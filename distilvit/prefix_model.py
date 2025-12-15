"""
Prefix-Conditioning Vision-Language Model

Architecture:
    Image → SigLIP2 encoder (frozen/partially frozen)
          ↓
    Linear/MLP projection (trainable)
          ↓
    SmolLM with LoRA adapters
          ↓
    Generate caption

This approach avoids cross-attention and works with decoder-only LMs.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Tuple, Union
from peft import get_peft_model, LoraConfig, TaskType


class PrefixConditioningConfig(PretrainedConfig):
    """Configuration for prefix-conditioning vision-language model."""

    model_type = "prefix_conditioning_vlm"

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        projection_dim=512,
        projection_type="linear",  # "linear" or "mlp"
        mlp_hidden_dim=2048,
        freeze_vision_encoder=True,
        vision_unfreeze_layers=0,  # Number of last layers to unfreeze
        lora_r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        lora_target_modules=None,  # None = auto-detect
        max_length=30,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vision_config = vision_config
        self.text_config = text_config
        self.projection_dim = projection_dim
        self.projection_type = projection_type
        self.mlp_hidden_dim = mlp_hidden_dim
        self.freeze_vision_encoder = freeze_vision_encoder
        self.vision_unfreeze_layers = vision_unfreeze_layers
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules
        self.max_length = max_length


class ProjectionHead(nn.Module):
    """Projects vision features to language model embedding space."""

    def __init__(self, vision_dim, text_dim, projection_type="linear", mlp_hidden_dim=2048):
        super().__init__()
        self.projection_type = projection_type

        if projection_type == "linear":
            self.projection = nn.Linear(vision_dim, text_dim)
        elif projection_type == "mlp":
            self.projection = nn.Sequential(
                nn.Linear(vision_dim, mlp_hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(mlp_hidden_dim, text_dim),
            )
        else:
            raise ValueError(f"Unknown projection type: {projection_type}")

    def forward(self, vision_features):
        """
        Args:
            vision_features: [batch_size, num_patches, vision_dim]
        Returns:
            projected_features: [batch_size, num_patches, text_dim]
        """
        return self.projection(vision_features)


class PrefixConditioningVLM(PreTrainedModel):
    """
    Vision-Language Model using prefix conditioning with LoRA.

    Instead of cross-attention, this model:
    1. Encodes image with frozen vision encoder
    2. Projects vision features to text embedding space
    3. Uses projected features as prefix prompts
    4. Generates caption with LoRA-adapted language model
    """

    config_class = PrefixConditioningConfig

    def __init__(self, config, vision_encoder, language_model):
        super().__init__(config)
        self.config = config

        # Vision encoder (SigLIP)
        self.vision_encoder = vision_encoder

        # Freeze vision encoder according to config
        self._freeze_vision_encoder()

        # Get dimensions
        vision_dim = vision_encoder.config.hidden_size
        text_dim = language_model.config.hidden_size

        # Projection head (always trainable)
        self.projection = ProjectionHead(
            vision_dim=vision_dim,
            text_dim=text_dim,
            projection_type=config.projection_type,
            mlp_hidden_dim=config.mlp_hidden_dim,
        )

        # Language model with LoRA
        self.language_model = self._setup_lora(language_model)

        # Store tokenizer reference (will be set externally)
        self.tokenizer = None

    def _freeze_vision_encoder(self):
        """Freeze vision encoder according to config."""
        if self.config.freeze_vision_encoder:
            # Freeze all parameters
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

            # Optionally unfreeze last N layers
            if self.config.vision_unfreeze_layers > 0:
                if hasattr(self.vision_encoder, 'vision_model'):
                    encoder = self.vision_encoder.vision_model.encoder
                else:
                    encoder = self.vision_encoder.encoder

                layers = encoder.layers if hasattr(encoder, 'layers') else encoder.layer
                num_layers = len(layers)
                unfreeze_from = num_layers - self.config.vision_unfreeze_layers

                for i in range(unfreeze_from, num_layers):
                    for param in layers[i].parameters():
                        param.requires_grad = True

    def _setup_lora(self, language_model):
        """Setup LoRA adapters for language model."""
        # Auto-detect target modules if not specified
        target_modules = self.config.lora_target_modules
        if target_modules is None:
            # Common patterns for different model architectures
            if hasattr(language_model.config, 'model_type'):
                model_type = language_model.config.model_type
                if model_type in ['llama', 'mistral']:
                    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
                elif model_type == 'gpt2':
                    target_modules = ["c_attn", "c_proj"]
                elif model_type == 'opt':
                    target_modules = ["q_proj", "v_proj"]
                else:
                    # Default: try common attention projection names
                    target_modules = ["q_proj", "v_proj"]

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        return get_peft_model(language_model, lora_config)

    def get_vision_features(self, pixel_values):
        """Extract features from vision encoder."""
        with torch.set_grad_enabled(not self.config.freeze_vision_encoder or self.config.vision_unfreeze_layers > 0):
            outputs = self.vision_encoder(pixel_values, return_dict=True)

            # Get the sequence output (all patch embeddings)
            if hasattr(outputs, 'last_hidden_state'):
                vision_features = outputs.last_hidden_state
            elif hasattr(outputs, 'pooler_output'):
                # If only pooled output available, unsqueeze to add sequence dim
                vision_features = outputs.pooler_output.unsqueeze(1)
            else:
                raise ValueError("Could not extract vision features from encoder output")

            return vision_features

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        """
        Forward pass for training.

        Args:
            pixel_values: [batch_size, channels, height, width]
            input_ids: [batch_size, seq_len] - caption token IDs
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, seq_len] - target token IDs for loss
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 1. Get vision features
        vision_features = self.get_vision_features(pixel_values)  # [B, num_patches, vision_dim]

        # 2. Project to text embedding space
        vision_embeds = self.projection(vision_features)  # [B, num_patches, text_dim]

        # 3. Get text embeddings
        if input_ids is not None:
            text_embeds = self.language_model.get_base_model().get_input_embeddings()(input_ids)  # [B, seq_len, text_dim]

            # 4. Concatenate vision prefix with text
            combined_embeds = torch.cat([vision_embeds, text_embeds], dim=1)  # [B, num_patches + seq_len, text_dim]

            # 5. Create combined attention mask
            batch_size = vision_embeds.shape[0]
            num_vision_tokens = vision_embeds.shape[1]
            vision_attention_mask = torch.ones(
                batch_size, num_vision_tokens,
                dtype=attention_mask.dtype if attention_mask is not None else torch.long,
                device=vision_embeds.device
            )

            if attention_mask is not None:
                combined_attention_mask = torch.cat([vision_attention_mask, attention_mask], dim=1)
            else:
                combined_attention_mask = None

            # 6. Prepare labels (vision tokens don't have labels)
            if labels is not None:
                # Pad labels with -100 for vision tokens (ignore in loss)
                vision_labels = torch.full(
                    (batch_size, num_vision_tokens),
                    -100,
                    dtype=labels.dtype,
                    device=labels.device
                )
                combined_labels = torch.cat([vision_labels, labels], dim=1)
            else:
                combined_labels = None
        else:
            # Generation mode: only vision embeddings
            combined_embeds = vision_embeds
            combined_attention_mask = None
            combined_labels = None

        # 7. Forward through language model
        outputs = self.language_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            labels=combined_labels,
            return_dict=return_dict,
            **kwargs
        )

        return outputs

    def generate(
        self,
        pixel_values: torch.FloatTensor,
        max_length: int = 30,
        num_beams: int = 3,
        temperature: float = 1.0,
        top_p: float = 0.9,
        **kwargs
    ):
        """
        Generate captions for images.

        Args:
            pixel_values: [batch_size, channels, height, width]
            max_length: Maximum caption length
            num_beams: Beam search width
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
        """
        # Get vision features and project
        vision_features = self.get_vision_features(pixel_values)
        vision_embeds = self.projection(vision_features)

        batch_size = vision_embeds.shape[0]
        num_vision_tokens = vision_embeds.shape[1]

        # Create attention mask for vision tokens
        vision_attention_mask = torch.ones(
            batch_size, num_vision_tokens,
            dtype=torch.long,
            device=vision_embeds.device
        )

        # Generate with language model
        # When using inputs_embeds, we must use max_new_tokens instead of max_length
        # because the embeddings don't have corresponding token IDs in the output
        outputs = self.language_model.generate(
            inputs_embeds=vision_embeds,
            attention_mask=vision_attention_mask,
            max_new_tokens=max_length,  # Generate max_length NEW tokens
            num_beams=num_beams,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer else 0,
            eos_token_id=self.tokenizer.eos_token_id if self.tokenizer else 2,
            **kwargs
        )

        # When using inputs_embeds, the output only contains the generated token IDs
        # (not the input embeddings), so we return the outputs directly
        return outputs

    def print_trainable_parameters(self):
        """Print number of trainable parameters."""
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        print(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {all_param:,} || "
            f"Trainable%: {100 * trainable_params / all_param:.2f}%"
        )
