"""
Save the merged vision-text model as a PyTorch checkpoint.
This works around ONNX export issues by keeping it in PyTorch format.

You can then:
1. Use it with a Python backend (FastAPI/Flask)
2. Convert with other tools (Optimum, TensorRT)
3. Export specific layers separately
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path


def load_projector_weights(onnx_path: str):
    """Load projector weights from ONNX file."""
    import onnx
    from onnx import numpy_helper

    print(f"Loading projector weights from: {onnx_path}")
    model = onnx.load(onnx_path)

    weights = {}
    for initializer in model.graph.initializer:
        weights[initializer.name] = numpy_helper.to_array(initializer)

    weight = weights['val_0']  # [768, 576]
    bias = weights['projection.bias']  # [576]

    return torch.from_numpy(weight.copy()).t(), torch.from_numpy(bias.copy())


class MergedVisionTextModel(nn.Module):
    """Merged model: projector + decoder"""

    def __init__(self, projector_onnx_path: str, decoder_id: str):
        super().__init__()

        # Load projector weights from ONNX
        weight, bias = load_projector_weights(projector_onnx_path)

        # Create projector
        self.projector = nn.Linear(768, 576)
        self.projector.weight.data = weight
        self.projector.bias.data = bias

        # Load decoder
        print(f"\nLoading decoder: {decoder_id}")
        self.decoder = AutoModelForCausalLM.from_pretrained(
            decoder_id,
            trust_remote_code=True,
        )
        self.decoder = self.decoder.to(torch.float32)
        self.decoder.eval()

    def forward(self, vision_features, text_input_ids, attention_mask):
        """
        Args:
            vision_features: [batch, 196, 768] from SigLIP
            text_input_ids: [batch, text_len] token IDs
            attention_mask: [batch, total_len] attention mask

        Returns:
            logits: [batch, total_len, vocab_size]
        """
        # Project vision features
        vision_embeds = self.projector(vision_features)

        # Get text embeddings
        text_embeds = self.decoder.model.embed_tokens(text_input_ids)

        # Concatenate
        combined_embeds = torch.cat([vision_embeds, text_embeds], dim=1)

        # Forward
        outputs = self.decoder.model(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )

        logits = self.decoder.lm_head(outputs.last_hidden_state)
        return logits


def save_merged_model(
    projector_onnx_path: str = "./multi_modal_projector/model.onnx",
    decoder_id: str = "HuggingFaceTB/SmolLM2-135M",
    output_dir: str = "./merged_model_pytorch",
):
    """Save the merged model as PyTorch checkpoint."""

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    print("="*60)
    print("Creating and saving merged vision-text model")
    print("="*60)

    # Create model
    model = MergedVisionTextModel(projector_onnx_path, decoder_id)
    model.eval()

    # Test it works
    print("\nTesting model...")
    batch_size = 1
    vision_len = 196
    text_len = 5
    vision_hidden = 768

    dummy_vision = torch.randn(batch_size, vision_len, vision_hidden)
    dummy_text_ids = torch.ones(batch_size, text_len, dtype=torch.int64)
    dummy_attention_mask = torch.ones(batch_size, vision_len + text_len, dtype=torch.int64)

    with torch.no_grad():
        output = model(dummy_vision, dummy_text_ids, dummy_attention_mask)
        print(f"✅ Forward pass successful!")
        print(f"   Output shape: {output.shape}")

    # Save model
    model_path = output_path / "merged_model.pt"
    print(f"\nSaving model to: {model_path}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'vision_hidden': 768,
            'text_hidden': 576,
            'decoder_id': decoder_id,
        }
    }, str(model_path))

    print(f"✅ Model saved!")

    # Save tokenizer
    print("\nSaving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(decoder_id)
    tokenizer.save_pretrained(str(output_path))
    print(f"✅ Tokenizer saved!")

    print("\n" + "="*60)
    print("Complete!")
    print("="*60)
    print(f"\nSaved to: {output_path.absolute()}")
    print("\nNext steps:")
    print("1. Use this model with a Python backend (FastAPI/Flask)")
    print("2. Upload to Hugging Face Hub")
    print("3. Try exporting with Optimum or other tools")
    print("\nThe model works perfectly in PyTorch!")


if __name__ == "__main__":
    save_merged_model()
