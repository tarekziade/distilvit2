"""
Create a merged ONNX model that combines:
1. Vision features input (from SigLIP)
2. Projector (vision features -> embeddings)
3. Decoder embeddings + forward pass

This bypasses the inputs_embeds limitation by creating a single graph.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from pathlib import Path
import numpy as np


def load_projector_weights(onnx_path: str):
    """Load projector weights from ONNX file."""
    import onnx
    from onnx import numpy_helper

    print(f"Loading projector weights from: {onnx_path}")
    model = onnx.load(onnx_path)

    weights = {}
    for initializer in model.graph.initializer:
        weights[initializer.name] = numpy_helper.to_array(initializer)

    # Extract weight matrix and bias
    # val_0 is the weight matrix [768, 576]
    # projection.bias is the bias [576]
    weight = weights['val_0']  # [768, 576]
    bias = weights['projection.bias']  # [576]

    print(f"  Weight shape: {weight.shape}")
    print(f"  Bias shape: {bias.shape}")

    return torch.from_numpy(weight).t(), torch.from_numpy(bias)  # Transpose to [576, 768] for nn.Linear


class MergedVisionTextModel(nn.Module):
    """Merged model: projector + decoder"""

    def __init__(self, projector_onnx_path: str, decoder_id: str):
        super().__init__()

        # Load projector weights from ONNX
        weight, bias = load_projector_weights(projector_onnx_path)

        # Create projector as Linear layer
        self.projector = nn.Linear(768, 576)  # SigLIP hidden (768) -> SmolLM hidden (576)
        self.projector.weight.data = weight
        self.projector.bias.data = bias

        print(f"\nLoading decoder: {decoder_id}")
        self.decoder = AutoModelForCausalLM.from_pretrained(
            decoder_id,
            trust_remote_code=True,
        )
        self.decoder = self.decoder.to(torch.float32)
        self.decoder.eval()

        print(f"Decoder config:")
        print(f"  Hidden size: {self.decoder.config.hidden_size}")
        print(f"  Vocab size: {self.decoder.config.vocab_size}")

    def forward(self, vision_features, text_input_ids, attention_mask):
        """
        Args:
            vision_features: [batch, 196, 768] from SigLIP (pooled, not 1152)
            text_input_ids: [batch, text_len] token IDs for text prompt
            attention_mask: [batch, total_len] attention mask

        Returns:
            logits: [batch, total_len, vocab_size]
        """
        # Project vision features
        vision_embeds = self.projector(vision_features)  # [batch, 196, 576]

        # Get text embeddings
        text_embeds = self.decoder.model.embed_tokens(text_input_ids)  # [batch, text_len, 576]

        # Concatenate vision + text embeddings
        combined_embeds = torch.cat([vision_embeds, text_embeds], dim=1)  # [batch, 196+text_len, 576]

        # Forward through decoder
        outputs = self.decoder.model(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )

        # Get logits
        logits = self.decoder.lm_head(outputs.last_hidden_state)

        return logits


def export_merged_model(
    projector_onnx_path: str = "./multi_modal_projector/model.onnx",
    decoder_id: str = "HuggingFaceTB/SmolLM2-135M",
    output_dir: str = "./merged_model",
):
    """Export the merged model."""

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    print("="*60)
    print("Creating merged vision-text model")
    print("="*60)

    model = MergedVisionTextModel(projector_onnx_path, decoder_id)
    model.eval()

    # Dummy inputs
    batch_size = 1
    vision_len = 196  # SigLIP patches (14x14 grid)
    text_len = 5  # Short prompt
    vision_hidden = 768  # SigLIP hidden size (after pooling)

    dummy_vision = torch.randn(batch_size, vision_len, vision_hidden)
    dummy_text_ids = torch.ones(batch_size, text_len, dtype=torch.int64)
    dummy_attention_mask = torch.ones(batch_size, vision_len + text_len, dtype=torch.int64)

    print("\nDummy inputs for export:")
    print(f"  Vision features: {dummy_vision.shape}")
    print(f"  Text IDs: {dummy_text_ids.shape}")
    print(f"  Attention mask: {dummy_attention_mask.shape}")

    # Test forward pass before export
    print("\nTesting forward pass...")
    with torch.no_grad():
        output = model(dummy_vision, dummy_text_ids, dummy_attention_mask)
        print(f"  Output logits shape: {output.shape}")

    # Export
    onnx_path = output_path / "model.onnx"
    print(f"\nExporting to ONNX: {onnx_path}")
    print("This may take a few minutes...")

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_vision, dummy_text_ids, dummy_attention_mask),
            str(onnx_path),
            input_names=["vision_features", "text_input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "vision_features": {0: "batch"},
                "text_input_ids": {0: "batch", 1: "text_length"},
                "attention_mask": {0: "batch", 1: "total_length"},
                "logits": {0: "batch", 1: "total_length"},
            },
            opset_version=14,
            do_constant_folding=True,
            export_params=True,
            dynamo=False,  # Use legacy exporter
            verbose=False,
        )

    print(f"✅ Model exported to: {onnx_path}")

    # Verify with ONNX Runtime
    print("\nVerifying ONNX model...")
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(onnx_path))

        outputs = session.run(
            None,
            {
                "vision_features": dummy_vision.numpy(),
                "text_input_ids": dummy_text_ids.numpy(),
                "attention_mask": dummy_attention_mask.numpy(),
            }
        )

        print(f"✅ ONNX inference successful!")
        print(f"   Output logits shape: {outputs[0].shape}")

    except Exception as e:
        print(f"⚠️  ONNX Runtime test failed: {e}")
        print("   Model may still work in transformers.js")

    print("\n" + "="*60)
    print("Export complete!")
    print("="*60)
    print("\nThis merged model accepts:")
    print("  1. vision_features: [batch, 196, 768] from SigLIP")
    print("  2. text_input_ids: [batch, text_len] for text prompt")
    print("  3. attention_mask: [batch, 196+text_len]")
    print("\nNext steps:")
    print("  1. Upload this model to Hugging Face Hub")
    print("  2. Update demo_browser.js to use this merged model")
    print("  3. Now you can do true vision-conditioned generation!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export merged vision-text model")
    parser.add_argument("--projector", type=str, default="./multi_modal_projector/model.onnx")
    parser.add_argument("--decoder", type=str, default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--output", type=str, default="./merged_model")

    args = parser.parse_args()

    export_merged_model(
        projector_onnx_path=args.projector,
        decoder_id=args.decoder,
        output_dir=args.output,
    )
