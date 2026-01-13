#!/usr/bin/env python3
"""
Export prefix-conditioning VLM into four ONNX components for JS inference:

1) vision_encoder/model.onnx  – SigLIP vision encoder
2) projection.onnx            – Linear/MLP projection head
3) language_model/model.onnx  – Decoder with KV cache (text-generation-with-past)
4) prefix_init.onnx           – One-shot graph that consumes prefix embeddings + prompt ids
                               and returns logits + present key/values
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Iterable, Tuple

import torch
from torch import nn
from optimum.exporters.onnx import main_export

from distilvit.prefix_model import PrefixConditioningVLM, PrefixConditioningConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export prefix-conditioning VLM to ONNX (vision, projection, decoder, prefix_init).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-dir", type=Path, default=Path("."), help="Folder containing config.json and model.safetensors")
    parser.add_argument("--output-dir", type=Path, default=Path("onnx"), help="Where to store ONNX files")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset to use")
    return parser.parse_args()


def load_components(model_dir: Path) -> Tuple[nn.Module, nn.Module, nn.Module, PrefixConditioningConfig]:
    """
    Load the trained PrefixConditioningVLM, merge LoRA into the decoder, and return
    the vision encoder, projection head, merged decoder, and config.
    """
    model = PrefixConditioningVLM.from_pretrained(model_dir)
    model.eval()

    # Merge LoRA adapters for export
    merged_lm = model.language_model.merge_and_unload()
    merged_lm.config._attn_implementation = "eager"
    merged_lm.eval()

    model.projection.eval()
    model.vision_encoder.eval()

    return model.vision_encoder, model.projection, merged_lm, model.config


def export_submodels(vision_encoder, projection, language_model, output_dir: Path, opset: int):
    """
    Export vision encoder and decoder using optimum, projection via torch.onnx.export.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        vision_path = tmp_path / "vision_encoder"
        lm_path = tmp_path / "language_model"
        vision_encoder.save_pretrained(vision_path)
        language_model.save_pretrained(lm_path)

        # Vision encoder export
        main_export(
            model_name_or_path=str(vision_path),
            output=str(output_dir / "vision_encoder"),
            task="feature-extraction",
            opset=opset,
            device="cpu",
            trust_remote_code=True,
        )

        # Decoder with KV cache export
        main_export(
            model_name_or_path=str(lm_path),
            output=str(output_dir / "language_model"),
            task="text-generation-with-past",
            opset=opset,
            device="cpu",
            trust_remote_code=True,
        )

    # Export projection head via torch.onnx.export using a real shape sample
    sample_image = torch.zeros(
        1,
        3,
        vision_encoder.config.image_size,
        vision_encoder.config.image_size,
        dtype=torch.float32,
    )
    with torch.no_grad():
        vision_hidden = vision_encoder(sample_image, return_dict=True).last_hidden_state

    proj_path = output_dir / "projection.onnx"
    torch.onnx.export(
        projection,
        vision_hidden,
        proj_path,
        input_names=["vision_hidden_states"],
        output_names=["projected_prefix"],
        opset_version=opset,
        dynamic_axes={
            "vision_hidden_states": {0: "batch", 1: "sequence"},
            "projected_prefix": {0: "batch", 1: "sequence"},
        },
    )
    print(f"[OK] Projection exported to {proj_path}")


class PrefixDecoderInit(nn.Module):
    """
    ONNX-friendly wrapper for the first decoding pass:
    consumes prefix embeddings + prompt ids, returns logits + present key/values.
    """

    def __init__(self, language_model: nn.Module):
        super().__init__()
        self.language_model = language_model

    def forward(self, prefix_embeddings: torch.Tensor, input_ids: torch.Tensor):
        text_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([prefix_embeddings, text_embeds], dim=1)
        outputs = self.language_model(inputs_embeds=inputs_embeds, use_cache=True)
        pkv = outputs.past_key_values.to_legacy_cache()
        return (outputs.logits, *sum(pkv, ()))


def export_prefix_init(language_model, vision_hidden_shape: Iterable[int], output_dir: Path, opset: int):
    """
    Export the prefix-init graph that builds the initial cache using prefix embeddings.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    wrapper = PrefixDecoderInit(language_model)
    wrapper.eval()

    batch = 1
    prefix_len = vision_hidden_shape[1]
    text_len = 4

    prefix_dummy = torch.zeros(batch, prefix_len, language_model.config.hidden_size, dtype=torch.float32)
    input_ids = torch.zeros(batch, text_len, dtype=torch.long)

    present_names = [
        name
        for i in range(language_model.config.num_hidden_layers)
        for name in (f"present.{i}.key", f"present.{i}.value")
    ]

    dynamic_axes = {
        "prefix_embeddings": {0: "batch", 1: "prefix_sequence"},
        "input_ids": {0: "batch", 1: "text_sequence"},
        "logits": {0: "batch", 1: "sequence"},
    }
    # KV cache tensors: [batch, heads, sequence, head_dim]
    for name in present_names:
        dynamic_axes[name] = {0: "batch", 2: "sequence"}

    init_path = output_dir / "prefix_init.onnx"
    torch.onnx.export(
        wrapper,
        (prefix_dummy, input_ids),
        init_path,
        input_names=["prefix_embeddings", "input_ids"],
        output_names=["logits", *present_names],
        opset_version=opset,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    print(f"[OK] Prefix-init decoder exported to {init_path}")


def main() -> None:
    args = parse_args()
    vision_encoder, projection, language_model, config = load_components(args.model_dir)

    # Export vision encoder, decoder with KV cache, and projection
    export_submodels(vision_encoder, projection, language_model, args.output_dir, args.opset)

    # Use real hidden shape for prefix_init export
    sample_image = torch.zeros(
        1,
        3,
        vision_encoder.config.image_size,
        vision_encoder.config.image_size,
        dtype=torch.float32,
    )
    with torch.no_grad():
        vision_hidden = vision_encoder(sample_image, return_dict=True).last_hidden_state
    export_prefix_init(language_model, vision_hidden.shape, args.output_dir, args.opset)

    # Copy config for reference (optional consumer convenience)
    config.save_pretrained(args.output_dir)
    print(f"[DONE] ONNX components written to {args.output_dir}")


if __name__ == "__main__":
    main()
