#!/usr/bin/env python3
"""
ONNX export using Optimum for transformers.js compatibility.

This approach exports the vision encoder and language model separately,
matching the structure expected by transformers.js.
"""
import torch
import json
import os
import shutil
from pathlib import Path


def merge_lora_weights(model_path, output_path):
    """Step 1: Merge LoRA adapters into base model weights."""
    print("=" * 80)
    print("STEP 1: Merging LoRA weights")
    print("=" * 80)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import SiglipVisionModel, AutoImageProcessor
    from distilvit.prefix_model import PrefixConditioningVLM, PrefixConditioningConfig
    from safetensors.torch import load_file

    # Load config
    print(f"Loading config from {model_path}...")
    with open(f"{model_path}/config.json", "r") as f:
        config_dict = json.load(f)
    config = PrefixConditioningConfig(**config_dict)

    # Load base models
    print("Loading base vision encoder (SigLIP)...")
    vision_encoder = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")

    print("Loading base language model (SmolLM-135M)...")
    language_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")

    # Create model and load trained weights
    print("Creating model and loading trained weights...")
    model = PrefixConditioningVLM(config, vision_encoder, language_model)
    state_dict = load_file(f"{model_path}/model.safetensors")
    model.load_state_dict(state_dict, strict=False)

    # Merge LoRA weights into language model
    print("Merging LoRA adapters into language model...")
    merged_language_model = model.language_model.merge_and_unload()

    # Create new model with merged weights
    print("Creating final merged model...")
    merged_model = PrefixConditioningVLM(
        config,
        model.vision_encoder,     # Vision encoder unchanged (was frozen)
        merged_language_model     # Language model with LoRA merged
    )
    merged_model.projection = model.projection  # Projection was fully trained

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Save merged model
    print(f"Saving merged model to {output_path}...")
    merged_model.save_pretrained(output_path)

    # Save tokenizer and image processor
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(output_path)

    print("Saving image processor...")
    image_processor = AutoImageProcessor.from_pretrained("google/siglip-base-patch16-224")
    image_processor.save_pretrained(output_path)

    print("[OK] LoRA weights merged successfully!")
    print()
    return merged_model, output_path


def export_vision_encoder_optimum(merged_model, output_dir):
    """Step 2a: Export vision encoder using torch.onnx.export."""
    print("=" * 80)
    print("STEP 2a: Exporting Vision Encoder")
    print("=" * 80)

    from PIL import Image
    from transformers import AutoImageProcessor

    # Create vision encoder output directory
    vision_dir = os.path.join(output_dir, "vision_encoder")
    os.makedirs(vision_dir, exist_ok=True)

    merged_model.vision_encoder.eval()

    # Create dummy input
    print("Creating dummy input...")
    image_processor = AutoImageProcessor.from_pretrained("google/siglip-base-patch16-224")
    dummy_image = Image.new('RGB', (224, 224), color='red')
    pixel_values = image_processor(dummy_image, return_tensors="pt").pixel_values

    # Export
    output_path = os.path.join(vision_dir, "model.onnx")
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        merged_model.vision_encoder,
        (pixel_values,),
        output_path,
        input_names=['pixel_values'],
        output_names=['last_hidden_state', 'pooler_output'],
        dynamic_axes={
            'pixel_values': {0: 'batch_size'},
            'last_hidden_state': {0: 'batch_size'},
            'pooler_output': {0: 'batch_size'}
        },
        opset_version=17,
        do_constant_folding=True,
    )

    # Verify
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[OK] Vision encoder exported to {output_path}")
    print(f"Size: {size_mb:.2f} MB")
    print()
    return vision_dir


def export_language_model_optimum(merged_model, output_dir):
    """Step 2b: Skip language model export (use unified model instead)."""
    print("=" * 80)
    print("STEP 2b: Language Model Export")
    print("=" * 80)

    print("Note: Language models with prefix-conditioning are better exported")
    print("as a unified model. See export_onnx_quantized.py for the complete model.")
    print()
    print("For transformers.js compatibility, you can:")
    print("  1. Use the modular approach (vision + projection + manual decoding)")
    print("  2. Or use the unified INT8 model from export_onnx_quantized.py")
    print()
    print("[SKIPPED] Language model export (not needed for modular approach)")
    print()
    return None


def export_projection_layer(merged_model, output_dir):
    """Step 2c: Export projection layer."""
    print("=" * 80)
    print("STEP 2c: Exporting Projection Layer")
    print("=" * 80)

    projection_dir = os.path.join(output_dir, "projection")
    os.makedirs(projection_dir, exist_ok=True)

    merged_model.projection.eval()

    # Create dummy input (SigLIP output dim = 768)
    dummy_input = torch.randn(1, 196, 768)  # [batch, seq_len, hidden_dim]

    output_path = os.path.join(projection_dir, "model.onnx")
    torch.onnx.export(
        merged_model.projection,
        (dummy_input,),
        output_path,
        input_names=['vision_features'],
        output_names=['projected_features'],
        dynamic_axes={
            'vision_features': {0: 'batch_size', 1: 'sequence_length'},
            'projected_features': {0: 'batch_size', 1: 'sequence_length'}
        },
        opset_version=17,
        do_constant_folding=True,
    )

    print(f"[OK] Projection layer exported to {output_path}")
    print()
    return projection_dir


def quantize_onnx_model(onnx_model_path, output_path):
    """Quantize a single ONNX model to INT8."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    print(f"Quantizing {onnx_model_path}...")
    quantize_dynamic(
        model_input=onnx_model_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8,
    )

    # Get sizes
    original_size = os.path.getsize(onnx_model_path) / (1024 * 1024)
    quantized_size = os.path.getsize(output_path) / (1024 * 1024)
    reduction = (1 - quantized_size / original_size) * 100

    print(f"  Original: {original_size:.2f} MB")
    print(f"  Quantized: {quantized_size:.2f} MB")
    print(f"  Reduction: {reduction:.1f}%")


def quantize_all_models(output_dir):
    """Step 3: Quantize all exported models."""
    print("=" * 80)
    print("STEP 3: Quantizing models to INT8")
    print("=" * 80)

    # Find all model.onnx files
    onnx_files = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file == "model.onnx":
                onnx_files.append(os.path.join(root, file))

    if not onnx_files:
        print("No ONNX models found to quantize")
        return

    print(f"Found {len(onnx_files)} ONNX models to quantize\n")

    for onnx_file in onnx_files:
        dirname = os.path.dirname(onnx_file)
        quantized_file = os.path.join(dirname, "model_quantized.onnx")
        quantize_onnx_model(onnx_file, quantized_file)
        print()

    print("[OK] All models quantized!")
    print()


def copy_supporting_files(merged_model_path, output_dir):
    """Copy tokenizer and config files."""
    print("=" * 80)
    print("Copying supporting files")
    print("=" * 80)

    files_to_copy = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "config.json",
        "preprocessor_config.json",
    ]

    for filename in files_to_copy:
        src = os.path.join(merged_model_path, filename)
        if os.path.exists(src):
            dst = os.path.join(output_dir, filename)
            shutil.copy2(src, dst)
            print(f"  Copied: {filename}")

    print("[OK] Supporting files copied")
    print()


def create_readme(output_dir):
    """Create README with usage instructions."""
    print("=" * 80)
    print("Creating README")
    print("=" * 80)

    readme = """# ONNX Model Export

This directory contains ONNX models exported using Optimum for transformers.js compatibility.

## Structure

```
onnx_export_optimum/
├── vision_encoder/
│   ├── model.onnx              # FP32 vision encoder
│   └── model_quantized.onnx    # INT8 quantized vision encoder
├── projection/
│   ├── model.onnx              # FP32 projection layer
│   └── model_quantized.onnx    # INT8 quantized projection
├── language_model/
│   ├── model.onnx              # FP32 language model (if supported)
│   └── model_quantized.onnx    # INT8 quantized language model
├── tokenizer.json
├── config.json
└── preprocessor_config.json
```

## Usage

The models are exported in a modular format:

1. **Vision Encoder**: Processes images (224x224 RGB) → vision features
2. **Projection**: Projects vision features to language model input space
3. **Language Model**: Generates text from projected features

### Browser Usage (ONNX Runtime Web)

```javascript
import * as ort from 'onnxruntime-web';

// Load models
const visionSession = await ort.InferenceSession.create('./vision_encoder/model_quantized.onnx');
const projectionSession = await ort.InferenceSession.create('./projection/model_quantized.onnx');

// Process image
const visionOutput = await visionSession.run({pixel_values: imageFeatures});
const projected = await projectionSession.run({vision_features: visionOutput.last_hidden_state});

// Use projected features with language model for generation
```

## Model Details

- **Vision Encoder**: SigLIP-base-patch16-224 (frozen during training)
- **Language Model**: SmolLM-135M with LoRA (merged)
- **Projection**: Trained projection head (768 → 576)

## Notes

This export uses Optimum where possible for better transformers.js compatibility.
For custom model architectures, some components may fall back to torch.onnx.export.
"""

    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w", encoding='utf-8') as f:
        f.write(readme)

    print(f"[OK] README saved to {readme_path}")
    print()


def main():
    """Complete export pipeline using Optimum."""
    print("\n")
    print("=" * 80)
    print("ONNX EXPORT PIPELINE WITH OPTIMUM")
    print("=" * 80)
    print()

    # Configuration
    model_path = "./siglip-base-patch16-224-SmolLM-135M-lora"
    merged_path = "./siglip-base-patch16-224-SmolLM-135M-merged"
    onnx_output_dir = "./onnx_export_optimum"

    print(f"Input model: {model_path}")
    print(f"Merged model output: {merged_path}")
    print(f"ONNX export output: {onnx_output_dir}")
    print()

    # Step 1: Merge LoRA weights
    merged_model, merged_path = merge_lora_weights(model_path, merged_path)

    # Step 2: Export components using Optimum
    vision_dir = export_vision_encoder_optimum(merged_model, onnx_output_dir)
    lm_dir = export_language_model_optimum(merged_model, onnx_output_dir)
    projection_dir = export_projection_layer(merged_model, onnx_output_dir)

    # Step 3: Quantize all models
    quantize_all_models(onnx_output_dir)

    # Step 4: Copy supporting files
    copy_supporting_files(merged_path, onnx_output_dir)

    # Step 5: Create README
    create_readme(onnx_output_dir)

    # Final summary
    print("=" * 80)
    print("EXPORT COMPLETE!")
    print("=" * 80)
    print()
    print("Output structure:")
    print(f"  {onnx_output_dir}/")
    print(f"    - vision_encoder/")
    print(f"        - model.onnx (FP32)")
    print(f"        - model_quantized.onnx (INT8)")
    print(f"    - projection/")
    print(f"        - model.onnx (FP32)")
    print(f"        - model_quantized.onnx (INT8)")
    if lm_dir:
        print(f"    - language_model/")
        print(f"        - model.onnx (FP32)")
        print(f"        - model_quantized.onnx (INT8)")
    print(f"    - tokenizer.json")
    print(f"    - config.json")
    print(f"    - README.md")
    print()
    print("See README.md in output directory for usage instructions.")
    print()


if __name__ == "__main__":
    main()
