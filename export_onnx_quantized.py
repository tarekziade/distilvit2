#!/usr/bin/env python3
"""
Complete ONNX export pipeline with INT8 quantization for transformers.js compatibility.

Steps:
1. Merge LoRA weights into base model
2. Export to ONNX format
3. Apply INT8 quantization
"""
import torch
import torch.onnx
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


def export_to_onnx(merged_model_path, output_dir):
    """Step 2: Export merged model to ONNX format."""
    print("=" * 80)
    print("STEP 2: Exporting to ONNX")
    print("=" * 80)

    from transformers import AutoImageProcessor
    from distilvit.prefix_model import PrefixConditioningVLM, PrefixConditioningConfig
    from PIL import Image
    from safetensors.torch import load_file

    # Load merged model
    print(f"Loading merged model from {merged_model_path}...")
    with open(f"{merged_model_path}/config.json", "r") as f:
        config_dict = json.load(f)
    config = PrefixConditioningConfig(**config_dict)

    from transformers import AutoModelForCausalLM, SiglipVisionModel
    vision_encoder = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
    language_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")

    model = PrefixConditioningVLM(config, vision_encoder, language_model)
    state_dict = load_file(f"{merged_model_path}/model.safetensors")
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Create ONNX-compatible wrapper that returns only logits tensor
    print("Creating ONNX export wrapper...")
    class ONNXExportWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            # Call model and extract only the logits tensor
            outputs = self.model(pixel_values)
            # Handle both dict-like and tuple outputs
            if hasattr(outputs, 'logits'):
                return outputs.logits
            elif isinstance(outputs, (tuple, list)):
                return outputs[0]
            else:
                return outputs

    wrapped_model = ONNXExportWrapper(model)
    wrapped_model.eval()

    # Create dummy input for tracing
    print("Creating dummy input for ONNX tracing...")
    image_processor = AutoImageProcessor.from_pretrained(merged_model_path)
    dummy_image = Image.new('RGB', (224, 224), color='red')
    pixel_values = image_processor(dummy_image, return_tensors="pt").pixel_values

    print(f"Input shape: {pixel_values.shape}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "model.onnx")

    # Export with torch.onnx.export
    print(f"Exporting to {output_path}...")
    print("This may take several minutes...")

    torch.onnx.export(
        wrapped_model,
        (pixel_values,),
        output_path,
        input_names=['pixel_values'],
        output_names=['logits'],
        dynamic_axes={
            'pixel_values': {0: 'batch_size'},
            'logits': {0: 'batch_size', 1: 'sequence_length'}
        },
        opset_version=17,
        do_constant_folding=True,
        export_params=True,
    )

    print(f"[OK] Model exported to {output_path}")

    # Verify export
    print("Verifying ONNX model...")
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("[OK] ONNX model verified!")

    # Get model size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Model size: {size_mb:.2f} MB")
    print()

    return output_path


def quantize_onnx_model(onnx_model_path, output_dir):
    """Step 3: Apply INT8 quantization to ONNX model."""
    print("=" * 80)
    print("STEP 3: Applying INT8 quantization")
    print("=" * 80)

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        output_path = os.path.join(output_dir, "model_int8.onnx")

        print(f"Quantizing {onnx_model_path}...")
        print("This will reduce model size by ~4x...")

        quantize_dynamic(
            model_input=onnx_model_path,
            model_output=output_path,
            weight_type=QuantType.QUInt8,
        )

        # Get sizes
        original_size = os.path.getsize(onnx_model_path) / (1024 * 1024)
        quantized_size = os.path.getsize(output_path) / (1024 * 1024)
        reduction = (1 - quantized_size / original_size) * 100

        print(f"[OK] Quantized model saved to {output_path}")
        print(f"Original size: {original_size:.2f} MB")
        print(f"Quantized size: {quantized_size:.2f} MB")
        print(f"Size reduction: {reduction:.1f}%")
        print()

        return output_path

    except ImportError:
        print("ERROR: onnxruntime not installed with quantization support")
        print("Install with: pip install onnxruntime")
        return None


def copy_supporting_files(merged_model_path, output_dir):
    """Copy tokenizer and config files for browser deployment."""
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


def create_usage_example(output_dir):
    """Create a JavaScript usage example."""
    print("=" * 80)
    print("Creating usage example")
    print("=" * 80)

    example_js = """// Example: Load and use the quantized ONNX model
import * as ort from 'onnxruntime-web';

class ImageCaptionModel {
  constructor() {
    this.session = null;
  }

  async load(modelPath = './onnx_export/model_int8.onnx') {
    console.log('Loading ONNX model...');
    this.session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['webgpu', 'wasm']
    });
    console.log('Model loaded successfully!');
  }

  async preprocessImage(imageElement) {
    // Resize to 224x224
    const canvas = document.createElement('canvas');
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imageElement, 0, 0, 224, 224);

    // Get pixel data and normalize
    const imageData = ctx.getImageData(0, 0, 224, 224);
    const pixels = imageData.data;
    const float32Data = new Float32Array(1 * 3 * 224 * 224);

    // SigLIP normalization: scale to [-1, 1]
    for (let i = 0; i < 224 * 224; i++) {
      float32Data[i] = (pixels[i * 4] / 255) * 2 - 1;           // R
      float32Data[224 * 224 + i] = (pixels[i * 4 + 1] / 255) * 2 - 1;     // G
      float32Data[224 * 224 * 2 + i] = (pixels[i * 4 + 2] / 255) * 2 - 1; // B
    }

    return new ort.Tensor('float32', float32Data, [1, 3, 224, 224]);
  }

  async generateCaption(imageElement, maxLength = 30) {
    const pixelValues = await this.preprocessImage(imageElement);

    // Run inference (single forward pass)
    const feeds = { pixel_values: pixelValues };
    const results = await this.session.run(feeds);

    // Get logits and decode
    const logits = results.logits;

    // Note: You'll need to implement token decoding
    // using the tokenizer.json file
    console.log('Generated logits shape:', logits.dims);

    return logits;
  }
}

// Usage
const model = new ImageCaptionModel();
await model.load();

const img = document.getElementById('myImage');
const caption = await model.generateCaption(img);
console.log('Caption:', caption);
"""

    example_path = os.path.join(output_dir, "usage_example.js")
    with open(example_path, "w") as f:
        f.write(example_js)

    print(f"[OK] Usage example saved to {example_path}")
    print()


def main():
    """Complete export pipeline."""
    print("\n")
    print("=" * 80)
    print("ONNX EXPORT PIPELINE WITH INT8 QUANTIZATION")
    print("=" * 80)
    print()

    # Configuration
    model_path = "./siglip-base-patch16-224-SmolLM-135M-lora"
    merged_path = "./siglip-base-patch16-224-SmolLM-135M-merged"
    onnx_output_dir = "./onnx_export"

    print(f"Input model: {model_path}")
    print(f"Merged model output: {merged_path}")
    print(f"ONNX export output: {onnx_output_dir}")
    print()

    # Step 1: Merge LoRA weights
    merged_model, merged_path = merge_lora_weights(model_path, merged_path)

    # Step 2: Export to ONNX
    onnx_model_path = export_to_onnx(merged_path, onnx_output_dir)

    # Step 3: Quantize to INT8
    quantized_path = quantize_onnx_model(onnx_model_path, onnx_output_dir)

    # Step 4: Copy supporting files
    copy_supporting_files(merged_path, onnx_output_dir)

    # Step 5: Create usage example
    create_usage_example(onnx_output_dir)

    # Final summary
    print("=" * 80)
    print("EXPORT COMPLETE!")
    print("=" * 80)
    print()
    print("Output files:")
    print(f"  - FP32 ONNX model: {onnx_output_dir}/model.onnx")
    if quantized_path:
        print(f"  - INT8 ONNX model: {onnx_output_dir}/model_int8.onnx")
    print(f"  - Tokenizer: {onnx_output_dir}/tokenizer.json")
    print(f"  - Usage example: {onnx_output_dir}/usage_example.js")
    print()
    print("Next steps:")
    print("  1. Copy the onnx_export/ directory to your web project")
    print("  2. Install: npm install onnxruntime-web")
    print("  3. See usage_example.js for integration code")
    print()
    print("Note: The model requires custom generation loop implementation")
    print("      in JavaScript. See docs/onnx-export-guide.md for details.")
    print()


if __name__ == "__main__":
    main()
