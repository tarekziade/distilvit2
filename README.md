# distilvit

Parameter-efficient image captioning using prefix-conditioning and LoRA.

This project fine-tunes a vision-language model for image captioning using modern techniques:
- **Prefix-conditioning**: Projects vision features as prompt tokens (no cross-attention needed)
- **LoRA**: Parameter-efficient fine-tuning (only 1% of parameters trainable)
- **Frozen encoder**: SigLIP vision encoder remains frozen
- **Modern LLMs**: Works with any decoder-only language model (SmolLM, Qwen2, Phi, etc.)

Resulting model is available on Hugging Face model hub at https://huggingface.co/mozilla/distilvit

The train script is inspired from https://ankur3107.github.io/blogs/the-illustrated-image-captioning-using-transformers/#references

## Quick Start

Using the Makefile (recommended):

```bash
make install        # Set up environment and install dependencies
make train-quick    # Quick test training (100 samples)
make train          # Full training on Flickr dataset
make help           # Show all available commands
```

Or manually:

```bash
python3.11 -m venv .
bin/pip install -r requirements.txt
bin/pip install -e .
```

## Training

**Quick test**:
```bash
make train-quick
```

**Train on specific dataset**:
```bash
make train DATASET=flickr
make train DATASET=coco EPOCHS=5
```

**Train with different architectures**:
```bash
make train-modern    # SigLIP-2 + SmolLM-360M (better quality)
make train-large     # SigLIP-SO400M + SmolLM-1.7B (maximum quality)
```

**Customize LoRA parameters**:
```bash
bin/train --dataset flickr \
          --lora-r 32 --lora-alpha 16 \
          --projection-lr 1e-3 --lora-lr 5e-5
```

**Train with CLIP loss** (improves image-text alignment):
```bash
# Recommended: Use CLIP loss for better caption quality
bin/train --dataset flickr \
          --clip-loss-weight 0.1 \
          --clip-model openai/clip-vit-base-patch32

# High quality: Use larger CLIP model (slower but better)
bin/train --dataset flickr \
          --clip-loss-weight 0.2 \
          --clip-model openai/clip-vit-large-patch14-336
```

**Train on all datasets** (requires 2TB disk space):
```bash
make train-all
# Or manually:
bin/train --dataset all
```

**Merge all training datasets into one Hub-ready dataset**:
```bash
make merge-datasets MERGE_SAVE=merged_dataset \
    MERGE_DATASETS="coco flickr pexels docornot validation" \
    SAMPLE=500  # optional throttle
# Push to Hub:
make merge-datasets PUSH_TO_HUB=org/merged-distilvit PRIVATE=1
```

## Testing

Once trained, test the model:

```bash
make test
# Or manually:
bin/python distilvit/infere.py
```

## Browser Demo

### Native JavaScript Implementation (Recommended)

Complete browser-based implementation with no server required:

```bash
# Export your model components
python export_prefix_vlm.py --model-dir ./siglip-base-patch16-224-SmolLM-135M-lora --output-dir onnx
# Outputs:
#   onnx/vision_encoder/model.onnx
#   onnx/projection.onnx
#   onnx/language_model/model.onnx
#   onnx/prefix_init.onnx

# Upload to HuggingFace Hub (see docs)

# Open demo
python -m http.server 8000
open http://localhost:8000/demo_complete.html
```

**Features:**
- 🚀 Complete native JavaScript implementation
- 📦 Loads models directly from HuggingFace Hub
- ⚡ INT8 quantization for fast loading
- 🎯 Proper autoregressive generation
- 💾 Browser caching after first load
- 🔒 100% privacy (all processing in browser)

See [docs/javascript-implementation.md](docs/javascript-implementation.md) for complete documentation.

### Alternative: Python Backend

If you prefer server-side processing:

**Option 1: Flask API**
```bash
pip install flask flask-cors transformers
python server.py
open http://localhost:5000
```

**Option 2: Gradio App**
```bash
pip install gradio transformers
python app_gradio.py
open http://localhost:7860
```

Both use the native Python transformers pipeline for best quality.

## Dataset Quality Analysis

Analyze dataset quality, measure bias, and validate caption transformations:

```bash
make quality-report-quick  # Quick analysis (100 samples)
make quality-report        # Full dataset analysis
```

The quality report measures:
- **Image-text alignment** (CLIP scores)
- **Caption fidelity** (BERT scores)
- **Bias detection** with original vs transformed comparison
- **Object distribution** and imbalance metrics

See [docs/dataset_quality.md](docs/dataset_quality.md) for complete documentation.

### Generate Synthetic Data for Rare Classes

Balance your dataset by generating synthetic images for underrepresented objects:

```bash
# Step 1: Generate prompts for rare objects
make generate-prompts

# Step 2: Generate images using Stable Diffusion/DALL-E
# (Use synthetic_prompts.jsonl with your image generation tool)

# Step 3: Add generated images to training data
```

This creates bias-free captions for rare objects (objects with <50 samples) that you can use with image generation tools to augment your training dataset.

## Model Architecture

**Prefix-Conditioning with LoRA (2025)**:

Modern parameter-efficient architecture using prefix-conditioning instead of cross-attention:

```
Image → SigLIP (frozen) → Projection (trainable) → SmolLM + LoRA → Caption
```

**Default Configuration**:
- Encoder: SigLIP-2 Base (google/siglip-base-patch16-224) - 86M params (frozen)
- Projection: Linear layer - ~350K params (trainable)
- Decoder: SmolLM-135M (HuggingFaceTB/SmolLM-135M) - 135M params with LoRA adapters
- **Total**: ~221M parameters, **only 2.2M trainable (1%)**
- LoRA: rank=16, alpha=16, applied to attention matrices

**Key Features**:
- **Parameter Efficient**: LoRA reduces trainable params by 99%
- **Flexible**: Works with ANY decoder-only LM (SmolLM, Qwen2, Phi, Llama-based, etc.)
- **Fast Training**: Frozen encoder + LoRA means significantly faster training
- **Memory Efficient**: Batch size 4 + gradient accumulation fits on consumer GPUs
- **ONNX-Friendly**: Simpler architecture, easier deployment

The architecture is fully pluggable with LoRA parameters. See [docs/architecture.md](docs/architecture.md) for detailed documentation and model options.

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[architecture.md](docs/architecture.md)**: Complete architecture documentation, model options, and design decisions
- **[architecture-comparison.md](docs/architecture-comparison.md)**: Performance comparison between old and new architectures
- **[dataset_quality.md](docs/dataset_quality.md)**: Dataset quality analysis, bias detection, and validation metrics
- **[fighting_bias.md](docs/fighting_bias.md)**: Bias mitigation strategy and caption transformation guidelines
- **[multi-gpu-windows.md](docs/multi-gpu-windows.md)**: Guide for training on Windows 11 with multiple GPUs (2xRTX4090)
- **[onnx-export-guide.md](docs/onnx-export-guide.md)**: Guide for merging LoRA weights and exporting to ONNX
- **[transformers-js-compatibility.md](docs/transformers-js-compatibility.md)**: Browser deployment compatibility analysis
- **[transformers-js-explained.md](docs/transformers-js-explained.md)**: Detailed explanation of transformers.js integration
