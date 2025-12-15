# Architecture Documentation

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project fine-tunes a Visual Encoder Decoder model for image captioning. The resulting model (mozilla/distilvit) combines Google's ViT (Vision Transformer) encoder with DistilGPT2 decoder to generate captions for images.

## Environment Setup

**Python Version**: 3.11 (required, enforced in setup.py)

**Quick Setup with Makefile** (Recommended):
```bash
make install        # Creates venv and installs all dependencies
make train-quick    # Test training with small sample
make help           # Show all available commands
```

**Manual Installation**:
```bash
python3.11 -m venv .
bin/pip install -r requirements.txt
bin/pip install -e .
```

## Training Commands

### Using Makefile (Recommended)

**Quick test training**:
```bash
make train-quick    # 100 samples, 1 epoch, validation dataset
```

**Basic training** with specific dataset:
```bash
make train DATASET=flickr
make train DATASET=coco EPOCHS=5
```

**Training with different architectures**:
```bash
make train-modern    # SigLIP-2 + OPT-350M (larger decoder)
make train-large     # SigLIP-SO400M + OPT-350M (largest)
make train-legacy    # ViT + DistilGPT2 (original)
```

**Multi-dataset training**:
```bash
make train DATASET='flickr coco' EPOCHS=5
make train-multi     # Shortcut for Flickr + COCO
```

**Full training** with all datasets (requires 2TB disk space):
```bash
make train-all
```

**Custom configuration**:
```bash
make train \
    DATASET=flickr \
    ENCODER=google/siglip-so400m-patch14-384 \
    DECODER=HuggingFaceTB/SmolLM-360M \
    MAX_LENGTH=256 \
    EPOCHS=10
```

### Direct Command Line

**Basic training**:
```bash
bin/train --dataset flickr
```

**Full training** with all datasets:
```bash
bin/train --dataset all
```

**Training with Metaflow** (for GPU orchestration):
```bash
python DistilVitFlow.py run --arg_dataset flickr
```

**Key training parameters**:
- `--dataset`: Choices are `flickr`, `coco`, `docornot`, `pexels`, `validation`
- `--encoder-model`: Vision encoder model (default: `google/siglip-base-patch16-224`)
- `--decoder-model`: Language decoder model (default: `HuggingFaceTB/SmolLM-135M`)
- `--feature-extractor-model`: Feature extractor (default: same as encoder)
- `--max-length`: Maximum caption length in tokens (default: 128)
- `--num-train-epochs`: Number of training epochs (default: 3)
- `--eval-steps`: Evaluation interval (default: 100)
- `--save-steps`: Checkpoint save interval (default: 100)
- `--sample`: Sample a subset of data for quick testing
- `--debug`: Enable debug mode with prediction logging
- `--push-to-hub`: Upload trained model to HuggingFace

**Example with multiple datasets**:
```bash
bin/train --dataset flickr coco --num-train-epochs 5
```

## Testing

**Run inference** to test a trained model:
```bash
bin/python distilvit/infere.py --before mozilla/distilvit --after /path/to/local/model
```

## Model Architecture

**Prefix-Conditioning with LoRA (2025)**:

The model now uses a modern prefix-conditioning architecture with parameter-efficient LoRA adapters, which is simpler and more efficient than traditional cross-attention encoder-decoder models. This architecture is also easier to export to ONNX for deployment.

**Architecture Flow**:
```
Image → Vision Encoder (SigLIP, frozen)
      ↓
Linear/MLP Projection (trainable)
      ↓
Vision Embeddings as Prefix Tokens
      ↓
Language Model with LoRA adapters (SmolLM)
      ↓
Generated Caption (25-30 tokens)
```

**Components**:
- **Vision Encoder**: SigLIP-2 Base (`google/siglip-base-patch16-224`, 86M params) - Frozen during training
- **Projection Head**: Linear or MLP layer that projects vision features to language model embedding space (trainable)
- **Language Model**: SmolLM-135M (`HuggingFaceTB/SmolLM-135M`, 135M params) with LoRA adapters
- **Total Parameters**: ~221M total, only 2.2M trainable (1%)
- **LoRA Configuration**: Rank r=16, alpha=16, applied to attention projection matrices

**Key Advantages**:
- ✅ **Parameter Efficient**: Only 1% of parameters are trainable via LoRA
- ✅ **Flexible**: Works with any decoder-only language model (Llama-based, GPT-based, etc.)
- ✅ **ONNX-Friendly**: No cross-attention simplifies export
- ✅ **Fast Training**: Frozen encoder + LoRA means faster training with less memory
- ✅ **Modern LLMs**: Can use state-of-the-art decoder-only models (SmolLM, Qwen2, Phi, Llama, etc.)

**Supported Decoder Models**:
- ✅ **SmolLM** (HuggingFaceTB/SmolLM-135M, SmolLM-360M, SmolLM-1.7B) - Recommended, efficient Llama-based models
- ✅ **Qwen2** (Qwen/Qwen2-0.5B, Qwen/Qwen2-1.5B) - High-quality alternative
- ✅ **Phi** (microsoft/Phi-3-mini-4k-instruct) - Premium quality
- ✅ **Any decoder-only LM** with causal language modeling support

**Architecture is Fully Pluggable**:
```bash
# Use different encoder/decoder combinations with LoRA parameters
bin/train --encoder-model google/siglip-so400m-patch14-384 \
          --decoder-model HuggingFaceTB/SmolLM-360M \
          --lora-r 16 --lora-alpha 16 \
          --projection-lr 1e-3 --lora-lr 5e-5 \
          --dataset flickr
```

**Recommended Model Combinations**:

*Option A: Balanced (default)*
- Encoder: `google/siglip-base-patch16-224` (86M params)
- Decoder: `HuggingFaceTB/SmolLM-135M` (135M params)
- Total: ~221M parameters, 2.2M trainable (1%)
- Max Length: 30 tokens
- Use: Default configuration
- Best for: Fast training, efficient deployment

*Option B: Better Quality*
- Encoder: `google/siglip-base-patch16-224` (86M params)
- Decoder: `HuggingFaceTB/SmolLM-360M` (360M params)
- Total: ~446M parameters, ~3M trainable
- Max Length: 30 tokens
- Use: `--decoder-model HuggingFaceTB/SmolLM-360M`
- Best for: Improved caption quality

*Option C: Large Configuration*
- Encoder: `google/siglip-so400m-patch14-384` (400M params)
- Decoder: `HuggingFaceTB/SmolLM-1.7B` (1.7B params)
- Total: ~2.1B parameters, ~10M trainable
- Max Length: 30 tokens
- Use: `--encoder-model google/siglip-so400m-patch14-384 --decoder-model HuggingFaceTB/SmolLM-1.7B`
- Best for: Maximum quality, research use

*Option D: Premium Quality*
- Encoder: `google/siglip-base-patch16-224` (86M params)
- Decoder: `microsoft/Phi-3-mini-4k-instruct` (3.8B params)
- Total: ~3.9B parameters, ~15M trainable
- Max Length: 30 tokens
- Use: `--decoder-model microsoft/Phi-3-mini-4k-instruct`
- Best for: State-of-the-art caption quality

**Key Configuration Details**:
- `--max-length`: Maximum tokens for captions (default: 30, optimized for short descriptions)
- `--lora-r`: LoRA rank (default: 16, range: 8-32)
- `--lora-alpha`: LoRA alpha scaling (default: 16)
- `--lora-dropout`: LoRA dropout rate (default: 0.1)
- `--projection-lr`: Learning rate for projection head (default: 1e-3)
- `--lora-lr`: Learning rate for LoRA parameters (default: 5e-5)
- `--projection-type`: Projection type (default: "linear", options: "linear", "mlp")
- Training uses standard Trainer (not Seq2SeqTrainer) with custom optimizer for differential learning rates
- Metrics: ROUGE and METEOR for caption quality evaluation
- Early stopping enabled with patience=3
- Checkpoints saved to `./checkpoints/` directory
- Best model selection based on `eval_loss`
- Tokenizer automatically handles missing special tokens (pad_token, bos_token)
- Memory efficient: Gradient accumulation (effective batch size 64) + gradient checkpointing

## Dataset Architecture

Dataset loaders are in `distilvit/_datasets/`. Each dataset module provides a `get_dataset()` function that returns a HuggingFace `DatasetDict` with 'train' and 'validation' splits.

**Available datasets** (registered in `_datasets/__init__.py`):
- `flickr`: Flickr30k dataset
- `coco`: COCO captions dataset
- `docornot`: Document classification dataset
- `pexels`: Pexels image dataset
- `validation`: Validation dataset

Datasets are combined using `concatenate_datasets()` and shuffled with seed 42.

## Metaflow Integration

The project supports Metaflow for orchestrating GPU training jobs on GCP. The `GenAIFlow` base class provides utilities to convert argparse arguments to Metaflow parameters.

**GenAIFlow Features**:
- Automatic conversion from argparse to Metaflow parameters
- GCP secret management integration
- Kubernetes GPU scheduling support

**DistilVitFlow**: Subclass that runs training on GCP with GPU allocation (100GB disk, 1 GPU).

## Post-Training Pipeline

After training completes, the following happens automatically:

1. **Model saving**: Saves to `./vit-base-patch16-224-distilgpt2/` (or custom path)
2. **Quantization**: Converts model to ONNX and quantizes for inference efficiency
3. **Hub upload** (if `--push-to-hub` specified): Uploads to HuggingFace

**Manual quantization**:
```bash
bin/python distilvit/quantize.py --model_id /path/to/model --quantize --task image-to-text-with-past
```

**Manual upload**:
```bash
bin/python distilvit/upload.py --model-id mozilla/distilvit --save-path /path/to/model --commit-message "Training update"
```

## Important Constants and Configurations

- `THE_ANSWER_TO_LIFE_THE_UNIVERSE_AND_EVERYTHING = 42`: Random seed for reproducibility
- `MODEL_ID = "mozilla/distilvit"`: Default HuggingFace model identifier
- **Training Configuration**:
  - Per-device batch size: 4 (with gradient accumulation 16 = effective batch 64)
  - Learning rates: Projection 1e-3, LoRA 5e-5 (differential learning rates)
  - Weight decay: 0.01
  - Warmup steps: 500
  - LR scheduler: Cosine decay
  - LoRA: rank=16, alpha=16, dropout=0.1
  - Gradient checkpointing: Enabled on language model
- Generation config: 3 beams, max length 30 tokens
- Weights & Biases integration disabled by default (`report_to="none"`)

## Device Selection

Automatic device detection prioritizes: CUDA > MPS (Apple Silicon) > CPU

Override with `--device` flag: `--device cuda` or `--device mps` or `--device cpu`

## Caching and Checkpoints

- **Cache directory**: `./cache/` (use `--cache-dir` to override)
- **Checkpoints directory**: `./checkpoints/` (use `--checkpoints-dir` to override)
- **Prune cache**: Use `--prune-cache` to empty cache directory before training
- **Resume training**: Automatically detects and resumes from last checkpoint in checkpoints directory

## Environment Variables

Training sets these environment variables:
```python
NCCL_P2P_DISABLE=1              # Disable NCCL peer-to-peer
NCCL_IB_DISABLE=1               # Disable InfiniBand
PYTORCH_ENABLE_MPS_FALLBACK=1   # Enable MPS fallback for unsupported ops
WANDB_PROJECT=distilvit         # Weights & Biases project name
WANDB_LOG_MODEL=false           # Don't upload model to W&B
```

## Recent Changes (2025)

### Prefix-Conditioning Architecture with LoRA (Latest)

**Major Architectural Redesign**:

The project has migrated from traditional cross-attention encoder-decoder to a modern prefix-conditioning architecture with LoRA adapters.

**What Changed**:
- **Architecture**: Cross-attention encoder-decoder → Prefix-conditioning with LoRA
- **Model Class**: `VisionEncoderDecoderModel` → `PrefixConditioningVLM` (custom)
- **Trainer**: `Seq2SeqTrainer` → `Trainer` with custom optimizer
- **Parameter Efficiency**: 100% trainable → 1% trainable (via LoRA)
- **Default Decoder**: Back to SmolLM-135M (now works via prefix-conditioning)
- **Max Length**: 128 → 30 tokens (optimized for short captions)
- **Memory Usage**: Significantly reduced via gradient accumulation + checkpointing

**New Training Parameters**:
- `--lora-r`: LoRA rank (default: 16)
- `--lora-alpha`: LoRA alpha scaling (default: 16)
- `--lora-dropout`: LoRA dropout rate (default: 0.1)
- `--projection-lr`: Learning rate for projection head (default: 1e-3)
- `--lora-lr`: Learning rate for LoRA parameters (default: 5e-5)
- `--projection-type`: "linear" or "mlp" (default: "linear")
- `--freeze-vision`: Freeze vision encoder (default: True)
- `--unfreeze-vision-layers`: Number of last vision layers to unfreeze (default: 0)

**Why These Changes**:
- **Parameter Efficient**: LoRA reduces trainable parameters from ~200M to ~2M (99% reduction)
- **Flexibility**: Prefix-conditioning works with ANY decoder-only LM (no cross-attention required)
- **Modern Models**: Can now use state-of-the-art LLMs (SmolLM, Qwen2, Phi, Llama-based)
- **ONNX Export**: Simpler architecture is easier to export for deployment
- **Memory Efficient**: Frozen encoder + LoRA significantly reduces GPU memory requirements
- **Faster Training**: Fewer parameters to update means faster iteration

**Implementation Files**:
- `distilvit/prefix_model.py`: New PrefixConditioningVLM model class with ProjectionHead
- `distilvit/train.py`: Completely rewritten for prefix-conditioning + LoRA
- `requirements.txt`: Added `peft>=0.7.0` for LoRA support

**Testing New Architecture**:
Quick test with validation dataset:
```bash
make train DATASET=validation SAMPLE=5 EPOCHS=1
# Or manually:
bin/train --dataset validation --sample 5 --num-train-epochs 1
```

**Migration Benefits**:
1. **Memory**: Batch size 4 with gradient accumulation 16 (effective 64) fits on MPS/consumer GPUs
2. **Quality**: Modern LLMs (SmolLM, Qwen2) trained on recent data produce better captions
3. **Speed**: Only updating 1% of parameters accelerates training significantly
4. **Deployment**: Simpler architecture exports cleanly to ONNX/ONNX Runtime

### Previous: Modern Architecture Update (Early 2025)

The project was previously updated with modern 2025 vision-language models:
- **Default Encoder**: ViT → SigLIP-2 Base (`google/siglip-base-patch16-224`)
- **Default Decoder**: DistilGPT2 (with cross-attention)
- SigLIP-2 is specifically trained for vision-language tasks (released Feb 2025)
- Fully pluggable architecture for easy experimentation

## Makefile Targets

The project includes a comprehensive Makefile for common operations:

**Setup and Installation**:
- `make install` - Set up virtual environment and install dependencies
- `make check-python` - Verify Python 3.11 is available
- `make status` - Show environment and cache status

**Training**:
- `make train` - Train with custom parameters (DATASET, EPOCHS, etc.)
- `make train-quick` - Quick test (100 samples, 1 epoch)
- `make train-modern` - High-quality modern architecture
- `make train-premium` - Premium quality with Phi-3
- `make train-legacy` - Original ViT + DistilGPT2
- `make train-flickr/coco` - Train on specific dataset
- `make train-multi` - Train on Flickr + COCO
- `make train-all` - Train on all datasets (2TB required)

**Testing and Utilities**:
- `make test` - Run inference comparison
- `make quantize MODEL_PATH=...` - Quantize a trained model
- `make upload-hub MODEL_ID=... MODEL_PATH=...` - Upload to HuggingFace
- `make list-models` - Show available encoder/decoder options
- `make shell` - Open Python shell with environment

**Maintenance**:
- `make clean` - Remove build artifacts
- `make clean-all` - Remove artifacts, cache, and checkpoints
- `make lint` - Run code linting
- `make format` - Format code with black
- `make help` - Show all available targets

**Environment Variables**:
All training targets accept these variables:
- `DATASET` - Dataset(s) to train on (default: flickr)
- `EPOCHS` - Number of epochs (default: 3)
- `SAMPLE` - Sample size for testing (default: full)
- `MAX_LENGTH` - Max caption length (default: 128)
- `ENCODER` - Vision encoder model
- `DECODER` - Language decoder model
