# AI Agent Guidelines

This document provides guidance for AI coding agents (Claude Code, Cursor, Copilot, etc.) working with this codebase.

## Quick Reference

**Project**: Image captioning with prefix-conditioning and LoRA
**Language**: Python 3.11+
**Framework**: PyTorch, HuggingFace Transformers, PEFT

## Key Documentation

- **[docs/architecture.md](docs/architecture.md)** - Complete architecture documentation, design decisions, and implementation details
- **[docs/architecture-comparison.md](docs/architecture-comparison.md)** - Performance comparison: old vs new architecture (5.7x speedup)
- **[docs/multi-gpu-windows.md](docs/multi-gpu-windows.md)** - Windows 11 multi-GPU training guide (2xRTX4090)
- **[docs/onnx-export-guide.md](docs/onnx-export-guide.md)** - Merging LoRA weights and exporting to ONNX
- **[docs/transformers-js-compatibility.md](docs/transformers-js-compatibility.md)** - Browser deployment analysis
- **[README.md](README.md)** - Quick start guide and project overview

## Architecture Overview

**Current Architecture (Prefix-Conditioning + LoRA)**:
```
Image → SigLIP (frozen) → Projection (trainable) → SmolLM + LoRA → Caption
```

**Key Components**:
- `distilvit/prefix_model.py` - PrefixConditioningVLM, ProjectionHead, PrefixConditioningConfig
- `distilvit/train.py` - Training script with automatic multi-GPU configuration
- `distilvit/_datasets.py` - Dataset loaders (Flickr30k, COCO, etc.)

**Training Parameters**:
- Trainable: 2.2M / 221M total (1%)
- LoRA: rank=16, alpha=16, dropout=0.1
- Batch: 4-16 per device with gradient accumulation
- LR: Projection 1e-3, LoRA 5e-5
- Optimizer: AdamW with cosine schedule

## Code Guidelines

### When Making Changes

1. **Read First**: Never propose changes without reading the file first
2. **Test**: Run `make train-quick` for fast validation
3. **Follow Patterns**: Study existing code before implementing new features
4. **LoRA-First**: This architecture uses LoRA - don't add full fine-tuning
5. **Frozen Encoder**: SigLIP encoder stays frozen by default

### Training Script Behavior

The training script (`distilvit/train.py`) automatically detects:
- **Platform**: Windows vs Linux (sets Gloo vs NCCL backend)
- **GPUs**: Single vs multi-GPU (adjusts batch sizes)
- **Device**: CUDA vs MPS vs CPU (enables fp16 on CUDA)

No manual configuration needed for different hardware.

### Testing Changes

```bash
# Quick test (100 samples, 1 epoch, ~2 minutes)
make train-quick

# Full training
make train DATASET=flickr EPOCHS=5

# Custom training
bin/train --dataset flickr --num-train-epochs 5 --lora-r 32
```

## Common Tasks

### Adding a New Decoder Model

1. Model must be decoder-only (GPT-style, not encoder-decoder)
2. Update `--decoder-model` argument
3. LoRA automatically applies to attention layers
4. Tokenizer must have pad_token and bos_token (script handles this)

### Adjusting LoRA Parameters

```bash
bin/train --dataset flickr \
          --lora-r 32 \              # Increase rank for more capacity
          --lora-alpha 16 \          # Keep alpha same or adjust
          --projection-lr 1e-3 \     # Projection learning rate
          --lora-lr 5e-5             # LoRA learning rate
```

### Multi-GPU Training

**Windows 11** (automatic):
```bash
torchrun --nproc_per_node=2 distilvit/train.py --dataset flickr --num-train-epochs 5
```

**Linux** (automatic):
```bash
torchrun --nproc_per_node=4 distilvit/train.py --dataset flickr --num-train-epochs 5
```

Script automatically configures Gloo (Windows) or NCCL (Linux) backend.

## File Structure

```
distilvit2/
├── distilvit/
│   ├── prefix_model.py          # Core architecture (PrefixConditioningVLM)
│   ├── train.py                 # Training script (auto multi-GPU)
│   ├── _datasets.py             # Dataset loaders
│   ├── infere.py                # Inference script
│   └── upload.py                # HuggingFace Hub upload
├── docs/                        # All documentation
├── Makefile                     # Training shortcuts
├── requirements.txt             # Dependencies (including peft>=0.7.0)
└── README.md                    # Quick start
```

## Performance Expectations

### Training Speed (vs Old Architecture)

- **Old (Cross-Attention)**: 62 minutes for test run
- **New (Prefix + LoRA)**: 11 minutes for test run
- **Speedup**: 5.7x faster

### GPU Estimates (Flickr30k, 5 epochs)

- **Single RTX 4090**: ~2-3 hours
- **2x RTX 4090 (Windows/Gloo)**: ~1.4 hours (1.6-1.8x speedup)
- **2x RTX 4090 (Linux/NCCL)**: ~1.3 hours (1.9x speedup)

## Important Notes

1. **No Cross-Attention**: This architecture uses prefix-conditioning, not VisionEncoderDecoderModel
2. **LoRA Only**: Don't add full fine-tuning - defeats the purpose of this architecture
3. **Frozen Encoder**: SigLIP encoder is frozen for efficiency
4. **Decoder-Only**: Language model must be decoder-only (GPT-style), not encoder-decoder
5. **Auto Configuration**: Training script handles multi-GPU, fp16, batch sizes automatically

## Troubleshooting

### Out of Memory
- Reduce `per_device_train_batch_size` (script defaults: 16 multi-GPU, 8 single GPU, 4 MPS)
- Increase `gradient_accumulation_steps` to maintain effective batch size

### Slow Training
- Check GPU utilization: `nvidia-smi -l 1`
- Ensure fp16 is enabled (automatic on CUDA)
- Multi-GPU: Use `torchrun` or `accelerate launch`, not plain `python`

### Windows Multi-GPU Not Working
- Use `torchrun --nproc_per_node=2` or `accelerate launch`
- Script automatically sets Gloo backend
- See [docs/multi-gpu-windows.md](docs/multi-gpu-windows.md)

## Getting Help

- Check [docs/architecture.md](docs/architecture.md) for detailed architecture documentation
- Run `make help` to see all available commands
- Review recent commits for examples of changes

---

**Note**: This is a LoRA-based architecture optimized for parameter efficiency and fast training. Keep changes aligned with this design philosophy.
