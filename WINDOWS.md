# Windows Usage Guide

This guide shows how to use the Windows-specific scripts as alternatives to the Unix Makefile.

## Setup Summary

**Environment:**
- Python 3.13.9 (in virtual environment)
- PyTorch 2.6+ with CUDA 12.4 support
- 2x NVIDIA GeForce RTX 4090 (24GB each)
- Multi-GPU: DataParallel (automatic)

**What Was Configured:**
1. ✓ Virtual environment created at `.\venv`
2. ✓ All dependencies installed (PyTorch, Transformers, PEFT, etc.)
3. ✓ CUDA support enabled and verified
4. ✓ Multi-GPU training configured with DataParallel
5. ✓ Python 3.13 compatibility fixes applied
6. ✓ Training scripts tested and working

**Current Status:**
- Single-GPU training: ✓ Working
- Multi-GPU training: ✓ Working (DataParallel, ~1.4x speedup)
- CUDA/GPU acceleration: ✓ Enabled
- FP16 mixed precision: ✓ Enabled

**Known Limitations:**
- DistributedDataParallel with Gloo has a Windows bug in PyTorch 2.6
- DataParallel shows uneven GPU usage (GPU 0 at 100%, GPU 1 lower) - this is expected
- For optimal multi-GPU performance (~1.9x), use Linux with NCCL or WSL2

## Quick Start

Two options are available:

### Option 1: PowerShell Script (Recommended)
```powershell
# Show help
.\run.ps1 help

# Install dependencies
.\run.ps1 install

# Quick test training
.\run.ps1 train-quick

# Full training
.\run.ps1 train -Dataset flickr -Epochs 5
```

### Option 2: Batch File
```cmd
# Show help
run help

# Install dependencies
run install

# Quick test training
run train-quick

# Full training
run train -Dataset flickr -Epochs 5
```

## Common Commands

### Setup and Environment
```powershell
.\run.ps1 install          # Install all dependencies
.\run.ps1 status           # Show environment status
.\run.ps1 info             # Show project information
.\run.ps1 list-models      # List available models
```

### Training
```powershell
# Quick test (100 samples, 1 epoch, ~2 minutes)
.\run.ps1 train-quick

# Train on Flickr30k
.\run.ps1 train -Dataset flickr -Epochs 5

# Train on COCO
.\run.ps1 train-coco -Epochs 3

# Train on multiple datasets
.\run.ps1 train -Dataset "flickr coco" -Epochs 5

# Train with custom models
.\run.ps1 train -Dataset flickr -Encoder google/siglip-base-patch16-224 -Decoder HuggingFaceTB/SmolLM-360M

# Train with sample (for testing)
.\run.ps1 train -Dataset flickr -Sample 1000 -Epochs 1
```

### Pre-configured Training Profiles
```powershell
# Modern architecture (SmolLM-360M)
.\run.ps1 train-modern -Dataset flickr -Epochs 5

# Large architecture (SmolLM-1.7B)
.\run.ps1 train-large -Dataset flickr -Epochs 5

# Legacy architecture (ViT + DistilGPT2)
.\run.ps1 train-legacy -Dataset flickr -Epochs 5
```

### Testing and Inference
```powershell
.\run.ps1 test             # Run inference test
```

### Model Management
```powershell
# Quantize a model
.\run.ps1 quantize -ModelPath ./checkpoints/my-model

# Upload to HuggingFace Hub
.\run.ps1 upload-hub -ModelId username/model-name -ModelPath ./checkpoints/my-model -Message "Training run description"

# Upload with tag
.\run.ps1 upload-hub -ModelId username/model-name -ModelPath ./checkpoints/my-model -Tag "v1.0" -Message "Release v1.0"
```

### Development
```powershell
.\run.ps1 lint             # Run code linting
.\run.ps1 format           # Format code with black
.\run.ps1 shell            # Open Python shell
.\run.ps1 clean            # Clean build artifacts
.\run.ps1 clean-all        # Clean everything including cache
```

## Parameters Reference

### Training Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `-Dataset` | Dataset to train on | `flickr` | `flickr`, `coco`, `docornot`, `pexels`, `validation`, `all` |
| `-Epochs` | Number of training epochs | `3` | Any integer |
| `-Sample` | Sample size for testing | (none) | Any integer |
| `-MaxLength` | Maximum caption length | `30` | Any integer |
| `-Encoder` | Vision encoder model | `google/siglip-base-patch16-224` | See list-models |
| `-Decoder` | Language decoder model | `HuggingFaceTB/SmolLM-135M` | See list-models |

### Upload Parameters

| Parameter | Description | Required |
|-----------|-------------|----------|
| `-ModelId` | HuggingFace model ID (e.g., `username/model-name`) | Yes |
| `-ModelPath` | Path to model directory | Yes |
| `-Tag` | Git tag for the upload | No |
| `-Message` | Commit message | No (default: "New training run") |

## Examples

### Basic Workflow
```powershell
# 1. Install dependencies
.\run.ps1 install

# 2. Check environment
.\run.ps1 status

# 3. Quick test to verify everything works
.\run.ps1 train-quick

# 4. Full training on Flickr30k
.\run.ps1 train -Dataset flickr -Epochs 5

# 5. Test the trained model
.\run.ps1 test

# 6. Upload to HuggingFace Hub
.\run.ps1 upload-hub -ModelId myusername/distilvit-flickr -ModelPath ./checkpoints/flickr-final -Message "Trained on Flickr30k for 5 epochs"
```

### Multi-GPU Training on Windows

**Current Setup: DataParallel (Windows Compatible)**

The training script automatically detects multiple GPUs and uses PyTorch DataParallel when multiple GPUs are available. This works out of the box on Windows without additional configuration:

```powershell
# Multi-GPU training is automatic when you have multiple GPUs
.\run.ps1 train -Dataset flickr -Epochs 5

# Or directly with the train command
.\Scripts\activate
train --dataset flickr --num-train-epochs 5
```

**Performance Expectations:**
- 2x RTX 4090: ~1.4x speedup vs single GPU
- GPU 0 will show higher utilization (acts as coordinator)
- GPU 1 will show lower utilization (this is expected with DataParallel)

**Note on DistributedDataParallel (DDP):**
- DDP with Gloo backend has a known bug in PyTorch 2.6 on Windows (libuv error)
- DataParallel provides reliable multi-GPU support on Windows
- For better scaling (~1.9x), use Linux with NCCL backend or WSL2

See [docs/multi-gpu-windows.md](docs/multi-gpu-windows.md) for detailed multi-GPU setup and troubleshooting.

### Experimentation
```powershell
# Try different model sizes
.\run.ps1 train -Decoder HuggingFaceTB/SmolLM-135M -Sample 1000 -Epochs 2
.\run.ps1 train -Decoder HuggingFaceTB/SmolLM-360M -Sample 1000 -Epochs 2
.\run.ps1 train -Decoder HuggingFaceTB/SmolLM-1.7B -Sample 1000 -Epochs 2

# Try different datasets with quick iterations
.\run.ps1 train -Dataset flickr -Sample 5000 -Epochs 1
.\run.ps1 train -Dataset coco -Sample 5000 -Epochs 1
```

## Troubleshooting

### PowerShell Execution Policy
If you get an error about execution policy, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Or run the script with:
```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 help
```

### Virtual Environment Issues
If the virtual environment gets corrupted:
```powershell
# Remove the virtual environment directories
Remove-Item -Recurse -Force Scripts, Lib, Include, pyvenv.cfg

# Reinstall
.\run.ps1 install
```

### Out of Memory
- Reduce batch size by editing `distilvit/train.py` (default is auto-configured based on hardware)
- Use smaller models: SmolLM-135M instead of SmolLM-1.7B
- Use sampling: `-Sample 1000` to train on fewer examples

### Path Issues
Make sure you're running the scripts from the project root directory:
```powershell
cd D:\github\distilvit2
.\run.ps1 <command>
```

## Comparison with Makefile

The PowerShell script (`run.ps1`) provides the same functionality as the Unix Makefile:

| Makefile | PowerShell | Description |
|----------|-----------|-------------|
| `make help` | `.\run.ps1 help` | Show help |
| `make install` | `.\run.ps1 install` | Install dependencies |
| `make train DATASET=flickr` | `.\run.ps1 train -Dataset flickr` | Train model |
| `make train-quick` | `.\run.ps1 train-quick` | Quick test |
| `make clean` | `.\run.ps1 clean` | Clean artifacts |
| `make status` | `.\run.ps1 status` | Show status |
| `make list-models` | `.\run.ps1 list-models` | List models |

## Notes

- The batch file (`run.bat`) is a simple wrapper around the PowerShell script
- Both scripts support all the same commands and parameters
- The scripts automatically detect Windows paths (Scripts/ instead of bin/)
- Virtual environment activation is handled automatically by the scripts
- For multi-GPU training, use `torchrun` directly (see Multi-GPU section above)

## Getting Help

```powershell
# Show all commands
.\run.ps1 help

# Show available models
.\run.ps1 list-models

# Show project info
.\run.ps1 info

# Check environment status
.\run.ps1 status
```

For more information, see:
- [README.md](README.md) - Project overview
- [docs/architecture.md](docs/architecture.md) - Architecture details
- [docs/multi-gpu-windows.md](docs/multi-gpu-windows.md) - Multi-GPU training on Windows
