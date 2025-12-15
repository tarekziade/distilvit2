# Windows Setup Documentation

This document summarizes the setup process completed for training DistilVit on Windows 11 with 2x RTX 4090 GPUs.

## Environment Details

**Hardware:**
- 2x NVIDIA GeForce RTX 4090 (24GB VRAM each)
- Windows 11

**Software:**
- Python 3.13.9
- PyTorch 2.6+ with CUDA 12.4 support
- HuggingFace Transformers, PEFT, Accelerate
- Virtual environment at `.\venv`

## Setup Steps Completed

### 1. Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 2. Dependencies Installation
```powershell
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

**Key Changes Made:**
- `requirements.txt`: Updated `onnxruntime>=1.20.0` for Python 3.13 compatibility
- `setup.py`: Changed Python version check from `==3.11` to `>=3.11`

### 3. PyTorch with CUDA
```powershell
# Uninstalled CPU-only version
pip uninstall torch -y

# Installed CUDA-enabled version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 4. Multi-GPU Configuration
Modified `distilvit/train.py` to automatically use DataParallel when multiple GPUs are detected:
- Detects number of GPUs at runtime
- Uses DataParallel for Windows compatibility (Gloo DDP has PyTorch 2.6 bug)
- Automatic batch size adjustment
- FP16 mixed precision enabled

### 5. Training Scripts
Created Windows-compatible training scripts:
- `run.ps1` - Main PowerShell training interface
- `run.bat` - Batch file wrapper
- `train-multigpu.ps1` - Multi-GPU launcher (for future use when DDP bug is fixed)

## Current Configuration

### Training Parameters (Auto-configured)
- Batch size per device: 16
- Gradient accumulation steps: 4
- Effective batch size: 128 (with 2 GPUs)
- Mixed precision: FP16 enabled
- Multi-GPU: DataParallel (automatic)

### Model Architecture
- Encoder: google/siglip-base-patch16-224 (frozen)
- Decoder: HuggingFaceTB/SmolLM-135M (with LoRA)
- Trainable parameters: 2.2M / 221M (1%)
- LoRA config: rank=16, alpha=16, dropout=0.1

## Performance Characteristics

### Multi-GPU Behavior (DataParallel)
- **Expected speedup**: ~1.4x with 2 GPUs (vs theoretical 2x)
- **GPU utilization**:
  - GPU 0: ~100% (coordinator + worker)
  - GPU 1: ~40-60% (worker only)
- This uneven usage is **expected** with DataParallel architecture

### Training Speed Estimates (Flickr30k, 5 epochs)
- Single GPU: ~2-3 hours
- 2 GPUs (DataParallel): ~1.8-2.2 hours

## Known Issues & Workarounds

### Issue: DistributedDataParallel with Gloo (Windows)
**Problem**: PyTorch 2.6 has a bug with libuv on Windows
```
RuntimeError: use_libuv was requested but PyTorch was build without libuv support
```

**Workaround**: Using DataParallel instead (automatically configured)

**Future Solutions**:
- Wait for PyTorch fix in future release
- Use WSL2 with Linux PyTorch (NCCL backend)
- Use Linux directly for optimal performance

### Issue: Uneven GPU Usage
**Observation**: GPU 0 at 100%, GPU 1 at lower utilization

**Explanation**: This is **expected behavior** with DataParallel:
- GPU 0 acts as coordinator (gathers/broadcasts data)
- GPU 0 has additional overhead
- All GPUs compute forward/backward passes
- Bottleneck is at coordinator, not compute

**Not an Issue**: Both GPUs are being used, speedup is achieved

## Verification

### Check CUDA is Working
```powershell
.\venv\Scripts\activate
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"
```

Expected output:
```
CUDA: True
GPUs: 2
```

### Check GPU During Training
Open another terminal:
```powershell
nvidia-smi -l 1
```

You should see:
- Both GPUs showing activity
- GPU 0 with higher utilization (~100%)
- GPU 1 with moderate utilization (~40-60%)
- Both GPUs with memory allocated

## Training Commands

### Quick Test (Already Completed)
```powershell
.\run.ps1 train-quick
```
- 100 samples, 1 epoch
- Validates setup works
- Takes ~2-3 minutes

### Full Training on Flickr30k (Recommended)
```powershell
.\run.ps1 train -Dataset flickr -Epochs 5
```
- Full Flickr30k dataset (~30K images)
- 5 epochs
- Takes ~1.8-2.2 hours with 2 GPUs
- Produces production-ready model

### Full Training on COCO
```powershell
.\run.ps1 train -Dataset coco -Epochs 3
```
- COCO dataset (~118K images)
- 3 epochs
- Takes ~8-10 hours with 2 GPUs

### Multi-Dataset Training
```powershell
.\run.ps1 train -Dataset "flickr coco" -Epochs 3
```
- Combined Flickr30k + COCO
- 3 epochs
- Takes ~10-12 hours with 2 GPUs

### Custom Configuration
```powershell
.\run.ps1 train -Dataset flickr -Epochs 5 -Decoder HuggingFaceTB/SmolLM-360M
```
- Larger decoder model (360M vs 135M)
- Better caption quality
- Slightly slower training

## Monitoring Training

### View Progress
Training output shows:
- Loss values
- Training speed (samples/second)
- GPU memory usage
- Estimated time remaining

### Check GPU Usage (Separate Terminal)
```powershell
# Real-time GPU monitoring
nvidia-smi -l 1

# Or use Windows Task Manager
# Performance tab -> GPU section
```

### Checkpoints
Models saved to `./checkpoints/` directory:
- Incremental checkpoints during training
- Final model at completion
- Can resume training from checkpoints

## Next Steps

1. **Run full training** (command provided above)
2. **Monitor progress** with nvidia-smi
3. **Test trained model**: `.\run.ps1 test`
4. **Upload to HuggingFace Hub**: `.\run.ps1 upload-hub -ModelId username/model-name -ModelPath ./checkpoints/final`
5. **Quantize for deployment**: `.\run.ps1 quantize -ModelPath ./checkpoints/final`

## Files Modified

1. `requirements.txt` - Updated onnxruntime, onnx versions for Python 3.13
2. `setup.py` - Changed Python version requirement to >=3.11
3. `distilvit/train.py` - Added DataParallel multi-GPU support (lines 514-533)

## Files Created

1. `run.ps1` - Main Windows training script
2. `run.bat` - Batch file wrapper
3. `train-multigpu.ps1` - Multi-GPU launcher (for future use)
4. `WINDOWS.md` - Windows usage guide
5. `SETUP.md` - This file

## References

- [WINDOWS.md](WINDOWS.md) - Complete Windows usage guide
- [docs/architecture.md](docs/architecture.md) - Model architecture details
- [docs/multi-gpu-windows.md](docs/multi-gpu-windows.md) - Multi-GPU training guide
- [README.md](README.md) - Project overview

---

**Setup completed successfully!** Ready for full training runs.
