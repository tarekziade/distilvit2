# Multi-GPU Training on Windows 11 (2xRTX4090)

## ✅ Automatic Configuration (Latest Version)

**The training script (`distilvit/train.py`) now automatically detects and configures Windows multi-GPU training!**

No manual configuration needed:
- ✅ Automatically detects Windows platform
- ✅ Automatically sets Gloo backend for Windows
- ✅ Automatically optimizes batch sizes for multi-GPU (16 per GPU)
- ✅ Automatically enables FP16 for RTX 4090
- ✅ Automatically adjusts dataloader workers
- ✅ Logs GPU configuration at startup

**Just run:**
```bash
# Single GPU
python distilvit/train.py --dataset flickr --num-train-epochs 5

# Multi-GPU with accelerate
accelerate launch distilvit/train.py --dataset flickr --num-train-epochs 5

# Multi-GPU with torchrun
torchrun --nproc_per_node=2 distilvit/train.py --dataset flickr --num-train-epochs 5
```

The script will automatically configure everything for optimal Windows multi-GPU performance!

---

## Background: Multi-GPU Training on Windows

Multi-GPU training on Windows 11 has limitations compared to Linux:

### Key Differences

| Backend | Linux | Windows | Performance |
|---------|-------|---------|-------------|
| **NCCL** | ✅ Supported | ❌ Not available | Fastest |
| **Gloo** | ✅ Supported | ✅ Supported | Slower |
| **MPI** | ✅ Supported | ⚠️ Complex setup | Medium |

**Windows limitation:** NCCL (fastest GPU communication) is Linux-only, forcing use of slower Gloo backend.

## Solution 1: Enable Gloo Backend (HuggingFace Trainer)

### Configuration Changes

Add to your training script:

```python
# distilvit/train.py
import os

# Set backend to Gloo for Windows
os.environ['PL_TORCH_DISTRIBUTED_BACKEND'] = 'gloo'

# Training arguments for multi-GPU
training_args = TrainingArguments(
    # ... existing args ...

    # Multi-GPU settings
    per_device_train_batch_size=16,  # Per GPU (32 total with 2 GPUs)
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,    # Effective batch = 16*2*2 = 64

    # Distributed training
    ddp_backend="gloo",               # Windows requires Gloo
    ddp_find_unused_parameters=False, # Faster training
    local_rank=-1,                    # Auto-detect

    # Other optimizations
    dataloader_num_workers=4,         # Per GPU
    dataloader_pin_memory=True,
)
```

### Launch Training

**Option A: HuggingFace Accelerate (Recommended)**

```bash
# First time: Configure accelerate
accelerate config

# Select:
# - Compute environment: This machine
# - Which type of machine: multi-GPU
# - How many machines: 1
# - How many processes: 2 (for 2 GPUs)
# - GPU ids: 0,1
# - Mixed precision: fp16
# - Backend: gloo (important for Windows!)

# Run training
accelerate launch distilvit/train.py \
    --dataset flickr \
    --num-train-epochs 5
```

**Option B: PyTorch Launch (Manual)**

```bash
# Windows PowerShell
python -m torch.distributed.launch `
    --nproc_per_node=2 `
    --use_env `
    distilvit/train.py `
    --dataset flickr `
    --num-train-epochs 5
```

**Option C: torchrun (PyTorch 2.0+)**

```bash
torchrun --nproc_per_node=2 distilvit/train.py \
    --dataset flickr \
    --num-train-epochs 5
```

## Solution 2: DataParallel (Simpler but Slower)

If distributed training fails, use DataParallel (single-process multi-GPU):

```python
# distilvit/train.py
import torch

# After model creation
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
    model = torch.nn.DataParallel(model)

# Rest of training code unchanged
```

**Pros:**
- ✅ Simpler setup (no distributed launch needed)
- ✅ Works reliably on Windows

**Cons:**
- ❌ Slower than DistributedDataParallel (~20-30% overhead)
- ❌ Single-process bottleneck
- ❌ Less memory efficient

## Solution 3: WSL2 with Linux (Best Performance)

For best multi-GPU performance on Windows hardware:

```bash
# Install WSL2 with Ubuntu
wsl --install -d Ubuntu-22.04

# In WSL2, NCCL works properly
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Clone repo in WSL2
git clone https://github.com/tarekziade/distilvit2
cd distilvit2
make install

# Train with NCCL backend (much faster)
torchrun --nproc_per_node=2 distilvit/train.py \
    --dataset flickr \
    --num-train-epochs 5
```

**Performance difference:**
- Native Windows (Gloo): ~1.5-1.8x speedup with 2 GPUs
- WSL2 (NCCL): ~1.9-2.0x speedup with 2 GPUs

## Expected Speedup with 2xRTX4090

### Single GPU (Baseline)

| Dataset | Epochs | Time (1x4090) |
|---------|--------|---------------|
| Flickr30k | 5 | ~2.5 hours |
| COCO | 3 | ~6 hours |

### Dual GPU Performance

| Method | Scaling Efficiency | Flickr30k (5 epochs) | COCO (3 epochs) |
|--------|-------------------|----------------------|-----------------|
| **DataParallel (Windows)** | ~1.4-1.5x | ~1.7 hours | ~4 hours |
| **DistributedDataParallel + Gloo (Windows)** | ~1.6-1.8x | ~1.4 hours | ~3.5 hours |
| **DistributedDataParallel + NCCL (WSL2)** | ~1.9-2.0x | ~1.3 hours | ~3 hours |

### Why Not Perfect 2x Speedup?

**Communication overhead:**
- Gradient synchronization between GPUs
- Model weight broadcasting
- Collective operations

**With LoRA (1% trainable params):**
- Minimal gradients to sync
- Better scaling efficiency than full fine-tuning
- Gloo overhead less noticeable

## Recommended Configuration for Your Setup

### Best Option: Accelerate + Gloo

```bash
# Windows PowerShell
cd C:\path\to\distilvit2

# Setup virtual environment
python3.11 -m venv .
.\bin\Activate.ps1
pip install -r requirements.txt
pip install -e .

# Configure accelerate (first time only)
accelerate config
# Choose: multi-GPU, 2 processes, Gloo backend, fp16

# Train
accelerate launch .\bin\train `
    --dataset flickr `
    --num-train-epochs 5 `
    --per-device-train-batch-size 16 `
    --gradient-accumulation-steps 2
```

### Verify GPU Usage During Training

```python
# Add to train.py for monitoring
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
```

```bash
# Monitor in separate terminal
nvidia-smi -l 1
```

You should see both GPUs at ~95-100% utilization.

## Troubleshooting

### Issue: Only One GPU Showing Activity

**Cause:** Training not launched with distributed launcher.

**Fix:** Use `accelerate launch` or `torchrun`, not `python` directly.

### Issue: NCCL Error on Windows

```
RuntimeError: NCCL is not supported on Windows
```

**Fix:** Set backend to Gloo:
```python
os.environ['PL_TORCH_DISTRIBUTED_BACKEND'] = 'gloo'
training_args.ddp_backend = "gloo"
```

### Issue: Slow Performance with 2 GPUs

**Possible causes:**
1. **Small model + Gloo overhead**: With LoRA (2.2M params), communication overhead may exceed benefits
2. **Batch size too small**: Try increasing per-device batch size
3. **Slow interconnect**: Check PCIe lanes (should be x16 for both)

**Fix:** Increase batch size to amortize overhead:
```python
training_args = TrainingArguments(
    per_device_train_batch_size=32,  # Increase from 16
    gradient_accumulation_steps=1,   # Less overhead
)
```

### Issue: Out of Memory with Larger Batches

**Fix:** Use gradient accumulation:
```python
training_args = TrainingArguments(
    per_device_train_batch_size=16,  # Fits in memory
    gradient_accumulation_steps=4,   # Effective batch = 16*4*2 = 128
)
```

## Is Multi-GPU Worth It for LoRA?

### With LoRA (This Architecture)

**Trainable parameters:** 2.2M (1% of model)

**Considerations:**
- ✅ Gradient sync is very fast (only 2.2M params)
- ✅ Good scaling efficiency expected
- ⚠️ Gloo overhead on Windows may reduce benefits

**Expected speedup:**
- **Windows (Gloo)**: 1.5-1.7x faster
- **WSL2 (NCCL)**: 1.8-1.9x faster

### Recommendation

**If staying on Windows:**
- Use 2 GPUs with Accelerate + Gloo
- Expected time: ~1.4 hours (vs 2.5 hours single GPU)
- Worth it for the time savings

**If maximum performance needed:**
- Use WSL2 for NCCL support
- Expected time: ~1.3 hours
- 15-20% faster than native Windows

**If simplicity preferred:**
- Use single RTX 4090
- Still very fast: ~2.5 hours for Flickr30k
- No distributed training complexity

## Summary

### Quick Start (Windows 11)

```bash
# Install and configure
pip install accelerate
accelerate config  # Choose: multi-GPU, 2 GPUs, Gloo, fp16

# Train with both GPUs
accelerate launch bin/train --dataset flickr --num-train-epochs 5

# Expected time: ~1.4 hours (vs 2.5 hours on single GPU)
```

### Expected Results

| Configuration | Flickr30k (5 epochs) | Speedup |
|--------------|---------------------|---------|
| 1x RTX 4090 | 2.5 hours | 1.0x |
| 2x RTX 4090 (Windows Gloo) | 1.4 hours | 1.8x |
| 2x RTX 4090 (WSL2 NCCL) | 1.3 hours | 1.9x |

The multi-GPU setup is definitely worth it for the time savings, even with Windows/Gloo limitations!
