# Multi-GPU Training Launcher for Windows with Gloo Backend
# This script sets the required environment variable to fix the libuv error on Windows

param(
    [Parameter(Position=0, Mandatory=$false)]
    [string]$Dataset = "validation",

    [Parameter(Position=1, Mandatory=$false)]
    [int]$Epochs = 1,

    [Parameter(Position=2, Mandatory=$false)]
    [string]$Sample = "100",

    [Parameter(Position=3, Mandatory=$false)]
    [int]$NumGPUs = 2
)

# Activate virtual environment
& ".\venv\Scripts\Activate.ps1"

# Set environment variable to disable libuv (fixes Windows torchrun issue)
$env:TORCH_DISTRIBUTED_USE_LIBUV = "0"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Multi-GPU Training with Gloo Backend" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "GPUs: $NumGPUs" -ForegroundColor Green
Write-Host "Dataset: $Dataset" -ForegroundColor Green
Write-Host "Epochs: $Epochs" -ForegroundColor Green
Write-Host "Sample: $Sample" -ForegroundColor Green
Write-Host "Backend: Gloo (Windows)" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Build arguments
$args_list = @(
    "--dataset", $Dataset,
    "--num-train-epochs", $Epochs
)

if ($Sample) {
    $args_list += "--sample", $Sample
}

# Launch training with torchrun
torchrun --nproc_per_node=$NumGPUs distilvit/train.py @args_list
