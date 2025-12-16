# DistilVit - Image Captioning Model Training
# PowerShell script for Windows
# Usage: .\run.ps1 <command> [options]

param(
    [Parameter(Position=0)]
    [string]$Command = "help",

    # Training parameters
    [string]$Dataset = "flickr pexels coco",
    [int]$Epochs = 5,
    [string]$Sample = "",
    [int]$MaxLength = 30,
    [string]$Encoder = "google/siglip-base-patch16-224",
    [string]$Decoder = "HuggingFaceTB/SmolLM-135M",
    [switch]$DeleteOldCheckpoints,

    # Upload parameters
    [string]$ModelId = "",
    [string]$ModelPath = "",
    [string]$Tag = "",
    [string]$Message = "New training run"
)

# Color output functions
function Write-Success {
    param([string]$msg)
    Write-Host "[OK] $msg" -ForegroundColor Green
}
function Write-Info {
    param([string]$msg)
    Write-Host $msg -ForegroundColor Cyan
}
function Write-Warning {
    param([string]$msg)
    Write-Host "[WARN] $msg" -ForegroundColor Yellow
}
function Write-Error {
    param([string]$msg)
    Write-Host "[ERROR] $msg" -ForegroundColor Red
}

# Python paths
$PYTHON = "python"
$VENV = "venv"
$SCRIPTS = Join-Path $VENV "Scripts"
$PYTHON_VENV = Join-Path $SCRIPTS "python.exe"
$PIP = Join-Path $SCRIPTS "pip.exe"
$TRAIN = Join-Path $SCRIPTS "train.exe"

# Check if virtual environment exists
function Test-VirtualEnv {
    return Test-Path $PYTHON_VENV
}

# Commands
function Show-Help {
    Write-Host "DistilVit - Image Captioning Model Training" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\run.ps1 `<command`> [options]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Available commands:" -ForegroundColor Green
    Write-Host "  help              Show this help message"
    Write-Host "  install           Install all dependencies and set up environment"
    Write-Host "  clean             Clean up build artifacts and cache"
    Write-Host "  clean-all         Clean everything including cache and checkpoints"
    Write-Host "  train             Train model with specified dataset"
    Write-Host "  train-quick       Quick test training - 100 samples 1 epoch"
    Write-Host "  train-modern      Train with modern larger architecture - SmolLM-360M"
    Write-Host "  train-large       Train with large architecture - SmolLM-1.7B"
    Write-Host "  train-legacy      Train with legacy architecture - ViT DistilGPT2"
    Write-Host "  train-flickr      Train on Flickr30k dataset"
    Write-Host "  train-coco        Train on COCO dataset"
    Write-Host "  train-multi       Train on multiple datasets - Flickr and COCO"
    Write-Host "  test              Run inference test"
    Write-Host "  quantize          Quantize a trained model"
    Write-Host "  upload-hub        Upload model to HuggingFace Hub"
    Write-Host "  status            Show current environment status"
    Write-Host "  list-models       List available encoder and decoder options"
    Write-Host "  shell             Open Python shell with environment loaded"
    Write-Host "  lint              Run code linting"
    Write-Host "  format            Format code with black"
    Write-Host "  info              Show detailed project information"
    Write-Host ""
    Write-Host "Training parameters:" -ForegroundColor Green
    Write-Host "  -Dataset NAME     Dataset to train on - default: 'flickr pexels coco'"
    Write-Host "                    Options: flickr, coco, docornot, pexels, validation"
    Write-Host "  -Epochs N         Number of training epochs - default: 5"
    Write-Host "  -Sample N         Sample size for quick testing - default: full dataset"
    Write-Host "  -MaxLength N      Maximum caption length - default: 30"
    Write-Host "  -Encoder MODEL    Vision encoder model - default: google/siglip-base-patch16-224"
    Write-Host "  -Decoder MODEL    Language decoder model - default: HuggingFaceTB/SmolLM-135M"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\run.ps1 install"
    Write-Host "  .\run.ps1 train -Dataset flickr -Epochs 5"
    Write-Host "  .\run.ps1 train-quick"
    Write-Host "  .\run.ps1 train -Dataset 'flickr coco' -Epochs 5"
    Write-Host "  .\run.ps1 upload-hub -ModelId user/model-name -ModelPath ./checkpoints/model"
}

function Install-Dependencies {
    Write-Info "Checking Python version..."
    $pythonVersion = & $PYTHON --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Python not found. Please install Python 3.11+"
        exit 1
    }
    Write-Success "Python version: $pythonVersion"

    if (-not (Test-Path (Join-Path $VENV "pyvenv.cfg"))) {
        Write-Info "Creating virtual environment..."
        & $PYTHON -m venv $VENV
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to create virtual environment"
            exit 1
        }
        Write-Success "Virtual environment created"
    }

    Write-Info "Installing dependencies..."
    & $PIP install --upgrade pip setuptools wheel
    & $PIP install -r requirements.txt
    & $PIP install -e .

    Write-Success "Installation complete!"
    Write-Host ""
    Write-Host "To activate the virtual environment manually:" -ForegroundColor Cyan
    Write-Host "  .\Scripts\activate" -ForegroundColor Yellow
}

function Clean-Build {
    Write-Info "Cleaning build artifacts..."
    $paths = @("build", "dist", "*.egg-info", ".pytest_cache", "__pycache__")
    foreach ($path in $paths) {
        if (Test-Path $path) {
            Remove-Item -Recurse -Force $path
        }
    }
    Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Recurse -Filter "*.pyc" | Remove-Item -Force -ErrorAction SilentlyContinue
    Write-Success "Cleaned build artifacts"
}

function Clean-All {
    Clean-Build
    Write-Info "Cleaning cache and checkpoints..."
    $paths = @("cache", "checkpoints")
    foreach ($path in $paths) {
        if (Test-Path $path) {
            Remove-Item -Recurse -Force $path
        }
    }
    Write-Success "Cleaned all artifacts"
}

function Start-Training {
    param(
        [string]$DatasetName,
        [int]$NumEpochs,
        [string]$SampleSize,
        [int]$Length,
        [string]$EncoderModel,
        [string]$DecoderModel,
        [bool]$DeleteCheckpoints = $false
    )

    if (-not (Test-VirtualEnv)) {
        Write-Error "Virtual environment not found. Run: .\run.ps1 install"
        exit 1
    }

    Write-Info "Training with:"
    Write-Host "  Dataset: $DatasetName"
    Write-Host "  Epochs: $NumEpochs"
    Write-Host "  Max Length: $Length"
    Write-Host "  Encoder: $EncoderModel"
    Write-Host "  Decoder: $DecoderModel"
    if ($SampleSize) {
        Write-Host "  Sample: $SampleSize"
    }
    Write-Host ""

    # Build args array, splitting dataset names if multiple
    $args = @("--dataset")
    $args += $DatasetName -split '\s+'  # Split on whitespace
    $args += @(
        "--num-train-epochs", $NumEpochs,
        "--encoder-model", $EncoderModel,
        "--decoder-model", $DecoderModel,
        "--max-length", $Length
    )

    if ($SampleSize) {
        $args += "--sample", $SampleSize
    }

    if ($DeleteCheckpoints) {
        $args += "--delete-old-checkpoints"
    }

    & $TRAIN @args
}

function Show-Status {
    Write-Host "DistilVit Environment Status" -ForegroundColor Cyan
    Write-Host "==============================" -ForegroundColor Cyan

    if (Test-VirtualEnv) {
        Write-Host "[OK] Virtual environment: active" -ForegroundColor Green
        $pythonVersion = & $PYTHON_VENV --version 2>&1
        Write-Host "  Python: $pythonVersion"
        Write-Host "  Location: $VENV"
    } else {
        Write-Host "[X] Virtual environment: not found" -ForegroundColor Red
        Write-Host "  Run '.\run.ps1 install' to set up"
    }
    Write-Host ""

    if (Test-Path "checkpoints") {
        Write-Host "Checkpoints:"
        Get-ChildItem "checkpoints" | ForEach-Object {
            Write-Host "  $($_.Name) ($([math]::Round($_.Length/1MB, 2)) MB)"
        }
    } else {
        Write-Host "Checkpoints: none"
    }
    Write-Host ""

    if (Test-Path "cache") {
        $cacheSize = (Get-ChildItem "cache" -Recurse | Measure-Object -Property Length -Sum).Sum
        Write-Host "Cache size: $([math]::Round($cacheSize/1GB, 2)) GB"
    } else {
        Write-Host "Cache: not created yet"
    }
}

function Show-Models {
    Write-Host "Recommended Encoder Models:" -ForegroundColor Cyan
    Write-Host "  google/siglip-base-patch16-224       [86M]  - Default SigLIP-2 Base"
    Write-Host "  google/siglip-so400m-patch14-384     [400M] - SigLIP-2 larger variant"
    Write-Host "  google/vit-base-patch16-224          [86M]  - Legacy ViT"
    Write-Host ""
    Write-Host "Recommended Decoder Models:" -ForegroundColor Cyan
    Write-Host "  HuggingFaceTB/SmolLM-135M            [135M]  - Default efficient"
    Write-Host "  HuggingFaceTB/SmolLM-360M            [360M]  - Better quality"
    Write-Host "  HuggingFaceTB/SmolLM-1.7B            [1.7B]  - High quality"
    Write-Host "  microsoft/Phi-3-mini-4k-instruct     [3.8B]  - Premium quality"
    Write-Host "  Qwen/Qwen2-1.5B                      [1.5B]  - Alternative high quality"
    Write-Host "  distilbert/distilgpt2                [82M]   - Legacy"
    Write-Host ""
    Write-Host "Usage: .\run.ps1 train -Encoder MODEL -Decoder MODEL" -ForegroundColor Yellow
}

function Show-Info {
    Write-Host "DistilVit - Visual Encoder Decoder Model for Image Captioning" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Project Details:"
    Write-Host "  Name: distilvit"
    Write-Host "  Version: 0.1"
    Write-Host "  Python Required: 3.11"
    Write-Host "  Model Hub: https://huggingface.co/mozilla/distilvit"
    Write-Host ""
    Write-Host "Architecture:"
    Write-Host "  Default Encoder: SigLIP-2 Base (google/siglip-base-patch16-224)"
    Write-Host "  Default Decoder: SmolLM-135M (HuggingFaceTB/SmolLM-135M)"
    Write-Host "  Total Parameters: ~221M"
    Write-Host ""
    Write-Host "Documentation: See docs/architecture.md for detailed information"
    Write-Host "Repository: Check README.md for project details"
}

function Start-Test {
    if (-not (Test-VirtualEnv)) {
        Write-Error "Virtual environment not found. Run: .\run.ps1 install"
        exit 1
    }
    Write-Info "Running inference test..."
    & $PYTHON_VENV distilvit/infere.py
}

function Start-Quantize {
    if (-not $ModelPath) {
        Write-Error "ModelPath not set"
        Write-Host "Usage: .\run.ps1 quantize -ModelPath ./path/to/model" -ForegroundColor Yellow
        exit 1
    }
    Write-Info "Quantizing model at $ModelPath..."
    & $PYTHON_VENV distilvit/quantize.py --model_id $ModelPath --quantize --task image-to-text-with-past
}

function Start-Upload {
    if (-not $ModelId) {
        Write-Error "ModelId not set"
        Write-Host "Usage: .\run.ps1 upload-hub -ModelId user/model-name -ModelPath ./path/to/model" -ForegroundColor Yellow
        exit 1
    }
    if (-not $ModelPath) {
        Write-Error "ModelPath not set"
        Write-Host "Usage: .\run.ps1 upload-hub -ModelId user/model-name -ModelPath ./path/to/model" -ForegroundColor Yellow
        exit 1
    }
    Write-Info "Uploading model to HuggingFace Hub..."

    $args = @(
        "--model-id", $ModelId,
        "--save-path", $ModelPath,
        "--commit-message", $Message
    )
    if ($Tag) {
        $args += "--tag", $Tag
    }

    & $PYTHON_VENV distilvit/upload.py @args
}

function Start-Lint {
    if (-not (Test-VirtualEnv)) {
        Write-Error "Virtual environment not found. Run: .\run.ps1 install"
        exit 1
    }
    Write-Info "Running linter..."
    & $PIP list | Select-String "ruff" -Quiet
    if ($LASTEXITCODE -ne 0) {
        & $PIP install ruff
    }
    & $PYTHON_VENV -m ruff check distilvit/
}

function Start-Format {
    if (-not (Test-VirtualEnv)) {
        Write-Error "Virtual environment not found. Run: .\run.ps1 install"
        exit 1
    }
    Write-Info "Formatting code..."
    & $PIP list | Select-String "black" -Quiet
    if ($LASTEXITCODE -ne 0) {
        & $PIP install black
    }
    & $PYTHON_VENV -m black distilvit/
}

function Start-Shell {
    if (-not (Test-VirtualEnv)) {
        Write-Error "Virtual environment not found. Run: .\run.ps1 install"
        exit 1
    }
    Write-Info "Opening Python shell..."
    & $PYTHON_VENV
}

# Main command dispatcher
switch ($Command.ToLower()) {
    "help" { Show-Help }
    "install" { Install-Dependencies }
    "clean" { Clean-Build }
    "clean-all" { Clean-All }
    "train" { Start-Training -DatasetName $Dataset -NumEpochs $Epochs -SampleSize $Sample -Length $MaxLength -EncoderModel $Encoder -DecoderModel $Decoder -DeleteCheckpoints $DeleteOldCheckpoints }
    "train-quick" { Start-Training -DatasetName "validation" -NumEpochs 1 -SampleSize "100" -Length $MaxLength -EncoderModel $Encoder -DecoderModel $Decoder -DeleteCheckpoints $DeleteOldCheckpoints }
    "train-modern" { Start-Training -DatasetName $Dataset -NumEpochs $Epochs -SampleSize $Sample -Length 30 -EncoderModel $Encoder -DecoderModel "HuggingFaceTB/SmolLM-360M" -DeleteCheckpoints $DeleteOldCheckpoints }
    "train-large" { Start-Training -DatasetName $Dataset -NumEpochs $Epochs -SampleSize $Sample -Length 30 -EncoderModel "google/siglip-so400m-patch14-384" -DecoderModel "HuggingFaceTB/SmolLM-1.7B" -DeleteCheckpoints $DeleteOldCheckpoints }
    "train-legacy" { Start-Training -DatasetName $Dataset -NumEpochs $Epochs -SampleSize $Sample -Length 128 -EncoderModel "google/vit-base-patch16-224" -DecoderModel "distilbert/distilgpt2" -DeleteCheckpoints $DeleteOldCheckpoints }
    "train-flickr" { Start-Training -DatasetName "flickr" -NumEpochs $Epochs -SampleSize $Sample -Length $MaxLength -EncoderModel $Encoder -DecoderModel $Decoder -DeleteCheckpoints $DeleteOldCheckpoints }
    "train-coco" { Start-Training -DatasetName "coco" -NumEpochs $Epochs -SampleSize $Sample -Length $MaxLength -EncoderModel $Encoder -DecoderModel $Decoder -DeleteCheckpoints $DeleteOldCheckpoints }
    "train-multi" { Start-Training -DatasetName "flickr coco" -NumEpochs $Epochs -SampleSize $Sample -Length $MaxLength -EncoderModel $Encoder -DecoderModel $Decoder -DeleteCheckpoints $DeleteOldCheckpoints }
    "test" { Start-Test }
    "quantize" { Start-Quantize }
    "upload-hub" { Start-Upload }
    "status" { Show-Status }
    "list-models" { Show-Models }
    "shell" { Start-Shell }
    "lint" { Start-Lint }
    "format" { Start-Format }
    "info" { Show-Info }
    default {
        Write-Error "Unknown command: $Command"
        Write-Host ""
        Show-Help
        exit 1
    }
}
