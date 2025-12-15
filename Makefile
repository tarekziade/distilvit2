# Makefile for DistilVit Image Captioning Model
# Requires Python 3.11

.PHONY: help install clean train train-quick train-modern train-large train-legacy test quantize upload-hub lint check-python

# Default target - show help
.DEFAULT_GOAL := help

# Python version check
PYTHON := python3.11
REQUIRED_PYTHON := 3.11

# Virtual environment paths
VENV := .
BIN := $(VENV)/bin
PYTHON_VENV := $(BIN)/python
PIP := $(BIN)/pip
TRAIN := $(BIN)/train

# Training defaults
DATASET ?= flickr
EPOCHS ?= 3
SAMPLE ?=
MAX_LENGTH ?= 30
ENCODER ?= google/siglip-base-patch16-224
DECODER ?= HuggingFaceTB/SmolLM-135M

help: ## Show this help message
	@echo "DistilVit - Image Captioning Model Training"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Environment variables:"
	@echo "  DATASET     Dataset to train on (default: flickr)"
	@echo "              Options: flickr, coco, docornot, pexels, validation, all"
	@echo "  EPOCHS      Number of training epochs (default: 3)"
	@echo "  SAMPLE      Sample size for quick testing (default: full dataset)"
	@echo "  MAX_LENGTH  Maximum caption length (default: 128)"
	@echo "  ENCODER     Vision encoder model (default: google/siglip-base-patch16-224)"
	@echo "  DECODER     Language decoder model (default: HuggingFaceTB/SmolLM-135M)"
	@echo ""
	@echo "Examples:"
	@echo "  make install                    # Set up environment"
	@echo "  make train DATASET=flickr       # Train on Flickr dataset"
	@echo "  make train-quick                # Quick test training"
	@echo "  make train DATASET='flickr coco' EPOCHS=5  # Multi-dataset training"

check-python: ## Check Python version
	@$(PYTHON) --version 2>/dev/null || (echo "Error: Python 3.11 not found. Please install Python 3.11" && exit 1)
	@echo "✓ Python version check passed"

$(VENV)/pyvenv.cfg: check-python ## Create virtual environment
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "✓ Virtual environment created"

install: $(VENV)/pyvenv.cfg ## Install all dependencies and set up environment
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo ""
	@echo "✓ Installation complete!"
	@echo ""
	@echo "To activate the virtual environment manually:"
	@echo "  source bin/activate"

clean: ## Clean up build artifacts, cache, and checkpoints
	@echo "Cleaning up..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✓ Cleaned build artifacts"

clean-all: clean ## Clean everything including cache and checkpoints
	@echo "Cleaning cache and checkpoints..."
	rm -rf cache/
	rm -rf checkpoints/
	@echo "✓ Cleaned all artifacts"

clean-venv: ## Remove virtual environment (use with caution!)
	@echo "Removing virtual environment..."
	@read -p "Are you sure you want to remove the virtual environment? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf bin/ lib/ include/ pyvenv.cfg share/ lib64 2>/dev/null || true; \
		echo "✓ Virtual environment removed"; \
	else \
		echo "Cancelled"; \
	fi

train: $(TRAIN) ## Train model with specified dataset (use DATASET, EPOCHS, SAMPLE vars)
	@echo "Training with:"
	@echo "  Dataset: $(DATASET)"
	@echo "  Epochs: $(EPOCHS)"
	@echo "  Max Length: $(MAX_LENGTH)"
	@echo "  Encoder: $(ENCODER)"
	@echo "  Decoder: $(DECODER)"
	@if [ -n "$(SAMPLE)" ]; then echo "  Sample: $(SAMPLE)"; fi
	@echo ""
	$(TRAIN) \
		--dataset $(DATASET) \
		--num-train-epochs $(EPOCHS) \
		--encoder-model $(ENCODER) \
		--decoder-model $(DECODER) \
		--max-length $(MAX_LENGTH) \
		$(if $(SAMPLE),--sample $(SAMPLE),)

train-quick: ## Quick test training (validation dataset, 100 samples, 1 epoch)
	@echo "Quick test training..."
	$(MAKE) train DATASET=validation SAMPLE=100 EPOCHS=1

train-modern: ## Train with modern larger architecture (SigLIP-2 + SmolLM-360M)
	@echo "Training with modern larger architecture..."
	$(MAKE) train DECODER=HuggingFaceTB/SmolLM-360M MAX_LENGTH=30 EPOCHS=$(EPOCHS)

train-large: ## Train with large architecture (SigLIP-SO400M + SmolLM-1.7B)
	@echo "Training with large architecture..."
	$(MAKE) train ENCODER=google/siglip-so400m-patch14-384 DECODER=HuggingFaceTB/SmolLM-1.7B MAX_LENGTH=30 EPOCHS=$(EPOCHS)

train-legacy: ## Train with legacy architecture (ViT + DistilGPT2)
	@echo "Training with legacy architecture..."
	$(MAKE) train \
		ENCODER=google/vit-base-patch16-224 \
		DECODER=distilbert/distilgpt2 \
		MAX_LENGTH=128

train-all: ## Train on all datasets (requires 2TB disk space!)
	@echo "WARNING: Training on all datasets requires 2TB of disk space"
	@read -p "Continue? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(MAKE) train DATASET=all EPOCHS=$(EPOCHS); \
	else \
		echo "Cancelled"; \
	fi

train-flickr: ## Train on Flickr30k dataset
	$(MAKE) train DATASET=flickr

train-coco: ## Train on COCO dataset
	$(MAKE) train DATASET=coco

train-multi: ## Train on multiple datasets (Flickr + COCO)
	$(MAKE) train DATASET='flickr coco'

test: $(PYTHON_VENV) ## Run inference test comparing models
	@echo "Running inference test..."
	$(PYTHON_VENV) distilvit/infere.py

quantize: $(PYTHON_VENV) ## Quantize a trained model (requires MODEL_PATH)
ifndef MODEL_PATH
	@echo "Error: MODEL_PATH not set"
	@echo "Usage: make quantize MODEL_PATH=./path/to/model"
	@exit 1
endif
	@echo "Quantizing model at $(MODEL_PATH)..."
	$(PYTHON_VENV) distilvit/quantize.py \
		--model_id $(MODEL_PATH) \
		--quantize \
		--task image-to-text-with-past

upload-hub: $(PYTHON_VENV) ## Upload model to HuggingFace Hub (requires MODEL_ID and MODEL_PATH)
ifndef MODEL_ID
	@echo "Error: MODEL_ID not set"
	@echo "Usage: make upload-hub MODEL_ID=user/model-name MODEL_PATH=./path/to/model"
	@exit 1
endif
ifndef MODEL_PATH
	@echo "Error: MODEL_PATH not set"
	@echo "Usage: make upload-hub MODEL_ID=user/model-name MODEL_PATH=./path/to/model"
	@exit 1
endif
	@echo "Uploading model to HuggingFace Hub..."
	$(PYTHON_VENV) distilvit/upload.py \
		--model-id $(MODEL_ID) \
		--save-path $(MODEL_PATH) \
		$(if $(TAG),--tag $(TAG),) \
		--commit-message "$(if $(MESSAGE),$(MESSAGE),New training run)"

status: ## Show current environment status
	@echo "DistilVit Environment Status"
	@echo "=============================="
	@if [ -f $(PYTHON_VENV) ]; then \
		echo "✓ Virtual environment: active"; \
		echo "  Python: $$($(PYTHON_VENV) --version)"; \
		echo "  Location: $(VENV)"; \
	else \
		echo "✗ Virtual environment: not found"; \
		echo "  Run 'make install' to set up"; \
	fi
	@echo ""
	@if [ -d checkpoints ]; then \
		echo "Checkpoints:"; \
		ls -lh checkpoints/ | tail -n +2 | awk '{print "  " $$9 " (" $$5 ")"}' || echo "  (empty)"; \
	else \
		echo "Checkpoints: none"; \
	fi
	@echo ""
	@if [ -d cache ]; then \
		CACHE_SIZE=$$(du -sh cache 2>/dev/null | cut -f1); \
		echo "Cache size: $$CACHE_SIZE"; \
	else \
		echo "Cache: not created yet"; \
	fi

list-models: ## List available encoder and decoder options
	@echo "Recommended Encoder Models:"
	@echo "  google/siglip-base-patch16-224       (86M)  - Default, SigLIP-2 Base"
	@echo "  google/siglip-so400m-patch14-384     (400M) - SigLIP-2 larger variant"
	@echo "  google/vit-base-patch16-224          (86M)  - Legacy ViT"
	@echo ""
	@echo "Recommended Decoder Models:"
	@echo "  HuggingFaceTB/SmolLM-135M            (135M)  - Default, efficient"
	@echo "  HuggingFaceTB/SmolLM-360M            (360M)  - Better quality"
	@echo "  HuggingFaceTB/SmolLM-1.7B            (1.7B)  - High quality"
	@echo "  microsoft/Phi-3-mini-4k-instruct     (3.8B)  - Premium quality"
	@echo "  Qwen/Qwen2-1.5B                      (1.5B)  - Alternative high quality"
	@echo "  distilbert/distilgpt2                (82M)   - Legacy"
	@echo ""
	@echo "Usage: make train ENCODER=<model> DECODER=<model>"

shell: $(PYTHON_VENV) ## Open Python shell with environment loaded
	@echo "Opening Python shell..."
	$(PYTHON_VENV)

notebook: $(PYTHON_VENV) ## Launch Jupyter notebook (requires jupyter to be installed)
	@$(PYTHON_VENV) -c "import jupyter" 2>/dev/null || (echo "Installing jupyter..." && $(PIP) install jupyter)
	$(BIN)/jupyter notebook

lint: $(PYTHON_VENV) ## Run code linting (requires ruff/flake8)
	@echo "Running linter..."
	@$(PYTHON_VENV) -m pip list | grep -q ruff || $(PIP) install ruff
	$(PYTHON_VENV) -m ruff check distilvit/ || true

format: $(PYTHON_VENV) ## Format code with black
	@echo "Formatting code..."
	@$(PYTHON_VENV) -m pip list | grep -q black || $(PIP) install black
	$(PYTHON_VENV) -m black distilvit/

requirements-update: $(PYTHON_VENV) ## Update requirements.txt with current environment
	$(PIP) freeze > requirements.txt
	@echo "✓ requirements.txt updated"

info: ## Show detailed project information
	@echo "DistilVit - Visual Encoder Decoder Model for Image Captioning"
	@echo ""
	@echo "Project Details:"
	@echo "  Name: distilvit"
	@echo "  Version: 0.1"
	@echo "  Python Required: 3.11"
	@echo "  Model Hub: https://huggingface.co/mozilla/distilvit"
	@echo ""
	@echo "Architecture:"
	@echo "  Default Encoder: SigLIP-2 Base (google/siglip-base-patch16-224)"
	@echo "  Default Decoder: SmolLM-135M (HuggingFaceTB/SmolLM-135M)"
	@echo "  Total Parameters: ~221M"
	@echo ""
	@echo "Documentation: See CLAUDE.md for detailed information"
	@echo "Repository: Check README.md for project details"

.PHONY: all
all: install ## Install and set up everything
	@echo ""
	@echo "Setup complete! Try:"
	@echo "  make train-quick    # Quick test"
	@echo "  make train          # Full training"
	@echo "  make help           # Show all options"
