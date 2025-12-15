#!/usr/bin/env python3
"""
Compare outputs from old and new architecture models.
"""
import torch
from datasets import load_dataset
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoImageProcessor
from distilvit.prefix_model import PrefixConditioningVLM

def load_old_model(path):
    """Load old cross-attention model."""
    model = VisionEncoderDecoderModel.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    image_processor = AutoImageProcessor.from_pretrained(path)
    return model, tokenizer, image_processor

def load_new_model(path):
    """Load new prefix-conditioning model."""
    from transformers import AutoModelForCausalLM
    from transformers import SiglipVisionModel
    from distilvit.prefix_model import PrefixConditioningConfig
    import json

    # Load config manually
    with open(f"{path}/config.json", "r") as f:
        config_dict = json.load(f)

    # Create config object
    config = PrefixConditioningConfig(**config_dict)

    # Load components from scratch (will be overwritten by saved weights)
    vision_encoder = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
    language_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")

    # Create model
    model = PrefixConditioningVLM(config, vision_encoder, language_model)

    # Load trained weights (safetensors format)
    from safetensors.torch import load_file
    state_dict = load_file(f"{path}/model.safetensors")
    model.load_state_dict(state_dict, strict=False)

    tokenizer = AutoTokenizer.from_pretrained(path)
    image_processor = AutoImageProcessor.from_pretrained("google/siglip-base-patch16-224")

    return model, tokenizer, image_processor

def generate_caption_old(model, tokenizer, image_processor, image, device):
    """Generate caption with old model."""
    pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)
    # Use greedy search instead of beam search to avoid GPT2 issues
    generated_ids = model.generate(
        pixel_values,
        max_length=50,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return caption

def generate_caption_new(model, tokenizer, image_processor, image, device):
    """Generate caption with new model."""
    # Set tokenizer for model
    model.tokenizer = tokenizer
    pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)
    # Use model's generate method (it handles do_sample internally)
    generated_ids = model.generate(
        pixel_values,
        max_length=30,
        temperature=0.0  # Greedy decoding
    )
    caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return caption

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    print()

    # Model paths
    old_model_path = "./siglip-base-patch16-224-gpt2"
    new_model_path = "./siglip-base-patch16-224-SmolLM-135M-lora"

    print("Loading models...")
    print("- Old model (cross-attention + GPT2):", old_model_path)
    try:
        old_model, old_tokenizer, old_processor = load_old_model(old_model_path)
        old_model.to(device)
        old_model.eval()
        print("  ✓ Loaded successfully")
    except Exception as e:
        print(f"  ✗ Failed to load: {e}")
        old_model = None

    print("- New model (prefix-conditioning + LoRA):", new_model_path)
    try:
        new_model, new_tokenizer, new_processor = load_new_model(new_model_path)
        new_model.to(device)
        new_model.eval()
        print("  ✓ Loaded successfully")
    except Exception as e:
        print(f"  ✗ Failed to load: {e}")
        new_model = None

    if old_model is None and new_model is None:
        print("\nERROR: Both models failed to load!")
        return

    # Load test images
    print("\nLoading validation dataset...")
    dataset = load_dataset("mozilla/alt-text-validation")
    test_samples = [
        item for item in dataset["train"]
        if item["alt_text"] != ""
    ][:5]  # Take first 5 samples

    print(f"Testing on {len(test_samples)} images\n")
    print("=" * 100)

    # Generate captions
    with torch.no_grad():
        for idx, sample in enumerate(test_samples, 1):
            image = sample["image"]
            ground_truth = sample["alt_text"]

            print(f"\n**Image {idx}:**")
            print(f"Ground Truth: {ground_truth}")

            if old_model:
                try:
                    old_caption = generate_caption_old(
                        old_model, old_tokenizer, old_processor, image, device
                    )
                    print(f"Old Model:    {old_caption}")
                except Exception as e:
                    print(f"Old Model:    ERROR - {e}")

            if new_model:
                try:
                    new_caption = generate_caption_new(
                        new_model, new_tokenizer, new_processor, image, device
                    )
                    print(f"New Model:    {new_caption}")
                except Exception as e:
                    print(f"New Model:    ERROR - {e}")

            print("-" * 100)

if __name__ == "__main__":
    main()
