"""
Quick test script to verify trained model works.

This script loads the complete model from a single model.safetensors file,
which contains:
- Vision encoder weights (SigLIP)
- Language model base weights (SmolLM)
- LoRA adapter weights (lora_A/lora_B matrices)
- Projection layer weights

No pretrained weights are downloaded from HuggingFace.
"""

import torch
from transformers import AutoTokenizer, AutoImageProcessor
from distilvit.prefix_model import PrefixConditioningVLM
from PIL import Image

# Model path - using the newly trained model with all fixes
# model_path = "D:/github/distilvit2/siglip-base-patch16-224-SmolLM-135M-lora"
model_path = "/Volumes/Shared/siglip-base-patch16-224-SmolLM-135M-merged/"

print("Loading model...")
try:
    # Load tokenizer from saved model
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load image processor - we need to know which encoder was used
    # Get this from config
    from distilvit.prefix_model import PrefixConditioningConfig
    config = PrefixConditioningConfig.from_pretrained(model_path)
    encoder_model_name = config.vision_config.get("_name_or_path", "google/siglip-base-patch16-224")
    image_processor = AutoImageProcessor.from_pretrained(encoder_model_name)

    # Initialize model architecture with random weights (will be overwritten)
    print("Initializing model architecture...")
    from transformers import SiglipVisionModel, AutoModelForCausalLM, SiglipVisionConfig
    from transformers import LlamaConfig

    # Create empty models from config (no pretrained weights download)
    vision_config = SiglipVisionConfig(**config.vision_config)
    vision_encoder = SiglipVisionModel(vision_config)

    text_config = LlamaConfig(**config.text_config)
    language_model = AutoModelForCausalLM.from_config(text_config)

    # Initialize full model
    model = PrefixConditioningVLM(
        config=config, vision_encoder=vision_encoder, language_model=language_model
    )

    # Set tokenizer
    model.tokenizer = tokenizer

    # Load all trained weights from single file
    print("Loading all weights from model.safetensors...")
    from safetensors.torch import load_file
    import os

    weights_path = os.path.join(model_path, "model.safetensors")
    state_dict = load_file(weights_path)

    # Remove 'module.' prefix from keys if present (added by DataParallel during training)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    load_result = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {len(load_result.missing_keys)}")
    print(f"Unexpected keys: {len(load_result.unexpected_keys)}")
    if load_result.missing_keys:
        print(f"First missing: {load_result.missing_keys[:3]}")
    if load_result.unexpected_keys:
        print(f"First unexpected: {load_result.unexpected_keys[:3]}")
    model.eval()

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"[OK] Model loaded successfully on {device}")

    # Load Mozilla/alt-text-validation dataset
    print("\nLoading Mozilla/alt-text-validation dataset...")
    from datasets import load_dataset

    dataset = load_dataset("Mozilla/alt-text-validation", split="train")
    print(f"[OK] Loaded dataset with {len(dataset)} images")

    # Test on 10 images
    num_images = min(10, len(dataset))
    print(f"\nTesting on {num_images} images...\n")
    print("=" * 80)

    for i in range(num_images):
        example = dataset[i]
        image = example['image'].convert("RGB")
        reference_alt_text = example['gpt_alt_text']

        # Process image
        pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        # Generate caption
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values=pixel_values,
                max_length=30,
                num_beams=3,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        # Decode
        generated_caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Display results
        print(f"\n[Image {i+1}/{num_images}]")
        print(f"Reference (GPT):  {reference_alt_text}")
        print(f"Generated:        {generated_caption}")
        print("-" * 80)

    print("\n[OK] Model test completed successfully!")

except Exception as e:
    print(f"\n[ERROR] Error testing model: {e}")
    import traceback

    traceback.print_exc()
