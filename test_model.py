"""
Quick test script to verify trained model works
"""
import torch
from transformers import AutoTokenizer, AutoImageProcessor
from distilvit.prefix_model import PrefixConditioningVLM
from PIL import Image
import requests
from io import BytesIO

# Model path - using the newly trained model with all fixes
model_path = "D:/github/distilvit2/siglip-base-patch16-224-SmolLM-135M-lora"

print("Loading model...")
try:
    # Load tokenizer from saved model
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load image processor from original encoder (not saved with model)
    encoder_model_name = "google/siglip-base-patch16-224"
    decoder_model_name = "HuggingFaceTB/SmolLM-135M"
    image_processor = AutoImageProcessor.from_pretrained(encoder_model_name)

    # Load encoder and decoder
    print("Loading encoder and decoder...")
    from transformers import SiglipVisionModel, AutoModelForCausalLM
    from distilvit.prefix_model import PrefixConditioningConfig

    vision_encoder = SiglipVisionModel.from_pretrained(encoder_model_name)
    language_model = AutoModelForCausalLM.from_pretrained(decoder_model_name)

    # Create config
    config = PrefixConditioningConfig.from_pretrained(model_path)

    # Initialize model with encoder/decoder
    model = PrefixConditioningVLM(
        config=config,
        vision_encoder=vision_encoder,
        language_model=language_model
    )

    # Set tokenizer (set externally after init)
    model.tokenizer = tokenizer

    # Load trained weights
    print("Loading trained weights...")
    from safetensors.torch import load_file
    import os
    weights_path = os.path.join(model_path, "model.safetensors")
    state_dict = load_file(weights_path)

    # Remove 'module.' prefix from keys if present (added by DataParallel during training)
    # Only strip if keys actually have the prefix (for backward compatibility)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

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

    # Test with a sample image (using a URL or local file)
    print("\nTesting inference...")

    # Try to use an image from the internet (small test image)
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"

    try:
        response = requests.get(image_url, timeout=10)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        print("[OK] Loaded test image from URL")
    except Exception as e:
        print(f"Could not load image from URL: {e}")
        print("Creating a blank test image instead")
        image = Image.new("RGB", (224, 224), color=(128, 128, 128))

    # Process image
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    # Generate caption
    print("Generating caption...")
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values=pixel_values,
            max_length=30,
            num_beams=3,
            repetition_penalty=1.2,  # Penalize repetition
            no_repeat_ngram_size=3,  # Don't repeat 3-grams
            early_stopping=True,
        )

    print(f"Generated token IDs shape: {generated_ids.shape}")
    print(f"Generated token IDs: {generated_ids[0].tolist()}")

    # Decode
    caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    caption_with_special = tokenizer.decode(generated_ids[0], skip_special_tokens=False)

    print("\n" + "="*60)
    print("GENERATED CAPTION (without special tokens):")
    print(repr(caption))
    print("\nGENERATED CAPTION (with special tokens):")
    print(repr(caption_with_special))
    print("="*60)

    print("\n[OK] Model test completed successfully!")

except Exception as e:
    print(f"\n[ERROR] Error testing model: {e}")
    import traceback
    traceback.print_exc()
