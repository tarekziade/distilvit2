"""
Compare new model (SigLIP + SmolLM + LoRA) vs old Mozilla/distilvit v0.5.0 (ViT + GPT2)
Using CLIPScore for evaluation
"""
import torch
from transformers import AutoTokenizer, AutoImageProcessor, VisionEncoderDecoderModel
from distilvit.prefix_model import PrefixConditioningVLM, PrefixConditioningConfig
from transformers import SiglipVisionModel, AutoModelForCausalLM
from PIL import Image
import requests
from io import BytesIO
import numpy as np

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

print("="*80)
print("MODEL COMPARISON: New (SigLIP+SmolLM+LoRA) vs Old Mozilla/distilvit v0.5.0")
print("="*80)

# ============================================================================
# Load CLIP for scoring
# ============================================================================
print("\n[1/4] Loading CLIP model for scoring...")
try:
    import clip
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    print("[OK] CLIP model loaded")
except ImportError:
    print("[ERROR] Please install clip: pip install git+https://github.com/openai/CLIP.git")
    exit(1)

# ============================================================================
# Load OLD model (Mozilla/distilvit v0.5.0)
# ============================================================================
print("\n[2/4] Loading OLD model (Mozilla/distilvit v0.5.0)...")
try:
    old_model = VisionEncoderDecoderModel.from_pretrained("Mozilla/distilvit", revision="v0.5.0")
    old_tokenizer = AutoTokenizer.from_pretrained("Mozilla/distilvit", revision="v0.5.0")
    old_image_processor = AutoImageProcessor.from_pretrained("Mozilla/distilvit", revision="v0.5.0")
    old_model = old_model.to(device)
    old_model.eval()
    print("[OK] Old model loaded (ViT-base + DistilGPT2)")
except Exception as e:
    print(f"[WARNING] Could not load Mozilla/distilvit: {e}")
    print("Continuing with new model only...")
    old_model = None

# ============================================================================
# Load NEW model (siglip-base-patch16-224-SmolLM-135M-lora)
# ============================================================================
print("\n[3/4] Loading NEW model (newly trained with fixes)...")
new_model_path = "D:/github/distilvit2/siglip-base-patch16-224-SmolLM-135M-lora"
encoder_model_name = "google/siglip-base-patch16-224"
decoder_model_name = "HuggingFaceTB/SmolLM-135M"

# Load components
new_tokenizer = AutoTokenizer.from_pretrained(new_model_path)
new_image_processor = AutoImageProcessor.from_pretrained(encoder_model_name)
vision_encoder = SiglipVisionModel.from_pretrained(encoder_model_name)
language_model = AutoModelForCausalLM.from_pretrained(decoder_model_name)

# Create model
config = PrefixConditioningConfig.from_pretrained(new_model_path)
new_model = PrefixConditioningVLM(
    config=config,
    vision_encoder=vision_encoder,
    language_model=language_model
)
new_model.tokenizer = new_tokenizer

# Load trained weights
from safetensors.torch import load_file
import os
weights_path = os.path.join(new_model_path, "model.safetensors")
state_dict = load_file(weights_path)

# Remove 'module.' prefix if present
if any(k.startswith('module.') for k in state_dict.keys()):
    state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

new_model.load_state_dict(state_dict, strict=False)
new_model = new_model.to(device)
new_model.eval()
print("[OK] New model loaded (SigLIP-base + SmolLM-135M + LoRA)")

# ============================================================================
# Test images
# ============================================================================
print("\n[4/4] Loading test images...")
test_images = [
    {
        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg",
        "name": "Diagram"
    },
    {
        "url": "https://images.unsplash.com/photo-1517849845537-4d257902454a?w=400",
        "name": "Dog"
    },
    {
        "url": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400",
        "name": "Mountain landscape"
    },
    {
        "url": "https://images.unsplash.com/photo-1551782450-a2132b4ba21d?w=400",
        "name": "Burger"
    },
    {
        "url": "https://images.unsplash.com/photo-1519125323398-675f0ddb6308?w=400",
        "name": "Coffee"
    },
]

images = []
for img_info in test_images:
    try:
        response = requests.get(img_info["url"], timeout=10)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        images.append((image, img_info["name"]))
        print(f"  [OK] Loaded: {img_info['name']}")
    except Exception as e:
        print(f"  [SKIP] Failed to load {img_info['name']}: {e}")

print(f"\n[OK] Loaded {len(images)} test images")

# ============================================================================
# Generate captions and compute CLIPScore
# ============================================================================
print("\n" + "="*80)
print("EVALUATION RESULTS")
print("="*80)

def compute_clipscore(image, caption, clip_model, preprocess):
    """Compute CLIPScore for image-caption pair"""
    # Preprocess image for CLIP
    image_input = preprocess(image).unsqueeze(0).to(device)

    # Tokenize caption for CLIP
    text_input = clip.tokenize([caption], truncate=True).to(device)

    # Get CLIP features
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_input)

        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute similarity (CLIPScore)
        similarity = (image_features @ text_features.T).item()

    return similarity * 100  # Scale to 0-100

old_scores = []
new_scores = []

for idx, (image, name) in enumerate(images):
    print(f"\n{'='*80}")
    print(f"Image {idx+1}/{len(images)}: {name}")
    print(f"{'='*80}")

    # Generate with OLD model
    if old_model is not None:
        try:
            pixel_values_old = old_image_processor(images=image, return_tensors="pt").pixel_values
            pixel_values_old = pixel_values_old.to(device)

            with torch.no_grad():
                output_ids_old = old_model.generate(
                    pixel_values_old,
                    max_length=30,
                    num_beams=3,
                    early_stopping=True,
                )

            old_caption = old_tokenizer.decode(output_ids_old[0], skip_special_tokens=True)
            old_score = compute_clipscore(image, old_caption, clip_model, preprocess)
            old_scores.append(old_score)

            print(f"\nOLD (ViT + DistilGPT2):")
            print(f"  Caption: {old_caption}")
            print(f"  CLIPScore: {old_score:.2f}")
        except Exception as e:
            print(f"\nOLD model failed: {e}")
            old_caption = "[ERROR]"
            old_score = 0

    # Generate with NEW model
    try:
        pixel_values_new = new_image_processor(images=image, return_tensors="pt").pixel_values
        pixel_values_new = pixel_values_new.to(device)

        with torch.no_grad():
            output_ids_new = new_model.generate(
                pixel_values=pixel_values_new,
                max_length=30,
                num_beams=3,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        new_caption = new_tokenizer.decode(output_ids_new[0], skip_special_tokens=True)
        new_score = compute_clipscore(image, new_caption, clip_model, preprocess)
        new_scores.append(new_score)

        print(f"\nNEW (SigLIP + SmolLM + LoRA):")
        print(f"  Caption: {new_caption}")
        print(f"  CLIPScore: {new_score:.2f}")
    except Exception as e:
        print(f"\nNEW model failed: {e}")
        import traceback
        traceback.print_exc()
        new_caption = "[ERROR]"
        new_score = 0

    # Compare
    if old_model is not None:
        diff = new_score - old_score
        winner = "NEW" if diff > 0 else "OLD" if diff < 0 else "TIE"
        print(f"\nDifference: {diff:+.2f} (Winner: {winner})")

# ============================================================================
# Summary statistics
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

if old_scores:
    print(f"\nOLD Model (ViT + DistilGPT2) - Mozilla/distilvit v0.5.0:")
    print(f"  Average CLIPScore: {np.mean(old_scores):.2f}")
    print(f"  Std Dev: {np.std(old_scores):.2f}")
    print(f"  Min: {np.min(old_scores):.2f}, Max: {np.max(old_scores):.2f}")

if new_scores:
    print(f"\nNEW Model (SigLIP + SmolLM + LoRA):")
    print(f"  Average CLIPScore: {np.mean(new_scores):.2f}")
    print(f"  Std Dev: {np.std(new_scores):.2f}")
    print(f"  Min: {np.min(new_scores):.2f}, Max: {np.max(new_scores):.2f}")

if old_scores and new_scores:
    avg_improvement = np.mean(new_scores) - np.mean(old_scores)
    print(f"\nImprovement: {avg_improvement:+.2f} CLIPScore points")
    pct_improvement = (avg_improvement / np.mean(old_scores)) * 100
    print(f"Relative improvement: {pct_improvement:+.1f}%")

    wins = sum(1 for n, o in zip(new_scores, old_scores) if n > o)
    losses = sum(1 for n, o in zip(new_scores, old_scores) if n < o)
    ties = len(new_scores) - wins - losses

    print(f"\nWin/Loss/Tie: {wins}/{losses}/{ties}")

print("\n" + "="*80)
print("\nNOTE: CLIPScore measures image-text alignment.")
print("Higher scores indicate better semantic matching between image and caption.")
print("="*80)
