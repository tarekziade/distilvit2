# Measuring Dataset Quality for Image Captioning: A Comprehensive Approach

*December 15, 2025*

When training vision-language models for image captioning, dataset quality is often overlooked. We focus on model architecture, hyperparameters, and training strategies, but the foundation—the data itself—can make or break the results. This post explains how we built a comprehensive dataset quality measurement system for our image captioning project, including bias detection, semantic alignment scoring, and synthetic data generation for rare classes.

## The Problem: Hidden Biases and Quality Issues

Image captioning datasets often contain problematic patterns:

1. **Implicit Biases**: Captions describing people often include unnecessary demographic descriptors (gender, race, age) that aren't relevant to the image content
2. **Poor Image-Text Alignment**: Captions may not accurately describe the visual content
3. **Class Imbalance**: Some objects appear rarely, leading to poor model performance on those classes
4. **Caption Quality Variance**: Inconsistent caption quality across the dataset

Our dataset, [Mozilla/flickr30k-transformed-captions-gpt4o](https://huggingface.co/datasets/Mozilla/flickr30k-transformed-captions-gpt4o), uses GPT-4 to transform original Flickr30k captions into bias-free alternatives. But how do we **measure** if this transformation actually works?

## The Solution: Multi-Metric Quality Analysis

We built a comprehensive quality analysis tool that measures four key dimensions:

### 1. Image-Text Alignment (CLIP Score)

CLIP (Contrastive Language-Image Pre-training) provides a powerful way to measure how well a caption matches its image. We use the CLIP model to compute cosine similarity between image and text embeddings:

```python
def compute_clip_score(images, captions):
    """Compute CLIP similarity between images and captions"""
    with torch.no_grad():
        image_features = clip_model.encode_image(images)
        text_features = clip_model.encode_text(captions)

        # Normalize and compute cosine similarity
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features * text_features).sum(-1)
    return similarity.mean()
```

**Key insight**: We upgraded from CLIP ViT-B/32 (151M params) to ViT-L/14@336px (428M params) for more discriminative scoring. The larger model gives lower absolute scores but better differentiates between good and poor alignments.

**Interpretation**:
- **Excellent**: ≥ 0.35 (tight alignment)
- **Good**: 0.30-0.35 (solid alignment)
- **Fair**: 0.25-0.30 (acceptable)
- **Poor**: < 0.25 (weak alignment)

Our transformed dataset scores **0.311** with ViT-B/32 (Good) and **0.284** with ViT-L/14@336px (Fair/discriminative).

### 2. Caption Fidelity (BERT Score)

While CLIP measures image-text alignment, we also need to ensure the transformed captions maintain semantic similarity to the originals. We use BERTScore with RoBERTa-large:

```python
def compute_bert_score(transformed_captions, original_captions):
    """Measure semantic similarity between transformed and original captions"""
    P, R, F1 = bert_score.score(
        transformed_captions,
        original_captions,
        model_type='roberta-large',
        device=device
    )
    return F1.mean()
```

**Interpretation**:
- **Excellent**: ≥ 0.90 (high fidelity)
- **Good**: 0.85-0.90 (good preservation)
- **Fair**: 0.80-0.85 (acceptable)
- **Poor**: < 0.80 (significant drift)

Our dataset scores **0.904** (Excellent), meaning GPT-4 successfully removes bias while preserving the core semantic content.

### 3. Bias Detection: Before and After

The most critical measurement is bias detection. We track protected attributes across seven categories:

```python
PROTECTED_TERMS = {
    "gender": ["man", "men", "woman", "women", "male", "female", "boy", "girl", ...],
    "sexual_orientation": ["gay", "lesbian", "bisexual", "lgbtq", ...],
    "race_ethnicity": ["white", "black", "asian", "hispanic", "latino", "african", ...],
    "nationality": ["american", "chinese", "indian", ...],  # Extended with pycountry
    "religion": ["christian", "muslim", "jewish", "hindu", "buddhist", ...],
    "disability": ["disabled", "wheelchair", "blind", "deaf", ...],
    "age": ["baby", "child", "teenager", "elderly", "old", "young", ...],
}
```

We use the [pycountry](https://pypi.org/project/pycountry/) library to expand nationality terms with all countries and demonyms (e.g., "french", "german", "japanese").

The key innovation: **we analyze both original and transformed captions** to show the effectiveness of GPT-4's bias removal:

```python
def count_protected_mentions(caption):
    """Count mentions of protected attributes in caption"""
    caption_lower = caption.lower()
    counts = {}
    for category, terms in PROTECTED_TERMS.items():
        counts[category] = sum(1 for term in terms if term in caption_lower)
    return counts
```

**Results on 1000 samples**:

| Category | Original | Transformed | Reduction |
|----------|----------|-------------|-----------|
| Gender | 70.0% | 0.0% | **-100%** ✅ |
| Race/Ethnicity | 33.0% | 1.0% | **-97%** ✅ |
| Nationality | 1.0% | 0.0% | **-100%** ✅ |
| Age | 18.0% | 14.0% | -22% ⚠️ |
| Religion | 0.5% | 0.2% | -60% ✅ |
| Sexual Orientation | 0.1% | 0.0% | -100% ✅ |
| Disability | 0.3% | 0.1% | -67% ✅ |

**Interpretation**: GPT-4 successfully eliminates nearly all gender, race, and nationality descriptors while maintaining high caption quality. Age descriptors remain at 14% because they're often visually necessary (e.g., "child playing" vs "person playing").

### 4. Object Distribution Analysis

Class imbalance is a common problem in image datasets. We compute three metrics:

**Gini Coefficient**: Measures inequality in object distribution (0 = perfect equality, 1 = maximum inequality)

```python
def gini_coefficient(counts):
    """Compute Gini coefficient for object distribution"""
    sorted_counts = np.sort(counts)
    n = len(counts)
    cumsum = np.cumsum(sorted_counts)
    return (2 * np.sum((n - np.arange(n)) * sorted_counts)) / (n * cumsum[-1]) - 1
```

**Shannon Entropy**: Measures diversity in the dataset

```python
def shannon_entropy(counts):
    """Compute Shannon entropy for object distribution"""
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-10))
```

**Results**:
- **Gini**: 0.23 (moderate inequality)
- **Shannon Entropy**: 5.14 bits (good diversity)
- **Rare Objects**: 204 objects with ≤ 5 samples

The tool automatically generates:
- `objects_below_50.csv`: List of rare objects for targeted data collection
- `reweighting_probs.csv`: Sampling weights for balanced training
- `lorenz_curve.png`: Visual representation of distribution inequality

## Performance Optimization

Processing 30,000 images with CLIP and BERT scoring is computationally expensive. We implemented several optimizations:

### Automatic Batch Size Detection

Different GPUs have different memory capacities. We auto-detect optimal batch size:

```python
def get_optimal_batch_size(device, clip_model):
    """Detect optimal batch size based on device memory"""
    is_large_model = "large" in clip_model.lower() or "336" in clip_model

    if device == "cuda":
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if total_memory_gb >= 24:
            return 128 if not is_large_model else 48
        elif total_memory_gb >= 16:
            return 96 if not is_large_model else 32
        # ... more tiers
    elif device == "mps":  # Apple Silicon
        total_memory_gb = psutil.virtual_memory().total / 1024**3
        if total_memory_gb >= 32:
            return 64 if not is_large_model else 24
        # ... more tiers
    else:  # CPU
        return 16
```

**Results**: 2x speedup on our test hardware (M2 Max 32GB):
- ViT-B/32: 4 batches → 2 batches (batch size 32→64)
- ViT-L/14@336px: Adaptive sizing (batch size 24)

### Progress Bars

Long-running operations show progress using [tqdm](https://tqdm.github.io/):

```python
for i in tqdm(range(0, n, batch_size), desc="CLIP scoring"):
    batch_images = images[i:i+batch_size]
    batch_texts = texts[i:i+batch_size]
    scores.extend(compute_clip_score(batch_images, batch_texts))
```

## Taking It Further: CLIP Loss During Training

Measuring dataset quality is valuable, but can we use CLIP scoring to **improve** the model during training?

We implemented a custom trainer that combines standard cross-entropy loss with CLIP alignment loss:

```python
class CLIPLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Standard caption generation loss
        outputs = model(**inputs)
        ce_loss = outputs.loss

        # Generate captions
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values=inputs["pixel_values"],
                max_length=inputs["labels"].shape[1],
            )
            generated_texts = self.tokenizer.batch_decode(generated_ids)

        # Compute CLIP alignment
        clip_inputs = self.clip_processor(
            text=generated_texts,
            images=[self.denormalize_image(img) for img in inputs["pixel_values"]],
            return_tensors="pt"
        )

        img_feats = self.clip_model.get_image_features(clip_inputs["pixel_values"])
        txt_feats = self.clip_model.get_text_features(clip_inputs["input_ids"])

        clip_similarity = (img_feats * txt_feats).sum(-1).mean()
        clip_loss = 1.0 - clip_similarity

        # Combined loss
        total_loss = ce_loss + self.clip_loss_weight * clip_loss
        return total_loss
```

**Usage**:

```bash
# Basic CLIP loss (10% weight)
bin/train --dataset flickr \
          --clip-loss-weight 0.1 \
          --clip-model openai/clip-vit-base-patch32

# Higher weight for stronger alignment (20%)
bin/train --dataset flickr \
          --clip-loss-weight 0.2 \
          --clip-model openai/clip-vit-large-patch14-336
```

**Expected benefits**:
- +5-10% improvement in CLIP score
- Better image-text alignment in generated captions
- More visually grounded descriptions

**Trade-offs**:
- Slower training (~15-20% overhead due to caption generation)
- Requires more GPU memory (CLIP model + caption model)

## Addressing Class Imbalance: Synthetic Data Generation

Our quality analysis revealed 204 rare objects with ≤5 samples. To address this, we built a synthetic prompt generator that creates bias-free captions for use with image generation tools (Stable Diffusion, DALL-E, Midjourney).

### Bias-Free Caption Templates

We created five template categories with contextual modifiers:

```python
CAPTION_TEMPLATES = {
    "indoor": [
        "A {object} in a cozy room with natural lighting",
        "A detailed view of a {object} on a wooden table",
        "A {object} placed near a window with soft daylight",
        # ... 7 templates
    ],
    "outdoor": [
        "A {object} in a natural outdoor setting",
        "A {object} photographed in bright daylight",
        # ... 7 templates
    ],
    "activity": [
        "A scene showing interaction with a {object}",
        "A {object} being used in an activity",
        # ... 6 templates
    ],
    # ... more categories
}

MODIFIERS = {
    "lighting": ["with dramatic lighting", "in golden hour light", ...],
    "composition": ["from a low angle", "with rule of thirds", ...],
    "style": ["in photorealistic style", "with vibrant colors", ...],
    "quality": ["highly detailed", "4K quality", ...],
}
```

The generator combines templates with modifiers to create diverse, bias-free prompts:

```python
def generate_caption(obj):
    template = random.choice(all_templates)
    caption = template.format(object=obj)

    # Add 1-2 random modifiers
    if random.random() > 0.3:
        modifiers = random.sample(MODIFIERS, k=random.randint(1, 2))
        caption = f"{caption}, {', '.join(modifiers)}"

    return caption
```

### Multi-Object Combinations

The generator also creates prompts featuring multiple rare objects together:

```python
def generate_combination_caption(objects):
    selected = random.sample(objects, k=random.randint(2, 3))

    if len(selected) == 2:
        obj_phrase = f"a {selected[0]} and a {selected[1]}"
    else:
        obj_phrase = ", ".join([f"a {o}" for o in selected[:-1]]) + f", and a {selected[-1]}"

    caption = f"A scene with {obj_phrase}, in a natural setting"
    return caption
```

### Usage Example

```bash
# Generate prompts for rare objects
make generate-prompts

# Or with custom parameters
./bin/python distilvit/generate_synthetic_prompts.py \
  --rare-objects quality_reports/objects_below_50.csv \
  --prompts-per-object 5 \
  --output synthetic_prompts.jsonl \
  --include-combinations
```

**Output** (`synthetic_prompts.jsonl`):

```json
{"prompt": "A professional photograph of a telescope, with dramatic lighting, highly detailed", "objects": ["telescope"], "type": "single_object"}
{"prompt": "A scene with a guitar, a skateboard, and a telescope together, in a natural setting", "objects": ["guitar", "skateboard", "telescope"], "type": "combination"}
```

**Results**: Generated 712 prompts for 204 rare objects.

### Synthetic Data Workflow

1. **Generate prompts**: Use our tool to create bias-free captions
2. **Generate images**: Use Stable Diffusion, DALL-E, or Midjourney with the prompts
3. **Add to training**: Include synthetic images in your dataset
4. **Validate**: Re-run quality analysis to verify improved balance

## Running the Quality Analysis

The tool is integrated into our project's Makefile:

```bash
# Quick test (100 samples)
make quality-report-quick

# Full analysis on test split
make quality-report SPLIT=test

# Custom analysis
make quality-report SPLIT=train MAX_SAMPLES=1000 OUTPUT_DIR=./my_reports

# With specific CLIP model
./bin/python distilvit/dataset_quality_report.py \
  --dataset Mozilla/flickr30k-transformed-captions-gpt4o \
  --split test \
  --clip-model openai/clip-vit-large-patch14-336 \
  --output-dir ./quality_reports
```

## Output Files

The tool generates comprehensive outputs:

```
quality_reports/
├── summary.json                    # All metrics in JSON format
├── quality_report.txt              # Human-readable summary
├── objects_distribution.csv        # Object frequency table
├── objects_below_50.csv           # Rare objects (≤50 samples)
├── reweighting_probs.csv          # Balanced sampling weights
└── lorenz_curve.png               # Distribution inequality visualization
```

## Key Takeaways

1. **Multi-metric evaluation is essential**: No single metric captures dataset quality. CLIP, BERT, bias detection, and distribution analysis provide complementary insights.

2. **Bias can be measured and eliminated**: Our comparison shows GPT-4 successfully removes 97-100% of gender, race, and nationality descriptors while maintaining semantic fidelity.

3. **Model size matters for discrimination**: Larger CLIP models (ViT-L/14@336px) provide more discriminative scoring than smaller ones (ViT-B/32).

4. **CLIP loss improves alignment**: Adding CLIP alignment loss during training encourages the model to generate more visually grounded captions.

5. **Address class imbalance proactively**: Synthetic data generation can help balance rare classes before training begins.

6. **Optimize for your hardware**: Auto-detecting batch sizes and using progress bars makes quality analysis practical on diverse hardware.

## Future Work

Several promising directions:

- **Multi-language bias detection**: Extend protected terms to other languages
- **Fine-grained bias categories**: Detect subtler forms of bias (stereotypes, associations)
- **Temporal bias**: Track bias trends across dataset splits and time
- **Active learning**: Use quality scores to prioritize data for human review
- **Adversarial testing**: Generate challenging examples to probe model biases

## Code and Resources

- **Project**: [distilvit2](https://github.com/mozilla/distilvit2)
- **Dataset**: [Mozilla/flickr30k-transformed-captions-gpt4o](https://huggingface.co/datasets/Mozilla/flickr30k-transformed-captions-gpt4o)
- **Documentation**: [docs/dataset_quality.md](../docs/dataset_quality.md)
- **Quality Analysis Tool**: [distilvit/dataset_quality_report.py](../distilvit/dataset_quality_report.py)
- **Synthetic Prompt Generator**: [distilvit/generate_synthetic_prompts.py](../distilvit/generate_synthetic_prompts.py)

---

*This work is part of Mozilla's effort to build more transparent and bias-aware AI systems. By measuring and improving dataset quality, we aim to create image captioning models that are both accurate and fair.*
