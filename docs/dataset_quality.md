# Dataset Quality Analysis

Comprehensive quality analysis and bias detection for image captioning datasets.

## Overview

The dataset quality analysis tool measures multiple dimensions of dataset quality:
- **Image-text alignment** (CLIP scores)
- **Caption fidelity** (BERT scores comparing transformed vs original)
- **Bias detection** (protected attribute mentions with original vs transformed comparison)
- **Caption statistics** (length, vocabulary, duplicates)
- **Object distribution** (coverage, imbalance metrics)

## Quick Start

```bash
# Quick analysis (100 samples)
make quality-report-quick

# Full analysis
make quality-report

# Custom options
make quality-report DATASET=Mozilla/flickr30k-transformed-captions-gpt4o MAX_SAMPLES=1000 SPLIT=train
```

## Command Options

### Makefile Targets

| Target | Description |
|--------|-------------|
| `make quality-report-quick` | Analyze 100 samples from test split |
| `make quality-report` | Full analysis with customizable options |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATASET` | `Mozilla/flickr30k-transformed-captions-gpt4o` | HuggingFace dataset name |
| `SPLIT` | `test` | Dataset split (train/test/validation) |
| `MAX_SAMPLES` | (all) | Maximum samples to analyze |
| `OUTPUT_DIR` | `./quality_reports` | Output directory for reports |

### Direct Script Usage

```bash
./bin/python distilvit/dataset_quality_report.py \
  --dataset Mozilla/flickr30k-transformed-captions-gpt4o \
  --split test \
  --max-samples 100 \
  --output-dir ./quality_reports \
  --batch-size 32 \
  --device mps
```

#### Script Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `Mozilla/flickr30k-transformed-captions-gpt4o` | Dataset to analyze |
| `--split` | `test` | Dataset split |
| `--max-samples` | None | Limit number of samples |
| `--output-dir` | `./quality_reports` | Output directory |
| `--batch-size` | `32` | Batch size for CLIP scoring |
| `--device` | auto | Device (cuda/mps/cpu) |
| `--topk` | `300` | Number of failure examples to export |
| `--rare-threshold` | `50` | Threshold for rare object classes |
| `--plot` | False | Generate distribution plots |
| `--imbalance-gini-alert` | `0.5` | Gini threshold for warnings |
| `--imbalance-topk-alert` | `0.5` | Top-k share threshold for warnings |

## Metrics Explained

### CLIP Score (Image-Text Alignment)

Measures how well captions match their images using OpenAI's CLIP model.

**Scale**: 0.0 to 1.0 (higher is better)

**Thresholds**:
- **>0.35**: Excellent - Very strong alignment
- **0.30-0.35**: Good - Solid alignment
- **0.25-0.30**: Fair - Acceptable but could improve
- **0.20-0.25**: Poor - Weak alignment
- **<0.20**: Very Poor - Major issues

**Typical values**:
- High-quality captioning datasets: 0.30-0.35
- Transformed captions: 0.28-0.32
- Low-quality captions: <0.25

### BERT Score (Caption Fidelity)

Measures semantic similarity between transformed and original captions using BERT embeddings.

**Scale**: 0.0 to 1.0 (higher is better)

**Thresholds**:
- **>0.90**: Excellent - Very high fidelity to original
- **0.85-0.90**: Good - Preserves most meaning
- **0.80-0.85**: Fair - Moderate changes
- **0.70-0.80**: Poor - Significant deviation
- **<0.70**: Very Poor - Major rewrite

**Interpretation**:
- High BERT scores indicate the transformation preserves semantic content
- Combined with low bias rates, shows successful bias removal without losing meaning

### Bias Detection

Detects mentions of protected attributes across 7 categories, comparing original vs transformed captions.

#### Categories

| Category | Description | Acceptable Rate |
|----------|-------------|-----------------|
| **Gender** | Gender terms, pronouns | <1% (should be eliminated) |
| **Sexual Orientation** | Orientation identifiers | <1% (should be eliminated) |
| **Race/Ethnicity** | Racial/ethnic descriptors | <2% (should be eliminated) |
| **Nationality** | Country/nationality terms | <2% (minimize) |
| **Religion** | Religious identifiers | <2% (minimize) |
| **Disability** | Disability-related terms | <5% (context-dependent) |
| **Age** | Age descriptors | <20% (context-dependent) |

#### Protected Terms Detection

The tool uses an extensive vocabulary of protected terms:

- **Gender**: 30+ terms including pronouns, gender identities
- **Sexual Orientation**: 12+ terms for various orientations
- **Race/Ethnicity**: 20+ high-level and regional categories
- **Nationality**: 200+ countries and demonyms (auto-generated via pycountry)
- **Religion**: 20+ major religions and denominations
- **Disability**: 18+ disability-related terms
- **Age**: 15+ age-related descriptors

Terms are detected using word boundary matching on normalized text (lowercase, alphanumeric only).

#### Comparison Output

The bias detection section shows:

```
Category                    Original  Transformed     Change                    Status
────────────────────────────────────────────────────────────────────────────────
Gender                        70.00%        0.00%    -100.0%               ✅ EXCELLENT
Race Ethnicity                33.00%        1.00%     -97.0%               ✅ EXCELLENT
Age                           18.00%       14.00%     -22.2%                    ✅ GOOD
```

- **Original**: Baseline bias rate from original captions
- **Transformed**: Bias rate after GPT-4 transformation
- **Change**: Percentage reduction (negative = improvement)
- **Status**: Quality assessment of transformed state

### Caption Statistics

| Metric | Description | Typical Values |
|--------|-------------|----------------|
| **Average length** | Mean words per caption | 8-12 words |
| **Median length** | Median words per caption | 8-10 words |
| **Vocabulary** | Unique words in dataset | 200-500 (100 samples) |
| **Duplicate rate** | Identical captions | <5% is good |
| **Entropy** | Vocabulary diversity | Higher = more diverse |
| **Gini coefficient** | Word distribution inequality | 0=equal, 1=unequal |

### Object Distribution

Analyzes how objects/tags are distributed across the dataset.

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Entropy** | Object distribution diversity | Higher = more diverse |
| **Gini coefficient** | Distribution inequality | <0.5 is good, >0.6 shows imbalance |
| **Top-5 share** | % of mentions from top 5 objects | <30% is good |
| **Top-10 share** | % of mentions from top 10 objects | <40% is good |
| **Rare classes** | Objects below threshold | Identifies underrepresented classes |

## Output Files

All files are saved to `quality_reports/` (or custom `OUTPUT_DIR`):

| File | Description |
|------|-------------|
| `quality_summary.json` | All metrics in JSON format |
| `per_example_scores.csv` | Detailed scores for each sample |
| `ranked_by_combined.csv` | Samples ranked by quality (worst first) |
| `object_counts.csv` | Object frequency distribution |
| `objects_below_50.csv` | Rare/underrepresented objects |
| `reweighting_probs.csv` | Suggested sampling probabilities for balanced training |
| `top_failures/` | Images and metadata for worst examples |

### Per-Example Scores CSV

Contains detailed per-sample analysis:

```csv
idx,alt_text,original_text,objects,word_count,has_image,
protected_gender_transformed,protected_gender_original,
protected_race_ethnicity_transformed,protected_race_ethnicity_original,
...,clip_score,bert_score,objs_matched,objs_total,obj_miss_frac
```

### Quality Summary JSON

Complete metrics in JSON format:

```json
{
  "dataset": "Mozilla/flickr30k-transformed-captions-gpt4o",
  "split": "test",
  "total_samples": 100,
  "clip_score_mean": 0.3113,
  "bert_score_mean": 0.9035,
  "protected_gender_rate_original": 0.70,
  "protected_gender_rate_transformed": 0.0,
  "protected_race_ethnicity_rate_original": 0.33,
  "protected_race_ethnicity_rate_transformed": 0.01,
  ...
}
```

### Top Failures Directory

Contains the worst-performing examples for manual review:

- `{idx}.txt`: Metadata (caption, scores, objects)
- `{idx}_c{clip}_b{bert}.jpg`: Original image

Useful for identifying systematic issues or edge cases.

## Example Analysis Results

### Mozilla/flickr30k-transformed-captions-gpt4o (100 samples)

```
CLIP Score:        0.3113 ± 0.0306  ✅ GOOD
BERT Score:        0.9035 ± 0.0232  ✅ EXCELLENT
Duplicates:        0.00%            ✅ EXCELLENT

BIAS DETECTION COMPARISON:
Gender:            70% → 0%   (-100.0%)  ✅ EXCELLENT
Race/Ethnicity:    33% → 1%   (-97.0%)   ✅ EXCELLENT
Nationality:       1% → 0%    (-100.0%)  ✅ EXCELLENT
Age:               18% → 14%  (-22.2%)   ✅ GOOD

OBJECT DISTRIBUTION:
Unique objects:    215
Gini coefficient:  0.462 (good)
Top-10 share:      37.8% (good)
```

**Interpretation**: The GPT-4 transformation successfully eliminates gender and race bias while maintaining high semantic fidelity (BERT 0.90+) and good image-text alignment (CLIP 0.31).

## Use Cases

### 1. Dataset Validation

Verify dataset quality before training:

```bash
make quality-report-quick
```

Check for:
- Low CLIP scores (poor image-text alignment)
- High duplicate rates
- Unexpected bias patterns
- Object distribution imbalances

### 2. Transformation Effectiveness

Measure bias reduction after caption transformation:

```bash
make quality-report DATASET=your-transformed-dataset
```

Compare original vs transformed bias rates to quantify improvement.

### 3. Data Cleaning

Identify problematic samples:

```bash
make quality-report MAX_SAMPLES=1000
```

Review `top_failures/` directory for:
- Misaligned captions
- Object coverage issues
- Quality outliers

Use `ranked_by_combined.csv` to prioritize data cleaning efforts.

### 4. Training Data Balancing

Use `reweighting_probs.csv` for balanced sampling:

```python
import pandas as pd

probs = pd.read_csv('quality_reports/reweighting_probs.csv')
weights = probs['reweight_norm'].values

# Use in DataLoader
sampler = WeightedRandomSampler(weights, len(weights))
dataloader = DataLoader(dataset, sampler=sampler, ...)
```

This upweights rare object classes for more balanced training.

### 5. Dataset Comparison

Compare multiple datasets:

```bash
make quality-report DATASET=dataset-a OUTPUT_DIR=./reports_a
make quality-report DATASET=dataset-b OUTPUT_DIR=./reports_b
```

Compare `quality_summary.json` files to choose the best dataset for training.

## Interpreting Results

### Good Dataset Characteristics

- **CLIP score** >0.30: Strong image-text alignment
- **BERT score** >0.85: Good semantic fidelity (for transformed captions)
- **Gender/Race bias** <2%: Effective bias elimination
- **Duplicates** <5%: Good diversity
- **Gini coefficient** <0.5: Balanced distribution
- **No imbalance warnings**: Even object coverage

### Warning Signs

- **CLIP score** <0.25: Poor alignment, may need caption improvement
- **BERT score** <0.80: Transformation changed meaning too much
- **Gender/Race bias** >10%: Bias removal failed
- **Duplicates** >10%: Low diversity, may need more data
- **Gini coefficient** >0.6: Highly imbalanced, consider reweighting
- **Imbalance warnings**: May need balanced sampling

## Technical Details

### Models Used

- **CLIP**: `openai/clip-vit-base-patch32` for image-text alignment
- **BERT**: `roberta-large` (via bert-score) for semantic similarity

### Computation

- **Device support**: CUDA, MPS (Apple Silicon), CPU
- **Batch processing**: Configurable batch size for efficiency
- **Memory usage**: ~2-4GB GPU RAM for 100 samples

### Performance

Approximate runtimes (100 samples on M1 Max):
- CLIP scoring: ~10 seconds
- BERT scoring: ~30 seconds
- Total: ~60 seconds

Full dataset (1000 samples): ~10 minutes

## Extending the Tool

### Adding Custom Protected Terms

Edit `distilvit/dataset_quality_report.py`:

```python
PROTECTED_TERMS = {
    "gender": ["man", "woman", ...],
    "custom_category": ["term1", "term2", ...],
}
```

### Custom Quality Metrics

Add metrics in `analyze_dataset()`:

```python
# Custom metric
custom_score = your_metric_function(df)
summary['custom_metric'] = float(custom_score)
```

### Custom Visualizations

Enable plotting:

```bash
make quality-report --plot
```

Or add custom plots in the `if args.plot:` section.

## Troubleshooting

### Out of Memory

Reduce batch size:

```bash
./bin/python distilvit/dataset_quality_report.py --batch-size 8
```

### Slow Performance

- Use GPU if available: `--device cuda` or `--device mps`
- Reduce sample size: `--max-samples 100`
- Disable plots: Remove `--plot` flag

### Dataset Loading Errors

Ensure dataset format has required columns:
- `alt_text` or `caption`: Current/transformed captions
- `original_alt_text` or `original_caption`: Original captions (optional)
- `image`: PIL Image objects or file paths
- `objects` or `tags`: Object annotations (optional)

## Related Documentation

- [Fighting Bias](fighting_bias.md): Bias mitigation strategy and guidelines
- [Architecture](architecture.md): Model architecture details
- [Model Card](model_card.md): Model documentation and limitations

## References

- CLIP: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- BERTScore: [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675)
- Protected attributes vocabulary inspired by [Hugging Face Datasets bias detection](https://huggingface.co/docs/datasets/)
