#!/usr/bin/env python3
"""
dataset_quality_report.py

Compute dataset quality signals:
 - CLIPScore (image vs alt_text)
 - BERTScore (alt_text vs original_caption)
 - object coverage & mismatch
 - class/object frequency distribution + imbalance metrics (entropy, gini, top-k share)
 - caption length distribution, duplicate rates, vocabulary stats
 - protected attribute mentions (gender/race/age terms)
 - export CSV and summary JSON + top-K failure examples (images + metadata)

Usage:
    # Analyze Mozilla transformed Flickr30k
    python distilvit/dataset_quality_report.py \
        --dataset Mozilla/flickr30k-transformed-captions-gpt4o \
        --output-dir ./quality_reports

    # Analyze with custom split and sample
    python distilvit/dataset_quality_report.py \
        --dataset Mozilla/flickr30k-transformed-captions-gpt4o \
        --split test \
        --max-samples 1000 \
        --output-dir ./quality_reports

Note: Run locally. If dataset rows include image URLs and you want to fetch images,
this script will attempt to download them (requires internet). If images are stored as
HF 'image' features, it will use them directly.
"""

import os
import re
import json
import math
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from bert_score import score as bert_score

# ------------------------
# Utilities
# ------------------------
def normalize_text(txt):
    if txt is None:
        return ""
    txt = txt.lower()
    txt = re.sub(r"[^\w\s]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def simple_object_match(alt_text, objects):
    """Check how many objects from list are mentioned in alt_text"""
    if not objects:
        return (0, 0)
    t = normalize_text(alt_text)
    matched = 0
    total = 0
    for o in objects:
        if not o:
            continue
        total += 1
        o_norm = normalize_text(o)
        if re.search(rf"\b{re.escape(o_norm)}\b", t):
            matched += 1
        else:
            # plural fallback
            if o_norm.endswith("s") and re.search(rf"\b{re.escape(o_norm[:-1])}\b", t):
                matched += 1
    return matched, total

def load_image_from_row(row):
    """Try to load image from HF image feature, file path, or URL"""
    # handle direct HF image feature
    if "image" in row and row["image"] is not None:
        try:
            img = row["image"]
            if isinstance(img, Image.Image):
                return img.convert("RGB")
            else:
                return Image.open(img).convert("RGB")
        except Exception:
            pass

    # local filename heuristics
    for k in ("image_path", "image_file", "filename", "img_path", "file_name"):
        if k in row and row[k]:
            try:
                return Image.open(row[k]).convert("RGB")
            except Exception:
                pass

    # url keys
    for k in ("image_url", "img_url", "url", "flickr_url"):
        if k in row and row[k]:
            try:
                import requests, io
                r = requests.get(row[k], timeout=20)
                r.raise_for_status()
                return Image.open(io.BytesIO(r.content)).convert("RGB")
            except Exception:
                return None
    return None

# ------------------------
# CLIP scorer (batch)
# ------------------------
class CLIPScorer:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def image_text_sim(self, images, texts, batch_size=32):
        """Compute CLIP similarity between images and texts"""
        sims = []
        n = len(texts)
        for i in range(0, n, batch_size):
            batch_imgs = images[i:i+batch_size]
            batch_texts = texts[i:i+batch_size]
            # Handle missing images
            proc_imgs = [img if img is not None else Image.new("RGB",(224,224),(0,0,0)) for img in batch_imgs]

            inputs = self.processor(
                text=batch_texts,
                images=proc_imgs,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            img_feats = self.model.get_image_features(pixel_values=inputs["pixel_values"])
            txt_feats = self.model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

            # Normalize features
            img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
            txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)

            # Compute cosine similarity
            batch_sim = (img_feats * txt_feats).sum(-1).cpu().tolist()
            sims.extend(batch_sim)
        return sims

# ------------------------
# Imbalance metrics
# ------------------------
def entropy_from_counts(counter):
    """Shannon entropy of distribution"""
    total = sum(counter.values())
    if total == 0:
        return 0.0
    probs = np.array([v/total for v in counter.values()])
    ent = -np.sum(probs * np.log2(probs + 1e-12))
    return float(ent)

def gini_from_counts(counter):
    """Gini coefficient (0=equal, 1=maximally unequal)"""
    vals = np.array(sorted(counter.values()))
    if vals.sum() == 0:
        return 0.0
    n = len(vals)
    cum = np.cumsum(vals)
    gini = (n + 1 - 2 * np.sum(cum / cum[-1])) / n
    return float(gini)

def topk_share(counter, k=10):
    """Share of total count held by top-k items"""
    total = sum(counter.values())
    if total == 0:
        return 0.0
    top = counter.most_common(k)
    top_sum = sum(v for _, v in top)
    return float(top_sum / total)

# ------------------------
# Protected attributes
# ------------------------
PROTECTED_TERMS = {
    "gender": [
        "man", "men", "woman", "women", "male", "female", "boy", "girl",
        "guy", "guys", "lady", "ladies", "gentleman", "gentlemen",
        "he", "she", "his", "her", "him"
    ],
    "race": [
        "white", "black", "asian", "hispanic", "latino", "latina",
        "african", "caucasian", "chinese", "indian", "arab"
    ],
    "age": [
        "young", "old", "elderly", "senior", "youth", "child", "children",
        "kid", "kids", "baby", "toddler", "teenager", "teen"
    ],
}

def count_protected_mentions(text):
    """Count mentions of protected attributes in text"""
    t = normalize_text(text)
    counts = {cat: 0 for cat in PROTECTED_TERMS}

    for category, terms in PROTECTED_TERMS.items():
        for term in terms:
            if re.search(rf"\b{re.escape(term)}\b", t):
                counts[category] += 1

    return counts

# ------------------------
# Main analysis
# ------------------------
def analyze_dataset(args):
    print(f"Loading dataset: {args.dataset} (split: {args.split})")

    # Load dataset
    if args.max_samples:
        split_str = f"{args.split}[:{args.max_samples}]"
        ds = load_dataset(args.dataset, split=split_str)
    else:
        ds = load_dataset(args.dataset, split=args.split)

    print(f"Loaded {len(ds)} samples")

    # Initialize CLIP scorer
    print("Loading CLIP model...")
    clip_scorer = CLIPScorer(device=args.device)

    # Prepare output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect data
    results = []
    images = []
    alt_texts = []
    original_texts = []
    word_counter = Counter()
    length_dist = []
    protected_counts = defaultdict(list)

    print("Processing samples...")
    for idx, row in enumerate(tqdm(ds)):
        # Get alt_text (transformed caption)
        alt_text = row.get("alt_text") or row.get("caption") or ""
        if isinstance(alt_text, list):
            alt_text = alt_text[0] if alt_text else ""

        # Get original caption for comparison
        original_text = row.get("original_alt_text") or row.get("original_caption") or ""
        if isinstance(original_text, list):
            original_text = original_text[0] if original_text else ""

        # Load image
        img = load_image_from_row(row)

        # Collect for batch processing
        images.append(img)
        alt_texts.append(alt_text)
        original_texts.append(original_text)

        # Text analysis
        words = normalize_text(alt_text).split()
        word_counter.update(words)
        length_dist.append(len(words))

        # Protected attributes
        prot = count_protected_mentions(alt_text)
        for cat, count in prot.items():
            protected_counts[cat].append(count)

        # Object detection (if available)
        objects = row.get("objects") or []
        if objects and isinstance(objects, list):
            matched, total = simple_object_match(alt_text, objects)
            obj_coverage = matched / total if total > 0 else None
        else:
            obj_coverage = None

        results.append({
            "idx": idx,
            "alt_text": alt_text,
            "original_text": original_text,
            "word_count": len(words),
            "obj_coverage": obj_coverage,
            "has_image": img is not None,
            **{f"protected_{k}": v for k, v in prot.items()}
        })

    # Compute CLIP scores
    print("Computing CLIP scores...")
    clip_scores = clip_scorer.image_text_sim(images, alt_texts, batch_size=args.batch_size)

    # Compute BERT scores (alt_text vs original)
    print("Computing BERT scores...")
    valid_pairs = [(a, o) for a, o in zip(alt_texts, original_texts) if a and o]
    if valid_pairs:
        alts, origs = zip(*valid_pairs)
        P, R, F1 = bert_score(
            list(alts),
            list(origs),
            lang="en",
            verbose=False,
            device=args.device
        )
        bert_scores = F1.cpu().numpy().tolist()
        # Pad with None for rows without original text
        bert_idx = 0
        for i, (a, o) in enumerate(zip(alt_texts, original_texts)):
            if a and o:
                results[i]["bert_score"] = bert_scores[bert_idx]
                bert_idx += 1
            else:
                results[i]["bert_score"] = None
    else:
        for r in results:
            r["bert_score"] = None

    # Add CLIP scores to results
    for i, score in enumerate(clip_scores):
        results[i]["clip_score"] = score

    # Create DataFrame
    df = pd.DataFrame(results)

    # Compute summary statistics
    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "total_samples": len(ds),
        "samples_with_images": df["has_image"].sum(),

        # CLIP scores
        "clip_score_mean": df["clip_score"].mean(),
        "clip_score_std": df["clip_score"].std(),
        "clip_score_min": df["clip_score"].min(),
        "clip_score_max": df["clip_score"].max(),
        "clip_score_median": df["clip_score"].median(),

        # BERT scores
        "bert_score_mean": df["bert_score"].mean(),
        "bert_score_std": df["bert_score"].std(),
        "bert_score_min": df["bert_score"].min(),
        "bert_score_max": df["bert_score"].max(),

        # Caption lengths
        "avg_word_count": df["word_count"].mean(),
        "median_word_count": df["word_count"].median(),
        "min_word_count": df["word_count"].min(),
        "max_word_count": df["word_count"].max(),

        # Vocabulary
        "unique_words": len(word_counter),
        "total_words": sum(word_counter.values()),
        "vocabulary_entropy": entropy_from_counts(word_counter),

        # Protected attributes
        "protected_gender_rate": (df["protected_gender"] > 0).mean(),
        "protected_race_rate": (df["protected_race"] > 0).mean(),
        "protected_age_rate": (df["protected_age"] > 0).mean(),

        # Object coverage
        "avg_object_coverage": df["obj_coverage"].mean() if "obj_coverage" in df else None,

        # Duplicates
        "duplicate_rate": (len(alt_texts) - len(set(alt_texts))) / len(alt_texts),

        # Distribution stats
        "word_dist_entropy": entropy_from_counts(word_counter),
        "word_dist_gini": gini_from_counts(word_counter),
        "word_dist_top10_share": topk_share(word_counter, k=10),
    }

    # Top-10 most common words
    summary["top_10_words"] = dict(word_counter.most_common(10))

    # Top-10 failures (lowest CLIP scores)
    failures = df.nsmallest(10, "clip_score")[["idx", "clip_score", "alt_text", "original_text"]]
    summary["top_10_failures"] = failures.to_dict(orient="records")

    # Top-10 successes (highest CLIP scores)
    successes = df.nlargest(10, "clip_score")[["idx", "clip_score", "alt_text", "original_text"]]
    summary["top_10_successes"] = successes.to_dict(orient="records")

    # Save results
    csv_path = output_dir / "quality_report.csv"
    json_path = output_dir / "quality_summary.json"

    print(f"\nSaving results to {output_dir}")
    df.to_csv(csv_path, index=False)

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("DATASET QUALITY SUMMARY")
    print("="*60)
    print(f"Dataset: {summary['dataset']}")
    print(f"Samples: {summary['total_samples']} ({summary['samples_with_images']} with images)")
    print(f"\nCLIP Score (image-text alignment):")
    print(f"  Mean: {summary['clip_score_mean']:.4f} ± {summary['clip_score_std']:.4f}")
    print(f"  Median: {summary['clip_score_median']:.4f}")
    print(f"  Range: [{summary['clip_score_min']:.4f}, {summary['clip_score_max']:.4f}]")

    if summary['bert_score_mean']:
        print(f"\nBERT Score (alt vs original):")
        print(f"  Mean: {summary['bert_score_mean']:.4f} ± {summary['bert_score_std']:.4f}")

    print(f"\nCaption Statistics:")
    print(f"  Avg words: {summary['avg_word_count']:.1f}")
    print(f"  Median words: {summary['median_word_count']:.0f}")
    print(f"  Unique words: {summary['unique_words']}")
    print(f"  Duplicate rate: {summary['duplicate_rate']*100:.2f}%")

    print(f"\nProtected Attributes:")
    print(f"  Gender mentions: {summary['protected_gender_rate']*100:.1f}%")
    print(f"  Race mentions: {summary['protected_race_rate']*100:.1f}%")
    print(f"  Age mentions: {summary['protected_age_rate']*100:.1f}%")

    print(f"\nDistribution Balance:")
    print(f"  Entropy: {summary['word_dist_entropy']:.2f}")
    print(f"  Gini: {summary['word_dist_gini']:.3f}")
    print(f"  Top-10 words share: {summary['word_dist_top10_share']*100:.1f}%")

    print(f"\nTop 10 most common words:")
    for word, count in summary['top_10_words'].items():
        print(f"  {word}: {count}")

    print(f"\n✓ Detailed CSV saved to: {csv_path}")
    print(f"✓ Summary JSON saved to: {json_path}")
    print("="*60)

    return df, summary

# ------------------------
# CLI
# ------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compute dataset quality metrics for image captioning datasets"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="Mozilla/flickr30k-transformed-captions-gpt4o",
        help="HuggingFace dataset name or path"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to analyze (default: test)"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to analyze (default: all)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./quality_reports",
        help="Output directory for reports (default: ./quality_reports)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for CLIP scoring (default: 32)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/mps/cpu, default: auto-detect)"
    )

    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    print(f"Using device: {args.device}")

    analyze_dataset(args)

if __name__ == "__main__":
    main()
