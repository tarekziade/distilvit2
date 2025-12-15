#!/usr/bin/env python3
"""
dataset_quality_report.py (updated)

Adds:
 - imbalance warnings (Gini, top-k share) with configurable thresholds
 - rare-class report (classes with counts < rare-threshold)
 - suggested reweighting probabilities per image (inverse-frequency over objects)
 - plots: Lorenz curve, log-log object frequency, caption length histogram
 - saves outputs: CSVs, summary JSON, plots, top failure artifacts

Usage examples (same as before, with new flags):
 python dataset_quality_report.py --dataset Mozilla/flickr30k-transformed-captions-gpt4o \
    --split train --max-samples 2000 --output-dir ./quality_reports --plot

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
import matplotlib.pyplot as plt


# ------------------------
# Utilities (unchanged / extended)
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
    if "image" in row and row["image"] is not None:
        try:
            img = row["image"]
            if isinstance(img, Image.Image):
                return img.convert("RGB")
            else:
                return Image.open(img).convert("RGB")
        except Exception:
            pass

    for k in ("image_path", "image_file", "filename", "img_path", "file_name"):
        if k in row and row[k]:
            try:
                return Image.open(row[k]).convert("RGB")
            except Exception:
                pass

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
            batch_imgs = images[i : i + batch_size]
            batch_texts = texts[i : i + batch_size]
            proc_imgs = [
                img if img is not None else Image.new("RGB", (224, 224), (0, 0, 0))
                for img in batch_imgs
            ]
            inputs = self.processor(
                text=batch_texts, images=proc_imgs, return_tensors="pt", padding=True
            ).to(self.device)
            img_feats = self.model.get_image_features(
                pixel_values=inputs["pixel_values"]
            )
            txt_feats = self.model.get_text_features(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
            img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
            txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
            batch_sim = (img_feats * txt_feats).sum(-1).cpu().tolist()
            sims.extend(batch_sim)
        return sims


# ------------------------
# Imbalance metrics (same + lorenz helper)
# ------------------------
def entropy_from_counts(counter):
    total = sum(counter.values())
    if total == 0:
        return 0.0
    probs = np.array([v / total for v in counter.values()])
    ent = -np.sum(probs * np.log2(probs + 1e-12))
    return float(ent)


def gini_from_counts(counter):
    vals = np.array(sorted(counter.values()))
    if vals.sum() == 0:
        return 0.0
    n = len(vals)
    cum = np.cumsum(vals)
    gini = (n + 1 - 2 * np.sum(cum / cum[-1])) / n
    return float(gini)


def topk_share(counter, k=10):
    total = sum(counter.values())
    if total == 0:
        return 0.0
    top = counter.most_common(k)
    top_sum = sum(v for _, v in top)
    return float(top_sum / total)


def lorenz_curve(counter):
    """Return x (cumulative share of classes) and y (cumulative share of counts) for Lorenz curve"""
    vals = np.array(sorted(counter.values()))
    if vals.sum() == 0:
        return np.array([0, 1]), np.array([0, 1])
    vals = vals.astype(float)
    cum_vals = np.cumsum(vals)
    total = cum_vals[-1]
    prop_counts = cum_vals / total
    prop_classes = np.linspace(1 / len(vals), 1.0, len(vals))
    return prop_classes, prop_counts


# ------------------------
# Protected attributes
# ------------------------
PROTECTED_TERMS = {
    "gender": [
        "man",
        "men",
        "woman",
        "women",
        "male",
        "female",
        "boy",
        "girl",
        "guy",
        "guys",
        "lady",
        "ladies",
        "gentleman",
        "gentlemen",
        "he",
        "she",
        "his",
        "her",
        "him",
    ],
    "race": [
        "white",
        "black",
        "asian",
        "hispanic",
        "latino",
        "latina",
        "african",
        "caucasian",
        "chinese",
        "indian",
        "arab",
    ],
    "age": [
        "young",
        "old",
        "elderly",
        "senior",
        "youth",
        "child",
        "children",
        "kid",
        "kids",
        "baby",
        "toddler",
        "teenager",
        "teen",
    ],
}


def count_protected_mentions(text):
    t = normalize_text(text)
    counts = {cat: 0 for cat in PROTECTED_TERMS}
    for category, terms in PROTECTED_TERMS.items():
        for term in terms:
            if re.search(rf"\b{re.escape(term)}\b", t):
                counts[category] += 1
    return counts


# ------------------------
# Reweighting utilities
# ------------------------
def compute_inverse_freq_weights_for_row(objects, obj_counter, eps=1e-6):
    """Compute a suggested sample weight for an image row based on objects in it.
    If no objects, return None (can't compute)."""
    if not objects:
        return None
    weights = []
    for o in objects:
        o_norm = normalize_text(o)
        if not o_norm:
            continue
        freq = obj_counter.get(o_norm, 0) + eps
        weights.append(1.0 / freq)
    if not weights:
        return None
    # average inverse-frequency across objects
    return float(np.mean(weights))


# ------------------------
# Main analysis (enhanced)
# ------------------------
def analyze_dataset(args):
    print(f"Loading dataset: {args.dataset} (split: {args.split})")

    # Load dataset (optionally sample)
    if args.max_samples:
        split_str = f"{args.split}[:{args.max_samples}]"
        ds = load_dataset(args.dataset, split=split_str)
    else:
        ds = load_dataset(args.dataset, split=args.split)

    total_rows = len(ds)
    print(f"Loaded {total_rows} samples")

    # Initialize CLIP scorer
    print("Loading CLIP model...")
    clip_scorer = CLIPScorer(device=args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    images = []
    alt_texts = []
    original_texts = []
    word_counter = Counter()
    length_dist = []
    protected_counts = defaultdict(list)
    obj_counter = Counter()
    rows_objects = []

    print("Collecting rows...")
    for idx, row in enumerate(tqdm(ds, desc="reading rows")):
        alt_text = row.get("alt_text") or row.get("caption") or ""
        if isinstance(alt_text, list):
            alt_text = alt_text[0] if alt_text else ""

        original_text = (
            row.get("original_alt_text") or row.get("original_caption") or ""
        )
        if isinstance(original_text, list):
            original_text = original_text[0] if original_text else ""

        img = load_image_from_row(row)

        # objects field (normalize)
        objects = row.get("objects") or row.get("object_tags") or row.get("tags") or []
        if isinstance(objects, str):
            objects = [o.strip() for o in re.split(r"[;,]", objects) if o.strip()]
        objects_norm = [normalize_text(o) for o in objects if o and normalize_text(o)]
        for o in objects_norm:
            obj_counter[o] += 1

        rows_objects.append(objects_norm)

        # text stats
        words = normalize_text(alt_text).split()
        word_counter.update(words)
        length_dist.append(len(words))
        protected = count_protected_mentions(alt_text)
        for cat, c in protected.items():
            protected_counts[cat].append(c)

        images.append(img)
        alt_texts.append(alt_text)
        original_texts.append(original_text)

        # add preliminary result row (we'll fill CLIP/BERT later)
        results.append(
            {
                "idx": idx,
                "alt_text": alt_text,
                "original_text": original_text,
                "objects": objects_norm,
                "word_count": len(words),
                "has_image": img is not None,
                **{f"protected_{k}": v for k, v in protected.items()},
            }
        )

    # 1) CLIP scores
    print("Computing CLIP scores...")
    clip_scores = clip_scorer.image_text_sim(
        images, alt_texts, batch_size=args.batch_size
    )
    for i, s in enumerate(clip_scores):
        results[i]["clip_score"] = s

    # 2) BERTScore (alt vs original) in batches
    print("Computing BERT scores...")
    valid_pairs = [(a, o) for a, o in zip(alt_texts, original_texts) if a and o]
    bert_map = {}
    if valid_pairs:
        alts, origs = zip(*valid_pairs)
        P, R, F1 = bert_score(
            list(alts), list(origs), lang="en", verbose=False, device=args.device
        )
        bert_scores = F1.cpu().numpy().tolist()
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

    # 3) object coverage / mismatch
    total_with_objs = 0
    miss_count = 0
    for i, r in enumerate(results):
        objs = r["objects"]
        if objs:
            total_with_objs += 1
            matched, total = simple_object_match(r["alt_text"], objs)
            r["objs_matched"] = int(matched)
            r["objs_total"] = int(total)
            r["obj_miss_frac"] = (
                float((total - matched) / total) if total > 0 else float("nan")
            )
            if matched == 0:
                miss_count += 1
        else:
            r["objs_matched"] = 0
            r["objs_total"] = 0
            r["obj_miss_frac"] = None

    # 4) assemble dataframe
    df = pd.DataFrame(results)

    # 5) imbalance metrics & warnings (object-level)
    obj_entropy = entropy_from_counts(obj_counter)
    obj_gini = gini_from_counts(obj_counter)
    obj_top5 = topk_share(obj_counter, k=5)
    obj_top10 = topk_share(obj_counter, k=10)

    imbalance_warnings = []
    if obj_gini >= args.imbalance_gini_alert:
        imbalance_warnings.append(
            f"Gini ({obj_gini:.3f}) >= alert threshold {args.imbalance_gini_alert}"
        )
    if obj_top5 >= args.imbalance_topk_alert:
        imbalance_warnings.append(
            f"Top-5 share ({obj_top5:.3f}) >= alert threshold {args.imbalance_topk_alert}"
        )
    if obj_top10 >= args.imbalance_topk_alert:
        imbalance_warnings.append(
            f"Top-10 share ({obj_top10:.3f}) >= alert threshold {args.imbalance_topk_alert}"
        )

    # 6) rare-class report
    rare_threshold = args.rare_threshold
    rare_classes = [(c, cnt) for c, cnt in obj_counter.items() if cnt < rare_threshold]
    rare_classes_sorted = sorted(rare_classes, key=lambda x: x[1])
    rare_df = pd.DataFrame(rare_classes_sorted, columns=["object", "count"])
    rare_df.to_csv(output_dir / f"objects_below_{rare_threshold}.csv", index=False)

    # 7) suggested reweighting probabilities per image (inverse-frequency of objects)
    reweight_probs = []
    for i, objs in enumerate(rows_objects):
        w = compute_inverse_freq_weights_for_row(objs, obj_counter)
        reweight_probs.append(w)
        df.at[i, "reweight_suggested"] = w
    # normalize non-null probs to sum to 1 for sampling
    probs_arr = np.array(
        [p if p is not None else 0.0 for p in reweight_probs], dtype=float
    )
    if probs_arr.sum() > 0:
        probs_norm = probs_arr / probs_arr.sum()
    else:
        probs_norm = probs_arr
    reweight_df = pd.DataFrame(
        {
            "idx": range(len(probs_norm)),
            "reweight_raw": probs_arr.tolist(),
            "reweight_norm": probs_norm.tolist(),
        }
    )
    reweight_df.to_csv(output_dir / "reweighting_probs.csv", index=False)

    # 8) word / caption statistics
    vocab_size = len([w for w in word_counter.keys()])
    dup_rate = (len(alt_texts) - len(set(alt_texts))) / max(1, len(alt_texts))
    avg_len = np.mean(length_dist) if length_dist else 0
    median_len = np.median(length_dist) if length_dist else 0

    # 9) combine summary
    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "total_samples": total_rows,
        "samples_with_images": int(df["has_image"].sum()),
        "clip_score_mean": float(np.nanmean(df["clip_score"].values)),
        "clip_score_std": float(np.nanstd(df["clip_score"].values)),
        "clip_score_min": float(np.nanmin(df["clip_score"].values)),
        "clip_score_max": float(np.nanmax(df["clip_score"].values)),
        "clip_score_median": float(np.nanmedian(df["clip_score"].values)),
        "bert_score_mean": float(np.nanmean(df["bert_score"].dropna().values))
        if df["bert_score"].dropna().size > 0
        else None,
        "bert_score_std": float(np.nanstd(df["bert_score"].dropna().values))
        if df["bert_score"].dropna().size > 0
        else None,
        "avg_word_count": float(avg_len),
        "median_word_count": float(median_len),
        "unique_words": int(vocab_size),
        "total_words": int(sum(word_counter.values())),
        "vocabulary_entropy": float(entropy_from_counts(word_counter)),
        "protected_gender_rate": float((df["protected_gender"] > 0).mean()),
        "protected_race_rate": float((df["protected_race"] > 0).mean()),
        "protected_age_rate": float((df["protected_age"] > 0).mean()),
        "avg_object_coverage": float(
            np.nanmean([v for v in df["obj_miss_frac"].dropna()])
        )
        if df["obj_miss_frac"].dropna().size > 0
        else None,
        "duplicate_rate": float(dup_rate),
        "word_dist_entropy": float(entropy_from_counts(word_counter)),
        "word_dist_gini": float(gini_from_counts(word_counter)),
        "word_dist_top10_share": float(topk_share(word_counter, k=10)),
        # object-level imbalance
        "obj_num_unique": int(len(obj_counter)),
        "obj_total_mentions": int(sum(obj_counter.values())),
        "obj_entropy_bits": float(obj_entropy),
        "obj_gini": float(obj_gini),
        "obj_top5_share": float(obj_top5),
        "obj_top10_share": float(obj_top10),
        "obj_rare_count": int(len(rare_classes_sorted)),
        "obj_rare_threshold": int(rare_threshold),
        "obj_miss_rate": float(miss_count / max(1, total_with_objs)),
        "imbalance_warnings": imbalance_warnings,
    }

    # 10) Export CSVs & JSONs
    df.to_csv(output_dir / "per_example_scores.csv", index=False)
    reweight_df.to_csv(output_dir / "reweighting_probs.csv", index=False)
    pd.DataFrame(obj_counter.most_common(), columns=["object", "count"]).to_csv(
        output_dir / "object_counts.csv", index=False
    )
    rare_df.to_csv(output_dir / f"objects_below_{rare_threshold}.csv", index=False)

    # Convert numpy types to Python native types for JSON serialization
    def convert_to_json_serializable(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        return obj

    summary_json_safe = convert_to_json_serializable(summary)

    with open(output_dir / "quality_summary.json", "w") as fh:
        json.dump(summary_json_safe, fh, indent=2)

    # 11) Rank failures & export top-K artifacts (same as earlier)
    df["combined_score"] = 0.0
    # combined score: low CLIP, low BERT, high obj_miss_frac
    clip_norm = (df["clip_score"].fillna(-1).values + 1.0) / 2.0
    berts = df["bert_score"].fillna(0).values
    objmiss = df["obj_miss_frac"].fillna(0).values
    combined = (
        args.w_clip * (1.0 - clip_norm)
        + args.w_bert * (1.0 - berts)
        + args.w_obj * objmiss
    )
    if combined.max() - combined.min() > 1e-12:
        combined_n = (combined - combined.min()) / (combined.max() - combined.min())
    else:
        combined_n = combined * 0.0
    df["combined_score"] = combined_n
    df_sorted = df.sort_values("combined_score", ascending=False)
    df_sorted.to_csv(output_dir / "ranked_by_combined.csv", index=False)

    sample_dir = output_dir / "top_failures"
    sample_dir.mkdir(exist_ok=True)
    topk = min(args.topk, len(df_sorted))
    for _, row in df_sorted.head(topk).iterrows():
        idx = int(row["idx"])
        meta_txt = (
            f"idx: {idx}\n"
            f"alt_text: {row['alt_text']}\n"
            f"original_text: {row['original_text']}\n"
            f"objects: {row['objects']}\n"
            f"clip_score: {row['clip_score']}\n"
            f"bert_score: {row['bert_score']}\n"
            f"obj_miss_frac: {row['obj_miss_frac']}\n"
            f"combined_score: {row['combined_score']}\n"
        )
        with open(sample_dir / f"{idx}.txt", "w", encoding="utf-8") as fh:
            fh.write(meta_txt)
        img = load_image_from_row(ds[idx])
        if img is not None:
            try:
                save_name = (
                    sample_dir
                    / f"{idx}_c{row['clip_score']:.3f}_b{(row['bert_score'] or 0):.3f}.jpg"
                )
                img.save(save_name)
            except Exception:
                pass

    # 12) Plots (Lorenz, log-log freq, caption length)
    if args.plot:
        print("Generating plots...")
        # Lorenz curve
        x, y = lorenz_curve(obj_counter)
        plt.figure(figsize=(6, 6))
        plt.plot(np.concatenate([[0.0], x]), np.concatenate([[0.0], y]), marker="o")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.title("Lorenz Curve (object distribution)")
        plt.xlabel("Cumulative share of classes")
        plt.ylabel("Cumulative share of mentions")
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / "lorenz_curve.png", bbox_inches="tight")
        plt.close()

        # log-log object frequency
        obj_df = pd.DataFrame(obj_counter.most_common(), columns=["object", "count"])
        plt.figure(figsize=(8, 5))
        plt.loglog(
            range(1, len(obj_df) + 1),
            obj_df["count"].values,
            marker="o",
            linestyle="None",
        )
        plt.title("Object frequency (log-log)")
        plt.xlabel("Rank")
        plt.ylabel("Count")
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.savefig(output_dir / "object_freq_loglog.png", bbox_inches="tight")
        plt.close()

        # caption length histogram
        plt.figure(figsize=(6, 4))
        plt.hist([int(x) for x in length_dist if x is not None], bins=30)
        plt.title("Caption length distribution (tokens)")
        plt.xlabel("Tokens")
        plt.ylabel("Number of captions")
        plt.savefig(output_dir / "caption_length_hist.png", bbox_inches="tight")
        plt.close()

    # Helper functions for quality assessment
    def assess_clip_score(score):
        """Assess CLIP score quality (0-1 scale, higher is better)"""
        if score >= 0.35:
            return "✅ EXCELLENT", "Very strong image-text alignment"
        elif score >= 0.30:
            return "✅ GOOD", "Good image-text alignment"
        elif score >= 0.25:
            return "⚠️ FAIR", "Acceptable but could be improved"
        elif score >= 0.20:
            return "❌ POOR", "Weak image-text alignment"
        else:
            return "❌ VERY POOR", "Very weak alignment, major issues"

    def assess_bert_score(score):
        """Assess BERT score quality (0-1 scale, higher is better)"""
        if score >= 0.90:
            return "✅ EXCELLENT", "Very high fidelity to original"
        elif score >= 0.85:
            return "✅ GOOD", "Good fidelity to original"
        elif score >= 0.80:
            return "⚠️ FAIR", "Moderate changes from original"
        elif score >= 0.70:
            return "❌ POOR", "Significant deviation from original"
        else:
            return "❌ VERY POOR", "Major rewrite, low fidelity"

    def assess_protected_rate(rate, category):
        """Assess protected attribute mention rate"""
        if category == "gender":
            if rate <= 0.01:
                return "✅ EXCELLENT", "Bias successfully eliminated"
            elif rate <= 0.05:
                return "✅ GOOD", "Very low bias"
            elif rate <= 0.15:
                return "⚠️ FAIR", "Some bias remains"
            else:
                return "❌ POOR", "High bias, needs improvement"
        elif category == "race":
            if rate <= 0.02:
                return "✅ EXCELLENT", "Bias successfully eliminated"
            elif rate <= 0.10:
                return "✅ GOOD", "Low bias"
            elif rate <= 0.20:
                return "⚠️ FAIR", "Some bias remains"
            else:
                return "❌ POOR", "High bias, needs improvement"
        else:  # age
            # Age mentions (child, elderly) are more acceptable per guidelines
            if rate <= 0.20:
                return "✅ GOOD", "Acceptable age mentions"
            elif rate <= 0.40:
                return "⚠️ FAIR", "Moderate age mentions"
            else:
                return "❌ HIGH", "High age mentions"

    def assess_duplicate_rate(rate):
        """Assess duplicate rate"""
        if rate <= 0.01:
            return "✅ EXCELLENT", "Virtually no duplicates"
        elif rate <= 0.05:
            return "✅ GOOD", "Very low duplicates"
        elif rate <= 0.10:
            return "⚠️ FAIR", "Some duplicates"
        else:
            return "❌ HIGH", "High duplicate rate"

    # 13) Print enhanced human-friendly summary
    print("\n" + "=" * 80)
    print("DATASET QUALITY REPORT".center(80))
    print("=" * 80)

    print(f"\n📊 Dataset: {summary['dataset']}")
    print(f"📈 Samples: {summary['total_samples']} ({summary['samples_with_images']} with images)\n")

    # CLIP Score Assessment
    clip_rating, clip_desc = assess_clip_score(summary['clip_score_mean'])
    print("─" * 80)
    print("🖼️  IMAGE-TEXT ALIGNMENT (CLIP Score)")
    print("─" * 80)
    print(f"Score:       {summary['clip_score_mean']:.4f} ± {summary['clip_score_std']:.4f}")
    print(f"Range:       [{summary['clip_score_min']:.4f}, {summary['clip_score_max']:.4f}]")
    print(f"Median:      {summary['clip_score_median']:.4f}")
    print(f"Assessment:  {clip_rating} - {clip_desc}")
    print(f"Reference:   >0.35=Excellent, 0.30-0.35=Good, 0.25-0.30=Fair, <0.25=Poor")

    # BERT Score Assessment
    if summary["bert_score_mean"] is not None:
        bert_rating, bert_desc = assess_bert_score(summary['bert_score_mean'])
        print("\n" + "─" * 80)
        print("📝 CAPTION FIDELITY (BERT Score)")
        print("─" * 80)
        print(f"Score:       {summary['bert_score_mean']:.4f} ± {summary['bert_score_std']:.4f}")
        print(f"Assessment:  {bert_rating} - {bert_desc}")
        print(f"Reference:   >0.90=Excellent, 0.85-0.90=Good, 0.80-0.85=Fair, <0.80=Poor")

    # Caption Statistics
    dup_rating, dup_desc = assess_duplicate_rate(summary['duplicate_rate'])
    print("\n" + "─" * 80)
    print("📊 CAPTION STATISTICS")
    print("─" * 80)
    print(f"Average length:    {summary['avg_word_count']:.1f} words")
    print(f"Median length:     {summary['median_word_count']:.1f} words")
    print(f"Vocabulary:        {summary['unique_words']} unique words")
    print(f"Duplicates:        {summary['duplicate_rate']*100:.2f}% - {dup_rating} ({dup_desc})")

    # Protected Attributes
    gender_rating, gender_desc = assess_protected_rate(summary['protected_gender_rate'], "gender")
    race_rating, race_desc = assess_protected_rate(summary['protected_race_rate'], "race")
    age_rating, age_desc = assess_protected_rate(summary['protected_age_rate'], "age")

    print("\n" + "─" * 80)
    print("⚖️  BIAS DETECTION (Protected Attributes)")
    print("─" * 80)
    print(f"Gender mentions:   {summary['protected_gender_rate']*100:.2f}% - {gender_rating} ({gender_desc})")
    print(f"Race mentions:     {summary['protected_race_rate']*100:.2f}% - {race_rating} ({race_desc})")
    print(f"Age mentions:      {summary['protected_age_rate']*100:.2f}% - {age_rating} ({age_desc})")
    print(f"Note: Lower is better for gender/race; age mentions (child/elderly) are acceptable")

    # Object Distribution
    print("\n" + "─" * 80)
    print("🏷️  OBJECT DISTRIBUTION")
    print("─" * 80)
    print("Most common objects (top 10):")
    for i, (ob, cnt) in enumerate(obj_counter.most_common(10), 1):
        print(f"  {i:2}. {ob:20} {cnt:4} mentions")

    print(f"\nDistribution metrics:")
    print(f"  Unique objects:    {summary['obj_num_unique']}")
    print(f"  Total mentions:    {summary['obj_total_mentions']}")
    print(f"  Entropy:           {summary['obj_entropy_bits']:.3f} bits (higher = more diverse)")
    print(f"  Gini coefficient:  {summary['obj_gini']:.3f} (0=equal, 1=unequal)")
    print(f"  Top-5 share:       {summary['obj_top5_share']*100:.1f}%")
    print(f"  Top-10 share:      {summary['obj_top10_share']*100:.1f}%")
    print(f"  Rare classes:      {summary['obj_rare_count']} (count < {rare_threshold})")

    if imbalance_warnings:
        print("\n⚠️  IMBALANCE WARNINGS:")
        for w in imbalance_warnings:
            print(f"   • {w}")
    else:
        print("\n✅ No major imbalance warnings detected")

    # Overall Quality Summary
    print("\n" + "=" * 80)
    print("OVERALL QUALITY SUMMARY".center(80))
    print("=" * 80)

    quality_checks = []

    # Image-text alignment
    if summary['clip_score_mean'] >= 0.30:
        quality_checks.append("✅ Image-text alignment: GOOD")
    elif summary['clip_score_mean'] >= 0.25:
        quality_checks.append("⚠️ Image-text alignment: FAIR")
    else:
        quality_checks.append("❌ Image-text alignment: NEEDS IMPROVEMENT")

    # Caption fidelity
    if summary["bert_score_mean"] is not None:
        if summary['bert_score_mean'] >= 0.85:
            quality_checks.append("✅ Caption fidelity: GOOD")
        elif summary['bert_score_mean'] >= 0.80:
            quality_checks.append("⚠️ Caption fidelity: FAIR")
        else:
            quality_checks.append("❌ Caption fidelity: NEEDS IMPROVEMENT")

    # Bias elimination
    if summary['protected_gender_rate'] <= 0.05 and summary['protected_race_rate'] <= 0.10:
        quality_checks.append("✅ Bias elimination: SUCCESSFUL")
    elif summary['protected_gender_rate'] <= 0.15 and summary['protected_race_rate'] <= 0.20:
        quality_checks.append("⚠️ Bias elimination: PARTIAL")
    else:
        quality_checks.append("❌ Bias elimination: NEEDS IMPROVEMENT")

    # Duplicates
    if summary['duplicate_rate'] <= 0.05:
        quality_checks.append("✅ Duplicate rate: LOW")
    elif summary['duplicate_rate'] <= 0.10:
        quality_checks.append("⚠️ Duplicate rate: MODERATE")
    else:
        quality_checks.append("❌ Duplicate rate: HIGH")

    for check in quality_checks:
        print(check)

    # Output files
    print("\n" + "─" * 80)
    print("📁 OUTPUT FILES")
    print("─" * 80)
    print(f"Directory: {output_dir}/")
    print("  • per_example_scores.csv       - Detailed scores for each sample")
    print("  • ranked_by_combined.csv       - Samples ranked by quality")
    print("  • object_counts.csv            - Object frequency distribution")
    print("  • reweighting_probs.csv        - Sampling probabilities")
    print(f"  • objects_below_{rare_threshold}.csv       - Rare/underrepresented objects")
    print("  • quality_summary.json         - All metrics in JSON format")
    print(f"  • top_failures/                - Top {min(args.topk, len(df_sorted))} failure examples with images")

    if args.plot:
        print("\n📊 PLOTS:")
        print("  • lorenz_curve.png             - Distribution inequality curve")
        print("  • object_freq_loglog.png       - Object frequency (log-log)")
        print("  • caption_length_hist.png      - Caption length histogram")

    print("\n" + "=" * 80)

    return df, summary


# ------------------------
# CLI
# ------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compute dataset quality metrics for image captioning datasets (enhanced)"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="Mozilla/flickr30k-transformed-captions-gpt4o",
        help="HF dataset name or path",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to analyze (default: test)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to analyze (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./quality_reports",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for CLIP scoring"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cuda/mps/cpu)"
    )
    parser.add_argument(
        "--topk", type=int, default=300, help="How many top failures to export"
    )
    parser.add_argument(
        "--rare-threshold",
        type=int,
        default=50,
        help="Threshold below which a class is considered rare",
    )
    parser.add_argument(
        "--imbalance-gini-alert",
        type=float,
        default=0.5,
        help="Gini threshold to raise imbalance warning (0-1)",
    )
    parser.add_argument(
        "--imbalance-topk-alert",
        type=float,
        default=0.5,
        help="Top-k share threshold (e.g., top-5) to raise imbalance warning (0-1)",
    )
    parser.add_argument(
        "--w-clip",
        dest="w_clip",
        type=float,
        default=0.4,
        help="Weight for CLIP term in combined score",
    )
    parser.add_argument(
        "--w-bert", dest="w_bert", type=float, default=0.3, help="Weight for BERT term"
    )
    parser.add_argument(
        "--w-obj",
        dest="w_obj",
        type=float,
        default=0.3,
        help="Weight for object-miss term",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots (Lorenz, log-log, length hist)",
    )
    args = parser.parse_args()

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
