import argparse
import glob
import os
from typing import Dict, Iterable, List

from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Image,
    Sequence,
    Value,
    concatenate_datasets,
    load_dataset,
)


DEFAULT_SAMPLE = None
RANDOM_SEED = 42


def _empty_dataset(source: str) -> Dataset:
    """Return an empty dataset with the expected schema."""
    features = Features(
        {
            "image": Image(),
            "alt_text": Sequence(Value("string")),
            "source": Value("string"),
            "source_id": Value("string"),
        }
    )
    return Dataset.from_dict(
        {"image": [], "alt_text": [], "source": [], "source_id": []}, features=features
    )


def _split(ds: Dataset, seed: int = RANDOM_SEED) -> DatasetDict:
    """Create train/validation/test splits (80/10/10)."""
    ds = ds.train_test_split(test_size=0.2, seed=seed)
    test_and_val = ds["test"].train_test_split(test_size=0.5, seed=seed)
    return DatasetDict(
        {
            "train": ds["train"],
            "validation": test_and_val["train"],
            "test": test_and_val["test"],
        }
    )


def _normalize_alt_text(example):
    alt_text = example.get("alt_text") or example.get("caption") or ""
    if isinstance(alt_text, str):
        alt_text = [alt_text]
    return {"alt_text": alt_text}


def _add_source(example, idx, source: str):
    return {
        "source": source,
        "source_id": f"{source}-{idx}",
    }


def _strip_columns(ds: Dataset, keep: Iterable[str]) -> Dataset:
    columns_to_remove = [col for col in ds.column_names if col not in keep]
    if columns_to_remove:
        ds = ds.remove_columns(columns_to_remove)
    return ds


def load_coco(sample: int | None = DEFAULT_SAMPLE, cache_dir: str | None = None) -> DatasetDict:
    """
    Load COCO captions from the local HF cache (used in training to avoid schema issues).
    """
    cache_root = cache_dir or os.path.join("~", ".cache", "huggingface", "hub")
    cache_base = os.path.expanduser(
        os.path.join(cache_root, "datasets--Mozilla--coco-gpt4o", "snapshots")
    )

    if not os.path.exists(cache_base):
        return DatasetDict({"train": _empty_dataset("coco")})

    snapshot_dirs = [d for d in os.listdir(cache_base) if os.path.isdir(os.path.join(cache_base, d))]
    if not snapshot_dirs:
        return DatasetDict({"train": _empty_dataset("coco")})

    train_dir = os.path.join(cache_base, snapshot_dirs[0], "train")
    arrow_files = sorted(glob.glob(os.path.join(train_dir, "data-*.arrow")))
    if not arrow_files:
        return DatasetDict({"train": _empty_dataset("coco")})

    datasets = []
    target_features = None
    for arrow_file in arrow_files:
        ds_part = Dataset.from_file(arrow_file)
        if "alt_text" in ds_part.column_names:
            first_alt = ds_part[0]["alt_text"]
            if isinstance(first_alt, str):
                ds_part = ds_part.map(lambda ex: {"alt_text": [ex["alt_text"]]})
                if target_features is None:
                    target_features = ds_part.features.copy()
                    target_features["alt_text"] = Sequence(Value("string"))
                ds_part = ds_part.cast(target_features)
            elif target_features is None:
                target_features = ds_part.features
        datasets.append(ds_part)

    ds = concatenate_datasets(datasets)
    if sample:
        ds = ds.select(range(min(sample, len(ds))))

    ds = ds.map(_normalize_alt_text)
    ds = ds.map(lambda ex, idx: _add_source(ex, idx, "coco"), with_indices=True)
    ds = _strip_columns(ds, keep=["image", "alt_text", "source", "source_id"])
    return _split(ds)


def load_flickr(sample: int | None = DEFAULT_SAMPLE, cache_dir: str | None = None) -> DatasetDict:
    split = f"test[:{sample}]" if sample else "test"
    ds = load_dataset("Mozilla/flickr30k-transformed-captions-gpt4o", split=split, cache_dir=cache_dir)
    ds = ds.map(_normalize_alt_text)
    ds = ds.map(lambda ex, idx: _add_source(ex, idx, "flickr"), with_indices=True)
    ds = _strip_columns(ds, keep=["image", "alt_text", "source", "source_id"])
    return _split(ds)


def load_pexels(sample: int | None = DEFAULT_SAMPLE, cache_dir: str | None = None) -> DatasetDict:
    split = f"train[:{sample}]" if sample else "train"
    ds = load_dataset("Mozilla/pexels-gpt4o", split=split, cache_dir=cache_dir)
    ds = ds.map(_normalize_alt_text)
    ds = ds.map(lambda ex, idx: _add_source(ex, idx, "pexels"), with_indices=True)
    ds = _strip_columns(ds, keep=["image", "alt_text", "source", "source_id"])
    return _split(ds)


def load_docornot(sample: int | None = DEFAULT_SAMPLE, cache_dir: str | None = None) -> DatasetDict:
    sample = sample or 2000
    ds = load_dataset("mozilla/docornot", split=f"train[:{sample}]", cache_dir=cache_dir)
    ds = ds.filter(lambda example: example["is_document"] == 1)
    ds = ds.map(lambda _: {"caption": "Text document."})
    ds = ds.map(_normalize_alt_text)
    ds = ds.map(lambda ex, idx: _add_source(ex, idx, "docornot"), with_indices=True)
    ds = _strip_columns(ds, keep=["image", "alt_text", "source", "source_id"])
    return _split(ds)


def load_validation(sample: int | None = DEFAULT_SAMPLE, cache_dir: str | None = None) -> DatasetDict:
    ds = load_dataset("Mozilla/alt-text-validation", split="train", cache_dir=cache_dir)
    ds = ds.filter(lambda x: x["gpt_alt_text"] and x["gpt_alt_text"].strip() != "")
    ds = ds.map(lambda x: {"alt_text": x["gpt_alt_text"]})
    if sample:
        ds = ds.select(range(min(sample, len(ds))))
    ds = ds.map(_normalize_alt_text)
    ds = ds.map(lambda ex, idx: _add_source(ex, idx, "validation"), with_indices=True)
    ds = _strip_columns(ds, keep=["image", "alt_text", "source", "source_id"])
    # Create validation/test splits but keep all data as train as well (small dataset)
    val_size = max(1, int(len(ds) * 0.1)) if len(ds) else 0
    validation = ds.select(range(val_size)) if val_size else _empty_dataset("validation")
    test = ds.select(range(val_size, 2 * val_size)) if val_size else _empty_dataset("validation")
    return DatasetDict({"train": ds, "validation": validation, "test": test})


DATASET_LOADERS = {
    "coco": load_coco,
    "flickr": load_flickr,
    "pexels": load_pexels,
    "docornot": load_docornot,
    "validation": load_validation,
}


def merge_dataset_dicts(datasets: List[DatasetDict]) -> DatasetDict:
    merged: Dict[str, List[Dataset]] = {}
    for ds in datasets:
        for split_name, split_ds in ds.items():
            merged.setdefault(split_name, []).append(split_ds)

    merged_dict = DatasetDict()
    for split_name, split_list in merged.items():
        merged_dict[split_name] = concatenate_datasets(split_list).shuffle(seed=RANDOM_SEED)
    return merged_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge training datasets into a unified Hugging Face dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_LOADERS.keys()),
        choices=list(DATASET_LOADERS.keys()),
        help="Datasets to include in the merge",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=DEFAULT_SAMPLE,
        help="Optional per-dataset sample size (useful for quick tests)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory passed to load_dataset",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="merged_dataset",
        help="Local directory to save the merged dataset",
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        help="Optional Hugging Face repo id to push the merged dataset",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Push the dataset to the hub as a private repo",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    selected = []
    for name in args.datasets:
        loader = DATASET_LOADERS[name]
        ds = loader(sample=args.sample, cache_dir=args.cache_dir)
        selected.append(ds)

    merged = merge_dataset_dicts(selected)
    merged.save_to_disk(args.save_path)

    if args.push_to_hub:
        merged.push_to_hub(args.push_to_hub, private=args.private)


if __name__ == "__main__":
    main()
