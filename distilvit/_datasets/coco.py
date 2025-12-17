"""
Tokenizes the COCO dataset with GPT-4o captions
"""

from distilvit.utils import DatasetTokenizer, cached_ds


@cached_ds("coco")
def get_dataset(feature_extractor_model, text_decoder_model, args):
    import os
    import glob
    from datasets import Dataset, DatasetDict, concatenate_datasets

    # Load arrow files directly from cache to bypass schema validation issues
    cache_base = os.path.expanduser(
        os.path.join("~", ".cache", "huggingface", "hub", "datasets--Mozilla--coco-gpt4o", "snapshots")
    )

    if not os.path.exists(cache_base):
        print(f"Warning: Cache directory not found: {cache_base}")
        print("Falling back to empty dataset - training will skip COCO")
        import pyarrow as pa
        empty_table = pa.table({
            'image': [],
            'alt_text': []
        })
        empty_ds = Dataset(empty_table)
        return DatasetDict({
            'train': empty_ds,
            'test': empty_ds,
            'validation': empty_ds
        })

    # Find the snapshot directory (should be only one)
    snapshot_dirs = [d for d in os.listdir(cache_base) if os.path.isdir(os.path.join(cache_base, d))]
    if not snapshot_dirs:
        print(f"Warning: No snapshot directories found in {cache_base}")
        print("Falling back to empty dataset - training will skip COCO")
        import pyarrow as pa
        empty_table = pa.table({
            'image': [],
            'alt_text': []
        })
        empty_ds = Dataset(empty_table)
        return DatasetDict({
            'train': empty_ds,
            'test': empty_ds,
            'validation': empty_ds
        })

    train_dir = os.path.join(cache_base, snapshot_dirs[0], "train")
    if not os.path.exists(train_dir):
        print(f"Warning: Train directory not found: {train_dir}")
        print("Falling back to empty dataset - training will skip COCO")
        import pyarrow as pa
        empty_table = pa.table({
            'image': [],
            'alt_text': []
        })
        empty_ds = Dataset(empty_table)
        return DatasetDict({
            'train': empty_ds,
            'test': empty_ds,
            'validation': empty_ds
        })

    print(f"Loading COCO from cached arrow files: {train_dir}")

    # Load arrow files directly
    try:
        arrow_files = glob.glob(os.path.join(train_dir, "*.arrow"))
        if not arrow_files:
            raise FileNotFoundError("No arrow files found")

        # Load from arrow files (all 115 files for full dataset)
        datasets_list = []
        print(f"Loading {len(arrow_files)} arrow files...")
        for i, arrow_file in enumerate(sorted(arrow_files)):
            ds_part = Dataset.from_file(arrow_file)
            datasets_list.append(ds_part)
            if (i + 1) % 20 == 0:
                print(f"  Loaded {i + 1}/{len(arrow_files)} files...")

        ds = concatenate_datasets(datasets_list)

        print(f"[OK] Loaded {len(ds)} COCO examples from {len(datasets_list)} arrow files")

        # Wrap alt_text as a list for consistency with other datasets
        ds = ds.map(lambda ex: {"alt_text": [ex["alt_text"]]})

    except Exception as e:
        print(f"Warning: Could not load COCO arrow files: {e}")
        import traceback
        traceback.print_exc()
        print("Falling back to empty dataset - training will skip COCO")
        import pyarrow as pa
        empty_table = pa.table({
            'image': [],
            'alt_text': []
        })
        empty_ds = Dataset(empty_table)
        return DatasetDict({
            'train': empty_ds,
            'test': empty_ds,
            'validation': empty_ds
        })

    # Sample if requested
    if args.sample:
        ds = ds.select(range(min(args.sample, len(ds))))

    # Tokenize captions
    ds_tokenizer = DatasetTokenizer(
        feature_extractor_model,
        text_decoder_model,
        caption_column="alt_text",
    )

    ds = ds_tokenizer("coco", ds)

    # splitting in train (80%), test (10%), validation (10%)
    ds = ds.train_test_split(test_size=0.2)
    test_and_eval = ds["test"].train_test_split(test_size=0.5)
    ds["test"] = test_and_eval["test"]
    ds["validation"] = test_and_eval["train"]
    return ds
