"""
Tokenizes the COCO dataset with GPT-4o captions
"""

from distilvit.utils import DatasetTokenizer, cached_ds


@cached_ds("coco")
def get_dataset(feature_extractor_model, text_decoder_model, args):
    from datasets import Dataset, load_dataset
    import pyarrow.parquet as pq
    import os

    # Path to the cached arrow files
    cache_path = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--Mozilla--coco-gpt4o/snapshots"
    )

    # Find the snapshot directory
    snapshot_dirs = []
    if os.path.exists(cache_path):
        for item in os.listdir(cache_path):
            snapshot_dir = os.path.join(cache_path, item)
            if os.path.isdir(snapshot_dir):
                snapshot_dirs.append(snapshot_dir)

    if not snapshot_dirs:
        # Fallback to normal loading if cache not found
        ds = load_dataset("Mozilla/coco-gpt4o")
        if "train" in ds:
            ds = ds["train"]
        else:
            ds = ds[list(ds.keys())[0]]
    else:
        # Use the first (likely only) snapshot
        snapshot_dir = snapshot_dirs[0]

        # Look for train split arrow files
        train_dir = os.path.join(snapshot_dir, "train")
        if not os.path.exists(train_dir):
            # Try other split names
            for split_name in ["test", "validation"]:
                split_dir = os.path.join(snapshot_dir, split_name)
                if os.path.exists(split_dir):
                    train_dir = split_dir
                    break

        # Load arrow files directly without schema validation
        from datasets import load_from_disk
        import pyarrow as pa

        arrow_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.arrow')]

        if arrow_files:
            # Read first arrow file to get the actual schema
            import pyarrow.ipc as ipc
            with pa.memory_map(arrow_files[0], 'r') as source:
                reader = ipc.open_file(source)
                table = reader.read_all()

            # Concatenate all tables
            tables = [table]
            for arrow_file in arrow_files[1:]:
                with pa.memory_map(arrow_file, 'r') as source:
                    reader = ipc.open_file(source)
                    tables.append(reader.read_all())

            full_table = pa.concat_tables(tables)
            ds = Dataset(full_table)
        else:
            # Fallback
            ds = load_dataset("Mozilla/coco-gpt4o", split="train")

    # Sample if requested
    if args.sample:
        ds = ds.select(range(min(args.sample, len(ds))))

    # alt_text is already a list in this dataset, no need to wrap it again
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
