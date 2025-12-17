"""
Tokenizes the COCO dataset with GPT-4o captions
"""

from distilvit.utils import DatasetTokenizer, cached_ds


@cached_ds("coco")
def get_dataset(feature_extractor_model, text_decoder_model, args):
    from datasets import load_dataset, VerificationMode

    # Load dataset with verification disabled to ignore schema mismatches
    try:
        ds = load_dataset(
            "Mozilla/coco-gpt4o",
            verification_mode=VerificationMode.NO_CHECKS,
            trust_remote_code=True
        )

        # Use train split if available, otherwise use the first available split
        if "train" in ds:
            ds = ds["train"]
        elif "test" in ds:
            ds = ds["test"]
        else:
            ds = ds[list(ds.keys())[0]]
    except Exception as e:
        print(f"Warning: Could not load COCO dataset: {e}")
        print("Falling back to empty dataset - training will skip COCO")
        from datasets import Dataset
        import pyarrow as pa
        # Return minimal empty dataset with required columns
        empty_table = pa.table({
            'image': [],
            'alt_text': []
        })
        ds = Dataset(empty_table)
        # Create dummy splits
        return {
            'train': ds,
            'test': ds,
            'validation': ds
        }

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
