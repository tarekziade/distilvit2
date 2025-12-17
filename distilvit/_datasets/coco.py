"""
Tokenizes the COCO dataset with GPT-4o captions
"""

from distilvit.utils import DatasetTokenizer, cached_ds


@cached_ds("coco")
def get_dataset(feature_extractor_model, text_decoder_model, args):
    from datasets import load_dataset

    split = f"train[:{args.sample}]" if args.sample else "train"
    # Load dataset without specifying split to get all available splits
    ds = load_dataset("Mozilla/coco-gpt4o")

    # Use train split if available, otherwise use the first available split
    if "train" in ds:
        ds = ds["train"]
    else:
        # Get first available split
        ds = ds[list(ds.keys())[0]]

    # Sample if requested
    if args.sample:
        ds = ds.select(range(min(args.sample, len(ds))))

    # alt_text is already a list in this dataset, no need to wrap it again
    # Just ensure it's in the right format for DatasetTokenizer

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
