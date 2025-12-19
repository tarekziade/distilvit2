"""
Test loading COCO arrow files directly
"""
import os
import glob
from datasets import Dataset, concatenate_datasets

cache_path = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--Mozilla--coco-gpt4o/snapshots/*/train"
)

train_paths = glob.glob(cache_path)

if not train_paths:
    print("ERROR: No train paths found")
    exit(1)

train_dir = train_paths[0]
print(f"Loading from: {train_dir}")

arrow_files = glob.glob(os.path.join(train_dir, "*.arrow"))
print(f"Found {len(arrow_files)} arrow files")

if not arrow_files:
    print("ERROR: No arrow files found")
    exit(1)

# Load first file to inspect schema
print(f"\nLoading first arrow file: {arrow_files[0]}")
ds_first = Dataset.from_file(arrow_files[0])

print(f"\n{'='*80}")
print(f"First dataset loaded successfully!")
print(f"{'='*80}")
print(f"Number of examples: {len(ds_first)}")
print(f"Columns: {ds_first.column_names}")
print(f"\nFirst example keys: {list(ds_first[0].keys())}")

# Check image and alt_text fields
print(f"\n{'='*80}")
print("Checking 'image' field:")
print(f"{'='*80}")
image_field = ds_first[0]['image']
print(f"Type: {type(image_field)}")
if isinstance(image_field, dict):
    print(f"Keys: {list(image_field.keys())}")
    if 'bytes' in image_field:
        print(f"Has 'bytes' key with {len(image_field['bytes'])} bytes")
    if 'path' in image_field:
        print(f"Has 'path': {image_field['path']}")

print(f"\n{'='*80}")
print("Checking 'alt_text' field:")
print(f"{'='*80}")
alt_text_field = ds_first[0]['alt_text']
print(f"Type: {type(alt_text_field)}")
print(f"Value: {alt_text_field}")

# Try loading multiple files
print(f"\n{'='*80}")
print("Loading first 5 arrow files and concatenating...")
print(f"{'='*80}")
datasets_list = []
for arrow_file in sorted(arrow_files)[:5]:
    ds_part = Dataset.from_file(arrow_file)
    datasets_list.append(ds_part)
    print(f"  Loaded {len(ds_part)} examples from {os.path.basename(arrow_file)}")

ds_combined = concatenate_datasets(datasets_list)
print(f"\nCombined dataset: {len(ds_combined)} total examples")
print(f"Columns: {ds_combined.column_names}")

print(f"\n{'='*80}")
print("SUCCESS! Arrow files can be loaded directly")
print(f"{'='*80}")
