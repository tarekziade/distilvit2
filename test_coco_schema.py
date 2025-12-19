"""
Test COCO arrow file schema handling
"""
import os
import glob
from datasets import Dataset, concatenate_datasets, Features, Sequence, Value

cache_base = os.path.expanduser(
    os.path.join("~", ".cache", "huggingface", "hub", "datasets--Mozilla--coco-gpt4o", "snapshots")
)

print(f"Cache base: {cache_base}")
print(f"Exists: {os.path.exists(cache_base)}")

if not os.path.exists(cache_base):
    print("ERROR: Cache directory not found")
    exit(1)

snapshot_dirs = [d for d in os.listdir(cache_base) if os.path.isdir(os.path.join(cache_base, d))]
print(f"Snapshot dirs: {snapshot_dirs}")

if not snapshot_dirs:
    print("ERROR: No snapshot directories found")
    exit(1)

train_dir = os.path.join(cache_base, snapshot_dirs[0], "train")
print(f"Train dir: {train_dir}")
print(f"Train dir exists: {os.path.exists(train_dir)}")

if not os.path.exists(train_dir):
    print("ERROR: Train directory not found")
    exit(1)

# Only load data-*.arrow files, not cache-*.arrow files
arrow_files = glob.glob(os.path.join(train_dir, "data-*.arrow"))
print(f"Found {len(arrow_files)} data arrow files")

if not arrow_files:
    print("ERROR: No arrow files found")
    exit(1)

# Load first 3 arrow files to test schema
print(f"\n{'='*80}")
print("Testing schema normalization...")
print(f"{'='*80}\n")

datasets_list = []
target_features = None

for i, arrow_file in enumerate(sorted(arrow_files)[:3]):
    print(f"\nLoading file {i+1}/3: {os.path.basename(arrow_file)}")
    ds_part = Dataset.from_file(arrow_file)
    print(f"  Rows: {len(ds_part)}")
    print(f"  Columns: {ds_part.column_names}")

    if 'alt_text' in ds_part.column_names:
        first_alt_text = ds_part[0]['alt_text']
        print(f"  alt_text type: {type(first_alt_text)}")
        print(f"  alt_text value: {repr(first_alt_text)[:100]}")
        print(f"  alt_text feature: {ds_part.features['alt_text']}")

        if isinstance(first_alt_text, str):
            print(f"  -> Need to convert string to list")
            # Wrap string values in list
            ds_part = ds_part.map(lambda ex: {"alt_text": [ex["alt_text"]]}, batched=False)
            print(f"  -> After map, alt_text feature: {ds_part.features['alt_text']}")

            # Build target schema from the first file that needs conversion
            if target_features is None:
                target_features = ds_part.features.copy()
                target_features['alt_text'] = Sequence(Value('string'))
                print(f"  -> Set target alt_text feature: {target_features['alt_text']}")

            # Cast to target schema
            ds_part = ds_part.cast(target_features)
            print(f"  -> After cast, alt_text feature: {ds_part.features['alt_text']}")
        elif target_features is None:
            # First file has correct schema already
            target_features = ds_part.features
            print(f"  -> Already list format, using as target schema")

    datasets_list.append(ds_part)

print(f"\n{'='*80}")
print("Concatenating datasets...")
print(f"{'='*80}\n")

try:
    ds = concatenate_datasets(datasets_list)
    print(f"SUCCESS! Concatenated {len(ds)} examples")
    print(f"Columns: {ds.column_names}")
    print(f"alt_text feature: {ds.features['alt_text']}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
