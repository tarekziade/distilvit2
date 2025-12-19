"""
Debug script to understand COCO dataset loading issues
"""
import os
from datasets import load_dataset, VerificationMode

print("=" * 80)
print("COCO Dataset Debug Script")
print("=" * 80)

# Path to cache
cache_path = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--Mozilla--coco-gpt4o"
)
print(f"\nCache path: {cache_path}")
print(f"Cache exists: {os.path.exists(cache_path)}")

if os.path.exists(cache_path):
    print("\nCache contents:")
    for root, dirs, files in os.walk(cache_path):
        level = root.replace(cache_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files[:10]:  # Limit to first 10 files per directory
            print(f'{subindent}{file}')
        if len(files) > 10:
            print(f'{subindent}... and {len(files) - 10} more files')

print("\n" + "=" * 80)
print("Attempting to load dataset...")
print("=" * 80)

# Try 1: Load with no verification
print("\n[Try 1] Loading with VerificationMode.NO_CHECKS...")
try:
    ds = load_dataset(
        "Mozilla/coco-gpt4o",
        verification_mode=VerificationMode.NO_CHECKS
    )
    print(f"✓ SUCCESS! Loaded dataset")
    print(f"  Splits: {list(ds.keys())}")
    for split_name, split_ds in ds.items():
        print(f"  {split_name}: {len(split_ds)} examples")
        if len(split_ds) > 0:
            print(f"    Columns: {split_ds.column_names}")
            print(f"    First example keys: {list(split_ds[0].keys())}")
            # Check alt_text structure
            first_alt_text = split_ds[0]['alt_text']
            print(f"    alt_text type: {type(first_alt_text)}")
            print(f"    alt_text value: {first_alt_text}")
            break
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Try 2: Load without any parameters
print("\n[Try 2] Loading with default parameters...")
try:
    ds = load_dataset("Mozilla/coco-gpt4o")
    print(f"✓ SUCCESS! Loaded dataset")
    print(f"  Splits: {list(ds.keys())}")
    for split_name, split_ds in ds.items():
        print(f"  {split_name}: {len(split_ds)} examples")
        if len(split_ds) > 0:
            print(f"    Columns: {split_ds.column_names}")
            break
except Exception as e:
    print(f"✗ FAILED: {e}")

# Try 3: Check if we can read arrow files directly
print("\n[Try 3] Checking arrow files directly...")
try:
    import pyarrow as pa
    import pyarrow.ipc as ipc

    # Find snapshot directory
    if os.path.exists(cache_path):
        snapshot_path = os.path.join(cache_path, "snapshots")
        if os.path.exists(snapshot_path):
            snapshots = os.listdir(snapshot_path)
            if snapshots:
                snapshot_dir = os.path.join(snapshot_path, snapshots[0])
                print(f"  Snapshot: {snapshot_dir}")

                # Look for data directories
                for split_name in ["train", "test", "validation"]:
                    split_dir = os.path.join(snapshot_dir, split_name)
                    if os.path.exists(split_dir):
                        print(f"\n  Found split: {split_name}")
                        files = [f for f in os.listdir(split_dir) if f.endswith('.arrow')]
                        print(f"    Arrow files: {len(files)}")

                        if files:
                            # Try to read first file
                            first_file = os.path.join(split_dir, files[0])
                            print(f"    Reading: {files[0]}")
                            try:
                                with pa.memory_map(first_file, 'r') as source:
                                    reader = ipc.open_file(source)
                                    table = reader.read_all()
                                    print(f"    ✓ Successfully read arrow file")
                                    print(f"    Rows: {table.num_rows}")
                                    print(f"    Columns: {table.column_names}")
                                    print(f"    Schema:\n{table.schema}")
                            except Exception as e:
                                print(f"    ✗ Failed to read: {e}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Debug complete")
print("=" * 80)
