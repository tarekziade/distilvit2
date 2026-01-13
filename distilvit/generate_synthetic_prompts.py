#!/usr/bin/env python3
"""
generate_synthetic_prompts.py

Generate synthetic image prompts for rare objects to balance the dataset.

Usage:
    python distilvit/generate_synthetic_prompts.py \
        --rare-objects quality_reports/objects_below_50.csv \
        --output prompts.jsonl \
        --prompts-per-object 5 \
        --include-combinations

This generates bias-free captions featuring rare objects that can be used with:
- Stable Diffusion / DALL-E / Midjourney for synthetic image generation
- The generated images can then be added to your training set
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict


# Bias-free caption templates following GPT-4 transformation guidelines
# No gender, race, age descriptors - focus on actions, objects, and scenes
CAPTION_TEMPLATES = {
    # Indoor scenes
    "indoor": [
        "A {object} in a cozy room with natural lighting",
        "A detailed view of a {object} on a wooden table",
        "A {object} placed near a window with soft daylight",
        "An indoor scene featuring a {object} in the foreground",
        "A minimalist composition with a {object} as the focal point",
        "A {object} in a well-lit interior space",
        "A close-up photograph of a {object} indoors",
    ],

    # Outdoor scenes
    "outdoor": [
        "A {object} in a natural outdoor setting",
        "A {object} photographed in bright daylight",
        "An outdoor scene with a {object} in the center",
        "A {object} against a scenic background",
        "A {object} in a park or garden setting",
        "A landscape photograph featuring a {object}",
        "A {object} outdoors with natural surroundings",
    ],

    # Action/Activity scenes
    "activity": [
        "A scene showing interaction with a {object}",
        "A {object} being used in an activity",
        "An action shot featuring a {object}",
        "A dynamic scene with a {object} in motion",
        "A {object} in active use",
        "A scene capturing movement around a {object}",
    ],

    # Object-focused
    "object_focus": [
        "A professional photograph of a {object}",
        "A detailed close-up of a {object}",
        "A {object} with clear focus and lighting",
        "A studio photograph of a {object}",
        "A {object} captured with shallow depth of field",
        "An artistic composition featuring a {object}",
    ],

    # Contextual scenes
    "contextual": [
        "A {object} in its typical environment",
        "A realistic scene with a {object}",
        "A {object} in everyday context",
        "A candid photograph showing a {object}",
        "A {object} in a natural setting",
    ],
}


# Contextual modifiers to add variety (all bias-free)
MODIFIERS = {
    "lighting": [
        "with dramatic lighting",
        "in golden hour light",
        "with soft natural light",
        "in bright sunlight",
        "with ambient lighting",
        "at sunset",
        "in daylight",
        "with backlight",
    ],
    "composition": [
        "from a low angle",
        "from above",
        "with symmetrical composition",
        "with rule of thirds",
        "in the foreground",
        "centered in the frame",
        "with blurred background",
    ],
    "style": [
        "in photorealistic style",
        "as a high-quality photograph",
        "with vibrant colors",
        "in natural colors",
        "with shallow depth of field",
        "with sharp focus",
    ],
    "quality": [
        "highly detailed",
        "4K quality",
        "professional photography",
        "award-winning photograph",
        "magazine quality",
    ],
}


def load_rare_objects(csv_path, max_count=None):
    """
    Load rare objects from quality report CSV.

    Args:
        csv_path: Path to objects_below_N.csv
        max_count: Only include objects with count <= max_count (optional)

    Returns:
        List of (object, count) tuples
    """
    import csv

    rare_objects = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            obj = row['object'].strip()
            count = int(row['count'])

            if max_count is None or count <= max_count:
                rare_objects.append((obj, count))

    return rare_objects


def generate_caption(obj, template_category=None, add_modifiers=True):
    """
    Generate a bias-free caption for an object.

    Args:
        obj: Object name
        template_category: Specific template category (or random if None)
        add_modifiers: Whether to add contextual modifiers

    Returns:
        Generated caption string
    """
    # Choose template
    if template_category and template_category in CAPTION_TEMPLATES:
        templates = CAPTION_TEMPLATES[template_category]
    else:
        # Random category
        templates = []
        for cat_templates in CAPTION_TEMPLATES.values():
            templates.extend(cat_templates)

    template = random.choice(templates)
    caption = template.format(object=obj)

    # Add modifiers for variety
    if add_modifiers and random.random() > 0.3:  # 70% chance
        # Add 1-2 modifiers
        num_modifiers = random.randint(1, 2)
        modifier_categories = random.sample(list(MODIFIERS.keys()),
                                          min(num_modifiers, len(MODIFIERS)))

        for mod_cat in modifier_categories:
            modifier = random.choice(MODIFIERS[mod_cat])
            caption = f"{caption}, {modifier}"

    return caption


def generate_combination_caption(objects, max_objects=3):
    """
    Generate caption featuring multiple rare objects together.

    This helps the model learn relationships between objects.
    """
    # Sample 2-3 objects
    num_objs = random.randint(2, min(max_objects, len(objects)))
    selected = random.sample(objects, num_objs)

    # Create natural combination
    if len(selected) == 2:
        obj_phrase = f"a {selected[0]} and a {selected[1]}"
    else:
        obj_phrase = ", ".join([f"a {o}" for o in selected[:-1]]) + f", and a {selected[-1]}"

    templates = [
        f"A scene with {obj_phrase}",
        f"A photograph showing {obj_phrase}",
        f"An image featuring {obj_phrase} together",
        f"A composition with {obj_phrase} in view",
        f"A setting containing {obj_phrase}",
    ]

    caption = random.choice(templates)

    # Add context
    if random.random() > 0.5:
        contexts = [
            "in a natural setting",
            "in an indoor environment",
            "outdoors with good lighting",
            "arranged artistically",
            "in everyday context",
        ]
        caption = f"{caption}, {random.choice(contexts)}"

    return caption, selected


def generate_prompts(rare_objects, prompts_per_object=5, include_combinations=True,
                    combination_count=None):
    """
    Generate synthetic image prompts for rare objects.

    Returns:
        List of dicts with prompt information
    """
    prompts = []

    # Single-object prompts
    print(f"Generating {prompts_per_object} prompts for each of {len(rare_objects)} rare objects...")
    for obj, count in rare_objects:
        for i in range(prompts_per_object):
            # Vary template categories
            categories = list(CAPTION_TEMPLATES.keys())
            category = categories[i % len(categories)]

            caption = generate_caption(obj, template_category=category,
                                     add_modifiers=True)

            prompts.append({
                "prompt": caption,
                "objects": [obj],
                "original_count": count,
                "type": "single_object",
            })

    # Multi-object combination prompts
    if include_combinations:
        if combination_count is None:
            # Generate combinations proportional to number of rare objects
            combination_count = min(len(rare_objects) * 2, 100)

        print(f"Generating {combination_count} multi-object combination prompts...")
        object_names = [obj for obj, _ in rare_objects]

        for _ in range(combination_count):
            caption, selected_objs = generate_combination_caption(object_names)

            prompts.append({
                "prompt": caption,
                "objects": selected_objs,
                "original_count": [count for obj, count in rare_objects if obj in selected_objs],
                "type": "combination",
            })

    return prompts


def save_prompts(prompts, output_path, format="jsonl"):
    """
    Save prompts to file in specified format.

    Formats:
        - jsonl: One JSON object per line (recommended)
        - json: Single JSON array
        - txt: Plain text, one prompt per line
        - csv: CSV with metadata
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "jsonl":
        with open(output_path, 'w') as f:
            for prompt_data in prompts:
                f.write(json.dumps(prompt_data) + '\n')

    elif format == "json":
        with open(output_path, 'w') as f:
            json.dump(prompts, f, indent=2)

    elif format == "txt":
        with open(output_path, 'w') as f:
            for prompt_data in prompts:
                f.write(prompt_data['prompt'] + '\n')

    elif format == "csv":
        import csv
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['prompt', 'objects', 'type', 'original_count'])
            for prompt_data in prompts:
                writer.writerow([
                    prompt_data['prompt'],
                    ';'.join(prompt_data['objects']),
                    prompt_data['type'],
                    prompt_data.get('original_count', '')
                ])

    print(f"✅ Saved {len(prompts)} prompts to {output_path}")


def print_summary(prompts, rare_objects):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("SYNTHETIC PROMPT GENERATION SUMMARY")
    print("="*80)

    total = len(prompts)
    single_obj = sum(1 for p in prompts if p['type'] == 'single_object')
    combinations = sum(1 for p in prompts if p['type'] == 'combination')

    print(f"\n📊 Generated Prompts:")
    print(f"  Total:                {total}")
    print(f"  Single-object:        {single_obj}")
    print(f"  Multi-object combos:  {combinations}")

    print(f"\n🎯 Rare Objects Covered:")
    print(f"  Total rare objects:   {len(rare_objects)}")

    # Show most underrepresented
    print(f"\n⚠️  Most Underrepresented (1-2 samples):")
    very_rare = [obj for obj, count in rare_objects if count <= 2][:10]
    for obj in very_rare:
        print(f"    • {obj}")

    # Object coverage in prompts
    object_coverage = defaultdict(int)
    for prompt_data in prompts:
        for obj in prompt_data['objects']:
            object_coverage[obj] += 1

    print(f"\n📈 Objects per Prompt (avg): {sum(len(p['objects']) for p in prompts) / len(prompts):.1f}")
    print(f"\n💡 Next Steps:")
    print(f"  1. Generate images using these prompts with Stable Diffusion/DALL-E")
    print(f"  2. Use generated images to augment your training dataset")
    print(f"  3. Re-run quality analysis to verify improved balance")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic image prompts for rare objects",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--rare-objects",
        type=str,
        required=True,
        help="Path to objects_below_N.csv from quality report"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="synthetic_prompts.jsonl",
        help="Output file path"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="jsonl",
        choices=["jsonl", "json", "txt", "csv"],
        help="Output format"
    )
    parser.add_argument(
        "--prompts-per-object",
        type=int,
        default=5,
        help="Number of prompt variations per rare object"
    )
    parser.add_argument(
        "--max-count",
        type=int,
        default=None,
        help="Only include objects with count <= this value"
    )
    parser.add_argument(
        "--include-combinations",
        action="store_true",
        default=True,
        help="Generate multi-object combination prompts"
    )
    parser.add_argument(
        "--combination-count",
        type=int,
        default=None,
        help="Number of combination prompts (default: 2x rare objects, max 100)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Load rare objects
    print(f"Loading rare objects from {args.rare_objects}...")
    rare_objects = load_rare_objects(args.rare_objects, max_count=args.max_count)
    print(f"Found {len(rare_objects)} rare objects")

    # Generate prompts
    prompts = generate_prompts(
        rare_objects,
        prompts_per_object=args.prompts_per_object,
        include_combinations=args.include_combinations,
        combination_count=args.combination_count
    )

    # Save prompts
    save_prompts(prompts, args.output, format=args.format)

    # Print summary
    print_summary(prompts, rare_objects)

    print(f"\n✨ Done! Use these prompts with your image generation tool.")


if __name__ == "__main__":
    main()
