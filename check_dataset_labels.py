
import random
from pathlib import Path
from collections import Counter

import cv2
import matplotlib.pyplot as plt


# Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "data" / "raw"

CLASS_NAMES = [
    "Acne-Vulgaris",    # 0
    "Eczema",           # 1
    "Tinea-Versicolor", # 2
    "Psoriasis"         # 3
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
random.seed(42)


def check_split(split):
    images_dir = DATASET_DIR / split / "images"
    labels_dir = DATASET_DIR / split / "labels"

    counts = Counter()
    missing = []
    empty = []
    invalid = []
    multiple_classes = []
    samples = {i: [] for i in range(len(CLASS_NAMES))}

    if not images_dir.exists() or not labels_dir.exists():
        print(f"\n{split}: Images or labels folder not found.")
        return

    image_files = [
        p for p in images_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    for image_path in image_files:
        label_path = labels_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            missing.append(image_path.name)
            continue

        lines = label_path.read_text(encoding="utf-8").splitlines()

        if not lines:
            empty.append(image_path.name)
            continue

        image_class_ids = set()
        valid_file = True

        for line in lines:
            parts = line.split()

            try:
                if len(parts) != 5:
                    raise ValueError("Expected 5 YOLO values")

                class_id = int(parts[0])

                if not 0 <= class_id < len(CLASS_NAMES):
                    raise ValueError("Class ID outside 0-3")

                # Validate the bounding-box numbers
                box = [float(value) for value in parts[1:]]
                if not all(0 <= value <= 1 for value in box):
                    raise ValueError("Bounding-box values outside 0-1")

                image_class_ids.add(class_id)

            except ValueError:
                valid_file = False
                break

        if not valid_file or not image_class_ids:
            invalid.append(image_path.name)
            continue

        if len(image_class_ids) > 1:
            multiple_classes.append(
                (image_path.name, sorted(image_class_ids))
            )

        # Count each image under its first annotation's class.
        # This matches the current training script.
        first_class_id = int(lines[0].split()[0])
        counts[first_class_id] += 1
        samples[first_class_id].append(image_path)

    print(f"\n{'=' * 45}")
    print(f"{split.upper()} DATASET CHECK")
    print(f"{'=' * 45}")
    print("Total images:", len(image_files))

    for class_id, name in enumerate(CLASS_NAMES):
        print(f"{name}: {counts[class_id]} images")

    print("Missing label files:", len(missing))
    print("Empty label files:", len(empty))
    print("Invalid label files:", len(invalid))
    print("Images with multiple classes:", len(multiple_classes))

    if missing:
        print("\nMissing-label examples:", missing[:5])
    if invalid:
        print("Invalid-label examples:", invalid[:5])
    if multiple_classes:
        print("Multiple-class examples:")
        for name, ids in multiple_classes[:5]:
            labels = [CLASS_NAMES[i] for i in ids]
            print(f"  {name}: {labels}")

    # Display up to 4 randomly selected images per class.
    fig, axes = plt.subplots(4, 4, figsize=(14, 12))
    fig.suptitle(f"{split.upper()} — Check the actual image labels")

    for class_id, name in enumerate(CLASS_NAMES):
        chosen = random.sample(
            samples[class_id],
            min(4, len(samples[class_id]))
        )

        for col in range(4):
            ax = axes[class_id, col]
            ax.axis("off")

            if col < len(chosen):
                image = cv2.imread(str(chosen[col]))

                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    ax.imshow(image)
                    ax.set_title(name, fontsize=9)
                    ax.set_xlabel(chosen[col].name[:24], fontsize=7)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    for split in ["train", "valid", "test"]:
        check_split(split)
