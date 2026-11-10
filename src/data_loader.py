
import os
import cv2
import numpy as np

DATASET_PATH = "data/raw"
IMAGE_SIZE = 128

CLASS_NAMES = [
    "Psoriasis",
    "Eczema",
    "Tinea-Versicolor",
    "Acne-Vulgaris"
]

NUM_CLASSES = len(CLASS_NAMES)


def load_dataset(split):

    images_path = os.path.join(
        DATASET_PATH, split, "images"
    )

    labels_path = os.path.join(
        DATASET_PATH, split, "labels"
    )

    images = []
    labels = []

    image_files = [
        f for f in os.listdir(images_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    for image_file in image_files:

        image_path = os.path.join(images_path, image_file)

        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(labels_path, label_file)

        if not os.path.exists(label_path):
            continue

        with open(label_path, "r") as file:
            lines = file.readlines()

        if not lines:
            continue

        parts = lines[0].strip().split()

        if len(parts) != 5:
            continue

        try:
            class_id = int(parts[0])
        except ValueError:
            continue

        if not 0 <= class_id < NUM_CLASSES:
            continue

        image = cv2.imread(image_path)

        if image is None:
            continue

        # Use the full image (no bounding-box cropping)
        image = cv2.resize(
            image, (IMAGE_SIZE, IMAGE_SIZE)
        )

        image = cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB
        )

        # Keep pixel values in the 0–255 range.
        # Each model applies its own preprocessing.
        images.append(image)
        labels.append(class_id)

    if not images:
        raise ValueError(
            f"No usable images found in the {split} split."
        )

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    print(f"{split.upper()} images loaded: {len(images)}")

    return images, labels
