import os
from PIL import Image

DATASET_PATH = "data/raw"

CLASS_NAMES = {
    0: "Acne-Vulgaris",
    1: "Eczema",
    2: "Tinea-Versicolor",
    3: "Psoriasis"
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def check_split(split_name):

    images_path = os.path.join(DATASET_PATH, split_name, "images")
    labels_path = os.path.join(DATASET_PATH, split_name, "labels")

    image_files = [
        f for f in os.listdir(images_path)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ]

    label_files = [
        f for f in os.listdir(labels_path)
        if f.endswith(".txt")
    ]

    # Count images belonging to each class
    class_image_counts = {
        0: 0,
        1: 0,
        2: 0,
        3: 0
    }

    images_with_multiple_classes = 0

    for image_file in image_files:

        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(labels_path, label_file)

        if not os.path.exists(label_path):
            continue

        classes_in_image = set()

        with open(label_path, "r") as file:
            for line in file:

                parts = line.strip().split()

                if parts:
                    class_id = int(parts[0])
                    classes_in_image.add(class_id)

        # Count each disease once per image
        for class_id in classes_in_image:
            if class_id in class_image_counts:
                class_image_counts[class_id] += 1

        if len(classes_in_image) > 1:
            images_with_multiple_classes += 1

    # Check matching files
    image_names = {
        os.path.splitext(f)[0]
        for f in image_files
    }

    label_names = {
        os.path.splitext(f)[0]
        for f in label_files
    }

    missing_labels = image_names - label_names
    extra_labels = label_names - image_names

    # Check corrupted images
    corrupted_images = []

    for image_file in image_files:

        image_path = os.path.join(images_path, image_file)

        try:
            with Image.open(image_path) as img:
                img.verify()

        except Exception:
            corrupted_images.append(image_file)

    # Print results
    print("\n" + "=" * 50)
    print(split_name.upper())
    print("=" * 50)

    print(f"Total images: {len(image_files)}")

    print("\nImages by disease:")

    for class_id, count in class_image_counts.items():
        print(f"{CLASS_NAMES[class_id]}: {count}")

    print(f"\nImages containing multiple classes: {images_with_multiple_classes}")

    print("\nDataset health:")
    print(f"Missing labels: {len(missing_labels)}")
    print(f"Labels without images: {len(extra_labels)}")
    print(f"Corrupted images: {len(corrupted_images)}")


# Check all three folders
for split in ["train", "valid", "test"]:
    check_split(split)

print("\n" + "=" * 50)
print("VERIFICATION COMPLETE")
print("=" * 50)

# command: python src/verify_dataset.py