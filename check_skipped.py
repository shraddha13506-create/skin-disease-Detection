import os
import cv2


DATASET_PATH = "data/raw"

SPLITS = ["train", "valid", "test"]


for split in SPLITS:

    print("\n" + "=" * 60)
    print(split.upper())
    print("=" * 60)

    images_path = os.path.join(DATASET_PATH, split, "images")
    labels_path = os.path.join(DATASET_PATH, split, "labels")

    image_files = [
        f for f in os.listdir(images_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    skipped = 0

    for image_file in image_files:

        image_path = os.path.join(images_path, image_file)

        # ----------------------------------
        # Check image
        # ----------------------------------

        image = cv2.imread(image_path)

        if image is None:
            print(f"SKIPPED: {image_file}")
            print("Reason: Image could not be read")
            skipped += 1
            continue

        image_height, image_width = image.shape[:2]

        # ----------------------------------
        # Check label file
        # ----------------------------------

        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(labels_path, label_file)

        if not os.path.exists(label_path):
            print(f"SKIPPED: {image_file}")
            print("Reason: Label file missing")
            skipped += 1
            continue

        # ----------------------------------
        # Read labels
        # ----------------------------------

        with open(label_path, "r") as file:
            lines = file.readlines()

        if len(lines) == 0:
            print(f"SKIPPED: {image_file}")
            print("Reason: Label file is empty")
            skipped += 1
            continue

        # ----------------------------------
        # Check label format
        # ----------------------------------

        valid_label = True

        for line_number, line in enumerate(lines, start=1):

            parts = line.strip().split()

            if len(parts) != 5:

                print(f"SKIPPED: {image_file}")
                print(
                    f"Reason: Invalid label format on line {line_number}"
                )
                print(f"Label: {line.strip()}")

                valid_label = False
                break

        if not valid_label:
            skipped += 1
            continue

        # ----------------------------------
        # Check bounding boxes
        # ----------------------------------

        x1_all = image_width
        y1_all = image_height
        x2_all = 0
        y2_all = 0

        valid_boxes = True

        for line in lines:

            parts = line.strip().split()

            try:

                class_id, x_center, y_center, box_width, box_height = map(
                    float, parts
                )

            except ValueError:

                print(f"SKIPPED: {image_file}")
                print("Reason: Non-numeric value in label")
                valid_boxes = False
                break

            # Convert YOLO coordinates to pixels

            x_center *= image_width
            y_center *= image_height
            box_width *= image_width
            box_height *= image_height

            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)

            x1_all = min(x1_all, x1)
            y1_all = min(y1_all, y1)
            x2_all = max(x2_all, x2)
            y2_all = max(y2_all, y2)

        if not valid_boxes:
            skipped += 1
            continue

        # ----------------------------------
        # Check final crop
        # ----------------------------------

        x1_all = max(0, x1_all)
        y1_all = max(0, y1_all)
        x2_all = min(image_width, x2_all)
        y2_all = min(image_height, y2_all)

        cropped = image[y1_all:y2_all, x1_all:x2_all]

        if cropped.size == 0:

            print(f"SKIPPED: {image_file}")
            print("Reason: Bounding box produced an empty crop")

            skipped += 1
            continue

    print(f"\nTotal skipped in {split}: {skipped}")

    # command: C:\Users\shrad\anaconda3\python.exe src/check_skipped.py