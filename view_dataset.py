import os
import random
import cv2
import matplotlib.pyplot as plt

# Dataset location
DATASET_PATH = "data/raw"

# Class names
CLASS_NAMES = {
    0: "Acne-Vulgaris",
    1: "Eczema",
    2: "Tinea-Versicolor",
    3: "Psoriasis"
}


def show_images(split="train", num_images=8):

    images_path = os.path.join(DATASET_PATH, split, "images")
    labels_path = os.path.join(DATASET_PATH, split, "labels")

    # Get all image files
    image_files = [
        f for f in os.listdir(images_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # Randomly select images
    selected_images = random.sample(
        image_files,
        min(num_images, len(image_files))
    )

    for image_file in selected_images:

        # Image path
        image_path = os.path.join(images_path, image_file)

        # Corresponding label file
        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(labels_path, label_file)

        # Read image
        image = cv2.imread(image_path)

        # Convert BGR → RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Image dimensions
        height, width, _ = image.shape

        # Read labels
        if os.path.exists(label_path):

            with open(label_path, "r") as file:

                for line in file:

                    parts = line.strip().split()

                    if len(parts) != 5:
                        continue

                    class_id = int(parts[0])

                    # YOLO coordinates
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    box_width = float(parts[3])
                    box_height = float(parts[4])

                    # Convert to pixel coordinates
                    x_center *= width
                    y_center *= height
                    box_width *= width
                    box_height *= height

                    x1 = int(x_center - box_width / 2)
                    y1 = int(y_center - box_height / 2)
                    x2 = int(x_center + box_width / 2)
                    y2 = int(y_center + box_height / 2)

                    # Draw bounding box
                    cv2.rectangle(
                        image,
                        (x1, y1),
                        (x2, y2),
                        (255, 0, 0),
                        2
                    )

                    # Disease name
                    label = CLASS_NAMES[class_id]

                    cv2.putText(
                        image,
                        label,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2
                    )

        # Display image
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.axis("off")
        plt.title(image_file)
        plt.show()


# Display 8 random training images
show_images("train", 8)

# command: src/view_dataset.py