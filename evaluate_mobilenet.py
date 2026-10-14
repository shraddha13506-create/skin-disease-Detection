
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================
# SETTINGS
# ==========================================

DATASET_PATH = "data/raw"
MODEL_PATH = "models/skin_disease_mobilenet_full.keras"
IMAGE_SIZE = 128

CLASS_NAMES = [
    "Acne-Vulgaris",
    "Eczema",
    "Tinea-Versicolor",
    "Psoriasis"
]

NUM_CLASSES = len(CLASS_NAMES)


# ==========================================
# LOAD TEST DATA
# ==========================================

def load_test_data():

    images_path = os.path.join(
        DATASET_PATH, "test", "images"
    )

    labels_path = os.path.join(
        DATASET_PATH, "test", "labels"
    )

    images = []
    labels = []

    image_files = sorted([
        f for f in os.listdir(images_path)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ])

    for image_file in image_files:

        image_path = os.path.join(
            images_path, image_file
        )

        label_file = (
            os.path.splitext(image_file)[0] + ".txt"
        )

        label_path = os.path.join(
            labels_path, label_file
        )

        if not os.path.exists(label_path):
            continue

        with open(label_path, "r") as file:
            lines = file.readlines()

        # Skip images with empty labels
        if not lines:
            continue

        parts = lines[0].strip().split()

        if len(parts) != 5:
            continue

        class_id = int(parts[0])

        image = cv2.imread(image_path)

        if image is None:
            continue

        # Use the FULL image, without cropping
        image = cv2.resize(
            image, (IMAGE_SIZE, IMAGE_SIZE)
        )

        image = cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB
        )

        images.append(image)
        labels.append(class_id)

    images = np.array(
        images, dtype=np.float32
    ) / 255.0

    labels = np.array(labels, dtype=np.int32)

    print("Test images loaded:", len(images))

    return images, labels


# ==========================================
# LOAD MODEL AND TEST DATA
# ==========================================

print("Loading MobileNetV2 model...")

model = tf.keras.models.load_model(MODEL_PATH)

x_test, y_true = load_test_data()

print("\nGenerating predictions...")

probabilities = model.predict(
    x_test, verbose=0
)

y_pred = np.argmax(probabilities, axis=1)


# ==========================================
# ACCURACY
# ==========================================

accuracy = accuracy_score(y_true, y_pred)

print("\n===================================")
print("CLASSIFICATION EVALUATION RESULTS")
print("===================================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")


# ==========================================
# PRECISION, RECALL AND F1-SCORE
# ==========================================

print("\nClassification Report:\n")

print(classification_report(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES)),
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
))


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES))
)

print("Confusion Matrix:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

display.plot(
    values_format="d",
    xticks_rotation=30
)

plt.title("MobileNetV2 - Confusion Matrix")
plt.tight_layout()

os.makedirs("results", exist_ok=True)

plt.savefig(
    "results/mobilenet_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nConfusion matrix saved to:")
print("results/mobilenet_confusion_matrix.png")
