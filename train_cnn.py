import os
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ==========================================
# 1. SETTINGS
# ==========================================

DATASET_PATH = "data/raw"

IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 50

CLASS_NAMES = [
    "Acne-Vulgaris",
    "Eczema",
    "Tinea-Versicolor",
    "Psoriasis"
]

NUM_CLASSES = len(CLASS_NAMES)


# ==========================================
# 2. LOAD IMAGES USING YOLO LABELS
# ==========================================

def load_dataset(split):

    images_path = os.path.join(DATASET_PATH, split, "images")
    labels_path = os.path.join(DATASET_PATH, split, "labels")

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

        # Read image
        image = cv2.imread(image_path)

        if image is None:
            continue

        image_height, image_width = image.shape[:2]

        # Read YOLO label
        with open(label_path, "r") as file:
            lines = file.readlines()

        if len(lines) == 0:
            continue

        # Get class from first annotation
        first_parts = lines[0].strip().split()

        if len(first_parts) != 5:
            continue

        class_id = int(first_parts[0])

        # Find bounding box covering all annotations
        x1_all = image_width
        y1_all = image_height
        x2_all = 0
        y2_all = 0

        for line in lines:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            _, x_center, y_center, box_width, box_height = map(
                float, parts
            )

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

        # Keep coordinates inside image
        x1_all = max(0, x1_all)
        y1_all = max(0, y1_all)
        x2_all = min(image_width, x2_all)
        y2_all = min(image_height, y2_all)

        # Crop affected region
        cropped = image[y1_all:y2_all, x1_all:x2_all]

        if cropped.size == 0:
            continue

        # Resize
        cropped = cv2.resize(
            cropped,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        # Convert BGR → RGB
        cropped = cv2.cvtColor(
            cropped,
            cv2.COLOR_BGR2RGB
        )

        images.append(cropped)
        labels.append(class_id)

    # Convert to NumPy arrays
    images = np.array(images, dtype=np.float32) / 255.0
    labels = np.array(labels)

    print(f"{split.upper()} images loaded: {len(images)}")

    return images, labels


# ==========================================
# 3. LOAD TRAIN / VALIDATION / TEST DATA
# ==========================================

x_train, y_train = load_dataset("train")
x_valid, y_valid = load_dataset("valid")
x_test, y_test = load_dataset("test")


print("\nDataset shapes:")
print("Training:", x_train.shape)
print("Validation:", x_valid.shape)
print("Testing:", x_test.shape)


# ==========================================
# 4. BUILD CNN MODEL
# ==========================================

model = models.Sequential([

    # Input
    layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),

    # Data augmentation
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),

    # CNN Block 1
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # CNN Block 2
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # CNN Block 3
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Flatten
    layers.Flatten(),

    # Fully connected layer
    layers.Dense(128, activation="relu"),

    # Dropout
    layers.Dropout(0.5),

    # Output layer
    layers.Dense(NUM_CLASSES, activation="softmax")
])


# ==========================================
# 5. COMPILE MODEL
# ==========================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# Display architecture
model.summary()


# ==========================================
# 6. CALLBACKS
# ==========================================

os.makedirs("models", exist_ok=True)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "models/skin_disease_cnn.keras",
    monitor="val_accuracy",
    save_best_only=True
)


# ==========================================
# 7. TRAIN MODEL
# ==========================================

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_valid, y_valid),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        early_stopping,
        checkpoint
    ]
)


# ==========================================
# 8. TEST MODEL
# ==========================================

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=1
)

print("\n========================================")
print("FINAL TEST RESULTS")
print("========================================")

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

print("\nCNN training complete!")
print("Model saved to:")
print("models/skin_disease_cnn.keras")

#use command : C:\Users\shrad\anaconda3\python.exe src/train_cnn.py