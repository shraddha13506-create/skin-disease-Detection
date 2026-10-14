import os
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ==========================================
# SETTINGS
# ==========================================

DATASET_PATH = "data/raw"

IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 30

CLASS_NAMES = [
    "Psoriasis",        # Class ID 0
    "Eczema",           # Class ID 1
    "Tinea-Versicolor", # Class ID 2
    "Acne-Vulgaris"     # Class ID 3
]

NUM_CLASSES = len(CLASS_NAMES)


# ==========================================
# LOAD DATASET
# ==========================================

def load_dataset(split):

    images_path = os.path.join(
        DATASET_PATH,
        split,
        "images"
    )

    labels_path = os.path.join(
        DATASET_PATH,
        split,
        "labels"
    )

    images = []
    labels = []

    image_files = [
        f for f in os.listdir(images_path)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    for image_file in image_files:

        image_path = os.path.join(
            images_path,
            image_file
        )

        label_file = os.path.splitext(
            image_file
        )[0] + ".txt"

        label_path = os.path.join(
            labels_path,
            label_file
        )

        # Skip if label file doesn't exist
        if not os.path.exists(label_path):
            continue

        # Read image
        image = cv2.imread(image_path)

        if image is None:
            continue

        # Read label file
        with open(label_path, "r") as file:
            lines = file.readlines()

        # Skip empty label files
        if len(lines) == 0:
            continue

        # Get class ID from first label
        first_parts = lines[0].strip().split()

        if len(first_parts) != 5:
            continue

        class_id = int(first_parts[0])

        # ==========================================
        # IMPORTANT:
        # NO CROPPING
        # We use the FULL image.
        # ==========================================

        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        # Convert BGR → RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        images.append(image)
        labels.append(class_id)

    # Convert to NumPy arrays
    images = np.array(
        images,
        dtype=np.float32
    ) / 255.0

    labels = np.array(labels)

    print(
        f"{split.upper()} images loaded: "
        f"{len(images)}"
    )

    return images, labels


# ==========================================
# LOAD TRAIN / VALIDATION / TEST DATA
# ==========================================

x_train, y_train = load_dataset("train")

x_valid, y_valid = load_dataset("valid")

x_test, y_test = load_dataset("test")


print("\nDataset shapes:")

print("Training:", x_train.shape)

print("Validation:", x_valid.shape)

print("Testing:", x_test.shape)


# ==========================================
# MOBILE NET V2
# ==========================================

base_model = tf.keras.applications.MobileNetV2(

    input_shape=(
        IMAGE_SIZE,
        IMAGE_SIZE,
        3
    ),

    include_top=False,

    weights="imagenet"
)


# Freeze pretrained layers
base_model.trainable = False


# ==========================================
# DATA AUGMENTATION
# ==========================================

data_augmentation = tf.keras.Sequential([

    layers.RandomFlip(
        "horizontal"
    ),

    layers.RandomRotation(
        0.1
    ),

    layers.RandomZoom(
        0.1
    )

])


# ==========================================
# BUILD MODEL
# ==========================================

model = models.Sequential([

    layers.Input(
        shape=(
            IMAGE_SIZE,
            IMAGE_SIZE,
            3
        )
    ),

    # Data augmentation
    data_augmentation,

    # Pretrained MobileNetV2
    base_model,

    # Convert feature maps to vector
    layers.GlobalAveragePooling2D(),

    # Classification layer
    layers.Dense(
        128,
        activation="relu"
    ),

    # Reduce overfitting
    layers.Dropout(0.4),

    # Final 4 classes
    layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )

])


# ==========================================
# COMPILE
# ==========================================

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)


# Show model
model.summary()


# ==========================================
# CREATE MODELS FOLDER
# ==========================================

os.makedirs(
    "models",
    exist_ok=True
)


# ==========================================
# EARLY STOPPING
# ==========================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True

)


# ==========================================
# SAVE BEST MODEL
# ==========================================

checkpoint = ModelCheckpoint(

    "models/skin_disease_mobilenet_full.keras",

    monitor="val_accuracy",

    save_best_only=True

)


# ==========================================
# TRAIN
# ==========================================

history = model.fit(

    x_train,

    y_train,

    validation_data=(
        x_valid,
        y_valid
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        early_stopping,
        checkpoint
    ]

)


# ==========================================
# TEST MODEL
# ==========================================

test_loss, test_accuracy = model.evaluate(

    x_test,

    y_test,

    verbose=1

)


# ==========================================
# FINAL RESULTS
# ==========================================

print("\n========================================")

print(
    "MOBILENETV2 FULL-IMAGE FINAL RESULTS"
)

print("========================================")

print(
    f"Test Loss: {test_loss:.4f}"
)

print(
    f"Test Accuracy: {test_accuracy:.4f}"
)

print(
    f"Test Accuracy: {test_accuracy * 100:.2f}%"
)

print("\nMobileNetV2 training complete!")

print("Model saved to:")

print(
    "models/skin_disease_mobilenet_full.keras"
)

#command : C:\Users\shrad\anaconda3\python.exe src/train_mobilenet.py