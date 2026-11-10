
import os
import time
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from data_loader import (
    load_dataset,
    IMAGE_SIZE,
    NUM_CLASSES,
    CLASS_NAMES
)


# ==========================================
# SETTINGS
# ==========================================

BATCH_SIZE = 16
EPOCHS = 25

MODEL_PATH = "models/skin_disease_resnet50v2.keras"

os.makedirs("models", exist_ok=True)


# ==========================================
# LOAD DATASET
# ==========================================

x_train, y_train = load_dataset("train")
x_valid, y_valid = load_dataset("valid")
x_test, y_test = load_dataset("test")

print("\nClass order:", CLASS_NAMES)
print("Training shape:", x_train.shape)
print("Validation shape:", x_valid.shape)
print("Testing shape:", x_test.shape)


# ==========================================
# LOAD PRETRAINED RESNET50V2
# ==========================================

base_model = tf.keras.applications.ResNet50V2(
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False


# ==========================================
# DATA AUGMENTATION
# ==========================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])


# ==========================================
# BUILD MODEL
# ==========================================

inputs = layers.Input(
    shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
)

x = data_augmentation(inputs)

# ResNet50V2 preprocessing: convert 0–255 to -1–1
x = layers.Rescaling(
    scale=1.0 / 127.5,
    offset=-1
)(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.4)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = models.Model(
    inputs=inputs,
    outputs=outputs
)


# ==========================================
# COMPILE
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ==========================================
# CALLBACKS
# ==========================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)


# ==========================================
# TRAIN
# ==========================================

print("\nStarting ResNet50V2 training...")

start_time = time.perf_counter()

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_valid, y_valid),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

training_time = time.perf_counter() - start_time


# ==========================================
# TEST
# ==========================================

best_model = tf.keras.models.load_model(MODEL_PATH)

test_loss, test_accuracy = best_model.evaluate(
    x_test,
    y_test,
    verbose=1
)

print("\n===================================")
print("RESNET50V2 FINAL RESULTS")
print("===================================")

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Training Time: {training_time:.2f} seconds")
print(f"Model saved: {MODEL_PATH}")
