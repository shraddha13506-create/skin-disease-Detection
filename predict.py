import cv2
import numpy as np
import tensorflow as tf


# ==========================================
# 1. SETTINGS
# ==========================================

MODEL_PATH = "models/skin_disease_cnn.keras"

IMAGE_SIZE = 128

CLASS_NAMES = [
    "Acne-Vulgaris",
    "Eczema",
    "Tinea-Versicolor",
    "Psoriasis"
]


# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ==========================================
# 3. GIVE IMAGE PATH
# ==========================================

IMAGE_PATH = input("\nEnter image path: ")


# ==========================================
# 4. LOAD IMAGE
# ==========================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Error: Could not load image.")
    exit()


# ==========================================
# 5. PREPROCESS IMAGE
# ==========================================

image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

image = image.astype(np.float32) / 255.0

# Add batch dimension
image = np.expand_dims(image, axis=0)


# ==========================================
# 6. MAKE PREDICTION
# ==========================================

prediction = model.predict(image, verbose=0)

predicted_class = np.argmax(prediction[0])

confidence = prediction[0][predicted_class]


# ==========================================
# 7. DISPLAY RESULT
# ==========================================

print("\n========================================")
print("PREDICTION RESULT")
print("========================================")

print("Predicted disease:", CLASS_NAMES[predicted_class])
print(f"Confidence: {confidence * 100:.2f}%")

print("\nClass probabilities:")

for i in range(len(CLASS_NAMES)):
    print(
        f"{CLASS_NAMES[i]}: "
        f"{prediction[0][i] * 100:.2f}%"
    )

    # command for o/p: C:\Users\shrad\anaconda3\python.exe src/predict.py