
import os
import time
from pathlib import Path


import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ==========================================
# PATHS AND SETTINGS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "skin_disease_mobilenet_full.keras"
)

# Test dataset
TEST_IMAGES_PATH = PROJECT_DIR / "data" / "raw" / "test" / "images"
TEST_LABELS_PATH = PROJECT_DIR / "data" / "raw" / "test" / "labels"

# Validation dataset
VALID_IMAGES_PATH = PROJECT_DIR / "data" / "raw" / "valid" / "images"
VALID_LABELS_PATH = PROJECT_DIR / "data" / "raw" / "valid" / "labels"

IMAGE_SIZE = 128

CLASS_NAMES = [
    "Psoriasis",
    "Eczema",
    "Tinea-Versicolor",
    "Acne-Vulgaris"
]

NUM_CLASSES = len(CLASS_NAMES)


# ==========================================
# PAGE DESIGN
# ==========================================

st.set_page_config(
    page_title="Skin Disease Detection",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Skin Disease Detection")

st.write(
    "Upload a skin image to obtain a prediction "
    "from our trained MobileNetV2 model."
)

st.info(
    "This is a student project for image classification, "
    "not a medical diagnosis."
)

st.markdown("""
<style>

/* All buttons, including CSV download */
.stButton > button,
.stDownloadButton > button,
[data-testid="stDownloadButton"] button {
    background-color: #198754 !important;
    color: #FFFFFF !important;
    border: 1px solid #198754 !important;
    border-radius: 8px !important;
    padding: 0.55rem 1rem !important;
    font-weight: 600 !important;
}

/* Hover effect */
.stButton > button:hover,
.stDownloadButton > button:hover,
[data-testid="stDownloadButton"] button:hover {
    background-color: #146C43 !important;
    color: #FFFFFF !important;
    border-color: #146C43 !important;
}

/* Keyboard focus */
.stButton > button:focus,
.stDownloadButton > button:focus {
    outline: 2px solid #8BD3A9 !important;
    outline-offset: 2px !important;
}

/* Light table background */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    background-color: #FFFFFF !important;
    border: 1px solid #D7E2EE !important;
    border-radius: 8px !important;
    overflow: hidden;
}

/* Table header */
[data-testid="stDataFrame"] th,
[data-testid="stTable"] th {
    background-color: #EAF2F8 !important;
    color: #17365D !important;
}

/* Table cells */
[data-testid="stDataFrame"] td,
[data-testid="stTable"] td {
    color: #263746 !important;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(MODEL_PATH)


# ==========================================
# PREPROCESS IMAGE
# ==========================================

def preprocess_image(image):

    image = image.convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    image_array = np.array(
        image, dtype=np.float32
    ) / 255.0

    image_array = np.expand_dims(
        image_array, axis=0
    )

    return image_array


# ==========================================
# LOAD TEST DATA
# ==========================================

@st.cache_data

def load_test_data():

    images = []
    labels = []

    # Evaluate both validation and test datasets
    dataset_folders = [
        (VALID_IMAGES_PATH, VALID_LABELS_PATH),
        (TEST_IMAGES_PATH, TEST_LABELS_PATH)
    ]

    for images_path, labels_path in dataset_folders:

        image_files = sorted([
            f for f in os.listdir(images_path)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ])

        for image_file in image_files:

            image_path = images_path / image_file

            label_file = Path(image_file).stem + ".txt"
            label_path = labels_path / label_file

            if not label_path.exists():
                continue

            with open(label_path, "r") as file:
                lines = file.readlines()

            # Skip images with empty labels
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

            image = cv2.imread(str(image_path))

            if image is None:
                continue

            # Resize the full image; do not crop bounding boxes
            image = cv2.resize(
                image, (IMAGE_SIZE, IMAGE_SIZE)
            )

            image = cv2.cvtColor(
                image, cv2.COLOR_BGR2RGB
            )

            images.append(image)
            labels.append(class_id)

    if len(images) == 0:
        raise ValueError(
            "No usable labelled validation or test images were found."
        )

    images = np.array(
        images, dtype=np.float32
    ) / 255.0

    labels = np.array(
        labels, dtype=np.int32
    )

    return images, labels


# ==========================================
# EVALUATE MODEL
# ==========================================

@st.cache_data
def calculate_metrics():

    model = load_model()

    x_test, y_true = load_test_data()

    probabilities = model.predict(
        x_test, verbose=0
    )

    y_pred = np.argmax(
        probabilities, axis=1
    )

    accuracy = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES))
    )

    return accuracy, report, matrix, len(y_true)


# ==========================================
# SESSION STORAGE
# ==========================================

if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# ==========================================
# THREE MAIN SECTIONS
# ==========================================

prediction_tab, history_tab, evaluation_tab, comparison_tab, safety_tab = st.tabs([
    "🔍 Disease Prediction",
    "🕘 Prediction History",
    "📊 Model Evaluation",
    "⚖️ CNN vs MobileNetV2",
    "🛡️ Safety & Limitations"
])


# ==========================================
# PREDICTION WITH UNCERTAINTY WARNING
# ==========================================

# ==========================================
# TAB 1: PREDICTION + UPLOAD CONTROLS
# ==========================================

with prediction_tab:

    st.subheader("Upload a skin image")
    st.write("Choose a clear JPG, JPEG, or PNG image.")

    upload_col, clear_col = st.columns([4, 1])

    with upload_col:
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            key=f"skin_upload_{st.session_state.upload_key}"
        )

    with clear_col:
        st.write("")
        st.write("")
        if st.button("Clear image"):
            st.session_state.upload_key += 1
            st.rerun()

    if uploaded_file is not None:

        st.caption(f"Selected file: {uploaded_file.name}")

        try:
            image = Image.open(uploaded_file).convert("RGB")

            st.subheader("Uploaded Image")
            st.image(image, width=350)

            if st.button("Predict disease class"):
                model = load_model()
                processed_image = preprocess_image(image)

                probabilities = model.predict(
                    processed_image, verbose=0
                )[0]

                predicted_class = int(np.argmax(probabilities))
                confidence = float(probabilities[predicted_class])
                predicted_name = CLASS_NAMES[predicted_class]

                # Save this prediction in the current session
                timestamp = pd.Timestamp.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                st.session_state.prediction_history.insert(0, {
                    "Timestamp": timestamp,
                    "Filename": uploaded_file.name,
                    "Predicted class": predicted_name,
                    "Confidence (%)": round(confidence * 100, 2)
                })

                # Keep only the latest 100 predictions
                st.session_state.prediction_history = (
                    st.session_state.prediction_history[:100]
                )

                st.subheader("Prediction")

                if confidence < 0.70:
                    st.warning(
                        "The model has a low confidence score. "
                        "Try a clearer image. This does not reliably "
                        "detect conditions outside the trained classes."
                    )
                else:
                    st.success(
                        f"Predicted class: {predicted_name}"
                    )

                st.metric(
                    "Model confidence score",
                    f"{confidence * 100:.2f}%"
                )

                st.caption(
                    "This score is not the probability of a medical diagnosis. "
                    "The model can still make incorrect predictions."
                )

                st.subheader("Class Scores")

                for i, class_name in enumerate(CLASS_NAMES):
                    score = float(probabilities[i])
                    st.write(
                        f"{class_name}: {score * 100:.2f}%"
                    )
                    st.progress(float(np.clip(score, 0, 1)))

        except Exception as error:
            st.error(
                f"Could not process this image. "
                f"Please choose a valid image. Details: {error}"
            )


# ==========================================
# TAB 2: PREDICTION HISTORY
# ==========================================

with history_tab:

    st.subheader("Prediction History")

    st.write(
        "Your predictions are stored temporarily in this session. "
        "They are not automatically saved permanently."
    )

    history = st.session_state.prediction_history

    if history:
        history_df = pd.DataFrame(history)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download History as CSV",
            data=csv_data,
            file_name="skin_prediction_history.csv",
            mime="text/csv"
        )

        if st.button("Clear Prediction History"):
            st.session_state.prediction_history = []
            st.rerun()

    else:
        st.info(
            "No predictions yet. Go to Disease Prediction "
            "and classify an image to see it here."
        )


# ==========================================
# TAB 2: MODEL EVALUATION
# ==========================================

with evaluation_tab:

    st.subheader("Test Dataset Evaluation")

    st.write(
        "These metrics evaluate the model on the labelled "
        "test dataset, not just the uploaded image."
    )

    st.caption(
        "Evaluation uses full test images resized to 128×128, "
        "matching the full-image training approach."
    )

    try:

        accuracy, report, matrix, test_count = calculate_metrics()

        # Overall metrics
        macro_precision = report["macro avg"]["precision"]
        macro_recall = report["macro avg"]["recall"]
        macro_f1 = report["macro avg"]["f1-score"]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Accuracy",
            f"{accuracy * 100:.2f}%"
        )

        col2.metric(
            "Macro Precision",
            f"{macro_precision * 100:.2f}%"
        )

        col3.metric(
            "Macro Recall",
            f"{macro_recall * 100:.2f}%"
        )

        col4.metric(
            "Macro F1-score",
            f"{macro_f1 * 100:.2f}%"
        )

        st.caption(
            f"Number of evaluated test images: {test_count}"
        )

        # Per-class classification report
        st.subheader("Per-Class Classification Report")

        report_df = pd.DataFrame(
            {
                class_name: {
                    "Precision": report[class_name]["precision"],
                    "Recall": report[class_name]["recall"],
                    "F1-score": report[class_name]["f1-score"],
                    "Support": report[class_name]["support"]
                }
                for class_name in CLASS_NAMES
            }
        ).T

        report_df.index.name = "Disease Class"

        st.dataframe(
            report_df.style.format({
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "F1-score": "{:.3f}",
                "Support": "{:.0f}"
            }),
            use_container_width=True
        )


# Confusion matrix
        st.subheader("Confusion Matrix")

        st.write(
            "Rows represent actual classes. "
            "Columns represent predicted classes. "
            "Diagonal values are correct predictions."
        )

        fig, ax = plt.subplots(figsize=(9, 6))

        display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=CLASS_NAMES
)

        display.plot(
            ax=ax,
            values_format="d",
            xticks_rotation=35,
            colorbar=False
        )

        ax.set_title(
            f"MobileNetV2 Confusion Matrix - Test Set (n={test_count})"
        )

        fig.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


        # Raw counts
        with st.expander("View confusion matrix values"):

            matrix_df = pd.DataFrame(
                matrix,
                index=CLASS_NAMES,
                columns=CLASS_NAMES
            )

            matrix_df.index.name = "Actual / Predicted"

            st.dataframe(
                matrix_df,
                use_container_width=True
            )

    except Exception as error:

        st.error(
            f"Could not evaluate the model: {error}"
        )

        st.write(
            "Check that the full-image model has been trained "
            "and that the test images and label files exist."
        )



# ==========================================
# TAB 4: CNN VS MOBILENETV2 COMPARISON
# ==========================================

with comparison_tab:

    st.subheader("CNN vs MobileNetV2")
    st.write(
        "Compare the performance of your two trained models "
        "on the same labelled test images."
    )

    st.warning(
        "Before interpreting results, confirm that both models "
        "were trained with the same class-ID order and compatible "
        "image preprocessing."
    )

    cnn_path = PROJECT_DIR / "models" / "skin_disease_cnn.keras"
    mobile_path = PROJECT_DIR / "models" / "skin_disease_mobilenet_full.keras"

    if not cnn_path.exists() or not mobile_path.exists():
        st.error(
            "One or both model files are missing. Check your models folder."
        )

    else:

        if st.button("Run Model Comparison"):

            try:
                with st.spinner(
                    "Evaluating both models on the test dataset..."
                ):

                    # Use the same test images and labels for both models
                    x_test, y_true = load_test_data()

                    results = []

                    for model_name, model_path in [
                        ("CNN", cnn_path),
                        ("MobileNetV2", mobile_path)
                    ]:

                        model = tf.keras.models.load_model(model_path)

                        # Check expected input dimensions
                        input_shape = model.input_shape

                        if tuple(input_shape[1:]) != (
                            IMAGE_SIZE, IMAGE_SIZE, 3
                        ):
                            raise ValueError(
                                f"{model_name} expects input shape "
                                f"{input_shape}. The app supplies "
                                f"(None, {IMAGE_SIZE}, {IMAGE_SIZE}, 3)."
                            )

                        # Check output class count
                        output_shape = model.output_shape

                        if output_shape[-1] != NUM_CLASSES:
                            raise ValueError(
                                f"{model_name} outputs "
                                f"{output_shape[-1]} classes, but the app "
                                f"expects {NUM_CLASSES}."
                            )

                        start_time = time.perf_counter()

                        probabilities = model.predict(
                            x_test, verbose=0
                        )

                        elapsed = time.perf_counter() - start_time

                        predictions = np.argmax(
                            probabilities, axis=1
                        )

                        report = classification_report(
                            y_true,
                            predictions,
                            labels=list(range(NUM_CLASSES)),
                            target_names=CLASS_NAMES,
                            output_dict=True,
                            zero_division=0
                        )

                        results.append({
                            "Model": model_name,
                            "Accuracy (%)": accuracy_score(
                                y_true, predictions
                            ) * 100,
                            "Macro Precision (%)":
                                report["macro avg"]["precision"] * 100,
                            "Macro Recall (%)":
                                report["macro avg"]["recall"] * 100,
                            "Macro F1-score (%)":
                                report["macro avg"]["f1-score"] * 100,
                            "Model Size (MB)":
                                model_path.stat().st_size / (1024 * 1024),
                            "Test Prediction Time (s)": elapsed
                        })

                        del model

                comparison_df = pd.DataFrame(results)

                st.session_state["comparison_results"] = comparison_df

                st.success("Comparison completed!")

            except Exception as error:
                st.error(f"Comparison failed: {error}")

        # Display saved results after the button is clicked
        if "comparison_results" in st.session_state:

            comparison_df = st.session_state["comparison_results"]

            st.subheader("Comparison Results")

            st.dataframe(
                comparison_df.style.format({
                    "Accuracy (%)": "{:.2f}",
                    "Macro Precision (%)": "{:.2f}",
                    "Macro Recall (%)": "{:.2f}",
                    "Macro F1-score (%)": "{:.2f}",
                    "Model Size (MB)": "{:.2f}",
                    "Test Prediction Time (s)": "{:.3f}"
                }),
                use_container_width=True,
                hide_index=True
            )

            st.subheader("Accuracy Comparison")

            chart_df = comparison_df.set_index("Model")[
                ["Accuracy (%)"]
            ]

            st.bar_chart(chart_df)

            st.caption(
                "Inference time is for one batched prediction over the test set "
                "on your computer. It depends on hardware and model warm-up. "
                "File size is the saved model file size."
            )

            st.info(
                "Higher accuracy is not the only consideration. Also compare "
                "macro F1-score, model size, inference time, and per-class results."
            )



# ==========================================
# TAB 5: SAFETY AND LIMITATIONS
# ==========================================

with safety_tab:

    st.subheader("Safety and Limitations")

    st.warning(
        "This application is an educational image-classification project, "
        "not a medical diagnostic tool."
    )

    st.markdown("""
### Supported classes

This model is designed to classify images into these four classes:

- Psoriasis
- Eczema
- Tinea-Versicolor
- Acne-Vulgaris

It has not been designed to recognize every skin condition.

### Model limitations

- The model may misclassify an image.
- A high confidence score does not guarantee a correct prediction.
- The model may perform poorly on blurry, poorly lit, or unfamiliar images.
- Other skin conditions and normal skin may be incorrectly assigned to one of the four trained classes.
- Performance on a small test dataset may not reflect performance on images from new people or different cameras.
- The displayed class names must match the numeric class IDs used during training.

### Understanding the confidence score

The confidence score is the model's output for its selected class. It is not the probability that a medical diagnosis is correct.

The 70% threshold used by the application is experimental. It does not reliably identify unfamiliar conditions or guarantee that high-scoring predictions are correct.

### Responsible use

- Do not use the prediction to diagnose or treat a condition.
- Do not delay professional care because of a model prediction.
- Consult a qualified healthcare professional if you have a skin concern.
- Avoid uploading images containing identifying information.

### Data and privacy

Prediction history is stored temporarily in the current Streamlit session. It can be downloaded as a CSV file, but it is not automatically saved permanently.

The application displays the uploaded image for prediction and does not need to save that image to the project folder.
""")

    st.success(
        "Use this model for learning, experimentation, and "
        "demonstrating image classification—not for medical decisions."
    )



#C:\Users\shrad\anaconda3\python.exe -m streamlit run frontend/app.py
