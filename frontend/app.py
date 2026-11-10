
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
# 1. PATHS AND SETTINGS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATHS = {
    "CNN": PROJECT_DIR / "models" / "skin_disease_cnn.keras",
    "MobileNetV2": PROJECT_DIR / "models" / "skin_disease_mobilenet_full.keras",
    "ResNet50V2": PROJECT_DIR / "models" / "skin_disease_resnet50v2.keras",
    "EfficientNetB0": PROJECT_DIR / "models" / "skin_disease_efficientnetb0.keras"
}

IMAGE_SIZE = 128

CLASS_NAMES = [
    "Psoriasis",
    "Eczema",
    "Tinea-Versicolor",
    "Acne-Vulgaris"
]

NUM_CLASSES = len(CLASS_NAMES)

TEST_IMAGES_PATH = PROJECT_DIR / "data" / "raw" / "test" / "images"
TEST_LABELS_PATH = PROJECT_DIR / "data" / "raw" / "test" / "labels"


# ==========================================
# 2. PAGE DESIGN
# ==========================================

st.set_page_config(
    page_title="Skin Disease Detection",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Skin Disease Detection")

st.write(
    "An image classification application comparing "
    "four deep learning models for skin disease identification."
)

st.info(
    "This is an educational project for image classification, "
    "not a medical diagnosis."
)

st.markdown("""
<style>
.stButton > button,
.stDownloadButton > button {
    background-color: #198754 !important;
    color: white !important;
    border: 1px solid #198754 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background-color: #146C43 !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid #D7E2EE !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. MODEL LOADING
# ==========================================

@st.cache_resource
def load_model(model_name):

    model_path = MODEL_PATHS[model_name]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    return tf.keras.models.load_model(model_path)


# ==========================================
# 4. IMAGE PREPROCESSING
# ==========================================

def preprocess_image(image, model_name):

    image = image.convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    image_array = np.array(
        image,
        dtype=np.float32
    )

    # CNN and MobileNetV2 were trained using 0–1 inputs.
    # ResNet50V2 and EfficientNetB0 use 0–255 inputs
    # with preprocessing included inside their models.

    if model_name in ["CNN", "MobileNetV2"]:
        image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ==========================================
# 5. LOAD TEST DATASET
# ==========================================

@st.cache_data
def load_test_data():

    images = []
    labels = []

    image_files = sorted([
        f for f in os.listdir(TEST_IMAGES_PATH)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ])

    for image_file in image_files:

        image_path = TEST_IMAGES_PATH / image_file

        label_path = (
            TEST_LABELS_PATH /
            (Path(image_file).stem + ".txt")
        )

        if not label_path.exists():
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

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        images.append(image)
        labels.append(class_id)

    if not images:
        raise ValueError(
            "No usable labelled test images found."
        )

    return (
        np.array(images, dtype=np.float32),
        np.array(labels, dtype=np.int32)
    )


# ==========================================
# 6. EVALUATION FUNCTION
# ==========================================

def evaluate_model(model_name, x_test, y_true):

    model = load_model(model_name)

    # Apply preprocessing appropriate to each model.
    if model_name in ["CNN", "MobileNetV2"]:
        model_images = x_test / 255.0
    else:
        model_images = x_test

    start_time = time.perf_counter()

    probabilities = model.predict(
        model_images,
        batch_size=16,
        verbose=0
    )

    prediction_time = time.perf_counter() - start_time

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    report = classification_report(
        y_true,
        predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=list(range(NUM_CLASSES))
    )

    model_size = (
        MODEL_PATHS[model_name].stat().st_size /
        (1024 * 1024)
    )

    results = {
        "Model": model_name,
        "Accuracy (%)": accuracy * 100,
        "Precision (%)": report["macro avg"]["precision"] * 100,
        "Recall (%)": report["macro avg"]["recall"] * 100,
        "F1-score (%)": report["macro avg"]["f1-score"] * 100,
        "Model Size (MB)": model_size,
        "Prediction Time (s)": prediction_time
    }

    return results, report, matrix, predictions


# ==========================================
# 7. SESSION STORAGE
# ==========================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "comparison_results" not in st.session_state:
    st.session_state.comparison_results = None


# ==========================================
# 8. MAIN TABS
# ==========================================

prediction_tab, history_tab, evaluation_tab, comparison_tab, safety_tab = st.tabs([
    "🔍 Disease Prediction",
    "🕘 Prediction History",
    "📊 Model Evaluation",
    "⚖️ Four-Model Comparison",
    "🛡️ Safety & Limitations"
])


# ==========================================
# TAB 1: DISEASE PREDICTION
# ==========================================

with prediction_tab:

    st.subheader("Upload a Skin Image")

    selected_model = st.selectbox(
        "Choose a Deep Learning Model",
        list(MODEL_PATHS.keys())
    )

    uploaded_file = st.file_uploader(
        "Upload a JPG, JPEG or PNG image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        try:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            st.image(
                image,
                caption="Uploaded Skin Image",
                width=350
            )

            if st.button("Predict Disease Class"):

                with st.spinner("Analyzing image..."):

                    model = load_model(selected_model)

                    processed_image = preprocess_image(
                        image,
                        selected_model
                    )

                    start_time = time.perf_counter()

                    probabilities = model.predict(
                        processed_image,
                        verbose=0
                    )[0]

                    prediction_time = (
                        time.perf_counter() - start_time
                    )

                    predicted_class = int(
                        np.argmax(probabilities)
                    )

                    confidence = float(
                        probabilities[predicted_class]
                    )

                    predicted_name = CLASS_NAMES[
                        predicted_class
                    ]

                st.subheader("Prediction Result")

                st.write(f"**Selected Model:** {selected_model}")

                if confidence < 0.70:
                    st.warning(
                        f"Predicted class: {predicted_name}. "
                        "The model has a low confidence score."
                    )
                else:
                    st.success(
                        f"Predicted class: {predicted_name}"
                    )

                col1, col2 = st.columns(2)

                col1.metric(
                    "Confidence Score",
                    f"{confidence * 100:.2f}%"
                )

                col2.metric(
                    "Prediction Time",
                    f"{prediction_time:.4f} seconds"
                )

                st.caption(
                    "Confidence is the model's output score, "
                    "not the probability of a medical diagnosis."
                )

                st.subheader("Class Scores")

                for i, class_name in enumerate(CLASS_NAMES):

                    score = float(probabilities[i])

                    st.write(
                        f"{class_name}: {score * 100:.2f}%"
                    )

                    st.progress(
                        float(np.clip(score, 0, 1))
                    )

                timestamp = pd.Timestamp.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                st.session_state.prediction_history.insert(
                    0,
                    {
                        "Timestamp": timestamp,
                        "Filename": uploaded_file.name,
                        "Model": selected_model,
                        "Predicted Class": predicted_name,
                        "Confidence (%)": round(
                            confidence * 100, 2
                        )
                    }
                )

                st.session_state.prediction_history = (
                    st.session_state.prediction_history[:100]
                )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )


# ==========================================
# TAB 2: PREDICTION HISTORY
# ==========================================

with history_tab:

    st.subheader("Prediction History")

    st.write(
        "Your prediction history is stored temporarily "
        "during the current application session."
    )

    history = st.session_state.prediction_history

    if history:

        history_df = pd.DataFrame(history)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download History as CSV",
            data=csv_data,
            file_name="skin_prediction_history.csv",
            mime="text/csv"
        )

        if st.button("Clear Prediction History"):

            st.session_state.prediction_history = []

            st.rerun()

    else:

        st.info("No predictions have been made yet.")


# ==========================================
# TAB 3: INDIVIDUAL MODEL EVALUATION
# ==========================================

with evaluation_tab:

    st.subheader("Individual Model Evaluation")

    st.write(
        "Evaluate the selected model using the labelled "
        "test dataset."
    )

    evaluation_model = st.selectbox(
        "Select Model for Evaluation",
        list(MODEL_PATHS.keys()),
        key="evaluation_model"
    )

    if st.button("Evaluate Selected Model"):

        try:

            with st.spinner("Evaluating model..."):

                x_test, y_true = load_test_data()

                results, report, matrix, predictions = (
                    evaluate_model(
                        evaluation_model,
                        x_test,
                        y_true
                    )
                )

                st.session_state["individual_results"] = (
                    evaluation_model,
                    results,
                    report,
                    matrix
                )

            st.success("Evaluation completed!")

        except Exception as error:

            st.error(f"Evaluation failed: {error}")

    if "individual_results" in st.session_state:

        name, results, report, matrix = (
            st.session_state["individual_results"]
        )

        if name == evaluation_model:

            st.subheader(f"{name} Results")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Accuracy",
                f"{results['Accuracy (%)']:.2f}%"
            )

            col2.metric(
                "Precision",
                f"{results['Precision (%)']:.2f}%"
            )

            col3.metric(
                "Recall",
                f"{results['Recall (%)']:.2f}%"
            )

            col4.metric(
                "F1-score",
                f"{results['F1-score (%)']:.2f}%"
            )

            st.subheader("Per-Class Classification Report")

            report_df = pd.DataFrame({
                class_name: {
                    "Precision": report[class_name]["precision"],
                    "Recall": report[class_name]["recall"],
                    "F1-score": report[class_name]["f1-score"],
                    "Support": report[class_name]["support"]
                }
                for class_name in CLASS_NAMES
            }).T

            st.dataframe(
                report_df.style.format({
                    "Precision": "{:.3f}",
                    "Recall": "{:.3f}",
                    "F1-score": "{:.3f}",
                    "Support": "{:.0f}"
                }),
                use_container_width=True
            )

            st.subheader("Confusion Matrix")

            fig, ax = plt.subplots(figsize=(8, 6))

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
                f"{name} Confusion Matrix"
            )

            fig.tight_layout()

            st.pyplot(fig)

            plt.close(fig)


# ==========================================
# TAB 4: FOUR-MODEL COMPARISON
# ==========================================

with comparison_tab:

    st.subheader("Four-Model Performance Comparison")

    st.write(
        "Compare CNN, MobileNetV2, ResNet50V2 and "
        "EfficientNetB0 using the same labelled test images."
    )

    st.info(
        "The comparison uses the full test images for all models. "
        "The CNN was trained using cropped images, whereas the "
        "other models were trained using full images."
    )

    available_models = [
        name for name, path in MODEL_PATHS.items()
        if path.exists()
    ]

    missing_models = [
        name for name, path in MODEL_PATHS.items()
        if not path.exists()
    ]

    if missing_models:

        st.warning(
            "Missing model files: " + ", ".join(missing_models)
        )

    st.write(
        f"Available models: {len(available_models)} / 4"
    )

    if st.button("Run Four-Model Comparison"):

        if len(available_models) == 0:

            st.error("No trained model files were found.")

        else:

            try:

                with st.spinner(
                    "Evaluating all available models. This may take a while..."
                ):

                    x_test, y_true = load_test_data()

                    all_results = []
                    all_matrices = {}

                    for model_name in available_models:

                        st.write(
                            f"Evaluating {model_name}..."
                        )

                        results, report, matrix, predictions = (
                            evaluate_model(
                                model_name,
                                x_test,
                                y_true
                            )
                        )

                        all_results.append(results)

                        all_matrices[model_name] = matrix

                    comparison_df = pd.DataFrame(
                        all_results
                    )

                    st.session_state.comparison_results = (
                        comparison_df
                    )

                    st.session_state.comparison_matrices = (
                        all_matrices
                    )

                st.success(
                    "Model comparison completed successfully!"
                )

            except Exception as error:

                st.error(
                    f"Comparison failed: {error}"
                )

    if st.session_state.comparison_results is not None:

        comparison_df = (
            st.session_state.comparison_results
        )

        st.subheader("Comparison Results")

        st.dataframe(
            comparison_df.style.format({
                "Accuracy (%)": "{:.2f}",
                "Precision (%)": "{:.2f}",
                "Recall (%)": "{:.2f}",
                "F1-score (%)": "{:.2f}",
                "Model Size (MB)": "{:.2f}",
                "Prediction Time (s)": "{:.3f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Accuracy Comparison")

        accuracy_chart = comparison_df.set_index(
            "Model"
        )[["Accuracy (%)"]]

        st.bar_chart(accuracy_chart)

        st.subheader("Precision, Recall and F1-score")

        metrics_chart = comparison_df.set_index(
            "Model"
        )[[
            "Precision (%)",
            "Recall (%)",
            "F1-score (%)"
        ]]

        st.bar_chart(metrics_chart)

        st.subheader("Model Size Comparison")

        size_chart = comparison_df.set_index(
            "Model"
        )[["Model Size (MB)"]]

        st.bar_chart(size_chart)

        st.subheader("Prediction Time Comparison")

        time_chart = comparison_df.set_index(
            "Model"
        )[["Prediction Time (s)"]]

        st.bar_chart(time_chart)

        st.subheader("Confusion Matrices")

        matrices = st.session_state.comparison_matrices

        for model_name, matrix in matrices.items():

            with st.expander(
                f"{model_name} Confusion Matrix"
            ):

                fig, ax = plt.subplots(figsize=(8, 6))

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
                    f"{model_name} Confusion Matrix"
                )

                fig.tight_layout()

                st.pyplot(fig)

                plt.close(fig)

        csv_data = comparison_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Comparison Results",
            data=csv_data,
            file_name="four_model_comparison.csv",
            mime="text/csv"
        )

        st.caption(
            "Prediction time measures one batched prediction over "
            "the test set on your computer. It is hardware-dependent "
            "and should not be treated as single-image latency."
        )


# ==========================================
# TAB 5: SAFETY AND LIMITATIONS
# ==========================================

with safety_tab:

    st.subheader("Safety and Limitations")

    st.warning(
        "This application is an educational image-classification "
        "project, not a medical diagnostic tool."
    )

    st.markdown("""
### Supported disease classes

- Psoriasis
- Eczema
- Tinea-Versicolor
- Acne-Vulgaris

### Model limitations

- The model may make incorrect predictions.
- A high confidence score does not guarantee correctness.
- The model may perform poorly on blurry or unfamiliar images.
- It has not been trained to identify every skin condition.
- Results on a small test dataset may not generalize to new people.

### Responsible use

- Do not use predictions to diagnose or treat a condition.
- Consult a qualified healthcare professional for skin concerns.
- Do not delay medical care because of a model prediction.

### Privacy

Uploaded images are processed for prediction and are not
automatically saved to the project folder.

Prediction history is stored temporarily in the current session.
""")

    st.success(
        "Use this application for learning, experimentation "
        "and demonstrating deep learning image classification."
    )
