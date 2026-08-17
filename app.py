"""Streamlit demonstration interface for the explainable classifier."""

from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from config import MODEL_OUTPUT_PATH
from inference.predict import DeepfakePredictor


st.set_page_config(page_title="Explainable Deepfake Classifier", page_icon="🔎", layout="wide")
st.title("Explainable Deepfake Image Classification")
st.caption(
    "A baseline demonstration: image → REAL/FAKE → model confidence → Grad-CAM → text explanation"
)


@st.cache_resource(show_spinner=False)
def load_predictor(checkpoint_path: str, checkpoint_mtime: float) -> DeepfakePredictor:
    del checkpoint_mtime
    return DeepfakePredictor(checkpoint_path)


if not MODEL_OUTPUT_PATH.exists():
    st.error(f"Model checkpoint not found: {MODEL_OUTPUT_PATH}")
    st.code("python -m training.train", language="bash")
    st.stop()

uploaded = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png", "bmp", "webp"])
if uploaded is None:
    st.info("Upload an image to run the classifier and generate Grad-CAM.")
    st.stop()

try:
    image = Image.open(uploaded).convert("RGB")
except (UnidentifiedImageError, OSError):
    st.error("The uploaded file could not be read as an image.")
    st.stop()

with st.spinner("Running classification and Grad-CAM..."):
    predictor = load_predictor(str(MODEL_OUTPUT_PATH), MODEL_OUTPUT_PATH.stat().st_mtime)
    result = predictor.predict(image)

prediction_color = "red" if result["prediction"] == "FAKE" else "green"
metric_columns = st.columns(4)
metric_columns[0].markdown(f"### Prediction: :{prediction_color}[{result['prediction']}]")
metric_columns[1].metric("Model confidence", f"{result['confidence']:.2%}")
metric_columns[2].metric("REAL score", f"{result['real_probability']:.2%}")
metric_columns[3].metric("FAKE score", f"{result['fake_probability']:.2%}")

st.subheader("Grad-CAM explanation")
image_columns = st.columns(3)
image_columns[0].image(result["original"], caption="Original image", width="stretch")
image_columns[1].image(result["heatmap"], caption="Grad-CAM heatmap", width="stretch")
image_columns[2].image(result["overlay"], caption="Heatmap overlay", width="stretch")

st.subheader("Why did the model produce this output?")
st.write(result["explanation"])
st.warning(
    "Model confidence is not a calibrated forensic probability. Grad-CAM shows influential "
    "regions, not proof of manipulation or a semantic diagnosis of an artifact."
)
