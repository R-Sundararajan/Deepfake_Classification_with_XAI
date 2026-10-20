"""Command-line inference with confidence, Grad-CAM, and honest text output."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config import IMAGE_SIZE, MODEL_OUTPUT_PATH, OUTPUT_DIR
from explainability.gradcam import GradCAM, create_visualizations, describe_regions
from models.classifier import load_model_checkpoint, target_layer
from training.data import evaluation_transform


class DeepfakePredictor:
    def __init__(self, checkpoint_path: str | Path = MODEL_OUTPUT_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.checkpoint = load_model_checkpoint(checkpoint_path, self.device)
        self.class_to_idx = self.checkpoint.get("class_to_idx", {"fake": 0, "real": 1})
        self.transform = evaluation_transform()

    def predict(self, image: Image.Image) -> dict:
        image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            predicted_index = int(probabilities.argmax().item())

        index_to_class = {index: name for name, index in self.class_to_idx.items()}
        predicted_class = index_to_class[predicted_index].upper()
        confidence = float(probabilities[predicted_index].item())

        gradcam = GradCAM(self.model, target_layer(self.model))
        try:
            cam = gradcam.generate(input_tensor, predicted_index)
        finally:
            gradcam.close()

        original = np.asarray(image, dtype=np.uint8)
        heatmap, overlay = create_visualizations(original, cam)
        region_text = describe_regions(cam)
        region_label = (
            f"{region_text} image regions"
            if " and " in region_text
            else f"{region_text} image region"
        )
        confidence_word = "high" if confidence >= 0.9 else "moderate" if confidence >= 0.7 else "low"
        explanation = (
            f"The model classified this image as {predicted_class} with {confidence_word} "
            f"model confidence ({confidence:.2%}). The strongest Grad-CAM activations are "
            f"concentrated in the {region_label}. This indicates where the "
            "network relied most strongly for this prediction; it does not identify or prove "
            "a particular manipulation artifact."
        )
        if float(np.std(cam)) < 1e-8:
            region_text = 'unavailable (constant attribution)'
            explanation = f'The model classified this image as {predicted_class} ({confidence:.2%} model score). Grad-CAM is constant; spatial localization is unavailable.'

        return {
            "prediction": predicted_class,
            "confidence": confidence,
            "real_probability": float(probabilities[self.class_to_idx["real"]].item()),
            "fake_probability": float(probabilities[self.class_to_idx["fake"]].item()),
            "activation_regions": region_text,
            "explanation": explanation,
            "original": original,
            "heatmap": heatmap,
            "overlay": overlay,
        }


def save_result(result: dict, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result["original"]).save(output_dir / "original.png")
    Image.fromarray(result["heatmap"]).save(output_dir / "gradcam_heatmap.png")
    Image.fromarray(result["overlay"]).save(output_dir / "gradcam_overlay.png")
    serializable = {key: value for key, value in result.items() if not isinstance(value, np.ndarray)}
    (output_dir / "prediction.json").write_text(
        json.dumps(serializable, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=MODEL_OUTPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "inference_demo")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    with Image.open(arguments.image) as uploaded_image:
        prediction_result = DeepfakePredictor(arguments.checkpoint).predict(uploaded_image)
    save_result(prediction_result, arguments.output_dir)
    print(f"Prediction: {prediction_result['prediction']}")
    print(f"Model confidence: {prediction_result['confidence']:.2%}")
    print(f"REAL probability: {prediction_result['real_probability']:.2%}")
    print(f"FAKE probability: {prediction_result['fake_probability']:.2%}")
    print(f"Explanation: {prediction_result['explanation']}")
    print(f"Saved output: {arguments.output_dir.resolve()}")
