from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LABELS = ["cibo", "paesaggi", "persone"]


def extract_features(image_path: Path) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize((96, 96))
    arr = np.array(img, dtype=np.float32) / 255.0
    means = arr.mean(axis=(0, 1))
    stds = arr.std(axis=(0, 1))
    return np.concatenate([means, stds], axis=0)


@dataclass
class ImageClassifier:
    model: Pipeline | None = None

    def train(self, image_paths: list[Path], labels: list[str]) -> None:
        x = np.array([extract_features(path) for path in image_paths])
        y = np.array(labels)

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(max_iter=1500, multi_class="multinomial"),
                ),
            ]
        )
        pipeline.fit(x, y)
        self.model = pipeline

    def predict(self, image_paths: list[Path]) -> list[dict[str, object]]:
        if self.model is None:
            raise RuntimeError("Model is not trained or loaded.")

        x = np.array([extract_features(path) for path in image_paths])
        probs = self.model.predict_proba(x)
        classes = list(self.model.classes_)
        results = []

        for path, vector in zip(image_paths, probs, strict=True):
            score_map = {label: float(vector[classes.index(label)]) for label in classes}
            results.append(
                {
                    "image": str(path),
                    "label": max(score_map, key=score_map.get),
                    "scores": score_map,
                }
            )

        return results

    def save(self, output_path: Path) -> None:
        if self.model is None:
            raise RuntimeError("No model to save.")
        joblib.dump(self.model, output_path)

    @classmethod
    def load(cls, model_path: Path) -> "ImageClassifier":
        model = joblib.load(model_path)
        return cls(model=model)


class PretrainedImageClassifier:
    """Zero-shot classifier backed by transformers pipeline when available."""

    def __init__(self):
        try:
            from transformers import pipeline  # noqa: PLC0415
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "transformers is not installed. Install with `pip install transformers torch`."
            ) from exc

        self._pipeline = pipeline(model="openai/clip-vit-base-patch32", task="zero-shot-image-classification")

    def predict(self, image_paths: list[Path]) -> list[dict[str, object]]:
        candidate_labels = ["food", "landscape", "person"]
        translate = {"food": "cibo", "landscape": "paesaggi", "person": "persone"}

        out = []
        for path in image_paths:
            predictions = self._pipeline(str(path), candidate_labels=candidate_labels)
            scores = {translate[item["label"]]: float(item["score"]) for item in predictions}
            out.append({"image": str(path), "label": max(scores, key=scores.get), "scores": scores})
        return out
