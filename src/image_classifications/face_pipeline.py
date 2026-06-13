from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np


class FaceBackend(Protocol):
    saved_embeddings: dict[str, list[list[float]]]

    def detect_and_encode(self, image_path: Path) -> list[list[float]]:
        ...


@dataclass
class FaceRecognizer:
    backend: FaceBackend
    threshold: float = 0.45
    known_embeddings: dict[str, list[float]] = field(default_factory=dict)

    def enroll(self, person_to_images: dict[str, list[Path]]) -> dict[str, list[float]]:
        enrolled: dict[str, list[float]] = {}

        for person, image_paths in person_to_images.items():
            vectors: list[list[float]] = []
            for image_path in image_paths:
                vectors.extend(self.backend.detect_and_encode(image_path))

            if vectors:
                enrolled[person] = np.mean(np.array(vectors), axis=0).tolist()

        self.known_embeddings = enrolled
        return enrolled

    def recognize(self, image_path: Path) -> dict[str, object]:
        encodings = self.backend.detect_and_encode(image_path)
        matches: set[str] = set()

        for encoding in encodings:
            e = np.array(encoding)
            for person, centroid in self.known_embeddings.items():
                distance = np.linalg.norm(e - np.array(centroid))
                if distance <= self.threshold:
                    matches.add(person)

        return {"image": str(image_path), "people": sorted(matches), "face_count": len(encodings)}


class FaceRecognitionBackend:
    """Backend wrapper around face_recognition (optional dependency)."""

    def __init__(self):
        try:
            import face_recognition  # noqa: PLC0415
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "face-recognition is not installed. Install with `pip install 'image-classifications[face]'`."
            ) from exc

        self._lib = face_recognition
        self.saved_embeddings: dict[str, list[list[float]]] = {}

    def detect_and_encode(self, image_path: Path) -> list[list[float]]:
        image = self._lib.load_image_file(image_path)
        encodings = self._lib.face_encodings(image)
        return [encoding.tolist() for encoding in encodings]
