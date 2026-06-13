#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from image_classifications.face_pipeline import FaceRecognitionBackend, FaceRecognizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--people-json", required=True, help="JSON mapping: person -> [image paths]")
    parser.add_argument("--output", default="models/face_embeddings.joblib")
    args = parser.parse_args()

    mapping = json.loads(Path(args.people_json).read_text())
    normalized = {person: [Path(path) for path in paths] for person, paths in mapping.items()}

    recognizer = FaceRecognizer(backend=FaceRecognitionBackend())
    embeddings = recognizer.enroll(normalized)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(embeddings, output)
    print(f"Saved embeddings to {output}")


if __name__ == "__main__":
    main()
