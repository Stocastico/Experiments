#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from image_classifications.face_pipeline import FaceRecognitionBackend, FaceRecognizer
from image_classifications.utils import load_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="*", help="Image paths")
    parser.add_argument("--list-file", help="Text file with one image path per line")
    parser.add_argument("--embeddings", default="models/face_embeddings.joblib")
    parser.add_argument("--threshold", type=float, default=0.45)
    args = parser.parse_args()

    recognizer = FaceRecognizer(backend=FaceRecognitionBackend(), threshold=args.threshold)
    recognizer.known_embeddings = joblib.load(Path(args.embeddings))

    for path in load_paths(args.images, args.list_file):
        print(recognizer.recognize(path))


if __name__ == "__main__":
    main()
