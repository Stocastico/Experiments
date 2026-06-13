#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from image_classifications.classification_pipeline import ImageClassifier, PretrainedImageClassifier
from image_classifications.utils import load_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="*", help="Image paths")
    parser.add_argument("--list-file", help="Text file with one image path per line")
    parser.add_argument("--model-path")
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()

    paths = load_paths(args.images, args.list_file)
    predictor = PretrainedImageClassifier() if args.pretrained else ImageClassifier.load(Path(args.model_path))

    for item in predictor.predict(paths):
        print(item)


if __name__ == "__main__":
    main()
