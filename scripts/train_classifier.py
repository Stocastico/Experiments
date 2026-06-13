#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from image_classifications.classification_pipeline import ImageClassifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV with columns: path,label")
    parser.add_argument("--output", default="models/image_classifier.joblib")
    args = parser.parse_args()

    rows = list(csv.DictReader(Path(args.csv).read_text().splitlines()))
    paths = [Path(row["path"]) for row in rows]
    labels = [row["label"] for row in rows]

    clf = ImageClassifier()
    clf.train(paths, labels)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    clf.save(out)
    print(f"Saved model to {out}")


if __name__ == "__main__":
    main()
