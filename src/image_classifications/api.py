from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .classification_pipeline import ImageClassifier, PretrainedImageClassifier
from .face_pipeline import FaceRecognitionBackend, FaceRecognizer
from .utils import load_paths

app = FastAPI(title="Image Classifications API")


class PathInput(BaseModel):
    image_paths: list[str] = Field(default_factory=list)
    list_file: str | None = None


class TrainRequest(BaseModel):
    dataset: list[dict[str, str]]
    model_output: str = "models/image_classifier.joblib"


class PredictRequest(PathInput):
    model_path: str | None = None
    use_pretrained: bool = False


class EnrollRequest(BaseModel):
    people: dict[str, list[str]]
    output_path: str = "models/face_embeddings.joblib"


class FacePredictRequest(PathInput):
    embeddings_path: str = "models/face_embeddings.joblib"
    threshold: float = 0.45


@app.post("/classification/train")
def train_classifier(payload: TrainRequest):
    classifier = ImageClassifier()
    image_paths = [Path(item["path"]) for item in payload.dataset]
    labels = [item["label"] for item in payload.dataset]
    classifier.train(image_paths, labels)

    output = Path(payload.model_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    classifier.save(output)

    return {"saved_to": str(output), "trained_samples": len(image_paths)}


@app.post("/classification/predict")
def predict_classifier(payload: PredictRequest):
    paths = load_paths(payload.image_paths, payload.list_file)
    if not paths:
        raise HTTPException(status_code=400, detail="No image paths provided")

    if payload.use_pretrained:
        predictor = PretrainedImageClassifier()
    else:
        if not payload.model_path:
            raise HTTPException(status_code=400, detail="model_path is required when not using pretrained")
        predictor = ImageClassifier.load(Path(payload.model_path))

    return {"results": predictor.predict(paths)}


@app.post("/faces/enroll")
def enroll_faces(payload: EnrollRequest):
    recognizer = FaceRecognizer(backend=FaceRecognitionBackend())
    mapping = {name: [Path(path) for path in images] for name, images in payload.people.items()}
    embeddings = recognizer.enroll(mapping)

    output_path = Path(payload.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import joblib  # noqa: PLC0415

    joblib.dump(embeddings, output_path)
    return {"saved_to": str(output_path), "people": sorted(embeddings.keys())}


@app.post("/faces/predict")
def predict_faces(payload: FacePredictRequest):
    paths = load_paths(payload.image_paths, payload.list_file)
    if not paths:
        raise HTTPException(status_code=400, detail="No image paths provided")

    import joblib  # noqa: PLC0415

    known = joblib.load(Path(payload.embeddings_path))
    recognizer = FaceRecognizer(backend=FaceRecognitionBackend(), threshold=payload.threshold)
    recognizer.known_embeddings = known

    return {"results": [recognizer.recognize(path) for path in paths]}
