from pathlib import Path

from image_classifications.face_pipeline import FaceRecognizer
from image_classifications.utils import load_paths


def test_load_paths_from_text_file(tmp_path: Path):
    listing = tmp_path / "images.txt"
    listing.write_text("a.jpg\n\n b.jpg \n")

    assert load_paths(list_file=listing) == [Path("a.jpg"), Path("b.jpg")]


class DummyBackend:
    def __init__(self):
        self.saved_embeddings = {
            "Carlo": [[1.0, 0.0]],
            "Luigi": [[0.0, 1.0]],
        }

    def detect_and_encode(self, image_path):
        if image_path == Path("scene.jpg"):
            return [[1.0, 0.0], [0.0, 1.0]]
        return []


def test_face_recognizer_is_multi_label():
    recognizer = FaceRecognizer(backend=DummyBackend(), threshold=0.2)
    recognizer.known_embeddings = {
        "Carlo": [1.0, 0.0],
        "Luigi": [0.0, 1.0],
    }

    result = recognizer.recognize(Path("scene.jpg"))

    assert set(result["people"]) == {"Carlo", "Luigi"}
