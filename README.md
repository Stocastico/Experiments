# Image Classifications

Pipeline Python + FastAPI per:

1. Classificare immagini nelle classi `cibo`, `paesaggi`, `persone`.
2. Riconoscimento facciale multi-label (es. nella stessa immagine possono comparire sia Carlo che Luigi).

## Installazione

```bash
pip install -e '.[dev]'
# opzionale per face recognition
pip install -e '.[face]'
```

## Training classificatore

Prepara un CSV con colonne `path,label`.

```bash
python scripts/train_classifier.py --csv data/train.csv --output models/image_classifier.joblib
```

## Inferenza classificatore

Con modello allenato:

```bash
python scripts/predict_classifier.py --model-path models/image_classifier.joblib image1.jpg image2.jpg
```

Con modello pre-trained (CLIP zero-shot):

```bash
python scripts/predict_classifier.py --pretrained image1.jpg image2.jpg
```

Oppure lista da file:

```bash
python scripts/predict_classifier.py --model-path models/image_classifier.joblib --list-file image_paths.txt
```

## Face recognition multi-label

### Enroll persone

JSON input:

```json
{
  "Carlo": ["carlo_1.jpg", "carlo_2.jpg"],
  "Luigi": ["luigi_1.jpg"]
}
```

```bash
python scripts/enroll_faces.py --people-json data/people.json --output models/face_embeddings.joblib
```

### Inferenza facce

```bash
python scripts/predict_faces.py --embeddings models/face_embeddings.joblib scene.jpg
python scripts/predict_faces.py --embeddings models/face_embeddings.joblib --list-file image_paths.txt
```

## API FastAPI

```bash
uvicorn image_classifications.api:app --reload
```

Endpoint principali:
- `POST /classification/train`
- `POST /classification/predict`
- `POST /faces/enroll`
- `POST /faces/predict`

`/faces/predict` restituisce un array di persone trovate per immagine, supportando output multi-label.
