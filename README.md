# Photo Atlas

Navigate years of photos by **who's in them**, **what they show**, **where**
and **when** — with a modern deep-learning face pipeline, offline reverse
geocoding, automatic scene tagging and a clean web UI.

Point it at a folder (or 15 years of folders), let it build a catalog, then
browse and filter your library and put names to faces. New imports are
recognised automatically once a person has been named.

![pipeline](https://img.shields.io/badge/faces-YuNet%20%2B%20SFace-4c8dff) ![offline](https://img.shields.io/badge/geocoding-offline-2ea043)

---

## Features

- **Deep face recognition** — OpenCV's [YuNet](https://github.com/opencv/opencv_zoo)
  detector + [SFace](https://github.com/opencv/opencv_zoo) 128-d embeddings.
  On real photos same-person pairs sit at ~0.05 cosine distance and different
  people at ~0.9, so identities separate cleanly.
- **Name people once** — unrecognised faces are grouped into clusters; name a
  cluster and every matching photo becomes filterable. Future imports are
  auto-recognised against the people you've named.
- **Filter by anything** — person, scene type (`people` / `landscape` / `food` /
  `document` / `other`), country, city, year, camera, or filename — combined.
- **Offline reverse geocoding** — GPS EXIF → city + country using a bundled
  dataset (no network). Install `reverse_geocoder` for ~150k-city resolution.
- **Rich metadata** — capture date (EXIF, with file-mtime fallback), camera,
  dimensions, GPS, thumbnails.
- **Web UI** — gallery with lazy thumbnails, a detail lightbox, a People page
  and a "Name faces" workflow. No build step (vanilla JS).
- **Everything local** — a single SQLite catalog plus thumbnails/face crops
  under `~/.photo_atlas`. Your photos never leave your machine.

## Install

```bash
pip install -e .            # core app (Pillow, scikit-learn, OpenCV, FastAPI…)
pip install -e '.[dev]'     # + test tooling
pip install -e '.[dlib]'    # optional: face_recognition (dlib) backend
pip install -e '.[geo]'     # optional: high-resolution reverse geocoding
```

The YuNet + SFace ONNX weights (~38 MB) are downloaded on first use to
`~/.photo_atlas/models`. For offline/air-gapped setups, point
`PHOTO_ATLAS_YUNET` / `PHOTO_ATLAS_SFACE` at local model files.

## Try it in 30 seconds

```bash
photo-atlas demo      # paint & index a synthetic, geotagged library
photo-atlas serve     # open http://127.0.0.1:8000
```

The demo creates cartoon photos with EXIF dates spread across 2010–2024, GPS
near real cities, and three recurring "people" so you can exercise filtering,
clustering and naming without any real photos.

## Use it on your photos

```bash
photo-atlas index ~/Pictures        # walk the tree, extract metadata + faces
photo-atlas cluster                 # group the unnamed faces
photo-atlas serve                   # browse, filter, and name people
photo-atlas stats                   # quick catalog summary
```

`index` is incremental — already-known photos are skipped (use `--recompute` to
force). Choose the face backend with `--faces {auto,yunet,dlib,synthetic,none}`
(default `auto` → YuNet/SFace).

## How it works

```
indexer ─┬─ metadata.py   EXIF date / camera / GPS  + thumbnails
         ├─ geocode.py    GPS → city, country (offline nearest-city)
         ├─ faces.py      YuNet detect → SFace embed → DBSCAN cluster
         ├─ classify.py   colour/face heuristics → scene tag
         └─ db.py         SQLite catalog (photos / persons / faces)

api.py (FastAPI)  →  web/  (gallery · filters · people · name-faces)
```

- **Recognition.** Each named person gets an averaged "centroid" embedding. When
  new photos are indexed, every detected face within
  `face_match_threshold` cosine distance of a centroid is auto-assigned.
- **Clustering.** Unnamed faces are grouped with DBSCAN over cosine distance
  (`cluster_eps`, `cluster_min_samples`) so you can name a whole group at once.
- **Tunables** live in `photo_atlas.config.AtlasConfig`.

## API

`photo-atlas serve` exposes a small JSON API (see `src/photo_atlas/api.py`):

| Method & path | Purpose |
| --- | --- |
| `GET /api/facets` | counts for the filter sidebar |
| `GET /api/photos?person_id=&scene=&country=&city=&year=&camera=&q=` | filtered list |
| `GET /api/photos/{id}` | photo detail + faces |
| `GET /api/image\|thumb/{id}`, `GET /api/face/{id}` | media |
| `GET /api/persons`, `PATCH/DELETE /api/persons/{id}` | manage people |
| `GET /api/clusters`, `POST /api/clusters/{id}/assign` | name a face group |
| `POST /api/faces/{id}/assign` | name a single face |

## Tests

```bash
pytest                            # offline suite (deterministic, no network)
pytest tests/test_deep_faces.py   # deep YuNet/SFace test on real faces*
```

\* downloads the models + a few public sample faces; **skips** (never fails)
when offline. It asserts the deep model puts the same person far closer than two
different people, and that clustering groups repeat photos of one identity.

## Notes

- Built and verified with `opencv-python 4.13`, which already ships the DNN face
  module (`FaceDetectorYN` / `FaceRecognizerSF`). OpenCV 5 wheels were not yet on
  PyPI at the time of writing; the code targets `opencv>=4.10` and is forward
  compatible.
- The original `image_classifications` package (trainable scene classifier +
  multi-label face service, FastAPI) is retained under `src/image_classifications`
  and `scripts/`; Photo Atlas is the application built on top of those ideas.
