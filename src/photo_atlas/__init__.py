"""Photo Atlas.

A self-hosted application to navigate, search and curate a large personal
photo library (years of pictures) by **person**, **scene type**, **place**
and **date**.

The package is organised in small, composable layers:

* :mod:`photo_atlas.config`    -- runtime configuration and paths.
* :mod:`photo_atlas.db`        -- SQLite schema and repository helpers.
* :mod:`photo_atlas.metadata`  -- EXIF / file metadata + thumbnails.
* :mod:`photo_atlas.geocode`   -- offline reverse geocoding.
* :mod:`photo_atlas.classify`  -- lightweight scene tagging.
* :mod:`photo_atlas.faces`     -- face detection, embedding, clustering, naming.
* :mod:`photo_atlas.indexer`   -- ingest a directory tree into the library.
* :mod:`photo_atlas.search`    -- turn filters into SQL queries.
* :mod:`photo_atlas.api`       -- FastAPI app + web UI.
* :mod:`photo_atlas.cli`       -- command line entry point.
"""

__version__ = "0.1.0"
