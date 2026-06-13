from pathlib import Path
from typing import Iterable


def load_paths(paths: Iterable[str | Path] | None = None, list_file: str | Path | None = None) -> list[Path]:
    collected: list[Path] = []

    if paths:
        collected.extend(Path(p) for p in paths)

    if list_file:
        file_path = Path(list_file)
        rows = [line.strip() for line in file_path.read_text().splitlines()]
        collected.extend(Path(row) for row in rows if row)

    return collected
