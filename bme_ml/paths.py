"""Project paths + dataset download.

`setup_paths()` defaults to the project root (the directory containing
`bme_ml/`), so you can call it with no arguments from anywhere in the repo
or from a notebook in `notebooks/`. All generated artifacts live under
`<root>/data/processed/` and `<root>/models/`.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/340/cuff+less+blood+pressure+estimation.zip"
)
PART_FILES = ["Part_1.mat", "Part_2.mat", "Part_3.mat", "Part_4.mat"]


@dataclass(frozen=True)
class Paths:
    root: Path
    raw: Path
    processed: Path
    models: Path

    @property
    def features_parquet(self) -> Path:
        return self.processed / "features.parquet"

    @property
    def signals_h5(self) -> Path:
        return self.processed / "signals.h5"

    @property
    def splits_json(self) -> Path:
        return self.processed / "splits.json"


def project_root() -> Path:
    """Return the project root (parent of the `bme_ml/` package)."""
    return Path(__file__).resolve().parent.parent


def setup_paths(root: str | Path | None = None) -> Paths:
    """Create the project directory tree under `root`. If `root` is None,
    use the project root inferred from this file's location."""
    root = Path(root) if root is not None else project_root()
    raw = root / "data" / "raw"
    processed = root / "data" / "processed"
    models = root / "models"
    for d in (raw, processed, models):
        d.mkdir(parents=True, exist_ok=True)
    return Paths(root=root, raw=raw, processed=processed, models=models)


def download_dataset(paths: Paths, force: bool = False) -> list[Path]:
    """Download the UCI Cuff-Less BP dataset .mat parts into `paths.raw`.

    UCI ships the dataset as a single zip containing the four `Part_*.mat`
    files. Idempotent — if all four are present, returns immediately.
    """
    expected = [paths.raw / name for name in PART_FILES]
    if not force and all(p.exists() and p.stat().st_size > 0 for p in expected):
        return expected

    zip_path = paths.raw / "cuff-less-bp.zip"
    if force or not zip_path.exists():
        print(f"Downloading {UCI_ZIP_URL} ...")
        urllib.request.urlretrieve(UCI_ZIP_URL, zip_path)

    import zipfile

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(paths.raw)

    missing = [p for p in expected if not p.exists()]
    if missing:
        raise RuntimeError(
            f"Expected .mat files missing after extract: {missing}. "
            f"Zip contents: {list(paths.raw.iterdir())}"
        )
    return expected
