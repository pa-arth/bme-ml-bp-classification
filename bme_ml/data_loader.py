"""Load UCI Cuff-Less BP `.mat` files.

The dataset comes as four files (`Part_1.mat` through `Part_4.mat`). Each is a
MATLAB cell array where each cell is a `(3, N)` matrix:
    row 0 = PPG, row 1 = ABP (mmHg), row 2 = ECG.

Sample rate is 125 Hz. Record lengths vary; expect anywhere from a few
thousand to several hundred thousand samples per record.

Part_1.mat is saved as v7.3 (HDF5-backed) and must be opened with `h5py`.
The other parts are v7.0–v7.2 and load with `scipy.io.loadmat`. We dispatch
on the file's magic bytes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np


def _is_hdf5(path: Path) -> bool:
    """v7.3 .mat files are HDF5-backed but begin with a text header
    ('MATLAB 7.3 MAT-file...'); the HDF5 superblock starts at offset 128.
    Older v5–v7.2 .mat files begin with 'MATLAB 5.0 MAT-file'."""
    with open(path, "rb") as f:
        head = f.read(10)
    return head.startswith(b"MATLAB 7.3")


def _iter_records_h5(path: Path) -> Iterator[np.ndarray]:
    """Iterate `(3, N)` records from a v7.3 (HDF5) MATLAB file.

    UCI ships the cell array as `(N, 1)` (column vector) — flatten the
    refs so the iterator works regardless of `(1, N)` vs `(N, 1)` layout.
    """
    import h5py

    with h5py.File(path, "r") as f:
        cell_keys = [k for k in f.keys() if not k.startswith("#")]
        if not cell_keys:
            raise RuntimeError(f"No cell variable found in {path}")
        cell_ref = f[cell_keys[0]]
        refs = np.asarray(cell_ref[()]).reshape(-1)
        for ref in refs:
            data = np.array(f[ref])
            # HDF5 storage is column-major-flipped relative to MATLAB; UCI
            # records come back as `(N, 3)`. Transpose to `(3, N)` to match
            # the v7.2 convention used by the rest of the loader.
            if data.shape[0] != 3 and data.shape[-1] == 3:
                data = data.T
            yield data.astype(np.float32, copy=False)


def _iter_records_mat(path: Path) -> Iterator[np.ndarray]:
    """Iterate `(3, N)` records from a v7.0–v7.2 MATLAB file."""
    from scipy.io import loadmat

    mat = loadmat(path, squeeze_me=False)
    cell_keys = [k for k in mat.keys() if not k.startswith("__")]
    if not cell_keys:
        raise RuntimeError(f"No cell variable found in {path}")
    cell = mat[cell_keys[0]]
    # Shape is typically (1, n_cells); each entry is a (3, N) ndarray.
    flat = cell.flatten()
    for entry in flat:
        arr = np.asarray(entry)
        if arr.shape[0] != 3 and arr.shape[-1] == 3:
            arr = arr.T
        yield arr.astype(np.float32, copy=False)


def iter_records(path: str | Path) -> Iterator[np.ndarray]:
    """Yield `(3, N)` arrays — one per record in the given .mat file.

    Channel order: row 0 = PPG, row 1 = ABP, row 2 = ECG.
    """
    path = Path(path)
    if _is_hdf5(path):
        yield from _iter_records_h5(path)
    else:
        yield from _iter_records_mat(path)


def iter_all_records(
    paths: list[str | Path], *, max_records: int | None = None
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield `(part_idx, record_idx, arr)` across multiple .mat parts.

    `part_idx` is 1-indexed to match the file naming (`Part_1.mat`).
    `max_records` bounds the *total* number of records returned across all
    parts, which is convenient for smoke-test runs.
    """
    yielded = 0
    for part_idx, p in enumerate(paths, start=1):
        for rec_idx, arr in enumerate(iter_records(p)):
            yield part_idx, rec_idx, arr
            yielded += 1
            if max_records is not None and yielded >= max_records:
                return
