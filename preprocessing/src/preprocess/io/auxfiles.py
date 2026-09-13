"""Parsers for the sidecar text files Pixet writes alongside each run."""

import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from preprocess.core.constants import (
    SHUTTER_COUNT_COLUMNS,
    SHUTTER_TIMES_COLUMNS,
    SIDECAR_TABLE_DIMENSIONS,
    SPECTRA_COLUMNS,
)
from preprocess.core.preprocessing_types import AuxPaths, ShutterTimes
from preprocess.io.naming import acq_counter


def find_acquisitions(run_dir: str | os.PathLike) -> list[AuxPaths]:
    """Locate every acquisition in a run directory, keyed by its counter."""
    d = str(run_dir)
    dsc = sorted(str(p) for p in Path(d).glob("*.dsc"))
    dsc_path = dsc[0] if len(dsc) == 1 else None

    out: list[AuxPaths] = []
    for st in sorted(str(p) for p in Path(d).glob("*_ShutterTimes.txt")):
        stem = st[: -len("_ShutterTimes.txt")]
        counter = _counter_of(Path(st).name)
        missing = [
            name
            for name, p in (("ShutterCount.txt", f"{stem}_ShutterCount.txt"), ("Spectra.txt", f"{stem}_Spectra.txt"))
            if not Path(p).exists()
        ]
        if missing:
            raise FileNotFoundError(f"acquisition {counter} in {d} is missing {missing}")
        status = f"{stem}_Status.txt"
        out.append(
            AuxPaths(
                shutter_times=st,
                shutter_count=f"{stem}_ShutterCount.txt",
                spectra=f"{stem}_Spectra.txt",
                status=status if Path(status).exists() else None,
                dsc=dsc_path,
                stem=Path(stem).name,
                counter=counter,
            )
        )
    if not out:
        raise FileNotFoundError(f"no *_ShutterTimes.txt in {d}")
    return out


def _counter_of(basename: str) -> str:

    c = acq_counter(basename)
    if c is None:
        raise ValueError(f"cannot parse an acquisition counter from {basename!r}")
    return c


def read_shutter_times(path: str | os.PathLike) -> ShutterTimes:
    """
    Parse 'ShutterTimes.txt', keeping only the active windows.

    A window is active if it has a non-zero duration.
    """
    raw = np.loadtxt(path, dtype=np.float64)
    if raw.ndim != SIDECAR_TABLE_DIMENSIONS or raw.shape[1] != SHUTTER_TIMES_COLUMNS:
        raise ValueError(f"{path}: expected Nx3, got {raw.shape}")
    active = raw[:, 2] > 0.0
    if not active.any():
        raise ValueError(f"{path}: no window has a non-zero duration")

    idx = np.flatnonzero(active)
    if not np.array_equal(idx, np.arange(idx[0], idx[-1] + 1)):
        raise ValueError(f"{path}: active windows are not contiguous: {idx}")
    return ShutterTimes(delay_s=raw[active, 1].copy(), duration_s=raw[active, 2].copy())


def read_shutter_counts(path: str | os.PathLike, n_windows: int) -> NDArray[np.int64]:
    """Per-window trigger counts from 'ShutterCount.txt'."""
    raw = np.loadtxt(path, dtype=np.int64)
    if raw.ndim != SIDECAR_TABLE_DIMENSIONS or raw.shape[1] != SHUTTER_COUNT_COLUMNS:
        raise ValueError(f"{path}: expected Nx2, got {raw.shape}")
    counts = raw[:n_windows, 1].copy()
    if (counts <= 0).any():
        raise ValueError(f"{path}: non-positive trigger count in the first {n_windows} windows: {counts}")
    return counts


def read_spectra(path: str | os.PathLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """'(tof_start_seconds, total_counts)' from 'Spectra.txt'."""
    raw = np.loadtxt(path, dtype=np.float64)
    if raw.ndim != SIDECAR_TABLE_DIMENSIONS or raw.shape[1] != SPECTRA_COLUMNS:
        raise ValueError(f"{path}: expected Nx2, got {raw.shape}")
    return raw[:, 0].copy(), raw[:, 1].copy()
