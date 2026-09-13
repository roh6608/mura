"""Band-wise streaming of a run's frames."""

from collections.abc import Iterator, Sequence
from pathlib import Path
from types import ModuleType

import numpy as np
from numpy.typing import NDArray

from preprocess.core.constants import (
    DETECTOR_COLUMNS,
    DETECTOR_ROWS,
    DETECTOR_SHAPE,
    FITS_FRAME_BYTES,
    FITS_HEADER_BYTES,
    FITS_PIXEL_DTYPE,
)
from preprocess.core.preprocessing_types import ToFAxis


def read_into(path: str, out: NDArray[np.float32]) -> None:
    """Decode one frame directly into 'out' (a 'DETECTOR_SHAPE' host array)."""
    with Path(path).open("rb") as fh:
        buf = fh.read()
    if len(buf) != FITS_FRAME_BYTES:
        raise ValueError(f"{path}: {len(buf)} B, expected {FITS_FRAME_BYTES}")
    frame = np.frombuffer(buf, dtype=FITS_PIXEL_DTYPE, count=DETECTOR_ROWS * DETECTOR_COLUMNS, offset=FITS_HEADER_BYTES)
    np.copyto(out, frame.reshape(DETECTOR_SHAPE), casting="unsafe")


def iter_bands(
    paths: Sequence[str], axis: ToFAxis, band_size: int = 64, xp: ModuleType = np
) -> Iterator[tuple[int, slice, NDArray[np.float32]]]:
    """Yield '(window_index, bin_slice, band)' in ToF order."""
    if len(paths) != axis.n_bins:
        raise ValueError(f"{len(paths)} frames for a {axis.n_bins}-bin axis")

    host = np.empty((band_size, *DETECTOR_SHAPE), dtype=np.float32)
    for w, sl in enumerate(axis.window_slices):
        for start in range(sl.start, sl.stop, band_size):
            stop = min(start + band_size, sl.stop)
            n = stop - start
            for k in range(n):
                read_into(paths[start + k], host[k])
            band = host[:n] if xp is np else xp.asarray(host[:n])
            yield w, slice(start, stop), (band.copy() if xp is np else band)
