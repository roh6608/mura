"""Fast reader for the fixed-layout FITS frames."""

import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from preprocess.core.constants import (
    DETECTOR_COLUMNS,
    DETECTOR_ROWS,
    DETECTOR_SHAPE,
    FITS_FRAME_BYTES,
    FITS_HEADER_BYTES,
    FITS_HEADER_CARD_BYTES,
    FITS_PIXEL_DTYPE,
)
from preprocess.core.preprocessing_types import FrameHeader
from preprocess.io.naming import acq_counter, frame_index


def iter_frame_paths(run_dir: str | os.PathLike, counter: str | None = None) -> list[str]:
    """Numbered ToF frames in a run directory, ordered by parsed integer index."""
    hits: list[tuple[int, str]] = []
    for path in (str(p) for p in Path(run_dir).glob("*.fits")):
        base = Path(path).name
        idx = frame_index(base)
        if idx is None:
            continue
        if counter is not None and acq_counter(base) != counter:
            continue
        hits.append((idx, path))
    hits.sort()
    return [p for _, p in hits]


def parse_header(block: bytes) -> FrameHeader:
    """Parse a 2880-byte FITS header block."""
    cards: dict[str, float | str] = {}
    for off in range(0, len(block), 80):
        card = block[off : off + 80]
        if len(card) < FITS_HEADER_CARD_BYTES:
            break
        key = card[:8].decode("ascii", "replace").strip()
        if key == "END":
            break
        if not key or key in ("COMMENT", "HISTORY") or card[8:10] != b"= ":
            continue
        raw = card[10:].decode("ascii", "replace")
        # Strip the inline comment, but not a '/' inside a quoted string.
        if raw.lstrip().startswith("'"):
            end = raw.index("'", raw.index("'") + 1)
            value: float | str = raw[raw.index("'") + 1 : end].strip()
        else:
            token = raw.split("/", 1)[0].strip()
            try:
                value = float(token)
            except ValueError:
                value = token
        cards[key] = value
    return FrameHeader(cards)


def read_header(path: str | os.PathLike) -> FrameHeader:
    """Read just the header block of a frame."""
    with Path(path).open("rb") as fh:
        return parse_header(fh.read(FITS_HEADER_BYTES))


def read_frame(
    path: str | os.PathLike, out: NDArray[np.int32] | None = None, *, with_header: bool = False
) -> NDArray[np.int32] | tuple[NDArray[np.int32], FrameHeader]:
    """Decode one frame to '(rows, columns)' counts as int32."""
    with Path(path).open("rb") as fh:
        buf = fh.read()
    if len(buf) != FITS_FRAME_BYTES:
        raise ValueError(f"{path}: {len(buf)} B, expected {FITS_FRAME_BYTES} (truncated or not a VENUS frame)")

    data = np.frombuffer(
        buf, dtype=FITS_PIXEL_DTYPE, count=DETECTOR_ROWS * DETECTOR_COLUMNS, offset=FITS_HEADER_BYTES
    ).reshape(DETECTOR_SHAPE)
    if out is None:
        arr = data.astype(np.int32)
    else:
        np.copyto(out, data, casting="unsafe")
        arr = out
    if with_header:
        return arr, parse_header(buf[:FITS_HEADER_BYTES])
    return arr
