"""Timepix3 quad geometry, gain structure, and pixel masks."""

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from preprocess.core.constants import CHIP, DETECTOR_SHAPE

# Mask bit flags.
GOOD = np.uint8(0)
DEAD = np.uint8(1 << 0)
HOT = np.uint8(1 << 1)
SEAM = np.uint8(1 << 2)
BORDER = np.uint8(1 << 3)
SATURATED = np.uint8(1 << 4)

FLAG_NAMES = {DEAD: "dead", HOT: "hot", SEAM: "seam", BORDER: "border", SATURATED: "saturated"}


def chip_slices() -> dict[str, tuple[slice, slice]]:
    """Return the four 256x256 chips of the quad, keyed by corner."""
    return {
        "TL": (slice(0, CHIP), slice(0, CHIP)),
        "TR": (slice(0, CHIP), slice(CHIP, 2 * CHIP)),
        "BL": (slice(CHIP, 2 * CHIP), slice(0, CHIP)),
        "BR": (slice(CHIP, 2 * CHIP), slice(CHIP, 2 * CHIP)),
    }


@dataclass(frozen=True)
class ChipGain:
    """Per-chip mean response and its spread."""

    means: dict[str, float]

    @property
    def spread(self) -> float:
        """Return the ratio of the largest chip mean to the smallest."""
        return max(self.means.values()) / min(self.means.values())

    def ratios(self, global_mean: float) -> dict[str, float]:
        """Return each chip's mean divided by 'global_mean'."""
        return {k: v / global_mean for k, v in self.means.items()}


def measure_chip_gain(white_beam: NDArray[np.floating]) -> ChipGain:
    """Mean open-beam response of each chip."""
    return ChipGain({k: float(white_beam[ys, xs].mean()) for k, (ys, xs) in chip_slices().items()})


def border_mask(shape: tuple[int, int] = DETECTOR_SHAPE, n: int = 4) -> NDArray[np.bool_]:
    """Boolean mask of the outer 'n' rows and columns."""
    m = np.zeros(shape, dtype=bool)
    if n > 0:
        m[:n, :] = m[-n:, :] = m[:, :n] = m[:, -n:] = True
    return m


def seam_mask(shape: tuple[int, int] = DETECTOR_SHAPE, n: int = 1) -> NDArray[np.bool_]:
    """Boolean mask of the 'n' rows/columns either side of each chip seam."""
    m = np.zeros(shape, dtype=bool)
    lo, hi = CHIP - n, CHIP + n
    m[lo:hi, :] = True
    m[:, lo:hi] = True
    return m


def bad_pixel_map(  # noqa: PLR0913, PLR0917
    white_beam: NDArray[np.floating],
    *,
    hot_sigma: float = 6.0,
    dead_fraction: float = 0.2,
    border_px: int = 4,
    seam_px: int = 1,
    dilate: int = 1,
) -> NDArray[np.uint8]:
    """Build the packed pixel mask from an open-beam white-beam image."""
    mask = np.zeros(white_beam.shape, dtype=np.uint8)

    for ys, xs in chip_slices().values():
        sub = white_beam[ys, xs]
        med = float(np.median(sub))
        mad = float(np.median(np.abs(sub - med))) * 1.4826
        scale = mad if mad > 0 else max(np.sqrt(max(med, 1.0)), 1.0)
        mask[ys, xs] |= np.where(sub > med + hot_sigma * scale, HOT, GOOD).astype(np.uint8)
        mask[ys, xs] |= np.where(sub < dead_fraction * med, DEAD, GOOD).astype(np.uint8)

    if dilate > 0:
        defects = ((mask & (HOT | DEAD)) > 0).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilate + 1, 2 * dilate + 1))
        grown = cv2.dilate(defects, k) > 0
        mask[grown & ((mask & (HOT | DEAD)) == 0)] |= DEAD

    mask[seam_mask(white_beam.shape, seam_px)] |= SEAM
    mask[border_mask(white_beam.shape, border_px)] |= BORDER
    return mask


def usable(mask: NDArray[np.uint8]) -> NDArray[np.bool_]:
    """Boolean map of pixels with no flags set."""
    return mask == GOOD


def summarise_mask(mask: NDArray[np.uint8]) -> dict[str, int]:
    """Count pixels per flag, plus the usable total."""
    out = {name: int(((mask & flag) > 0).sum()) for flag, name in FLAG_NAMES.items()}
    out["usable"] = int(usable(mask).sum())
    out["total"] = int(mask.size)
    return out
