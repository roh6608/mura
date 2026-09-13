"""Gamma-spot despeckling and dead/hot-pixel repair.

Despeckling is transient and per frame; the pixel repair handles persistent defects
identified once from the open beam.
"""

from dataclasses import dataclass
from types import ModuleType

import cv2
import numpy as np
from numpy.typing import NDArray
from shared.core.backend import array_module

from preprocess.core.constants import CUBE_DIMENSIONS, CV2_FLOAT_MEDIAN_KSIZES, FRAME_DIMENSIONS


@dataclass
class DespeckleStats:
    """How much was replaced."""

    n_replaced: int = 0
    n_pixels: int = 0
    counts_removed: float = 0.0
    counts_total: float = 0.0
    max_excess_sigma: float = 0.0

    @property
    def frac_replaced(self) -> float:
        """Return the fraction of pixels whose value was replaced."""
        return self.n_replaced / self.n_pixels if self.n_pixels else 0.0

    @property
    def frac_counts_removed(self) -> float:
        """Return the fraction of total counts removed by despeckling."""
        return self.counts_removed / self.counts_total if self.counts_total else 0.0

    def merge(self, other: "DespeckleStats") -> None:
        """Add another chunk's despeckle statistics into this one."""
        self.n_replaced += other.n_replaced
        self.n_pixels += other.n_pixels
        self.counts_removed += other.counts_removed
        self.counts_total += other.counts_total
        self.max_excess_sigma = max(self.max_excess_sigma, other.max_excess_sigma)


def _median_filter(band: NDArray[np.float32], ksize: int, xp: ModuleType) -> NDArray[np.float32]:
    """3x3 (or 5x5) median over the last two axes of a 2-D or 3-D array."""
    if xp is np:
        if ksize not in CV2_FLOAT_MEDIAN_KSIZES:
            raise ValueError(f"cv2.medianBlur supports ksize {CV2_FLOAT_MEDIAN_KSIZES} for float32, not {ksize}")
        if band.ndim == FRAME_DIMENSIONS:
            return cv2.medianBlur(band, ksize)
        out = np.empty_like(band)
        for i in range(band.shape[0]):
            out[i] = cv2.medianBlur(band[i], ksize)
        return out

    from cupyx.scipy.ndimage import median_filter  # noqa: PLC0415

    size = (ksize, ksize) if band.ndim == FRAME_DIMENSIONS else (1, ksize, ksize)
    return median_filter(band, size=size, mode="reflect")


def _box_filter(band: NDArray[np.float32], ksize: int, xp: ModuleType) -> NDArray[np.float32]:
    """Normalised box filter over the last two axes."""
    if xp is np:
        if band.ndim == FRAME_DIMENSIONS:
            return cv2.blur(band, (ksize, ksize), borderType=cv2.BORDER_REFLECT_101)
        out = np.empty_like(band)
        for i in range(band.shape[0]):
            out[i] = cv2.blur(band[i], (ksize, ksize), borderType=cv2.BORDER_REFLECT_101)
        return out

    from cupyx.scipy.ndimage import uniform_filter  # noqa: PLC0415

    size = (ksize, ksize) if band.ndim == FRAME_DIMENSIONS else (1, ksize, ksize)
    return uniform_filter(band, size=size, mode="reflect")


def despeckle(
    band: NDArray[np.float32],
    k: float = 5.0,
    ksize: int = 3,
    stats: DespeckleStats | None = None,
    ref: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    """Replace bright Poisson outliers with the local median, in place."""
    xp = array_module(band)
    src = band if ref is None else ref
    med_src = _median_filter(src, ksize, xp)
    excess = src - med_src
    thresh = k * xp.sqrt(med_src + 1.0)
    hit = excess > thresh

    med_band = med_src if ref is None else _median_filter(band, ksize, xp)

    if stats is not None:
        n_hit = int(xp.asarray(hit).sum())
        removed = float(xp.asarray(xp.where(hit, band - med_band, 0.0)).sum())
        total = float(xp.asarray(band).sum())
        stats.merge(
            DespeckleStats(
                n_replaced=n_hit,
                n_pixels=int(band.size),
                counts_removed=removed,
                counts_total=total,
                max_excess_sigma=float(
                    xp.asarray(xp.where(thresh > 0, excess / xp.maximum(xp.sqrt(med_src + 1.0), 1e-9), 0.0)).max()
                ),
            )
        )

    band[...] = xp.where(hit, med_band, band)
    return band


def repair(
    band: NDArray[np.float32], invalid: NDArray[np.bool_], ksize: int = 5, passes: int = 2
) -> NDArray[np.float32]:
    """Fill invalid pixels by normalised convolution of their valid neighbours."""
    xp = array_module(band)
    valid = (~xp.asarray(invalid)).astype(band.dtype)
    if valid.ndim == FRAME_DIMENSIONS and band.ndim == CUBE_DIMENSIONS:
        valid = valid[None, :, :]

    out = band
    v = xp.broadcast_to(valid, out.shape).copy()
    for _ in range(max(passes, 1)):
        num = _box_filter(out * v, ksize, xp)
        den = _box_filter(v, ksize, xp)
        fill = (v == 0) & (den > 0)
        out = xp.where(fill, num / xp.maximum(den, 1e-12), out)
        v = xp.where(fill, xp.asarray(1.0, dtype=band.dtype), v)

    return xp.where(v > 0, out, xp.asarray(xp.nan, dtype=band.dtype))
