"""Event pile-up (dead-time) correction for the timepix detector.

Per pixel and per shutter window, 'N_true(i) = M(i) / (1 - cum_{j<i}(M) / N_trig_w)',
on RAW counts.
"""

from dataclasses import dataclass, field
from types import ModuleType

import numpy as np
from numpy.typing import NDArray
from shared.core.backend import asnumpy, check_same_module

from preprocess.core.constants import CUBE_DIMENSIONS
from preprocess.core.preprocessing_types import BoundaryStep, ToFAxis


@dataclass
class SaturationStats:
    """Diagnostics for one correction pass. Accumulates across streamed bands."""

    n_bins: int = 0
    n_pixels: int = 0
    min_denominator: float = 1.0
    n_below_floor: int = 0
    max_correction: float = 1.0
    _corr_sum: float = field(default=0.0, repr=False)
    _corr_n: int = field(default=0, repr=False)
    per_window: dict[int, tuple[float, int]] = field(default_factory=dict)
    pixel_hits: NDArray[np.int64] | None = field(default=None, repr=False)

    def add_pixel_hits(self, hits: NDArray[np.int64]) -> None:
        """Accumulate per-pixel saturation hits, allocating the buffer on first use."""
        if self.pixel_hits is None:
            self.pixel_hits = np.zeros(hits.shape, dtype=np.int64)
        self.pixel_hits += hits

    @property
    def n_saturating_pixels(self) -> int:
        """Return the number of pixels that saturated at least once."""
        return 0 if self.pixel_hits is None else int((self.pixel_hits > 0).sum())

    @property
    def mean_correction(self) -> float:
        """Return the mean overlap correction factor, NaN when nothing was measured."""
        return self._corr_sum / self._corr_n if self._corr_n else float("nan")

    @property
    def frac_below_floor(self) -> float:
        """Return the fraction of bins that fell below the count floor."""
        total = self.n_bins * self.n_pixels
        return self.n_below_floor / total if total else 0.0

    def window_mean(self, w: int) -> float:
        """Return the mean correction for ToF window 'w', NaN when it was not measured."""
        s, n = self.per_window.get(w, (0.0, 0))
        return s / n if n else float("nan")

    def update(  # noqa: PLR0913, PLR0917
        self,
        *,
        n_bins: int,
        n_pixels: int,
        min_denom: float,
        n_below: int,
        max_corr: float,
        corr_sum: float,
        corr_n: int,
        window: int | None = None,
    ) -> None:
        """Fold one band's statistics into the running totals."""
        self.n_bins += n_bins
        self.n_pixels = max(self.n_pixels, n_pixels)
        self.min_denominator = min(self.min_denominator, min_denom)
        self.n_below_floor += n_below
        self.max_correction = max(self.max_correction, max_corr)
        self._corr_sum += corr_sum
        self._corr_n += corr_n
        if window is not None:
            s, n = self.per_window.get(window, (0.0, 0))
            self.per_window[window] = (s + corr_sum, n + corr_n)


def overlap_correct_band(  # noqa: PLR0913, PLR0917
    band: NDArray[np.float32],
    accum: NDArray[np.float32],
    n_trig: int,
    *,
    floor: float = 0.20,
    half_bin: bool = False,
    stats: SaturationStats | None = None,
    window: int | None = None,
) -> NDArray[np.float32]:
    """Correct one contiguous band of frames from a single shutter window."""
    xp = check_same_module(band, accum, where="overlap_correct_band")
    if band.ndim != CUBE_DIMENSIONS:
        raise ValueError(f"band must be (n_bins, ny, nx), got {band.shape}")
    if n_trig <= 0:
        raise ValueError(f"n_trig must be positive, got {n_trig}")

    # Triggers in which each pixel has already fired before this bin.
    csum = xp.cumsum(band, axis=0)
    prior = csum - band + accum[None, :, :]
    if half_bin:
        prior = prior + 0.5 * band

    denom = 1.0 - prior / np.float32(n_trig)

    below = denom < floor
    min_denom = float(asnumpy(denom.min()))
    n_below = int(asnumpy(below.sum()))
    if stats is not None and n_below:
        stats.add_pixel_hits(asnumpy(below.sum(axis=0)))
    xp.clip(denom, floor, 1.0, out=denom)

    band /= denom

    if stats is not None:
        inv = 1.0 / denom
        stats.update(
            n_bins=band.shape[0],
            n_pixels=band.shape[1] * band.shape[2],
            min_denom=min_denom,
            n_below=n_below,
            max_corr=float(asnumpy(inv.max())),
            corr_sum=float(asnumpy(inv.sum())),
            corr_n=inv.size,
            window=window,
        )

    accum += csum[-1]
    return band


class WindowAccumulator:
    """Tracks the per-window occupancy accumulator across a streamed run."""

    def __init__(self, shape: tuple[int, int], xp: ModuleType = np) -> None:
        """Prepare a zeroed accumulator of 'shape' on the given array module."""
        self._xp = xp
        self._accum = xp.zeros(shape, dtype=xp.float32)
        self._window: int | None = None

    def for_window(self, w: int) -> NDArray[np.float32]:
        """Return the accumulator for window 'w', resetting on a new window."""
        if w != self._window:
            self._accum[...] = 0
            self._window = w
        return self._accum

    @property
    def current_window(self) -> int | None:
        """Return the ToF window currently being accumulated, None before the first."""
        return self._window


def boundary_steps(spectrum: NDArray[np.float64], axis: ToFAxis, n_ref: int = 16, trim: int = 1) -> list[BoundaryStep]:
    """Discontinuity of a spectrum across each readout gap."""
    t = axis.t_start_s
    out: list[BoundaryStep] = []
    for w in range(len(axis.window_slices) - 1):
        a = axis.window_slices[w]
        b = axis.window_slices[w + 1]
        li = slice(a.stop - trim - n_ref, a.stop - trim)
        ri = slice(b.start + trim, b.start + trim + n_ref)
        left, right = spectrum[li], spectrum[ri]
        lm, rm = float(np.mean(left)), float(np.mean(right))

        extrap_pct = left_slope = right_slope = None

        if (left > 0).all() and (right > 0).all():
            t_mid = 0.5 * (t[a.stop - 1] + t[b.start])
            pl = np.polyfit(t[li] - t_mid, np.log(left), 1)
            pr = np.polyfit(t[ri] - t_mid, np.log(right), 1)

            extrap_pct = (float(np.exp(pr[1] - pl[1])) - 1.0) * 100.0
            left_slope = float(pl[0] * axis.bin_width_s) * 100.0
            right_slope = float(pr[0] * axis.bin_width_s) * 100.0

        out.append(
            BoundaryStep(
                gap=w,
                left_mean=lm,
                right_mean=rm,
                step_pct=(rm / lm - 1.0) * 100.0 if lm else float("nan"),
                extrap_pct=extrap_pct,
                left_slope_pct_per_bin=left_slope,
                right_slope_pct_per_bin=right_slope,
            )
        )
    return out
