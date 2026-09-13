"""Centre of rotation and axis tilt from the 0 deg / 180 deg mirror pair.

Coarse search over the full plausible range first, then sub-pixel refinement.
"""

import numpy as np
from numpy.typing import NDArray

from preprocess.core.constants import (
    FRAME_DIMENSIONS,
    MIN_BAND_ROWS,
    MIN_BANDS_FOR_TILT,
    MIN_PROJECTIONS_FOR_CENTRE,
    NEAR_ZERO,
)
from preprocess.core.preprocessing_types import AxisOrientation, CentreOfRotationFit, TiltFit


def _projected_extent_px(profile: NDArray[np.float64], fraction_of_peak: float = 0.5) -> float:
    """Return how many pixels of an attenuation profile lie above 'fraction_of_peak' of its peak."""
    p = np.asarray(profile, float)
    p = p - p.min()
    if p.max() <= 0:
        return 0.0
    above = np.flatnonzero(p >= fraction_of_peak * p.max())
    return float(above[-1] - above[0] + 1) if above.size else 0.0


def detect_axis_orientation(col_profiles: NDArray[np.float64], row_profiles: NDArray[np.float64]) -> AxisOrientation:
    """Decide the axis direction from how the projection changes with angle."""
    wx = np.array([_projected_extent_px(p) for p in np.asarray(col_profiles)])
    wy = np.array([_projected_extent_px(p) for p in np.asarray(row_profiles)])
    sx = float(wx.max() / max(wx.min(), 1.0) - 1.0)
    sy = float(wy.max() / max(wy.min(), 1.0) - 1.0)
    return AxisOrientation(horizontal=sy > sx, swing_x=sx, swing_y=sy)


def _profile(img: NDArray[np.float64]) -> NDArray[np.float64]:
    """Column profile."""
    a = np.asarray(img, dtype=np.float64)
    if a.ndim == FRAME_DIMENSIONS:
        a = np.nanmean(a, axis=0)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a - a.mean()


def _circular_correlation(
    a: NDArray[np.float64], b: NDArray[np.float64], lags: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Normalised cross-correlation of 'b' rolled by each lag against 'a', wrapping round the ends."""
    dots = np.array([float(np.dot(a, np.roll(b, int(lag)))) for lag in lags])
    return dots / (np.linalg.norm(a) * np.linalg.norm(b))


def _profile_shift_px(a: NDArray[np.float64], b: NDArray[np.float64], max_shift_px: int) -> tuple[float, float, bool]:
    """
    Return the shift of 'b' against 'a' that best aligns them, with its correlation and whether it saturated.

    The best integer lag by normalised cross-correlation, refined to sub-pixel by a parabola through the peak.
    """
    lags = np.arange(-max_shift_px, max_shift_px + 1)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0, 0.0, False
    cc = _circular_correlation(a, b, lags)

    j = int(np.argmax(cc))
    saturated = j in (0, len(cc) - 1)
    sub = 0.0
    if not saturated:
        d = cc[j - 1] - 2 * cc[j] + cc[j + 1]
        if abs(d) > NEAR_ZERO:
            sub = float(np.clip(0.5 * (cc[j - 1] - cc[j + 1]) / d, -1.0, 1.0))
    return float(lags[j] + sub), float(cc[j]), saturated


def centre_of_rotation_from_mirror_pair(
    p0: NDArray[np.float64], p180: NDArray[np.float64], max_shift: int | None = None, horizontal_axis: bool = False
) -> CentreOfRotationFit:
    """Centre of rotation from a 0/180 pair."""
    arr0, arr180 = np.asarray(p0), np.asarray(p180)
    if horizontal_axis:
        a = _profile(arr0.T)
        b = _profile(arr180[::-1].T)
    else:
        a = _profile(arr0)
        b = _profile(np.fliplr(arr180))
    n = a.size
    if max_shift is None:
        max_shift = n // 3

    shift, corr, sat = _profile_shift_px(a, b, max_shift)
    return CentreOfRotationFit(
        centre_of_rotation_px=(n - 1) / 2.0 + shift / 2.0, shift_px=shift, correlation=corr, n_cols=n, saturated=sat
    )


def tilt_from_bands(  # noqa: PLR0913, PLR0917
    p0: NDArray[np.float64],
    p180: NDArray[np.float64],
    n_bands: int = 8,
    min_correlation: float = 0.8,
    max_shift: int | None = None,
    horizontal_axis: bool = False,
    extent_frac: float = 0.25,
) -> TiltFit:
    """Fit the apparent mirror shift against position along the axis, the slope is the tilt."""
    p0 = np.asarray(p0)
    p180 = np.asarray(p180)

    # Position along the axis. columns for a horizontal axis, rows for a vertical one.
    along_axis_len = p0.shape[1] if horizontal_axis else p0.shape[0]
    along = np.nanmean(p0, axis=0 if horizontal_axis else 1)
    along = np.nan_to_num(along - np.nanmin(along), nan=0.0)
    if along.max() > 0:
        inside = np.flatnonzero(along >= extent_frac * along.max())
        lo_e, hi_e = int(inside[0]), int(inside[-1]) + 1
    else:
        lo_e, hi_e = 0, along_axis_len
    if hi_e - lo_e < 8 * n_bands:
        lo_e, hi_e = 0, along_axis_len
    edges = np.linspace(lo_e, hi_e, n_bands + 1).astype(int)

    pos, shifts, corrs = [], [], []
    for i in range(n_bands):
        lo, hi = edges[i], edges[i + 1]
        if hi - lo < MIN_BAND_ROWS:
            continue
        if horizontal_axis:
            sub0, sub180 = p0[:, lo:hi], p180[:, lo:hi]
        else:
            sub0, sub180 = p0[lo:hi, :], p180[lo:hi, :]
        fit = centre_of_rotation_from_mirror_pair(sub0, sub180, max_shift=max_shift, horizontal_axis=horizontal_axis)
        pos.append(0.5 * (lo + hi - 1))
        shifts.append(fit.shift_px)
        corrs.append(0.0 if fit.saturated else fit.correlation)

    pos = np.asarray(pos, float)
    shifts = np.asarray(shifts, float)
    corrs = np.asarray(corrs, float)
    used = corrs >= min_correlation
    if used.sum() < MIN_BANDS_FOR_TILT:
        raise ValueError(
            f"only {int(used.sum())} of {len(pos)} bands reached correlation {min_correlation}. Cannot fit a tilt."
        )

    m, c = np.polyfit(pos[used], shifts[used], 1)
    resid = float(np.sqrt(np.mean((shifts[used] - (m * pos[used] + c)) ** 2)))
    return TiltFit(
        slope_px_per_row=float(m),
        intercept_px=float(c),
        residual_px=resid,
        rows=pos,
        shifts=shifts,
        correlations=corrs,
        used=used,
    )


def centre_of_rotation_from_sinogram(
    sino: NDArray[np.float64], angles_deg: NDArray[np.float64]
) -> tuple[float, float, float]:
    """Independent centre-of-rotation estimate from the full angular sweep."""
    s = np.asarray(sino, dtype=np.float64)
    s = np.clip(s - np.nanmin(s), 0, None)
    x = np.arange(s.shape[1], dtype=np.float64)
    w = np.nansum(s, axis=1)
    good = w > 0
    com = np.full(s.shape[0], np.nan)
    com[good] = (s[good] @ x) / w[good]

    th = np.radians(np.asarray(angles_deg, dtype=np.float64))

    m = np.isfinite(com) & np.isfinite(th)
    if m.sum() < MIN_PROJECTIONS_FOR_CENTRE:
        raise ValueError(f"only {int(m.sum())} usable projections for a centre-of-mass fit")

    design_matrix = np.stack([np.ones(m.sum()), np.cos(th[m]), np.sin(th[m])], axis=1)
    coef, *_ = np.linalg.lstsq(design_matrix, com[m], rcond=None)
    resid = float(np.std(com[m] - design_matrix @ coef))
    return float(coef[0]), float(np.hypot(coef[1], coef[2])), resid


def truncation_fraction(sino: NDArray[np.float64], edge_px: int = 6, rel: float = 0.15) -> float:
    """Fraction of projections whose sample reaches a detector edge."""
    s = np.asarray(sino, dtype=np.float64)
    s = np.clip(s - np.nanmin(s), 0, None)
    peak = s.max(axis=1)
    left = s[:, :edge_px].mean(axis=1)
    right = s[:, -edge_px:].mean(axis=1)
    return float(np.mean((left > rel * peak) | (right > rel * peak)))
