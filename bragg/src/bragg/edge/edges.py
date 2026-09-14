"""Locating a Bragg edge in the GP posterior, and turning it into strain."""

import cv2
import numpy as np
from numpy.typing import NDArray
from shared.core.backend import array_module, asnumpy

from bragg.core.bragg_types import ValidityCriteria
from bragg.core.constants import (
    AMPLITUDE_FLOOR_FACTOR,
    AMPLITUDE_FLOOR_QUANTILE,
    DERIVATIVE_POLYORDER,
    DERIVATIVE_WINDOW,
    MIN_DRAW_FRACTION,
    MIN_DRAWS_FOR_SIGMA,
    TINY_DENOMINATOR,
)
from bragg.gp.kron import KroneckerGP


def argmax_subbin(
    deriv: NDArray[np.float64], axis: int = 0, coords: NDArray[np.float64] | None = None
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.bool_], NDArray[np.bool_]]:
    """Argmax along one axis with parabolic refinement."""
    xp = array_module(deriv)
    n = deriv.shape[axis]
    idx = xp.argmax(deriv, axis=axis)
    on_boundary = (idx == 0) | (idx == n - 1)
    safe = xp.clip(idx, 1, n - 2)

    def at(offset: int) -> NDArray[np.float64]:
        return xp.take_along_axis(deriv, xp.expand_dims(safe + offset, axis), axis).squeeze(axis)

    fm, f0, fp = at(-1), at(0), at(1)
    denom = fm - 2.0 * f0 + fp
    curvature_ok = denom < 0  # a maximum must be concave
    delta = xp.where(xp.abs(denom) > TINY_DENOMINATOR, 0.5 * (fm - fp) / xp.where(denom == 0, 1.0, denom), 0.0)
    delta = xp.clip(delta, -1.0, 1.0)
    frac = xp.where(on_boundary, idx.astype(delta.dtype), safe + delta)

    if coords is None:
        return frac, idx, curvature_ok, on_boundary
    c = xp.asarray(coords, dtype=frac.dtype).ravel()
    lo = xp.clip(xp.floor(frac).astype(xp.int64), 0, n - 2)
    w = frac - lo
    pos = c[lo] * (1.0 - w) + c[lo + 1] * w
    return pos, idx, curvature_ok, on_boundary


def edge_from_posterior(
    gp: KroneckerGP, y: NDArray[np.float64], axis: int = 0
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.bool_], NDArray[np.bool_], NDArray[np.float64]]:
    """Edge position from the argmax of the posterior ToF derivative, plus the peak height."""
    deriv = gp.posterior_derivative(y, axis=axis)
    pos, idx, ok, boundary = argmax_subbin(deriv, axis=axis, coords=gp.coords[axis])
    xp = array_module(deriv)
    amp = xp.max(deriv, axis=axis)
    return pos, idx, ok, boundary, amp


def edge_uncertainty(
    gp: KroneckerGP, y: NDArray[np.float64], n_draws: int, seed: int, axis: int = 0
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Uncertainty on the edge position from posterior draws, and the draws each pixel kept."""
    xp = gp.xp
    mats = gp.derivative_mats(axis, DERIVATIVE_WINDOW, DERIVATIVE_POLYORDER)
    coords = gp.coords[axis]

    total = None
    total2 = None
    count = None
    for draw in gp.sample(y, n_draws=n_draws, seed=seed, mats=mats):
        pos, _, ok, boundary = argmax_subbin(draw, axis=axis, coords=coords)
        good = (~boundary) & ok
        z = xp.where(good, pos, 0.0)
        total = z if total is None else total + z
        total2 = z * z if total2 is None else total2 + z * z
        c = good.astype(xp.float64)
        count = c if count is None else count + c

    if total is None or total2 is None or count is None:
        raise ValueError("n_draws must be at least 1 to estimate an edge uncertainty")

    enough = count >= max(MIN_DRAWS_FOR_SIGMA, int(n_draws * MIN_DRAW_FRACTION))
    denom = xp.where(enough, count, 1.0)
    mean = total / denom
    var = xp.maximum(total2 / denom - mean * mean, 0.0)
    sigma = xp.where(enough, xp.sqrt(var * denom / xp.maximum(denom - 1.0, 1.0)), xp.nan)
    return sigma, count


def validity_mask(
    position: NDArray[np.float64],
    sigma: NDArray[np.float64],
    amplitude: NDArray[np.float64],
    bounds: tuple[float, float],
    criteria: ValidityCriteria | None = None,
) -> NDArray[np.bool_]:
    """Which pixels carry a real edge measurement."""
    criteria = criteria or ValidityCriteria()
    xp = array_module(position)
    pos = asnumpy(position)
    sig = asnumpy(sigma)
    amp = asnumpy(amplitude)

    lo, hi = float(bounds[0]), float(bounds[1])
    if hi < lo:
        lo, hi = hi, lo
    ok = np.isfinite(pos) & (pos > lo) & (pos < hi)
    ok &= np.isfinite(sig)
    if criteria.boundary is not None:
        ok &= ~asnumpy(criteria.boundary).astype(bool)
    if criteria.curvature_ok is not None:
        ok &= asnumpy(criteria.curvature_ok).astype(bool)
    if criteria.amplitude_min is not None:
        ok &= np.isfinite(amp) & (amp >= float(criteria.amplitude_min))
    if criteria.sigma_max is not None:
        ok &= sig <= float(criteria.sigma_max)

    m = ok.astype(np.uint8)
    if criteria.open_radius > 0:
        radius = criteria.open_radius
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if criteria.largest_component and m.any():
        n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n_lab > 1:
            biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            m = (lab == biggest).astype(np.uint8)
    return xp.asarray(m.astype(bool))


def auto_amplitude_floor(amplitude: NDArray[np.float64], valid_seed: NDArray[np.bool_]) -> float | None:
    """Return a data-driven floor for the derivative peak height."""
    xp = array_module(amplitude)
    a = amplitude[valid_seed]
    a = a[xp.isfinite(a)]
    if a.size == 0:
        return None
    return float(xp.quantile(a, AMPLITUDE_FLOOR_QUANTILE)) * AMPLITUDE_FLOOR_FACTOR


def strain(position: NDArray[np.float64], reference: float) -> NDArray[np.float64]:
    """Engineering strain from an edge shift, '(lambda - lambda0)/lambda0'."""
    ref = float(reference)
    if ref <= 0:
        raise ValueError("reference wavelength must be positive")
    return (position - ref) / ref


def reference_from_map(position: NDArray[np.float64], valid: NDArray[np.bool_]) -> float:
    """Return an internal, unstrained reference: the median edge position over the valid pixels."""
    xp = array_module(position)
    v = position[valid]
    v = v[xp.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(xp.median(v))
