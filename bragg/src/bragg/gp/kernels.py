"""Stationary kernels for the separable Gaussian process."""

from collections.abc import Callable
from types import ModuleType

import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_coeffs

SQRT3 = np.sqrt(3.0)
SQRT5 = np.sqrt(5.0)


def _dist(coords: NDArray[np.float64], length_scale: float, xp: ModuleType) -> NDArray[np.float64]:
    """Pairwise absolute distance along one axis, in units of the length scale."""
    c = xp.asarray(coords, dtype=xp.float64).ravel()
    return xp.abs(c[:, None] - c[None, :]) / float(length_scale)


def rbf(coords: NDArray[np.float64], length_scale: float, xp: ModuleType = np) -> NDArray[np.float64]:
    """Squared exponential. Analytic paths -- rings against a step."""
    r = _dist(coords, length_scale, xp)
    return xp.exp(-0.5 * r * r)


def matern32(coords: NDArray[np.float64], length_scale: float, xp: ModuleType = np) -> NDArray[np.float64]:
    """Matern nu=3/2."""
    r = SQRT3 * _dist(coords, length_scale, xp)
    return (1.0 + r) * xp.exp(-r)


def matern52(coords: NDArray[np.float64], length_scale: float, xp: ModuleType = np) -> NDArray[np.float64]:
    """Matern nu=5/2."""
    r = SQRT5 * _dist(coords, length_scale, xp)
    return (1.0 + r + r * r / 3.0) * xp.exp(-r)


KERNELS: dict[str, Callable[..., NDArray[np.float64]]] = {"rbf": rbf, "matern32": matern32, "matern52": matern52}
DEFAULT_TOF_KERNEL: str = "matern52"
DEFAULT_SPATIAL_KERNEL: str = "matern32"


def gram(
    coords: NDArray[np.float64], length_scale: float, kind: str = "matern52", jitter: float = 1e-8, xp: ModuleType = np
) -> NDArray[np.float64]:
    """Gram matrix of one axis, with a jitter for a "safe" eigendecomposition."""
    if kind not in KERNELS:
        raise ValueError(f"unknown kernel {kind!r}; choose from {sorted(KERNELS)}")
    k = KERNELS[kind](coords, length_scale, xp=xp)
    n = k.shape[0]
    return k + float(jitter) * xp.eye(n, dtype=k.dtype)


def savgol_derivative_matrix(
    coords: NDArray[np.float64], window: int = 7, polyorder: int = 3, xp: ModuleType = np
) -> NDArray[np.float64]:
    """Matrix 'D' with 'D @ y' the smoothed first derivative of 'y'."""
    c = np.asarray(coords, dtype=np.float64).ravel()
    n = c.size
    if n < window:
        raise ValueError(f"axis of length {n} is too short for a {window}-point stencil")
    if polyorder >= window:
        raise ValueError("polyorder must be smaller than the window length")
    step = float(np.median(np.diff(c)))
    if not np.isfinite(step) or step == 0:
        raise ValueError("cannot build a derivative matrix on a non-uniform axis")

    half = window // 2
    coefficients = [savgol_coeffs(window, polyorder, deriv=1, delta=step, pos=pos, use="dot") for pos in range(window)]
    d = np.zeros((n, n), dtype=np.float64)
    centre = coefficients[half]
    for i in range(n):
        if i < half:
            d[i, :window] = coefficients[i]
        elif i >= n - half:
            d[i, n - window :] = coefficients[window - (n - i)]
        else:
            d[i, i - half : i - half + window] = centre
    return xp.asarray(d)
