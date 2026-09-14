"""Hyperparameter selection by exact marginal likelihood."""

from collections.abc import Sequence
from types import ModuleType

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from shared.core.backend import asnumpy

from bragg.core.bragg_types import HyperFit, OptimiseOptions
from bragg.core.constants import (
    BOUND_SPANS,
    BOUND_STEPS,
    CACHE_KEY_DECIMALS,
    ETA_BOUNDS,
    GOLDEN_TOLERANCE,
    INIT_SPAN_FRACTION,
    INIT_STEPS,
    KERNEL_KINDS,
    LOWER_BOUND_FACTOR,
    MAX_AUTOCORRELATION_LAGS,
    NELDER_MEAD_FATOL,
    NELDER_MEAD_MAX_ITER,
    NELDER_MEAD_XATOL,
    UPPER_BOUND_FACTOR,
)
from bragg.gp.kron import KroneckerGP

GOLDEN: float = float(0.5 * (3.0 - np.sqrt(5.0)))


def profile_nlml(
    yt2: NDArray[np.float64], lam: NDArray[np.float64], eta: float, n: int, xp: ModuleType = np
) -> tuple[float, float]:
    """NLML with the amplitude profiled out, plus the amplitude it implies."""
    denom = lam + eta
    sf2 = float(xp.sum(yt2 / denom)) / n
    if not np.isfinite(sf2) or sf2 <= 0:
        return np.inf, np.nan
    val = 0.5 * (n + n * np.log(sf2) + float(xp.sum(xp.log(denom))) + n * np.log(2 * np.pi))
    return float(val), sf2


def _golden_eta(
    yt2: NDArray[np.float64], lam: NDArray[np.float64], n: int, xp: ModuleType = np
) -> tuple[float, float, float, int]:
    """Golden-section search over log(eta)."""
    lo, hi = float(np.log(ETA_BOUNDS[0])), float(np.log(ETA_BOUNDS[1]))
    c, d = lo + GOLDEN * (hi - lo), hi - GOLDEN * (hi - lo)
    fc, _ = profile_nlml(yt2, lam, float(np.exp(c)), n, xp)
    fd, _ = profile_nlml(yt2, lam, float(np.exp(d)), n, xp)
    evals = 2
    while hi - lo > GOLDEN_TOLERANCE:
        if fc < fd:
            hi, d, fd = d, c, fc
            c = lo + GOLDEN * (hi - lo)
            fc, _ = profile_nlml(yt2, lam, float(np.exp(c)), n, xp)
        else:
            lo, c, fc = c, d, fd
            d = hi - GOLDEN * (hi - lo)
            fd, _ = profile_nlml(yt2, lam, float(np.exp(d)), n, xp)
        evals += 1
    eta = float(np.exp(0.5 * (lo + hi)))
    val, sf2 = profile_nlml(yt2, lam, eta, n, xp)
    return eta, val, sf2, evals


def _centred(y: NDArray[np.float64], xp: ModuleType) -> NDArray[np.float64]:
    """Return 'y' on 'xp' with its mean removed, refusing a grid with holes in it."""
    y = xp.asarray(y)
    if not bool(xp.all(xp.isfinite(y))):
        raise ValueError("cube contains non-finite values; the exact solve needs a complete grid")
    return y - xp.mean(y)


def optimise(
    y: NDArray[np.float64], coords: Sequence[NDArray[np.float64]], options: OptimiseOptions | None = None
) -> HyperFit:
    """Maximise the exact marginal likelihood over length scales and noise."""
    options = options or OptimiseOptions()
    xp, work_dtype = options.xp, options.work_dtype
    kinds = options.kinds or KERNEL_KINDS[: len(coords)]
    coords = tuple(coords)

    y = _centred(y, xp)
    n = int(y.size)
    spans = [float(np.ptp(np.asarray(c))) for c in coords]
    steps = [float(np.median(np.diff(np.asarray(c).ravel()))) for c in coords]
    init = options.init or tuple(
        max(INIT_STEPS * st, INIT_SPAN_FRACTION * sp) for st, sp in zip(steps, spans, strict=True)
    )
    bounds = [(BOUND_STEPS * st, BOUND_SPANS * sp) for st, sp in zip(steps, spans, strict=True)]

    cache: dict[tuple[float, ...], tuple[float, float, float]] = {}
    n_evals = 0

    def evaluate(log_ls: NDArray[np.float64]) -> tuple[float, float, float]:
        nonlocal n_evals
        key = tuple(round(v, CACHE_KEY_DECIMALS) for v in log_ls)
        if key in cache:
            return cache[key]
        ls = tuple(float(np.exp(v)) for v in log_ls)
        gp = KroneckerGP(coords=coords, length_scales=ls, kinds=kinds, xp=xp, work_dtype=work_dtype)
        yt = gp.transform(y)
        eta, val, sf2, ev = _golden_eta(yt * yt, gp.eigenvalue_spectrum, n, xp=xp)
        n_evals += ev
        out = (val, float(np.sqrt(sf2)), float(np.sqrt(sf2 * eta)))
        cache[key] = out
        return out

    lo = np.log([b[0] for b in bounds])
    hi = np.log([b[1] for b in bounds])

    def objective(v: NDArray[np.float64]) -> float:
        return evaluate(np.clip(v, lo, hi))[0]

    x0 = np.log(np.clip(init, [b[0] for b in bounds], [b[1] for b in bounds]))
    res = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={"xatol": NELDER_MEAD_XATOL, "fatol": NELDER_MEAD_FATOL, "maxiter": NELDER_MEAD_MAX_ITER},
    )
    best = np.clip(res.x, lo, hi)
    val, sf, sn = evaluate(best)
    ls = tuple(float(np.exp(v)) for v in best)

    at_bound = tuple(
        i
        for i, (v, b) in enumerate(zip(ls, bounds, strict=True))
        if v <= b[0] * LOWER_BOUND_FACTOR or v >= b[1] * UPPER_BOUND_FACTOR
    )
    return HyperFit(
        length_scales=ls,
        sigma_f=sf,
        sigma_n=sn,
        nlml=val,
        n_evals=n_evals,
        converged=bool(res.success) and not at_bound,
        at_bound=at_bound,
    )


def initial_guess_from_data(y: NDArray[np.float64], coords: Sequence[NDArray[np.float64]]) -> tuple[float, ...]:
    """Return a starting point taken from the data rather than from a constant.

    Uses the lag at which the autocorrelation along each axis first falls below '1/e', the
    definition of a correlation length, so Nelder-Mead starts near the answer.
    """
    guesses = []
    host = asnumpy(y)
    for ax, c in enumerate(coords):
        step = float(np.median(np.diff(np.asarray(c).ravel())))
        v = np.moveaxis(host, ax, 0).reshape(host.shape[ax], -1)
        v = v - v.mean(axis=0, keepdims=True)
        denom = (v * v).sum(axis=0)
        keep = denom > 0
        if not keep.any():
            guesses.append(INIT_STEPS * step)
            continue
        v, denom = v[:, keep], denom[keep]
        n_lag = min(v.shape[0] - 1, MAX_AUTOCORRELATION_LAGS)
        ac = np.array([(v[: v.shape[0] - k] * v[k:]).sum(axis=0).sum() / denom.sum() for k in range(n_lag)])
        below = np.nonzero(ac < 1.0 / np.e)[0]
        lag = float(below[0]) if below.size else float(n_lag)
        guesses.append(max(lag * step, BOUND_STEPS * step))
    return tuple(guesses)


def fit_sigmas(
    y: NDArray[np.float64],
    coords: Sequence[NDArray[np.float64]],
    length_scales: Sequence[float],
    options: OptimiseOptions | None = None,
) -> HyperFit:
    """Amplitude and noise at fixed length scales, by the same profiling."""
    options = options or OptimiseOptions()
    xp = options.xp
    kinds = options.kinds or KERNEL_KINDS[: len(coords)]
    y = _centred(y, xp)
    gp = KroneckerGP(
        coords=tuple(coords), length_scales=tuple(length_scales), kinds=kinds, xp=xp, work_dtype=options.work_dtype
    )
    yt = gp.transform(y)
    eta, val, sf2, evals = _golden_eta(yt * yt, gp.eigenvalue_spectrum, int(y.size), xp=xp)
    return HyperFit(
        length_scales=tuple(float(v) for v in length_scales),
        sigma_f=float(np.sqrt(sf2)),
        sigma_n=float(np.sqrt(sf2 * eta)),
        nlml=val,
        n_evals=evals,
        converged=True,
    )
