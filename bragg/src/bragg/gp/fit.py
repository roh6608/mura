"""Driving the GP fit over a projection, and over a whole scan."""

import json
import time
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import h5py
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from shared.core.backend import asnumpy, free_bytes, get_backend

from bragg.core.bragg_types import (
    Crop,
    FitMaps,
    FitResult,
    HyperFit,
    NoiseCheck,
    OptimiseOptions,
    Projection,
    ProjectionFit,
    ValidityCriteria,
)
from bragg.core.config import BraggConfig, Fit
from bragg.core.constants import (
    DRAW_SEED,
    KERNEL_KINDS,
    LIVE_CUBES,
    MAX_CONDITION_RATIO,
    MAX_MEMORY_FRACTION,
    TINY_DENOMINATOR,
    VALIDITY_OPEN_RADIUS_PX,
)
from bragg.edge import edges
from bragg.gp.kron import noise_from_differences
from bragg.gp.optimise import fit_sigmas, initial_guess_from_data, optimise
from bragg.io import cube as cubemod
from bragg.utils import plots


def fixed_hypers(fit: Fit, step_a: float) -> HyperFit | None:
    """Return the length scales the configuration pins, sigmas left to be fitted, or 'None' to fit them all."""
    if fit.tof_bins is None:
        return None
    return HyperFit(
        length_scales=(fit.tof_bins * step_a, fit.spatial_ls, fit.spatial_ls),
        sigma_f=np.nan,
        sigma_n=np.nan,
        nlml=np.nan,
        n_evals=0,
        converged=True,
    )


def check_noise(sigma_scalar: float, sigma_pointwise: NDArray[np.float64] | None, mad: float) -> NoiseCheck:
    """Compare the scalar noise the GP assumes against the propagated per-point sigma."""
    if sigma_pointwise is None:
        return NoiseCheck(ok=None, mad_first_difference=mad)
    s = sigma_pointwise[np.isfinite(sigma_pointwise) & (sigma_pointwise > 0)]
    if s.size == 0:
        logger.warning("no finite pointwise sigma in the crop, so the noise check cannot run")
        return NoiseCheck(ok=False, mad_first_difference=mad)
    med = float(np.median(s))
    lo, hi = (float(v) for v in np.percentile(s, [5.0, 95.0]))
    return NoiseCheck(
        ok=bool(hi / max(lo, TINY_DENOMINATOR) < MAX_CONDITION_RATIO),
        mad_first_difference=mad,
        scalar=float(sigma_scalar),
        pointwise_median=med,
        ratio_scalar_over_median=float(sigma_scalar) / med if med else float("nan"),
        pointwise_p5=lo,
        pointwise_p95=hi,
        spread=hi / max(lo, TINY_DENOMINATOR),
    )


def _backend_for(crop: Crop, device: str) -> tuple[ModuleType, str, type[np.floating[Any]]]:
    """Return the backend and working dtype, falling back to CPU when the fit would not fit."""
    xp, dev = get_backend(device)
    work_dtype = np.float32 if xp is not np else np.float64
    need = int(np.prod(crop.shape)) * np.dtype(work_dtype).itemsize * LIVE_CUBES
    have = free_bytes(xp, include_cpu=True)
    if have is not None and need > MAX_MEMORY_FRACTION * have:
        xp, dev = np, f"cpu (fit needs {need / 2**30:.1f} GiB, {have / 2**30:.1f} GiB free)"
        work_dtype = np.float64
    if dev != device:
        logger.warning(f"requested device {device!r} but running on {dev}")
    return xp, dev, work_dtype


def fit_projection(
    projection: Projection, fit: Fit, hypers: HyperFit | None = None, reference: float | None = None
) -> ProjectionFit:
    """Fit one projection: crop, GP, edge, strain."""
    t_start = time.time()
    crop = cubemod.load_crop(projection.path, fit.target, half_width_a=fit.half_width, n_bins=fit.bins)
    coords = crop.coords()
    xp, dev, work_dtype = _backend_for(crop, fit.device)
    options = OptimiseOptions(kinds=KERNEL_KINDS, xp=xp, work_dtype=work_dtype)

    y = xp.asarray(crop.cube, dtype=np.float64)
    offset = float(xp.mean(y))
    y = y - offset

    if hypers is None:
        hypers = optimise(y, coords, replace(options, init=initial_guess_from_data(y, coords)))
    elif not hypers.sigmas_fitted:
        hypers = fit_sigmas(y, coords, hypers.length_scales, options)
    gp = hypers.gp(coords, KERNEL_KINDS, xp=xp, work_dtype=work_dtype)

    noise = check_noise(gp.sigma_n, crop.sigma, noise_from_differences(y, axis=0))
    pos, _idx, curv_ok, boundary, amp = edges.edge_from_posterior(gp, y, axis=0)
    sigma, n_used = edges.edge_uncertainty(gp, y, n_draws=fit.draws, seed=DRAW_SEED, axis=0)

    seed_valid = xp.isfinite(pos) & (~boundary) & curv_ok
    valid = edges.validity_mask(
        pos,
        sigma,
        amp,
        bounds=(crop.lam[0], crop.lam[-1]),
        criteria=ValidityCriteria(
            sigma_max=fit.sigma_max_bins * crop.step_a,
            amplitude_min=edges.auto_amplitude_floor(amp, valid_seed=seed_valid),
            open_radius=VALIDITY_OPEN_RADIUS_PX,
            boundary=boundary,
            curvature_ok=curv_ok,
        ),
    )
    valid = xp.asarray(valid) & ~xp.asarray(crop.filled)

    ref = float(reference) if reference is not None else edges.reference_from_map(pos, valid)
    eps = edges.strain(pos, ref) if np.isfinite(ref) else xp.full_like(pos, np.nan)

    maps = FitMaps(
        position=asnumpy(pos),
        sigma=asnumpy(sigma),
        valid=asnumpy(valid),
        strain=asnumpy(eps),
        amplitude=asnumpy(amp),
        n_draws_used=asnumpy(n_used),
    )
    p, s, e = maps.position[maps.valid], maps.sigma[maps.valid], maps.strain[maps.valid]
    result = FitResult(
        run_number=crop.run_number,
        angle_deg=crop.angle_deg,
        bbox=crop.bbox,
        lam_slice=crop.lam_slice,
        window_id=crop.window_id,
        length_scales=tuple(float(v) for v in hypers.length_scales),
        sigma_f=float(hypers.sigma_f),
        sigma_n=float(hypers.sigma_n),
        nlml=float(gp.nlml(y)),
        converged=hypers.converged,
        n_valid=int(valid.sum()),
        n_pixels=int(pos.size),
        edge_median=float(np.median(p)) if p.size else float("nan"),
        edge_iqr=float(np.subtract(*np.percentile(p, [75, 25]))) if p.size else float("nan"),
        sigma_median=float(np.nanmedian(s)) if s.size else float("nan"),
        strain_reference=ref,
        strain_std=float(np.nanstd(e)) if e.size else float("nan"),
        noise_check=noise,
        seconds=time.time() - t_start,
        device=dev,
    )
    return ProjectionFit(result=result, maps=maps, crop=crop, gp=gp, y=y, offset=offset)


def _shared_hypers(reference: FitResult) -> HyperFit:
    """Return the reference projection's hyperparameters, to hold across the scan."""
    if not reference.converged:
        logger.warning("Reference hyperparameter search did not converge")
    return HyperFit(
        length_scales=reference.length_scales,
        sigma_f=reference.sigma_f,
        sigma_n=reference.sigma_n,
        nlml=reference.nlml,
        n_evals=0,
        converged=reference.converged,
    )


def _write_fit(out: Path, projection: Projection, fit: ProjectionFit, lam0: float, target_lambda: float) -> None:
    """Write one projection's maps and summary to 'out/<run>_<acq>_<angle>.h5'."""
    result, maps = fit.result, fit.maps
    name = f"{result.run_number}_{projection.acq}_{result.angle_deg:08.3f}"
    with h5py.File(out / f"{name}.h5", "w") as f:
        for k in ("position", "sigma", "strain", "amplitude"):
            f.create_dataset(k, data=getattr(maps, k).astype(np.float32), compression="lzf")
        f.create_dataset("valid", data=maps.valid.astype(np.uint8), compression="lzf")
        f.create_dataset("lam", data=fit.crop.lam)
        f.create_dataset("bbox", data=np.asarray(result.bbox, dtype=np.int32))
        for k, v in result.to_attrs().items():
            f.attrs[k] = v
        f.attrs["lambda0"] = lam0
        f.attrs["target_lambda"] = float(target_lambda)


def _log_fit(i: int, n: int, result: FitResult) -> None:
    """Log one line per fitted projection."""
    logger.info(
        f"[{i + 1:3d}/{n}] {result.angle_deg:8.3f} deg  "
        f"valid {result.n_valid:6d}/{result.n_pixels}  "
        f"edge {result.edge_median:.5f} A  sigma {result.sigma_median:.2e}  "
        f"strain sd {result.strain_std:.2e}  {result.seconds:.1f}s"
    )


def fit_experiment(config: BraggConfig, out: Path) -> list[FitResult]:
    """Fit every projection of the scan with one shared set of hyperparameters, writing each to 'out'."""
    fit, chain = config.fit, config.chain
    out.mkdir(parents=True, exist_ok=True)
    projections = cubemod.require_projections(chain.experiment, chain.limit)
    n = len(projections)

    ref_i = int(np.argmin([abs(p.angle_deg - chain.reference_angle_deg) for p in projections]))
    reference = projections[ref_i]
    logger.info(f"fitting hyperparameters on {reference.path.name} ({reference.angle_deg:.3f} deg)")
    step_a = cubemod.load_crop(
        reference.path, fit.target, half_width_a=fit.half_width, n_bins=fit.bins, with_sigma=False
    ).step_a
    pinned = fixed_hypers(fit, step_a)
    if pinned is not None:
        logger.info(f"ToF length scale fixed at {fit.tof_bins:g} bins = {pinned.length_scales[0]:.5f} A")
    ref_fit = fit_projection(reference, fit, hypers=pinned)

    try:
        plots.edge_fit_figure(
            ref_fit, out / "edge_fit.png", fit.target, title=f"{reference.path.name}  {reference.angle_deg:.3f} deg"
        )
    except Exception:
        logger.exception("could not draw the edge-fit figure; the fit itself is unaffected")
    shared = _shared_hypers(ref_fit.result)
    lam0 = ref_fit.result.strain_reference
    logger.info(
        f"length scales {tuple(round(v, 4) for v in shared.length_scales)}  "
        f"sigma_f {shared.sigma_f:.4g}  sigma_n {shared.sigma_n:.4g}  lambda0 {lam0:.5f} A"
    )
    _write_fit(out, reference, ref_fit, lam0, fit.target)
    results: dict[int, FitResult] = {ref_i: ref_fit.result}
    _log_fit(ref_i, n, ref_fit.result)
    del ref_fit

    for i, projection in enumerate(projections):
        if i == ref_i:
            continue
        if chain.per_projection_hypers:
            projection_fit = fit_projection(projection, fit, hypers=None, reference=None)
        else:
            projection_fit = fit_projection(projection, fit, hypers=shared, reference=lam0)
        _write_fit(out, projection, projection_fit, lam0, fit.target)
        results[i] = projection_fit.result
        _log_fit(i, n, projection_fit.result)
        del projection_fit

    ordered = [results[i] for i in range(n)]
    (out / "fit_report.json").write_text(json.dumps([r.to_json_dict() for r in ordered], indent=2, default=float))
    return ordered
