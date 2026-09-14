"""Getting a fittable cube out of the preprocessed store."""

from os import PathLike
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from shared.core.h5 import dataset

from bragg.core.bragg_types import Crop, Projection
from bragg.core.constants import (
    BBOX_CLOSE_KERNEL_PX,
    BBOX_MIN_SIZE_PX,
    BBOX_OPEN_KERNEL_PX,
    BBOX_PAD_PX,
    BBOX_PERCENTILES,
    BBOX_PROBE_BINS,
    BBOX_QUANTILE,
    DEFAULT_CROP_BINS,
    FILL_KSIZE,
    FILL_PASSES,
    FRAME_NDIM,
    MIN_BAND_ROWS,
    SMALL_DENOMINATOR,
)


def window_slice(
    window_id: NDArray[np.int64],
    lam: NDArray[np.float64],
    target_lambda: float,
    half_width_a: float | None = None,
    n_bins: int | None = None,
) -> tuple[slice, int]:
    """Bins around 'target_lambda' that stay inside a single shutter window."""
    lam = np.asarray(lam)
    wid = np.asarray(window_id)
    centre = int(np.argmin(np.abs(lam - float(target_lambda))))
    w = int(wid[centre])
    same = np.nonzero(wid == w)[0]
    w_lo, w_hi = int(same[0]), int(same[-1]) + 1

    if half_width_a is not None:
        lo = int(np.searchsorted(lam, float(target_lambda) - float(half_width_a), "left"))
        hi = int(np.searchsorted(lam, float(target_lambda) + float(half_width_a), "right"))
    else:
        half = int(n_bins or DEFAULT_CROP_BINS) // 2
        lo, hi = centre - half, centre + half
    lo, hi = max(lo, w_lo), min(hi, w_hi)
    if hi - lo < MIN_BAND_ROWS:
        raise ValueError(
            f"only {hi - lo} bins of window {w} lie within the requested band around "
            f"{target_lambda} A; widen half_width_a or pick an edge further from a gap"
        )
    return slice(lo, hi), w


def sample_bbox(white: NDArray[np.float64], mask: NDArray[np.integer[Any]] | None) -> tuple[int, int, int, int]:
    """Bounding box of the attenuating object in a white-beam image."""
    w = np.asarray(white, dtype=np.float32)
    good = np.isfinite(w)
    if mask is not None:
        good &= np.asarray(mask) == 0
    if not good.any():
        raise ValueError("no valid pixels in the white-beam image")
    v = w[good]
    lo = float(np.percentile(v, BBOX_PERCENTILES[0]))  # through the thickest part
    hi = float(np.percentile(v, BBOX_PERCENTILES[1]))  # open beam
    thresh = lo + BBOX_QUANTILE * (hi - lo)
    m = ((w < thresh) & good).astype(np.uint8)
    # close first, across the 2-pixel chip seams, then open to drop speckle
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((BBOX_CLOSE_KERNEL_PX, BBOX_CLOSE_KERNEL_PX), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((BBOX_OPEN_KERNEL_PX, BBOX_OPEN_KERNEL_PX), np.uint8))
    if not m.any():
        raise ValueError("sample threshold selected no pixels")
    n_lab, _lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) if n_lab > 1 else 0
    x, y, bw, bh = (
        int(stats[k, i]) for i in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT)
    )
    y0, y1 = max(0, y - BBOX_PAD_PX), min(w.shape[0], y + bh + BBOX_PAD_PX)
    x0, x1 = max(0, x - BBOX_PAD_PX), min(w.shape[1], x + bw + BBOX_PAD_PX)
    if (y1 - y0) < BBOX_MIN_SIZE_PX or (x1 - x0) < BBOX_MIN_SIZE_PX:
        raise ValueError(f"bounding box {(y0, y1, x0, x1)} is too small to fit")
    return int(y0), int(y1), int(x0), int(x1)


def fill_invalid(
    cube: NDArray[np.float32], invalid: NDArray[np.bool_]
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """
    Normalised convolution to fill holes with a weighted mean of valid neighbours.

    This should really be moved into the preprocessing module.
    """
    out = np.array(cube, dtype=np.float32, copy=True)
    bad = np.asarray(invalid)
    if bad.ndim == FRAME_NDIM:
        bad = np.broadcast_to(bad, out.shape)
    still = bad.copy()
    k = (FILL_KSIZE, FILL_KSIZE)
    for i in range(out.shape[0]):
        b = still[i]
        if not b.any():
            continue
        v = out[i].copy()
        w = (~b).astype(np.float32)
        v[b] = 0.0
        for _ in range(FILL_PASSES):
            num = cv2.blur(v, k)
            den = cv2.blur(w, k)
            ok = den > SMALL_DENOMINATOR
            filled = np.where(ok, num / np.where(ok, den, 1.0), 0.0)
            newly = b & ok
            v = np.where(newly, filled, v)
            w = np.where(newly, 1.0, w)
            b = b & ~ok
            if not b.any():
                break
        out[i] = v
        still[i] = b
    return out, still


def load_crop(
    path: str | PathLike[str],
    target_lambda: float,
    half_width_a: float | None = None,
    n_bins: int | None = None,
    with_sigma: bool = True,
) -> Crop:
    """Read one projection, cropped in wavelength and space, as a complete float64 grid."""
    with h5py.File(path, "r") as f:
        lam_all = dataset(f, "axis/lambda_a")[:]
        wid = dataset(f, "axis/window_id")[:]
        tof_all = dataset(f, "axis/tof_s")[:]
        sl, w = window_slice(wid, lam_all, target_lambda, half_width_a, n_bins)

        probe = dataset(f, "T")[sl.start : sl.stop : max(1, (sl.stop - sl.start) // BBOX_PROBE_BINS)]
        white = np.nanmean(probe, axis=0)
        bbox = sample_bbox(white, dataset(f, "mask")[:] if "mask" in f else None)
        y0, y1, x0, x1 = bbox
        cube = np.asarray(dataset(f, "T")[sl, y0:y1, x0:x1], dtype=np.float32)
        sigma = None
        if with_sigma and "varT" in f:
            sigma = np.sqrt(np.maximum(np.asarray(dataset(f, "varT")[sl, y0:y1, x0:x1], dtype=np.float32), 0.0))
        det_mask = dataset(f, "mask")[y0:y1, x0:x1] if "mask" in f else None
        angle_deg = float(f.attrs.get("angle_deg", np.nan))
        run_number = int(f.attrs.get("run_number", -1))

    invalid = ~np.isfinite(cube)
    if det_mask is not None:
        invalid |= np.broadcast_to(np.asarray(det_mask) != 0, cube.shape)
    cube, still = fill_invalid(cube, invalid)
    if still.any():
        cube = np.where(still, float(np.nanmedian(cube[~still])), cube)
    if sigma is not None:
        sigma = np.where(invalid, np.nan, sigma)

    return Crop(
        cube=cube.astype(np.float64),
        lam=lam_all[sl],
        tof=tof_all[sl],
        sigma=sigma,
        bbox=bbox,
        lam_slice=(int(sl.start), int(sl.stop)),
        window_id=int(w),
        filled=invalid.any(axis=0),
        angle_deg=angle_deg,
        run_number=run_number,
    )


def list_projections(experiment_dir: str | PathLike[str]) -> list[Projection]:
    """Projection files with their angles, sorted by angle."""
    out = []
    for p in sorted((Path(experiment_dir) / "proj").glob("*.h5")):
        with h5py.File(p, "r") as f:
            out.append(
                Projection(
                    path=p,
                    angle_deg=float(f.attrs.get("angle_deg", np.nan)),
                    run_number=int(f.attrs.get("run_number", -1)),
                    acq=str(f.attrs.get("acq", "")),
                )
            )
    out.sort(key=lambda r: r.angle_deg)
    return out


def require_projections(experiment_dir: str | PathLike[str], limit: int | None = None) -> list[Projection]:
    """Return the projections of an experiment."""
    projections = list(list_projections(experiment_dir))
    if limit is not None:
        projections = projections[:limit]
    if not projections:
        logger.error(
            f"No projections to read under {Path(experiment_dir) / 'proj'}"
            + (f" with 'chain.limit' {limit}" if limit == 0 else "")
            + ". Point 'chain.experiment' at a processed experiment directory, or run "
            "'preprocess' to create one."
        )
        raise SystemExit(1)
    return projections
