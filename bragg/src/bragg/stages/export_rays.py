"""Export the ray data the tomography needs, into one compact file per axis."""

import json
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from shared.core.h5 import attr, dataset

from bragg.core.bragg_types import RayFit
from bragg.core.config import AxisSpec, BraggConfig
from bragg.core.constants import (
    AIR_PERCENTILE,
    ATTENUATION_BAND_A,
    ATTENUATION_BINS,
    DETECTOR_COLUMNS,
    DETECTOR_ROWS,
    MIN_TRANSMISSION,
    RAY_CHUNKS,
)

FIT_NAME_RE = re.compile(r"(\d+)_([^_]+)_")


def _read_calibration(path: Path) -> dict[str, Any]:
    """Return the per-experiment calibration entries."""
    if not path.is_file():
        logger.error(f"No calibration at {path}. ")
        raise SystemExit(1)
    return json.loads(path.read_text())["experiments"]


def _read_fits(fit_directory: Path) -> list[RayFit]:
    """Read every per-projection fit under 'fit_directory', sorted by angle, run and acquisition."""
    paths = sorted(fit_directory.glob("*.h5"))
    if not paths:
        logger.error(f"No GP fits in {fit_directory}.")
        raise SystemExit(1)
    fits = []
    for path in paths:
        m = FIT_NAME_RE.match(path.name)
        with h5py.File(path, "r") as f:
            y0, y1, x0, x1 = (int(v) for v in dataset(f, "bbox")[:])
            fits.append(
                RayFit(
                    angle_deg=float(attr(f, "angle_deg")),
                    run_number=int(attr(f, "run_number")),
                    acq=m.group(2) if m else "",
                    lambda0=float(attr(f, "lambda0")),
                    strain=dataset(f, "strain")[:],
                    sigma=dataset(f, "sigma")[:],
                    valid=dataset(f, "valid")[:].astype(bool),
                    bbox=(y0, y1, x0, x1),
                )
            )
    fits.sort(key=lambda r: (r.angle_deg, r.run_number, r.acq))
    return fits


def _place(fits: list[RayFit]) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.uint8]]:
    """Return the strain, sigma and validity of every fit placed back into the full detector frame."""
    shape = (len(fits), DETECTOR_ROWS, DETECTOR_COLUMNS)
    strain = np.zeros(shape, np.float32)
    sigma = np.zeros(shape, np.float32)
    valid = np.zeros(shape, np.uint8)
    for i, r in enumerate(fits):
        y0, y1, x0, x1 = r.bbox
        strain[i, y0:y1, x0:x1] = np.where(r.valid, r.strain, 0.0)
        sigma[i, y0:y1, x0:x1] = np.where(r.valid, r.sigma, 0.0)
        valid[i, y0:y1, x0:x1] = r.valid
    return strain, sigma, valid


def _attenuation_frames(experiment: Path) -> tuple[tuple[float, float], dict[tuple[int, str], NDArray[np.float32]]]:
    """Return the attenuation band actually used and one attenuation frame per (run, acq) of the experiment."""
    projections = sorted((experiment / "proj").glob("*.h5"))
    if not projections:
        logger.error(f"No projections under {experiment / 'proj'}")
        raise SystemExit(1)
    with h5py.File(projections[0], "r") as g0:
        lam = dataset(g0, "axis/lambda_a")[:]
    sl = np.nonzero((lam >= ATTENUATION_BAND_A[0]) & (lam <= ATTENUATION_BAND_A[1]))[0]
    lo, hi = int(sl[0]), min(int(sl[0]) + ATTENUATION_BINS, int(sl[-1]) + 1)
    frames = {}
    for path in projections:
        with h5py.File(path, "r") as f:
            key = (int(attr(f, "run_number")), str(f.attrs.get("acq", "")))
            t = np.nanmean(dataset(f, "T")[lo:hi], axis=0)
        good = np.isfinite(t) & (t > 0)
        air = np.percentile(t[good], AIR_PERCENTILE) if good.any() else 1.0
        frames[key] = np.where(good, -np.log(np.clip(t / air, MIN_TRANSMISSION, None)), 0.0).astype(np.float32)
    return (float(lam[lo]), float(lam[hi - 1])), frames


def _match_attenuation(
    fits: list[RayFit], frames: dict[tuple[int, str], NDArray[np.float32]], axis: str
) -> NDArray[np.float32]:
    """Return the attenuation frame of every fit, matched by (run, acq) so repeats stay distinct."""
    atten = np.zeros((len(fits), DETECTOR_ROWS, DETECTOR_COLUMNS), np.float32)
    missing = 0
    for i, r in enumerate(fits):
        frame = frames.get((r.run_number, r.acq))
        if frame is None:
            missing += 1
            continue
        atten[i] = np.clip(frame, 0, None)
    if missing:
        logger.error(
            f"{axis}: {missing}/{len(fits)} attenuation frames could not be matched to a fit. Fit keys "
            f"{sorted({(r.run_number, r.acq) for r in fits})[:3]} vs projection keys {sorted(frames)[:3]}"
        )
        raise SystemExit(1)
    return atten


def _export_axis(spec: AxisSpec, calibration: dict[str, Any], out: Path) -> None:
    """Write 'rays_<axis>.h5' for one rotation axis."""
    if spec.experiment.name not in calibration:
        logger.error(f"{spec.experiment.name} is not in the calibration; it has {sorted(calibration)}")
        raise SystemExit(1)
    cal = calibration[spec.experiment.name]
    fits = _read_fits(spec.fit_directory)
    strain, sigma, valid = _place(fits)
    band, frames = _attenuation_frames(spec.experiment)
    atten = _match_attenuation(fits, frames, spec.axis)

    with h5py.File(out, "w") as f:
        for k, v in (("strain", strain), ("sigma", sigma), ("valid", valid), ("attenuation", atten)):
            f.create_dataset(k, data=v, compression="lzf", chunks=RAY_CHUNKS)
        f.create_dataset("angle_deg", data=np.array([r.angle_deg for r in fits]))
        f.create_dataset("run_number", data=np.array([r.run_number for r in fits]))
        f.create_dataset("lambda0", data=np.array([r.lambda0 for r in fits]))
        f.attrs.update(
            axis=spec.axis,
            experiment=spec.experiment.name,
            n_projections=len(fits),
            pixel_um=cal["effective_pixel_um"],
            px_per_mm=cal["px_per_mm"],
            cor_row=cal["cor_row"],
            tilt_deg=cal["tilt_deg"],
            edge_lambda_a=float(np.mean([r.lambda0 for r in fits])),
            atten_band_a=list(band),
            note="strain is PATH-AVERAGED; multiply by geometric path "
            "length to get the ray transform If. Invalid rays are 0, "
            "never NaN. Rotation axis is HORIZONTAL: a slice is a "
            "detector COLUMN, the sinogram runs over ROWS.",
        )
    logger.info(
        f"{spec.axis}: {len(fits)} projections -> {out} ({out.stat().st_size / 2**20:.0f} MB); "
        f"{valid.sum() / 1e6:.2f}M valid rays, pixel {cal['effective_pixel_um']:.3f} um, "
        f"COR row {cal['cor_row']:.2f}, tilt {cal['tilt_deg']:+.4f} deg"
    )


def run(config: BraggConfig) -> bool:
    """Export the rays of every configured axis."""
    export = config.require_export()
    if not export.axes:
        logger.error("No 'export.axes' are configured, so there is nothing to export. ")
        return False
    calibration = _read_calibration(export.calibration)
    out_dir = config.chain.require_output_directory() / "rays"
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in export.axes:
        _export_axis(spec, calibration, out_dir / f"rays_{spec.axis}.h5")
    return True
