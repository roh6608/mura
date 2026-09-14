"""Centre of rotation and rotation-axis tilt."""

import json
import warnings
from pathlib import Path

import h5py
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from shared.core.h5 import dataset

from preprocess.calibrate import geometry
from preprocess.core.config import Geometry, PreprocessConfig
from preprocess.core.constants import GEOMETRY_NAME, INDEX_NAME
from preprocess.core.preprocessing_types import AxisOrientation, CentreOfMass, CentreOfRotationFit, Profiles, TiltFit
from preprocess.utils import plots


def _wavelength_band(index: Path, settings: Geometry) -> slice:
    """Return the ToF bins inside the configured wavelength band."""
    with h5py.File(index, "r") as f:
        lam = dataset(f, "axis/lambda_a")[:]

    sel = np.flatnonzero((lam >= settings.lambda_low_a) & (lam <= settings.lambda_high_a))
    if sel.size == 0:
        raise SystemExit(1)
    lo, hi = int(sel[0]), int(sel[-1]) + 1
    return slice(lo, hi)


def _attenuation(transmission: NDArray[np.float32]) -> NDArray[np.float64]:
    """Return '-ln T' averaged over the band."""
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        a = -np.log(np.clip(np.nanmean(transmission, axis=0), 1e-6, None))
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(a - np.percentile(a, 2), 0, None)


def _read_profiles(index: Path, band: slice) -> Profiles:
    """Reduce every projection in the index to an attenuation image."""
    with h5py.File(index, "r") as f:
        angles = dataset(f, "angle_deg")[:]
        runs = dataset(f, "run_number")[:]
        logger.info(f"{angles.size} projections, angles {np.nanmin(angles):.3f} to {np.nanmax(angles):.3f} deg")

        order = np.argsort(angles)
        logger.info("reading projections ...")
        attenuation = [_attenuation(dataset(f, "T")[int(i), band, :, :]) for i in order]

    return Profiles(
        angles_deg=angles[order],
        runs=runs[order],
        attenuation=attenuation,
        cols=np.array([a.mean(axis=0) for a in attenuation]),
        rows=np.array([a.mean(axis=1) for a in attenuation]),
    )


def _fit_mirror_pair(
    profiles: Profiles, i0: int, i180: int, orient: AxisOrientation, settings: Geometry
) -> CentreOfRotationFit:
    """Fit the centre of rotation from the 0/180 pair, and confirm the other axis direction carries none."""
    a0, a180 = profiles.attenuation[i0], profiles.attenuation[i180]

    return geometry.centre_of_rotation_from_mirror_pair(
        a0, a180, max_shift=settings.max_shift, horizontal_axis=orient.horizontal
    )


def _fit_tilt(profiles: Profiles, i0: int, i180: int, orient: AxisOrientation, settings: Geometry) -> TiltFit | None:
    """Fit the axis tilt from per-band mirror shifts."""
    try:
        tilt = geometry.tilt_from_bands(
            profiles.attenuation[i0],
            profiles.attenuation[i180],
            n_bands=settings.bands,
            min_correlation=settings.min_correlation,
            max_shift=settings.max_shift,
            horizontal_axis=orient.horizontal,
        )
    except ValueError as exc:
        logger.warning(f"tilt: not fitted ({exc})")
        return None

    return tilt


def _centre_of_mass(profiles: Profiles, orient: AxisOrientation) -> CentreOfMass:
    """Estimate the centre of rotation from the sinogram's centroid, as a check on the mirror pair."""
    sino = profiles.rows if orient.centre_of_rotation_is_row else profiles.cols
    trunc = geometry.truncation_fraction(sino)
    centre, amplitude, residual = geometry.centre_of_rotation_from_sinogram(sino, profiles.angles_deg)

    return CentreOfMass(centre_px=centre, amplitude_px=amplitude, residual_px=residual, truncation_fraction=trunc)


def _verdict(mirror: CentreOfRotationFit, com: CentreOfMass, settings: Geometry) -> bool:
    """Decide whether the mirror-pair centre stands, by the centre-of-mass cross-check when that is usable."""
    d = abs(com.centre_px - mirror.centre_of_rotation_px)

    if com.amplitude_px < settings.min_amplitude:
        ok = mirror.correlation >= settings.min_correlation and not mirror.saturated
    else:
        ok = d < settings.tolerance_px and not mirror.saturated
    return ok


def _payload(  # noqa: PLR0913, PLR0917
    experiment: str,
    profiles: Profiles,
    settings: Geometry,
    orient: AxisOrientation,
    com: CentreOfMass,
    mirror: CentreOfRotationFit | None,
    tilt: TiltFit | None,
) -> dict:
    """Assemble 'geometry.json'. The keys are the stored product's schema; do not rename them."""
    ny, nx = profiles.shape
    payload = {
        "experiment": experiment,
        "n_projections": int(profiles.n),
        "lambda_lo": settings.lambda_low_a,
        "lambda_hi": settings.lambda_high_a,
        "axis_horizontal": bool(orient.horizontal),
        "cor_is_row": bool(orient.centre_of_rotation_is_row),
        "swing_x": orient.swing_x,
        "swing_y": orient.swing_y,
        "cor_centre_of_mass": com.centre_px,
        "com_amplitude_px": com.amplitude_px,
        "com_residual_px": com.residual_px,
        "truncation_fraction": com.truncation_fraction,
        "detector_centre_px": ((ny if orient.centre_of_rotation_is_row else nx) - 1) / 2.0,
    }
    if mirror is not None:
        payload.update(
            {
                "cor_px": mirror.centre_of_rotation_px,
                "shift_px": mirror.shift_px,
                "correlation": mirror.correlation,
                "saturated": mirror.saturated,
            }
        )
    if tilt is not None:
        payload.update(
            {"tilt_deg": tilt.tilt_deg, "tilt_slope": tilt.slope_px_per_row, "tilt_residual_px": tilt.residual_px}
        )
    return payload


def _write(destination: Path, payload: dict) -> None:
    """Write the geometry record."""
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error(f"Cannot create the directory for {destination}: {exc}.")
        raise SystemExit(1) from exc
    destination.write_text(json.dumps(payload, indent=2))
    logger.info(f"written to {destination}")


def run(config: PreprocessConfig) -> bool:
    """Rotation-axis orientation, centre of rotation, and tilt."""
    settings = config.geometry
    output_directory = Path(config.chain.require_output_directory())
    index = output_directory / INDEX_NAME

    band = _wavelength_band(index, settings)
    profiles = _read_profiles(index, band)
    orient = geometry.detect_axis_orientation(profiles.cols, profiles.rows)
    logger.info(f"rotation axis is {orient.name}")

    angles = profiles.angles_deg
    i0 = int(np.nanargmin(np.abs(angles - 0.0)))
    i180 = int(np.nanargmin(np.abs(angles - 180.0)))
    mirror = tilt = None
    if abs(angles[i180] - 180.0) > settings.angle_tolerance_deg:
        logger.warning(f"no projection near 180 deg (closest {angles[i180]:.3f})")
    else:
        mirror = _fit_mirror_pair(profiles, i0, i180, orient, settings)
        tilt = _fit_tilt(profiles, i0, i180, orient, settings)

    com = _centre_of_mass(profiles, orient)
    ok = _verdict(mirror, com, settings) if mirror is not None else True

    payload = _payload(output_directory.name, profiles, settings, orient, com, mirror, tilt)
    destination = output_directory / GEOMETRY_NAME
    _write(destination, payload)
    if config.chain.figures:
        plots.geometry_figure(tilt, payload, destination.with_name("geometry_summary.png"))
    return ok
