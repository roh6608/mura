"""Summary figurre generation."""

from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from preprocess.core.constants import FIGURE_DPI, FIGURE_SIZE_IN
from preprocess.core.preprocessing_types import Status

if TYPE_CHECKING:
    from preprocess.core.preprocessing_types import ProjReport, TiltFit
    from preprocess.io.manifest import RunRecord

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


def _finish(fig: Figure, out_path: str | PathLike[str]) -> Path:
    """Save and close 'fig', returning where it went."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info(f"figure -> {path}")
    return path


def scan_figure(records: Sequence["RunRecord"], out_path: str | PathLike[str]) -> Path | None:
    """Draw the angular coverage and frame counts of the validated acquisitions."""
    angles = [r.key.nominal_angle_deg for r in records if r.key.nominal_angle_deg is not None]
    if not angles:
        logger.warning("No angles on the validated records. Skipping the scan figure")
        return None

    ok = np.array([r.status == Status.OK for r in records if r.key.nominal_angle_deg is not None])
    angles = np.array(angles, dtype=float)
    frames = np.array([r.n_frames for r in records if r.key.nominal_angle_deg is not None], dtype=float)

    fig, (left, right) = plt.subplots(1, 2, figsize=FIGURE_SIZE_IN)
    order = np.argsort(angles)
    left.plot(angles[order], np.arange(len(angles)), ".-", lw=0.8, ms=4, color="0.3")
    left.set_xlabel("Projection Angle [deg]")
    left.set_ylabel("Sorted Index")

    right.axhline(np.median(frames), color="0.7", lw=1.0, ls="--")
    right.plot(angles[ok], frames[ok], ".", ms=5, label="ok")
    if (~ok).any():
        right.plot(angles[~ok], frames[~ok], "x", ms=7, color="crimson", label="quarantined")
        right.legend(fontsize=8)
    right.set_xlabel("Projection Angle [deg]")
    right.set_ylabel("Frames")
    return _finish(fig, out_path)


def open_beam_figure(white_beams: Mapping[str, NDArray[np.float64]], out_path: str | PathLike[str]) -> Path | None:
    """Draw every open-beam run's white beam side by side on one colour scale.

    Runs that were summed together should look like repeats of one field; a shifted beam, a moved
    slit or a hot quadrant shows up here before it is averaged into the reference.
    """
    if not white_beams:
        logger.warning("no white beams to draw; skipping the open-beam figure")
        return None
    stack = np.stack(list(white_beams.values()))
    finite = stack[np.isfinite(stack)]
    lo, hi = np.percentile(finite, (1, 99)) if finite.size else (0.0, 1.0)

    n = len(white_beams)
    # The colour bar gets a grid column of its own, so 'tight_layout' places it rather than overlaps it.
    fig, axes = plt.subplots(
        1, n + 1, figsize=(2.4 * n + 1.2, 3.0), squeeze=False, gridspec_kw={"width_ratios": [1.0] * n + [0.06]}
    )
    images = []
    for ax, (uid, image) in zip(axes[0, :n], white_beams.items(), strict=True):
        images.append(ax.imshow(image, cmap="viridis", vmin=lo, vmax=hi, origin="lower"))
        ax.set_title(uid, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(images[-1], cax=axes[0, n], label="corrected counts, all bins")
    return _finish(fig, out_path)


def projection_figure(reports: Sequence["ProjReport"], out_path: str | PathLike[str]) -> Path | None:
    """Draw transmission against angle, for the sample and for the air region."""
    good = [r for r in reports if r.ok and r.angle_deg is not None]
    if not good:
        logger.warning("No successful projections with an angle. Skipping the projection figure")
        return None

    angle = np.array([r.angle_deg for r in good], dtype=float)
    sample = np.array([r.mean_transmission for r in good], dtype=float)
    air = np.array([r.air_transmission for r in good], dtype=float)

    fig, (left, right) = plt.subplots(1, 2, figsize=FIGURE_SIZE_IN)
    left.plot(angle, sample, ".", ms=6, color="tab:blue")
    left.set_xlabel("Projection Angle [deg]")
    left.set_ylabel("Mean transmission over the sample")

    if np.isfinite(air).any():
        right.plot(angle, air, ".", ms=6, color="tab:orange")

    else:
        right.set_xlabel("projection angle [deg]")
        right.set_ylabel("mean transmission over the air ROI")
    return _finish(fig, out_path)


def geometry_figure(tilt: "TiltFit | None", payload: dict[str, Any], out_path: str | PathLike[str]) -> Path | None:
    """Draw the rotation-axis tilt fit, and the centre of rotation."""
    if tilt is None:
        logger.warning("No tilt fit for this experiment. Skipping the geometry figure")
        return None

    fig, (left, right) = plt.subplots(1, 2, figsize=FIGURE_SIZE_IN)
    used = np.asarray(tilt.used, dtype=bool)
    rows, shifts = np.asarray(tilt.rows, dtype=float), np.asarray(tilt.shifts, dtype=float)
    left.plot(rows[used], shifts[used], ".", ms=6, label="used")
    if (~used).any():
        left.plot(rows[~used], shifts[~used], "x", ms=6, color="0.6", label="rejected")
    line = tilt.slope_px_per_row * rows + tilt.intercept_px
    left.plot(rows, line, "-", lw=1.2, color="crimson", label="fit")
    left.set_xlabel("position along the rotation axis [px]")
    left.set_ylabel("mirror shift [px]")
    left.legend(fontsize=8)

    right.axis("off")
    lines = [f"{key:<24s} {value}" for key, value in payload.items() if not isinstance(value, dict | list)]
    right.text(0.0, 1.0, "\n".join(lines), family="monospace", fontsize=8, va="top")
    right.set_title("geometry.json")
    return _finish(fig, out_path)
