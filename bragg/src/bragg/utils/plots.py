"""Diagnostic figures for a Bragg-edge fit."""

from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from shared.core.backend import asnumpy

from bragg.core.bragg_types import FitResult, ProjectionFit
from bragg.core.constants import FIGURE_DPI, MIN_POINTS_FOR_FIT

if TYPE_CHECKING:
    from matplotlib.gridspec import GridSpec, SubplotSpec

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

plt.rcParams.update(
    {"savefig.dpi": FIGURE_DPI, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.8, "font.size": 9}
)


def _finish(fig: Figure, out_path: str | PathLike[str]) -> Path:
    """Save 'fig' as PNG and PDF beside each other, close it, and return the PNG path."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(path.with_suffix(suffix), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"figure -> {path}")
    return path


def _panel(  # noqa: PLR0913, PLR0917
    fig: Figure,
    slot: "SubplotSpec",
    img: NDArray[np.float64],
    name: str,
    cmap: str = "viridis",
    pct: tuple[float, float] = (2, 98),
    mask: NDArray[np.bool_] | None = None,
    unit: str = "",
) -> None:
    """Draw one map, colour-scaled between two percentiles of its finite values."""
    ax = fig.add_subplot(slot)
    v = np.where(mask, img, np.nan) if mask is not None else img
    finite = v[np.isfinite(v)]
    lo, hi = np.percentile(finite, pct) if finite.size else (0, 1)
    im = ax.imshow(v, cmap=cmap, vmin=lo, vmax=hi, origin="lower", interpolation="nearest")
    ax.set_title(name, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046).set_label(unit, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def _spectra_panels(fig: Figure, gs: "GridSpec", fit: ProjectionFit, target_lambda: float, n_examples: int) -> None:
    """Draw a few valid pixels' spectra with their posterior mean, and the posterior derivative beside them."""
    crop, maps = fit.crop, fit.maps
    lam = np.asarray(crop.lam)
    ys, xs = np.nonzero(maps.valid)
    if not ys.size:
        return
    mean = asnumpy(fit.gp.posterior_mean(fit.y)) + fit.offset
    deriv = asnumpy(fit.gp.posterior_derivative(fit.y, axis=0))
    data = np.asarray(crop.cube)
    pick = np.linspace(0, ys.size - 1, min(n_examples, ys.size)).astype(int)
    ax = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2:])
    for j in pick:
        r, c = int(ys[j]), int(xs[j])
        ax.plot(lam, data[:, r, c], ".", ms=2.5, alpha=0.35)
        ax.plot(lam, mean[:, r, c], "-", lw=1.6, label=f"({r},{c})")
        ax2.plot(lam, deriv[:, r, c], "-", lw=1.4)
        ax2.axvline(maps.position[r, c], ls=":", lw=1)
    ax.axvline(target_lambda, color="k", ls="--", lw=1, label="predicted edge")
    ax.set_xlabel("wavelength [A]")
    ax.set_ylabel("T (arb.)")
    ax.set_title("data and posterior mean")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax2.axvline(target_lambda, color="k", ls="--", lw=1)
    ax2.set_xlabel("wavelength [A]")
    ax2.set_ylabel("dT/dlambda")
    ax2.set_title("posterior derivative; dotted = fitted edge")
    ax2.grid(alpha=0.3)


def _histogram(fig: Figure, slot: "SubplotSpec", values: NDArray[np.float64], xlabel: str, title: str) -> None:
    """Draw the histogram of the finite values, if there are any."""
    ax = fig.add_subplot(slot)
    if values.size:
        ax.hist(values[np.isfinite(values)], bins=60, alpha=0.85)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("pixels")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)


def edge_fit_figure(
    fit: ProjectionFit, out_path: str | PathLike[str], target_lambda: float, title: str = "", n_examples: int = 4
) -> Path:
    """Spectra with their posterior fit and derivative, plus the four maps."""
    crop, maps = fit.crop, fit.maps
    valid = maps.valid
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.15, 1, 1], hspace=0.34, wspace=0.28)
    _spectra_panels(fig, gs, fit, target_lambda, n_examples)

    step = crop.step_a
    _panel(fig, gs[1, 0], np.nanmean(np.asarray(crop.cube), axis=0), "white beam (crop)", unit="T")
    _panel(fig, gs[1, 1], maps.position, "edge position", mask=valid, unit="A")
    _panel(fig, gs[1, 2], maps.sigma / step, "sigma", cmap="magma", mask=valid, unit="bins")
    _panel(fig, gs[1, 3], valid.astype(float), "validity mask", cmap="gray", pct=(0, 100))
    _panel(fig, gs[2, 0], maps.amplitude, "derivative peak", cmap="cividis", mask=valid)
    _panel(fig, gs[2, 1], maps.strain * 1e6, "strain", cmap="coolwarm", mask=valid, unit="microstrain")

    e = maps.strain[valid] * 1e6
    _histogram(fig, gs[2, 2], e, "strain [microstrain]", f"sd {np.nanstd(e):.0f} ue over {valid.sum()} px")
    s = maps.sigma[valid] / step
    _histogram(fig, gs[2, 3], s, "sigma [bins]", f"median {np.nanmedian(s):.2f} bins")

    fig.suptitle(title or "Bragg-edge GP fit", fontsize=13)
    return _finish(fig, out_path)


def scan_summary_figure(results: Sequence[FitResult], out_path: str | PathLike[str], title: str = "") -> Path:
    """Per-projection behaviour across a scan."""
    a = np.array([r.angle_deg for r in results], float)
    o = np.argsort(a)
    a = a[o]
    series = [
        ("edge_median", "median edge [A]"),
        ("sigma_median", "median sigma [A]"),
        ("strain_std", "strain sd"),
        ("n_valid", "valid pixels"),
    ]
    fig, axes = plt.subplots(len(series), 1, figsize=(11, 10), sharex=True)
    for ax, (key, label) in zip(axes, series, strict=True):
        v = np.array([getattr(r, key) for r in results], float)[o]
        ax.plot(a, v, "o-", ms=4)
        m = np.nanmean(v)
        ax.axhline(m, color="tab:red", ls=":", label=f"mean {m:.5g}")
        th = np.radians(a)
        design_matrix = np.stack([np.ones(len(a)), np.cos(2 * th), np.sin(2 * th)], axis=1)
        good = np.isfinite(v)
        if good.sum() > MIN_POINTS_FOR_FIT:
            c, *_ = np.linalg.lstsq(design_matrix[good], v[good], rcond=None)
            resid = v[good] - design_matrix[good] @ c
            frac = 1.0 - np.var(resid) / max(np.var(v[good]), 1e-30)
            ax.plot(
                a,
                design_matrix @ c,
                "--",
                color="tab:green",
                lw=1,
                label=f"180 deg harmonic, {100 * frac:.0f}% of variance",
            )
        ax.set_ylabel(label, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("rotation angle [deg]")
    axes[0].set_title(
        title or "edge fit across the scan\na trend with ANGLE reconstructs into false strain; scatter does not"
    )
    fig.tight_layout()
    return _finish(fig, out_path)
