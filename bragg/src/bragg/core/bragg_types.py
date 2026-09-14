"""Types shared across the Bragg-edge processing."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from bragg.gp.kron import KroneckerGP


class Stage(StrEnum):
    """The pipeline stages, in the order the chain runs them."""

    SCAN = "scan"
    EXPORT_RAYS = "export-rays"


@dataclass(frozen=True)
class Projection:
    """One projection file of a preprocessed experiment."""

    path: Path
    angle_deg: float
    run_number: int
    acq: str


@dataclass(frozen=True)
class Crop:
    """One projection cropped in wavelength and space, as a complete grid."""

    cube: NDArray[np.float64]
    lam: NDArray[np.float64]
    tof: NDArray[np.float64]
    sigma: NDArray[np.float64] | None
    bbox: tuple[int, int, int, int]
    lam_slice: tuple[int, int]
    window_id: int
    filled: NDArray[np.bool_]
    angle_deg: float
    run_number: int

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the cropped cube, '(n_lambda, ny, nx)'."""
        return self.cube.shape

    @property
    def step_a(self) -> float:
        """Wavelength bin width of the crop, in Angstrom."""
        return float(np.median(np.diff(self.lam)))

    def coords(self) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Axis coordinates for the GP: wavelength along ToF, pixels in space."""
        ny, nx = self.cube.shape[1:]
        return (np.asarray(self.lam, dtype=float), np.arange(ny, dtype=float), np.arange(nx, dtype=float))


@dataclass(frozen=True)
class HyperFit:
    """One set of hyperparameters."""

    length_scales: tuple[float, ...]
    sigma_f: float
    sigma_n: float
    nlml: float
    n_evals: int
    converged: bool
    at_bound: tuple[int, ...] = ()

    @property
    def sigmas_fitted(self) -> bool:
        """Return whether both sigmas carry a value."""
        return bool(np.isfinite(self.sigma_f) and np.isfinite(self.sigma_n))

    def gp(
        self,
        coords: Sequence[NDArray[np.float64]],
        kinds: Sequence[str],
        xp: ModuleType = np,
        work_dtype: type[np.floating[Any]] = np.float64,
    ) -> KroneckerGP:
        """Return a 'KroneckerGP' on these coordinates carrying these hyperparameters."""
        return KroneckerGP(
            coords=tuple(coords),
            length_scales=self.length_scales,
            kinds=tuple(kinds),
            sigma_f=self.sigma_f,
            sigma_n=self.sigma_n,
            xp=xp,
            work_dtype=work_dtype,
        )


@dataclass(frozen=True)
class OptimiseOptions:
    """Tuning for the marginal-likelihood search: the kernel per axis, the start and the numerics."""

    kinds: tuple[str, ...] | None = None
    init: tuple[float, ...] | None = None
    xp: ModuleType = np
    work_dtype: type[np.floating[Any]] = np.float64


@dataclass(frozen=True)
class ValidityCriteria:
    """The cuts and morphology that decide whether a pixel carries a real edge."""

    sigma_max: float | None = None
    amplitude_min: float | None = None
    open_radius: int = 1
    largest_component: bool = True
    boundary: NDArray[np.bool_] | None = None
    curvature_ok: NDArray[np.bool_] | None = None


@dataclass(frozen=True)
class NoiseCheck:
    """The scalar noise the GP assumes against the propagated per-point sigma."""

    ok: bool | None
    mad_first_difference: float
    scalar: float = float("nan")
    pointwise_median: float = float("nan")
    ratio_scalar_over_median: float = float("nan")
    pointwise_p5: float = float("nan")
    pointwise_p95: float = float("nan")
    spread: float = float("nan")


@dataclass(frozen=True)
class FitResult:
    """Everything about one fitted projection."""

    run_number: int
    angle_deg: float
    bbox: tuple[int, int, int, int]
    lam_slice: tuple[int, int]
    window_id: int
    length_scales: tuple[float, ...]
    sigma_f: float
    sigma_n: float
    nlml: float
    converged: bool
    n_valid: int
    n_pixels: int
    edge_median: float
    edge_iqr: float
    sigma_median: float
    strain_reference: float
    strain_std: float
    noise_check: NoiseCheck
    seconds: float
    device: str

    def to_attrs(self) -> dict[str, Any]:
        """Return the scalar and tuple fields as HDF5 attributes."""
        return {k: v for k, v in asdict(self).items() if k != "noise_check"}

    def to_json_dict(self) -> dict[str, Any]:
        """Return the whole record, nested, for 'fit_report.json'."""
        return asdict(self)


@dataclass(frozen=True)
class FitMaps:
    """The per-pixel maps of one fitted projection."""

    position: NDArray[np.float64]
    sigma: NDArray[np.float64]
    valid: NDArray[np.bool_]
    strain: NDArray[np.float64]
    amplitude: NDArray[np.float64]
    n_draws_used: NDArray[np.float64]


@dataclass(frozen=True)
class ProjectionFit:
    """One fitted projection."""

    result: FitResult
    maps: FitMaps
    crop: Crop
    gp: KroneckerGP
    y: NDArray[np.float64]
    offset: float


@dataclass(frozen=True)
class RayFit:
    """One per-projection fit as the ray export reads it back."""

    angle_deg: float
    run_number: int
    acq: str
    lambda0: float
    strain: NDArray[np.float32]
    sigma: NDArray[np.float32]
    valid: NDArray[np.bool_]
    bbox: tuple[int, int, int, int]
