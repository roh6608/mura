"""Types shared across the pre-processing chain."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from preprocess.core.constants import CUBE_CHUNK_YX, DETECTOR_SHAPE, H_OVER_M, SPECTRA_ATOL_S, TOF_CARD_ATOL_S

if TYPE_CHECKING:
    from preprocess.core.config import PreprocessConfig


class Kind(StrEnum):
    """What an acquisition is."""

    CT = "ct"
    RADIOGRAPHY = "radiography"
    OPEN_BEAM = "ob"
    ALIGNMENT = "alignment"


class Status(StrEnum):
    """Where an acquisition stands after validation."""

    UNCHECKED = "unchecked"
    OK = "ok"
    QUARANTINE = "quarantine"


@dataclass(frozen=True)
class RunKey:
    """Everything recoverable from a run-directory name alone."""

    dirname: str
    date: str
    run_number: int
    body: str

    nominal_angle_deg: float | None = None
    seq: int | None = None
    ob_position: int | None = None
    charge_c: float | None = None
    lambda_min_a: float | None = None

    @property
    def is_open_beam(self) -> bool:
        """Return whether this run is an open-beam acquisition."""
        return self.ob_position is not None

    @property
    def is_projection(self) -> bool:
        """Return whether this run is a sample projection."""
        return self.nominal_angle_deg is not None


@dataclass(frozen=True)
class CentreOfRotationFit:
    """Centre-of-rotation estimate from one mirror pair."""

    centre_of_rotation_px: float
    shift_px: float
    correlation: float
    n_cols: int
    saturated: bool = False

    @property
    def offset_from_centre_px(self) -> float:
        """Return the centre of rotation's offset from the detector centre, in pixels."""
        return self.centre_of_rotation_px - (self.n_cols - 1) / 2.0


@dataclass(frozen=True)
class TiltFit:
    """Linear fit of the apparent mirror shift against detector row."""

    slope_px_per_row: float
    intercept_px: float
    residual_px: float
    rows: NDArray[np.int64]
    shifts: NDArray[np.float64]
    correlations: NDArray[np.float64]
    used: NDArray[np.bool_]

    @property
    def tilt_deg(self) -> float:
        """Tilt of the rotation axis in the detector plane."""
        return float(np.degrees(np.arctan(self.slope_px_per_row / 2.0)))


@dataclass(frozen=True)
class AxisOrientation:
    """Which way the rotation axis lies in the detector plane."""

    horizontal: bool
    swing_x: float
    swing_y: float

    @property
    def name(self) -> str:
        """Return a human-readable description of the axis orientation."""
        return "horizontal (along detector x)" if self.horizontal else "vertical (along detector y)"

    @property
    def centre_of_rotation_is_row(self) -> bool:
        """Return whether the centre of rotation is measured along a detector row."""
        return self.horizontal


@dataclass(frozen=True)
class Profiles:
    """Every projection reduced to an attenuation image and its two marginal profiles, in angle order."""

    angles_deg: NDArray[np.float64]
    runs: NDArray[np.int64]
    attenuation: list[NDArray[np.float64]]
    cols: NDArray[np.float64]
    rows: NDArray[np.float64]

    @property
    def n(self) -> int:
        """Return the number of projections."""
        return len(self.attenuation)

    @property
    def shape(self) -> tuple[int, int]:
        """Return the (ny, nx) shape of one image."""
        return self.rows.shape[1], self.cols.shape[1]


@dataclass(frozen=True)
class CentreOfMass:
    """The centre-of-rotation estimate from the sinogram's centroid, and the two numbers that qualify it."""

    centre_px: float
    amplitude_px: float
    residual_px: float
    truncation_fraction: float


@dataclass(frozen=True)
class ToFAxis:
    """Absolute ToF of every bin, plus its shutter-window membership."""

    t_start_s: NDArray[np.float64]
    bin_width_s: float
    window_id: NDArray[np.int8]
    window_slices: tuple[slice, ...]
    n_trig: NDArray[np.int64]

    @property
    def n_bins(self) -> int:
        """Return the number of ToF bins on the axis."""
        return int(self.t_start_s.size)

    @property
    def n_windows(self) -> int:
        """Return the number of ToF windows on the axis."""
        return len(self.window_slices)

    @property
    def t_centre_s(self) -> NDArray[np.float64]:
        """Return the centre time of each ToF bin, in seconds."""
        return self.t_start_s + 0.5 * self.bin_width_s

    @property
    def gaps_s(self) -> list[tuple[float, float]]:
        """'(end_of_window_w, start_of_window_w+1)' for each readout gap."""
        out = []
        for w in range(self.n_windows - 1):
            a = self.window_slices[w]
            b = self.window_slices[w + 1]
            out.append((float(self.t_start_s[a.stop - 1] + self.bin_width_s), float(self.t_start_s[b.start])))
        return out

    def wavelength_a(self, flight_path_m: float, t0_s: float) -> NDArray[np.float64]:
        """Bin-centre wavelength in Angstrom."""
        return H_OVER_M * (self.t_centre_s + t0_s) / flight_path_m

    def trig_per_bin(self) -> NDArray[np.int64]:
        """Per-bin trigger count, expanded from the per-window values."""
        out = np.empty(self.n_bins, dtype=np.int64)
        for w, sl in enumerate(self.window_slices):
            out[sl] = self.n_trig[w]
        return out

    def trim_mask(self, n_head: int = 1, n_tail: int = 1) -> NDArray[np.bool_]:
        """Boolean mask of *good* bins, dropping the anomalous last bin of each shutter window."""
        good = np.ones(self.n_bins, dtype=bool)
        for sl in self.window_slices:
            if n_head:
                good[sl.start : sl.start + n_head] = False
            if n_tail:
                good[sl.stop - n_tail : sl.stop] = False
        return good


@dataclass(frozen=True)
class AxisReport:
    """How far a built axis sits from the two other records of the same timing, 'Spectra.txt' and the frame headers."""

    n_bins: int
    window_bins: tuple[int, ...]
    boundary_indices: tuple[int, ...]
    spectra_rows: int
    max_dev_spectra_s: float
    worst_spectra_bin: int
    max_dev_headers_s: float | None
    worst_header_frame: int | None

    @property
    def spectra_ok(self) -> bool:
        """Return whether 'Spectra.txt' has one row per bin and agrees with the axis to 'SPECTRA_ATOL_S'."""
        return self.spectra_rows == self.n_bins and self.max_dev_spectra_s <= SPECTRA_ATOL_S

    @property
    def headers_ok(self) -> bool:
        """Return whether every sampled 'TOF' card agrees with the axis to 'TOF_CARD_ATOL_S'."""
        return self.max_dev_headers_s is None or self.max_dev_headers_s <= TOF_CARD_ATOL_S

    @property
    def ok(self) -> bool:
        """Return whether both records agree with the axis."""
        return self.spectra_ok and self.headers_ok


@dataclass(frozen=True)
class EdgeAnomaly:
    """How far a shutter window's first and last bin sit from their neighbours, as ratios to the mean of them.

    A ratio is NaN when the window is too short to have neighbours to compare with.
    """

    window: int
    n_bins: int
    first_ratio: float
    last_ratio: float


@dataclass
class RunMeta:
    """Per-run metadata from the NeXus file, with whatever could not be found recorded explicitly."""

    run_number: int
    path: str
    proton_charge: float | None = None
    duration_s: float | None = None
    total_pulses: int | None = None
    monitor_counts: dict[str, int] = field(default_factory=dict)
    rotation_deg: dict[str, float] = field(default_factory=dict)
    scalars: dict[str, float] = field(default_factory=dict)
    missing: tuple[str, ...] = ()


@dataclass(frozen=True)
class Calibration:
    """One run's wavelength calibration, from the instrument's own DASlogs."""

    flight_path_m: float
    t0_s: float
    lambda_min_a: float

    @classmethod
    def from_nexus(cls, meta: RunMeta) -> "Calibration | None":
        """Derive the calibration from the trigger delay and the chopper cut-off, or 'None' if either log is missing."""
        t0_us = meta.scalars.get("ti_delay_us")
        lam = meta.scalars.get("lambda_min_actual")
        if t0_us is None or not lam:
            return None
        t0 = float(t0_us) * 1e-6
        return cls(flight_path_m=H_OVER_M * t0 / float(lam), t0_s=t0, lambda_min_a=float(lam))


@dataclass(frozen=True)
class BoundaryStep:
    """The discontinuity of a spectrum across one readout gap."""

    gap: int
    left_mean: float
    right_mean: float
    step_pct: float
    extrap_pct: float | None
    left_slope_pct_per_bin: float | None
    right_slope_pct_per_bin: float | None


@dataclass(frozen=True)
class ShutterTimes:
    """Shutter window timing, already reduced to the active windows."""

    delay_s: NDArray[np.float64]
    duration_s: NDArray[np.float64]

    @property
    def n_windows(self) -> int:
        """Return the number of ToF windows in the acquisition."""
        return len(self.delay_s)


@dataclass(frozen=True)
class AuxPaths:
    """Sidecar files of a single *acquisition* within a run directory."""

    shutter_times: str
    shutter_count: str
    spectra: str
    status: str | None
    dsc: str | None
    stem: str  # "<run_dirname>_<NNN>"
    counter: str  # "<NNN>", the acquisition counter


@dataclass(frozen=True)
class FrameHeader:
    """One frame's FITS header: every card as read, with typed access to the four the chain relies on.

    A frame lacking one of those raises 'KeyError' from the property.
    """

    cards: Mapping[str, float | str]

    @property
    def tof_s(self) -> float:
        """Return the start time of this frame's ToF bin, in seconds after the trigger."""
        return float(self.cards["TOF"])

    @property
    def timebin_s(self) -> float:
        """Return the width of the ToF bin, in seconds."""
        return float(self.cards["TIMEBIN"])

    @property
    def n_counts(self) -> int:
        """Return the total counts in the frame as the writer summed them: the checksum of a decode."""
        return int(self.cards["N_COUNTS"])

    @property
    def n_trigs(self) -> int:
        """Return the triggers in this frame's shutter window."""
        return int(self.cards["N_TRIGS"])


@dataclass
class ProjectionMeta:
    """Everything about a projection that is not pixel data."""

    run_number: int
    acq: str
    angle_deg: float | None
    experiment: str
    kind: Kind
    flight_path_m: float
    t0_s: float
    lambda_min_a: float
    bin_width_s: float
    n_trig: list[int]
    ob_reference: str
    n_sample_px: int
    n_air_px: int
    mean_correction: float
    n_saturating_px: int

    def to_attrs(self) -> dict:
        """Return the metadata as HDF5 attributes: None as an empty string, an enum as its value."""
        d = {k: ("" if v is None else v.value if isinstance(v, StrEnum) else v) for k, v in self.__dict__.items()}
        d["n_trig"] = np.asarray(self.n_trig, dtype=np.int64)
        return d


@dataclass(frozen=True)
class CubeLayout:
    """How a transmission cube is laid out on disk."""

    shape_yx: tuple[int, int] = DETECTOR_SHAPE
    band: int = 64
    chunk_yx: int = CUBE_CHUNK_YX
    compression: str | None = "lzf"
    store_variance: bool = True


@dataclass
class ProjTask:
    """A single unit of work. Must stay picklable and small."""

    run_dir: str
    counter: str
    experiment: str
    kind: Kind
    run_number: int
    angle_deg: float | None
    calibration: Calibration
    out_path: str
    ob_reference: str
    rois_path: str
    config: "PreprocessConfig"
    device: str = "cpu"
    band: int = 128
    store_variance: bool = True
    despeckle_k: float = 5.0
    despeckle_ksize: int = 3
    repair_ksize: int = 5
    repair_passes: int = 3
    compression: str = "lzf"


@dataclass
class ProjReport:
    """One projection's outcome, as recorded in the run report."""

    run_number: int
    counter: str
    angle_deg: float | None
    out_path: str
    ok: bool
    seconds: float = 0.0
    n_bins: int = 0
    mean_correction: float = 1.0
    n_saturating_px: int = 0
    n_despeckled: int = 0
    frac_counts_despeckled: float = 0.0
    n_repaired_px: int = 0
    mean_transmission: float = float("nan")
    air_transmission: float = float("nan")
    message: str = ""
