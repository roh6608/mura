"""Configuration. One frozen object carrying every path, policy and threshold."""

import dataclasses
import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

from loguru import logger

from preprocess.core.preprocessing_types import Kind

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


_AT_LEAST_ONE = "at_least_one"


def at_least_one[T: (int, int | None)](default: T) -> T:
    """Declare a count or stride that must be at least one; 'None' means unset and passes."""
    return field(default=default, metadata={_AT_LEAST_ONE: True})


class _Checked:
    """Mix-in enforcing, at construction, every invariant a section declares on its fields."""

    def __post_init__(self) -> None:
        """Refuse a zero or negative value on any field declared with 'at_least_one'."""
        section = cast("DataclassInstance", self)
        for f in dataclasses.fields(section):
            value = getattr(section, f.name)
            if f.metadata.get(_AT_LEAST_ONE) and value is not None and value < 1:
                raise ValueError(f"{type(self).__name__}.{f.name} must be at least 1, got {value}")


@dataclass(frozen=True)
class TrimPolicy:
    """How many bins to drop at each shutter-window edge."""

    n_head: int = 1
    n_tail: int = 1
    max_residual_ratio: float = 1.02


@dataclass(frozen=True)
class OverlapPolicy:
    """Event pile-up correction settings."""

    denominator_floor: float = 0.20
    half_bin: bool = False


@dataclass(frozen=True)
class FilterPolicy:
    """Gamma-spot despeckle and bad-pixel handling."""

    despeckle_k: float = 5.0
    despeckle_ksize: int = 3
    border_px: int = 4
    hot_sigma: float = 6.0
    dead_fraction: float = 0.2


@dataclass(frozen=True)
class Paths:
    """Where the raw data is read from: the directory holding 'images/tpx1' and 'nexus'. Set per configuration file."""

    data_root: Path

    @property
    def raw_ct(self) -> Path:
        """Return the directory holding the raw CT projections."""
        return self.data_root / "images/tpx1/raw/ct"

    @property
    def raw_radiography(self) -> Path:
        """Return the directory holding the raw radiographs."""
        return self.data_root / "images/tpx1/raw/radiography"

    @property
    def open_beam(self) -> Path:
        """Return the directory holding the open-beam images."""
        return self.data_root / "images/tpx1/ob"

    @property
    def nexus(self) -> Path:
        """Return the directory holding the NeXus run files."""
        return self.data_root / "nexus"

    @property
    def alignment(self) -> Path:
        """Return the directory holding the alignment images."""
        return self.data_root / "images/tpx1/alignment"

    def root_for(self, kind: Kind) -> Path:
        """Return the raw tree holding acquisitions of 'kind'."""
        return {
            Kind.CT: self.raw_ct,
            Kind.RADIOGRAPHY: self.raw_radiography,
            Kind.OPEN_BEAM: self.open_beam,
            Kind.ALIGNMENT: self.alignment,
        }[kind]


@dataclass(frozen=True)
class Scan(_Checked):
    """Settings for the discovery and validation stage."""

    n_headers: int = at_least_one(12)
    limit: int | None = at_least_one(None)


@dataclass(frozen=True)
class Monitoring:
    """Settings for the system monitor worker."""

    logs_dir: Path = Path("logs")
    cleanup_threshold_gigabytes: float = 5.0
    logging_period_seconds: float = 60.0


@dataclass(frozen=True)
class Geometry(_Checked):
    """Settings for the centre-of-rotation and tilt stage."""

    lambda_low_a: float = 1.0
    lambda_high_a: float = 6.1
    bands: int = at_least_one(8)
    max_shift: int | None = None
    min_correlation: float = 0.8
    rows: int = at_least_one(16)
    min_amplitude: float = 1.0
    angle_tolerance_deg: float = 1.0
    tolerance_px: float = 0.5

    def __post_init__(self) -> None:
        """Enforce the field invariants; an inverted wavelength band would select nothing."""
        super().__post_init__()
        if self.lambda_high_a <= self.lambda_low_a:
            raise ValueError(
                f"Geometry.lambda_high_a ({self.lambda_high_a}) must be greater than "
                f"Geometry.lambda_low_a ({self.lambda_low_a}); as given the band is empty."
            )


@dataclass(frozen=True)
class OpenBeam(_Checked):
    """Settings for combining the open-beam runs."""

    band: int = at_least_one(128)
    block_gap: int = 10


@dataclass(frozen=True)
class Projection:
    """Settings for writing one projection's transmission cube."""

    store_variance: bool = True
    repair_ksize: int = 5
    repair_passes: int = 3


@dataclass(frozen=True)
class Rois(_Checked):
    """Settings for building the sample and air regions of interest."""

    wb_step: int = at_least_one(20)
    sample_threshold: float = 0.85
    air_threshold: float = 0.97
    air_percentile: float = 99.0
    angles: int = at_least_one(12)
    margin_px: int = 6
    #: Fewer always-clear air pixels than this and the air ROI is refused.
    min_air_px: int = at_least_one(500)
    #: Box-blur width applied to the min-over-angle ratio before thresholding; 1 disables it.
    ratio_smooth: int = at_least_one(5)
    #: The air threshold is capped here so a bright reference cannot push it out of reach.
    air_threshold_min: float = 0.93
    force_rebuild: bool = False


@dataclass(frozen=True)
class Chain(_Checked):
    """What an end-to-end run has to decide that no single stage owns."""

    experiment: str | None = None
    open_beam_experiment: str | None = None
    kind: Kind = Kind.CT
    output_directory: Path | None = None
    open_beam_reference: Path | None = None
    open_beam_block: int | None = None
    device: str = "cpu"
    band: int = at_least_one(64)
    limit: int | None = at_least_one(None)
    overwrite: bool = False
    compression: str = "lzf"
    fast: bool = False
    figures: bool = True

    def __post_init__(self) -> None:
        """Enforce the field invariants; a plain string for 'kind' becomes the enum or is refused."""
        super().__post_init__()
        object.__setattr__(self, "kind", Kind(self.kind))

    def require_output_directory(self) -> Path:
        """Return the output directory, failing clearly when the config leaves it unset."""
        if self.output_directory is None:
            logger.error(
                "'chain.output_directory' is not set. The same directory is handed to "
                "every stage, so it cannot fall back to a per-experiment default that "
                "is only known once the runs have been discovered and validated."
            )
            raise SystemExit(1)
        return self.output_directory


@dataclass(frozen=True)
class PreprocessConfig(_Checked):
    """Everything the chain needs that is not the data itself."""

    paths: Paths
    trim: TrimPolicy = field(default_factory=TrimPolicy)
    overlap: OverlapPolicy = field(default_factory=OverlapPolicy)
    filters: FilterPolicy = field(default_factory=FilterPolicy)
    chain: Chain = field(default_factory=Chain)
    scan: Scan = field(default_factory=Scan)
    geometry: Geometry = field(default_factory=Geometry)
    open_beam: OpenBeam = field(default_factory=OpenBeam)
    projection: Projection = field(default_factory=Projection)
    rois: Rois = field(default_factory=Rois)
    monitoring: Monitoring = field(default_factory=Monitoring)

    expected_frames: int | None = None
    n_workers: int = at_least_one(16)

    def to_json(self, indent: int = 2) -> str:
        """Serialise the whole configuration, for the 'config.json' written beside the projections."""
        settings = dataclasses.asdict(self)

        def default(o: object) -> str:
            if isinstance(o, Path):
                return str(o)
            raise TypeError(type(o))

        return json.dumps(settings, indent=indent, default=default)


CONFIG_ATTRIBUTE = "config"


def load_config(config_path: Path) -> PreprocessConfig:
    """Load a 'PreprocessConfig' from a Python configuration file."""
    if not config_path.is_file():
        logger.error(f"Configuration file not found: {config_path}")
        raise SystemExit(1)

    spec = importlib.util.spec_from_file_location(f"_preprocess_config_{config_path.stem}", config_path)
    if spec is None or spec.loader is None:
        logger.error(f"Not an importable Python file: {config_path}")
        raise SystemExit(1)

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        logger.exception(f"Error executing configuration file {config_path}.")
        raise SystemExit(1) from None

    config = getattr(module, CONFIG_ATTRIBUTE, None)
    if not isinstance(config, PreprocessConfig):
        logger.error(
            f"{config_path} must define '{CONFIG_ATTRIBUTE} = PreprocessConfig(...)', found {type(config).__name__}"
        )
        raise SystemExit(1)

    return config
