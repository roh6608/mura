"""Configuration. Typically partially overwritten by user supplied config."""

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from bragg.core.bragg_types import Stage
from bragg.core.constants import LAM_FE_110_A, SPATIAL_LS_PX


@dataclass(frozen=True)
class Fit:
    """Everything the GP fit needs that is not the data itself."""

    target: float = LAM_FE_110_A
    half_width: float = 0.05
    bins: int | None = None
    device: str = "cpu"
    draws: int = 32
    sigma_max_bins: float = 3.0
    tof_bins: float | None = None
    spatial_ls: float = SPATIAL_LS_PX

    def __post_init__(self) -> None:
        """Refuse a fit that could not run."""
        if self.half_width <= 0:
            raise ValueError(f"Fit.half_width must be positive, got {self.half_width}")
        if self.draws < 1:
            raise ValueError(f"Fit.draws must be at least 1, got {self.draws}")
        if self.tof_bins is not None and self.tof_bins <= 0:
            raise ValueError(f"Fit.tof_bins must be positive or None, got {self.tof_bins}")


@dataclass(frozen=True)
class AxisSpec:
    """One rotation axis: where its fits are, which experiment they came from, and its label."""

    fit_directory: Path
    experiment: Path
    axis: str


@dataclass(frozen=True)
class Export:
    """Settings for the ray export."""

    calibration: Path
    axes: tuple[AxisSpec, ...] = ()


@dataclass(frozen=True)
class Chain:
    """Whats required for a complete run."""

    experiment: Path
    output_directory: Path | None = None
    limit: int | None = None
    reference_angle_deg: float = 0.0
    per_projection_hypers: bool = False
    stages: tuple[Stage, ...] = (Stage.SCAN, Stage.EXPORT_RAYS)

    def __post_init__(self) -> None:
        """Enforce field invariants and coerce stage names to their enum values."""
        if self.limit is not None and self.limit < 1:
            raise ValueError(f"Chain.limit must be at least 1 or None, got {self.limit}")
        object.__setattr__(self, "stages", tuple(Stage(s) for s in self.stages))

    def require_output_directory(self) -> Path:
        """Return the output directory, failing clearly when the config leaves it unset."""
        if self.output_directory is None:
            logger.error(
                "'chain.output_directory' is not set. Every stage writes into the same "
                "directory and the next one reads what the last wrote, so it cannot be "
                "guessed per stage."
            )
            raise SystemExit(1)
        return self.output_directory


@dataclass(frozen=True)
class Monitoring:
    """Settings for the system monitor worker."""

    logs_dir: Path = Path("logs")
    cleanup_threshold_gigabytes: float = 5.0
    logging_period_seconds: float = 60.0


@dataclass(frozen=True)
class BraggConfig:
    """Everything a run needs that is not the data itself."""

    chain: Chain
    fit: Fit = field(default_factory=Fit)
    export: Export | None = None
    monitoring: Monitoring = field(default_factory=Monitoring)

    def require_export(self) -> Export:
        """Return the export settings, failing clearly when the config leaves them unset."""
        if self.export is None:
            logger.error("'export' is not set, and the export-rays stage needs the calibration and the axes.")
            raise SystemExit(1)
        return self.export


CONFIG_ATTRIBUTE = "config"


def load_config(config_path: Path) -> BraggConfig:
    """Load a 'BraggConfig' from a Python configuration file."""
    if not config_path.is_file():
        logger.error(f"Configuration file not found: {config_path}")
        raise SystemExit(1)

    spec = importlib.util.spec_from_file_location(f"_bragg_config_{config_path.stem}", config_path)
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
    if not isinstance(config, BraggConfig):
        logger.error(
            f"{config_path} must define '{CONFIG_ATTRIBUTE} = BraggConfig(...)', found {type(config).__name__}"
        )
        raise SystemExit(1)

    return config
