"""Simulation configuration dataclasses and the loader for a Python config file."""

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from fea.core.types import DirichletBC, EigenStrain, MaterialData

CONFIG_ATTRIBUTE = "config"


@dataclass(frozen=True)
class MonitoringConfig:
    """Settings for the system monitor worker."""

    logs_dir: Path = Path("logs")
    cleanup_threshold_gigabytes: float = 5.0
    logging_period_seconds: float = 60.0


@dataclass(frozen=True)
class SimConfig:
    """Simulation parameters for one finite element run.

    Defined in a user-supplied Python file as 'config = SimConfig(...)' and loaded with 'load_config'.
    """

    materials: MaterialData
    eigen_strains: list[EigenStrain] = field(default_factory=list[EigenStrain])
    dirichlet_bcs: list[DirichletBC] = field(default_factory=list[DirichletBC])
    files: list[Path] = field(default_factory=list[Path])
    results_dir: Path = Path("results")
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)


def load_config(config_path: Path) -> SimConfig:
    """
    Load a 'SimConfig' from a Python configuration file.

    The file is imported and must define a module-level 'config' attribute
    holding a 'SimConfig' instance.
    """
    if not config_path.is_file():
        logger.error(f"Configuration file not found: {config_path}")
        raise SystemExit(1)

    spec = importlib.util.spec_from_file_location(f"_sim_config_{config_path.stem}", config_path)
    if spec is None or spec.loader is None:
        logger.error(f"Not an importable Python file: {config_path}")
        raise SystemExit(1)

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        logger.error(f"Error executing configuration file {config_path}: {e}")
        raise SystemExit(1) from e

    config = getattr(module, CONFIG_ATTRIBUTE, None)
    if not isinstance(config, SimConfig):
        logger.error(f"{config_path} must define '{CONFIG_ATTRIBUTE} = SimConfig(...)', found {type(config).__name__}")
        raise SystemExit(1)

    return config
