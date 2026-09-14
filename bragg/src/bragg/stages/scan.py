"""Fit every projection of a scan with one shared set of hyperparameters."""

from loguru import logger

from bragg.core.config import BraggConfig
from bragg.gp.fit import fit_experiment
from bragg.utils.plots import scan_summary_figure


def run(config: BraggConfig) -> bool:
    """Fit the whole experiment and write the per-projection results and summary."""
    out_dir = config.chain.require_output_directory() / "fits"
    results = fit_experiment(config, out_dir)
    scan_summary_figure(
        results, out_dir / "scan_summary.png", title=f"{config.chain.experiment.name}  edge near {config.fit.target} A"
    )
    logger.info(f"{len(results)} projections -> {out_dir}")
    return bool(results)
