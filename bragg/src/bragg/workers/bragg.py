"""Worker process that runs the bragg chain."""

from collections.abc import Callable

from loguru import logger
from shared.core.worker import Worker

from bragg.core.bragg_types import Stage
from bragg.core.config import BraggConfig
from bragg.stages import export_rays, scan


class BraggWorker(Worker):
    """Worker that fits the scan and exports the rays."""

    def __init__(self, config: BraggConfig) -> None:
        """Store the configuration for the worker process."""
        super().__init__(daemon=False)
        self.config = config

    def steps(self) -> list[tuple[str, Callable[[BraggConfig], bool]]]:
        """Return the configured pipeline stages, in order, as (label, callable)."""
        chain = self.config.chain
        steps: list[tuple[str, Callable[[BraggConfig], bool]]] = []
        if Stage.SCAN in chain.stages:
            steps.append((Stage.SCAN, scan.run))
        if Stage.EXPORT_RAYS in chain.stages:
            steps.append((Stage.EXPORT_RAYS, export_rays.run))
        return steps

    def main(self) -> None:
        """Run the configured stages in order, stopping at the first failure."""
        output_directory = self.config.chain.require_output_directory()
        output_directory.mkdir(parents=True, exist_ok=True)

        steps = self.steps()
        for position, (label, run_stage) in enumerate(steps, start=1):
            if self.shutdown_requested:
                logger.info(f"Shutdown requested, stopping before '{label}'")
                return
            logger.info(f"[{position}/{len(steps)}] {label}")
            if not run_stage(self.config):
                remaining = [name for name, _ in steps[position:]]
                unreached = (
                    f" The stages after it read what it writes, so {' and '.join(remaining)} "
                    "would be working from missing or partial data."
                    if remaining
                    else ""
                )
                logger.error(f"'{label}' did not succeed, so the chain stops here.{unreached}")
                return

        logger.success(f"Chain complete. Results are under {output_directory}")
