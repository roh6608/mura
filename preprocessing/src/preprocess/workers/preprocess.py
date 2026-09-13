"""Worker process that runs the pre-processing chain."""

from collections.abc import Callable

from loguru import logger
from shared.core.worker import Worker

from preprocess.core.config import PreprocessConfig
from preprocess.stages import geometry, open_beam, project, scan

Stage = tuple[str, Callable[[PreprocessConfig], bool]]


class PreprocessWorker(Worker):
    """Worker that runs the producing stages in order."""

    STAGES: tuple[Stage, ...] = (
        ("scan", scan.run),
        ("open beam", open_beam.run),
        ("projections", project.run),
        ("geometry", geometry.run),
    )

    def __init__(self, config: PreprocessConfig) -> None:
        """Store the configuration for the worker process."""
        super().__init__(daemon=False)
        self.config = config

    def stages(self) -> tuple[Stage, ...]:
        """Return the stages to run, dropping the open-beam sum when a reference is given."""
        if self.config.chain.open_beam_reference is None:
            return self.STAGES
        return tuple(stage for stage in self.STAGES if stage[0] != "open beam")

    def main(self) -> None:
        """Run every stage in order, stopping at the first that fails."""
        stages = self.stages()
        for position, (name, run_stage) in enumerate(stages, start=1):
            if self.shutdown_requested:
                logger.info(f"Shutdown requested, stopping before '{name}'")
                return

            logger.info(f"[{position}/{len(stages)}] {name}")
            if not run_stage(self.config):
                remaining = [later for later, _ in stages[position:]]
                unreached = (
                    f" The stages after it read what it writes, so {' and '.join(remaining)} "
                    "would be working from missing or partial data."
                    if remaining
                    else ""
                )
                logger.error(f"'{name}' did not succeed, so the chain stops here.{unreached}")
                return

        logger.success(f"Chain complete. Results {self.config.chain.require_output_directory()}")
