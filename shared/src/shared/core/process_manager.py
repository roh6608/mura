"""Ownership and lifecycle management for an application's worker processes."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from loguru import logger

from shared.core.worker import Worker


class ProcessManager[ConfigT](ABC):
    """Owns an application's workers and manages their lifecycle.

    Each application subclasses this and implements '_create_workers' to build its
    worker set, e.g. a system monitor plus a compute worker (in the future multiple
    compute workers). 'config' is the application's runtime-loaded dataclass, stored
    for workers to read, never write.

    '_create_workers' is called at the end of '__init__', so subclasses that need extra
    constructor state must set it before calling 'super().__init__'.
    """

    def __init__(self, config: ConfigT) -> None:
        """Store the configuration and build the worker set, unstarted."""
        self.config = config

        logger.debug("Instantiating workers...")
        self.workers: tuple[Worker, ...] = tuple(self._create_workers())

    @abstractmethod
    def _create_workers(self) -> Sequence[Worker]:
        """Build the application's workers, unstarted."""

    def start(self) -> None:
        """Start every worker."""
        for worker in self.workers:
            worker.start()
        logger.debug("Workers started.")

    def stop(self, timeout: float | None = 60.0) -> None:
        """Stop every worker, allowing each 'timeout' seconds to exit."""
        logger.debug("Stopping workers...")

        # Signal every worker before joining any, so they all shut down in
        # parallel instead of consuming one timeout each in sequence.
        for worker in self.workers:
            worker.request_shutdown()

        for worker in self.workers:
            worker.stop(timeout)
