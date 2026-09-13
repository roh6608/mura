"""Process manager owning the pre-processing worker processes."""

from collections.abc import Sequence

from shared.core.process_manager import ProcessManager
from shared.core.system_monitor import SystemMonitorWorker
from shared.core.worker import Worker

from preprocess.core.config import PreprocessConfig
from preprocess.workers.preprocess import PreprocessWorker


class PreprocessProcessManager(ProcessManager[PreprocessConfig]):
    """Process manager for the pre-processing application."""

    def _create_workers(self) -> Sequence[Worker]:
        """Build the monitor and the chain worker, unstarted."""
        self.monitor = SystemMonitorWorker(
            logs_dir=self.config.monitoring.logs_dir,
            cleanup_threshold_gigabytes=self.config.monitoring.cleanup_threshold_gigabytes,
            logging_period_seconds=self.config.monitoring.logging_period_seconds,
        )
        self.preprocess = PreprocessWorker(self.config)
        return [self.monitor, self.preprocess]
