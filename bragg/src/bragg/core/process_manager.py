"""Process manager owning the bragg worker processes."""

from collections.abc import Sequence

from shared.core.process_manager import ProcessManager
from shared.core.system_monitor import SystemMonitorWorker
from shared.core.worker import Worker

from bragg.core.config import BraggConfig
from bragg.workers.bragg import BraggWorker


class BraggProcessManager(ProcessManager[BraggConfig]):
    """Process manager for the bragg application: a system monitor and the fit chain."""

    def _create_workers(self) -> Sequence[Worker]:
        """Build the monitor and the chain worker, unstarted."""
        self.monitor = SystemMonitorWorker(
            logs_dir=self.config.monitoring.logs_dir,
            cleanup_threshold_gigabytes=self.config.monitoring.cleanup_threshold_gigabytes,
            logging_period_seconds=self.config.monitoring.logging_period_seconds,
        )
        self.bragg = BraggWorker(self.config)
        return [self.monitor, self.bragg]
