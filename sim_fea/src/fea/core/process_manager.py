"""Process manager owning the simulation's worker processes."""

from collections.abc import Sequence

from shared.core.process_manager import ProcessManager
from shared.core.system_monitor import SystemMonitorWorker
from shared.core.worker import Worker

from fea.core.config import SimConfig
from fea.workers.solver import SolverWorker


class FeaProcessManager(ProcessManager[SimConfig]):
    """Process manager for the sim-fea application: a system monitor and the FEA solver worker."""

    def _create_workers(self) -> Sequence[Worker]:
        self.monitor = SystemMonitorWorker(
            logs_dir=self.config.monitoring.logs_dir,
            cleanup_threshold_gigabytes=self.config.monitoring.cleanup_threshold_gigabytes,
            logging_period_seconds=self.config.monitoring.logging_period_seconds,
        )
        self.solver = SolverWorker(self.config)
        return [self.monitor, self.solver]
