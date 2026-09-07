"""Worker that logs system utilisation and keeps the logs directory under a size budget."""

import contextlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import psutil
from loguru import logger

from shared.core.worker import Worker

BYTES_PER_GIGABYTE = 1024**3

# name, utilisation.gpu, temperature.gpu, memory.used, memory.total
NVIDIA_SMI_FIELD_COUNT = 5


class SystemMonitorWorker(Worker):
    """Periodically log system utilisation.

    Keeps the logs directory under a size budget by deleting the oldest files first.
    """

    def __init__(
        self, logs_dir: Path, cleanup_threshold_gigabytes: float, logging_period_seconds: float = 60.0
    ) -> None:
        """Set the directory to police and how often to sample."""
        super().__init__()
        self.logs_dir = logs_dir
        self.logging_period_seconds = logging_period_seconds
        self.max_logs_size_bytes = int(cleanup_threshold_gigabytes * BYTES_PER_GIGABYTE)

    def main(self) -> None:
        """Sample usage and prune the logs directory until a shutdown is requested."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        while not self.shutdown_requested:
            self._log_system_usage()
            self._cleanup_directory(self.logs_dir, self.max_logs_size_bytes)

            self.sleep(self.logging_period_seconds)

    def _collect_gpu_stats(self) -> dict[str, float | str]:
        """Return per-GPU statistics from 'nvidia-smi', empty if it is unavailable."""
        if not shutil.which("nvidia-smi"):
            return {}

        stats: dict[str, float | str] = {}
        command = [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,temperature.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        try:
            output = subprocess.check_output(command, encoding="utf-8").strip().split("\n")
        except Exception:
            logger.exception("Could not fetch GPU stats via nvidia-smi.")
            return {}

        for index, line in enumerate(output):
            values = [value.strip() for value in line.split(",")]
            if len(values) != NVIDIA_SMI_FIELD_COUNT:
                continue

            stats[f"gpu_{index}_name"] = values[0]
            stats[f"gpu_{index}_load_percent"] = float(values[1])
            stats[f"gpu_{index}_temp_c"] = float(values[2])

            memory_used = float(values[3])
            memory_total = float(values[4])
            stats[f"gpu_{index}_mem_used_gb"] = round(memory_used / 1024, 2)
            stats[f"gpu_{index}_mem_percent"] = (
                round((memory_used / memory_total) * 100, 2) if memory_total > 0 else 0.0
            )

        return stats

    def _collect_temperature_stats(self) -> dict[str, float]:
        """Return per-sensor temperatures in degrees Celsius, empty if unsupported."""
        if not hasattr(psutil, "sensors_temperatures"):
            return {}

        temperatures = psutil.sensors_temperatures()
        if not temperatures:
            return {}

        return {
            f"temp_{group_name}_{(sensor.label or str(index)).replace(' ', '_')}_c": sensor.current
            for group_name, sensor_list in temperatures.items()
            for index, sensor in enumerate(sensor_list)
        }

    def _log_system_usage(self) -> None:
        """Log one line of CPU, memory, disk and accelerator utilisation as JSON."""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            disk_io = psutil.disk_io_counters()
            cpu_freq = psutil.cpu_freq()

            stats: dict[str, object] = {
                "date_time": datetime.now().astimezone().strftime("%Y%m%d_%H%M%S"),
                "cpu_total_percent": psutil.cpu_percent(interval=None),
                "cpu_per_core_percent": psutil.cpu_percent(interval=None, percpu=True),
                "cpu_freq_mhz": round(cpu_freq.current, 2) if cpu_freq else None,
                "ram_percent": memory.percent,
                "ram_used_gb": round(memory.used / BYTES_PER_GIGABYTE, 2),
                "ram_total_gb": round(memory.total / BYTES_PER_GIGABYTE, 2),
                "swap_percent": swap.percent,
                "swap_used_gb": round(swap.used / BYTES_PER_GIGABYTE, 2),
                "disk_capacity_percent": psutil.disk_usage("/").percent,
                "disk_read_gb": round(disk_io.read_bytes / BYTES_PER_GIGABYTE, 2) if disk_io else 0.0,
                "disk_write_gb": round(disk_io.write_bytes / BYTES_PER_GIGABYTE, 2) if disk_io else 0.0,
            }
            stats.update(self._collect_temperature_stats())
            stats.update(self._collect_gpu_stats())

            logger.info(f"SYSTEM_USAGE: {json.dumps(stats)}")

        except Exception:
            logger.exception("Failed to log system usage.")

    def _cleanup_directory(self, directory: Path, max_size_bytes: int) -> None:
        """Delete the oldest files under 'directory' until it fits inside 'max_size_bytes'."""
        try:
            total_size = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())

            if total_size <= max_size_bytes:
                logger.debug(
                    f"Directory {directory} size {total_size / BYTES_PER_GIGABYTE:.2f} "
                    f"of {max_size_bytes / BYTES_PER_GIGABYTE:.2f} GB."
                )
                return

            logger.warning(
                f"Directory {directory} exceeded limit ({total_size / BYTES_PER_GIGABYTE:.2f} GB). Cleaning up..."
            )

            files = sorted((path for path in directory.rglob("*") if path.is_file()), key=lambda p: p.stat().st_mtime)

            for file_path in files:
                try:
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    total_size -= file_size
                    logger.debug(f"Deleted old file: {file_path.name}")
                except OSError:
                    logger.exception(f"Failed to delete {file_path}.")

                if total_size <= max_size_bytes:
                    break

            for path in sorted(directory.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                if path.is_dir() and not any(path.iterdir()):
                    with contextlib.suppress(OSError):
                        path.rmdir()

        except Exception:
            logger.exception(f"Directory cleanup failed for {directory}.")
