import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import psutil
from loguru import logger

from shared.core.worker import Worker


class SystemMonitorWorker(Worker):
    """Periodically logs system utilisation and keeps the logs directory
    under a size budget by deleting the oldest files first.
    """

    def __init__(
        self,
        logs_dir: Path,
        cleanup_threshold_gigabytes: float,
        logging_period_seconds: float = 60.0,
    ) -> None:
        super().__init__()
        self.logs_dir = logs_dir
        self.logging_period_seconds = logging_period_seconds
        self.max_logs_size_bytes = int(cleanup_threshold_gigabytes * 1024**3)

    def main(self) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        while not self.shutdown_requested:
            self._log_system_usage()
            self._cleanup_directory(self.logs_dir, self.max_logs_size_bytes)

            # Cooperative sleep: wakes immediately when a stop is requested so
            # shutdown isn't blocked for a full logging period.
            self.sleep(self.logging_period_seconds)

    def _log_system_usage(self) -> None:
        try:
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            disk_io = psutil.disk_io_counters()
            cpu_freq = psutil.cpu_freq()

            stats = {
                "date_time": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "cpu_total_percent": psutil.cpu_percent(interval=None),
                "cpu_per_core_percent": psutil.cpu_percent(interval=None, percpu=True),
                "cpu_freq_mhz": round(cpu_freq.current, 2) if cpu_freq else None,
                "ram_percent": mem.percent,
                "ram_used_gb": round(mem.used / (1024**3), 2),
                "ram_total_gb": round(mem.total / (1024**3), 2),
                "swap_percent": swap.percent,
                "swap_used_gb": round(swap.used / (1024**3), 2),
                "disk_capacity_percent": psutil.disk_usage("/").percent,
                "disk_read_gb": round(disk_io.read_bytes / (1024**3), 2)
                if disk_io
                else 0.0,
                "disk_write_gb": round(disk_io.write_bytes / (1024**3), 2)
                if disk_io
                else 0.0,
            }

            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    for group_name, sensor_list in temps.items():
                        for i, sensor in enumerate(sensor_list):
                            label = sensor.label or str(i)

                            clean_label = label.replace(" ", "_")

                            stats[f"temp_{group_name}_{clean_label}_c"] = sensor.current

            if shutil.which("nvidia-smi"):
                try:
                    cmd = [
                        "nvidia-smi",
                        "--query-gpu=name,utilization.gpu,temperature.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ]
                    output = (
                        subprocess.check_output(cmd, encoding="utf-8")
                        .strip()
                        .split("\n")
                    )

                    for i, line in enumerate(output):
                        vals = [x.strip() for x in line.split(",")]
                        if len(vals) == 5:
                            stats[f"gpu_{i}_name"] = vals[0]
                            stats[f"gpu_{i}_load_percent"] = float(vals[1])
                            stats[f"gpu_{i}_temp_c"] = float(vals[2])

                            mem_used = float(vals[3])
                            mem_total = float(vals[4])
                            stats[f"gpu_{i}_mem_used_gb"] = round(mem_used / 1024, 2)
                            stats[f"gpu_{i}_mem_percent"] = (
                                round((mem_used / mem_total) * 100, 2)
                                if mem_total > 0
                                else 0.0
                            )
                except Exception as e:  # noqa: BLE001 GPU stats are best effort
                    logger.debug(f"Could not fetch GPU stats via nvidia-smi: {e}")

            logger.info(f"SYSTEM_USAGE: {json.dumps(stats)}")

        except Exception as e:  # noqa: BLE001 monitoring must never kill the worker
            logger.error(f"Failed to log system usage: {e}")

    def _cleanup_directory(self, directory: Path, max_size_bytes: int) -> None:
        try:
            total_size = sum(
                f.stat().st_size for f in directory.rglob("*") if f.is_file()
            )

            if total_size <= max_size_bytes:
                logger.debug(
                    f"Directory {directory} size {total_size / 1024**3:.2f} of "
                    f"{max_size_bytes / 1024**3:.2f} GB."
                )
                return

            logger.warning(
                f"Directory {directory} exceeded limit ({total_size / 1024**3:.2f} GB). Cleaning up..."
            )

            files = sorted(
                (f for f in directory.rglob("*") if f.is_file()),
                key=lambda f: f.stat().st_mtime,
            )

            for f in files:
                try:
                    file_size = f.stat().st_size
                    f.unlink()
                    total_size -= file_size
                    logger.debug(f"Deleted old file: {f.name}")
                except Exception as e:  # noqa: BLE001 keep cleaning the remaining files
                    logger.error(f"Failed to delete {f}: {e}")

                if total_size <= max_size_bytes:
                    break

            for d in sorted(
                directory.rglob("*"), key=lambda x: len(x.parts), reverse=True
            ):
                if d.is_dir() and not any(d.iterdir()):
                    try:
                        d.rmdir()
                    except OSError:
                        pass
        except Exception as e:  # noqa: BLE001 cleanup must never kill the worker
            logger.error(f"Directory cleanup failed for {directory}: {e}")
