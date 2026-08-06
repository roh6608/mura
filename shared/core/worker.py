import multiprocessing
from abc import ABC, abstractmethod

from loguru import logger


class Worker(ABC):
    """Base class for workers.
    A worker runs its ``main`` method in a dedicated child process and shuts
    down cooperatively.
    """

    def __init__(self, name: str | None = None) -> None:
        self.name = name or type(self).__name__
        self._stop_event = multiprocessing.Event()
        self._process: multiprocessing.Process | None = None

    @abstractmethod
    def main(self) -> None:
        """Worker body"""

    @property
    def shutdown_requested(self) -> bool:
        return self._stop_event.is_set()

    @property
    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def start(self) -> None:
        if self.is_alive:
            logger.warning(f"{self.name} is already running")
            return

        self._stop_event.clear()

        process = multiprocessing.Process(target=self._run, name=self.name, daemon=True)
        process.start()
        self._process = process

    def request_shutdown(self) -> None:
        """Signal the worker to stop without waiting for it to exit."""
        self._stop_event.set()

    def stop(self, timeout: float | None = None) -> None:
        """Request shutdown and wait for the worker process to exit.

        If the process is still alive after ``timeout`` seconds it is
        terminated so no orphan is left behind.
        """
        self.request_shutdown()
        if self._process is None:
            return

        self._process.join(timeout)
        if self._process.is_alive():
            logger.warning(f"{self.name} did not stop within {timeout} s, terminating")
            self._process.terminate()
            self._process.join()

    def sleep(self, seconds: float) -> bool:
        """Cooperative sleep: returns False immediately if a stop is requested."""
        return not self._stop_event.wait(seconds)

    def _run(self) -> None:
        logger.info(f"{self.name} started")
        try:
            self.main()
        except Exception:  # noqa: BLE001 - crash barrier for the child process
            logger.exception(f"{self.name} crashed")
        finally:
            logger.info(f"{self.name} stopped")
