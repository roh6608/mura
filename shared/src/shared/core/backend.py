"""Select NumPy or CuPy and make array transfers explicit."""

import importlib
import os
from types import ModuleType
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray


def get_backend(device: str = "auto") -> tuple[ModuleType, str]:
    """Select a backend, raising when an explicitly requested GPU is unusable."""
    if device == "auto":
        device = os.environ.get("MURA_DEVICE", "auto")
    if device in ("cpu", "numpy"):
        return np, "cpu"

    want_gpu = device in ("gpu", "cuda")
    try:
        cp = importlib.import_module("cupy")

        if cp.cuda.runtime.getDeviceCount() > 0:
            cp.zeros(1)
            return cp, "gpu"
        if want_gpu:
            raise RuntimeError("cupy imported but no CUDA device is visible")
    except Exception as exc:
        if want_gpu:
            raise RuntimeError(f"GPU backend requested but cupy/CUDA is unusable: {exc}") from exc

    return np, "cpu"


def array_module(*arrays: ArrayLike) -> ModuleType:
    """Return the array module of the inputs without importing CuPy for NumPy arrays."""
    for array in arrays:
        if type(array).__module__.split(".")[0] == "cupy":
            return importlib.import_module("cupy")
    return np


def check_same_module(*arrays: ArrayLike, where: str = "") -> ModuleType:
    """Reject inputs on mixed devices and return their array module."""
    modules = {type(array).__module__.split(".")[0] for array in arrays if array is not None}
    modules = {module if module in ("numpy", "cupy") else "numpy" for module in modules}
    if len(modules) > 1:
        raise TypeError(
            f"mixed array modules {sorted(modules)}"
            + (f" in {where}" if where else "")
            + " -- move everything to one device before combining"
        )
    return array_module(*[array for array in arrays if array is not None])


def asnumpy(array: ArrayLike) -> NDArray[Any]:
    """Return a host array, copying device arrays through their 'get' method."""
    get = getattr(array, "get", None)
    return cast("NDArray[Any]", get()) if callable(get) else np.asarray(array)


def free_bytes(xp: ModuleType, *, include_cpu: bool = False) -> int | None:
    """Return free device bytes, optionally estimating available CPU memory."""
    if xp is np:
        if not include_cpu:
            return None
        try:
            return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (AttributeError, ValueError, OSError):
            return None
    free, _total = xp.cuda.runtime.memGetInfo()
    return int(free)
