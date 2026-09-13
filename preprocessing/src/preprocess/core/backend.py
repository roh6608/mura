"""CPU/GPU array-module indirection.

Every array expression goes through a handle 'xp' that is either numpy or cupy, with
transfers explicit at the boundaries.
"""

import os
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

ArrayModule = ModuleType

_DEVICE_ENV = "MURA_DEVICE"


def get_backend(device: str = "auto") -> tuple[Any, str]:
    """
    Return '(array_module, device_name)'.

    'device' is '"auto"' (env var, else GPU if usable, else CPU), '"cpu"',
    or '"gpu"'/'"cuda"'.  Requesting a GPU that is not usable raises rather
    than silently falling back.
    """
    if device == "auto":
        device = os.environ.get(_DEVICE_ENV, "auto")

    if device in ("cpu", "numpy"):
        return np, "cpu"

    want_gpu = device in ("gpu", "cuda")

    try:
        import cupy as cp  # noqa: PLC0415

        if cp.cuda.runtime.getDeviceCount() > 0:
            return cp, "gpu"
        if want_gpu:
            raise RuntimeError("cupy imported but no CUDA device is visible")
    except Exception as exc:
        if want_gpu:
            raise RuntimeError(f"GPU backend requested but cupy/CUDA is unusable: {exc}") from exc

    return np, "cpu"


def array_module(*arrays: ArrayLike) -> ArrayModule:
    """Return the array module of the given arrays, dispatching on the arrays themselves."""
    for a in arrays:
        mod = type(a).__module__.split(".")[0]
        if mod == "cupy":
            import cupy as cp  # noqa: PLC0415

            return cp
    return np


def check_same_module(*arrays: ArrayLike, where: str = "") -> ArrayModule:
    """Assert every argument lives on the same device; return that module."""
    mods = {type(a).__module__.split(".")[0] for a in arrays if a is not None}
    mods = {m if m in ("numpy", "cupy") else "numpy" for m in mods}
    if len(mods) > 1:
        raise TypeError(
            f"mixed array modules {sorted(mods)}"
            + (f" in {where}" if where else "")
            + " -- move everything to one device before combining"
        )
    return array_module(*[a for a in arrays if a is not None])


def asnumpy(a: ArrayLike) -> NDArray[Any]:
    """Device-to-host copy. 'numpy' input passes through untouched."""
    get = getattr(a, "get", None)
    return get() if callable(get) else np.asarray(a)


def free_bytes(xp: ArrayModule) -> int | None:
    """Free device memory in bytes, or 'None' on CPU."""
    if xp is np:
        return None
    free, _total = xp.cuda.runtime.memGetInfo()
    return int(free)
