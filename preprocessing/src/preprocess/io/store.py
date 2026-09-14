"""HDF5 storage for calibrated transmission cubes. One file per projection, plus a per-experiment virtual dataset."""

import json
from pathlib import Path
from types import TracebackType
from typing import Self

import h5py
import numpy as np
from numpy.typing import NDArray
from shared.core.h5 import dataset

from preprocess.core.constants import CUBE_SCHEMA_VERSION, INDEX_NAME
from preprocess.core.preprocessing_types import CubeLayout, ProjectionMeta, ToFAxis


def _as_float(v: object) -> float:
    """Attribute to float, mapping absent/empty to NaN."""
    if v is None or (isinstance(v, (str, bytes)) and not v):
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


class ProjectionWriter:
    """Streaming writer for one projection, used as a context manager."""

    def __init__(
        self,
        path: str | Path,
        axis: ToFAxis,
        meta: ProjectionMeta,
        mask: NDArray[np.uint8],
        layout: CubeLayout | None = None,
    ) -> None:
        """Prepare a cube writer for 'path' without opening the file."""
        layout = layout or CubeLayout()
        shape_yx, band, chunk_yx = layout.shape_yx, layout.band, layout.chunk_yx
        compression, store_variance = layout.compression, layout.store_variance

        self.path = Path(path)
        self.axis = axis
        self.meta = meta
        self.mask = mask
        self.shape = (axis.n_bins, *shape_yx)

        chunks = (band, chunk_yx, chunk_yx)
        self.chunks = tuple(min(c, s) for c, s in zip(chunks, self.shape, strict=False))
        self.compression = compression
        self.store_variance = store_variance
        self._f = None

    def __enter__(self) -> Self:
        """Open the temporary output file and create its datasets."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._tmp = self.path.with_name(self.path.name + ".partial")
        self._f = h5py.File(self._tmp, "w")
        f = self._f
        f.attrs["schema_version"] = CUBE_SCHEMA_VERSION
        for k, v in self.meta.to_attrs().items():
            f.attrs[k] = v

        comp = None if self.compression in (None, "none") else self.compression
        kw = {"chunks": self.chunks, "compression": comp, "shuffle": comp is not None}
        self.T = f.create_dataset("T", shape=self.shape, dtype=np.float32, **kw)
        self.T.attrs["long_name"] = "transmission = sample / open beam"
        self.T.attrs["normalisation"] = "none"
        if self.store_variance:
            self.V = f.create_dataset("varT", shape=self.shape, dtype=np.float32, **kw)
            self.V.attrs["long_name"] = "variance of T"
        else:
            self.V = None

        g = f.create_group("axis")
        g.create_dataset("tof_s", data=self.axis.t_start_s)
        g.create_dataset("lambda_a", data=self.axis.wavelength_a(self.meta.flight_path_m, self.meta.t0_s))
        g.create_dataset("window_id", data=self.axis.window_id)
        g.create_dataset("n_trig_per_bin", data=self.axis.trig_per_bin())
        g.attrs["bin_width_s"] = self.axis.bin_width_s
        g.attrs["gaps_s"] = np.asarray(self.axis.gaps_s, dtype=np.float64)
        g.attrs["window_bins"] = np.asarray([sl.stop - sl.start for sl in self.axis.window_slices], dtype=np.int64)

        f.create_dataset("mask", data=self.mask, compression="lzf")
        f["mask"].attrs["bits"] = "1=dead 2=hot 4=seam 8=border 16=saturated"
        return self

    def write_band(self, sl: slice, t: NDArray[np.float32], var: NDArray[np.float32] | None = None) -> None:
        """Write one band of transmission, and its variance when tracked."""
        self.T[sl] = t
        if self.V is not None and var is not None:
            self.V[sl] = var

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        """Close the file and move the temporary cube into place unless an exception occurred."""
        if self._f is not None:
            self._f.close()
            self._f = None
        if exc_type is None:
            self._tmp.replace(self.path)
        else:
            self._tmp.unlink(missing_ok=True)
        return False


def build_index(experiment_dir: str | Path, out_path: str | Path | None = None) -> Path:
    """Virtual dataset presenting every projection as one logical array."""
    experiment_dir = Path(experiment_dir)
    files = sorted(p for p in (experiment_dir / "proj").glob("*.h5") if not p.name.endswith(".partial"))
    if not files:
        raise FileNotFoundError(f"no projections under {experiment_dir / 'proj'}")

    entries = []
    for p in files:
        with h5py.File(p, "r") as f:
            entries.append(
                {
                    "path": p,
                    "angle": _as_float(f.attrs.get("angle_deg")),
                    "run": int(f.attrs["run_number"]),
                    "acq": str(f.attrs["acq"]),
                    "shape": f["T"].shape,
                    "dtype": f["T"].dtype,
                    "lambda_a": dataset(f, "axis/lambda_a")[:],
                    "window_id": dataset(f, "axis/window_id")[:],
                }
            )
    entries.sort(key=lambda e: (e["angle"], e["run"], e["acq"]))

    shp = entries[0]["shape"]
    if any(e["shape"] != shp for e in entries):
        raise ValueError("projections have differing shapes; cannot build one index")
    lambda_a, window_id = entries[0]["lambda_a"], entries[0]["window_id"]
    if any(not np.array_equal(e["lambda_a"], lambda_a) for e in entries):
        raise ValueError("projections have differing wavelength axes; cannot build one index")

    out_path = Path(out_path or experiment_dir / INDEX_NAME)
    layout = h5py.VirtualLayout(shape=(len(entries), *shp), dtype=entries[0]["dtype"])
    for i, e in enumerate(entries):
        layout[i] = h5py.VirtualSource(str(e["path"]), "T", shape=shp)

    with h5py.File(out_path, "w") as f:
        f.create_virtual_dataset("T", layout, fillvalue=np.nan)
        f.create_dataset("angle_deg", data=np.array([e["angle"] for e in entries]))
        f.create_dataset("run_number", data=np.array([e["run"] for e in entries]))
        f.create_dataset("acq", data=np.array([e["acq"].encode() for e in entries]))
        g = f.create_group("axis")
        g.create_dataset("lambda_a", data=lambda_a)
        g.create_dataset("window_id", data=window_id)
        f.attrs["n_projections"] = len(entries)
        f.attrs["schema_version"] = CUBE_SCHEMA_VERSION
        f.attrs["note"] = "virtual dataset over proj/*.h5; keep them beside this file"
    return out_path


def write_experiment_meta(experiment_dir: str | Path, cfg_json: str, extra: dict | None = None) -> Path:
    """Config for a whole experiment, next to its projections."""
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    p = experiment_dir / "config.json"
    payload = {"schema_version": CUBE_SCHEMA_VERSION, "config": json.loads(cfg_json)}
    if extra:
        payload.update(extra)
    p.write_text(json.dumps(payload, indent=2, default=str))
    return p
