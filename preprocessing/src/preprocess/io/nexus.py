"""Run metadata from the NeXus files."""

import os
from pathlib import Path

import h5py
import numpy as np

from preprocess.core.constants import DASLOG_SCALARS, MONITOR_BANKS, NEXUS_FILE_RE
from preprocess.core.preprocessing_types import RunMeta


def find_nexus(nexus_dir: str | os.PathLike, run_number: int) -> str | None:
    """Path to a run's NeXus file, or 'None'."""
    hits = [str(p) for p in Path(nexus_dir).glob(f"VENUS_{run_number}.nxs.h5")]
    return hits[0] if hits else None


def _log_value(f: "h5py.File", name: str) -> float | None:
    """Mean of a DASlog's 'value' dataset, or 'None' if absent/empty."""
    node = f.get(f"/entry/DASlogs/{name}/value")
    if node is None or node.size == 0:
        return None
    v = np.asarray(node[()], dtype=np.float64)
    return float(np.mean(v))


def read_run_meta(nexus_path: str | os.PathLike, *, with_monitors: bool = True) -> RunMeta:
    """Read one run's metadata."""
    m = NEXUS_FILE_RE.search(Path(str(nexus_path)).name)
    meta = RunMeta(run_number=int(m.group(1)) if m else -1, path=str(nexus_path))
    missing: list[str] = []

    with h5py.File(nexus_path, "r") as f:
        for key, node in (
            ("proton_charge", "/entry/proton_charge"),
            ("duration_s", "/entry/duration"),
            ("total_pulses", "/entry/total_pulses"),
        ):
            d = f.get(node)
            if d is None:
                missing.append(node)
                continue
            val = np.asarray(d[()]).ravel()
            if val.size:
                setattr(meta, key, float(val[0]) if key != "total_pulses" else int(val[0]))

        for key, log in DASLOG_SCALARS.items():
            v = _log_value(f, log)
            if v is None:
                missing.append(log)
            else:
                meta.scalars[key] = v

        for i in range(1, 10):
            v = _log_value(f, f"BL10:Mot:rot{i}.RBV")
            if v is not None:
                meta.rotation_deg[f"rot{i}"] = v

        if with_monitors:
            for b in MONITOR_BANKS:
                d = f.get(f"/entry/instrument/{b}/total_counts")
                if d is None:
                    missing.append(f"{b}/total_counts")
                else:
                    meta.monitor_counts[b] = int(np.asarray(d[()]).ravel()[0])

    meta.missing = tuple(missing)
    return meta
