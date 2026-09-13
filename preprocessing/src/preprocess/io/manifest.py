"""Run discovery and validation."""

import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from preprocess.calibrate import tofaxis
from preprocess.core.constants import (
    FITS_FRAME_BYTES,
    NAME_ROUNDING_TOLERANCE,
    ROTATION_STAGE,
    SPECTRA_ATOL_S,
    TOF_CARD_ATOL_S,
)
from preprocess.core.preprocessing_types import Calibration, Kind, RunKey, Status
from preprocess.io import auxfiles, fitsframe, nexus
from preprocess.io.naming import frame_index, parse_run_dirname


@dataclass
class RunRecord:
    """One acquisition: identity, sidecar data, and validation verdict."""

    key: RunKey
    path: str
    experiment: str
    kind: Kind
    counter: str = ""
    angle_deg: float | None = None
    calibration: Calibration | None = None

    n_frames: int = -1
    n_windows: int = -1
    window_bins: tuple[int, ...] = ()
    n_trig: tuple[int, ...] = ()
    bin_width_s: float = float("nan")
    t_first_s: float = float("nan")
    t_last_s: float = float("nan")

    status: Status = Status.UNCHECKED
    problems: tuple[str, ...] = ()
    checks: dict = field(default_factory=dict)

    @property
    def run_number(self) -> int:
        """Return the run number this entry belongs to."""
        return self.key.run_number

    @property
    def uid(self) -> str:
        """Identifier run number plus acquisition counter."""
        return f"{self.key.run_number}_{self.counter}" if self.counter else str(self.key.run_number)

    def to_dict(self) -> dict:
        """Return the entry as a plain dictionary, including its derived 'uid'."""
        d = dataclasses.asdict(self)
        d["key"] = dataclasses.asdict(self.key)
        d["uid"] = self.uid
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RunRecord":
        """Rebuild an entry from 'to_dict' output, restoring the enums, tuples and nested types JSON flattened."""
        calibration = d["calibration"]
        return cls(
            key=RunKey(**d["key"]),
            path=d["path"],
            experiment=d["experiment"],
            kind=Kind(d["kind"]),
            counter=d["counter"],
            angle_deg=d["angle_deg"],
            calibration=Calibration(**calibration) if calibration is not None else None,
            n_frames=d["n_frames"],
            n_windows=d["n_windows"],
            window_bins=tuple(d["window_bins"]),
            n_trig=tuple(d["n_trig"]),
            bin_width_s=d["bin_width_s"],
            t_first_s=d["t_first_s"],
            t_last_s=d["t_last_s"],
            status=Status(d["status"]),
            problems=tuple(d["problems"]),
            checks=d["checks"],
        )


def _records_for_dir(run_dir: Path, key: RunKey, experiment: str, kind: Kind) -> list[RunRecord]:
    """One record per acquisition in a run directory."""
    try:
        acqs = auxfiles.find_acquisitions(run_dir)
    except (FileNotFoundError, ValueError) as exc:
        rec = RunRecord(key=key, path=str(run_dir), experiment=experiment, kind=kind)
        rec.status = Status.QUARANTINE
        rec.problems = (f"sidecar files: {exc}",)
        return [rec]
    return [RunRecord(key=key, path=str(run_dir), experiment=experiment, kind=kind, counter=a.counter) for a in acqs]


def discover(root: str | os.PathLike, kind: Kind) -> list[RunRecord]:
    """Find every acquisition beneath an experiment root."""
    out: list[RunRecord] = []
    root = Path(root)
    if not root.is_dir():
        return out

    for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
            key = parse_run_dirname(run_dir.name)
            if key is not None:
                out.extend(_records_for_dir(run_dir, key, exp_dir.name, kind))
                continue

            for sub in sorted(p for p in run_dir.iterdir() if p.is_dir()):
                k2 = parse_run_dirname(sub.name)
                if k2 is not None:
                    out.extend(_records_for_dir(sub, k2, exp_dir.name, kind))
    return out


def validate_run(rec: RunRecord, *, nexus_dir: Path, check_sizes: bool = True, n_header_samples: int = 12) -> RunRecord:  # noqa: PLR0911
    """Validate one acquisition and fill in its axis metadata."""
    if rec.status == Status.QUARANTINE:
        return rec

    problems: list[str] = []
    checks: dict = {}

    try:
        acqs = {a.counter: a for a in auxfiles.find_acquisitions(rec.path)}
        aux = acqs[rec.counter] if rec.counter else next(iter(acqs.values()))
    except (FileNotFoundError, ValueError, KeyError, StopIteration) as exc:
        rec.status = Status.QUARANTINE
        rec.problems = (f"sidecar files: {exc}",)
        return rec

    try:
        shutter = auxfiles.read_shutter_times(aux.shutter_times)
        n_trig = auxfiles.read_shutter_counts(aux.shutter_count, shutter.n_windows)
        spec_tof, spec_counts = auxfiles.read_spectra(aux.spectra)
    except (ValueError, OSError) as exc:
        rec.status = Status.QUARANTINE
        rec.problems = (f"sidecar parse: {exc}",)
        return rec

    paths = fitsframe.iter_frame_paths(rec.path, counter=rec.counter or None)
    rec.n_frames = len(paths)
    rec.n_windows = shutter.n_windows
    rec.n_trig = tuple(int(x) for x in n_trig)

    if not paths:
        rec.status = Status.QUARANTINE
        rec.problems = ("no numbered ToF frames for this acquisition",)
        return rec

    idx = [frame_index(Path(p).name) for p in paths]
    if idx != list(range(len(idx))):
        present = {i for i in idx if i is not None}
        missing = sorted(set(range(max(present) + 1)) - present)
        problems.append(f"frame indices not contiguous from 0; {len(missing)} missing, e.g. {missing[:5]}")

    try:
        bin_width = fitsframe.read_header(paths[0]).timebin_s
    except (OSError, KeyError, ValueError) as exc:
        rec.status = Status.QUARANTINE
        rec.problems = (f"cannot read TIMEBIN from first frame: {exc}",)
        return rec
    rec.bin_width_s = bin_width

    try:
        axis = tofaxis.build_tof_axis(shutter, n_trig, bin_width)
    except ValueError as exc:
        rec.status = Status.QUARANTINE
        rec.problems = (f"ToF axis: {exc}",)
        return rec

    rec.window_bins = tuple(int(sl.stop - sl.start) for sl in axis.window_slices)
    rec.t_first_s = float(axis.t_start_s[0])
    rec.t_last_s = float(axis.t_start_s[-1])

    if rec.n_frames != axis.n_bins:
        problems.append(f"{rec.n_frames} frames but shutter windows imply {axis.n_bins} bins")

    if check_sizes:
        bad = [p for p in paths if Path(p).stat().st_size != FITS_FRAME_BYTES]
        checks["n_wrong_size"] = len(bad)
        if bad:
            problems.append(f"{len(bad)} frames not {FITS_FRAME_BYTES} B, e.g. {Path(bad[0]).name}")

    if rec.n_frames == axis.n_bins and n_header_samples > 0:
        sample = np.unique(
            np.concatenate(
                [
                    np.linspace(0, rec.n_frames - 1, n_header_samples).astype(int),
                    np.array([sl.start for sl in axis.window_slices]),
                    np.array([sl.stop - 1 for sl in axis.window_slices]),
                ]
            )
        )
        try:
            hdrs = [fitsframe.read_header(paths[i]) for i in sample]
        except (OSError, ValueError) as exc:
            problems.append(f"header sampling failed: {exc}")
        else:
            widths = np.array([h.timebin_s for h in hdrs])
            checks["timebin_unique"] = sorted({float(w) for w in widths})
            if not np.allclose(widths, bin_width, rtol=0, atol=1e-12):
                problems.append(f"TIMEBIN not constant across frames: {checks['timebin_unique']}")

            rep = tofaxis.cross_validate(
                axis, spec_tof, header_tof_s=np.array([h.tof_s for h in hdrs]), header_indices=sample
            )
            checks["max_dev_spectra_ns"] = rep.max_dev_spectra_s * 1e9
            checks["max_dev_headers_ps"] = rep.max_dev_headers_s * 1e12 if rep.max_dev_headers_s is not None else None
            checks["boundary_indices"] = list(rep.boundary_indices)
            if rep.spectra_rows != rep.n_bins:
                problems.append(f"Spectra.txt has {rep.spectra_rows} rows, axis has {rep.n_bins} bins")
            elif not rep.spectra_ok:
                problems.append(
                    f"max |axis - Spectra.txt| = {rep.max_dev_spectra_s * 1e9:.1f} ns exceeds "
                    f"{SPECTRA_ATOL_S * 1e9:.0f} ns at bin {rep.worst_spectra_bin}"
                )
            if not rep.headers_ok and rep.max_dev_headers_s is not None:
                problems.append(
                    f"max |axis - TOF card| = {rep.max_dev_headers_s * 1e12:.1f} ps exceeds "
                    f"{TOF_CARD_ATOL_S * 1e12:.0f} ps at frame {rep.worst_header_frame}"
                )

            for i, h in enumerate(hdrs):
                fi = int(sample[i])
                w = int(axis.window_id[fi])
                if h.n_trigs != int(n_trig[w]):
                    problems.append(
                        f"window {w}: ShutterCount says {int(n_trig[w])} but frame {fi} header says {h.n_trigs}"
                    )
                    break

    if spec_counts.size == axis.n_bins:
        checks["edge_anomaly"] = tofaxis.measure_edge_anomaly(axis, spec_counts)

    problems.extend(_read_nexus(rec, nexus_dir))

    rec.checks = checks
    rec.problems = tuple(problems)
    rec.status = Status.OK if not problems else Status.QUARANTINE
    return rec


def _read_nexus(rec: RunRecord, nexus_dir: Path) -> list[str]:
    """Set the calibration and, for a projection, the stage angle from the run's DASlogs, reporting what is missing."""
    path = nexus.find_nexus(nexus_dir, rec.run_number)
    if path is None:
        return [f"no NeXus file VENUS_{rec.run_number}.nxs.h5 under {nexus_dir}"]
    meta = nexus.read_run_meta(path, with_monitors=False)
    problems: list[str] = []

    calibration = Calibration.from_nexus(meta)
    if calibration is None:
        problems.append("no wavelength calibration: the TIDelay or LambdaMinActual DASlog is missing")
    elif (
        rec.key.lambda_min_a is not None
        and abs(calibration.lambda_min_a - rec.key.lambda_min_a) > NAME_ROUNDING_TOLERANCE
    ):
        problems.append(
            f"name says lambda_min {rec.key.lambda_min_a:.3f} A but the chopper log is {calibration.lambda_min_a:.6f} A"
        )
    else:
        rec.calibration = calibration

    if rec.key.is_projection:
        achieved = meta.rotation_deg.get(ROTATION_STAGE)
        nominal = rec.key.nominal_angle_deg
        if achieved is None:
            problems.append(f"no rotation readback: the {ROTATION_STAGE} DASlog is missing")
        elif nominal is not None and abs(achieved - nominal) > NAME_ROUNDING_TOLERANCE:
            problems.append(f"name says {nominal:.3f} deg but the stage readback is {achieved:.6f} deg")
        else:
            rec.angle_deg = achieved
    return problems


def read_manifest(path: str | os.PathLike) -> list[RunRecord]:
    """Read back the records a scan wrote, in the order it validated them."""
    return [RunRecord.from_dict(d) for d in json.loads(Path(path).read_text())["runs"]]


def write_manifest(records: list[RunRecord], path: str | os.PathLike) -> None:
    """Serialise a manifest, plus a quarantine summary next to it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_acquisitions": len(records),
        "n_ok": sum(r.status == Status.OK for r in records),
        "n_quarantine": sum(r.status == Status.QUARANTINE for r in records),
        "runs": [r.to_dict() for r in records],
    }
    path.write_text(json.dumps(payload, indent=2, default=str))

    bad = {f"{r.path}#{r.counter}": list(r.problems) for r in records if r.status == Status.QUARANTINE}
    (path.parent / "quarantine.json").write_text(json.dumps(bad, indent=2))
