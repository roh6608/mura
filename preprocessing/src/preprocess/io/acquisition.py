"""One acquisition's files on disk, loaded together or not at all."""

from dataclasses import dataclass
from pathlib import Path

from preprocess.calibrate import tofaxis
from preprocess.core.preprocessing_types import AuxPaths, ToFAxis
from preprocess.io import auxfiles, fitsframe
from preprocess.io.manifest import RunRecord


class IncompleteAcquisitionError(ValueError):
    """An acquisition whose frames or sidecars cannot be read."""


@dataclass(frozen=True)
class Acquisition:
    """An acquisition's ToF frames and the axis its sidecars define, complete by construction."""

    record: RunRecord
    frame_paths: list[str]
    axis: ToFAxis

    @classmethod
    def load(cls, record: RunRecord) -> "Acquisition":
        """Read the shutter sidecars and list the frames, refusing an acquisition that is missing any of them."""
        run_dir = Path(record.path)
        counter = record.counter or None
        if not run_dir.is_dir():
            raise IncompleteAcquisitionError(
                f"Not a directory: {run_dir}. It should hold the numbered *.fits frames and their "
                "*_ShutterTimes.txt sidecars."
            )

        aux = _select_sidecars(run_dir, counter)
        try:
            shutter = auxfiles.read_shutter_times(aux.shutter_times)
            n_trig = auxfiles.read_shutter_counts(aux.shutter_count, shutter.n_windows)
        except (OSError, ValueError) as exc:
            raise IncompleteAcquisitionError(
                f"Cannot read the shutter sidecars of acquisition {aux.counter}: {exc}. Both "
                f"{Path(aux.shutter_times).name} and {Path(aux.shutter_count).name} must be present and well "
                "formed."
            ) from exc

        frame_paths = fitsframe.iter_frame_paths(run_dir, counter=counter)
        if not frame_paths:
            raise IncompleteAcquisitionError(
                f"No numbered *.fits frames in {run_dir}"
                + (f" for acquisition counter {counter}" if counter else "")
                + "."
            )

        try:
            bin_width_s = fitsframe.read_header(frame_paths[0]).timebin_s
            axis = tofaxis.build_tof_axis(shutter, n_trig, bin_width_s)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise IncompleteAcquisitionError(
                f"Cannot build the ToF axis of {run_dir}: {exc!r}. Check that {Path(frame_paths[0]).name} is a "
                "complete FITS file with a TIMEBIN card and that the sidecars agree with it."
            ) from exc
        return cls(record=record, frame_paths=frame_paths, axis=axis)


def _select_sidecars(run_dir: Path, counter: str | None) -> AuxPaths:
    """Return the sidecars of the acquisition asked for."""
    try:
        candidates = auxfiles.find_acquisitions(run_dir)
    except (OSError, ValueError) as exc:
        raise IncompleteAcquisitionError(
            f"No readable acquisition in {run_dir}: {exc}. A run directory shoudld hold "
            "<run>_<NNN>_ShutterTimes.txt with matching _ShutterCount.txt and _Spectra.txt files."
        ) from exc

    if counter is None:
        return candidates[0]
    found = next((a for a in candidates if a.counter == counter), None)
    if found is None:
        available = ", ".join(a.counter for a in candidates)
        raise IncompleteAcquisitionError(
            f"No acquisition {counter!r} in {run_dir}; it holds {len(candidates)} ({available}). Select one of "
            "those, or leave the counter unset to take the first."
        )
    return found
