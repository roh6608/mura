"""Parsing of VENUS/Pixet run-directory and file names; the observed layouts are in docs/naming.md."""

from preprocess.core.constants import (
    ACQ_AUX_RE,
    ACQ_FRAME_RE,
    ANGLE_RE,
    CHARGE_RE,
    FRAME_RE,
    LAMBDA_MIN_RE,
    OPEN_BEAM_RE,
    RUN_DIRNAME_RE,
)
from preprocess.core.preprocessing_types import RunKey


def parse_run_dirname(dirname: str) -> RunKey | None:
    """Parse a run-directory name, or return 'None' if it is not one."""
    m = RUN_DIRNAME_RE.match(dirname)
    if m is None:
        return None

    rest = m.group("rest")
    angle_deg = seq = ob_position = None

    a = ANGLE_RE.match(rest)
    if a is not None:
        angle_deg = int(a.group("deg")) + int(a.group("mdeg")) / 1000.0
        seq = int(a.group("seq"))
        body = a.group("body")
    else:
        o = OPEN_BEAM_RE.match(rest)
        if o is not None:
            ob_position = int(o.group("pos"))
            body = o.group("body")
        else:
            body = rest

    charge = CHARGE_RE.search(rest)
    lam = LAMBDA_MIN_RE.search(rest)

    return RunKey(
        dirname=dirname,
        date=m.group("date"),
        run_number=int(m.group("run")),
        body=body,
        nominal_angle_deg=angle_deg,
        seq=seq,
        ob_position=ob_position,
        charge_c=(int(charge.group(1)) + int(charge.group(2)) / 1000.0) if charge else None,
        lambda_min_a=(int(lam.group(1)) + int(lam.group(2)) / 1000.0) if lam else None,
    )


def frame_index(filename: str) -> int | None:
    """ToF bin index of a numbered frame, or 'None' if the name is not one."""
    m = FRAME_RE.search(filename)
    return int(m.group(1)) if m else None


def acq_counter(filename: str) -> str | None:
    """Acquisition counter of a frame or sidecar file."""
    m = ACQ_FRAME_RE.search(filename) or ACQ_AUX_RE.search(filename)
    return m.group(1) if m else None
