"""The time-of-flight axis: shutter windows, wavelength, and edge trimming."""

import numpy as np
from numpy.typing import NDArray

from preprocess.core.constants import H_OVER_M, WINDOW_BIN_COUNT_TOLERANCE
from preprocess.core.preprocessing_types import AxisReport, EdgeAnomaly, ShutterTimes, ToFAxis


def build_tof_axis(shutter: ShutterTimes, n_trig: NDArray[np.int64], bin_width_s: float) -> ToFAxis:
    """Build the ToF axis from shutter timings."""
    n_w = shutter.n_windows
    if len(n_trig) != n_w:
        raise ValueError(f"n_trig has {len(n_trig)} entries, shutter has {n_w} windows")

    starts: list[float] = []
    counts: list[int] = []
    t = 0.0
    for w in range(n_w):
        t += float(shutter.delay_s[w])
        starts.append(t)
        exact = float(shutter.duration_s[w]) / bin_width_s
        nb = round(exact)
        if abs(exact - nb) > WINDOW_BIN_COUNT_TOLERANCE:
            raise ValueError(
                f"window {w}: duration {shutter.duration_s[w]:.9g} s is {exact:.6f} bins "
                f"of {bin_width_s:.6g} s, not an integer"
            )
        counts.append(nb)
        t += float(shutter.duration_s[w])

    n_bins = int(sum(counts))
    t_start = np.empty(n_bins, dtype=np.float64)
    window_id = np.empty(n_bins, dtype=np.int8)
    slices: list[slice] = []

    off = 0
    for w, nb in enumerate(counts):
        sl = slice(off, off + nb)
        slices.append(sl)
        t_start[sl] = starts[w] + np.arange(nb, dtype=np.float64) * bin_width_s
        window_id[sl] = w
        off += nb

    return ToFAxis(
        t_start_s=t_start,
        bin_width_s=float(bin_width_s),
        window_id=window_id,
        window_slices=tuple(slices),
        n_trig=np.asarray(n_trig, dtype=np.int64),
    )


def cross_validate(
    axis: ToFAxis,
    spectra_tof_s: NDArray[np.float64],
    header_tof_s: NDArray[np.float64] | None = None,
    header_indices: NDArray[np.int64] | None = None,
) -> AxisReport:
    """Measure a built axis against 'Spectra.txt' and, optionally, sampled frame headers; the report judges."""
    max_dev_spectra_s, worst_spectra_bin = float("nan"), -1
    if spectra_tof_s.size == axis.n_bins:
        dev = np.abs(spectra_tof_s - axis.t_start_s)
        max_dev_spectra_s, worst_spectra_bin = float(dev.max()), int(dev.argmax())

    max_dev_headers_s, worst_header_frame = None, None
    if header_tof_s is not None:
        idx = np.arange(axis.n_bins) if header_indices is None else np.asarray(header_indices)
        dev = np.abs(np.asarray(header_tof_s) - axis.t_start_s[idx])
        max_dev_headers_s, worst_header_frame = float(dev.max()), int(idx[dev.argmax()])

    return AxisReport(
        n_bins=axis.n_bins,
        window_bins=tuple(int(sl.stop - sl.start) for sl in axis.window_slices),
        boundary_indices=tuple(int(sl.start) for sl in axis.window_slices[1:]),
        spectra_rows=int(spectra_tof_s.size),
        max_dev_spectra_s=max_dev_spectra_s,
        worst_spectra_bin=worst_spectra_bin,
        max_dev_headers_s=max_dev_headers_s,
        worst_header_frame=worst_header_frame,
    )


def measure_edge_anomaly(axis: ToFAxis, spectra_counts: NDArray[np.float64], n_ref: int = 5) -> list[EdgeAnomaly]:
    """Quantify how anomalous the first and last bin of each window are, against the mean of the 'n_ref' beside them."""
    out: list[EdgeAnomaly] = []
    for w, sl in enumerate(axis.window_slices):
        lo, hi = sl.start, sl.stop
        if hi - lo < n_ref + 2:
            out.append(EdgeAnomaly(window=w, n_bins=hi - lo, first_ratio=float("nan"), last_ratio=float("nan")))
            continue
        first_ref = float(spectra_counts[lo + 1 : lo + 1 + n_ref].mean())
        last_ref = float(spectra_counts[hi - 1 - n_ref : hi - 1].mean())
        out.append(
            EdgeAnomaly(
                window=w,
                n_bins=hi - lo,
                first_ratio=float(spectra_counts[lo]) / first_ref if first_ref else float("nan"),
                last_ratio=float(spectra_counts[hi - 1]) / last_ref if last_ref else float("nan"),
            )
        )
    return out


def tof_to_wavelength_a(t_s: NDArray[np.float64] | float, flight_path_m: float, t0_s: float) -> NDArray[np.float64]:
    """'lambda[A] = 3956.034 * (t + t0) / L'."""
    return H_OVER_M * (np.asarray(t_s, dtype=np.float64) + t0_s) / flight_path_m
