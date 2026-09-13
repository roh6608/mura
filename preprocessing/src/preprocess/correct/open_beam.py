"""Combining the open-beam runs into one reference cube."""

import json
import time
from pathlib import Path
from types import ModuleType

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from preprocess.core.backend import asnumpy
from preprocess.core.config import PreprocessConfig
from preprocess.core.constants import DETECTOR_COLUMNS, DETECTOR_ROWS, DETECTOR_SHAPE, FRAME_FLOAT32_BYTES
from preprocess.correct import overlap
from preprocess.io import stream
from preprocess.io.acquisition import Acquisition


def sum_open_beam(
    acquisitions: list[Acquisition], config: PreprocessConfig, destination: Path, xp: ModuleType, dev: str
) -> dict[str, NDArray[np.float64]]:
    """Build the combined open-beam reference cube, returning each run's corrected white beam by 'uid'."""
    if not acquisitions:
        logger.error("No open-beam acquisition survived to the summing step, so the reference cube would be empty.")
        raise SystemExit(1)

    # Every contributing run must share one ToF axis, or "summing" is meaningless.
    ref_axis = acquisitions[0].axis
    for acquisition in acquisitions[1:]:
        ax = acquisition.axis
        if ax.n_bins != ref_axis.n_bins or not np.allclose(ax.t_start_s, ref_axis.t_start_s):
            logger.error(
                f"Run {acquisition.record.run_number} does not share the reference ToF axis "
                f"({ax.n_bins} bins vs {ref_axis.n_bins}), so its counts cannot be "
                "added to it."
            )
            raise SystemExit(1)

    n = ref_axis.n_bins
    logger.info(
        f"summing {len(acquisitions)} open-beam acquisitions into a {n}x{DETECTOR_ROWS}x{DETECTOR_COLUMNS} reference "
        f"({n * FRAME_FLOAT32_BYTES / 1e9:.2f} GB float32), device {dev}"
    )

    total = np.zeros((n, *DETECTOR_SHAPE), dtype=np.float32)
    var_total = np.zeros((n, *DETECTOR_SHAPE), dtype=np.float32)
    n_trig_runs: dict[str, list[int]] = {}
    white_beams: dict[str, NDArray[np.float64]] = {}
    t_all = time.time()

    for k, acquisition in enumerate(acquisitions, 1):
        record, ax, paths, nt = acquisition.record, acquisition.axis, acquisition.frame_paths, acquisition.axis.n_trig
        acc = overlap.WindowAccumulator(DETECTOR_SHAPE, xp=xp)
        stats = overlap.SaturationStats()
        white_beam = np.zeros(DETECTOR_SHAPE, dtype=np.float64)
        t0 = time.time()
        for w, sl, band in stream.iter_bands(paths, ax, band_size=config.open_beam.band, xp=xp):
            raw_band = band.copy()
            overlap.overlap_correct_band(
                band,
                acc.for_window(w),
                int(nt[w]),
                floor=config.overlap.denominator_floor,
                half_bin=config.overlap.half_bin,
                stats=stats,
                window=w,
            )
            corrected = asnumpy(band)
            total[sl] += corrected
            var_total[sl] += asnumpy(band * band / xp.maximum(raw_band, 1.0))
            white_beam += corrected.sum(axis=0, dtype=np.float64)
        n_trig_runs[str(record.run_number)] = [int(x) for x in nt]
        white_beams[record.uid] = white_beam
        logger.info(
            f"  [{k}/{len(acquisitions)}] run {record.run_number} acq {record.counter}  "
            f"{time.time() - t0:5.1f} s   mean correction x{stats.mean_correction:.4f}   "
            f"saturated px {stats.n_saturating_pixels}"
        )

    out = destination
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, total)
        var_path = out.with_name(out.stem + "_var.npy")
        np.save(var_path, var_total)
    except OSError as exc:
        logger.error(f"Cannot write the open-beam reference to {out}: {exc}.")
        raise SystemExit(1) from exc
    meta = {
        "n_runs": len(acquisitions),
        "runs": [
            {"run": a.record.run_number, "acq": a.record.counter, "position": a.record.key.ob_position}
            for a in acquisitions
        ],
        "n_bins": int(n),
        "shape": [int(n), *DETECTOR_SHAPE],
        "bin_width_s": float(ref_axis.bin_width_s),
        "t_start_s": ref_axis.t_start_s.tolist(),
        "window_id": ref_axis.window_id.tolist(),
        "n_trig_per_run": n_trig_runs,
        "total_counts": float(total.sum()),
        "variance_file": var_path.name,
    }
    (out.with_suffix(".json")).write_text(json.dumps(meta))
    logger.info(f"  total {time.time() - t_all:.1f} s")
    logger.info(
        f"  wrote {out} ({out.stat().st_size / 1e9:.2f} GB), "
        f"{var_path.name} ({var_path.stat().st_size / 1e9:.2f} GB), {out.with_suffix('.json').name}"
    )
    logger.info(f"  combined counts {total.sum():.6g}")
    logger.info(
        f"  statistics gain vs a single run: sqrt({len(acquisitions)}) = {len(acquisitions) ** 0.5:.2f}x "
        "(exact only if the runs are repeats of one field -- see docs/open_beam.md)"
    )
    return white_beams
