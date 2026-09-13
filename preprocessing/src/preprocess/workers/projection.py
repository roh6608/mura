"""Per-projection driver: raw frames to a calibrated transmission cube."""

import json
import multiprocessing as mp
import os
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from loguru import logger
from shared.core.backend import asnumpy, get_backend

from preprocess.calibrate import tofaxis
from preprocess.core.constants import DETECTOR_SHAPE
from preprocess.core.preprocessing_types import CubeLayout, ProjectionMeta, ProjReport, ProjTask

if TYPE_CHECKING:
    from preprocess.core.config import PreprocessConfig
from preprocess.correct import detector, filters, overlap
from preprocess.io import auxfiles, fitsframe, store, stream


def worker_init() -> None:
    """Pin every nested thread pool to a single thread."""
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = "1"

    cv2.setNumThreads(1)


def process_projection(task: ProjTask) -> ProjReport:
    """Correct one projection and write its transmission cube."""
    t_start = time.time()
    rep = ProjReport(task.run_number, task.counter, task.angle_deg, task.out_path, ok=False)
    cfg = task.config
    try:
        xp, _dev = get_backend(task.device)

        acqs = {a.counter: a for a in auxfiles.find_acquisitions(task.run_dir)}
        aux = acqs[task.counter] if task.counter else next(iter(acqs.values()))
        shutter = auxfiles.read_shutter_times(aux.shutter_times)
        n_trig = auxfiles.read_shutter_counts(aux.shutter_count, shutter.n_windows)
        paths = fitsframe.iter_frame_paths(task.run_dir, counter=task.counter or None)
        bin_width = fitsframe.read_header(paths[0]).timebin_s
        axis = tofaxis.build_tof_axis(shutter, n_trig, bin_width)
        cal = task.calibration
        rep.n_bins = axis.n_bins

        if len(paths) != axis.n_bins:
            rep.message = f"{len(paths)} frames for a {axis.n_bins}-bin axis"
            return rep

        refp = Path(task.ob_reference)
        meta = json.loads(refp.with_suffix(".json").read_text())
        ob = np.load(refp, mmap_mode="r")
        obv = np.load(refp.with_name(refp.stem + "_var.npy"), mmap_mode="r")
        if ob.shape[0] != axis.n_bins or not np.allclose(meta["t_start_s"], axis.t_start_s):
            rep.message = "open-beam reference does not share this projection's ToF axis"
            return rep

        rois = np.load(task.rois_path)
        roi_sample = rois["sample"].ravel()
        roi_air = rois["air"].ravel()
        mask = rois["mask"]

        pmeta = ProjectionMeta(
            run_number=task.run_number,
            acq=task.counter,
            angle_deg=task.angle_deg,
            experiment=task.experiment,
            kind=task.kind,
            flight_path_m=cal.flight_path_m,
            t0_s=cal.t0_s,
            lambda_min_a=cal.lambda_min_a,
            bin_width_s=axis.bin_width_s,
            n_trig=[int(x) for x in n_trig],
            ob_reference=refp.name,
            n_sample_px=int(roi_sample.sum()),
            n_air_px=int(roi_air.sum()),
            mean_correction=1.0,
            n_saturating_px=0,
        )

        acc = overlap.WindowAccumulator(DETECTOR_SHAPE, xp=xp)
        stats = overlap.SaturationStats()
        dstats = filters.DespeckleStats()
        repair_mask = (mask & (detector.DEAD | detector.HOT)) > 0
        repair_mask = repair_mask if repair_mask.any() else None
        repair_mask_dev = xp.asarray(repair_mask) if repair_mask is not None else None
        t_sum = 0.0
        t_n = 0
        air_sum = 0.0
        air_n = 0

        with store.ProjectionWriter(
            task.out_path,
            axis,
            pmeta,
            mask,
            CubeLayout(band=task.band, compression=task.compression, store_variance=task.store_variance),
        ) as w:
            for win, sl, band in stream.iter_bands(paths, axis, band_size=task.band, xp=xp):
                raw = band.copy()
                overlap.overlap_correct_band(
                    band,
                    acc.for_window(win),
                    int(n_trig[win]),
                    floor=cfg.overlap.denominator_floor,
                    half_bin=cfg.overlap.half_bin,
                    stats=stats,
                    window=win,
                )

                if task.despeckle_k > 0:
                    filters.despeckle(band, k=task.despeckle_k, ksize=task.despeckle_ksize, stats=dstats, ref=raw)

                o_counts = xp.asarray(np.array(ob[sl], dtype=np.float32))

                rel = None
                if task.store_variance:
                    rel = 1.0 / xp.maximum(raw, 1.0)
                    vo = xp.asarray(np.array(obv[sl], dtype=np.float32))
                    rel += vo / xp.maximum(o_counts, 1.0) ** 2
                    del vo
                del raw

                ok_px = o_counts > 0
                t = xp.where(ok_px, band / xp.where(ok_px, o_counts, xp.float32(1.0)), xp.float32(np.nan)).astype(
                    xp.float32
                )
                del o_counts

                var = None
                if rel is not None:
                    rel *= t * t
                    var = xp.where(ok_px, rel, xp.float32(np.nan)).astype(xp.float32)
                    del rel
                del ok_px

                if repair_mask is not None:
                    t = filters.repair(t, repair_mask_dev, ksize=task.repair_ksize, passes=task.repair_passes)
                    if var is not None:
                        var = filters.repair(var, repair_mask_dev, ksize=task.repair_ksize, passes=task.repair_passes)

                th = asnumpy(t)
                w.write_band(sl, th, asnumpy(var) if var is not None else None)
                del t, var
                if xp is not np:
                    xp.get_default_memory_pool().free_all_blocks()

                flat = th.reshape(th.shape[0], -1)
                blk = flat[:, roi_sample] if roi_sample.any() else flat[:, :0]
                t_sum += float(np.nansum(blk))
                t_n += int(np.isfinite(blk).sum())
                blk = flat[:, roi_air] if roi_air.any() else flat[:, :0]
                air_sum += float(np.nansum(blk))
                air_n += int(np.isfinite(blk).sum())

        rep.mean_correction = stats.mean_correction
        rep.n_saturating_px = stats.n_saturating_pixels
        rep.n_despeckled = dstats.n_replaced
        rep.frac_counts_despeckled = dstats.frac_counts_removed
        rep.n_repaired_px = int(repair_mask.sum()) if repair_mask is not None else 0
        rep.mean_transmission = t_sum / t_n if t_n else float("nan")
        rep.air_transmission = air_sum / air_n if air_n else float("nan")
        rep.ok = True
    except Exception as exc:
        name = type(exc).__name__
        gpu_side = "OutOfMemory" in name or "CUDA" in name or "cuda" in str(exc).lower()
        if gpu_side and task.device != "cpu":
            logger.warning(f"Projection {task.run_dir} failed on the device with {name}. Retrying on the CPU")
            fallback = replace(task, device="cpu")
            rep2 = process_projection(fallback)
            rep2.message = f"retried on CPU after {name}" + (f"; {rep2.message}" if rep2.message else "")
            rep2.seconds = time.time() - t_start
            return rep2
        logger.exception(f"Projection {task.run_dir} failed.")
        rep.message = f"{name}: {exc}"
    rep.seconds = time.time() - t_start
    return rep


def build_rois(
    ob_reference: str | Path, out_path: str | Path, cfg: "PreprocessConfig", sample_runs: list[tuple[str, str]]
) -> dict:
    """Build the pixel mask and the sample/air ROIs once for a whole experiment."""
    refp = Path(ob_reference)
    ob = np.load(refp, mmap_mode="r")
    n = ob.shape[0]

    rois = cfg.rois
    wb_step, sample_thresh = rois.wb_step, rois.sample_threshold
    air_thresh, air_percentile = rois.air_threshold, rois.air_percentile
    min_air_px, ratio_smooth = rois.min_air_px, rois.ratio_smooth
    air_margin_px, air_thresh_min = rois.margin_px, rois.air_threshold_min

    wb_o = np.zeros(ob.shape[1:], dtype=np.float64)
    for i in range(0, n, wb_step):
        wb_o += ob[i]

    mask = detector.bad_pixel_map(
        wb_o, hot_sigma=cfg.filters.hot_sigma, dead_fraction=cfg.filters.dead_fraction, border_px=cfg.filters.border_px
    )
    good = detector.usable(mask)

    ratios = []
    for run_dir, counter in sample_runs:
        paths = fitsframe.iter_frame_paths(run_dir, counter=counter or None)
        if not paths:
            continue
        wb_s = np.zeros(ob.shape[1:], dtype=np.float64)
        buf = np.empty(ob.shape[1:], dtype=np.float32)
        for i in range(0, len(paths), wb_step):
            stream.read_into(paths[i], buf)
            wb_s += buf

        r = wb_s / np.maximum(wb_o, 1e-12)

        v = r[good]
        cut = float(np.percentile(v, air_percentile))
        upper = v[v > cut]
        scale = float(np.median(upper)) if upper.size else cut
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"cannot normalise the ratio for {run_dir}")
        rn = (r / scale).astype(np.float32)
        if ratio_smooth > 1:
            rn = cv2.blur(rn, (ratio_smooth, ratio_smooth), borderType=cv2.BORDER_REFLECT_101)
        ratios.append(rn)

    if not ratios:
        raise ValueError("no usable projections for ROI construction")

    ratio = ratios[0]
    ratio_min = np.min(np.stack(ratios, axis=0), axis=0)

    sample = good & (ratio < sample_thresh)

    thresh = min(air_thresh, air_thresh_min)
    air = good & (ratio_min > thresh)
    n_air_raw = int(air.sum())
    if air_margin_px > 0:
        occluded = (ratio_min <= thresh).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * air_margin_px + 1, 2 * air_margin_px + 1))
        air = air & (cv2.dilate(occluded, k) == 0)

    air_ok = int(air.sum()) >= min_air_px
    if not air_ok:
        logger.warning(f"Only {int(air.sum())} always-clear air pixels (< {min_air_px})")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path, sample=sample, air=air, mask=mask, ratio=ratio, ratio_min=ratio_min, wb_ob=wb_o.astype(np.float32)
    )
    pcts = np.percentile(ratio_min[good], [1, 5, 25, 50, 75, 95, 99])

    gain = detector.measure_chip_gain(wb_o)
    return {
        "chip_gain": gain.ratios(float(wb_o.mean())),
        "chip_gain_spread": gain.spread,
        "n_sample": int(sample.sum()),
        "n_air": int(air.sum()),
        "n_air_before_margin": n_air_raw,
        "air_ok": air_ok,
        "air_threshold": thresh,
        "n_angles_used": len(ratios),
        "air_margin_px": air_margin_px,
        "mask": detector.summarise_mask(mask),
        "ratio_percentiles": {str(q): float(v) for q, v in zip([1, 5, 25, 50, 75, 95, 99], pcts, strict=False)},
    }


def run_experiment(
    tasks: list[ProjTask], n_workers: int, progress: Callable[[int, int, ProjReport], None] | None = None
) -> list[ProjReport]:
    """Run a list of projection tasks, sequentially if 'n_workers <= 1'."""
    if n_workers <= 1:
        worker_init()
        out = []
        for i, t in enumerate(tasks, 1):
            r = process_projection(t)
            out.append(r)
            if progress:
                progress(i, len(tasks), r)
        return out

    needs_spawn = any(t.device != "cpu" for t in tasks)
    ctx = mp.get_context("spawn" if needs_spawn else "fork")

    out = []
    with ProcessPoolExecutor(max_workers=n_workers, initializer=worker_init, mp_context=ctx) as ex:
        futs = {ex.submit(process_projection, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            out.append(r)
            if progress:
                progress(i, len(tasks), r)
    out.sort(key=lambda r: (r.angle_deg if r.angle_deg is not None else 1e9, r.run_number))
    return out


def write_report(reports: list[ProjReport], path: str | Path) -> None:
    """Write 'reports' to 'path' as JSON, creating the parent directory."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([asdict(r) for r in reports], indent=2, default=str))
