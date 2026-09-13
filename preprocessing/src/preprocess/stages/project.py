"""Process a whole experiment into transmission cubes."""

import time
from pathlib import Path

from loguru import logger

from preprocess.core.backend import free_bytes, get_backend
from preprocess.core.config import PreprocessConfig
from preprocess.core.constants import FRAME_FLOAT32_BYTES, MANIFEST_NAME, OPEN_BEAM_REFERENCE_NAME
from preprocess.core.preprocessing_types import ProjReport, ProjTask, Status
from preprocess.io import manifest, store
from preprocess.io.manifest import RunRecord
from preprocess.utils import plots
from preprocess.workers import projection


def _require_open_beam_reference(reference: Path) -> Path:
    """Return the open-beam reference."""
    missing = [
        p
        for p in (reference, reference.with_suffix(".json"), reference.with_name(reference.stem + "_var.npy"))
        if not p.is_file()
    ]
    if missing:
        logger.error("Open-beam reference incomplete.")
        raise SystemExit(1)
    return reference


def _prepare_output(
    out_dir: Path, config: PreprocessConfig, ok: list[RunRecord], quarantined: list[RunRecord], reference: Path
) -> None:
    """Create the output directory and write the experiment's configuration beside it."""
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error(f"Cannot create the output directory {out_dir}: {exc}.")
        raise SystemExit(1) from exc
    store.write_experiment_meta(
        out_dir,
        config.to_json(),
        extra={
            "experiment": ok[0].experiment,
            "kind": config.chain.kind,
            "open_beam_reference": str(reference),
            "n_acquisitions": len(ok),
            "quarantined": [{"run": r.run_number, "acq": r.counter, "problems": list(r.problems)} for r in quarantined],
        },
    )


def _build_rois(out_dir: Path, config: PreprocessConfig, all_ok: list[RunRecord], reference: Path) -> Path:
    """Build the sample and air regions of interest once, from projections spread over angle, unless they exist."""
    rois_path = out_dir / "rois.npz"
    if not config.rois.force_rebuild and rois_path.exists():
        logger.info(f"ROIs: reusing {rois_path}")
        return rois_path

    with_ang = [r for r in all_ok if r.angle_deg is not None]
    pick = with_ang or all_ok
    pick = sorted(pick, key=lambda r: r.angle_deg or 0.0)
    step = max(1, len(pick) // config.rois.angles)
    chosen = pick[::step][: config.rois.angles]
    logger.info(
        f"building ROIs from {len(chosen)} projections at angles " + ", ".join(f"{r.angle_deg:.1f}" for r in chosen)
    )
    info = projection.build_rois(str(reference), str(rois_path), config, [(r.path, r.counter) for r in chosen])
    logger.info(
        f"ROIs: sample {info['n_sample']} px, air {info['n_air']} px, usable {info['mask']['usable']}; "
        "chip gain relative to the frame mean: "
        + " ".join(f"{chip} {ratio:.3f}" for chip, ratio in info["chip_gain"].items())
    )
    return rois_path


def _tasks(
    ok: list[RunRecord], config: PreprocessConfig, out_dir: Path, reference: Path, rois_path: Path
) -> list[ProjTask]:
    """One task per acquisition whose cube is not already present."""
    tasks: list[ProjTask] = []
    for r in ok:
        if r.calibration is None:
            raise RuntimeError(f"run {r.run_number} was validated ok without a wavelength calibration")
        name = f"{r.run_number}_{r.counter}.h5" if r.counter else f"{r.run_number}.h5"
        out_path = out_dir / "proj" / name
        if out_path.exists() and not config.chain.overwrite:
            continue
        tasks.append(
            ProjTask(
                run_dir=r.path,
                counter=r.counter,
                experiment=r.experiment,
                kind=r.kind,
                run_number=r.run_number,
                angle_deg=r.angle_deg,
                calibration=r.calibration,
                out_path=str(out_path),
                ob_reference=str(reference),
                rois_path=str(rois_path),
                device=config.chain.device,
                band=config.chain.band,
                store_variance=bool(config.projection.store_variance),
                despeckle_k=config.filters.despeckle_k,
                despeckle_ksize=config.filters.despeckle_ksize,
                repair_ksize=config.projection.repair_ksize,
                repair_passes=config.projection.repair_passes,
                compression=config.chain.compression,
                config=config,
            )
        )
    skipped = len(ok) - len(tasks)
    if skipped:
        logger.info(f"{skipped} projections already present (set 'chain.overwrite' to redo)")
    return tasks


def _workers_that_fit(config: PreprocessConfig) -> int:
    """Return the worker count."""
    n_workers = config.n_workers
    if config.chain.device not in ("gpu", "auto"):
        return n_workers
    xp, dev = get_backend(config.chain.device)
    if dev != "gpu":
        return n_workers

    per_worker = 7 * config.chain.band * FRAME_FLOAT32_BYTES + 400e6
    free = free_bytes(xp)
    if free is None:
        raise RuntimeError("the GPU backend reported no free-memory figure")
    fits = int(free // per_worker)
    logger.info(
        f"VRAM: {free / 1e9:.1f} GB free, ~{per_worker / 1e9:.2f} GB per worker "
        f"at band={config.chain.band} -> room for ~{fits}"
    )
    if n_workers > fits:
        logger.warning(f"reducing workers {n_workers} -> {max(fits, 1)} to stay inside VRAM ")
        n_workers = max(fits, 1)
    return n_workers


def _log_progress(i: int, n: int, r: ProjReport) -> None:
    """Log one projection's outcome as the pool reports it."""
    flag = "ok  " if r.ok else "FAIL"
    ang = f"{r.angle_deg:7.3f}" if r.angle_deg is not None else "     --"
    logger.info(
        f"  [{i:3d}/{n}] {flag} run={r.run_number} acq={r.counter} ang={ang} "
        f"{r.seconds:5.1f}s  corr=x{r.mean_correction:.3f} "
        f"meanT={r.mean_transmission:.4f} airT={r.air_transmission:.4f} "
        f"spk={r.n_despeckled}({100 * r.frac_counts_despeckled:.4f}%) "
        f"rep={r.n_repaired_px}"
    )
    if r.message:
        logger.warning(f"run={r.run_number} acq={r.counter}: {r.message}")


def _write_index(out_dir: Path) -> None:
    """Index every complete cube under 'out_dir/proj'."""
    try:
        logger.info(f"  index       : {store.build_index(out_dir)}")
    except (FileNotFoundError, ValueError) as exc:
        logger.warning(f"index skipped: {exc}")


def run(config: PreprocessConfig) -> bool:
    """Process a whole experiment into per-projection transmission cubes."""
    out_dir = Path(config.chain.require_output_directory())
    reference = _require_open_beam_reference(
        Path(config.chain.open_beam_reference or out_dir / OPEN_BEAM_REFERENCE_NAME)
    )

    validated = manifest.read_manifest(out_dir / MANIFEST_NAME)
    all_ok = sorted(
        (r for r in validated if r.status == Status.OK),
        key=lambda r: (r.angle_deg if r.angle_deg is not None else 1e9, r.run_number),
    )
    quarantined = [r for r in validated if r.status != Status.OK]
    ok = all_ok[: config.chain.limit] if config.chain.limit else all_ok
    if config.chain.limit:
        logger.info(f"  limited to {len(ok)} acquisitions")

    _prepare_output(out_dir, config, ok, quarantined, reference)
    rois_path = _build_rois(out_dir, config, all_ok, reference)
    tasks = _tasks(ok, config, out_dir, reference, rois_path)
    if not tasks:
        logger.info("nothing to do")
        _write_index(out_dir)
        return True

    n_workers = _workers_that_fit(config)
    est_gb = len(tasks) * ok[0].n_frames * FRAME_FLOAT32_BYTES * (2 if config.projection.store_variance else 1) / 1e9
    logger.info(f"processing {len(tasks)} projections with {n_workers} workers on {config.chain.device}")
    logger.info(f"  output {out_dir}   ~{est_gb:.0f} GB uncompressed before {config.chain.compression}")

    started = time.time()
    reports = projection.run_experiment(tasks, n_workers, progress=_log_progress)
    elapsed = time.time() - started

    if not reports:
        logger.error(f"{len(tasks)} projection(s) were queued but no report came back, so nothing was written. ")
        raise SystemExit(1)
    n_ok = sum(r.ok for r in reports)
    logger.info(f"{n_ok}/{len(reports)} projections written in {elapsed:.1f} s ({elapsed / len(reports):.1f} s each)")

    projection.write_report(reports, out_dir / "run_report.json")
    if config.chain.figures:
        plots.projection_figure(reports, out_dir / "projection_summary.png")
    if n_ok:
        _write_index(out_dir)
    return n_ok == len(reports)
