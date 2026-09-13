"""Discover and validate runs, and write a manifest."""

from loguru import logger

from preprocess.core.config import PreprocessConfig
from preprocess.core.constants import MANIFEST_NAME
from preprocess.core.preprocessing_types import Status
from preprocess.io import manifest
from preprocess.utils import plots


def discover(config: PreprocessConfig) -> list[manifest.RunRecord]:
    """Find every run of the configured kind, refusing an empty result."""
    kind = config.chain.kind
    root = config.paths.root_for(kind)
    if not root.is_dir():
        logger.error(
            f"No raw data root at {root}. Mount /data, or point PreprocessConfig at the tree that holds images/tpx1."
        )
        raise SystemExit(1)

    records = manifest.discover(root, kind)
    logger.info(f"{kind}: {len(records)} run directories under {root}")

    if config.chain.experiment:
        records = [r for r in records if config.chain.experiment in r.experiment]
        logger.info(f"Filtered to {len(records)} runs matching {config.chain.experiment!r}")
    if config.scan.limit is not None:
        records = records[: config.scan.limit]

    # Reporting "0/0 acquisitions OK" and exiting 0 says the data is sound when
    # nothing was looked at, which is indistinguishable from a real pass in a log.
    if not records:
        logger.error(
            "No run directories to validate"
            + (f" matching {config.chain.experiment!r}" if config.chain.experiment else "")
            + f" of kind {kind!r} under {root}. Nothing was "
            "validated, so there is nothing to report -- check the experiment "
            "substring against a configuration with no filter."
        )
        raise SystemExit(1)

    return records


def run(config: PreprocessConfig) -> bool:
    """Validate every acquisition found and write the manifest, reporting success."""
    records = discover(config)
    check_sizes = not config.chain.fast
    logger.info(f"Validating {len(records)} runs (check_sizes={check_sizes})")

    validated: list[manifest.RunRecord] = []
    for position, record in enumerate(records, start=1):
        validated.append(
            manifest.validate_run(
                record, nexus_dir=config.paths.nexus, check_sizes=check_sizes, n_header_samples=config.scan.n_headers
            )
        )
        latest = validated[-1]
        angle = f"{latest.angle_deg:7.3f}" if latest.angle_deg is not None else "     --"
        logger.debug(
            f"[{position}/{len(records)}] {latest.status} run={latest.run_number} "
            f"acq={latest.counter or '-'} ang={angle} frames={latest.n_frames} {latest.key.dirname[:36]}"
        )
        for problem in latest.problems:
            logger.warning(f"run {latest.run_number}: {problem}")

    n_ok = sum(record.status == Status.OK for record in validated)
    logger.info(f"{n_ok}/{len(validated)} acquisitions OK, {len(validated) - n_ok} quarantined")

    destination = config.chain.require_output_directory() / MANIFEST_NAME
    try:
        manifest.write_manifest(validated, destination)
    except OSError as exc:
        logger.error(
            f"Cannot write the manifest to {destination}: {exc}. The scan itself "
            "succeeded; give the output directory a writable path and re-run."
        )
        raise SystemExit(1) from exc

    logger.info(f"Manifest written to {destination}")
    if config.chain.figures:
        plots.scan_figure(validated, destination.with_name("scan_summary.png"))
    return n_ok == len(validated)
