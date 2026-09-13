"""Combine the open-beam runs of one time block into the reference cube."""

import itertools

from loguru import logger

from preprocess.core.backend import get_backend
from preprocess.core.config import PreprocessConfig
from preprocess.core.constants import OPEN_BEAM_REFERENCE_NAME
from preprocess.core.preprocessing_types import Kind
from preprocess.correct.open_beam import sum_open_beam
from preprocess.io import manifest
from preprocess.io.acquisition import Acquisition, IncompleteAcquisitionError
from preprocess.io.manifest import RunRecord
from preprocess.utils import plots


def _discover(config: PreprocessConfig) -> list[RunRecord]:
    """Find the open-beam acquisitions in run order, refusing an empty result."""
    if not config.paths.open_beam.is_dir():
        logger.error(
            f"No open-beam tree at {config.paths.open_beam}. Mount /data, or point "
            "PreprocessConfig at the tree that holds images/tpx1/ob."
        )
        raise SystemExit(1)

    records = list(manifest.discover(config.paths.open_beam, Kind.OPEN_BEAM))
    if config.chain.open_beam_experiment:
        records = [r for r in records if config.chain.open_beam_experiment in r.experiment]
    records.sort(key=lambda r: (r.key.run_number, r.counter))
    if not records:
        logger.error(f"No open-beam runs under {config.paths.open_beam}")
        raise SystemExit(1)
    return records


def _time_blocks(records: list[RunRecord], block_gap: int) -> list[list[RunRecord]]:
    """Group consecutive records into time blocks, split where the run number jumps by more than 'block_gap'."""
    blocks = [[records[0]]]
    for previous, record in itertools.pairwise(records):
        if record.run_number - previous.run_number > block_gap:
            blocks.append([])
        blocks[-1].append(record)
    return blocks


def _select_block(blocks: list[list[RunRecord]], index: int) -> list[RunRecord]:
    """Return one time block's records, refusing an index the experiment does not have."""
    if not 0 <= index < len(blocks):
        logger.error(
            f"'chain.open_beam_block' {index} is out of range: this experiment has "
            f"{len(blocks)} time blocks, so pass 0..{len(blocks) - 1}."
        )
        raise SystemExit(1)
    return blocks[index]


def _load_complete(records: list[RunRecord]) -> list[Acquisition]:
    """Load every acquisition whose files are all there; a partial transfer is skipped with a warning, not fatal."""
    acquisitions: list[Acquisition] = []
    for record in records:
        try:
            acquisitions.append(Acquisition.load(record))
        except IncompleteAcquisitionError as exc:
            logger.warning(f"skip {record.key.dirname[:52]} -- {exc}")
    return acquisitions


def run(config: PreprocessConfig) -> bool:
    """Sum the open-beam runs into the reference cube."""
    settings = config.open_beam
    output_directory = config.chain.require_output_directory()

    records = _discover(config)
    blocks = _time_blocks(records, settings.block_gap)
    logger.info(
        f"{len(records)} open-beam acquisitions in {len(blocks)} time blocks "
        f"(split on a run-number gap > {settings.block_gap})"
    )
    if config.chain.open_beam_block is not None:
        records = _select_block(blocks, config.chain.open_beam_block)

    acquisitions = _load_complete(records)
    if not acquisitions:
        logger.error(
            "No complete open-beam acquisition. Widen 'chain.open_beam_experiment', or clear "
            "'chain.open_beam_block', so more runs are included."
        )
        raise SystemExit(1)
    logger.info("summing runs " + ", ".join(str(a.record.run_number) for a in acquisitions))

    xp, dev = get_backend(config.chain.device)
    white_beams = sum_open_beam(acquisitions, config, output_directory / OPEN_BEAM_REFERENCE_NAME, xp, dev)
    if config.chain.figures:
        plots.open_beam_figure(white_beams, output_directory / "open_beam_white_beams.png")
    return True
