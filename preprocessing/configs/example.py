"""Example configuration for the pre-processing application."""

from pathlib import Path

from preprocess.core.config import Chain, Paths, PreprocessConfig

DATA = Path("/data/projects/neutron")
DERIVED = DATA / "derived/preprocess/v1"

config = PreprocessConfig(
    paths=Paths(data_root=DATA),
    chain=Chain(
        experiment="20260306_FeCube_CT_2_500C",
        open_beam_experiment="20260306_OB_CT",
        output_directory=DERIVED / "20260306_FeCube_CT_2_500C_1_000AngsMin",
        open_beam_reference=DERIVED / "openbeam/ob_block0.npy",
        device="cpu",
        limit=4,
    ),
    n_workers=8,
)
