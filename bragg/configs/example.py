"""Example configuration for the Bragg-edge fitting application."""

from pathlib import Path

from bragg.core.config import AxisSpec, BraggConfig, Chain, Export, Fit

DERIVED = Path("/data/projects/neutron/derived")
PREPROCESSED = DERIVED / "preprocess/v1"
OUT = DERIVED / "bragg/v1"

config = BraggConfig(
    fit=Fit(tof_bins=24.0, device="cpu"),
    chain=Chain(experiment=PREPROCESSED / "20260306_FeCube_CT_2_500C_1_000AngsMin", output_directory=OUT, limit=4),
    export=Export(
        calibration=PREPROCESSED / "calibration_geometry.json",
        axes=(
            AxisSpec(OUT / "fits", PREPROCESSED / "20260306_FeCube_CT_2_500C_1_000AngsMin", "X"),
            AxisSpec(DERIVED / "bragg/v1_y/fits", PREPROCESSED / "20260311_FeCube_CT_Y_Axis_2_500C_1_000AngsMin", "Y"),
            AxisSpec(DERIVED / "bragg/v1_z/fits", PREPROCESSED / "20260314_FeCube_CT_Z_axis_2_500C_1_000AngsMin", "Z"),
        ),
    ),
)
