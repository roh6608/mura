"""Measured constants, instrument values and canonical data locations.

Each number's provenance is in the comment above it; the module is a leaf that imports
nothing else in the package, so a calibration-critical value has exactly one definition.
"""

import re
from typing import Final

# --- Instrument -----------------------------------------------------------------

H_OVER_M: Final[float] = 3956.034
TI_DELAY_LOG: Final[str] = "BL10:Det:TH:DSPT1:TIDelay"
LAMBDA_MIN_LOG: Final[str] = "BL10:Exp:Chop:LambdaMinActual"
ROTATION_STAGE: Final[str] = "rot1"

DASLOG_SCALARS: Final[dict[str, str]] = {
    "ti_delay_us": TI_DELAY_LOG,
    "proton_charge_rtdl": "BL10:Det:rtdl:ProtonCharge",
    "pcharge_integrated": "BL10:Det:T1:PChargeIntegrated_RBV",
    "lambda_min_actual": LAMBDA_MIN_LOG,
    "chopper_wavelength_req": "BL10:Chop:Gbl:WavelengthReq",
    "chop_skf1_delay": "BL10:Chop:Skf1:PhaseTimeDelaySet",
    "chop_skf4_delay": "BL10:Chop:Skf4:PhaseTimeDelaySet",
    "acq_freq": "BL10:Det:dsp1:AcqFreq",
    "lattice_a": "BL10:CS:ITEMS:LatticeA",
    "lattice_b": "BL10:CS:ITEMS:LatticeB",
    "lattice_c": "BL10:CS:ITEMS:LatticeC",
}
MONITOR_BANKS: Final[tuple[str, ...]] = ("bank100", "bank200", "bank300")

# --- Detector geometry ----------------------------------------------------------

CHIP: Final[int] = 256
DETECTOR_ROWS: Final[int] = 2 * CHIP
DETECTOR_COLUMNS: Final[int] = 2 * CHIP
DETECTOR_SHAPE: Final[tuple[int, int]] = (DETECTOR_ROWS, DETECTOR_COLUMNS)

# --- Library limits -------------------------------------------------------------

CV2_FLOAT_MEDIAN_KSIZES: Final[tuple[int, ...]] = (3, 5)

# --- ToF axis tolerances --------------------------------------------------------

TOF_CARD_ATOL_S: Final[float] = 1e-9
SPECTRA_ATOL_S: Final[float] = 5e-8
WINDOW_BIN_COUNT_TOLERANCE: Final[float] = 0.05

# --- Data shapes ----------------------------------------------------------------

SIDECAR_TABLE_DIMENSIONS: Final[int] = 2
SHUTTER_TIMES_COLUMNS: Final[int] = 3
SHUTTER_COUNT_COLUMNS: Final[int] = 2
SPECTRA_COLUMNS: Final[int] = 2
FRAME_DIMENSIONS: Final[int] = 2
CUBE_DIMENSIONS: Final[int] = 3

# --- FITS frame layout ----------------------------------------------------------

FITS_BLOCK_BYTES: Final[int] = 2880
FITS_HEADER_CARD_BYTES: Final[int] = 80
FITS_HEADER_BYTES: Final[int] = FITS_BLOCK_BYTES
FITS_PIXEL_DTYPE: Final[str] = ">i2"
FITS_DATA_BYTES: Final[int] = DETECTOR_ROWS * DETECTOR_COLUMNS * 2
FITS_FRAME_BYTES: Final[int] = FITS_HEADER_BYTES + -(-FITS_DATA_BYTES // FITS_BLOCK_BYTES) * FITS_BLOCK_BYTES
FRAME_FLOAT32_BYTES: Final[int] = DETECTOR_ROWS * DETECTOR_COLUMNS * 4

# --- Transmission cubes ---------------------------------------------------------

CUBE_SCHEMA_VERSION: Final[int] = 1
CUBE_CHUNK_YX: Final[int] = 64

# --- File and directory names -----------------------------------------------------


FRAME_RE: Final[re.Pattern[str]] = re.compile(r"_(\d{5})\.fits$")

ACQ_FRAME_RE: Final[re.Pattern[str]] = re.compile(r"_(\d{3})_(\d{5})\.fits$")
ACQ_AUX_RE: Final[re.Pattern[str]] = re.compile(
    r"_(\d{3})_(?:ShutterTimes|ShutterCount|Spectra|Status|SummedImg)\.(?:txt|fits)$"
)
RUN_DIRNAME_RE: Final[re.Pattern[str]] = re.compile(r"^(?P<date>\d{8})_Run_(?P<run>\d+)_(?P<rest>.+)$")
NEXUS_FILE_RE: Final[re.Pattern[str]] = re.compile(r"VENUS_(\d+)\.nxs\.h5$")
ANGLE_RE: Final[re.Pattern[str]] = re.compile(r"^(?P<body>.+)_Ang_(?P<deg>\d+)_(?P<mdeg>\d{3})_(?P<seq>\d+)$")
OPEN_BEAM_RE: Final[re.Pattern[str]] = re.compile(r"^(?P<body>.+)_ob_(?P<pos>\d+)$")
CHARGE_RE: Final[re.Pattern[str]] = re.compile(r"_(\d+)_(\d{3})C(?:_|$)")
LAMBDA_MIN_RE: Final[re.Pattern[str]] = re.compile(r"_(\d+)_(\d{3})AngsMin(?:_|$)")

# --- Acceptance thresholds ------------------------------------------------------

NEAR_ZERO: Final[float] = 1e-12
MIN_BAND_ROWS: Final[int] = 8
MIN_BANDS_FOR_TILT: Final[int] = 3
MIN_PROJECTIONS_FOR_CENTRE: Final[int] = 4
MAX_CONTROL_SHIFT_PX: Final[float] = 6.0
MAX_TRUNCATION_FRACTION: Final[float] = 0.02
NAME_ROUNDING_TOLERANCE: Final[float] = 1e-3

# --- Summary figures -------------------------------------------------------------

FIGURE_SIZE_IN: Final[tuple[float, float]] = (10.0, 4.0)
FIGURE_DPI: Final[int] = 600

# --- Stage outputs, beside each other in 'chain.output_directory' ---------------

MANIFEST_NAME: Final[str] = "manifest.json"
OPEN_BEAM_REFERENCE_NAME: Final[str] = "open_beam_sum.npy"
INDEX_NAME: Final[str] = "index.h5"
GEOMETRY_NAME: Final[str] = "geometry.json"
