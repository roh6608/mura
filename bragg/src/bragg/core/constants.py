"""Measured constants and fixed model choices for the Bragg-edge chain.

Each value is used by the production chain; the calibration studies that established
the measured ones are recorded in the handover.
"""

from typing import Final

# --- Lattice -----------------------------------------------------------------------------

LAM_FE_110_A: Final[float] = 4.0538

# --- GP ----------------------------------------------------------------------------------

KERNEL_KINDS: Final[tuple[str, str, str]] = ("matern52", "matern32", "matern32")
GRAM_JITTER: Final[float] = 1e-8
SPATIAL_LS_PX: Final[float] = 4.0
DERIVATIVE_WINDOW: Final[int] = 7
DERIVATIVE_POLYORDER: Final[int] = 3
DRAW_SEED: Final[int] = 0
NELDER_MEAD_XATOL: Final[float] = 1e-3
NELDER_MEAD_FATOL: Final[float] = 1e-4
NELDER_MEAD_MAX_ITER: Final[int] = 120
ETA_BOUNDS: Final[tuple[float, float]] = (1e-8, 1e4)
GOLDEN_TOLERANCE: Final[float] = 1e-3
LOWER_BOUND_FACTOR: Final[float] = 1.02
UPPER_BOUND_FACTOR: Final[float] = 0.98
INIT_STEPS: Final[float] = 4.0
INIT_SPAN_FRACTION: Final[float] = 0.05
BOUND_STEPS: Final[float] = 1.5
BOUND_SPANS: Final[float] = 1.5
MAX_AUTOCORRELATION_LAGS: Final[int] = 64
CACHE_KEY_DECIMALS: Final[int] = 6
MAD_TO_SIGMA: Final[float] = 1.4826
MAX_CONDITION_RATIO: Final[float] = 4.0
LIVE_CUBES: Final[int] = 7
MAX_MEMORY_FRACTION: Final[float] = 0.6

# --- ROI crop ----------------------------------------------------------------------------

MIN_BAND_ROWS: Final[int] = 8
DEFAULT_CROP_BINS: Final[int] = 128
BBOX_PROBE_BINS: Final[int] = 12
BBOX_QUANTILE: Final[float] = 0.5
BBOX_PERCENTILES: Final[tuple[float, float]] = (10.0, 90.0)
BBOX_PAD_PX: Final[int] = 4
BBOX_MIN_SIZE_PX: Final[int] = 16
BBOX_CLOSE_KERNEL_PX: Final[int] = 9
BBOX_OPEN_KERNEL_PX: Final[int] = 5
FILL_KSIZE: Final[int] = 5
FILL_PASSES: Final[int] = 4
FRAME_NDIM: Final[int] = 2

# --- Edge ----------------------------------------------------------------------------

MIN_DRAWS_FOR_SIGMA: Final[int] = 4
MIN_DRAW_FRACTION: Final[float] = 0.25
AMPLITUDE_FLOOR_QUANTILE: Final[float] = 0.5
AMPLITUDE_FLOOR_FACTOR: Final[float] = 0.25
VALIDITY_OPEN_RADIUS_PX: Final[int] = 1

# --- Ray export ----------------------------------------------------------------------

DETECTOR_ROWS: Final[int] = 512
DETECTOR_COLUMNS: Final[int] = 512
ATTENUATION_BAND_A: Final[tuple[float, float]] = (3.60, 4.00)
ATTENUATION_BINS: Final[int] = 24
AIR_PERCENTILE: Final[float] = 97.0
MIN_TRANSMISSION: Final[float] = 1e-6
RAY_CHUNKS: Final[tuple[int, int, int]] = (1, 128, 128)
MIN_POINTS_FOR_FIT: Final[int] = 4

# --- Numerical guards --------------------------------------------------------------------

SMALL_DENOMINATOR: Final[float] = 1e-6
TINY_DENOMINATOR: Final[float] = 1e-30

# --- Figures -----------------------------------------------------------------------------

FIGURE_DPI: Final[int] = 600
