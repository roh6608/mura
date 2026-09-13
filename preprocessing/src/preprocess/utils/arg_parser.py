"""Command-line argument parsing for the pre-processing application."""

import argparse
from collections.abc import Sequence
from pathlib import Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse 'argv' into the application's command-line arguments."""
    parser = argparse.ArgumentParser(prog="preprocess", description="Pre-processing.")
    parser.add_argument("config", type=Path, help="Python file defining 'config = PreprocessConfig(...)'")
    return parser.parse_args(argv)
