"""Command-line argument parsing for the Bragg-edge fitting application."""

import argparse
from collections.abc import Sequence
from pathlib import Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse 'argv' into the application's command-line arguments."""
    parser = argparse.ArgumentParser(prog="bragg", description="Bragg-edge fitting.")
    parser.add_argument("config", type=Path, help="Python file defining 'config = BraggConfig(...)'")
    return parser.parse_args(argv)
