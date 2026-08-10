import argparse
from collections.abc import Sequence
from pathlib import Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sim-fea",
        description="Eigenstrain computation with finite element analysis.",
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Python configuration file defining `config = SimConfig(...)`",
    )
    return parser.parse_args(argv)
