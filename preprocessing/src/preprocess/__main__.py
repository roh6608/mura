"""Allow the package to be run as 'python -m preprocess'."""

import sys

from preprocess.main import main

sys.exit(main() or 0)
