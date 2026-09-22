"""Allow ``python -m product_normalization ...`` alongside the console script."""

import sys

from .main import main

if __name__ == "__main__":
    sys.exit(main())
