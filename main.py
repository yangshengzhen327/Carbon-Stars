# -*- coding: utf-8 -*-
"""Main entry point for the full CSpec-DB-Net training workflow."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from cspec_db_net.training import main as _train_main
else:
    from .training import main as _train_main


def main(argv=None):
    """Run data loading, preprocessing, model construction, training, and evaluation."""

    return _train_main(argv)


if __name__ == "__main__":
    main()
