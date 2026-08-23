# -*- coding: utf-8 -*-
"""Direct training launcher for CSpec-DB-Net."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from cspec_db_net.main import main
else:
    from .main import main


if __name__ == "__main__":
    main()
