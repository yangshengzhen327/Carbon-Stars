# -*- coding: utf-8 -*-
"""Short entry point for the SNR Gradient SHAP experiment."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_snr_shap_attribution_experiment import main
else:
    from .run_snr_shap_attribution_experiment import main


if __name__ == "__main__":
    main()
