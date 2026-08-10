# CSpec-DB-Net modular version

This folder is split from `fif_test/train_wdcnn_transformer_idx.py` without changing the training flow.

- `config.py`: shared constants, feature names, and CLI parsing helpers
- `preprocessing.py`: FITS loading, resampling, normalization, augmentation, and `LAMOSTDataset`
- `models/`: model zoo copied from `D:\deeplearning study\model\model`, plus `cspec_db_net.py`
- `models/registry.py`: unified model factory and adapters for all model files
- `training.py`: losses, metrics, EMA, scheduler, threshold search, and the train/eval loop
- `main.py`: main function that runs data loading, preprocessing, training, and evaluation
- `recall.py`: recall pipeline entry point that delegates to the workspace recall implementation
- `interpretability analysis experiments/`: SHAP and band-occlusion experiment scripts
- `interpretability_analysis_experiments.py`: import-friendly wrappers for those scripts
- `train.py`: direct launcher that calls `main.py`

Run directly from the repository root:

```powershell
python .\cspec_db_net\main.py
python .\cspec_db_net\main.py --model_name wdcnn1d
python .\cspec_db_net\recall.py
python ".\cspec_db_net\interpretability analysis experiments\run_snr_band_occlusion_experiment.py"
python ".\cspec_db_net\interpretability analysis experiments\run_snr_shap_attribution_experiment.py"
```

Or import from another script:

```python
from cspec_db_net.main import main
from cspec_db_net.models import create_model
from cspec_db_net.recall import main as recall_main
from cspec_db_net.interpretability_analysis_experiments import (
    run_band_occlusion_experiment,
    run_shap_experiment,
)

model = create_model("wdcnn1d")
outputs = model(flux, x_idx=flux_idx)  # returns {"bin_logits": ...}

main(["--model_name", "cspec_db_net", "--epochs", "1", "--batch_size", "8"])
run_band_occlusion_experiment(["--render-only"])
run_shap_experiment(["--max-samples", "16", "--skip-figures"])
```
