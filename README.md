# CSpec-DB-Net modular version

This folder is a self-contained extraction of the original DESI-recall workspace.
It does not import code from the original `desi_recall` or `fif_test` folders.

- `config.py`: shared constants, feature names, and CLI parsing helpers
- `preprocessing.py`: FITS loading, resampling, normalization, augmentation, and `LAMOSTDataset`
- `models/`: the complete local model zoo, including `cspec_db_net.py`
- `models/registry.py`: unified model factory and adapters for all model files
- `training.py`: losses, metrics, EMA, scheduler, validation-only checkpoint/threshold selection, and final test evaluation
- `main.py`: main function that runs data loading, preprocessing, training, and evaluation
- `desi_io.py`: memory-conscious DESI pickle reader and writer
- `recall.py`: standalone DESI recall pipeline using a default carbon-star threshold of 0.97
- `train.py`: direct launcher that calls `main.py`

Run directly from the repository root:

```powershell
python .\cspec_db_net\main.py
python .\cspec_db_net\main.py --model_name wdcnn1d
python .\cspec_db_net\recall.py --checkpoint .\cspec_db_net\runs\wdcnn_transformer_idx\best_model.pt --input D:\path\to\desi_pickles
```

Install Python dependencies with:

```powershell
python -m pip install -r .\cspec_db_net\requirements.txt
```

The default local data layout is:

```text
cspec_db_net/
  data/
    lamost/
      trainval/<class folders>/
      test/<class folders>/
    desi/*.pkl
  runs/wdcnn_transformer_idx/best_model.pt
  outputs/
```

Data files and model checkpoints are runtime inputs and are intentionally not
vendored into the source package. All Python code required for training,
evaluation, and DESI recall is contained in this directory. Checkpoint and
decision-threshold selection use only the validation set; the test set is
evaluated only after the best checkpoint has been fixed.

Or import from another script:

```python
from cspec_db_net.main import main
from cspec_db_net.models import create_model
from cspec_db_net.recall import main as recall_main

model = create_model("wdcnn1d")
outputs = model(flux, x_idx=flux_idx)  # returns {"bin_logits": ...}

main(["--model_name", "cspec_db_net", "--epochs", "1", "--batch_size", "8"])
recall_main(["--checkpoint", "best_model.pt", "--input", "desi_pickles", "--dry_run"])
```
