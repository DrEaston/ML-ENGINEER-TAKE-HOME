# Credit Card Fraud Anomaly Detection

The take-home submission has one main notebook:

`fraud_anomaly_detection.ipynb`

Its reusable model training, scoring, and evaluation interfaces live in the
small supporting file `fraud_scoring.py`, which the notebook imports directly.

## Run the notebook

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Open `fraud_anomaly_detection.ipynb` in Jupyter or VS Code and run all cells.
The first executable cell installs the pinned dependencies, and dataset
acquisition is handled by KaggleHub.

The notebook contains data loading, model comparison, operational benchmarks,
plots, and design recommendations. The supporting Python file contains model
training plus production-style scoring, validation, serialization, and
operating-point evaluation.
