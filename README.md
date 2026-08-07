# ML Engineer Take-Home: Credit Card Fraud Anomaly Detection

This project is a standalone implementation for the ML Engineer take-home exercise on unsupervised anomaly detection in credit-card transaction data.

The submission is designed to emphasize production-oriented ML engineering:

- train without fraud labels;
- use labels only for evaluation;
- choose thresholds from analyst review capacity;
- expose a tested inference interface;
- document monitoring, retraining, rollback, and multi-client operation.

## Project layout

```text
.
├── data/
│   └── creditcard.csv              # downloaded locally; not intended for git
├── notebooks/
│   └── fraud_anomaly_detection.ipynb
├── scripts/
│   └── download_data.py
├── src/
│   └── fraud_scorer.py
├── tests/
│   └── test_fraud_scorer.py
├── .gitignore
└── requirements.txt
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts\download_data.py
python -m pytest
```

## Dataset

The assessment references the Kaggle Credit Card Fraud Detection dataset originally published by Université Libre de Bruxelles.

This scaffold downloads the Kaggle dataset through `kagglehub`:

```powershell
python scripts\download_data.py
```

If that download fails, manually download the Kaggle dataset and place it at:

```text
data/creditcard.csv
```

Expected columns:

- `Time`
- `Amount`
- `V1` through `V28`
- `Class` for evaluation only

## Recommended workflow

1. Run the notebook top-to-bottom.
2. Keep the model section simple and strong: Isolation Forest as the primary unsupervised model.
3. Evaluate at fixed review rates, especially 5%, to match the current analyst workload.
4. Use `src/fraud_scorer.py` for the productionization section instead of keeping all scoring logic in notebook cells.
5. Run tests before submission.
