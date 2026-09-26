# Applied Machine Learning & Predictive Modeling

A collection of learning implementations and notebook experiments, including Naive Bayes, matrix-based linear regression, and Solana price/volume classification experiments using XGBoost and LightGBM.

## Scope

- Two Naive Bayes learning notebooks.
- A linear-regression notebook using lagged financial features.
- A 15-minute Solana experiment with chronological train/calibration/test slices.
- An XGBoost/LightGBM experiment with engineered features and exploratory backtest code.

## Setup

```bash
git clone https://github.com/arslantariq364/machine-learning-algorithms.git
cd machine-learning-algorithms
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas scikit-learn xgboost lightgbm ccxt ta plotly matplotlib scipy jupyter
jupyter notebook
```

Inspect each notebook's data paths and configuration first. Some experiments fetch exchange data; results depend on available data and package versions. No training or external requests run merely by cloning this repository.

## Evaluation limitations

Chronological slicing is present, but it does not by itself prevent leakage. `ANTI_GRAVITY REVISED SOL .ipynb` constructs swing features using future observations via `shift(-1)`; these must be removed or time-aligned before out-of-sample claims. This collection does not establish a complete walk-forward validation procedure or leakage-free live strategy. Saved backtest output is exploratory, not independently reproduced evidence of profitability.

There is no production inference service or verified execution latency, and the reviewed files do not support claims of PCA, genetic feature selection or clustering pipelines. Notebook names containing strong marketing terms are original filenames, not validated quality guarantees.
