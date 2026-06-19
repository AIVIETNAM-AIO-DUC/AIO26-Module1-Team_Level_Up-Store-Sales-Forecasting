# Store Sales — Time Series Forecasting

A learning project that forecasts daily unit sales for Corporación Favorita (Ecuador) over
a 16-day horizon, following the Kaggle *Time Series* course techniques (trend, seasonality,
lags, hybrid models). Metric: **RMSLE**.

## Setup

This project uses [**uv**](https://docs.astral.sh/uv/) to manage Python and dependencies.

```bash
# 1. Install uv (https://docs.astral.sh/uv/getting-started/installation/)
# 2. From the repo root, create the locked environment:
uv sync                 # builds .venv/ from pyproject.toml + uv.lock (Python 3.11+)

# 3. Run anything through uv:
uv run jupyter lab      # launch notebooks
uv run python -c "from src import data; print(data.load_train().shape)"
```

## ⚠️ You must add the competition data yourself

The dataset is **not** included in this repository (`store-sales-data/` is git-ignored to
keep the repo small and respect competition data terms). Each teammate downloads it
separately:

1. Go to the Kaggle competition:
   **Store Sales – Time Series Forecasting**
   <https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data>
   (You need a Kaggle account and must accept the competition rules.)
2. Download the data and unzip it into a folder named **`store-sales-data/`** at the repo
   root, so the layout is exactly:

   ```text
   store-sales-data/
   ├── train.csv
   ├── test.csv
   ├── stores.csv
   ├── transactions.csv
   ├── oil.csv
   ├── holidays_events.csv
   └── sample_submission.csv
   ```

   (Alternatively, with the Kaggle CLI: `kaggle competitions download -c
   store-sales-time-series-forecasting -p store-sales-data && unzip` it in place.)
3. Verify the data loads:

   ```bash
   uv run python -c "import sys; sys.path.insert(0,'.'); from src import data; print('train', data.load_train().shape)"
   # expect: train (3000888, 6)
   ```

The loaders in `src/data.py` resolve paths relative to the repo root, so once the folder is
in place everything works without configuration.

## Project layout

```text
src/                 # reusable logic (loaders, features, validation, models)
notebooks/           # 01_eda onward — teaching/exploration, run via uv
analyze/             # learner docs: one small topic per file (start at analyze/README.md)
docs/                # published website (GitHub Pages): English + vi/ mirror
submissions/         # generated submission CSVs (git-ignored)
iteration_log.md     # running log of each technique's holdout RMSLE
store-sales-data/    # raw competition CSVs — ADD YOURSELF (git-ignored)
```

> **New to the project?** Start with `analyze/README.md` — the index to the learner docs. It
> explains the data's quirks (closed-day gaps, oil gaps, holiday handling) and the core concepts,
> one small topic per file.

## Notes

- The shared learner docs live under `analyze/` (Markdown, one topic per file); `docs/` is the
  published website (GitHub Pages) and this `README.md` is the entry point for setup.
- Validation uses a single time-respecting 16-day holdout; all models are scored by RMSLE in
  log space with predictions clipped to non-negative.
  
