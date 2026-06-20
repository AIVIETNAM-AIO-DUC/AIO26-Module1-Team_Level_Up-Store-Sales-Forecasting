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
analyze/             # learner docs (Markdown): README + concepts/ + data-traps/ + eda/
docs/                # published website (GitHub Pages): en/ + vi/, organized by stage
submissions/         # generated submission CSVs (git-ignored)
iteration_log.md     # running log of each technique's holdout RMSLE
store-sales-data/    # raw competition CSVs — ADD YOURSELF (git-ignored)
specs/               # Spec Kit planning artifacts (internal — readers can ignore)
```

## How to read this repo

Pick the entry point that matches what you want:

- **Just want to understand the project (recommended for most readers)** →
  open the **website**: <https://nqnguyen86.github.io/learning-time-series/>.
  It walks the story stage by stage (foundations → stage 0 → stage 1 → …) in English
  and Vietnamese, with the same content mirrored under `docs/en/` and `docs/vi/`.
- **Want to focus on the tasks / dig into a specific concept or data quirk** →
  read `analyze/README.md`. It's the index to the learner docs: small Markdown files,
  one topic per file (closed-day gaps, oil gaps, holiday handling, lag/rolling features, etc.).
- **Want to run or extend the code** → start in `src/` (reusable logic) and `notebooks/`
  (exploration, run via `uv run jupyter lab`).

