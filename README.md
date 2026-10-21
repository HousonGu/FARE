# FARE

**Factor-Aware Retrieval with Error-Guided Fusion for Retail Forecasting**

FARE augments a backbone forecast with shape-based and factor-conditioned historical references. Two learned gates combine the references and control how much they correct the backbone. Candidate errors from completed development weeks provide reliability inputs; calibration stays fixed during testing.

This repository contains a reusable CPU implementation of the method, dataset loaders, backbone adapters, and experiment runners.

[Quick start](#quick-start) · [Data preparation](#data-preparation) · [External implementations](#external-implementations) · [Experiment commands](#reproduce-the-experiments) · [Method details](#method)

## Quick start

Use Python 3.10 or newer. From the repository root:

```bash
python -m pip install -e .
python run.py --list
python run.py --dataset favorita --backbone chronos --stage audit --dry-run
```

These commands install the core method, list supported experiments, and preview a command without loading data or models. Full experiments additionally require the dataset and the selected backbone's dependencies.

## Repository

```text
src/fare/             Reusable retrieval, features, error state, and fusion
data_provider/        Raw-data loaders, eligibility rules, and prepared panels
backbones/            Backbone training/inference adapters and source fetcher
exp/                  Retrieval, fusion fitting, evaluation, and experiment runners
configs/              Protocols, hyperparameters, and experiment environments
scripts/              Per-dataset and per-backbone launch scripts
.github/workflows/    Installation and entry-point checks on Linux and Windows
run.py                Unified experiment entry point
```

## Environment

Use Python 3.12 for the original experiment environments. Install a PyTorch build for the machine using the [official installation selector](https://pytorch.org/get-started/locally/), then install the requirements for the selected experiment:

```bash
python -m pip install -e .
python -m pip install -r configs/environments/retail.txt
```

`retail.txt` covers Favorita/M5, and provides the Chronos dependencies used by Rossmann/Rohlik. Rohlik LightGBM/TFT uses `rohlik_supervised.txt`; PatchTST/SparseTSF uses `neural.txt`. These retain the original version pins. Use separate environments for configurations with different pins. Root `requirements.txt` installs the lightweight method/data-check dependencies, not every backbone.

## Data preparation

Download the datasets from their official competition pages using your own access:

- [Favorita / Store Sales – Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data): `train.csv`, `stores.csv`, `holidays_events.csv`.
- [M5 Forecasting – Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data): `sales_train_evaluation.csv`, `calendar.csv`, `sell_prices.csv`.
- [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales/data): `train.csv`, `store.csv`.
- [Rohlik Sales Forecasting Challenge V2](https://www.kaggle.com/competitions/rohlik-sales-forecasting-challenge-v2/data): `sales_train.csv`, `sales_test.csv`, `inventory.csv`, `calendar.csv`.

The loaders use the files listed above, not arbitrary similarly named datasets. The Favorita data is the 1,782 store–family panel, rather than the older item-level grocery competition. Rohlik uses the V2 panel. Place them in `data/raw/favorita`, `data/raw/m5`, `data/raw/rossmann`, and `data/raw/rohlik`, respectively. Use `run.py --raw-root` to choose another parent directory. See the [experiment guide](#reproduce-the-experiments) for prepared panels and output conventions.

### Protocol

Every query has a complete 56-day history and a seven-day forecast horizon. Origins are nonoverlapping Mondays. The retained protocol/configuration files define exact split dates.

- Favorita: promotion category and holidays; retrieve within family. The controlled reference library excludes high-promotion and holiday combinations. Preserve missing calendar days rather than filling them with zero.
- M5: SNAP and calendar events; retrieve within item. Listing eligibility is derived from the first available price week; price values are excluded as forecasting inputs.
- Rossmann: promotion and school holidays; retrieve within store type and assortment. Evaluate all days without forcing forecasts to zero on closed days.
- Rohlik: any positive value among seven discount fields, plus warehouse holidays; retrieve within warehouse and level-2 category. Retain the provided availability-adjusted sales target without a second adjustment.

The benchmark assumes the two future factors are known at the forecast origin. Observed future sales are never an input. Historical test weeks can enter later query histories only after completion.

### Backbones

The original Chronos-2 experiments use model `amazon/chronos-2`, package `chronos-forecasting==2.3.2`, revision `29ec3766d36d6f73f0696f85560a422f50e8498c`, a 56-day context, seven-day median predictions, and `cross_learning=False`. Weekday one-hot values and the two dataset factors form nine known covariate channels.

Weights are not included. The native runners use the model identifiers/revisions in `configs/`; Favorita/M5 also expose `python -m exp.retail --chronos-model` for a local checkpoint. LightGBM and TFT require the experiment dependencies; PatchTST and SparseTSF additionally require the external source checkout described in the experiment guide. The lightweight `fare` package can work with forecasts from any separately fitted backbone.

## External implementations

### Installed model packages

- Chronos-2 is called through [`chronos-forecasting`](https://github.com/amazon-science/chronos-forecasting). The protocol fixes the model revision in `configs/`.
- LightGBM is called through [`lightgbm`](https://github.com/microsoft/LightGBM).
- TFT is called through [`pytorch-forecasting`](https://github.com/sktime/pytorch-forecasting). The local files construct samples, select configurations, and call the library model.

Their model libraries and pretrained weights are not copied into `src/fare/`.

The Chronos runner downloads the pretrained checkpoint through the model library when needed and keeps its weights frozen. LightGBM and TFT are trained on the configured fitting period; FARE gates are fitted on development predictions.

### PatchTST and SparseTSF source

`python -m backbones.fetch_upstream` reads `configs/upstream_sources.json` and retrieves:

- [PatchTST](https://github.com/yuqinie98/PatchTST), commit `204c21efe0b39603ad6e2ca640ef5896646ab1a9`.
- [SparseTSF](https://github.com/lss-1138/SparseTSF), commit `b8c2740eecc84d8095ffce49ba5acafe68e53bb8`.

The fetcher verifies original file hashes, applies the recorded import-namespace adaptation, and verifies the adapted hashes. It also retrieves the upstream license files. Downloaded files live under `vendor/` and are excluded from the source release by default. The manifest records original URLs and exact files; model computations retain the original implementation.

Keep the downloaded upstream license files with those sources.

## Reproduce the experiments

Run commands from the repository root. The installed `fare` package provides the core method; `data_provider/`, `backbones/`, and `exp/` run from this checkout.

### Available combinations

- Favorita and M5: Chronos-2, LightGBM, TFT, PatchTST, SparseTSF.
- Rossmann: Chronos-2, PatchTST, SparseTSF. The research notes record LightGBM/TFT results, but the local snapshot did not contain that training adapter.
- Rohlik: Chronos-2, LightGBM, TFT, PatchTST, SparseTSF.

Use `python run.py --list` to inspect this coverage. Select one dataset and backbone per run; `all` stages execute only that selection.

### Unified entry point

```bash
python run.py --dataset favorita --backbone chronos --stage audit --device cpu
python run.py --dataset favorita --backbone chronos --stage all --device cuda
python run.py --dataset m5 --backbone tft --stage all --device cuda
python run.py --dataset rossmann --backbone chronos --stage all --device cuda
python run.py --dataset rohlik --backbone lightgbm --stage all --device cpu
```

Default raw CSV root is `data/raw/`; default output is `results/DATASET/BACKBONE/`. `--raw-root`, `--out`, `--device`, and `--data-root` override these. Add `--dry-run` to inspect the dispatched command before execution. `scripts/DATASET/BACKBONE.sh` forwards its arguments to the same entry point, for example:

```bash
bash scripts/rossmann/chronos.sh --stage audit --device cpu
```

Rossmann/Rohlik raw-data runners also accept `--raw /path/to/actual-csv-directory` when the downloaded folder has a different name.

Stages are `audit`, `references`, `backbone`, `fusion`, `score`, and `all`. The Rohlik supervised runner retains a combined gate-fitting-and-scoring stage, so both `fusion` and `score` invoke that operation. The other runners separate their native fusion and scoring stages. Test labels never fit the gates.

Favorita/M5 outputs add `favorita_original/` or `m5_full/` below the selected output root. The native modules also expose their detailed CLI, such as `python -m exp.retail --help` or `python -m exp.rohlik --help`.

### PatchTST and SparseTSF

These two backbones use prepared panels and separately downloaded upstream source. First fetch their recorded revisions and licenses:

```bash
python -m pip install -r configs/environments/neural.txt
python -m backbones.fetch_upstream
```

Generate references using the raw-data runner, then export the sales panel and the exact reference files:

```bash
python run.py --dataset favorita --backbone chronos --stage references --out results/favorita/references --device cpu
python -m exp.prepare_panel --dataset favorita --raw-root data/raw --reference-root results/favorita/references/favorita_original --out data/prepared/favorita
python run.py --dataset favorita --backbone patchtst --stage all --data-root data/prepared --device cuda
```

For M5 the corresponding reference output adds `m5_full/`. Rossmann and Rohlik reference files are directly under the output folder's `references/`, for example:

```bash
python run.py --dataset rossmann --backbone chronos --stage references --out results/rossmann/references --device cpu
python -m exp.prepare_panel --dataset rossmann --raw-root data/raw --reference-root results/rossmann/references --out data/prepared/rossmann
python run.py --dataset rossmann --backbone sparsetsf --stage all --device cuda
```

Panel export requires a new output directory and verifies reference coverage and evaluation keys. Neural outputs add a dataset subdirectory below `--out`. Neural `score` summarizes the selected dataset/backbone's completed fusion metrics; it does not rerun predictions. Generate references with the raw-data runner before panel export, rather than requesting a neural `references` stage.

### Configuration and results

`configs/retail.json` defines the M5 165/52/52-origin schedule and Favorita's fitting/validation settings. `configs/favorita_origins.json` supplies Favorita's 62/24/25 fitting/development/test origins. Rossmann/Rohlik use their dataset-specific configurations. Gates and standardization use development rows; calibration is frozen in testing.

Existing result column `soft_history` denotes the full FARE method. Overall WAPE is total absolute error divided by total actual sales across all evaluation rows. Do not average weekly WAPE values. Per-seed summaries are averages of metrics, not prediction ensembles.

Use a new output directory after code, data, configuration, or environment changes. Resume manifests intentionally reject changed inputs.

### Configuration files

- `retail.json`: Favorita/M5 backbone settings, gate optimization, and final M5 weekly schedule.
- `rossmann.json`, `rohlik.json`: natural-protocol Chronos and reference settings.
- `rohlik_supervised.json`: Rohlik LightGBM/TFT configurations and seeds.
- `neural.json`: PatchTST/SparseTSF search spaces, seeds, and streaming fusion settings.
- `upstream_sources.json`: fixed upstream source revisions, URLs, hashes, and license files.
- `favorita_origins.json`: Favorita's fitting, development, and test origins. M5 dates are generated from `retail.json`.
- `environments/`: original environment pins for each experiment family.

Changing a configuration changes the experiment identity. Use a new output directory for changed configurations.

## Method

Each query uses 56 days of history to forecast seven days. Shape retrieval selects eight windows by normalized history distance. Factor retrieval ranks the same library separately for each of two known future factors, alternates between the rankings, and skips duplicate windows.

For backbone `B`, shape reference `S`, and factor reference `F`:

```text
alpha = sigmoid(X @ theta_alpha)
beta  = sigmoid(X @ theta_beta)
R     = S + alpha * (F - S)
P     = max(0, B + beta * (R - B))
```

The effective weights are `(1-beta, beta*(1-alpha), beta*alpha)`. The gates use origin features, current candidate disagreements, and shrunk historical errors.

### Historical references

For a 56-day window, the scale is `max(mean(history), 1)`. Shape distance is the mean squared distance between scale-normalized histories. For each factor, mismatch is the fraction of the seven future factor values that differ. The factor ranking score is:

```text
0.5 * (shape_distance / (1 + shape_distance) + factor_mismatch)
```

The published configuration uses eight unique references: four turns from factor 1 and four from factor 2. Stable sorting resolves ties in input-bank order. Construct each group's bank in chronological origin order and then series-ID order to match the experiment drivers. Each reference forecast averages scale-normalized observed continuations and rescales by the query's historical mean floor.

The reusable retrieval function processes queries in batches of eight by default (`query_batch_size=8`). This limits the size of temporary history-distance arrays without truncating the eligible bank. Changing the query batch size preserves the per-query arithmetic and selection order.

A reference continuation must be fully observed before the query origin: `reference_origin + 6 days < query_origin`. Callers must also restrict the bank to the correct dataset group. The reusable retrieval API accepts reference and query dates to check this boundary. Without date metadata, eligibility is the caller's responsibility. A bank with fewer than eight windows falls back to the query's preceding seven days; the controlled Favorita experiment instead requires a sufficient bank.

### Historical error state

Only development labels calibrate reliability. On each development origin:

1. Construct features from the state of previously completed origins.
2. Record the current origin's candidate forecasts and labels.
3. Update the error state for subsequent origins after the entire target week has completed.

Per-day error is `clip(abs(actual - candidate) / max(history_mean, 1), 0, 10)`. A series' accumulated errors are shrunk toward the global mean with pseudo-count 14. The fourth historical input is `log1p(number_of_completed_rows)`. Empty global history gives zero errors and support; unseen series use the global prior.

Freeze this state before test inference. Completed test weeks may supply later retrieval windows under the time boundary, while test labels never update calibration or gate parameters.

### Features and fitting

The core exposes the same dataset-specific base feature definitions as the experiment code. Add the four historical inputs to obtain 15 inputs for Favorita, 12 for M5, and 14 for Rossmann or Rohlik. Standardize continuous columns using development-row means and population standard deviations with a floor of `1e-8`; leave binary factors unchanged and add an intercept.

Bank-size features use `log1p(bank_size)` on all four datasets. This is the implementation of the quantity written as `log(bank_size)` for Favorita and M5 in the manuscript.

Both gates start at zero parameters. The loss is:

```text
sum(sqrt((prediction - actual)**2 + 1)) / sum(actual)
    + 0.002 * sum(non_intercept_parameters**2)
```

The optimizer is SciPy L-BFGS-B with analytical gradients, `maxiter=250`, `ftol=3*1e-9`, and `gtol=3*1e-6`. Optimization failure raises an error. The low-level `fit_gate` / `predict_gate` functions accept an already standardized design matrix including an intercept. The `fit_fusion` wrapper fits normalization on the supplied development features and saves it with the gate parameters. Neither wrapper chooses temporal splits for the caller.

### Scoring

WAPE is `sum(abs(actual - prediction)) / sum(actual)`, reported as a ratio unless multiplied by 100. MAE is the mean absolute error on the same rows. Aggregate errors across all evaluation rows before dividing; do not average weekly WAPE values. Test labels are joined only for scoring.
