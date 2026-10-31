# Reproducing the experiments

Run every command from the repository root. Method details are in [`METHOD.md`](METHOD.md).

## Quick start

Use Python 3.10 or newer for the core package. From the repository root:

```bash
python -m pip install -e .
python run.py --list
python run.py --dataset favorita --backbone chronos --stage audit --dry-run
```

These commands install the core method, list supported experiments, and preview a command without loading data or models. Full experiments additionally require the dataset and the selected backbone's dependencies.

## Environment

Use Python 3.12 for the original experiment environments. Install a PyTorch build for the machine using the [official installation selector](https://pytorch.org/get-started/locally/), then install the requirements for the selected experiment:

```bash
python -m pip install -e .
python -m pip install -r configs/environments/retail.txt
```

`retail.txt` covers Favorita/M5, and provides the Chronos dependencies used by Rossmann/Rohlik. Rossmann/Rohlik LightGBM/TFT use `rohlik_supervised.txt`; PatchTST/SparseTSF uses `neural.txt`. The retail, neural, conventional and residual/RAF files retain recorded version pins; the supervised and comparison/ablation files also contain version ranges and are not full environment lockfiles. Use separate environments for configurations with different pins, and retain the installed package versions with your run. Root `requirements.txt` installs the lightweight method/data-check dependencies, not every backbone.

## Data preparation

Download the datasets from their official competition pages using your own access:

- [Favorita / Store Sales – Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data): `train.csv`, `stores.csv`, `holidays_events.csv`.
- [M5 Forecasting – Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data): `sales_train_evaluation.csv`, `calendar.csv`, `sell_prices.csv`.
- [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales/data): `train.csv`, `store.csv`.
- [Rohlik Sales Forecasting Challenge V2](https://www.kaggle.com/competitions/rohlik-sales-forecasting-challenge-v2/data): `sales_train.csv`, `sales_test.csv`, `inventory.csv`, `calendar.csv`.

The loaders use the files listed above, not arbitrary similarly named datasets. The Favorita data is the 1,782 store–family panel, rather than the older item-level grocery competition. Rohlik uses the V2 panel. Place them in `data/raw/favorita`, `data/raw/m5`, `data/raw/rossmann`, and `data/raw/rohlik`, respectively. Use `run.py --raw-root` to choose another parent directory. See [Running the experiments](#running-the-experiments) for prepared panels and output conventions.

### Protocol

Every query has a complete 56-day history and a seven-day forecast horizon. Origins are nonoverlapping Mondays. The retained protocol/configuration files define exact split dates.

- Favorita: promotion category and holidays; retrieve within family. The controlled reference library excludes high-promotion and holiday combinations. Preserve missing calendar days rather than filling them with zero.
- M5: SNAP and calendar events; retrieve within item. Listing eligibility is derived from the first available price week; price values are excluded as forecasting inputs.
- Rossmann: promotion and school holidays; retrieve within store type and assortment. Evaluate all days without forcing forecasts to zero on closed days.
- Rohlik: any positive value among seven discount fields, plus warehouse holidays; retrieve within warehouse and level-2 category. Retain the provided availability-adjusted sales target without a second adjustment.

The benchmark assumes the two future factors are known at the forecast origin. Observed future sales are never an input. Historical test weeks can enter later query histories only after completion.

### Backbones

The Chronos-2 experiments use model `amazon/chronos-2`, package `chronos-forecasting==2.3.2`, a 56-day context, seven-day median predictions, and `cross_learning=False`. Weekday one-hot values and the two dataset factors form nine known covariate channels.

Weights are not included. The native runners use the model identifier in `configs/`; Favorita/M5 also expose `python -m exp.retail --chronos-model` for a local checkpoint. LightGBM and TFT require the experiment dependencies; PatchTST and SparseTSF additionally require the external source checkout described [below](#patchtst-and-sparsetsf-source). The lightweight `fare` package can work with forecasts from any separately fitted backbone.

## External implementations

### Installed model packages

- Chronos-2 is called through [`chronos-forecasting`](https://github.com/amazon-science/chronos-forecasting).
- LightGBM is called through [`lightgbm`](https://github.com/microsoft/LightGBM).
- TFT is called through [`pytorch-forecasting`](https://github.com/sktime/pytorch-forecasting). The local files construct samples, select configurations, and call the library model.

Their model libraries and pretrained weights are not copied into `src/fare/`.

The Chronos runner downloads the pretrained checkpoint through the model library when needed and keeps its weights frozen. LightGBM and TFT are trained on the configured fitting period; FARE gates are fitted on development predictions.

### PatchTST and SparseTSF source

`python -m backbones.fetch_upstream` reads `configs/upstream_sources.json` and retrieves:

- [PatchTST](https://github.com/yuqinie98/PatchTST)
- [SparseTSF](https://github.com/lss-1138/SparseTSF)

The fetcher downloads the listed files and applies the recorded import-namespace adaptation. It also retrieves the upstream license files. Downloaded files live under `vendor/` and are excluded from the source release by default. The manifest records original URLs and exact files; model computations retain the original implementation.

Keep the downloaded upstream license files with those sources.

## Running the experiments

Run commands from the repository root. The installed `fare` package provides the core method; `data_provider/`, `backbones/`, and `exp/` run from this checkout.

### Available combinations

- Favorita and M5: Chronos-2, LightGBM, TFT, PatchTST, SparseTSF.
- Rossmann: Chronos-2, LightGBM, TFT, PatchTST, SparseTSF.
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

Default raw CSV root is `data/raw/`; default output is `results/DATASET/BACKBONE/`. `--raw-root`, `--out`, `--device`, and `--data-root` override these. Add `--dry-run` to inspect the dispatched command before execution. `scripts/fare/DATASET/BACKBONE.sh` forwards its arguments to the same entry point, for example:

```bash
bash scripts/fare/rossmann/chronos.sh --stage audit --device cpu
```

Rossmann/Rohlik raw-data runners also accept `--raw /path/to/actual-csv-directory` when the downloaded folder has a different name.

Stages are `audit`, `references`, `backbone`, `fusion`, `score`, and `all`. The Rossmann and Rohlik supervised runners retain a combined gate-fitting-and-scoring stage, so both `fusion` and `score` invoke that operation. The Chronos runners separate their native fusion and scoring stages; the neural runner verifies completed fusion outputs in `score`. Test labels never fit the gates.

Favorita/M5 outputs add `favorita_original/` or `m5_full/` below the selected output root. The native modules also expose their detailed CLI, such as `python -m exp.retail --help` or `python -m exp.rohlik --help`.

### PatchTST and SparseTSF

These two backbones use prepared panels and separately downloaded upstream source. First fetch their sources and licenses:

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

Panel export requires a new output directory and verifies reference coverage and evaluation keys. Neural outputs add a dataset subdirectory below `--out`. For PatchTST/SparseTSF, `fusion` writes the test metrics; `score` re-verifies completed fusion outputs and does not rerun predictions. Generate references with the raw-data runner before panel export, rather than requesting a neural `references` stage.

### Configuration and results

`configs/retail.json` defines the M5 165/52/52-origin schedule and Favorita's fitting/validation settings. `configs/favorita_origins.json` supplies Favorita's 62/24/25 fitting/development/test origins. Rossmann/Rohlik use their dataset-specific configurations. Gates and standardization use development rows; calibration is frozen in testing.

Each run writes `metrics.json` with overall and weekly metrics. Result column `soft_history` (Favorita/M5/Rossmann/Rohlik runners) or `with` (PatchTST/SparseTSF) denotes the full FARE method. Overall WAPE is total absolute error divided by total actual sales across all evaluation rows; do not average weekly WAPE values. Seeds are scored separately; average their metrics rather than ensembling predictions.

Use a new output directory after code, data, configuration, or environment changes. Resume manifests reject a changed configuration, schedule or environment.

### Configuration files

- `retail.json`: Favorita/M5 backbone settings, gate optimization, and final M5 weekly schedule.
- `rossmann.json`, `rohlik.json`: natural-protocol Chronos and reference settings.
- `rohlik_supervised.json`: shared Rossmann/Rohlik LightGBM/TFT configurations and seeds; each dataset adapter supplies its own fitting schedule and features.
- `neural.json`: PatchTST/SparseTSF search spaces, seeds, and streaming fusion settings.
- `upstream_sources.json`: PatchTST/SparseTSF source URLs and license files.
- `third_party.json`: repositories of TimeFuse, PFRP, PIR and SARAF.
- `favorita_origins.json`: Favorita's fitting, development, and test origins. M5 dates are generated from `retail.json`.
- `environments/`: original environment pins for each experiment family (`retail`, `rohlik_supervised`, `neural`, `conventional`, `residual_raf`, `comparisons`, `ablations`).

Changing a configuration changes the experiment identity. Use a new output directory for changed configurations.

## Comparison methods

All comparison methods read one shared input folder per dataset. Build it after the main Chronos-2 run of that dataset (`bash scripts/fare/DATASET/chronos.sh`):

```bash
bash scripts/baselines/prepare_inputs.sh DATASET
```

This exports the prepared panel (`python -m exp.prepare_panel`) to `data/prepared/DATASET` and then writes `data/baselines/DATASET/` with the sales panel, the two known factors from the raw CSV files, the references, the frozen Chronos-2 forecasts (`backbone/`), and the FARE test predictions and gate scaler (`fare/`). Every file is listed in `complete.json`, and each method checks that they are present before running.

| Method | Command | Environment |
| --- | --- | --- |
| Seasonal naive, ETS, SARIMA, Ridge, XGBoost | `bash scripts/baselines/conventional.sh [--dataset D] [--method M]` | `conventional.txt` |
| LightGBM residual correction, RAF | `bash scripts/download_chronos.sh`, then `bash scripts/baselines/residual_raf.sh [--dataset D] [--method raf\|residual]` | `residual_raf.txt` |
| TimeFuse on the stored B/S/F candidates | `bash scripts/baselines/timefuse.sh [D ...]` | `retail.txt` |
| PFRP and its DLinear backbone | `bash scripts/baselines/pfrp.sh [D ...]` | `comparisons.txt` |
| PIR on the frozen Chronos-2 forecasts | `bash scripts/baselines/pir.sh [D ...]` | `comparisons.txt` |
| SARAF and its linear control | `bash scripts/baselines/saraf.sh [D ...]` | `comparisons.txt` |

TimeFuse, PFRP, PIR and SARAF use the upstream implementations of the repositories in `configs/third_party.json`. Fetch them once with `bash scripts/fetch_third_party.sh`; the adapters load the upstream modules from `third_party/` and leave the fetched files unchanged. RAF and the residual corrector load the `amazon/chronos-2` weights from `models/chronos-2`.

TimeFuse on Rossmann and Rohlik (`baselines/timefuse/rossmann_rohlik.py`) reads the references and Chronos-2 forecasts of the completed main run (`results/D/chronos`) and the 56-day histories from the raw CSV files; Favorita and M5 read the shared input folder.

Outputs go to `results/baselines/METHOD/`. Each run writes per-origin predictions and a `metrics.json` with overall and weekly WAPE and MAE on the test rows. Seeded methods (XGBoost, LightGBM residual, PFRP, PIR, SARAF) write one folder per seed; average their metrics rather than ensembling predictions.

## Ablations and sensitivity

The FARE Chronos-2 runners fit several ablations and allocation controls next to the full model (`soft_history`). On all four datasets: `soft_no_history` (w/o historical errors), `shape_half` and `factor_half` (fixed 50:50 blends), `global_static` (development-fitted global weights), `history_hard` (historical hard selection) and `flat_softmax` (one-level softmax). On Favorita and M5 also `soft_history_no_factor` (w/o factor reference) and `soft_history_no_disagreement` (w/o disagreements). They appear as columns of the gate predictions and in `metrics.json`.

Two further studies read the shared input folder above:

```bash
bash scripts/ablations/candidate_paths.sh [D ...]   # learned B+S, learned B+F, constant alpha and beta
bash scripts/ablations/sensitivity.sh [D ...]       # lambda, clip, smoothing constant, L2 weight
```

The candidate-path controls keep every gate input and the full-model scaler, and change only the output path. The sensitivity study changes one setting at a time (lambda in {0, 3.5, 7, 28, 56}, clip in {5, 20}, smoothing constant in {0.1, 10}, L2 weight in {1e-4, 1e-2}), refits both gates on development rows, and scores the test rows once. Both use `ablations.txt` and run on CPU.

The retrieval and error-estimation ablations read the completed Chronos-2 run of each dataset (`bash scripts/fare/D/chronos.sh`) and leave it unchanged:

```bash
bash scripts/ablations/extended.sh [D ...]
```

| Variant | Change |
| --- | --- |
| `A1_joint` | factor reference F from the joint shape-and-factor Top-8 |
| `A2_random` | F from 8 random library entries; seeds 20261005, 20261006, 20261007 |
| `A3_global_errors` | historical errors replaced by the global past mean |
| `A4_no_shrink` | per-series historical errors without shrinkage |
| `A5_constant_alpha` | alpha reduced to a constant |
| `A6_constant_beta` | beta reduced to a constant |
| `no_factor` | Rossmann/Rohlik: factor reference F and its gate inputs removed |
| `no_disagreement` | Rossmann/Rohlik: the two disagreement inputs removed |

A1 and A2 rebuild the references from the same past-origin library and keep S unchanged; the other variants reuse the stored references and Chronos-2 forecasts. Each variant refits the gates on development rows and scores the test rows once. Outputs go to `results/ablations/extended/D/VARIANT_SEED/` with `predictions.parquet`, `fit.json` and `metrics.json` (variant and full model on the same rows). Run in the environment of the main run (`retail.txt`): the raw files and package versions are checked against the run's `audit.json`. A single variant is `python ablations/extended/run.py --source results/m5/chronos/m5_full --out OUT --variant A4_no_shrink`.

On Rossmann and Rohlik the variants run in `ablations/extended/rossmann_rohlik.py` (`--dataset D --variant V [--seed S]`), which reads `results/D/chronos` and `data/raw/D` (`--source`, `--raw` to change them). Outputs go to `results/ablations/extended/D/VARIANT_SEED/`, rebuilt A1/A2 references to `results/ablations/extended/D/references/`. Use `retail.txt`. On Favorita and M5, `no_factor` and `no_disagreement` are the built-in controls above.

## Computational cost

`efficiency/` measures wall time and peak memory of the stages of FARE, TimeFuse and RAF at one test origin per process. `efficiency/profiler.py` wraps the functions listed in `efficiency/specs/METHOD.json` and runs `efficiency/entry.py`, which calls the same reference, gate, TimeFuse and RAF functions as the experiment runners.

| Stage | FARE | TimeFuse | RAF |
| --- | --- | --- | --- |
| `reference` | references S/F for the origin, with every preceding completed origin in the library (shared by FARE and TimeFuse) | as FARE | — |
| `fusion` | gate inputs, gate fit on development rows, test prediction | meta features, fusor training, test prediction | no training |
| `raf` | — | — | retrieval plan and Chronos-2 prediction on the augmented context |

`fusion` reads the stored B/S/F candidates; the Chronos-2 forecasts are not recomputed. Inputs are the shared folders in `data/baselines/` (`bash scripts/baselines/prepare_inputs.sh D`), the raw CSV files for `reference`, and `models/chronos-2` for RAF (`bash scripts/download_chronos.sh`).

```bash
bash scripts/efficiency/run.sh check   # resolve every measured function
bash scripts/efficiency/run.sh smoke   # at most 8 query series per stage
bash scripts/efficiency/run.sh run     # first, middle and last test origin, 3 processes each
bash scripts/efficiency/run.sh scale   # FARE reference stage, library fractions 1/8, 1/4, 1/2, 1
```

Run on one otherwise idle machine and one measurement at a time. FARE and TimeFuse use `retail.txt`; RAF uses `residual_raf.txt`. Choose the interpreter and methods with `EFFICIENCY_PYTHON` and `EFFICIENCY_METHODS`, for example `EFFICIENCY_METHODS=raf EFFICIENCY_PYTHON=/path/to/raf/python`. Other settings: `EFFICIENCY_DEVICE` (`cpu`, `cuda`), `EFFICIENCY_THREADS` (default 8), `EFFICIENCY_DATASETS`, `EFFICIENCY_POSITIONS`, `EFFICIENCY_REPS` (default 3), `EFFICIENCY_SCALE_DATASETS` (default `m5 rossmann`) and `EFFICIENCY_OUTPUT` (a new directory; default `results/efficiency/TIMESTAMP`). The FARE gate optimizer keeps its internal four-thread limit. `psutil` is optional for memory sampling.

The cost table at the last test origin is measured with

```bash
bash scripts/efficiency/cost_table.sh check|smoke|run
```

For each method it measures the one-time stage (development-period references, features and fitting, or the RAF retrieval plan) and the stage for one new origin (retrieval and prediction): FARE (`dev_reference`, `fusion`, `reference`), TimeFuse (`fusion`, `reference`), the Chronos-2 forecast B (`backbone_dev`, `backbone`), the LightGBM residual corrector, PIR, PFRP and SARAF (`fit`, `test`) and RAF (`raf_plan`, `raf_test`). These stages are in `efficiency/runtime_extra.py` with the specs `efficiency/specs/cost_*.json`. Each method runs in its own environment (`EFFICIENCY_METHODS`, `EFFICIENCY_PYTHON`); PIR, PFRP and SARAF use `comparisons.txt` and the fetched upstream sources, the residual corrector and B use `residual_raf.txt`. Only the first seed of each configuration is timed.

Each measurement writes `profiles/LABEL/profile.json` with wall time, inclusive time and calls per measured function, sampled and `getrusage` peak RSS, GPU peak memory, the query count and, for `scale`, the full and used library sizes. The scaling study keeps a fixed random subset of each group library before the distance computation, nested across fractions. Stage outputs in `stages/` are for checking only and are not used for accuracy.

## Paper tables and scripts

| Paper content | Scripts |
| --- | --- |
| Main comparison (conventional and conference methods, FARE) | `scripts/fare/D/chronos.sh`, `scripts/baselines/*.sh` |
| Component ablations | built-in controls of `scripts/fare/D/chronos.sh`; `scripts/ablations/candidate_paths.sh` |
| Mechanism analysis (fixed blends, global weights, hard selection, softmax) | built-in controls of `scripts/fare/D/chronos.sh` |
| Backbones with and without FARE | `scripts/fare/D/{chronos,lightgbm,tft,patchtst,sparsetsf}.sh` |
| Retrieval and error-estimation ablations | `scripts/ablations/extended.sh` |
| Hyperparameter sensitivity | `scripts/ablations/sensitivity.sh` |
| Computational cost | `scripts/efficiency/cost_table.sh` |
| Library-size scaling | `scripts/efficiency/run.sh scale` |
