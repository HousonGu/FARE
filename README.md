# FARE

**Factor-Aware Retrieval with Error-Guided Fusion for Retail Forecasting**

Retail demand is driven by known future factors such as promotions, holidays and calendar events, yet a single forecasting backbone often reacts to them poorly. FARE is a lightweight layer on top of any backbone forecast. It looks up similar past weeks, builds reference forecasts from what happened after them, and learns how far to move the backbone towards those references.

## How it works

- **Two kinds of historical references.** A *shape* reference comes from past windows whose recent sales look like the query. A *factor* reference comes from past windows whose upcoming factor pattern matches the forecast week.
- **Two gates.** One gate decides how to mix the two references; the other decides how strongly the mixed reference corrects the backbone. Both are small logistic models fitted on development weeks.
- **Error-guided reliability.** Each gate also sees how accurate the backbone and each reference have been for that series in earlier, already completed weeks. The calibration is frozen before testing.

The method needs only CPU, and it works with forecasts from any separately fitted backbone.

## Repository layout

```text
src/fare/        Core method: retrieval, features, error state, and gate fusion
data_provider/   Dataset loaders and prepared panels
backbones/       Backbone adapters (Chronos-2, LightGBM, TFT, PatchTST, SparseTSF)
exp/             Experiment runners for references, fusion, and scoring
configs/         Experiment settings and environment pins
scripts/         Per-dataset launch scripts
run.py           Unified entry point
docs/            Method details and reproduction guide
.github/         Installation and entry-point checks
```

## Documentation

- [`docs/METHOD.md`](docs/METHOD.md): retrieval, error state, gate features and fitting, scoring.
- [`docs/REPRODUCE.md`](docs/REPRODUCE.md): environment, data preparation, and how to run the experiments.

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
