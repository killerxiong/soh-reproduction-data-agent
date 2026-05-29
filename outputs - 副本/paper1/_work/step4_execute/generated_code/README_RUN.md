# Wang 2024 PINN Battery SOH Executable Reproduction

This package is a self-contained executable approximation of the paper-specific PINN described in the normalized plan.

## Model implemented

`model_definitions.py` defines `ExecutablePINN`, not a generic model router. It contains:

- Solution network `F(.)`: maps 16 engineered current/voltage charge-curve features plus `cycle_index` to SOH.
- Dynamics network `G(.)`: maps constructed degradation-state variables to a degradation rate.
- PINN training loss: data MSE + monotonicity loss + PDE/degradation residual loss.

Because the original raw battery datasets and exact PDE details are unavailable, this is an executable feature-level synthetic reproduction rather than a strict numerical reproduction.

## Run

```bash
python run_pipeline.py --plan normalized_plan.json --out_dir outputs/paper1_run
```

The `--plan` file is loaded for interface compatibility; this package already encodes the paper-specific plan supplied at generation time.

## Outputs

The run creates:

- `data/cell_metadata.csv`
- `data/cycle_soh_trajectories.csv`
- `data/features.csv`
- `data/labels.csv`
- `data/model_dataset.csv`
- `data/train.csv`, `data/val.csv`, `data/test.csv`
- `model/trained_model.pth`
- `model/scaler.pkl`
- `results/metrics.json`
- `results/test_predictions.csv`
- `results/test_soh_true_vs_predicted.png`
- `results/training_history.csv`
- `results/training_report.md`
- `results/model_alignment_report.md`

## Dependencies

Python packages: `numpy`, `pandas`, `scikit-learn`, `torch`, `matplotlib`.
