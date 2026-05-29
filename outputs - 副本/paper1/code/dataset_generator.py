import json
from pathlib import Path

import numpy as np
import pandas as pd


def _entropy_like(x):
    x = np.asarray(x)
    p = np.abs(x) + 1e-6
    p = p / p.sum()
    return -np.sum(p * np.log(p + 1e-12))


def generate_dataset(cfg):
    rng = np.random.default_rng(cfg.seed)
    rows_meta = []
    rows_traj = []
    rows_feat = []
    rows_lab = []

    feature_specs = {
        "current_mean": (0.35, 0.55, -0.15),
        "current_standard_deviation": (0.02, 0.08, 0.10),
        "current_kurtosis": (2.4, 0.35, 0.40),
        "current_skewness": (0.15, 0.25, 0.35),
        "current_window_charging_time": (850.0, 120.0, 420.0),
        "current_window_accumulated_charge": (0.45, 0.08, -0.20),
        "current_curve_slope": (-0.003, 0.002, -0.018),
        "current_curve_entropy": (3.2, 0.25, 0.55),
        "voltage_mean": (4.05, 0.05, -0.20),
        "voltage_standard_deviation": (0.025, 0.01, 0.060),
        "voltage_kurtosis": (2.6, 0.30, 0.38),
        "voltage_skewness": (-0.05, 0.25, -0.30),
        "voltage_window_charging_time": (760.0, 100.0, 360.0),
        "voltage_window_accumulated_charge": (0.55, 0.08, -0.24),
        "voltage_curve_slope": (0.004, 0.002, 0.016),
        "voltage_curve_entropy": (3.0, 0.20, 0.50),
    }

    for i in range(cfg.num_cells):
        cell_id = f"cell_{i+1:03d}"
        chemistry = "NMC" if i < 5 else "LFP"
        chemistry_code = 0 if chemistry == "NMC" else 1
        protocol_code = int(i % 3)
        n_cycles = int(rng.integers(cfg.min_cycles, cfg.max_cycles + 1))
        initial_soh = float(rng.uniform(0.98, 1.02))
        target_end = float(rng.uniform(0.75, 0.90))
        knee = float(rng.uniform(0.45, 0.82))
        b = float(rng.uniform(0.015, 0.060))
        a = max(0.04, initial_soh - target_end - b * max(0.0, 1.0 - knee) ** 2)
        cell_effect = float(rng.normal(0.0, 0.04))
        feature_noise = float(rng.uniform(0.015, 0.055))
        nominal_capacity = float((2.0 if chemistry == "NMC" else 1.5) + rng.normal(0, 0.06))
        upper_v = float(4.2 if chemistry == "NMC" else 3.6)
        lower_v = float(3.0 if chemistry == "NMC" else 2.5)
        ambient = float(rng.choice([20, 25, 30, 35]) + rng.normal(0, 0.8))
        age_group = int(rng.integers(0, 3))

        rows_meta.append({
            "cell_id": cell_id,
            "chemistry": chemistry,
            "chemistry_code": chemistry_code,
            "nominal_capacity_ah": nominal_capacity,
            "upper_cutoff_voltage": upper_v,
            "lower_cutoff_voltage": lower_v,
            "ambient_temperature_c": ambient,
            "protocol_code": protocol_code,
            "cell_age_group": age_group,
            "num_cycles": n_cycles,
        })

        prev_noise = 0.0
        for c in range(1, n_cycles + 1):
            z = (c - 1) / max(1, n_cycles - 1)
            regen_noise = 0.0015 * rng.normal() + 0.45 * prev_noise
            prev_noise = regen_noise
            soh = initial_soh - a * z - b * max(0.0, z - knee) ** 2 + regen_noise
            soh = float(np.clip(soh, 0.70, 1.05))
            cap_loss = float(1.0 - soh)

            common = {
                "cell_id": cell_id,
                "cycle_index": c,
                "normalized_cycle": z,
                "split": "train" if cell_id in cfg.train_cells else ("val" if cell_id in cfg.val_cells else "test"),
            }
            rows_traj.append({**common, "soh": soh, "capacity_loss": cap_loss, "Q_cell_loss": cap_loss})
            rows_lab.append({"cell_id": cell_id, "cycle_index": c, "soh": soh, "capacity_loss": cap_loss, "Q_cell_loss": cap_loss})

            frow = {"cell_id": cell_id, "cycle_index": c}
            chem_offset = -0.04 if chemistry == "NMC" else 0.04
            proto_offset = 0.015 * (protocol_code - 1)
            for j, (fname, (base, scale, trend)) in enumerate(feature_specs.items()):
                nonlinear = 0.25 * trend * cap_loss ** 2
                cyc = 0.10 * scale * z * np.sign(trend if trend != 0 else 1)
                val = base + scale * cell_effect + trend * cap_loss + nonlinear + cyc + chem_offset * scale + proto_offset * scale
                val += rng.normal(0, feature_noise * max(abs(scale), 0.03))
                frow[fname] = float(val)

            frow.update({
                "chemistry_code": chemistry_code,
                "nominal_capacity_ah": nominal_capacity,
                "upper_cutoff_voltage": upper_v,
                "lower_cutoff_voltage": lower_v,
                "ambient_temperature_c": ambient,
                "protocol_code": protocol_code,
                "cell_age_group": age_group,
            })
            rows_feat.append(frow)

    meta = pd.DataFrame(rows_meta)
    traj = pd.DataFrame(rows_traj)
    features = pd.DataFrame(rows_feat)
    labels = pd.DataFrame(rows_lab)
    dataset = features.merge(labels, on=["cell_id", "cycle_index"], how="inner")
    dataset["split"] = dataset["cell_id"].map(lambda x: "train" if x in cfg.train_cells else ("val" if x in cfg.val_cells else "test"))

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    meta.to_csv(cfg.data_dir / "cell_metadata.csv", index=False)
    traj.to_csv(cfg.data_dir / "cycle_soh_trajectories.csv", index=False)
    features.to_csv(cfg.data_dir / "features.csv", index=False)
    labels.to_csv(cfg.data_dir / "labels.csv", index=False)
    dataset.to_csv(cfg.data_dir / "model_dataset.csv", index=False)
    dataset[dataset["split"] == "train"].to_csv(cfg.data_dir / "train.csv", index=False)
    dataset[dataset["split"] == "val"].to_csv(cfg.data_dir / "val.csv", index=False)
    dataset[dataset["split"] == "test"].to_csv(cfg.data_dir / "test.csv", index=False)

    corr = {}
    for f in cfg.input_features:
        corr[f] = float(dataset[f].corr(dataset["soh"]))
    report = {
        "num_cells": int(dataset["cell_id"].nunique()),
        "rows": int(len(dataset)),
        "train_cells": cfg.train_cells,
        "val_cells": cfg.val_cells,
        "test_cells": cfg.test_cells,
        "input_feature_target_correlations": corr,
        "note": "Feature-level synthetic data: raw voltage/current extraction windows are represented by correlated engineered features, not by raw signal processing.",
    }
    with open(cfg.data_dir / "feature_generation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
