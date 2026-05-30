import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from model_definitions import ExecutablePINN
from utils import rmse, r2_score_np, mae_np, mape_np


def evaluate_model(cfg):
    test_df = pd.read_csv(cfg.data_dir / "test.csv")
    with open(cfg.model_dir / "scaler.pkl", "rb") as f:
        scaler_obj = pickle.load(f)
    scaler = scaler_obj["scaler"]
    x_test = scaler.transform(test_df[cfg.input_features].values)
    y_true = test_df[cfg.target_column].values.astype(float)

    ckpt = torch.load(cfg.model_dir / "trained_model.pth", map_location="cpu")
    model = ExecutablePINN(input_dim=ckpt.get("input_dim", cfg.input_dim), cycle_feature_index=ckpt.get("cycle_feature_index", cfg.cycle_feature_index))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    preds = []
    with torch.no_grad():
        for i in range(0, len(x_test), 4096):
            xb = torch.tensor(x_test[i:i + 4096], dtype=torch.float32)
            pred = model.predict_soh(xb).numpy().reshape(-1)
            preds.append(pred)
    pred_target = np.concatenate(preds)

    if cfg.target_column in ["capacity_loss", "Q_cell_loss"]:
        true_soh = 1.0 - y_true
        pred_soh = 1.0 - pred_target
    else:
        true_soh = y_true
        pred_soh = pred_target

    metrics = {
        "test_RMSE": float(rmse(true_soh, pred_soh)),
        "test_R2": float(r2_score_np(true_soh, pred_soh)),
        "test_MAE": float(mae_np(true_soh, pred_soh)),
        "test_MAPE": float(mape_np(true_soh, pred_soh)),
        "num_test_samples": int(len(test_df)),
        "test_cells": sorted(test_df["cell_id"].unique().tolist()),
    }
    with open(cfg.results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    out = pd.DataFrame({
        "cell_id": test_df["cell_id"].values,
        "cycle_index": test_df["cycle_index"].values,
        "true_target": y_true,
        "predicted_target": pred_target,
        "true_soh": true_soh,
        "predicted_soh": pred_soh,
        "split": "test",
    })
    out.to_csv(cfg.results_dir / "test_predictions.csv", index=False)

    plt.figure(figsize=(10, 5))
    for cell_id, g in out.groupby("cell_id"):
        g = g.sort_values("cycle_index")
        plt.plot(g["cycle_index"], g["true_soh"], label=f"{cell_id} true", linewidth=2)
        plt.plot(g["cycle_index"], g["predicted_soh"], "--", label=f"{cell_id} pred", linewidth=1.5)
    plt.xlabel("Cycle index")
    plt.ylabel("SOH")
    plt.title("Test SOH: true vs predicted")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(cfg.results_dir / "test_soh_true_vs_predicted.png", dpi=160)
    plt.close()
