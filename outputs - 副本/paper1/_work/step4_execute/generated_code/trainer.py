import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from model_definitions import ExecutablePINN
from utils import rmse, r2_score_np, mae_np, mape_np


def _load_split(cfg, split):
    return pd.read_csv(cfg.data_dir / f"{split}.csv")


def _make_loader(x, y, batch_size, shuffle):
    ds = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y.reshape(-1, 1), dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def pinn_loss(model, xb, yb, cfg):
    xb = xb.clone().detach().requires_grad_(True)
    pred, dyn = model(xb)
    data_loss = torch.mean((pred - yb) ** 2)
    grad = torch.autograd.grad(pred.sum(), xb, create_graph=True, retain_graph=True)[0]
    d_soh_d_cycle = grad[:, cfg.cycle_feature_index:cfg.cycle_feature_index + 1]
    mono_loss = torch.mean(torch.relu(d_soh_d_cycle) ** 2)
    pde_loss = torch.mean((d_soh_d_cycle - dyn) ** 2)
    total = data_loss + cfg.alpha_mono * mono_loss + cfg.beta_pde * pde_loss
    return total, data_loss.detach(), mono_loss.detach(), pde_loss.detach()


def _predict(model, x, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), 4096):
            xb = torch.tensor(x[i:i + 4096], dtype=torch.float32, device=device)
            pred = model.predict_soh(xb).detach().cpu().numpy().reshape(-1)
            preds.append(pred)
    return np.concatenate(preds)


def train_model(cfg):
    train_df = _load_split(cfg, "train")
    val_df = _load_split(cfg, "val")

    for col in cfg.input_features + [cfg.target_column]:
        if col not in train_df.columns:
            raise ValueError(f"Missing required column {col} in train.csv")

    scaler = MinMaxScaler(feature_range=cfg.feature_range)
    x_train = scaler.fit_transform(train_df[cfg.input_features].values)
    x_val = scaler.transform(val_df[cfg.input_features].values)
    y_train = train_df[cfg.target_column].values.astype(np.float32)
    y_val = val_df[cfg.target_column].values.astype(np.float32)

    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.model_dir / "scaler.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "input_features": cfg.input_features, "target_column": cfg.target_column}, f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ExecutablePINN(input_dim=cfg.input_dim, cycle_feature_index=cfg.cycle_feature_index).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    train_loader = _make_loader(x_train, y_train, cfg.batch_size, cfg.shuffle_train_batches)

    best_state = None
    best_val_rmse = float("inf")
    best_epoch = -1
    patience_count = 0
    history = []

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        losses = []
        data_losses = []
        mono_losses = []
        pde_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            total, dl, ml, pl = pinn_loss(model, xb, yb, cfg)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            losses.append(float(total.detach().cpu()))
            data_losses.append(float(dl.cpu()))
            mono_losses.append(float(ml.cpu()))
            pde_losses.append(float(pl.cpu()))

        val_pred = _predict(model, x_val, device)
        val_rmse = rmse(y_val, val_pred)
        val_r2 = r2_score_np(y_val, val_pred)
        val_mae = mae_np(y_val, val_pred)
        row = {
            "epoch": epoch,
            "train_total_loss": float(np.mean(losses)),
            "train_data_loss": float(np.mean(data_losses)),
            "train_mono_loss": float(np.mean(mono_losses)),
            "train_pde_loss": float(np.mean(pde_losses)),
            "val_RMSE": float(val_rmse),
            "val_R2": float(val_r2),
            "val_MAE": float(val_mae),
        }
        history.append(row)

        if val_rmse < best_val_rmse - cfg.early_stopping_min_delta:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= cfg.early_stopping_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": cfg.input_dim,
        "cycle_feature_index": cfg.cycle_feature_index,
        "input_features": cfg.input_features,
        "model_class": cfg.generated_model_class,
    }, cfg.model_dir / "trained_model.pth")

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(cfg.results_dir / "training_history.csv", index=False)

    report = f"""# Training Report\n\nPaper: {cfg.paper_name}\n\nModel family: PINN\n\nGenerated class: `{cfg.generated_model_class}`\n\nTarget: SOH\n\nInput features: {len(cfg.input_features)} engineered features/cycle variables.\n\nPreprocessing: MinMaxScaler(feature_range=[-1, 1]) fit on train split only.\n\nOptimizer: Adam(lr={cfg.learning_rate}, weight_decay={cfg.weight_decay})\n\nLoss: L = L_data + {cfg.alpha_mono} * L_mono + {cfg.beta_pde} * L_pde.\n\nBest validation RMSE: {best_val_rmse:.6f} at epoch {best_epoch}.\n\nExecuted epochs: {len(history)}.\n\nFallback used: false. This is an executable approximation of the paper-specific F/G PINN because exact raw data and full PDE equation details were unavailable.\n"""
    with open(cfg.results_dir / "training_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    alignment = f"""# Model Alignment Report\n\n1. Paper-reported model name: {cfg.paper_model_name}\n\n2. Generated model class name: {cfg.generated_model_class}\n\n3. Paper-reported input features: 16 engineered current/voltage charge-curve statistics plus cycle/time.\n\n4. Generated model input features: {', '.join(cfg.input_features)}\n\n5. Implemented architecture components:\n   - Solution network F(.): 17 -> 60 -> 60 -> 32 -> 32 -> 1 with sine activations and sigmoid-scaled SOH output.\n   - Dynamics network G(.): 4 -> 60 -> 60 -> 1 with sine activations.\n   - Data MSE loss.\n   - Monotonicity penalty using autograd derivative d(SOH)/d(cycle).\n   - PDE/degradation residual loss matching d(SOH)/d(cycle) to G(.).\n\n6. Approximated architecture components:\n   - Exact degradation PDE is approximated by residual mean((d_soh_d_cycle - G(.))^2).\n   - Raw signal feature extraction is replaced by feature-level synthetic generation.\n\n7. Missing details from the paper:\n   - Original raw datasets and exact 10-cell split.\n   - Complete PDE implementation details.\n   - Exact learning rate, epoch count, early stopping, and architecture context for this synthetic setting.\n\n8. Fallback used: false\n\n9. Reason for fallback, if any: none; a paper-specific executable PINN approximation was implemented.\n\n10. Reproduction status: executable approximation, not strict numerical reproduction.\n"""
    with open(cfg.results_dir / "model_alignment_report.md", "w", encoding="utf-8") as f:
        f.write(alignment)
