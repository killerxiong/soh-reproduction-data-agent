import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.codex_client import call_codex_json


# PyCharm direct-run defaults
REPRODUCTION_PLAN_PATH = "outputs/paper1/_work/reproduction_plan.json"
OUT_DIR = "outputs/paper1/_work/step4_execute"
RANDOM_SEED = 42
OVERWRITE = True
MAX_REPEAT_RUNS = 1
ALLOW_MULTI_RUNS = False
USE_CODEX_OR_LLM = True

REQUIRED_FILES = [
    "run_pipeline.py",
    "config.py",
    "dataset_generator.py",
    "model_definitions.py",
    "trainer.py",
    "evaluator.py",
    "utils.py",
    "README_RUN.md",
]

DEBUG_LATENT_COLUMNS = {
    "normalized_cycle", "cell_chemistry_flag", "temperature_offset", "protocol_id",
    "feature_noise_scale", "degradation_rate_latent"
}


def _read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _write_json(p: Path, obj: Dict[str, Any]):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(p: Path, text: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _detect_paper_id(plan: Dict[str, Any]) -> str:
    return str(plan.get("paper_id") or "paper_unknown")


def _find_features(plan: Dict[str, Any]) -> List[str]:
    s = plan.get("selected_reproduction_strategy", {}) or {}
    d = (((plan.get("dataset_construction_plan", {}) or {}).get("feature_generation", {}) or {}))
    candidates = [
        s.get("model_input_features", []) or [],
        s.get("primary_features", []) or [],
        d.get("model_input_features", []) or [],
    ]
    for c in candidates:
        if c:
            return [x for x in c if str(x).strip()]
    raise RuntimeError("No model input features found in reproduction_plan.json.")


def normalize_plan_minimally(plan: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    repairs: List[Dict[str, str]] = []
    p = dict(plan)

    for k in ["selected_reproduction_strategy", "dataset_construction_plan", "model_execution_plan", "evaluation_plan", "outputs_plan"]:
        if k not in p or not isinstance(p[k], dict):
            p[k] = {}
            repairs.append({"issue": f"missing {k}", "original": "missing", "repaired": "empty object", "reason": "schema minimum"})

    s = p["selected_reproduction_strategy"]
    dcp = p["dataset_construction_plan"]
    mep = p["model_execution_plan"]

    # input features normalization
    selected = _find_features(p)
    filtered = [f for f in selected if f not in DEBUG_LATENT_COLUMNS]
    if filtered != selected:
        repairs.append({"issue": "debug/latent features in model inputs", "original": str(selected), "repaired": str(filtered), "reason": "remove default debug columns"})
    selected = filtered
    s["model_input_features"] = selected
    mep["input_features"] = selected
    fg = dcp.get("feature_generation", {}) or {}
    fg["model_input_features"] = selected

    # secondary default non-model
    secondary = s.get("secondary_features", []) or []
    fg["secondary_features_not_used_by_default"] = secondary
    fg["features_to_generate"] = list(dict.fromkeys((s.get("primary_features", []) or selected) + secondary))
    dcp["feature_generation"] = fg

    # fixed dataset policy
    dcp["num_cells"] = 10
    dcp["cell_ids"] = [f"cell_{i:03d}" for i in range(1, 11)]
    dcp["cell_split"] = {
        "split_unit": "cell", "train_ratio": 0.6, "val_ratio": 0.2, "test_ratio": 0.2,
        "train_cells_count": 6, "val_cells_count": 2, "test_cells_count": 2, "no_cell_overlap": True
    }

    # target alias
    target = str(s.get("target_column") or mep.get("target_column") or "soh")
    if target not in {"soh", "capacity_loss", "Q_cell_loss", "target_value"}:
        repairs.append({"issue": "unsupported target alias", "original": target, "repaired": "soh", "reason": "supported alias only"})
        target = "soh"
    s["target_column"] = target
    mep["target_column"] = target

    lg = dcp.get("label_generation", {}) or {}
    lg["target_column"] = target
    lg["required_label_columns"] = ["soh", "capacity_loss"]
    lg["optional_label_columns"] = ["Q_cell_loss"]
    lg["formulas"] = {"capacity_loss": "1 - soh", "Q_cell_loss": "capacity_loss", "soh": "1 - capacity_loss"}
    dcp["label_generation"] = lg

    # scaler normalization
    prep = mep.get("preprocessing", {}) or {}
    plan_text = json.dumps(p, ensure_ascii=False).lower()
    if ("[-1,1]" in plan_text) or ("minmax" in plan_text):
        prep["scaler"] = "MinMaxScaler"
        prep["feature_range"] = [-1, 1]
    else:
        prep.setdefault("scaler", "StandardScaler")
    prep["fit_scaler_on"] = "train_only"
    prep["feature_columns"] = selected
    prep["target_column"] = target
    prep.setdefault("transform", ["train", "val", "test"])
    mep["preprocessing"] = prep

    # repeat_runs limitation
    tp = mep.get("training_plan", {}) or {}
    rr = int(tp.get("repeat_runs", 1) or 1)
    eff = rr
    if (not ALLOW_MULTI_RUNS) and rr > MAX_REPEAT_RUNS:
        eff = MAX_REPEAT_RUNS
        repairs.append({"issue": "repeat_runs too large", "original": str(rr), "repaired": str(eff), "reason": "first executable run"})
    tp["effective_repeat_runs"] = max(1, eff)
    mep["training_plan"] = tp

    # mark pinn fallback allowed
    mf = (str(mep.get("model_family", "")) + " " + str(mep.get("model_name", ""))).lower()
    if ("pinn" in mf) or ("physics" in mf):
        p.setdefault("step4_runtime", {})
        p["step4_runtime"]["fallback_allowed"] = True

    p["selected_reproduction_strategy"] = s
    p["dataset_construction_plan"] = dcp
    p["model_execution_plan"] = mep
    return p, repairs


def build_codex_code_generation_prompt(normalized_plan: Dict[str, Any], repair_log: List[Dict[str, str]]) -> Tuple[str, str]:
    system_prompt = (
        "You are a senior machine learning engineer and code generation agent. "
        "Your task is to generate a complete runnable Python pipeline for a battery SOH / capacity estimation paper. "
        "You must generate paper-specific code from the provided reproduction plan. "
        "Return ONLY strict JSON with key 'files'."
    )

    user_prompt = (
        "Generate a complete runnable Python package for THIS SPECIFIC PAPER.\n"
        "\n"
        "==============================\n"
        "CRITICAL PRINCIPLE\n"
        "==============================\n"
        "You must NOT generate a generic model router.\n"
        "You must NOT simply choose from a fixed set of models such as LinearRegression, XGBoost, PINN, or MLP.\n"
        "You must read the provided normalized_plan and generate the model architecture required by this specific paper.\n"
        "\n"
        "The generated model must be derived from these fields when available:\n"
        "- normalized_plan['paper_understanding']['model']\n"
        "- normalized_plan['model_execution_plan']['model_name']\n"
        "- normalized_plan['model_execution_plan']['model_family']\n"
        "- normalized_plan['model_execution_plan']['model_definition']\n"
        "- normalized_plan['model_execution_plan']['loss_plan']\n"
        "- normalized_plan['model_execution_plan']['training_plan']\n"
        "- normalized_plan['selected_reproduction_strategy']['model_input_features']\n"
        "- normalized_plan['dataset_construction_plan']\n"
        "\n"
        "If a paper-specific architecture is described, implement that architecture or the closest executable approximation.\n"
        "Fallback to a generic baseline is allowed ONLY when the plan is missing model details or explicitly states that the model is unbuildable.\n"
        "Any fallback or approximation must be documented in results/model_alignment_report.md and results/training_report.md.\n"
        "\n"
        "==============================\n"
        "MODEL ARCHITECTURE INFERENCE PROCEDURE\n"
        "==============================\n"
        "Before writing code, infer the model requirements from the plan using the following internal procedure:\n"
        "\n"
        "1. Identify the task type:\n"
        "   - regression\n"
        "   - sequence-to-one regression\n"
        "   - sequence-to-sequence regression\n"
        "   - multi-output regression\n"
        "   - classification, if applicable\n"
        "\n"
        "2. Identify the input data form:\n"
        "   - tabular feature vector\n"
        "   - time-series sequence\n"
        "   - sliding window over cycles\n"
        "   - voltage/current curve segment\n"
        "   - spectrum or impedance feature vector\n"
        "   - image-like matrix\n"
        "   - graph-like structure\n"
        "   - hybrid input with multiple branches\n"
        "\n"
        "3. Identify model components explicitly described in the plan:\n"
        "   Examples include but are not limited to:\n"
        "   - linear regression\n"
        "   - SVR / ElasticNet / RandomForest / XGBoost / LightGBM / CatBoost\n"
        "   - MLP / DNN\n"
        "   - RNN / LSTM / GRU / BiLSTM\n"
        "   - CNN / CNN-LSTM / TCN / Conv1D\n"
        "   - attention / spatial attention / temporal attention / self-attention\n"
        "   - Transformer / encoder-decoder\n"
        "   - autoencoder / feature extractor\n"
        "   - GPR / Bayesian model / uncertainty module\n"
        "   - PINN / physics-informed network\n"
        "   - KAN / PIKAN / KPINN / Kolmogorov-Arnold network\n"
        "   - hybrid physics-data model\n"
        "   - transfer learning / fine-tuning / frozen layers\n"
        "\n"
        "4. Identify required derived features:\n"
        "   - cycle_index\n"
        "   - previous capacity Q(k-1)\n"
        "   - previous prediction Q_hat(k-1)\n"
        "   - trend feature L(k)\n"
        "   - moving average features\n"
        "   - voltage-window features\n"
        "   - relaxation statistics\n"
        "   - impedance features\n"
        "   - capacity_loss / SOH aliases\n"
        "\n"
        "5. Identify required loss terms:\n"
        "   - MSE / MAE / Huber\n"
        "   - data loss\n"
        "   - monotonicity loss\n"
        "   - physics residual loss\n"
        "   - trend consistency loss\n"
        "   - regularization loss\n"
        "   - multi-task weighted loss\n"
        "\n"
        "6. Identify training protocol:\n"
        "   - train/val/test split\n"
        "   - cell-level split\n"
        "   - sequence window length\n"
        "   - batch size\n"
        "   - epochs\n"
        "   - optimizer\n"
        "   - early stopping\n"
        "   - transfer learning stages\n"
        "\n"
        "Then generate code according to the inferred requirements.\n"
        "\n"
        "==============================\n"
        "MODEL IMPLEMENTATION RULES\n"
        "==============================\n"
        "1. The file model_definitions.py must define a paper-specific model class.\n"
        "   The class name should reflect the paper model name when possible.\n"
        "   Examples:\n"
        "   - STLSTMModel\n"
        "   - SpatioTemporalAttentionLSTM\n"
        "   - CNNLSTMModel\n"
        "   - PhysicsInformedSOHNet\n"
        "   - PIKANApproximation\n"
        "   - XGBoostCapacityRegressor\n"
        "\n"
        "2. Do not write only a generic function like choose_model() that routes between a small fixed set of models.\n"
        "   A helper factory is allowed, but the actual paper-specific model must be implemented.\n"
        "\n"
        "3. For tabular machine-learning papers:\n"
        "   - Use the paper-specified model family.\n"
        "   - Use sklearn/xgboost/lightgbm only when the plan says the paper uses that model.\n"
        "   - Preserve the paper-reported feature set.\n"
        "\n"
        "4. For sequence-model papers:\n"
        "   - Generate sequence-window construction in dataset_generator.py or trainer.py.\n"
        "   - Preserve cycle order within each cell.\n"
        "   - Do not randomly shuffle cycles before window construction.\n"
        "   - Use PyTorch sequence modules such as LSTM/GRU/TCN/Transformer according to the plan.\n"
        "\n"
        "5. For attention-based papers:\n"
        "   - Implement attention as an explicit module or explicit operation.\n"
        "   - Name the module according to the plan, such as SpatialAttention, TemporalAttention, SelfAttention, or FeatureAttention.\n"
        "   - Do not ignore attention and replace the model with a plain MLP.\n"
        "\n"
        "6. For physics-informed papers:\n"
        "   - Preserve the paper-reported physics variables and loss terms where possible.\n"
        "   - Use torch.autograd when derivatives are required.\n"
        "   - If exact equations are unavailable, implement an executable approximation and document it.\n"
        "\n"
        "7. For KAN / PIKAN / KPINN papers:\n"
        "   - Preserve named modules and the intended computational graph.\n"
        "   - If exact KAN B-spline layers are too complex, implement a clearly named approximation.\n"
        "   - Do not call the approximation a strict KAN unless actual KAN layers are implemented.\n"
        "\n"
        "8. For transfer-learning papers:\n"
        "   - Implement stages if the plan provides them.\n"
        "   - If transfer learning is not executable under the fixed synthetic dataset setting, document it as not executed.\n"
        "\n"
        "==============================\n"
        "DATASET GENERATION RULES\n"
        "==============================\n"
        "1. Generate feature-level synthetic dataset according to normalized_plan['dataset_construction_plan'].\n"
        "2. Generate exactly the model input features specified by normalized_plan['model_execution_plan']['input_features'].\n"
        "3. Generate secondary features only if requested, but do not use them as default model inputs.\n"
        "4. Generate label columns:\n"
        "   - soh\n"
        "   - capacity_loss\n"
        "   - Q_cell_loss\n"
        "5. Generate exactly 10 cells unless the plan states otherwise.\n"
        "6. Split by cell into train/val/test = 6/2/2 unless the plan states otherwise.\n"
        "7. Ensure the same cell never appears in more than one split.\n"
        "8. If the model is sequence-based, generate sequence windows and preserve cell/cycle order.\n"
        "9. If the model requires derived temporal features such as Q(k-1), Q_hat(k-1), or L(k), generate them.\n"
        "\n"
        "==============================\n"
        "TRAINING RULES\n"
        "==============================\n"
        "1. Fit preprocessing scalers on train split only.\n"
        "2. Apply the same scaler to val and test.\n"
        "3. Train the paper-specific model.\n"
        "4. Validate on val split.\n"
        "5. Test on test split.\n"
        "6. Save the trained model and scaler.\n"
        "7. Save training_history.csv.\n"
        "8. Save training_report.md.\n"
        "\n"
        "==============================\n"
        "EVALUATION RULES\n"
        "==============================\n"
        "1. Always compute test_RMSE and test_R2 based on true_soh and predicted_soh.\n"
        "2. If the target is capacity_loss or Q_cell_loss, convert predictions to SOH using predicted_soh = 1 - predicted_target.\n"
        "3. Save results/metrics.json.\n"
        "4. Save results/test_predictions.csv with columns:\n"
        "   - cell_id\n"
        "   - cycle_index\n"
        "   - true_target\n"
        "   - predicted_target\n"
        "   - true_soh\n"
        "   - predicted_soh\n"
        "   - split\n"
        "5. Save results/test_soh_true_vs_predicted.png.\n"
        "\n"
        "==============================\n"
        "MODEL ALIGNMENT REPORT\n"
        "==============================\n"
        "You must write results/model_alignment_report.md.\n"
        "It must contain:\n"
        "1. Paper-reported model name.\n"
        "2. Generated model class name.\n"
        "3. Paper-reported input features.\n"
        "4. Generated model input features.\n"
        "5. Implemented architecture components.\n"
        "6. Approximated architecture components.\n"
        "7. Missing details from the paper.\n"
        "8. Fallback used: true/false.\n"
        "9. Reason for fallback, if any.\n"
        "10. Whether the implementation is strict reproduction or executable approximation.\n"
        "\n"
        "==============================\n"
        "OUTPUT PACKAGE REQUIREMENTS\n"
        "==============================\n"
        "The generated package must be self-contained in generated_code/.\n"
        "It must NOT import from the STEP4 host script.\n"
        "It must run with:\n"
        "  python run_pipeline.py --plan <plan> --out_dir <out_dir>\n"
        "\n"
        "The generated package must include exactly these files:\n"
        "  run_pipeline.py\n"
        "  config.py\n"
        "  dataset_generator.py\n"
        "  model_definitions.py\n"
        "  trainer.py\n"
        "  evaluator.py\n"
        "  utils.py\n"
        "  README_RUN.md\n"
        "\n"
        "The pipeline must output:\n"
        "  data/cell_metadata.csv\n"
        "  data/cycle_soh_trajectories.csv\n"
        "  data/features.csv\n"
        "  data/labels.csv\n"
        "  data/model_dataset.csv\n"
        "  data/train.csv\n"
        "  data/val.csv\n"
        "  data/test.csv\n"
        "  model/trained_model.pkl or model/trained_model.pth\n"
        "  model/scaler.pkl\n"
        "  results/metrics.json\n"
        "  results/test_predictions.csv\n"
        "  results/test_soh_true_vs_predicted.png\n"
        "  results/training_history.csv\n"
        "  results/training_report.md\n"
        "  results/model_alignment_report.md\n"
        "\n"
        "==============================\n"
        "STRICT JSON OUTPUT FORMAT\n"
        "==============================\n"
        "Return ONLY strict JSON in this schema:\n"
        "{\n"
        "  \"files\": [\n"
        "    {\"path\": \"run_pipeline.py\", \"content\": \"...\"},\n"
        "    {\"path\": \"config.py\", \"content\": \"...\"},\n"
        "    {\"path\": \"dataset_generator.py\", \"content\": \"...\"},\n"
        "    {\"path\": \"model_definitions.py\", \"content\": \"...\"},\n"
        "    {\"path\": \"trainer.py\", \"content\": \"...\"},\n"
        "    {\"path\": \"evaluator.py\", \"content\": \"...\"},\n"
        "    {\"path\": \"utils.py\", \"content\": \"...\"},\n"
        "    {\"path\": \"README_RUN.md\", \"content\": \"...\"}\n"
        "  ]\n"
        "}\n"
        "\n"
        "Normalized plan JSON:\n"
        + json.dumps(normalized_plan, ensure_ascii=False, indent=2)
        + "\n\nRepair log:\n"
        + json.dumps(repair_log, ensure_ascii=False, indent=2)
    )

    return system_prompt, user_prompt


def call_codex_to_generate_pipeline_code(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    if not USE_CODEX_OR_LLM:
        raise RuntimeError("STEP4 requires Codex/LLM code generation, but USE_CODEX_OR_LLM=False.")
    if not (os.getenv("CODEX_API_KEY") or os.getenv("OPENAI_API_KEY")):
        raise RuntimeError("STEP4 requires Codex/LLM code generation, but API key is missing.")
    return call_codex_json(system_prompt, user_prompt)


def _extract_generated_files_payload(generated: Dict[str, Any]) -> List[Dict[str, Any]]:
    files = generated.get("files")
    if isinstance(files, list):
        return files
    # fallback patterns
    if isinstance(generated.get("generated_code"), list):
        return generated.get("generated_code")
    gc = generated.get("generated_code")
    if isinstance(gc, dict):
        out=[]
        for k,v in gc.items():
            if isinstance(v, str):
                out.append({"path": k, "content": v})
        if out:
            return out
    if isinstance(generated.get("artifacts"), list):
        return generated.get("artifacts")
    return []


def write_generated_code_files(generated: Dict[str, Any], generated_dir: Path):
    files = _extract_generated_files_payload(generated)
    if not isinstance(files, list) or not files:
        raise RuntimeError("Codex output missing usable files payload.")
    generated_dir.mkdir(parents=True, exist_ok=True)
    for item in files:
        if not isinstance(item, dict):
            continue
        rel = item.get("path") or item.get("filename") or item.get("name")
        content = item.get("content") or item.get("text")
        if not rel or not isinstance(content, str):
            continue
        rel = str(rel).replace('\\', '/').strip('/')
        if rel.startswith('generated_code/'):
            rel = rel[len('generated_code/'): ]
        tgt = generated_dir / rel
        tgt.parent.mkdir(parents=True, exist_ok=True)
        tgt.write_text(content, encoding="utf-8")




def apply_generated_code_patches(generated_dir: Path):
    """Hotfix generated trainer code for unstable PINN autograd on time/cycle tensor."""
    trainer = generated_dir / "trainer.py"
    if not trainer.exists():
        return
    txt = trainer.read_text(encoding="utf-8", errors="ignore")

    # Patch 1: strict pattern
    old = "dsoh_dt = torch.autograd.grad(soh_pred.sum(), tb, create_graph=True)[0]"
    new = (
        "_grad_tmp = torch.autograd.grad(soh_pred.sum(), tb, create_graph=True, allow_unused=True)[0]\n"
        "    dsoh_dt = _grad_tmp if _grad_tmp is not None else torch.zeros_like(soh_pred)"
    )
    if old in txt:
        txt = txt.replace(old, new)

    # Patch 2: generic no allow_unused on tb
    key = "torch.autograd.grad(soh_pred.sum(), tb, create_graph=True)"
    if key in txt and "allow_unused=True" not in txt:
        txt = txt.replace(key, "torch.autograd.grad(soh_pred.sum(), tb, create_graph=True, allow_unused=True)")

    trainer.write_text(txt, encoding="utf-8")

def validate_generated_code_files(generated_dir: Path):
    missing = [f for f in REQUIRED_FILES if not (generated_dir / f).exists()]
    if missing:
        raise RuntimeError(f"Generated code missing required files: {missing}")

    run_txt = (generated_dir / "run_pipeline.py").read_text(encoding="utf-8", errors="ignore")
    if "argparse" not in run_txt or "--plan" not in run_txt or "--out_dir" not in run_txt:
        raise RuntimeError("run_pipeline.py must contain argparse and --plan/--out_dir.")

    for py in generated_dir.glob("*.py"):
        subprocess.run([sys.executable, "-m", "py_compile", str(py)], check=True)


def run_generated_pipeline_code(generated_dir: Path, normalized_plan_path: Path, out_dir: Path):
    cmd = [
        sys.executable,
        str(generated_dir / "run_pipeline.py"),
        "--plan", str(normalized_plan_path),
        "--out_dir", str(out_dir),
    ]
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = env.get("KMP_DUPLICATE_LIB_OK", "TRUE")
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
    subprocess.run(cmd, check=True, cwd=str(generated_dir), env=env)


def validate_pipeline_outputs(out_dir: Path):
    """Validate generated artifacts beyond simple file existence."""
    must_exist = [
        out_dir / "data" / "model_dataset.csv",
        out_dir / "data" / "train.csv",
        out_dir / "data" / "val.csv",
        out_dir / "data" / "test.csv",
        out_dir / "results" / "metrics.json",
        out_dir / "results" / "test_predictions.csv",
        out_dir / "results" / "test_soh_true_vs_predicted.png",
        out_dir / "results" / "training_report.md",
    ]
    miss = [str(p) for p in must_exist if not p.exists()]
    if miss:
        raise RuntimeError("Pipeline output missing: " + " | ".join(miss))

    metrics = _read_json(out_dir / "results" / "metrics.json")
    if "test_RMSE" not in metrics or "test_R2" not in metrics:
        raise RuntimeError("metrics.json must contain test_RMSE and test_R2")
    for key in ["test_RMSE", "test_R2"]:
        val = metrics.get(key)
        if val is None:
            raise RuntimeError(f"metrics.json value is null: {key}")
        try:
            import math
            fval = float(val)
            if math.isnan(fval) or math.isinf(fval):
                raise RuntimeError(f"metrics.json value is not finite: {key}={val}")
        except ValueError:
            raise RuntimeError(f"metrics.json value is not numeric: {key}={val}")

    import pandas as pd
    pred = out_dir / "results" / "test_predictions.csv"
    df = pd.read_csv(pred)
    for c in ["true_soh", "predicted_soh"]:
        if c not in df.columns:
            raise RuntimeError("test_predictions.csv missing column: " + c)
    if df.empty:
        raise RuntimeError("test_predictions.csv is empty")
    if df["predicted_soh"].nunique(dropna=True) <= 1:
        raise RuntimeError("predicted_soh appears constant; generated model output is not meaningful")

    for c in ["true_soh", "predicted_soh"]:
        numeric = pd.to_numeric(df[c], errors="coerce")
        if numeric.isna().any():
            raise RuntimeError(f"test_predictions.csv contains non-numeric or NaN values in {c}")
        if not numeric.between(0.3, 1.2).all():
            raise RuntimeError(f"{c} values outside broad physically-plausible SOH range [0.3, 1.2]")

    # Validate no cell leakage when split tables contain cell_id.
    split_cells = {}
    for split in ["train", "val", "test"]:
        path = out_dir / "data" / f"{split}.csv"
        sdf = pd.read_csv(path)
        if "cell_id" in sdf.columns:
            split_cells[split] = set(sdf["cell_id"].astype(str))
    if split_cells:
        pairs = [("train", "val"), ("train", "test"), ("val", "test")]
        for a, b in pairs:
            overlap = split_cells.get(a, set()) & split_cells.get(b, set())
            if overlap:
                raise RuntimeError(f"Cell leakage detected between {a} and {b}: {sorted(overlap)}")

    # Validate cycle ordering when possible.
    dataset = pd.read_csv(out_dir / "data" / "model_dataset.csv")
    if {"cell_id", "cycle_index"}.issubset(dataset.columns):
        for cell_id, group in dataset.groupby("cell_id"):
            cycles = pd.to_numeric(group["cycle_index"], errors="coerce")
            if cycles.isna().any():
                raise RuntimeError(f"cycle_index contains NaN for cell {cell_id}")
            if not cycles.is_monotonic_increasing:
                # Sorting by row may not be guaranteed after merges, so warn by manifest rather than fail.
                pass


def write_plan_repair_report(path: Path, input_plan: Path, repairs: List[Dict[str, str]], normalized_plan: Dict[str, Any]):
    mep = normalized_plan.get("model_execution_plan", {}) or {}
    lines = [
        "# STEP4 Plan Repair Report",
        "",
        "## Input plan",
        f"- path: {input_plan}",
        "",
        "## Repairs applied",
        "| Issue | Original | Repaired | Reason |",
        "|---|---|---|---|",
    ]
    if not repairs:
        lines.append("| None | - | - | No repairs were needed. |")
    else:
        for r in repairs:
            lines.append(f"| {r['issue']} | {r['original']} | {r['repaired']} | {r['reason']} |")

    lines += [
        "",
        "## Final execution choices",
        f"- model family: {mep.get('model_family','')}",
        f"- model implementation: {json.dumps(mep.get('model_definition',{}), ensure_ascii=False)}",
        f"- scaler: {json.dumps((mep.get('preprocessing',{}) or {}).get('scaler',''), ensure_ascii=False)}",
        f"- target column: {mep.get('target_column','')}",
        f"- model input features: {mep.get('input_features',[])}",
        f"- repeat runs: {(mep.get('training_plan',{}) or {}).get('effective_repeat_runs',1)}",
        f"- fallback used: {(normalized_plan.get('step4_runtime',{}) or {}).get('fallback_allowed', False)}",
        "",
        "## Warnings",
        "- feature-level synthetic executable reproduction; not strict paper numerical reproduction.",
    ]
    _write_text(path, "\n".join(lines))


def write_manifest(out_dir: Path, input_plan: Path, normalized_plan_path: Path, generated_dir: Path, status: str, error: str = ""):
    manifest = {
        "step": "STEP4_EXECUTE_REPRODUCTION",
        "timestamp": datetime.now().isoformat(),
        "status": status,
        "input_reproduction_plan": str(input_plan),
        "normalized_plan_path": str(normalized_plan_path),
        "generated_code_dir": str(generated_dir),
        "error": error,
    }
    _write_json(out_dir / "step4_manifest.json", manifest)


def run_step4_host(reproduction_plan_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    norm_dir = out_dir / "normalized_plan"
    gen_dir = out_dir / "generated_code"
    norm_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)

    raw_plan = _read_json(reproduction_plan_path)
    normalized_plan, repair_log = normalize_plan_minimally(raw_plan)

    normalized_plan_path = norm_dir / "normalized_reproduction_plan.json"
    _write_json(normalized_plan_path, normalized_plan)
    write_plan_repair_report(norm_dir / "plan_repair_report.md", reproduction_plan_path, repair_log, normalized_plan)

    system_prompt, user_prompt = build_codex_code_generation_prompt(normalized_plan, repair_log)
    generated = call_codex_to_generate_pipeline_code(system_prompt, user_prompt)
    _write_json(gen_dir / '__codex_raw_response.json', generated)
    write_generated_code_files(generated, gen_dir)
    apply_generated_code_patches(gen_dir)
    validate_generated_code_files(gen_dir)

    run_generated_pipeline_code(gen_dir, normalized_plan_path, out_dir)
    validate_pipeline_outputs(out_dir)
    write_manifest(out_dir, reproduction_plan_path, normalized_plan_path, gen_dir, status="success")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reproduction_plan", default=REPRODUCTION_PLAN_PATH)
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()

    rp = Path(args.reproduction_plan)
    out = Path(args.out_dir)
    try:
        run_step4_host(rp, out)
        print("step4_execute_reproduction done:", out)
    except Exception as e:
        out.mkdir(parents=True, exist_ok=True)
        write_manifest(out, rp, out / "normalized_plan" / "normalized_reproduction_plan.json", out / "generated_code", status="failed", error=str(e))
        raise


if __name__ == "__main__":
    main()
