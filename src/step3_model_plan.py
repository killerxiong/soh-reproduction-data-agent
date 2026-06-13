import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.codex_client import call_codex_json

STEP2_SPEC_PATH = "outputs/paper1/_work/repro_spec.json"
PAPER_MD_PATH = "outputs/paper1/_work/paper.md"
SUPPLEMENTARY_MD_PATH = "outputs/paper1/_work/Supplementary information.md"
OUT_DIR = "outputs/paper1/_work/step3_plan"
USE_CODEX_OR_LLM = True
SCHEMA_VERSION = "step3_executable_reproduction_plan_v1"

KEYWORDS = [
    "model", "architecture", "network", "pinn", "physics-informed", "linear regression", "loss", "optimizer",
    "learning rate", "batch size", "epoch", "feature", "target", "soh", "capacity", "q_cell_loss", "rmse", "r2",
]


def _read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _read_text(p: Path) -> str:
    if not p or not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="ignore")


def _write_json(p: Path, obj: Dict[str, Any]):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _extract_context(text: str, keywords: List[str], window: int = 1400, max_hits: int = 10) -> str:
    if not text:
        return ""
    low = text.lower()
    spans: List[Tuple[int, int, str]] = []
    for kw in keywords:
        pos = low.find(kw.lower())
        if pos >= 0:
            spans.append((max(0, pos - window), min(len(text), pos + len(kw) + window), kw))
        if len(spans) >= max_hits:
            break
    if not spans:
        return text[:12000]
    chunks = []
    for i, (s, e, k) in enumerate(spans):
        chunks.append(f"[chunk_{i+1}|{k}]\n{text[s:e]}")
    return "\n\n".join(chunks)


def _default_cell_ids() -> List[str]:
    return [f"cell_{i:03d}" for i in range(1, 11)]


def read_inputs(step2_spec: Path, paper_md: Path, supp_md: Path) -> Dict[str, Any]:
    return {
        "step2_json": _read_json(step2_spec),
        "paper_text": _read_text(paper_md),
        "supp_text": _read_text(supp_md),
    }


def _template(source_files: Dict[str, str]) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "paper_id": "",
        "paper_name": "",
        "source_files": source_files,
        "reproduction_mode": {
            "strict_paper_reproduction_possible": False,
            "selected_mode": "feature_level_synthetic_executable_reproduction",
            "reason": "",
            "what_is_reproduced": [
                "paper-reported model input-output structure",
                "paper-reported engineered feature space",
                "paper-reported primary model family",
                "complete train/validation/test execution pipeline",
            ],
            "what_is_not_reproduced": [
                "paper original raw experimental dataset",
                "paper exact numerical results",
                "raw voltage/current/time feature extraction unless explicitly implemented",
            ],
        },
        "paper_understanding": {
            "task": {"task_type": "regression", "target": "", "target_formula": "", "sample_granularity": ""},
            "model": {"primary_model_name": "", "model_family": "", "baseline_models": [], "is_primary_model_clear": False},
            "paper_reported_feature_sets": [],
            "paper_reported_training_protocol": "",
            "paper_reported_metrics": [],
            "important_notes": [],
        },
        "selected_reproduction_strategy": {
            "strategy_name": "primary_feature_strategy",
            "selection_rule": "Use primary_features as the default model input features.",
            "primary_features": [],
            "secondary_features": [],
            "model_input_features": [],
            "features_to_generate_but_not_use_by_default": [],
            "target_column": "soh",
            "also_generate_label_columns": ["capacity_loss", "Q_cell_loss"],
            "reason": "",
        },
        "dataset_construction_plan": {
            "dataset_mode": "feature_level_synthetic_dataset",
            "num_cells": 10,
            "cell_ids": _default_cell_ids(),
            "cell_split": {
                "split_unit": "cell",
                "train_ratio": 0.6,
                "val_ratio": 0.2,
                "test_ratio": 0.2,
                "train_cells_count": 6,
                "val_cells_count": 2,
                "test_cells_count": 2,
                "no_cell_overlap": True,
            },
            "cell_variability": {
                "chemistry_assignment": {
                    "default": "mixed_NMC_LFP",
                    "num_nmc": 5,
                    "num_lfp": 5,
                    "note": "In feature-level synthetic mode, chemistry controls metadata/statistical offsets rather than raw electrochemical simulation.",
                },
                "per_cell_random_effects": [
                    "initial_capacity_offset",
                    "degradation_rate_offset",
                    "knee_cycle_offset",
                    "feature_baseline_offset",
                    "feature_noise_level",
                ],
                "require_different_soh_trajectories": True,
                "require_different_feature_distributions": True,
            },
            "cycle_soh_generation": {
                "min_cycles_per_cell": 180,
                "max_cycles_per_cell": 300,
                "initial_soh_range": [0.98, 1.02],
                "end_soh_range": [0.75, 0.90],
                "trajectory_shape": "nonlinear_degradation_with_optional_knee",
                "allow_small_regeneration_noise": True,
                "formula_template": "soh = initial_soh - a * normalized_cycle - b * max(0, normalized_cycle - knee)^2 + noise",
                "constraints": [
                    "SOH should generally decrease with cycle_index",
                    "SOH must be clipped to a physically plausible range",
                    "different cells must not have identical degradation trajectories",
                ],
            },
            "feature_generation": {
                "features_to_generate": [],
                "model_input_features": [],
                "secondary_features_not_used_by_default": [],
                "generation_method": "feature_target_correlated_synthetic_generation",
                "latent_variables": ["normalized_cycle", "soh", "capacity_loss", "cell_random_effect"],
                "feature_value_formula_template": "feature = cell_offset + trend_strength * degradation_latent + nonlinear_component + noise",
                "feature_correlation_policy": {
                    "primary_features": "medium_to_high_correlation_with_target",
                    "secondary_features": "weak_to_medium_correlation_or_optional",
                    "avoid_perfect_correlation": True,
                },
                "feature_report_required": True,
            },
            "label_generation": {
                "target_column": "soh",
                "required_label_columns": ["soh", "capacity_loss"],
                "optional_label_columns": ["Q_cell_loss"],
                "formulas": {"capacity_loss": "1 - soh", "Q_cell_loss": "capacity_loss", "soh": "1 - capacity_loss"},
            },
            "output_tables": {
                "data_dir": "outputs/step4_dataset/<paper_id>/data",
                "reports_dir": "outputs/step4_dataset/<paper_id>/reports",
                "required_files": [
                    "cell_metadata.csv",
                    "cycle_soh_trajectories.csv",
                    "features.csv",
                    "labels.csv",
                    "model_dataset.csv",
                    "train.csv",
                    "val.csv",
                    "test.csv",
                ],
                "reports": ["feature_generation_report.json", "dataset_generation_report.md"],
            },
        },
        "model_execution_plan": {
            "model_family": "",
            "model_name": "",
            "framework": "",
            "input_features": [],
            "target_column": "soh",
            "train_val_test_data": {"train_file": "train.csv", "val_file": "val.csv", "test_file": "test.csv"},
            "preprocessing": {
                "feature_columns": [],
                "target_column": "soh",
                "scaler": "StandardScaler",
                "fit_scaler_on": "train_only",
                "transform": ["train", "val", "test"],
                "handle_missing_values": "raise_error_unless_user_allows_imputation",
            },
            "model_definition": {},
            "loss_plan": {},
            "training_plan": {},
            "validation_plan": {},
            "testing_plan": {},
        },
        "evaluation_plan": {
            "metrics": [
                {"name": "RMSE", "split": "test", "formula": "sqrt(mean((y_true - y_pred)^2))", "output_key": "test_RMSE"},
                {"name": "R2", "split": "test", "formula": "1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)", "output_key": "test_R2"},
            ],
            "soh_conversion_rule": {
                "if_target_is_soh": "use prediction directly",
                "if_target_is_capacity_loss": "predicted_soh = 1 - predicted_capacity_loss",
                "if_target_is_Q_cell_loss": "predicted_soh = 1 - predicted_Q_cell_loss",
            },
            "plots": [
                {
                    "name": "test_soh_true_vs_predicted",
                    "type": "scatter_or_line",
                    "x": "cycle_index or sample_index",
                    "y_true": "true_soh",
                    "y_pred": "predicted_soh",
                    "split": "test",
                    "output_file": "test_soh_true_vs_predicted.png",
                }
            ],
            "prediction_outputs": ["test_predictions.csv"],
        },
        "outputs_plan": {
            "dataset_outputs": [
                "cell_metadata.csv",
                "cycle_soh_trajectories.csv",
                "features.csv",
                "labels.csv",
                "model_dataset.csv",
                "train.csv",
                "val.csv",
                "test.csv",
            ],
            "model_outputs": [
                "trained_model.pkl or trained_model.pth",
                "metrics.json",
                "test_predictions.csv",
                "test_soh_true_vs_predicted.png",
                "training_report.md",
            ],
        },
        "missing_details_and_filled_assumptions": [],
        "codex_decisions": [],
        "validation_rules_for_step4": [
            "Dataset must contain exactly 10 unique cells.",
            "Train/val/test split must be cell-level with no cell overlap.",
            "Train/val/test cell counts must be 6/2/2.",
            "All model_input_features must exist in train.csv, val.csv, and test.csv.",
            "Target column must exist.",
            "If target is capacity_loss or Q_cell_loss, SOH must be computable as 1 - target.",
            "Test predictions must include true_soh and predicted_soh.",
            "metrics.json must include test_RMSE and test_R2.",
            "test_soh_true_vs_predicted.png must be generated.",
        ],
        "evidence": [],
    }


def build_prompt(step2_json: Dict[str, Any], paper_ctx: str, supp_ctx: str, source_files: Dict[str, str]) -> Tuple[str, str]:
    system = (
        "You are a rigorous research-paper reproduction planner. "
        "Output ONLY JSON. Build a full executable reproduction_plan. "
        "Do not fabricate paper-reported values; put filled items into missing_details_and_filled_assumptions."
    )
    tpl = _template(source_files)
    user = (
        "Generate reproduction_plan JSON.\n"
        "Hard constraints:\n"
        "1) model_input_features must equal primary_features by default.\n"
        "2) secondary features are generated but not used by default.\n"
        "3) keep 10 cells, split 6/2/2 by cell.\n"
        "4) include executable model_definition/loss_plan/training_plan/validation_plan/testing_plan.\n"
        "5) evaluation must include test RMSE, test R2, and test_soh_true_vs_predicted plot.\n"
        "6) if details missing, fill executable assumptions and mark paper_reported=false.\n"
        "7) if assumptions exist, strict_paper_reproduction_possible must be false.\n"
        "Return JSON only.\n\n"
        f"Template:\n{json.dumps(tpl, ensure_ascii=False, indent=2)}\n\n"
        f"STEP2 repro_spec:\n{json.dumps(step2_json, ensure_ascii=False)}\n\n"
        f"paper context:\n{paper_ctx}\n\n"
        f"supplementary context:\n{supp_ctx}\n"
    )
    return system, user


def call_codex_or_llm(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    if not USE_CODEX_OR_LLM:
        raise RuntimeError("STEP3 requires Codex/LLM; USE_CODEX_OR_LLM=False")
    return call_codex_json(system_prompt, user_prompt)


def _ensure(plan: Dict[str, Any], path: List[str], default: Any):
    cur = plan
    for k in path[:-1]:
        if not isinstance(cur.get(k), dict):
            cur[k] = {}
        cur = cur[k]
    if path[-1] not in cur:
        cur[path[-1]] = default


def _append_assumption(plan: Dict[str, Any], field: str, value: Any, why: str, risk: str):
    arr = plan.setdefault("missing_details_and_filled_assumptions", [])
    if not any(isinstance(x, dict) and x.get("field") == field for x in arr):
        arr.append({
            "field": field,
            "paper_reported": False,
            "filled_value": value,
            "why_needed": why,
            "why_reasonable": "Standard executable default for reproduction pipeline.",
            "risk": risk,
            "used_in_execution": True,
        })


def validate_and_patch(plan: Dict[str, Any], source_files: Dict[str, str]) -> Dict[str, Any]:
    if not isinstance(plan, dict):
        raise RuntimeError("LLM output is not JSON object")

    tpl = _template(source_files)
    for k, v in tpl.items():
        if k not in plan:
            plan[k] = v
    plan["schema_version"] = SCHEMA_VERSION
    plan["source_files"] = source_files

    s = plan.get("selected_reproduction_strategy") or {}
    primary = s.get("primary_features") or []
    secondary = s.get("secondary_features") or []
    if not s.get("model_input_features"):
        s["model_input_features"] = list(primary)
    else:
        s["model_input_features"] = list(primary) if primary else s["model_input_features"]
    s["features_to_generate_but_not_use_by_default"] = list(secondary)
    s.setdefault("also_generate_label_columns", ["capacity_loss", "Q_cell_loss"])
    plan["selected_reproduction_strategy"] = s

    ds = plan.get("dataset_construction_plan") or {}
    ds["num_cells"] = 10
    ds["cell_ids"] = _default_cell_ids()
    ds["cell_split"] = {
        "split_unit": "cell",
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "train_cells_count": 6,
        "val_cells_count": 2,
        "test_cells_count": 2,
        "no_cell_overlap": True,
    }
    fg = ds.get("feature_generation") or {}
    fg["features_to_generate"] = list(primary) + [x for x in secondary if x not in primary]
    fg["model_input_features"] = list(primary)
    fg["secondary_features_not_used_by_default"] = list(secondary)
    fg.setdefault("generation_method", "feature_target_correlated_synthetic_generation")
    ds["feature_generation"] = fg

    lg = ds.get("label_generation") or {}
    target = s.get("target_column") or lg.get("target_column") or "soh"
    lg["target_column"] = target
    lg["required_label_columns"] = ["soh", "capacity_loss"]
    lg["optional_label_columns"] = ["Q_cell_loss"]
    lg["formulas"] = {"capacity_loss": "1 - soh", "Q_cell_loss": "capacity_loss", "soh": "1 - capacity_loss"}
    ds["label_generation"] = lg
    plan["dataset_construction_plan"] = ds

    mep = plan.get("model_execution_plan") or {}
    mep["input_features"] = list(primary)
    mep["target_column"] = target
    mep["train_val_test_data"] = {"train_file": "train.csv", "val_file": "val.csv", "test_file": "test.csv"}
    mep["preprocessing"] = {
        "feature_columns": list(primary),
        "target_column": target,
        "scaler": "StandardScaler",
        "fit_scaler_on": "train_only",
        "transform": ["train", "val", "test"],
        "handle_missing_values": "raise_error_unless_user_allows_imputation",
    }

    mf = str(mep.get("model_family") or (plan.get("paper_understanding", {}).get("model", {}).get("model_family", ""))).lower()
    if "linear" in mf:
        mep.setdefault("framework", "sklearn")
        mep["model_definition"] = {
            "type": "linear_regression",
            "implementation": "sklearn.linear_model.LinearRegression",
            "fit_intercept": True,
            "regularization": "none",
            "assumption_source": "codex_filled_if_not_paper_reported",
            "reason": "Standard ordinary least squares executable default.",
        }
        mep["loss_plan"] = {
            "type": "ordinary_least_squares",
            "implementation": "sklearn default",
            "primary_training_objective": "minimize squared error",
            "source": "standard implementation assumption",
        }
        mep["training_plan"] = {
            "training_mode": "fit_on_train_validate_on_val_test_on_test",
            "hyperparameters": {"fit_intercept": True},
            "random_seed": 42,
        }
    else:
        mep.setdefault("framework", "pytorch")
        if not mep.get("model_definition"):
            mep["model_definition"] = {
                "type": "pinn",
                "implementation": "pytorch",
                "strict_implementation_possible": False,
                "approximate_implementation_required": True,
                "components": {
                    "F_solution_network": {"input_dim": "len(input_features)", "output_dim": 1, "hidden_layers": [64, 64, 32], "activation": "tanh", "output_activation": "none", "role": "map features + cycle_index to SOH"},
                    "G_dynamics_network": {"input_dim": "len(input_features) + 1", "output_dim": 1, "hidden_layers": [64, 32], "activation": "tanh", "output_activation": "none", "role": "approximate degradation rate / dSOH_dcycle"},
                },
                "assumption_source": "codex_filled_executable_assumption",
                "reason": "Paper reports PINN but exact architecture details are incomplete.",
            }
            _append_assumption(plan, "pinn_hidden_layers", {"F_solution_network": [64, 64, 32], "G_dynamics_network": [64, 32]}, "STEP4 must instantiate executable PINN.", "Numerical results may differ from paper.")
        if not mep.get("loss_plan"):
            mep["loss_plan"] = {
                "total_loss": "L_data + alpha_mono * L_mono + alpha_pde * L_pde",
                "loss_terms": {
                    "L_data": {"formula": "MSE(soh_pred, soh_true)", "weight": 1.0, "source": "codex_filled_executable_assumption"},
                    "L_mono": {"formula": "mean(ReLU(soh_pred_next - soh_pred_current)^2) grouped by cell_id sorted by cycle_index", "weight": 0.1, "source": "codex_filled_executable_assumption"},
                    "L_pde": {"formula": "MSE(dF_dcycle - G(features, soh_pred), 0)", "weight": 0.01, "source": "codex_filled_executable_assumption"},
                },
                "notes": ["Exact PDE residual may be missing; this is executable approximation."],
            }
            _append_assumption(plan, "pinn_loss_weights", {"alpha_mono": 0.1, "alpha_pde": 0.01}, "Need executable total loss for training.", "Different weights may change performance.")
        if not mep.get("training_plan"):
            mep["training_plan"] = {
                "training_mode": "fit_on_train_validate_on_val_test_on_test",
                "optimizer": "Adam",
                "learning_rate": 0.001,
                "batch_size": 256,
                "epochs": 200,
                "early_stopping": {"enabled": True, "monitor": "val_RMSE", "patience": 30, "mode": "min"},
                "random_seed": 42,
            }
            _append_assumption(plan, "training_hyperparameters", {"optimizer": "Adam", "learning_rate": 0.001, "batch_size": 256, "epochs": 200}, "Need executable training loop params.", "Paper likely used different hyperparameters.")

    mep.setdefault("validation_plan", {"split": "val.csv", "metrics": ["RMSE", "R2"], "model_selection_rule": "best_val_RMSE"})
    mep.setdefault("testing_plan", {"split": "test.csv", "metrics": ["RMSE", "R2"], "save_predictions": "test_predictions.csv"})
    plan["model_execution_plan"] = mep

    plan["evaluation_plan"] = tpl["evaluation_plan"]
    plan["outputs_plan"] = tpl["outputs_plan"]
    plan["validation_rules_for_step4"] = tpl["validation_rules_for_step4"]

    rm = plan.get("reproduction_mode") or {}
    rm["selected_mode"] = "feature_level_synthetic_executable_reproduction"
    if plan.get("missing_details_and_filled_assumptions"):
        rm["strict_paper_reproduction_possible"] = False
    rm.setdefault("reason", "Original dataset/details incomplete; executable Codex-filled assumptions used.")
    plan["reproduction_mode"] = rm

    if not plan.get("codex_decisions"):
        plan["codex_decisions"] = [
            {"decision": "Use primary_features as model input features.", "reason": "User policy for default pipeline."},
            {"decision": "Generate secondary_features but not use them by default.", "reason": "Secondary features are optional experiments."},
            {"decision": "Use 10 synthetic cells with 6/2/2 cell split.", "reason": "User fixed project-level dataset policy."},
            {"decision": "Evaluate with RMSE, R2 and SOH true-vs-pred plot on test set.", "reason": "User fixed evaluation outputs."},
        ]

    return plan


def write_outputs(out_dir: Path, plan: Dict[str, Any]):
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "reproduction_plan.json", plan)

    rs = plan.get("reproduction_mode") or {}
    st = plan.get("selected_reproduction_strategy") or {}
    ds = plan.get("dataset_construction_plan") or {}
    mep = plan.get("model_execution_plan") or {}
    ep = plan.get("evaluation_plan") or {}

    rows = []
    for a in (plan.get("missing_details_and_filled_assumptions") or []):
        if isinstance(a, dict):
            rows.append(f"| {a.get('field','')} | {json.dumps(a.get('filled_value',''),ensure_ascii=False)} | {a.get('why_needed','')} | {a.get('risk','')} |")
    if not rows:
        rows = ["| (none) |  |  |  |"]

    report = "\n".join([
        "# STEP3 Executable Reproduction Plan Report",
        "",
        "## Paper",
        f"- paper_id: {plan.get('paper_id','')}",
        f"- paper_name: {plan.get('paper_name','')}",
        "",
        "## Selected reproduction mode",
        f"- strict paper reproduction possible: {rs.get('strict_paper_reproduction_possible')}",
        f"- selected mode: {rs.get('selected_mode','')}",
        f"- what is reproduced: {rs.get('what_is_reproduced',[])}",
        f"- what is not reproduced: {rs.get('what_is_not_reproduced',[])}",
        "",
        "## Selected strategy",
        f"- primary features: {st.get('primary_features',[])}",
        f"- secondary features: {st.get('secondary_features',[])}",
        f"- model input features: {st.get('model_input_features',[])}",
        f"- target: {st.get('target_column','')}",
        f"- reason: {st.get('reason','')}",
        "",
        "## Dataset construction plan",
        f"- num cells: {ds.get('num_cells')}",
        f"- split rule: {json.dumps(ds.get('cell_split',{}), ensure_ascii=False)}",
        f"- cycle-SOH generation: {json.dumps(ds.get('cycle_soh_generation',{}), ensure_ascii=False)}",
        f"- feature generation method: {(ds.get('feature_generation') or {}).get('generation_method','')}",
        f"- labels: {json.dumps(ds.get('label_generation',{}), ensure_ascii=False)}",
        f"- output files: {(ds.get('output_tables') or {}).get('required_files',[])}",
        "",
        "## Model execution plan",
        f"- model family: {mep.get('model_family','')}",
        f"- model implementation: {json.dumps(mep.get('model_definition',{}), ensure_ascii=False)}",
        f"- input features: {mep.get('input_features',[])}",
        f"- target: {mep.get('target_column','')}",
        f"- preprocessing: {json.dumps(mep.get('preprocessing',{}), ensure_ascii=False)}",
        f"- loss: {json.dumps(mep.get('loss_plan',{}), ensure_ascii=False)}",
        f"- training plan: {json.dumps(mep.get('training_plan',{}), ensure_ascii=False)}",
        f"- validation plan: {json.dumps(mep.get('validation_plan',{}), ensure_ascii=False)}",
        f"- testing plan: {json.dumps(mep.get('testing_plan',{}), ensure_ascii=False)}",
        "",
        "## Evaluation plan",
        "- test RMSE",
        "- test R2",
        "- true-vs-predicted SOH plot",
        f"- details: {json.dumps(ep, ensure_ascii=False)}",
        "",
        "## Filled assumptions",
        "| Field | Filled value | Why needed | Risk |",
        "|---|---|---|---|",
        *rows,
        "",
        "## Evidence from paper/spec",
        f"- evidence count: {len(plan.get('evidence',[]))}",
    ])
    (out_dir / "reproduction_plan_report.md").write_text(report, encoding="utf-8")

    _write_json(out_dir / "step3_manifest.json", {
        "step": "STEP3_REPRODUCTION_PLAN",
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "outputs": {
            "reproduction_plan_json": str(out_dir / "reproduction_plan.json"),
            "reproduction_plan_report_md": str(out_dir / "reproduction_plan_report.md"),
            "step3_manifest_json": str(out_dir / "step3_manifest.json"),
        },
    })


def build_reproduction_plan(step2_spec: Path, paper_md: Path, supp_md: Path, out_dir: Path):
    inp = read_inputs(step2_spec, paper_md, supp_md)
    step2_json = inp["step2_json"]
    paper_ctx = _extract_context(inp["paper_text"], KEYWORDS)
    supp_ctx = _extract_context(inp["supp_text"], KEYWORDS)
    source_files = {
        "step2_spec_path": str(step2_spec),
        "paper_md_path": str(paper_md),
        "supplementary_md_path": str(supp_md),
    }

    system_prompt, user_prompt = build_prompt(step2_json, paper_ctx, supp_ctx, source_files)
    raw = call_codex_or_llm(system_prompt, user_prompt)
    plan = validate_and_patch(raw, source_files)
    write_outputs(out_dir, plan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step2_spec", default=STEP2_SPEC_PATH)
    ap.add_argument("--paper_md", default=PAPER_MD_PATH)
    ap.add_argument("--supp_md", default=SUPPLEMENTARY_MD_PATH)
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()

    build_reproduction_plan(Path(args.step2_spec), Path(args.paper_md), Path(args.supp_md), Path(args.out_dir))
    print("step3_reproduction_plan done:", args.out_dir)


if __name__ == "__main__":
    main()
