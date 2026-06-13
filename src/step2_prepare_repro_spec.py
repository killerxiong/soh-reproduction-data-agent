import json
from pathlib import Path
from typing import Dict, Any, List

DEFAULTS = {
    "split": {
        "mode": "group_by_battery_if_possible_else_random",
        "train_val_test": [0.6, 0.2, 0.2],
        "seed": 42,
    },
    "framework_defaults_adam": {
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_eps": 1e-8,
    },
    "runnable_defaults": {
        "learning_rate": 1e-3,
        "epochs": 200,
        "batch_size": 256,
        "weight_decay": 0.0,
        "seed": 42,
    },
}

NN_FAMILIES = {"neural_network", "pinn", "transformer"}


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_missing(v):
    return v is None or v == "" or v == [] or v == {}


def _ensure_list(v):
    return v if isinstance(v, list) else ([] if v is None else [v])


def _first_nonempty(*vals):
    for v in vals:
        if not _is_missing(v):
            return v
    return None


def _normalize_step1(step1: Dict[str, Any]) -> Dict[str, Any]:
    s = dict(step1)

    # data_requirements???????
    dr = s.get("data_requirements") or {}
    if isinstance(dr, dict) and "data_requirements" in dr and isinstance(dr["data_requirements"], dict):
        dr = dr["data_requirements"]
    s["data_requirements"] = dr if isinstance(dr, dict) else {}

    # target_source_signal????????????
    td = s.get("target_definition") or {}
    tss = str(td.get("target_source_signal", "") or "")
    if len(tss) > 120:
        low = tss.lower()
        if "capacity" in low:
            td["target_source_signal"] = "capacity"
        elif "soh" in low:
            td["target_source_signal"] = "soh"
    s["target_definition"] = td

    # feature_recipe??
    fr = _ensure_list(s.get("feature_recipe"))
    norm_fr = []
    for i, f in enumerate(fr):
        if not isinstance(f, dict):
            continue
        x = dict(f)
        x.setdefault("feature_id", f"f{i+1:03d}")
        x.setdefault("name", f"feature_{i+1}")
        if not isinstance(x.get("source_signals"), list):
            x["source_signals"] = [] if _is_missing(x.get("source_signals")) else [str(x.get("source_signals"))]
        x.setdefault("operation", "unknown")
        x.setdefault("group_level", x.get("aggregation_level", "unknown"))
        x.setdefault("status", x.get("paper_status", "missing"))
        x.setdefault("paper_status", x.get("status", "missing"))
        if not isinstance(x.get("construction_steps"), list):
            cs = x.get("construction_steps")
            x["construction_steps"] = [str(cs)] if cs else []
        if not isinstance(x.get("unresolved_details"), list):
            ud = x.get("unresolved_details")
            x["unresolved_details"] = [str(ud)] if ud else []
        if not isinstance(x.get("missing_details"), list):
            md = x.get("missing_details")
            x["missing_details"] = [str(md)] if md else []
        norm_fr.append(x)
    s["feature_recipe"] = norm_fr
    if not isinstance(s.get("feature_formulas"), list):
        s["feature_formulas"] = []
    if not isinstance(s.get("model_equations"), list):
        s["model_equations"] = []
    if not isinstance(s.get("model_architecture"), dict):
        s["model_architecture"] = {}
    if not isinstance(s.get("training_hyperparameters"), dict):
        s["training_hyperparameters"] = {}
    if not isinstance(s.get("cross_references"), list):
        s["cross_references"] = []
    return s


def _contains_adam(step1: Dict[str, Any]) -> bool:
    md = step1.get("model_definition") or {}
    txt = f"{md.get('optimizer_or_solver','')} {md.get('training_paradigm','')} {md.get('architecture','')}".lower()
    return "adam" in txt


def _normalize_hparams(step1: Dict[str, Any]) -> Dict[str, Any]:
    hp = step1.get("training_and_hyperparameters") or {}
    paper_reported = hp.get("paper_reported") or {}
    code_reported = hp.get("code_reported") or {}
    inferred = hp.get("codex_inferred_with_evidence") or {}
    missing = _ensure_list(hp.get("missing_in_paper"))

    framework_defaults = DEFAULTS["framework_defaults_adam"].copy() if _contains_adam(step1) else {}

    model_family = str((step1.get("model_definition") or {}).get("model_family", "unknown")).lower()
    runnable_defaults = dict(DEFAULTS["runnable_defaults"]) if model_family in NN_FAMILIES else {"seed": DEFAULTS["runnable_defaults"]["seed"]}

    unresolved = [str(x) for x in missing if str(x).strip()]

    return {
        "paper_reported": paper_reported,
        "code_reported": code_reported,
        "codex_inferred_with_evidence": inferred,
        "framework_defaults": framework_defaults,
        "runnable_defaults": runnable_defaults,
        "unresolved_for_strict_reproduction": unresolved,
        "warnings": [],
    }


def _build_feature_build_plan(feature_recipe: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    plans = []
    for f in feature_recipe:
        op = str(f.get("operation", "unknown")).lower()
        plan = {
            "feature_id": f.get("feature_id"),
            "name": f.get("name"),
            "operation": op,
            "source_signals": f.get("source_signals", []),
            "group_level": f.get("group_level", "unknown"),
            "window_or_segment": f.get("window_or_segment", ""),
            "status": f.get("status", "missing"),
            "buildability": "not_buildable" if f.get("status") == "missing" else ("approx" if f.get("status") in {"partially_reported", "ambiguous"} or (f.get("missing_details") or f.get("unresolved_details")) else "strict"),
            "notes": [],
        }
        if op in {"unknown", "model_specific"}:
            plan["notes"].append("custom operator required")
        if not f.get("source_signals"):
            plan["notes"].append("source_signals missing")
        plans.append(plan)
    return plans


def _build_assumption_plan(step1: Dict[str, Any], hp: Dict[str, Any]) -> List[Dict[str, Any]]:
    assumptions = []
    for f in step1.get("feature_recipe", []) or []:
        for d in (f.get("missing_details") or []) + (f.get("unresolved_details") or []):
            assumptions.append({
                "scope": "feature",
                "feature_id": f.get("feature_id"),
                "feature_name": f.get("name"),
                "assumption_item": str(d),
                "impact": "strict_reproduction_blocker",
            })
    for m in hp.get("unresolved_for_strict_reproduction", []):
        assumptions.append({
            "scope": "training",
            "assumption_item": str(m),
            "impact": "strict_reproduction_blocker",
        })
    return assumptions


def _build_model_plan(step1: Dict[str, Any]) -> Dict[str, Any]:
    md = step1.get("model_definition") or {}
    fam = str(md.get("model_family", "unknown")).lower()
    primary = {
        "model_name": md.get("model_name", ""),
        "model_family": fam,
        "training_paradigm": md.get("training_paradigm", "unknown"),
        "optimizer_or_solver": md.get("optimizer_or_solver", ""),
    }
    baselines = []
    for x in ["MLP", "CNN", "Linear Regression", "RF", "XGBoost"]:
        txt = json.dumps(step1.get("experiment_protocol", {}), ensure_ascii=False) + json.dumps(step1.get("output_metrics", {}), ensure_ascii=False)
        if x.lower() in txt.lower():
            baselines.append(x)
    return {
        "primary_model": primary,
        "baseline_models": sorted(set(baselines)),
        "benchmark_suite": {
            "paper_reported_comparisons": sorted(set(baselines)),
            "notes": "from paper spec text; no new assumptions",
        },
    }


def _build_protocol_blocks(step1: Dict[str, Any]) -> Dict[str, Any]:
    ep = step1.get("experiment_protocol") or {}
    paper_protocol = dict(ep)
    user_template = {
        "dataset_source": "user_provided",
        "is_original_paper_dataset": False,
        "split_mode": "user_fixed",
        "train_val_test": DEFAULTS["split"]["train_val_test"],
        "seed": DEFAULTS["split"]["seed"],
        "notes": "Used when user dataset differs from original paper datasets.",
    }
    deviation = {
        "allowed": True,
        "rules": [
            "If user dataset != original paper dataset, mark protocol deviation.",
            "Do not claim original-paper numerical reproduction under user_fixed split.",
        ],
    }
    return {
        "paper_experiment_protocol": paper_protocol,
        "user_experiment_protocol_template": user_template,
        "protocol_deviation_policy": deviation,
    }


def _build_reproduction_feasibility(step1: Dict[str, Any], hp: Dict[str, Any], assumption_plan: List[Dict[str, Any]]) -> Dict[str, Any]:
    strict_blockers = []
    for f in step1.get("feature_recipe", []) or []:
        if f.get("status") in {"missing", "partially_reported", "ambiguous"} or (f.get("missing_details") or f.get("unresolved_details")):
            strict_blockers.append(f"feature:{f.get('name')}:insufficient_details")
    strict_blockers += [f"hyperparam:{x}" for x in hp.get("unresolved_for_strict_reproduction", [])]

    strict_ok = len(strict_blockers) == 0
    approx_ok = True
    rec_mode = "strict_reproduction" if strict_ok else "approximate_reimplementation"

    return {
        "strict_reproduction": {
            "boolean": strict_ok,
            "reason": "ok" if strict_ok else "paper details incomplete",
            "blockers": strict_blockers,
        },
        "approximate_reimplementation": {
            "boolean": approx_ok,
            "reason": "run with explicit assumptions" if not strict_ok else "also feasible",
            "required_assumptions": [a.get("assumption_item") for a in assumption_plan],
        },
        "method_adaptation_on_user_dataset": {
            "boolean": True,
            "reason": "Allowed when user provides dataset and fixed split; results are not original-paper numerical reproduction.",
        },
        "recommended_mode": rec_mode,
    }


def _build_readiness(step1: Dict[str, Any], hp: Dict[str, Any], feasibility: Dict[str, Any]) -> Dict[str, Any]:
    strict_blockers = feasibility["strict_reproduction"]["blockers"]
    feat_strict = {
        "boolean": all(not b.startswith("feature:") for b in strict_blockers),
        "reason": "ok" if all(not b.startswith("feature:") for b in strict_blockers) else "feature_spec_not_strictly_complete",
        "blockers": [b for b in strict_blockers if b.startswith("feature:")],
    }
    feat_approx = {
        "boolean": True,
        "reason": "ok",
        "blockers": [],
        "warnings": [b for b in strict_blockers if b.startswith("feature:")],
    }
    model_ready = {
        "boolean": len([b for b in strict_blockers if b.startswith("hyperparam:")]) == 0,
        "reason": "ok" if len([b for b in strict_blockers if b.startswith("hyperparam:")]) == 0 else "model_or_training_spec_incomplete",
        "blockers": [b for b in strict_blockers if b.startswith("hyperparam:")],
    }
    return {
        "feature_construction_ready_strict": feat_strict,
        "feature_construction_ready_approx": feat_approx,
        "model_training_ready": model_ready,
        "strict_reproduction_ready": {
            "boolean": feasibility["strict_reproduction"]["boolean"],
            "reason": feasibility["strict_reproduction"]["reason"],
            "blockers": strict_blockers,
        },
    }


def _build_repro_spec(step1: Dict[str, Any]) -> Dict[str, Any]:
    step1 = _normalize_step1(step1)
    hp = _normalize_hparams(step1)
    feature_build_plan = _build_feature_build_plan(step1.get("feature_recipe") or [])
    assumption_plan = _build_assumption_plan(step1, hp)
    model_plan = _build_model_plan(step1)
    proto = _build_protocol_blocks(step1)
    feasibility = _build_reproduction_feasibility(step1, hp, assumption_plan)
    readiness = _build_readiness(step1, hp, feasibility)

    warnings = []
    if not feasibility["strict_reproduction"]["boolean"]:
        warnings.append("strict reproduction blocked; use approximate or adaptation mode")

    td = step1.get("target_definition") or {}
    fr = step1.get("feature_recipe") or []
    md = step1.get("model_definition") or {}
    loss_terms_raw = md.get("loss_terms", []) or []
    loss_terms_names = []
    for lt in loss_terms_raw:
        if isinstance(lt, str):
            if lt.strip():
                loss_terms_names.append(lt.strip())
        elif isinstance(lt, dict):
            name = str(lt.get("name") or lt.get("term") or "").strip()
            if name:
                loss_terms_names.append(name)
    ep = step1.get("experiment_protocol") or {}
    om = step1.get("output_metrics") or {}
    impl = {
        "target": {
            "name": td.get("target_name", ""),
            "formula": td.get("target_formula", ""),
            "granularity": td.get("label_granularity", ""),
        },
        "features": {
            "input_feature_count": len(fr),
            "feature_names": [f.get("name") for f in fr if isinstance(f, dict)],
            "source_signals": sorted(set([s for f in fr if isinstance(f, dict) for s in (f.get("source_signals") or [])])),
            "segments": [f.get("window_or_segment", "") for f in fr if isinstance(f, dict) and f.get("window_or_segment")],
            "formulas": [x.get("formula") for x in (step1.get("feature_formulas") or []) if isinstance(x, dict) and x.get("formula")],
            "normalization": "; ".join(sorted(set([str(f.get("normalization_or_scaling", "")) for f in fr if isinstance(f, dict) and f.get("normalization_or_scaling")]))),
        },
        "model": {
            "name": md.get("model_name", ""),
            "family": md.get("model_family", ""),
            "inputs": md.get("inputs", []),
            "outputs": md.get("outputs", []),
            "architecture": md.get("architecture", ""),
            "loss": ", ".join(loss_terms_names),
            "optimizer": md.get("optimizer_or_solver", "") or (step1.get("training_hyperparameters", {}) or {}).get("optimizer", ""),
            "known_missing": (step1.get("training_hyperparameters", {}) or {}).get("missing", []),
        },
        "experiment": {
            "split_unit": ep.get("split_unit", ""),
            "split_ratio": ep.get("train_val_test_split", "") or ep.get("split_ratio", ""),
            "metrics": om.get("metrics", []),
            "baselines": ep.get("baselines", []),
        },
        "reproduction_mode": feasibility.get("recommended_mode", "approximate_reimplementation").replace("approximate_reimplementation", "approximate").replace("strict_reproduction", "strict"),
    }

    return {
        "task": (step1.get("paper_identity") or {}).get("task_type", ""),
        "target_definition": step1.get("target_definition") or {},
        "feature_recipe": step1.get("feature_recipe") or [],
        "feature_build_plan": feature_build_plan,
        "data_requirements": step1.get("data_requirements") or {},
        "model_definition": step1.get("model_definition") or {},
        "model_plan": model_plan,
        "training_and_hyperparameters": hp,
        "experiment_protocol": step1.get("experiment_protocol") or {},
        "paper_experiment_protocol": proto["paper_experiment_protocol"],
        "user_experiment_protocol_template": proto["user_experiment_protocol_template"],
        "protocol_deviation_policy": proto["protocol_deviation_policy"],
        "output_metrics": step1.get("output_metrics") or {},
        "feature_formulas": step1.get("feature_formulas") or [],
        "model_equations": step1.get("model_equations") or [],
        "model_architecture": step1.get("model_architecture") or {},
        "training_hyperparameters": step1.get("training_hyperparameters") or {},
        "cross_references": step1.get("cross_references") or [],
        "implementation_summary": impl,
        "readiness": readiness,
        "reproduction_feasibility": feasibility,
        "assumption_plan": assumption_plan,
        "warnings": warnings,
        "assumptions": [a.get("assumption_item") for a in assumption_plan],
    }


def run_step2_for_one_paper(paper_dir: Path, step1_dir: Path, step2_dir: Path) -> str:
    step2_dir.mkdir(parents=True, exist_ok=True)
    step1_path = step1_dir / f"{paper_dir.name}_paper_spec.json"
    step1 = _read_json(step1_path)

    paper_name = (step1.get("paper_identity") or {}).get("paper_name") or paper_dir.name
    out = {
        "paper_name": paper_name,
        "source_step1": str(step1_path),
        "repro_spec": _build_repro_spec(step1),
    }

    out_path = step2_dir / f"{paper_dir.name}_repro_spec.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out_path)


def run_step2_for_all_papers(raw_data_dir: Path, step1_dir: Path, step2_dir: Path) -> List[str]:
    step2_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for p in sorted(raw_data_dir.glob("paper*")):
        if p.is_dir():
            outputs.append(run_step2_for_one_paper(p, step1_dir, step2_dir))
    typo = raw_data_dir / "pape2"
    if typo.exists() and typo.is_dir():
        outputs.append(run_step2_for_one_paper(typo, step1_dir, step2_dir))
    return outputs
