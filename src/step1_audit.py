from typing import Any, Dict, List


def _bool_blockers(ok: bool, blockers: List[str]) -> Dict[str, Any]:
    return {"boolean": ok, "blockers": blockers}


def audit_reproduction_readiness(spec: Dict[str, Any]) -> Dict[str, Any]:
    target_blockers = []
    td = spec.get("target_definition", {})
    if not td.get("target_name"):
        target_blockers.append("target_name missing")
    if not td.get("target_formula") and not td.get("target_source_signal"):
        target_blockers.append("target formula/source missing")

    feat_blockers, feat_warnings = [], []
    for f in spec.get("feature_recipe", []) or []:
        fid = f.get("feature_id", f.get("name", "feature"))
        if not f.get("source_signals"):
            feat_blockers.append(f"{fid}: source_signals missing")
        if not f.get("operation") or f.get("operation") == "unknown":
            feat_blockers.append(f"{fid}: operation missing")
        if not f.get("group_level") or f.get("group_level") == "unknown":
            feat_warnings.append(f"{fid}: group_level unknown")
        if (f.get("status") in {"partially_reported", "ambiguous", "missing"}) or f.get("unresolved_details"):
            feat_warnings.append(f"{fid}: unresolved details present")

    model_blockers = []
    md = spec.get("model_definition", {})
    if not md.get("model_name"):
        model_blockers.append("model_name missing")
    if not md.get("model_family"):
        model_blockers.append("model_family missing")

    dr = spec.get("data_requirements", {})
    data_ready = {
        "boolean": len(dr.get("required_signals", []) or []) > 0,
        "required_signals": dr.get("required_signals", []),
        "required_experiment_types": dr.get("required_experiment_types", []),
    }

    formula_warnings = []
    for f in spec.get("feature_recipe", []) or []:
        fid = f.get("feature_id", f.get("name", "feature"))
        fam = str(f.get("feature_family", "")).lower()
        op = str(f.get("operation", "")).lower()
        formula = str(f.get("formula", "")).lower()
        fd = f.get("formula_details", {}) if isinstance(f.get("formula_details", {}), dict) else {}
        op_formula = str(fd.get("operation_formula", "")).lower()
        agg_formula = str(fd.get("aggregation_formula", "")).lower()
        inc_formula = str(fd.get("incremental_formula", "")).lower()
        norm_formula = str(fd.get("normalization_formula", "")).lower()
        fev = fd.get("formula_evidence", []) if isinstance(fd.get("formula_evidence", []), list) else []

        if "resist" in fam:
            if not any(k in (op_formula + formula) for k in ["delta", "v", "i", "/"]):
                formula_warnings.append(f"{fid}: resistance feature missing clear ΔV/ΔI-style formula")
        if "impedance" in fam:
            if not op_formula:
                formula_warnings.append(f"{fid}: impedance feature missing operation_formula")
        if any(k in op for k in ["integral", "integrate", "energy"]):
            if not any(k in (op_formula + agg_formula + formula) for k in ["v*i", "integral", "∫", "trapz"]):
                formula_warnings.append(f"{fid}: integration/energy feature missing power/integral formula")
        if any(k in op for k in ["norm", "normalized"]):
            if not norm_formula:
                formula_warnings.append(f"{fid}: normalized feature missing normalization_formula")
        if any(k in op for k in ["increment", "subtract"]):
            if not inc_formula:
                formula_warnings.append(f"{fid}: incremental feature missing incremental_formula")
        if not fev:
            formula_warnings.append(f"{fid}: formula_evidence missing")

    feature_formula_ready = len(formula_warnings) == 0 or len(spec.get("feature_formulas", []) or []) > 0
    model_equation_ready = len(spec.get("model_equations", []) or []) > 0
    model_arch_ready = len((spec.get("model_architecture", {}) or {}).get("architectures", []) or []) > 0
    th = spec.get("training_hyperparameters", {}) or {}
    th_missing = th.get("missing", []) if isinstance(th, dict) else []
    th_ready = "false" if (not th or (not th.get("optimizer") and not th.get("batch_size"))) else ("partial" if th_missing else "true")
    ep = spec.get("experiment_protocol", {}) or {}
    exp_ready = "partial" if not ep.get("train_val_test_split") and not ep.get("split_type") else "true"
    strict_ok = len(target_blockers) == 0 and len(feat_blockers) == 0 and feature_formula_ready and len(model_blockers) == 0 and model_equation_ready and model_arch_ready and th_ready == "true" and exp_ready == "true"
    approx_ok = len(target_blockers) == 0 and len(feat_blockers) == 0 and len(model_blockers) == 0

    known_missing = []
    known_missing.extend(target_blockers)
    known_missing.extend(feat_blockers)
    known_missing.extend(model_blockers)
    known_missing.extend([f"training:{x}" for x in th_missing])

    return {
        "target_ready": len(target_blockers) == 0,
        "feature_list_ready": len(feat_blockers) == 0,
        "feature_formula_ready": feature_formula_ready,
        "feature_segment_ready": len([w for w in feat_warnings if "segment" in w.lower()]) == 0,
        "model_family_ready": len(model_blockers) == 0,
        "model_equation_ready": model_equation_ready,
        "model_architecture_ready": model_arch_ready,
        "training_hyperparams_ready": th_ready,
        "experiment_protocol_ready": exp_ready,
        "strict_reproduction_ready": strict_ok,
        "approximate_implementation_ready": approx_ok,
        "known_missing": known_missing,
        "warnings": feat_warnings + formula_warnings,
        "data_matching_ready": data_ready,
    }
