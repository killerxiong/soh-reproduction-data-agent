from typing import Any, Dict, List


def ensure_step1_keys(spec: Dict[str, Any]) -> Dict[str, Any]:
    defaults = {
        "paper_identity": {},
        "reproduction_scope": {},
        "target_definition": {},
        "feature_recipe": [],
        "data_requirements": {},
        "model_definition": {},
        "training_and_hyperparameters": {},
        "experiment_protocol": {},
        "output_metrics": {},
        "feature_formulas": [],
        "model_equations": [],
        "model_architecture": {},
        "training_hyperparameters": {},
        "cross_references": [],
        "readiness": {},
        "readiness_audit": {},
        "missing_items_global": [],
        "paper_dir": "",
        "inputs": {},
        "urls": [],
        "parse_status": "",
        "parse_warnings": [],
    }
    out = dict(defaults)
    out.update(spec or {})
    if not isinstance(out["feature_recipe"], list):
        out["feature_recipe"] = []
    if not isinstance(out["feature_formulas"], list):
        out["feature_formulas"] = []
    if not isinstance(out["model_equations"], list):
        out["model_equations"] = []
    if not isinstance(out["cross_references"], list):
        out["cross_references"] = []
    if not isinstance(out["missing_items_global"], list):
        out["missing_items_global"] = []
    if not isinstance(out["parse_warnings"], list):
        out["parse_warnings"] = [str(out["parse_warnings"])]
    return out


def normalize_signals(v) -> List[str]:
    raw = v if isinstance(v, list) else [v]
    out = []
    for x in raw:
        for t in str(x or "").replace("|", "/").replace(",", "/").split("/"):
            s = t.strip().lower()
            if not s:
                continue
            if "volt" in s:
                out.append("voltage")
            elif "curr" in s or s in {"i", "a", "ma"}:
                out.append("current")
            elif "time" in s:
                out.append("time")
            elif "cap" in s:
                out.append("capacity")
            elif "temp" in s:
                out.append("temperature")
            elif "cycle" in s:
                out.append("cycle")
            elif "soc" in s:
                out.append("soc")
            elif "soh" in s:
                out.append("soh")
            elif "eis" in s:
                out.append("eis")
            elif "resist" in s or "impedance" in s:
                out.append("resistance")
            else:
                out.append(s)
    return sorted(set(out))


def normalize_feature_recipe(spec: Dict[str, Any]) -> Dict[str, Any]:
    fr = []
    for i, f in enumerate(spec.get("feature_recipe", []) or []):
        if not isinstance(f, dict):
            continue
        x = dict(f)
        x.setdefault("feature_id", f"f{i+1:03d}")
        x["source_signals"] = normalize_signals(x.get("source_signals", []))
        x.setdefault("operation", "unknown")
        x.setdefault("modality", "unknown")
        x.setdefault("construction_steps", [])
        if not isinstance(x["construction_steps"], list):
            x["construction_steps"] = [str(x["construction_steps"])] if x["construction_steps"] else []
        x.setdefault("formula", "")
        x.setdefault("formula_details", {
            "paper_formula_raw": [],
            "operation_formula": "",
            "aggregation_formula": "",
            "incremental_formula": "",
            "normalization_formula": "",
            "label_alignment_formula": "",
            "variable_definitions": {},
            "formula_evidence": [],
        })
        if not isinstance(x["formula_details"], dict):
            x["formula_details"] = {
                "paper_formula_raw": [],
                "operation_formula": "",
                "aggregation_formula": "",
                "incremental_formula": "",
                "normalization_formula": "",
                "label_alignment_formula": "",
                "variable_definitions": {},
                "formula_evidence": [],
            }
        fd = x["formula_details"]
        fd.setdefault("paper_formula_raw", [])
        fd.setdefault("operation_formula", "")
        fd.setdefault("aggregation_formula", "")
        fd.setdefault("incremental_formula", "")
        fd.setdefault("normalization_formula", "")
        fd.setdefault("label_alignment_formula", "")
        fd.setdefault("variable_definitions", {})
        fd.setdefault("formula_evidence", [])
        if not isinstance(fd["paper_formula_raw"], list):
            fd["paper_formula_raw"] = [str(fd["paper_formula_raw"])] if fd["paper_formula_raw"] else []
        if not isinstance(fd["variable_definitions"], dict):
            fd["variable_definitions"] = {}
        if not isinstance(fd["formula_evidence"], list):
            fd["formula_evidence"] = []
        x.setdefault("source_modality", {
            "primary": "unknown",
            "allowed": [],
            "not_primary": [],
            "notes": ""
        })
        if not isinstance(x["source_modality"], dict):
            x["source_modality"] = {"primary": "unknown", "allowed": [], "not_primary": [], "notes": ""}
        x["source_modality"].setdefault("primary", "unknown")
        x["source_modality"].setdefault("allowed", [])
        x["source_modality"].setdefault("not_primary", [])
        x["source_modality"].setdefault("notes", "")
        x.setdefault("formula_completeness", {
            "operation_formula_present": False,
            "aggregation_formula_present": False,
            "incremental_formula_present": False,
            "normalization_formula_present": False,
            "variable_definitions_present": False,
            "status": "missing",
            "missing_formula_parts": [],
        })
        if not isinstance(x["formula_completeness"], dict):
            x["formula_completeness"] = {
                "operation_formula_present": False,
                "aggregation_formula_present": False,
                "incremental_formula_present": False,
                "normalization_formula_present": False,
                "variable_definitions_present": False,
                "status": "missing",
                "missing_formula_parts": [],
            }
        fc = x["formula_completeness"]
        fc["operation_formula_present"] = bool(fd.get("operation_formula"))
        fc["aggregation_formula_present"] = bool(fd.get("aggregation_formula"))
        fc["incremental_formula_present"] = bool(fd.get("incremental_formula"))
        fc["normalization_formula_present"] = bool(fd.get("normalization_formula"))
        fc["variable_definitions_present"] = bool(fd.get("variable_definitions"))
        fc.setdefault("missing_formula_parts", [])
        if not isinstance(fc["missing_formula_parts"], list):
            fc["missing_formula_parts"] = []
        if not fc.get("status"):
            if fc["operation_formula_present"] and (fc["aggregation_formula_present"] or fc["incremental_formula_present"] or fc["normalization_formula_present"]):
                fc["status"] = "partial"
            elif fc["operation_formula_present"]:
                fc["status"] = "partial"
            else:
                fc["status"] = "missing"
        x.setdefault("window_or_segment", "")
        x.setdefault("group_level", "unknown")
        x.setdefault("unit", "")
        x.setdefault("normalization_or_scaling", "")
        x.setdefault("status", x.get("paper_status", "missing"))
        x.setdefault("paper_status", x.get("status", "missing"))
        x.setdefault("evidence", [])
        x.setdefault("missing_details", [])
        x.setdefault("unresolved_details", x.get("missing_details", []))
        x.setdefault("segment", {
            "type": "unknown", "voltage_min": None, "voltage_max": None,
            "soc_min": None, "soc_max": None, "time_min": None, "time_max": None,
            "cycle_min": None, "cycle_max": None, "description": ""
        })
        x.setdefault("resampling", {
            "required": False, "method": "", "grid_signal": "", "num_points": None, "description": ""
        })
        x.setdefault("normalization", {
            "required": False, "method": "", "scope": "unknown", "description": ""
        })
        x.setdefault("label_alignment", {
            "target": "SOH", "alignment": "same_cycle", "description": ""
        })
        fr.append(x)
    spec["feature_recipe"] = fr
    return spec


def normalize_target_definition(spec: Dict[str, Any]) -> Dict[str, Any]:
    td = spec.get("target_definition", {}) or {}
    if not isinstance(td, dict):
        td = {}
    ts = str(td.get("target_source_signal", "") or "").strip()
    formula = str(td.get("target_formula", "") or "").lower()
    tname = str(td.get("target_name", "") or "").lower()

    # 避免把输入特征描述误写进 target_source_signal
    noisy = any(k in ts.lower() for k in ["statistical feature", "features extracted", "voltage/current/time", "input feature"])
    if noisy or len(ts) > 64 or " " in ts:
        td["target_source_signal"] = ""

    # SOH + 容量比公式时，优先规范到容量来源
    if ("soh" in tname or "state of health" in tname or "soh" in formula) and (
        "capacity" in formula or "q0" in formula or "q_" in formula
    ):
        if not td.get("target_source_signal"):
            td["target_source_signal"] = "available_capacity_per_cycle"
        if not td.get("status"):
            td["status"] = "inferred_from_text"

    spec["target_definition"] = td
    return spec


def validate_or_repair_spec(spec: Dict[str, Any], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    spec = ensure_step1_keys(spec)
    spec = normalize_target_definition(spec)
    spec = normalize_feature_recipe(spec)
    warnings = spec.get("parse_warnings", [])
    for f in spec.get("feature_recipe", []):
        if not f.get("source_signals"):
            f["source_signals"] = ["unknown"]
            warnings.append(f"{f.get('feature_id')}: missing source_signals")
        if not f.get("operation"):
            f["operation"] = "unknown"
            warnings.append(f"{f.get('feature_id')}: missing operation")
        if not f.get("group_level"):
            f["group_level"] = "unknown"
            warnings.append(f"{f.get('feature_id')}: missing group_level")
    spec["parse_warnings"] = warnings
    return spec
