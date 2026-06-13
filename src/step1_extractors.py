import json
from typing import Any, Dict, List

from .codex_client import call_codex_json


def _chunks_to_text(chunks: List[Dict[str, Any]], max_chars: int = 42000) -> str:
    parts = []
    total = 0
    for c in chunks:
        block = f"[{c['chunk_id']}|{c['source']}|{c['section_title']}]\n{c['text']}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts)


def _extract(name: str, chunks: List[Dict[str, Any]], schema: Dict[str, Any], instruction: str):
    system = (
        "You are a paper-information extraction agent for reproducibility. "
        "Return STRICT JSON only (no markdown, no prose outside JSON). "
        "Use only provided chunks, and cite supporting chunk_id in evidence."
    )
    user = f"""
task={name}
instruction={instruction}
chunks:
{_chunks_to_text(chunks)}
schema:
{json.dumps(schema, ensure_ascii=False)}
"""
    return call_codex_json(system, user)


def extract_paper_identity(chunks):
    return _extract("paper_identity", chunks,
                    {"paper_name": "", "task_type": "", "domain": "", "evidence": []},
                    "Extract paper name, task type and domain from the paper.")


def extract_target_definition(chunks):
    instruction = """
Extract ONLY target/label definition.

Rules:
1) target_source_signal means where label y comes from (raw or directly-derived label signal),
   NOT model input features.
2) For SOH tasks, if SOH is defined by capacity ratio (e.g., Qk/Q0), use:
   - target_source_signal: "available_capacity_per_cycle" or "capacity"
3) Do NOT put feature descriptions (voltage/current/time statistical features) into target_source_signal.
4) If unclear, set status='ambiguous' and keep target_source_signal empty.
5) evidence must cite chunk_id.
"""
    return _extract("target_definition", chunks,
                    {
                        "target_name": "",
                        "target_formula": "",
                        "target_source_signal": "",
                        "label_source_signal": "",
                        "label_granularity": "",
                        "model_input_signals": [],
                        "model_input_features_summary": "",
                        "status": "",
                        "evidence": []
                    },
                    instruction)


def extract_feature_recipe(chunks):
    schema = {"feature_recipe": [{"name": "", "feature_family": "unknown", "source_signals": [], "operation": "unknown", "cycle_phase": "unknown", "modality": "unknown", "segment": {"type": "unknown", "voltage_min": None, "voltage_max": None, "soc_min": None, "soc_max": None, "time_min": None, "time_max": None, "cycle_min": None, "cycle_max": None, "description": ""}, "resampling": {"required": False, "method": "", "grid_signal": "", "num_points": None, "description": ""}, "smoothing": {"required": False, "method": "", "window": None, "description": ""}, "aggregation_level": "unknown", "group_level": "unknown", "unit": "", "normalization": {"required": False, "method": "", "scope": "unknown", "description": ""}, "label_alignment": {"target": "SOH", "alignment": "same_cycle", "description": ""}, "formula": "", "formula_details": {"paper_formula_raw": [], "operation_formula": "", "aggregation_formula": "", "incremental_formula": "", "normalization_formula": "", "label_alignment_formula": "", "variable_definitions": {}, "formula_evidence": []}, "source_modality": {"primary": "unknown", "allowed": [], "not_primary": [], "notes": ""}, "formula_completeness": {"operation_formula_present": False, "aggregation_formula_present": False, "incremental_formula_present": False, "normalization_formula_present": False, "variable_definitions_present": False, "status": "missing", "missing_formula_parts": []}, "construction_steps": [], "paper_status": "missing", "status": "missing", "evidence": [], "missing_details": [], "unresolved_details": []}]}
    instruction = """
Extract feature construction information for downstream code reproduction.

Critical rule (general, not paper-specific):
- Do NOT merge multiple real features into one item.
- If a paper says "N statistics from voltage curve" and "N statistics from current curve",
  expand into separate items per statistic per signal.
- Each statistic must be one feature item (e.g., mean/std/kurtosis/skewness/slope/entropy/duration/integral).

Expected behavior:
- One feature item = one computable feature column.
- Keep source_signals atomic list (e.g., ["voltage"], ["current"], ["time"]).
- Use operation to reflect the specific statistic/operation.
- If cycle index is used as an input feature, output it as a separate feature item.
- If details are unclear, keep the feature item but mark status/ambiguous and unresolved_details.
- Preserve formulas with high fidelity. Search main paper and supplementary chunks.
- Do not only summarize formulas: keep raw formula text whenever available.
- Fill formula_details:
  - paper_formula_raw
  - operation_formula
  - aggregation_formula
  - incremental_formula
  - normalization_formula
  - label_alignment_formula
  - variable_definitions
  - formula_evidence (chunk_id + text)
- For resistance / impedance / energy / autocorrelation / entropy / slope / derivative /
  capacity / incremental / normalized / percent-loss features, explicitly search formulas.
- Do NOT invent formulas. If referenced but missing, record missing_details/unresolved_details.
- Identify source_modality:
  aging_cycle_charge_curve / aging_cycle_discharge_curve /
  aging_cycle_charge_discharge_timeseries / rpt_capacity_test / hppc_pulse_test /
  eis_impedance_spectrum / ocv_test / metadata / unknown.
- If feature is defined from charging/discharging V-I curves, do not mark EIS/HPPC as primary.
- Fill formula_completeness status and missing_formula_parts.

Return JSON only.
"""
    out = _extract("feature_recipe", chunks, schema, instruction)
    return out.get("feature_recipe", []) if isinstance(out, dict) else []


def extract_data_requirements(chunks):
    return _extract("data_requirements", chunks,
                    {"required_signals": [], "required_experiment_types": [], "required_granularity": "", "optional_signals": [], "blocking_missing_if_absent": []},
                    "Infer data requirements from target, features and model.")


def extract_model_definition(chunks):
    return _extract("model_definition", chunks,
                    {"model_name": "", "model_family": "unknown", "architecture": "", "layers": [], "loss_terms": [], "optimizer_or_solver": "", "training_paradigm": "unknown", "status": "missing", "evidence": []},
                    "Extract model definition; support linear/statistical/ML/NN/PINN/physics models.")


def extract_training_and_hyperparameters(chunks):
    return _extract("training_and_hyperparameters", chunks,
                    {"paper_reported": {}, "code_reported": {}, "codex_inferred_with_evidence": {}, "missing_in_paper": [], "evidence": []},
                    "Extract training/hyperparameters with clear source split and evidence.")


def extract_experiment_protocol(chunks):
    return _extract("experiment_protocol", chunks,
                    {"train_val_test_split": "", "cross_validation": "", "train_test_by_battery_or_random": "", "transfer_setting": "", "dataset_names": [], "evaluation_flow": [], "evidence": [], "missing_or_ambiguous_items": []},
                    "Extract experiment protocol, split strategy, transfer setup, datasets and evaluation flow.")


def extract_output_metrics(chunks):
    return _extract("output_metrics", chunks,
                    {"target_output": "", "metrics": [], "metric_formulas_if_available": {}, "evidence": []},
                    "Extract target output and evaluation metrics.")


def extract_feature_formulas(chunks):
    return _extract(
        "feature_formulas",
        chunks,
        {"feature_formulas": [{"name": "", "formula": "", "variables": {}, "applies_to": [], "evidence": []}]},
        "Extract feature formulas from both paper and supplementary. Include formulas in equations/tables/plain text."
    ).get("feature_formulas", [])


def extract_model_equations(chunks):
    return _extract(
        "model_equations",
        chunks,
        {"model_equations": [{"equation_id": "", "equation_type": "other", "formula": "", "meaning": "", "variables": {}, "used_for": "", "evidence": []}]},
        "Extract model equations (target/dynamics/residual/loss/metric/feature/other) from paper and supplementary."
    ).get("model_equations", [])


def extract_model_architecture(chunks):
    return _extract(
        "model_architecture",
        chunks,
        {"model_architecture": {"architectures": [{"model_name": "", "component_name": "", "layers": [], "parameter_count": "", "inference_time": "", "evidence": []}], "missing_details": []}},
        "Extract model architecture from text/tables/figure captions in paper and supplementary."
    ).get("model_architecture", {})


def extract_training_hyperparameters(chunks):
    return _extract(
        "training_hyperparameters",
        chunks,
        {
            "training_hyperparameters": {
                "optimizer": "",
                "learning_rate": "",
                "batch_size": "",
                "epochs": "",
                "early_stopping": "",
                "loss_weights": {},
                "regularization": "",
                "normalization": "",
                "framework": "",
                "hardware": "",
                "hyperparameter_search": "",
                "random_seed": "",
                "dataset_specific_values": {},
                "missing": [],
                "evidence": [],
            }
        },
        "Extract training hyperparameters from paper and supplementary, including dataset-specific values."
    ).get("training_hyperparameters", {})


def extract_cross_references(chunks):
    return _extract(
        "cross_references",
        chunks,
        {"cross_references": [{"from_chunk": "", "reference_text": "", "resolved_to": [], "confidence": "low"}]},
        "Resolve cross references like Supplementary Note/Table S/Fig S/Appendix from paper to supplementary chunks."
    ).get("cross_references", [])
