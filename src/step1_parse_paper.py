import json
from pathlib import Path
from typing import Dict, List, Any

from .step1_chunking import load_paper_inputs, build_paper_chunks
from .step1_extractors import (
    extract_paper_identity,
    extract_target_definition,
    extract_feature_recipe,
    extract_data_requirements,
    extract_model_definition,
    extract_training_and_hyperparameters,
    extract_experiment_protocol,
    extract_output_metrics,
    extract_feature_formulas,
    extract_model_equations,
    extract_model_architecture,
    extract_training_hyperparameters,
    extract_cross_references,
)
from .step1_schemas import ensure_step1_keys, validate_or_repair_spec
from .step1_audit import audit_reproduction_readiness


def _fallback_spec(paper_dir: Path, inputs: Dict[str, Any], err: str) -> Dict[str, Any]:
    return {
        "paper_identity": {"paper_name": "", "task_type": "unknown", "domain": "battery", "evidence": []},
        "reproduction_scope": {"primary_focus": "mixed", "reason": "codex_parse_failed", "evidence": []},
        "target_definition": {"target_name": "", "target_formula": "", "target_source_signal": "", "label_granularity": "unknown", "status": "missing", "evidence": []},
        "feature_recipe": [],
        "data_requirements": {"required_signals": [], "required_experiment_types": [], "required_granularity": "unknown", "optional_signals": [], "blocking_missing_if_absent": []},
        "model_definition": {"model_name": "", "model_family": "unknown", "architecture": "", "layers": [], "loss_terms": [], "optimizer_or_solver": "", "training_paradigm": "unknown", "status": "missing", "evidence": []},
        "training_and_hyperparameters": {"paper_reported": {}, "code_reported": {}, "codex_inferred_with_evidence": {}, "missing_in_paper": ["codex_parse_failed"], "evidence": []},
        "experiment_protocol": {"train_val_test_split": "", "cross_validation": "", "train_test_by_battery_or_random": "", "transfer_setting": "", "dataset_names": [], "evaluation_flow": [], "evidence": [], "missing_or_ambiguous_items": []},
        "output_metrics": {"target_output": "", "metrics": [], "metric_formulas_if_available": {}, "evidence": []},
        "feature_formulas": [],
        "model_equations": [],
        "model_architecture": {"architectures": [], "missing_details": []},
        "training_hyperparameters": {},
        "cross_references": [],
        "readiness": {},
        "readiness_audit": {},
        "missing_items_global": ["codex_parse_failed"],
        "paper_dir": str(paper_dir),
        "inputs": {
            "paper_md": inputs.get("paper_md"),
            "supplementary_md": inputs.get("supplementary_md"),
            "paper_exists": inputs.get("paper_exists"),
            "supplementary_exists": inputs.get("supplementary_exists"),
        },
        "urls": inputs.get("urls", []),
        "parse_status": "fallback_rule",
        "parse_warnings": [str(err)],
    }


def _merge_step1_spec(inputs: Dict[str, Any], **parts) -> Dict[str, Any]:
    return {
        "paper_identity": parts.get("identity", {}),
        "reproduction_scope": parts.get("scope", {}),
        "target_definition": parts.get("target", {}),
        "feature_recipe": parts.get("features", []),
        "data_requirements": parts.get("data_requirements", {}),
        "model_definition": parts.get("model", {}),
        "training_and_hyperparameters": parts.get("training", {}),
        "experiment_protocol": parts.get("protocol", {}),
        "output_metrics": parts.get("metrics", {}),
        "feature_formulas": parts.get("feature_formulas", []),
        "model_equations": parts.get("model_equations", []),
        "model_architecture": parts.get("model_architecture", {}),
        "training_hyperparameters": parts.get("training_hyperparameters", {}),
        "cross_references": parts.get("cross_references", []),
        "readiness": parts.get("readiness", {}),
        "missing_items_global": parts.get("missing_items_global", []),
        "paper_dir": inputs.get("paper_dir", ""),
        "inputs": {
            "paper_md": inputs.get("paper_md"),
            "supplementary_md": inputs.get("supplementary_md"),
            "paper_exists": inputs.get("paper_exists"),
            "supplementary_exists": inputs.get("supplementary_exists"),
        },
        "urls": inputs.get("urls", []),
        "parse_status": "codex_success_validated",
        "parse_warnings": [],
    }


def parse_paper_bundle(paper_dir: Path) -> Dict[str, Any]:
    inputs = load_paper_inputs(paper_dir)
    try:
        chunks = build_paper_chunks(inputs)

        identity = extract_paper_identity(chunks)
        target = extract_target_definition(chunks)
        features = extract_feature_recipe(chunks)
        data_requirements = extract_data_requirements(chunks)
        model = extract_model_definition(chunks)
        training = extract_training_and_hyperparameters(chunks)
        protocol = extract_experiment_protocol(chunks)
        metrics = extract_output_metrics(chunks)
        feature_formulas = extract_feature_formulas(chunks)
        model_equations = extract_model_equations(chunks)
        model_architecture = extract_model_architecture(chunks)
        training_hparams = extract_training_hyperparameters(chunks)
        cross_references = extract_cross_references(chunks)

        scope = {
            "primary_focus": "mixed",
            "reason": "assembled_from_multi_extractors",
            "evidence": [],
        }

        spec = _merge_step1_spec(
            inputs,
            identity=identity,
            scope=scope,
            target=target,
            features=features,
            data_requirements=data_requirements,
            model=model,
            training=training,
            protocol=protocol,
            metrics=metrics,
            feature_formulas=feature_formulas,
            model_equations=model_equations,
            model_architecture=model_architecture,
            training_hyperparameters=training_hparams,
            cross_references=cross_references,
        )

        spec = ensure_step1_keys(spec)
        spec = validate_or_repair_spec(spec, chunks)
        spec["readiness_audit"] = audit_reproduction_readiness(spec)
        if spec.get("parse_warnings"):
            spec["parse_status"] = "codex_partial_with_warnings"
        return spec
    except Exception as e:
        return _fallback_spec(paper_dir, inputs, str(e))


def run_step1_for_one_paper(paper_dir: Path, out_root: Path) -> str:
    out_root.mkdir(parents=True, exist_ok=True)
    spec = parse_paper_bundle(paper_dir)
    out_path = out_root / f"{paper_dir.name}_paper_spec.json"
    out_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out_path)


def run_step1_for_all_papers(raw_data_dir: Path, out_root: Path) -> List[str]:
    out_root.mkdir(parents=True, exist_ok=True)
    outputs = []
    for p in sorted(raw_data_dir.glob("paper*")):
        if p.is_dir():
            outputs.append(run_step1_for_one_paper(p, out_root))
    typo = raw_data_dir / "pape2"
    if typo.exists() and typo.is_dir():
        outputs.append(run_step1_for_one_paper(typo, out_root))
    return outputs
