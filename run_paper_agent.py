from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.agent_trace import AgentTrace
from src.final_report import (
    collect_final_result,
    read_json_if_exists,
    write_final_summary,
    write_json,
    write_reproduction_story,
    write_paper_case_report,
)
from src.step0_pdf_to_md import run_step0_pdf_to_md
from src.step1_parse_paper import parse_paper_bundle
from src.step2_prepare_repro_spec import _build_repro_spec
from src.step3_model_plan import build_reproduction_plan
from src.step4_execute_reproduction import run_step4_host
from src.workspace import copy_input_files, copy_tree_contents, create_workspace

STEP_ORDER = ["step0", "step1", "step2", "step3", "step4", "final"]
STEP_ALIASES = {
    "0": "step0",
    "1": "step1",
    "2": "step2",
    "3": "step3",
    "4": "step4",
    "5": "final",
    "report": "final",
    "finalization": "final",
}


def _copy_if_exists(src: Path, dst: Path) -> None:
    src = Path(src)
    dst = Path(dst)
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _stage_file(path: Path) -> str:
    return str(Path(path))


def normalize_steps(steps: Optional[str | Sequence[str]]) -> List[str]:
    """Normalize user step selection.

    Supported values:
    - "all"
    - "step1"
    - "step1,step2,step3"
    - "step1:step3"  # inclusive range
    - ["step1", "step2"]
    """
    if steps is None:
        return list(STEP_ORDER)
    if isinstance(steps, str):
        raw = steps.strip().lower()
        if raw in {"", "all", "full"}:
            return list(STEP_ORDER)
        parts: List[str] = []
        for token in raw.replace("，", ",").split(","):
            token = token.strip()
            if not token:
                continue
            if ":" in token:
                left, right = [x.strip() for x in token.split(":", 1)]
                left = STEP_ALIASES.get(left, left)
                right = STEP_ALIASES.get(right, right)
                if left not in STEP_ORDER or right not in STEP_ORDER:
                    raise ValueError(f"Invalid step range: {token}")
                i, j = STEP_ORDER.index(left), STEP_ORDER.index(right)
                if i > j:
                    raise ValueError(f"Invalid reversed step range: {token}")
                parts.extend(STEP_ORDER[i:j + 1])
            else:
                token = STEP_ALIASES.get(token, token)
                parts.append(token)
    else:
        parts = [STEP_ALIASES.get(str(x).strip().lower(), str(x).strip().lower()) for x in steps]

    out: List[str] = []
    for step in parts:
        if step not in STEP_ORDER:
            raise ValueError(f"Invalid step '{step}'. Valid steps: {STEP_ORDER}")
        if step not in out:
            out.append(step)
    return out


def _require(path: Path, message: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"{message}: {path}")


def _partial_result(ws, selected_steps: List[str], status: str = "partial_success") -> Dict[str, Any]:
    return {
        "paper_id": ws.root.name,
        "workspace": str(ws.root),
        "agent_status": status,
        "selected_steps": selected_steps,
        "available_artifacts": {
            "paper_md": str(ws.paper_md) if ws.paper_md.exists() else None,
            "paper_spec_json": str(ws.paper_spec_json) if ws.paper_spec_json.exists() else None,
            "repro_spec_json": str(ws.repro_spec_json) if ws.repro_spec_json.exists() else None,
            "reproduction_plan_json": str(ws.reproduction_plan_json) if ws.reproduction_plan_json.exists() else None,
            "metrics_json": str(ws.results_dir / "metrics.json") if (ws.results_dir / "metrics.json").exists() else None,
            "final_result_json": str(ws.final_result_json) if ws.final_result_json.exists() else None,
            "agent_trace_jsonl": str(ws.agent_trace_jsonl),
        },
    }


def run_paper_agent(
    paper_pdf: Path,
    out_root: Path,
    supp_pdf: Optional[Path] = None,
    paper_id: Optional[str] = None,
    overwrite: bool = False,
    keep_work: bool = True,
    steps: Optional[str | Sequence[str]] = None,
) -> Dict[str, Any]:
    """Run selected Paper2SOH Agent steps for one PDF.

    Internal logic is still STEP0-STEP4, but outputs are organized by paper.

    steps examples:
      - "all"
      - "step1"
      - "step2,step3"
      - "step1:step3"
      - "step4,final"

    Notes:
      - If you run a later step only, prerequisite files must already exist in
        outputs/<paper_id>/_work.
      - Do not use overwrite=True when running only a later step, because it
        deletes the paper workspace and therefore deletes prerequisites.
    """
    selected_steps = normalize_steps(steps)
    paper_pdf = Path(paper_pdf)
    supp_pdf = Path(supp_pdf) if supp_pdf else None

    if overwrite and selected_steps != STEP_ORDER and "step0" not in selected_steps:
        raise ValueError(
            "overwrite=True would delete existing intermediate files. "
            "Use overwrite=False when running selected later steps."
        )

    ws = create_workspace(out_root=Path(out_root), paper_id=paper_id, paper_pdf=paper_pdf, overwrite=overwrite)
    trace = AgentTrace(ws.agent_trace_jsonl)
    input_files = copy_input_files(ws, paper_pdf, supp_pdf)

    paper_spec: Dict[str, Any] = read_json_if_exists(ws.paper_spec_json)
    repro_spec: Dict[str, Any] = read_json_if_exists(ws.repro_spec_json)
    reproduction_plan: Dict[str, Any] = read_json_if_exists(ws.reproduction_plan_json)
    execution_manifest: Dict[str, Any] = read_json_if_exists(ws.run_manifest_json)

    trace.log(
        stage="agent_run",
        status="start",
        message="Run selected Paper2SOH Agent steps.",
        inputs={"selected_steps": selected_steps, "input_files": input_files},
    )

    try:
        if "step0" in selected_steps:
            trace.start(
                stage="step0_document_parsing",
                tool="MinerU",
                message="Parse paper PDF and optional supplementary PDF into Markdown.",
                inputs=input_files,
            )
            step0_manifest = run_step0_pdf_to_md(
                out_dir=ws.work_dir,
                paper_pdf=paper_pdf,
                supp_pdf=supp_pdf,
                overwrite=True,
            )
            trace.success(
                stage="step0_document_parsing",
                tool="MinerU",
                outputs={
                    "paper_md": _stage_file(ws.paper_md),
                    "supplementary_md": _stage_file(ws.supplementary_md) if ws.supplementary_md.exists() else None,
                    "step0_manifest": step0_manifest,
                },
            )
        else:
            if any(s in selected_steps for s in ["step1", "step2", "step3", "step4", "final"]):
                _require(ws.paper_md, "STEP0 output paper.md is required before running later steps")

        if "step1" in selected_steps:
            trace.start(
                stage="step1_paper_understanding",
                tool="Codex",
                message="Extract paper identity, target, features, model, protocol, metrics and readiness.",
                inputs={
                    "paper_md": _stage_file(ws.paper_md),
                    "supplementary_md": _stage_file(ws.supplementary_md),
                },
            )
            paper_spec = parse_paper_bundle(ws.work_dir)
            write_json(ws.paper_spec_json, paper_spec)
            trace.success(
                stage="step1_paper_understanding",
                tool="Codex",
                outputs={
                    "paper_spec": _stage_file(ws.paper_spec_json),
                    "parse_status": paper_spec.get("parse_status"),
                    "readiness_audit": paper_spec.get("readiness_audit", {}),
                },
            )
        elif any(s in selected_steps for s in ["step2", "step3", "step4", "final"]):
            _require(ws.paper_spec_json, "STEP1 output paper_spec.json is required before running later steps")
            paper_spec = read_json_if_exists(ws.paper_spec_json)

        if "step2" in selected_steps:
            trace.start(
                stage="step2_reproduction_specification",
                message="Build reproduction specification and strict/approx readiness decision.",
                inputs={"paper_spec": _stage_file(ws.paper_spec_json)},
            )
            if not paper_spec:
                paper_spec = read_json_if_exists(ws.paper_spec_json)
            repro_spec = {
                "paper_name": (paper_spec.get("paper_identity") or {}).get("paper_name") or ws.root.name,
                "source_step1": _stage_file(ws.paper_spec_json),
                "repro_spec": _build_repro_spec(paper_spec),
            }
            write_json(ws.repro_spec_json, repro_spec)
            trace.success(
                stage="step2_reproduction_specification",
                outputs={
                    "repro_spec": _stage_file(ws.repro_spec_json),
                    "recommended_mode": (repro_spec.get("repro_spec", {}) or {}).get("reproduction_feasibility", {}).get("recommended_mode"),
                },
            )
        elif any(s in selected_steps for s in ["step3", "step4", "final"]):
            _require(ws.repro_spec_json, "STEP2 output repro_spec.json is required before running later steps")
            repro_spec = read_json_if_exists(ws.repro_spec_json)

        if "step3" in selected_steps:
            trace.start(
                stage="step3_execution_planning",
                tool="Codex",
                message="Generate executable reproduction plan.",
                inputs={
                    "repro_spec": _stage_file(ws.repro_spec_json),
                    "paper_md": _stage_file(ws.paper_md),
                    "supplementary_md": _stage_file(ws.supplementary_md),
                },
            )
            plan_tmp_dir = ws.work_dir / "step3_plan"
            build_reproduction_plan(
                step2_spec=ws.repro_spec_json,
                paper_md=ws.paper_md,
                supp_md=ws.supplementary_md,
                out_dir=plan_tmp_dir,
            )
            generated_plan = plan_tmp_dir / "reproduction_plan.json"
            if not generated_plan.exists():
                raise RuntimeError(f"Missing reproduction plan: {generated_plan}")
            shutil.copy2(generated_plan, ws.reproduction_plan_json)
            reproduction_plan = read_json_if_exists(ws.reproduction_plan_json)
            _copy_if_exists(plan_tmp_dir / "reproduction_plan_report.md", ws.reports_dir / "reproduction_plan_report.md")
            _copy_if_exists(ws.reproduction_plan_json, ws.final_dir / "reproduction_plan.json")
            trace.success(
                stage="step3_execution_planning",
                tool="Codex",
                outputs={
                    "reproduction_plan": _stage_file(ws.reproduction_plan_json),
                    "selected_mode": (reproduction_plan.get("reproduction_mode") or {}).get("selected_mode"),
                },
            )
        elif any(s in selected_steps for s in ["step4", "final"]):
            _require(ws.reproduction_plan_json, "STEP3 output reproduction_plan.json is required before running later steps")
            reproduction_plan = read_json_if_exists(ws.reproduction_plan_json)

        if "step4" in selected_steps:
            trace.start(
                stage="step4_code_generation_and_execution",
                tool="Codex + Python",
                message="Generate runnable reproduction code, execute pipeline, and validate outputs.",
                inputs={"reproduction_plan": _stage_file(ws.reproduction_plan_json)},
            )
            step4_tmp_out = ws.work_dir / "step4_execute"
            run_step4_host(reproduction_plan_path=ws.reproduction_plan_json, out_dir=step4_tmp_out)
            copy_tree_contents(step4_tmp_out / "generated_code", ws.code_dir)
            copy_tree_contents(step4_tmp_out / "data", ws.data_dir)
            copy_tree_contents(step4_tmp_out / "model", ws.model_dir)
            copy_tree_contents(step4_tmp_out / "results", ws.results_dir)
            copy_tree_contents(step4_tmp_out / "normalized_plan", ws.work_dir / "normalized_plan")
            for report_name in ["training_report.md", "model_alignment_report.md"]:
                _copy_if_exists(ws.results_dir / report_name, ws.reports_dir / report_name)
            _copy_if_exists(step4_tmp_out / "normalized_plan" / "plan_repair_report.md", ws.reports_dir / "plan_repair_report.md")
            execution_manifest = read_json_if_exists(step4_tmp_out / "step4_manifest.json")
            write_json(ws.run_manifest_json, execution_manifest)
            trace.success(
                stage="step4_code_generation_and_execution",
                tool="Codex + Python",
                outputs={
                    "code_dir": _stage_file(ws.code_dir),
                    "data_dir": _stage_file(ws.data_dir),
                    "model_dir": _stage_file(ws.model_dir),
                    "results_dir": _stage_file(ws.results_dir),
                    "run_manifest": _stage_file(ws.run_manifest_json),
                },
            )
        elif "final" in selected_steps:
            # Final report can still be generated from previous STEP4 output.
            execution_manifest = read_json_if_exists(ws.run_manifest_json)

        if "final" in selected_steps:
            trace.start(stage="finalization", message="Collect final JSON/Markdown reports.")
            if not paper_spec:
                paper_spec = read_json_if_exists(ws.paper_spec_json)
            if not repro_spec:
                repro_spec = read_json_if_exists(ws.repro_spec_json)
            if not reproduction_plan:
                reproduction_plan = read_json_if_exists(ws.reproduction_plan_json)
            if not execution_manifest:
                execution_manifest = read_json_if_exists(ws.run_manifest_json)
            metrics = read_json_if_exists(ws.results_dir / "metrics.json")
            final_result = collect_final_result(
                paper_id=ws.root.name,
                workspace_root=ws.root,
                input_files=input_files,
                paper_spec=paper_spec,
                repro_spec=repro_spec,
                reproduction_plan=reproduction_plan,
                execution_manifest=execution_manifest,
                metrics=metrics,
            )
            write_json(ws.final_result_json, final_result)
            write_final_summary(ws.final_summary_md, final_result)
            write_reproduction_story(ws.reports_dir / "reproduction_story.md", final_result)
            write_paper_case_report(
                ws.reports_dir / "paper_case_report.md",
                final_result=final_result,
                paper_spec=paper_spec,
                repro_spec=repro_spec,
                reproduction_plan=reproduction_plan,
                workspace_root=ws.root,
            )
            trace.success(
                stage="finalization",
                outputs={
                    "primary_report_md": _stage_file(ws.reports_dir / "paper_case_report.md"),
                    "final_result_json": _stage_file(ws.final_result_json),
                    "final_summary_md": _stage_file(ws.final_summary_md),
                    "reproduction_story_md": _stage_file(ws.reports_dir / "reproduction_story.md"),
                },
            )
            if not keep_work:
                shutil.rmtree(ws.work_dir, ignore_errors=True)
            return final_result

        result = _partial_result(ws, selected_steps, status="partial_success")
        trace.success(stage="agent_run", message="Selected steps finished.", outputs=result)
        return result

    except Exception as exc:
        trace.fail(stage="agent_run", exc=exc)
        error_manifest = {
            "status": "failed",
            "error": str(exc),
            "workspace": str(ws.root),
            "input": input_files,
            "selected_steps": selected_steps,
        }
        write_json(ws.run_manifest_json, error_manifest)
        write_json(ws.final_result_json, {
            "paper_id": ws.root.name,
            "agent_status": "failed",
            "error": str(exc),
            "workspace": str(ws.root),
            "input": input_files,
            "selected_steps": selected_steps,
            "paper_spec_path": str(ws.paper_spec_json) if ws.paper_spec_json.exists() else None,
            "repro_spec_path": str(ws.repro_spec_json) if ws.repro_spec_json.exists() else None,
            "reproduction_plan_path": str(ws.reproduction_plan_json) if ws.reproduction_plan_json.exists() else None,
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper2SOH: SOH estimation paper reproduction Agent")
    parser.add_argument("--paper_pdf", required=True, help="Path to main paper PDF")
    parser.add_argument("--supp_pdf", default=None, help="Path to supplementary PDF")
    parser.add_argument("--out_root", default="outputs", help="Output root directory")
    parser.add_argument("--paper_id", default=None, help="Paper output folder name")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing paper workspace")
    parser.add_argument("--clean_work", action="store_true", help="Remove internal _work directory after success")
    parser.add_argument(
        "--steps",
        default="all",
        help="Steps to run: all, step1, step1,step2, step1:step3, step4,final. Valid: step0 step1 step2 step3 step4 final",
    )
    args = parser.parse_args()

    result = run_paper_agent(
        paper_pdf=Path(args.paper_pdf),
        supp_pdf=Path(args.supp_pdf) if args.supp_pdf else None,
        out_root=Path(args.out_root),
        paper_id=args.paper_id,
        overwrite=args.overwrite,
        keep_work=not args.clean_work,
        steps=args.steps,
    )

    workspace = Path(result.get("workspace", ""))
    print(json.dumps({
        "status": result.get("agent_status"),
        "paper_id": result.get("paper_id"),
        "selected_steps": result.get("selected_steps", args.steps),
        "workspace": str(workspace),
        "primary_report": str(workspace / "reports" / "paper_case_report.md"),
        "final_result": str(workspace / "final" / "final_result.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
