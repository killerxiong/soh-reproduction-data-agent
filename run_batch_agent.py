from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from run_paper_agent import run_paper_agent

SUPP_KEYWORDS = ["supp", "supplement", "support", "supporting", "si", "补充"]


def _natural_key(path: Path):
    import re
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(x) if x.isdigit() else x for x in parts]


def discover_paper_dirs(raw_data_dir: Path) -> List[Path]:
    raw_data_dir = Path(raw_data_dir)
    if raw_data_dir.name.lower() == "pdf":
        return [raw_data_dir.parent]
    if (raw_data_dir / "PDF").exists():
        return [raw_data_dir]
    return sorted([p for p in raw_data_dir.glob("paper*") if p.is_dir()], key=_natural_key)


def pick_pdfs(paper_dir: Path) -> Tuple[Path, Optional[Path]]:
    pdf_dir = paper_dir / "PDF" if (paper_dir / "PDF").exists() else paper_dir
    pdfs = sorted([p for p in pdf_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"], key=_natural_key)
    if not pdfs:
        raise RuntimeError(f"No PDF found in {pdf_dir}")

    supps = [p for p in pdfs if any(k in p.name.lower() for k in SUPP_KEYWORDS)]
    mains = [p for p in pdfs if p not in supps]
    paper_pdf = mains[0] if mains else pdfs[0]
    supp_pdf = supps[0] if supps else None
    return paper_pdf, supp_pdf


def write_submission_case_summary(path: Path, summary: Dict[str, object]) -> None:
    papers = summary.get("papers", []) if isinstance(summary.get("papers"), list) else []
    lines = [
        "# Paper2SOH Agent 案例汇总",
        "",
        "本文件用于汇总多篇论文案例。每篇论文的主阅读入口是 `reports/paper_case_report.md`。",
        "",
        f"- Total: {summary.get('total', 0)}",
        f"- Success: {summary.get('success', 0)}",
        f"- Failed: {summary.get('failed', 0)}",
        f"- Steps: {summary.get('steps', '')}",
        "",
        "| Paper ID | Status | Mode | Test RMSE | Test R2 | 主案例报告 |",
        "|---|---|---|---:|---:|---|",
    ]
    for item in papers:
        if not isinstance(item, dict):
            continue
        metrics = item.get("metrics") or {}
        if not isinstance(metrics, dict):
            metrics = {}
        lines.append(
            "| {paper_id} | {status} | {mode} | {rmse} | {r2} | `{report}` |".format(
                paper_id=item.get("paper_id", ""),
                status=item.get("status", ""),
                mode=item.get("selected_mode", ""),
                rmse=metrics.get("test_RMSE", ""),
                r2=metrics.get("test_R2", ""),
                report=item.get("primary_report", ""),
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_batch_agent(
    raw_data_dir: Path,
    out_root: Path,
    overwrite: bool = False,
    keep_work: bool = True,
    steps: str = "all",
) -> Dict[str, object]:
    raw_data_dir = Path(raw_data_dir)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {
        "total": 0,
        "success": 0,
        "failed": 0,
        "steps": steps,
        "papers": [],
    }

    for paper_dir in discover_paper_dirs(raw_data_dir):
        summary["total"] = int(summary["total"]) + 1
        paper_id = paper_dir.name
        try:
            paper_pdf, supp_pdf = pick_pdfs(paper_dir)
            result = run_paper_agent(
                paper_pdf=paper_pdf,
                supp_pdf=supp_pdf,
                out_root=out_root,
                paper_id=paper_id,
                overwrite=overwrite,
                keep_work=keep_work,
                steps=steps,
            )
            summary["success"] = int(summary["success"]) + 1
            summary["papers"].append({
                "paper_id": paper_id,
                "status": result.get("agent_status", "success"),
                "selected_steps": steps,
                "workspace": result.get("workspace"),
                "primary_report": str(Path(result.get("workspace", "")) / "reports" / "paper_case_report.md"),
                "final_result": str(Path(result.get("workspace", "")) / "final" / "final_result.json"),
                "metrics": (result.get("execution") or {}).get("metrics", {}),
                "selected_mode": (result.get("reproduction_mode") or {}).get("selected_mode"),
            })
        except Exception as exc:
            summary["failed"] = int(summary["failed"]) + 1
            summary["papers"].append({
                "paper_id": paper_id,
                "status": "failed",
                "selected_steps": steps,
                "error": str(exc),
            })

    summary_path = out_root / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_submission_case_summary(out_root / "submission_case_summary.md", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch runner for Paper2SOH Agent")
    parser.add_argument("--raw_data_dir", default="raw_data", help="Directory containing paper1/paper2... folders")
    parser.add_argument("--out_root", default="outputs", help="Output root directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite each paper workspace")
    parser.add_argument("--clean_work", action="store_true", help="Remove each paper _work directory after success")
    parser.add_argument(
        "--steps",
        default="all",
        help="Steps to run: all, step1, step1,step2, step1:step3, step4,final. Valid: step0 step1 step2 step3 step4 final",
    )
    args = parser.parse_args()

    summary = run_batch_agent(
        raw_data_dir=Path(args.raw_data_dir),
        out_root=Path(args.out_root),
        overwrite=args.overwrite,
        keep_work=not args.clean_work,
        steps=args.steps,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Batch summary written to: {Path(args.out_root) / 'batch_summary.json'}")
    print(f"Submission case summary written to: {Path(args.out_root) / 'submission_case_summary.md'}")


if __name__ == "__main__":
    main()
