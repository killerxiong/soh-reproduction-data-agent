from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any


def safe_slug(text: str, default: str = "paper") -> str:
    text = (text or "").strip()
    if not text:
        return default
    text = re.sub(r"[^\w\-.]+", "_", text, flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or default


@dataclass
class PaperWorkspace:
    root: Path
    input_dir: Path
    work_dir: Path
    code_dir: Path
    data_dir: Path
    model_dir: Path
    results_dir: Path
    reports_dir: Path
    final_dir: Path
    logs_dir: Path

    @property
    def paper_md(self) -> Path:
        return self.work_dir / "paper.md"

    @property
    def supplementary_md(self) -> Path:
        return self.work_dir / "Supplementary information.md"

    @property
    def paper_spec_json(self) -> Path:
        return self.work_dir / "paper_spec.json"

    @property
    def repro_spec_json(self) -> Path:
        return self.work_dir / "repro_spec.json"

    @property
    def reproduction_plan_json(self) -> Path:
        return self.work_dir / "reproduction_plan.json"

    @property
    def normalized_plan_json(self) -> Path:
        return self.work_dir / "normalized_reproduction_plan.json"

    @property
    def final_result_json(self) -> Path:
        return self.final_dir / "final_result.json"

    @property
    def final_summary_md(self) -> Path:
        return self.final_dir / "final_summary.md"

    @property
    def agent_trace_jsonl(self) -> Path:
        return self.logs_dir / "agent_trace.jsonl"

    @property
    def run_manifest_json(self) -> Path:
        return self.logs_dir / "run_manifest.json"


def create_workspace(out_root: Path, paper_id: Optional[str], paper_pdf: Path, overwrite: bool = False) -> PaperWorkspace:
    out_root = Path(out_root)
    paper_pdf = Path(paper_pdf)
    pid = safe_slug(paper_id or paper_pdf.stem)
    root = out_root / pid

    if root.exists() and overwrite:
        shutil.rmtree(root)

    dirs = {
        "input_dir": root / "input",
        "work_dir": root / "_work",
        "code_dir": root / "code",
        "data_dir": root / "data",
        "model_dir": root / "model",
        "results_dir": root / "results",
        "reports_dir": root / "reports",
        "final_dir": root / "final",
        "logs_dir": root / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return PaperWorkspace(root=root, **dirs)


def copy_input_files(ws: PaperWorkspace, paper_pdf: Path, supp_pdf: Optional[Path] = None) -> Dict[str, Any]:
    paper_pdf = Path(paper_pdf)
    copied: Dict[str, Any] = {}

    target_paper = ws.input_dir / paper_pdf.name
    if paper_pdf.resolve() != target_paper.resolve():
        shutil.copy2(paper_pdf, target_paper)
    copied["paper_pdf"] = str(target_paper)

    if supp_pdf:
        supp_pdf = Path(supp_pdf)
        target_supp = ws.input_dir / supp_pdf.name
        if supp_pdf.resolve() != target_supp.resolve():
            shutil.copy2(supp_pdf, target_supp)
        copied["supplementary_pdf"] = str(target_supp)
    else:
        copied["supplementary_pdf"] = None
    return copied


def copy_tree_contents(src: Path, dst: Path, overwrite: bool = True) -> None:
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.exists() and overwrite:
                shutil.rmtree(target)
            if not target.exists():
                shutil.copytree(item, target)
        else:
            if target.exists() and not overwrite:
                continue
            shutil.copy2(item, target)
