"""
PyCharm one-click runner for a single paper.

Usage:
1. Open this project in PyCharm.
2. Edit the configuration block below.
3. Right click this file -> Run 'run_local_paper'.

Step control:
- RUN_STEPS = "all"          run full workflow
- RUN_STEPS = "step1"        run only STEP1 using existing _work/paper.md
- RUN_STEPS = "step2"        run only STEP2 using existing _work/paper_spec.json
- RUN_STEPS = "step3"        run only STEP3 using existing _work/repro_spec.json
- RUN_STEPS = "step4"        run only STEP4 using existing _work/reproduction_plan.json
- RUN_STEPS = "final"        regenerate final reports only
- RUN_STEPS = "step1:step3"  run STEP1, STEP2, STEP3
- RUN_STEPS = "step4,final"  run STEP4 and then final report

Do not commit real API keys to GitHub. Prefer configuring them in PyCharm
Run/Debug Configuration -> Environment variables.
"""
from __future__ import annotations

import os
from pathlib import Path


# =========================
# PyCharm configuration area
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent

# Main paper PDF. Change this to your local paper path.
# PAPER_PDF = PROJECT_ROOT / "raw_data" / "paper2" / "PDF" / "2.pdf"
# PAPER_PDF = PROJECT_ROOT / "raw_data" / "paper3" / "PDF" / "3.pdf"
# PAPER_PDF = PROJECT_ROOT / "raw_data" / "paper4" / "PDF" / "4.pdf"
# PAPER_PDF = PROJECT_ROOT / "raw_data" / "paper5" / "PDF" / "5.pdf"
PAPER_PDF = PROJECT_ROOT / "raw_data" / "paper6" / "PDF" / "6.pdf"

# Supplementary PDF. Set to None when there is no supplementary file.
# SUPP_PDF = None
# Example:
# SUPP_PDF = PROJECT_ROOT / "raw_data" / "paper2" / "PDF" / "2-supp.pdf"
# SUPP_PDF = None
# SUPP_PDF = PROJECT_ROOT / "raw_data" / "paper4" / "PDF" / "4-supp.pdf"
# SUPP_PDF = PROJECT_ROOT / "raw_data" / "paper5" / "PDF" / "5-supp.pdf"
SUPP_PDF = PROJECT_ROOT / "raw_data" / "paper6" / "PDF" / "6-supp.pdf"

# Output root. The agent will create OUT_ROOT / PAPER_ID.
OUT_ROOT = PROJECT_ROOT / "outputs"
# PAPER_ID = "paper2"
# PAPER_ID = "paper3"
# PAPER_ID = "paper4"
# PAPER_ID = "paper5"
PAPER_ID = "paper6"

# Choose which steps to run.
# Recommended while debugging:
#   First run: RUN_STEPS = "all" or "step0:step2"
#   Rerun only paper understanding: RUN_STEPS = "step1"
#   Rerun only reproduction planning: RUN_STEPS = "step3"
#   Rerun code generation/execution: RUN_STEPS = "step4,final"
RUN_STEPS = "all"

# Whether to delete an existing outputs/PAPER_ID folder before running.
# IMPORTANT: keep this False when RUN_STEPS is not "all" or "step0...",
# otherwise prerequisites in _work will be deleted.
OVERWRITE = False

# Keep internal intermediate files in outputs/PAPER_ID/_work for debugging and traceability.
KEEP_WORK = True

# Optional local API configuration.
# Recommended: leave secrets empty here and set them in PyCharm Environment variables.
MINERU_TOKEN = os.getenv("MINERU_TOKEN", "")
CODEX_API_KEY = os.getenv("CODEX_API_KEY") or os.getenv("OPENAI_API_KEY") or ""

# Optional Codex/OpenAI compatible endpoint settings.
# Leave empty to use defaults from src/config.py or environment variables.
CODEX_BASE_URL = os.getenv("CODEX_BASE_URL")
CODEX_MODEL_NAME = os.getenv("CODEX_MODEL_NAME", "gpt-5.5")
CODEX_PROXY = os.getenv("CODEX_PROXY")
CODEX_TIMEOUT = int(os.getenv("CODEX_TIMEOUT", "600"))
CODEX_REASONING_EFFORT = os.getenv("CODEX_REASONING_EFFORT", "medium")


def set_env_if_value(name: str, value: object) -> None:
    value = str(value or "").strip()
    if value:
        os.environ[name] = value


def main() -> None:
    # These must be set before importing run_paper_agent because src/config.py
    # reads environment variables at import time.
    set_env_if_value("MINERU_TOKEN", MINERU_TOKEN)
    set_env_if_value("CODEX_API_KEY", CODEX_API_KEY)
    # set_env_if_value("OPENAI_API_KEY", OPENAI_API_KEY)
    set_env_if_value("CODEX_BASE_URL", CODEX_BASE_URL)
    set_env_if_value("CODEX_MODEL_NAME", CODEX_MODEL_NAME)
    set_env_if_value("CODEX_PROXY", CODEX_PROXY)
    set_env_if_value("CODEX_TIMEOUT", CODEX_TIMEOUT)
    set_env_if_value("CODEX_REASONING_EFFORT", CODEX_REASONING_EFFORT)

    from run_paper_agent import run_paper_agent

    if not PAPER_PDF.exists():
        raise FileNotFoundError(f"PAPER_PDF does not exist: {PAPER_PDF}")
    if SUPP_PDF is not None and not Path(SUPP_PDF).exists():
        raise FileNotFoundError(f"SUPP_PDF does not exist: {SUPP_PDF}")

    result = run_paper_agent(
        paper_pdf=Path(PAPER_PDF),
        supp_pdf=Path(SUPP_PDF) if SUPP_PDF else None,
        out_root=Path(OUT_ROOT),
        paper_id=PAPER_ID,
        overwrite=OVERWRITE,
        keep_work=KEEP_WORK,
        steps=RUN_STEPS,
    )

    workspace = Path(result.get("workspace", OUT_ROOT / PAPER_ID))
    print("\nPaper2SOH Agent finished.")
    print("Selected steps:", RUN_STEPS)
    print("Status:", result.get("agent_status"))
    print("Workspace:", workspace)
    print("Primary case report:", workspace / "reports" / "paper_case_report.md")
    print("Final result:", workspace / "final" / "final_result.json")
    print("Final summary:", workspace / "final" / "final_summary.md")
    print("Agent trace:", workspace / "logs" / "agent_trace.jsonl")


if __name__ == "__main__":
    main()
