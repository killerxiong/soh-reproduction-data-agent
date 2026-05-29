"""
PyCharm one-click runner for a batch of papers.

Expected default structure:
raw_data/
  paper1/PDF/*.pdf
  paper2/PDF/*.pdf
  ...

Step control is the same as run_local_paper.py:
- RUN_STEPS = "all"
- RUN_STEPS = "step1"
- RUN_STEPS = "step1:step3"
- RUN_STEPS = "step4,final"

Right click this file in PyCharm -> Run 'run_local_batch'.
"""
from __future__ import annotations

import os
from pathlib import Path


# =========================
# PyCharm configuration area
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "raw_data"
OUT_ROOT = PROJECT_ROOT / "outputs"

RUN_STEPS = "all"
OVERWRITE = False
KEEP_WORK = True

# Optional local API configuration.
# Recommended: leave secrets empty here and set them in PyCharm Environment variables.
MINERU_TOKEN = ""
CODEX_API_KEY = ""
OPENAI_API_KEY = ""
CODEX_BASE_URL = ""
CODEX_MODEL_NAME = ""
CODEX_PROXY = ""
CODEX_TIMEOUT = ""
CODEX_REASONING_EFFORT = ""


def set_env_if_value(name: str, value: object) -> None:
    value = str(value or "").strip()
    if value:
        os.environ[name] = value


def main() -> None:
    set_env_if_value("MINERU_TOKEN", MINERU_TOKEN)
    set_env_if_value("CODEX_API_KEY", CODEX_API_KEY)
    set_env_if_value("OPENAI_API_KEY", OPENAI_API_KEY)
    set_env_if_value("CODEX_BASE_URL", CODEX_BASE_URL)
    set_env_if_value("CODEX_MODEL_NAME", CODEX_MODEL_NAME)
    set_env_if_value("CODEX_PROXY", CODEX_PROXY)
    set_env_if_value("CODEX_TIMEOUT", CODEX_TIMEOUT)
    set_env_if_value("CODEX_REASONING_EFFORT", CODEX_REASONING_EFFORT)

    from run_batch_agent import run_batch_agent

    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(f"RAW_DATA_DIR does not exist: {RAW_DATA_DIR}")

    summary = run_batch_agent(
        raw_data_dir=Path(RAW_DATA_DIR),
        out_root=Path(OUT_ROOT),
        overwrite=OVERWRITE,
        keep_work=KEEP_WORK,
        steps=RUN_STEPS,
    )

    print("\nPaper2SOH batch finished.")
    print("Selected steps:", RUN_STEPS)
    print("Output root:", OUT_ROOT)
    print("Batch summary:", OUT_ROOT / "batch_summary.json")
    print("Success:", summary.get("success"), "/", summary.get("total"))


if __name__ == "__main__":
    main()
