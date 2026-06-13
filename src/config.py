"""Runtime configuration for Paper2SOH Agent.

All secrets and environment-specific values should be provided through
environment variables. Do not hard-code API keys or local paths in source code.
"""

from __future__ import annotations

import os

# MinerU API config
MINERU_BASE_URL = os.getenv("MINERU_BASE_URL", "https://mineru.net")
MINERU_TOKEN = os.getenv("MINERU_TOKEN", "")

# Codex / LLM API config. Override these in your local environment if you use
# a custom gateway.
CODEX_BASE_URL = os.getenv("CODEX_BASE_URL", "https://sorryios.ai/codex")
CODEX_MODEL_NAME = os.getenv("CODEX_MODEL_NAME", "gpt-5.4")
CODEX_API_KEY = os.getenv("CODEX_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
CODEX_PROXY = os.getenv("CODEX_PROXY", "")
CODEX_TIMEOUT = int(os.getenv("CODEX_TIMEOUT", "600"))
CODEX_REASONING_EFFORT = os.getenv("CODEX_REASONING_EFFORT", "medium")

# Agent defaults
DEFAULT_MINERU_MODEL_VERSION = os.getenv("DEFAULT_MINERU_MODEL_VERSION", "vlm")
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "en")
DEFAULT_RANDOM_SEED = int(os.getenv("DEFAULT_RANDOM_SEED", "42"))
