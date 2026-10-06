"""GEPA prompt optimization experiments for business requirement generation."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

__version__ = "0.1.0"

# Load .env from project root; disable interactive CrewAI tracing by default.
_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_ROOT / ".env")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
