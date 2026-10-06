"""Per-run output directories and concise colorful CLI progress."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def safe_tag(model_tag: str) -> str:
    return model_tag.replace(":", "_").replace("/", "_")


class _Style:
    """Minimal ANSI colors; disabled when not a TTY or NO_COLOR is set."""

    def __init__(self) -> None:
        self.enabled = sys.stdout.isatty() and not os.environ.get("NO_COLOR")

    def _c(self, code: str, text: str) -> str:
        if not self.enabled:
            return text
        return f"\033[{code}m{text}\033[0m"

    def bold(self, t: str) -> str:
        return self._c("1", t)

    def dim(self, t: str) -> str:
        return self._c("2", t)

    def cyan(self, t: str) -> str:
        return self._c("36", t)

    def green(self, t: str) -> str:
        return self._c("32", t)

    def yellow(self, t: str) -> str:
        return self._c("33", t)

    def red(self, t: str) -> str:
        return self._c("31", t)

    def magenta(self, t: str) -> str:
        return self._c("35", t)

    def blue(self, t: str) -> str:
        return self._c("34", t)

    def score(self, value: float, *, digits: int = 2) -> str:
        text = f"{value:.{digits}f}"
        if value >= 0.85:
            return self.green(text)
        if value >= 0.6:
            return self.yellow(text)
        return self.red(text)


style = _Style()


class RunOutput:
    """
    Timestamped run directory with full file logs + concise colorful CLI.

    File (`progress.log`, jsonl, candidates/) keeps full detail.
    Terminal shows short status lines only.
    """

    def __init__(self, root: Path, kind: str, label: str = "") -> None:
        stamp = utc_timestamp()
        name = f"{kind}_{label}_{stamp}" if label else f"{kind}_{stamp}"
        self.root = root / name
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "candidates").mkdir(exist_ok=True)
        (self.root / "final").mkdir(exist_ok=True)
        (self.root / "arms").mkdir(exist_ok=True)
        self.progress_path = self.root / "progress.log"
        self.scores_jsonl = self.root / "evaluation_scores.jsonl"
        self._scores_fh = self.scores_jsonl.open("a", encoding="utf-8")
        self.file_log(f"Run directory: {self.root}")
        self.cli(
            f"{style.cyan('●')} {style.bold(kind)} "
            f"{style.dim('→')} {self.root.name}"
        )

    def file_log(self, message: str) -> None:
        """Append plain detail to progress.log only (no CLI)."""
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
        with self.progress_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def file_block(self, title: str, body: str, *, max_chars: int = 4000) -> None:
        self.file_log(f"--- {title} ---")
        text = body if len(body) <= max_chars else body[:max_chars] + "\n… [truncated]"
        with self.progress_path.open("a", encoding="utf-8") as fh:
            for row in text.splitlines() or [""]:
                fh.write(f"  {row}\n")

    def cli(self, message: str) -> None:
        """Print a colorful line and mirror (plain) to the log file."""
        print(message, flush=True)
        self.file_log(_strip_ansi(message))

    def cli_block(self, title: str, body: str, *, max_chars: int | None = None) -> None:
        """Print a titled multi-line block to CLI + progress.log."""
        self.cli(f"{style.cyan('┌')} {style.bold(title)}")
        text = body if max_chars is None or len(body) <= max_chars else body[:max_chars] + "\n… [truncated]"
        for row in text.splitlines() or [""]:
            line = f"{style.cyan('│')} {row}"
            print(line, flush=True)
            self.file_log(f"| {row}")
        self.cli(f"{style.cyan('└')}")

    # Back-compat aliases used by older call sites
    def log(self, message: str) -> None:
        self.file_log(message)

    def log_block(self, title: str, body: str, *, max_chars: int = 2000) -> None:
        self.file_block(title, body, max_chars=max_chars)

    def write_json(self, relative: str, data: Any) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return path

    def write_text(self, relative: str, text: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def append_score_row(self, row: dict[str, Any]) -> None:
        self._scores_fh.write(json.dumps(row, default=str) + "\n")
        self._scores_fh.flush()

    def close(self) -> None:
        self._scores_fh.close()

    def write_candidate(
        self,
        index: int,
        prompt: str,
        *,
        score: float | None = None,
        extras: dict[str, Any] | None = None,
    ) -> Path:
        folder = self.root / "candidates" / f"{index:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "prompt.txt").write_text(prompt, encoding="utf-8")
        payload = {"index": index, "score": score, **(extras or {})}
        (folder / "score.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return folder

    def write_final_prompt(
        self,
        prompt: str,
        *,
        score: float | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.write_text("final/prompt.txt", prompt)
        payload = {"system_prompt": prompt, "score": score, **(meta or {})}
        self.write_json("final/prompt.json", payload)
        self.write_json(
            "final/scores.json",
            {"best_score": score, **(meta or {})},
        )
        self.file_block("Final prompt", prompt, max_chars=8000)
        score_txt = style.score(score) if score is not None else style.dim("n/a")
        self.cli(
            f"{style.green('✓')} final prompt  score={score_txt}  "
            f"{style.dim('→ final/prompt.txt')}"
        )
        self.cli_block("Final prompt", prompt)


def _strip_ansi(text: str) -> str:
    import re

    return re.sub(r"\033\[[0-9;]*m", "", text)


def create_optimize_run(outputs_dir: Path, task_model: str) -> RunOutput:
    return RunOutput(outputs_dir, kind="optimize", label=safe_tag(task_model))


def create_evaluate_run(outputs_dir: Path) -> RunOutput:
    return RunOutput(outputs_dir, kind="evaluate")


def find_latest_optimized_prompt(outputs_dir: Path, model_tag: str) -> str | None:
    """Find final/prompt.txt from the newest optimize run for a model tag."""
    tag = safe_tag(model_tag)
    if not outputs_dir.exists():
        return None
    runs = sorted(
        [
            p
            for p in outputs_dir.iterdir()
            if p.is_dir() and p.name.startswith(f"optimize_{tag}_")
        ],
        key=lambda p: p.name,
        reverse=True,
    )
    for run in runs:
        prompt_file = run / "final" / "prompt.txt"
        if prompt_file.exists():
            return prompt_file.read_text(encoding="utf-8")
        prompt_json = run / "final" / "prompt.json"
        if prompt_json.exists():
            data = json.loads(prompt_json.read_text(encoding="utf-8"))
            return data.get("system_prompt")
    return None
