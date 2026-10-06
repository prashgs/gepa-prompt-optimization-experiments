"""Scoring helpers: structural checklist metric for GEPA + judge JSON parsing."""

from __future__ import annotations

import json
import re
from typing import Any


REQUIRED_SECTIONS = (
    "user stories",
    "functional requirements",
    "non-functional requirements",
    "acceptance criteria",
)

JUDGE_KEYS = (
    "correctness",
    "completeness",
    "clarity",
    "testability",
    "specificity",
    "grounding",
    "relevance",
    "structural",
    "overall",
)

# Dimension keys only (excludes aggregate overall)
JUDGE_DIMENSIONS = tuple(k for k in JUDGE_KEYS if k != "overall")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def structural_score(output: str, checklist: list[str]) -> tuple[float, list[str]]:
    """
    Deterministic score in [0, 1] from section presence + checklist coverage.
    Returns (score, feedback_bullets).
    """
    text = _normalize(output)
    feedback: list[str] = []

    section_hits = 0
    # Treat non-functional variants as one bucket
    has_user = "user stor" in text
    has_functional = "functional" in text
    has_nfr = "non-functional" in text or "nonfunctional" in text or "non functional" in text
    has_ac = "acceptance" in text
    for label, hit in (
        ("User Stories", has_user),
        ("Functional Requirements", has_functional),
        ("Non-Functional Requirements", has_nfr),
        ("Acceptance Criteria", has_ac),
    ):
        if hit:
            section_hits += 1
        else:
            feedback.append(f"Missing or unclear section: {label}.")

    section_score = section_hits / 4.0

    if not checklist:
        checklist_score = 1.0
    else:
        hits = 0
        for item in checklist:
            tokens = [t for t in _normalize(item).split() if len(t) > 2]
            if not tokens:
                continue
            # Require majority of meaningful tokens to appear
            matched = sum(1 for t in tokens if t in text)
            if matched >= max(1, len(tokens) // 2):
                hits += 1
            else:
                feedback.append(f"Checklist topic under-covered: '{item}'.")
        checklist_score = hits / len(checklist)

    # Soft penalties for restating-only / very short outputs
    length_penalty = 0.0
    if len(output.strip()) < 200:
        length_penalty = 0.15
        feedback.append("Output is too short for a complete requirements document.")

    score = max(0.0, min(1.0, 0.45 * section_score + 0.55 * checklist_score - length_penalty))
    if score >= 0.85 and not feedback:
        feedback.append("Strong structure and checklist coverage.")
    return score, feedback


def score_with_feedback(
    output: str,
    checklist: list[str],
    gold_requirements: str = "",
) -> tuple[float, str]:
    score, bullets = structural_score(output, checklist)
    if gold_requirements and score < 0.7:
        bullets.append(
            "Compare against the gold reference: ensure acceptance criteria are measurable "
            "and requirements decompose the brief rather than restating it."
        )
    feedback = " ".join(bullets) if bullets else "Adequate requirements quality."
    return score, feedback


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of a JSON object from model output."""
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence:
        try:
            data = json.loads(fence.group(1))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            return None
    return None


def parse_judge_scores(raw: str) -> dict[str, Any]:
    """Parse judge JSON into normalized float scores + rationale."""
    data = extract_json_object(raw) or {}
    scores: dict[str, float] = {}
    for key in JUDGE_KEYS:
        try:
            scores[key] = float(data.get(key, 0.0))
        except (TypeError, ValueError):
            scores[key] = 0.0
        scores[key] = max(0.0, min(1.0, scores[key]))

    dims = [scores[k] for k in JUDGE_DIMENSIONS]
    if scores["overall"] == 0.0 and dims:
        scores["overall"] = sum(dims) / len(dims)

    return {
        **scores,
        "rationale": str(data.get("rationale", "")),
        "raw": raw,
    }
