# GEPA Prompt Optimization Experiments

**Hypotheses** (scored by the CrewAI judge on held-out BRG examples):

1. **H1 — GEPA improves the base model:** `gepa_small` overall score **>** `baseline_small` (same `qwen3.5:4b`, seed vs GEPA prompt).
2. **H2 — GEPA closes/beats the larger model:** `gepa_small` overall score **≥** `baseline_large` (`qwen3.5:9b` with seed prompt).

Stack: **UV**, **Ollama**, **GEPA**, **CrewAI** (JSONC agents/tasks), **MLflow**.

## Prerequisites

1. [UV](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com/) installed
2. Pull models:

```bash
ollama pull qwen3.5:4b
ollama pull qwen3.5:9b
```

3. Install the project:

```bash
uv sync
```

## Configuration

| Path | Purpose |
|------|---------|
| [`config/experiment.jsonc`](config/experiment.jsonc) | Model tags, temperatures, GEPA budget, MLflow, paths |
| [`config/generator/`](config/generator/) | Generator `crew.jsonc` + `agents/*.jsonc` |
| [`config/judge/`](config/judge/) | Judge `crew.jsonc` + `agents/*.jsonc` |

GEPA evolves the generator agent **`backstory`** field. Edit JSONC to change roles, tasks, or rubrics without code changes.

**Override precedence:** CLI flags > `experiment.jsonc` > code defaults.

### CLI model tags

```bash
# Optimize BOTH small and large with GEPA reflection (default)
uv run gepa-optimize

# Or one slot at a time
uv run gepa-optimize --model small
uv run gepa-optimize --model large

# Override tags
uv run gepa-optimize --model both \
  --small-model qwen3.5:4b --large-model qwen3.5:9b \
  --reflection-model qwen3.5:4b

# Evaluate with explicit tags
uv run gepa-evaluate \
  --small-model qwen3.5:4b \
  --large-model qwen3.5:9b \
  --judge-model qwen3.5:9b
```

Other useful flags: `--max-metric-calls`, `--temperature`, `--judge-temperature`, `--config`, `--generator-crew`, `--judge-crew`, `--arms`.

## Run the experiment

```bash
# 1) Synthetic dataset (~24 examples: 12 train / 6 val / 6 test)
uv run gepa-prepare-data

# 2) GEPA optimize prompts (separately per task model)
uv run gepa-optimize --model small --max-metric-calls 30
uv run gepa-optimize --model large --max-metric-calls 30

# 3) Held-out evaluate: baselines + GEPA arms, judged by 9B
uv run gepa-evaluate

# 4) Inspect MLflow
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Each run writes a timestamped folder under `outputs/`:

```
outputs/optimize_qwen3.5_4b_<timestamp>/
  progress.log              # full CLI progress mirror
  seed_prompt.txt
  evaluation_scores.jsonl   # per-example scores during GEPA
  candidates/000/prompt.txt + score.json
  final/prompt.txt          # winning prompt
  final/scores.json
  scores_summary.json
  summary.json

outputs/evaluate_<timestamp>/
  progress.log
  arms/<arm>/prompt.txt
  arms/<arm>/scores.json
  arms/<arm>/examples/<id>.json
  summary.json
```

Compat copies also land in `artifacts/optimized_prompt_<model>.json` and `artifacts/evaluation_results.json`.

## Experiment arms

| Arm | Model | Prompt |
|-----|-------|--------|
| `baseline_small` | `qwen3.5:4b` | seed JSONC backstory |
| `baseline_large` | `qwen3.5:9b` | seed JSONC backstory |
| `gepa_small` | `qwen3.5:4b` | GEPA-optimized |
| `gepa_large` | `qwen3.5:9b` | GEPA-optimized |

Logged to MLflow / `outputs/.../hypotheses.json`:

- H1: `delta.gepa_small_minus_baseline_small` (**> 0** supports)
- H2: `delta.gepa_small_minus_baseline_large` (**≥ 0** supports)

## How scoring works

- **During GEPA:** deterministic structural/checklist score + textual feedback (ASI) drives reflection.
- **On the test set:** CrewAI judge scores correctness, completeness, clarity, testability, specificity, grounding, relevance, structural (0–1); `overall` is the mean of those dimensions.

## Notes

- **Qwen3.5 thinking:** `llm.disable_thinking` is `true` by default in `experiment.jsonc`. Without it, LiteLLM/CrewAI often get empty `content` from Ollama for these models.
- Raise `--max-metric-calls` (e.g. 50–150) for fuller GEPA searches; `30` is a smoke-friendly default.
- Keep train/val/test splits as written by `gepa-prepare-data`; do not use the test set during optimization.

```
config/           # experiment + CrewAI JSONC
data/             # synthetic BRG dataset
src/gepa_experiments/
  crews/          # JSONC → CrewAI builder
  optimize.py     # GEPA + CrewAI adapter
  evaluate.py     # multi-arm judge evaluation
  scripts/        # CLI entry points
artifacts/        # optimized prompts + eval results
mlruns/           # MLflow tracking store
```
