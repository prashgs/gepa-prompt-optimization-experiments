"""GEPA optimization of CrewAI generator backstory prompts."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import gepa
from gepa import EvaluationBatch, GEPAAdapter
from gepa.logging.logger import StdOutLogger

from gepa_experiments.config_loader import ExperimentConfig
from gepa_experiments.crews.builder import get_seed_backstory, run_generator, run_judge
from gepa_experiments.data import load_dataset, split_dataset, to_gepa_examples
from gepa_experiments.metrics import JUDGE_DIMENSIONS, parse_judge_scores, score_with_feedback
from gepa_experiments.mlflow_utils import (
    log_json_artifact,
    log_params_flat,
    resolve_mlflow_tracking_uri,
    setup_mlflow,
)
from gepa_experiments.run_io import RunOutput, create_optimize_run, safe_tag, style


class _FileOnlyLogger(StdOutLogger):
    """GEPA logger: full detail to progress.log, quiet on the CLI."""

    def __init__(self, run: RunOutput) -> None:
        self._run = run

    def log(self, message: str) -> None:  # type: ignore[override]
        self._run.file_log(f"[gepa] {message}")


class CrewAIBRGAdapter(GEPAAdapter[dict[str, Any], dict[str, Any], dict[str, Any]]):
    """Evaluate candidate backstories by running the JSONC CrewAI generator crew."""

    COMPONENT = "system_prompt"

    def __init__(
        self,
        cfg: ExperimentConfig,
        task_model: str,
        run: RunOutput | None = None,
        reflection_model: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.task_model = task_model
        self.reflection_model = reflection_model or cfg.models.reflection
        self.crew_dir = cfg.generator_crew_dir()
        self.run = run
        self._eval_batch_idx = 0
        self._seen_prompts: set[str] = set()
        self._prompt_versions: dict[str, int] = {}

    def _models_line(self, *, include_reflection: bool = False) -> str:
        parts = [
            f"task={style.bold(self.task_model)}",
        ]
        if self.cfg.gepa.use_judge:
            parts.append(f"judge={style.bold(self.cfg.models.judge)}")
        if include_reflection:
            parts.append(f"reflection={style.bold(self.reflection_model)}")
        return style.dim("models ") + "  ".join(parts)

    def _score_example(
        self,
        *,
        generated: str,
        brief: str,
        gold: str,
        checklist: list[str],
    ) -> tuple[float, str, dict[str, Any]]:
        """Checklist (+ optional judge) → GEPA score, feedback, detail dict."""
        checklist_score, checklist_fb = score_with_feedback(generated, checklist, gold)
        detail: dict[str, Any] = {
            "checklist": checklist_score,
            "checklist_feedback": checklist_fb,
        }

        if not self.cfg.gepa.use_judge:
            return checklist_score, checklist_fb, detail

        try:
            raw_judge = run_judge(
                self.cfg.judge_crew_dir(),
                brief=brief,
                gold_requirements=gold,
                candidate_output=generated,
                model_tag=self.cfg.models.judge,
                base_url=self.cfg.ollama_base_url,
                temperature=self.cfg.llm.judge_temperature,
                max_tokens=self.cfg.llm.max_tokens,
                disable_thinking=self.cfg.llm.disable_thinking,
            )
            judged = parse_judge_scores(raw_judge)
        except Exception as exc:  # noqa: BLE001
            judged = {k: 0.0 for k in ("overall", *JUDGE_DIMENSIONS)}
            judged["rationale"] = f"Judge failed: {exc}"
            raw_judge = ""

        detail["judge"] = {k: judged[k] for k in ("overall", *JUDGE_DIMENSIONS)}
        detail["rationale"] = judged.get("rationale", "")
        detail["judge_raw_preview"] = (raw_judge or "")[:500]

        w = max(0.0, min(1.0, float(self.cfg.gepa.checklist_weight)))
        score = w * checklist_score + (1.0 - w) * float(judged["overall"])
        dim_txt = ", ".join(f"{k}={judged[k]:.2f}" for k in JUDGE_DIMENSIONS)
        feedback = (
            f"checklist={checklist_score:.2f}; judge_overall={judged['overall']:.2f}; "
            f"{dim_txt}. {checklist_fb} "
            f"Rationale: {judged.get('rationale', '')}".strip()
        )
        return score, feedback, detail

    def _prompt_label(self, prompt: str) -> str:
        if prompt not in self._prompt_versions:
            self._prompt_versions[prompt] = len(self._prompt_versions)
        idx = self._prompt_versions[prompt]
        return "seed" if idx == 0 else f"mutated-{idx:03d}"

    def evaluate(
        self,
        batch: list[dict[str, Any]],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[dict[str, Any], dict[str, Any]]:
        prompt = candidate.get(self.COMPONENT) or next(iter(candidate.values()))
        self._eval_batch_idx += 1
        batch_id = self._eval_batch_idx
        label = self._prompt_label(prompt)
        is_new = prompt not in self._seen_prompts
        self._seen_prompts.add(prompt)

        if self.run:
            self.run.file_log(
                f"[eval #{batch_id}] candidate prompt ({len(prompt)} chars), "
                f"batch_size={len(batch)}, capture_traces={capture_traces}, label={label}"
            )
            self.run.file_block(f"eval #{batch_id} prompt", prompt, max_chars=8000)
            source = "seed" if label == "seed" else "gepa"
            self.run.cli(
                f"{style.blue('▸')} {style.bold(label)}  "
                + style.dim(f"source={source}")
            )
            self.run.cli(
                f"  {style.dim('models')}  "
                f"generator={style.bold(self.task_model)}"
                + (
                    f"  judge={style.bold(self.cfg.models.judge)}"
                    if self.cfg.gepa.use_judge
                    else ""
                )
            )
            if is_new:
                title = (
                    f"{label} prompt (seed / baseline)"
                    if label == "seed"
                    else f"{label} prompt (GEPA-mutated)"
                )
                self.run.cli_block(title, prompt)
                if self.cfg.gepa.use_judge:
                    self.run.cli(
                        f"  {style.yellow('⚖')} judging generator outputs for this prompt with "
                        f"{style.bold(self.cfg.models.judge)}"
                    )

        outputs: list[dict[str, Any]] = []
        scores: list[float] = []
        trajectories: list[dict[str, Any]] | None = [] if capture_traces else None

        for i, example in enumerate(batch):
            brief = example.get("brief") or example.get("input") or ""
            example_id = example.get("id") or example.get("additional_context", {}).get(
                "id", f"ex{i}"
            )
            checklist = example.get("checklist") or example.get("additional_context", {}).get(
                "checklist", []
            )
            gold = example.get("gold_requirements") or example.get("answer") or ""
            if self.run:
                self.run.file_log(
                    f"[eval #{batch_id}] example {i + 1}/{len(batch)} id={example_id} "
                    f"generating with {self.task_model}…"
                )
                self.run.cli(
                    f"  {style.cyan('→')} {style.bold(str(example_id))}  "
                    f"{i + 1}/{len(batch)}  "
                    + style.dim(
                        f"models generator={self.task_model}"
                        + (
                            f"  judge={self.cfg.models.judge}"
                            if self.cfg.gepa.use_judge
                            else ""
                        )
                    )
                )
            detail: dict[str, Any] = {}
            try:
                generated = run_generator(
                    self.crew_dir,
                    brief,
                    model_tag=self.task_model,
                    base_url=self.cfg.ollama_base_url,
                    temperature=self.cfg.llm.temperature,
                    max_tokens=self.cfg.llm.max_tokens,
                    prompt_override=prompt,
                    disable_thinking=self.cfg.llm.disable_thinking,
                )
                score, feedback, detail = self._score_example(
                    generated=generated,
                    brief=brief,
                    gold=gold,
                    checklist=checklist,
                )
            except Exception as exc:  # noqa: BLE001 — per-example failures must not abort GEPA
                generated = ""
                score = 0.0
                feedback = f"Generation failed: {exc}"
                detail = {"checklist": 0.0, "checklist_feedback": feedback}

            if self.run:
                self.run.file_log(
                    f"[eval #{batch_id}] id={example_id} score={score:.4f} "
                    f"output_chars={len(generated)} feedback={feedback}"
                )
                checklist_val = float(detail.get("checklist", 0.0))
                judge = detail.get("judge") or {}
                if judge:
                    self.run.cli(
                        f"  {style.bold(str(example_id))}  "
                        f"checklist={style.score(checklist_val)}  "
                        f"overall={style.score(float(judge.get('overall', 0.0)))}"
                    )
                    mid = (len(JUDGE_DIMENSIONS) + 1) // 2
                    self.run.cli(
                        "    "
                        + "  ".join(
                            f"{k}={style.score(float(judge.get(k, 0.0)))}"
                            for k in JUDGE_DIMENSIONS[:mid]
                        )
                    )
                    self.run.cli(
                        "    "
                        + "  ".join(
                            f"{k}={style.score(float(judge.get(k, 0.0)))}"
                            for k in JUDGE_DIMENSIONS[mid:]
                        )
                    )
                    self.run.cli(
                        f"    {style.dim('checklist feedback:')} "
                        f"{detail.get('checklist_feedback', feedback)}"
                    )
                    if detail.get("rationale"):
                        self.run.cli(
                            f"    {style.dim('judge rationale:')} {detail['rationale']}"
                        )
                    elif float(judge.get("overall", 0.0)) == 0.0:
                        raw = str(detail.get("judge_raw_preview") or "")
                        self.run.cli(
                            f"    {style.red('judge parse/empty response — raw:')} "
                            f"{style.dim(raw[:300])}"
                        )
                else:
                    self.run.cli(
                        f"  {style.bold(str(example_id))}  "
                        f"checklist={style.score(checklist_val)}"
                    )
                    self.run.cli(
                        f"    {style.dim('checklist feedback:')} "
                        f"{detail.get('checklist_feedback', feedback)}"
                    )
                self.run.append_score_row(
                    {
                        "batch": batch_id,
                        "example_id": example_id,
                        "score": score,
                        "feedback": feedback,
                        "prompt_label": label,
                        "output_chars": len(generated),
                        "brief_preview": brief[:200],
                        "output_preview": generated[:500],
                        **detail,
                    }
                )

            outputs.append({"full_assistant_response": generated})
            scores.append(score)
            if trajectories is not None:
                trajectories.append(
                    {
                        "data": example,
                        "full_assistant_response": generated,
                        "feedback": feedback,
                    }
                )

        if self.run and scores:
            mean = sum(scores) / len(scores)
            self.run.file_log(
                f"[eval #{batch_id}] batch mean score={mean:.4f} "
                f"(min={min(scores):.4f}, max={max(scores):.4f})"
            )
            self.run.cli(
                f"{style.green('✓')} {style.bold(label)} metrics  "
                f"mean={style.score(mean)}  "
                f"min={style.score(min(scores))}  max={style.score(max(scores))}  "
                f"n={len(scores)}"
            )

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[dict[str, Any], dict[str, Any]],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        assert len(components_to_update) == 1
        comp = components_to_update[0]
        trajectories = eval_batch.trajectories
        assert trajectories is not None

        items: list[dict[str, Any]] = []
        for traj in trajectories:
            data = traj["data"]
            brief = data.get("brief") or data.get("input") or ""
            items.append(
                {
                    "Inputs": brief,
                    "Generated Outputs": traj["full_assistant_response"],
                    "Feedback": traj["feedback"],
                }
            )
        if not items:
            raise RuntimeError("No trajectories available for reflection.")
        if self.run:
            parent = candidate.get(self.COMPONENT) or next(iter(candidate.values()), "")
            label = self._prompt_label(parent)
            self.run.file_log(
                f"[reflection] building reflective dataset for '{comp}' "
                f"with {len(items)} examples from {label}"
            )
            self.run.cli(
                f"{style.magenta('✧')} reflect on {style.bold(label)}  "
                f"{style.dim(f'{len(items)} examples → propose mutated prompt')}"
            )
            self.run.cli(f"  {self._models_line(include_reflection=True)}")
            for i, item in enumerate(items, start=1):
                self.run.cli(
                    f"  {style.dim(f'ref[{i}]')}  {style.dim(str(item['Feedback']))}"
                )
        return {comp: items}


def _write_gepa_candidates(run: RunOutput, result: Any) -> None:
    """Persist every candidate prompt + val score explored by GEPA."""
    for idx, candidate in enumerate(result.candidates):
        prompt = candidate.get(CrewAIBRGAdapter.COMPONENT) or next(
            iter(candidate.values()), ""
        )
        score = (
            float(result.val_aggregate_scores[idx])
            if idx < len(result.val_aggregate_scores)
            else None
        )
        parents = result.parents[idx] if idx < len(result.parents) else []
        run.write_candidate(
            idx,
            prompt,
            score=score,
            extras={
                "parents": parents,
                "is_best": idx == result.best_idx,
            },
        )
        run.file_log(f"Candidate {idx:03d}: val_score={score} parents={parents}")
        mark = style.green("★") if idx == result.best_idx else style.dim("·")
        score_txt = style.score(score) if score is not None else style.dim("n/a")
        kind = "seed" if idx == 0 else f"mutated-{idx:03d}"
        run.cli(
            f"  {mark} {style.bold(kind)}  val={score_txt}  "
            f"{style.dim(f'parents={parents}')}"
        )
        run.cli_block(f"Candidate {kind} prompt", prompt)

def run_optimization(
    cfg: ExperimentConfig,
    task_model: str | None = None,
    *,
    slot: str = "task",
    reflection_model: str | None = None,
) -> dict[str, Any]:
    """Run GEPA for a task model and persist the best prompt artifact."""
    task_model = task_model or cfg.task_model_for(slot)
    reflection = reflection_model or cfg.reflection_for(slot)
    setup_mlflow(cfg)

    outputs_root = cfg.outputs_dir()
    outputs_root.mkdir(parents=True, exist_ok=True)
    run = create_optimize_run(outputs_root, task_model)

    dataset_path = cfg.dataset_path()
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Run: uv run gepa-prepare-data"
        )

    examples = load_dataset(dataset_path)
    train, val, _test = split_dataset(examples)
    trainset = to_gepa_examples(train)
    valset = to_gepa_examples(val)

    seed_backstory = get_seed_backstory(cfg.generator_crew_dir())
    seed_candidate = {CrewAIBRGAdapter.COMPONENT: seed_backstory}

    run.write_json(
        "meta.json",
        {
            "kind": "optimize",
            "slot": slot,
            "task_model": task_model,
            "reflection_model": reflection,
            "max_metric_calls": cfg.gepa.max_metric_calls,
            "train_size": len(trainset),
            "val_size": len(valset),
            "config": cfg.to_log_dict(),
        },
    )
    run.write_text("seed_prompt.txt", seed_backstory)
    run.file_log(
        f"Starting GEPA optimize | slot={slot} | task={task_model} | "
        f"reflection={reflection} | max_metric_calls={cfg.gepa.max_metric_calls} | "
        f"train={len(trainset)} val={len(valset)}"
    )
    run.file_block("Seed prompt", seed_backstory)
    run.cli(
        f"{style.cyan('●')} optimize {style.bold(slot)}  "
        f"{style.dim(f'budget={cfg.gepa.max_metric_calls}  '
                     f'train={len(trainset)} val={len(valset)}')}"
    )
    run.cli(
        f"  {style.dim('models')}  "
        f"task={style.bold(task_model)}  "
        f"reflection={style.bold(reflection)}"
        + (
            f"  judge={style.bold(cfg.models.judge)}"
            if cfg.gepa.use_judge
            else ""
        )
    )
    adapter = CrewAIBRGAdapter(
        cfg,
        task_model=task_model,
        run=run,
        reflection_model=reflection,
    )
    artifacts = cfg.artifacts_dir()
    artifacts.mkdir(parents=True, exist_ok=True)
    tag = safe_tag(task_model)
    # GEPA internal checkpoint dir inside our run folder
    gepa_dir = run.root / "gepa_engine"
    gepa_dir.mkdir(parents=True, exist_ok=True)

    reflection_lm = cfg.ollama_model_id(reflection)
    reflection_kwargs: dict[str, Any] = {
        "api_base": cfg.ollama_base_url,
    }
    if cfg.llm.disable_thinking:
        reflection_kwargs["extra_body"] = {"think": False}

    from gepa.lm import LM

    reflection_lm_client = LM(
        reflection_lm,
        temperature=1.0,
        max_tokens=cfg.llm.max_tokens,
        **reflection_kwargs,
    )

    try:
        with _MlflowRun(cfg, task_model):
            result = gepa.optimize(
                seed_candidate=seed_candidate,
                trainset=trainset,
                valset=valset,
                adapter=adapter,
                reflection_lm=reflection_lm_client,
                max_metric_calls=cfg.gepa.max_metric_calls,
                candidate_selection_strategy=cfg.gepa.candidate_selection_strategy,  # type: ignore[arg-type]
                display_progress_bar=cfg.gepa.display_progress_bar,
                use_mlflow=True,
                mlflow_tracking_uri=resolve_mlflow_tracking_uri(cfg),
                mlflow_experiment_name=cfg.mlflow.experiment_name,
                mlflow_attach_existing=True,
                run_dir=str(gepa_dir),
                logger=_FileOnlyLogger(run),
                seed=0,
            )

        best_prompt = result.best_candidate[CrewAIBRGAdapter.COMPONENT]
        best_score = float(result.val_aggregate_scores[result.best_idx])

        run.file_log(
            f"GEPA finished | candidates={result.num_candidates} | "
            f"best_idx={result.best_idx} | best_score={best_score:.4f} | "
            f"metric_calls={result.total_metric_calls}"
        )
        run.cli(f"{style.cyan('●')} candidates")
        _write_gepa_candidates(run, result)

        scores_table = [
            {
                "index": i,
                "val_score": float(result.val_aggregate_scores[i]),
                "parents": result.parents[i],
                "is_best": i == result.best_idx,
            }
            for i in range(len(result.candidates))
        ]
        run.write_json("scores_summary.json", scores_table)

        out = {
            "slot": slot,
            "task_model": task_model,
            "reflection_model": reflection,
            "best_score_val": best_score,
            "best_idx": result.best_idx,
            "total_metric_calls": result.total_metric_calls,
            "best_candidate": result.best_candidate,
            "seed_candidate": seed_candidate,
            "run_dir": str(run.root),
            "candidate_scores": scores_table,
        }

        run.write_final_prompt(
            best_prompt,
            score=best_score,
            meta={
                "best_idx": result.best_idx,
                "slot": slot,
                "task_model": task_model,
                "reflection_model": reflection,
                "total_metric_calls": result.total_metric_calls,
            },
        )
        run.write_json("summary.json", out)
        run.write_json("final/result.json", out)

        # Compat copy for evaluate loader
        prompt_path = artifacts / f"optimized_prompt_{tag}.json"
        prompt_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        run.file_log(f"Also wrote compat artifact: {prompt_path}")

        try:
            import mlflow

            if mlflow.active_run() is not None:
                log_params_flat(
                    {
                        "slot": slot,
                        "task_model": task_model,
                        "reflection_model": reflection,
                        "max_metric_calls": cfg.gepa.max_metric_calls,
                        "run_dir": str(run.root),
                    }
                )
                mlflow.log_metric("best_score_val", best_score)
                log_json_artifact(out, f"optimized_prompt_{tag}.json", artifacts)
        except Exception:  # noqa: BLE001
            pass

        run.cli(
            f"{style.green('✓')} done  best={style.score(best_score)}  "
            f"candidates={result.num_candidates}  "
            f"calls={result.total_metric_calls}  "
            f"{style.dim(str(run.root))}"
        )
        return out
    finally:
        run.close()


class _MlflowRun:
    """Start an outer MLflow run so GEPA can attach with mlflow_attach_existing."""

    def __init__(self, cfg: ExperimentConfig, task_model: str) -> None:
        self.cfg = cfg
        self.task_model = task_model
        self._run = None

    def __enter__(self):
        import mlflow

        setup_mlflow(self.cfg)
        self._run = mlflow.start_run(
            run_name=f"optimize_{self.task_model.replace(':', '_')}"
        )
        log_params_flat(self.cfg.to_log_dict())
        log_params_flat({"effective_task_model": self.task_model})
        return self._run

    def __exit__(self, *exc):
        import mlflow

        mlflow.end_run()
        return False
