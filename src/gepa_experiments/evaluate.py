"""Held-out evaluation of baseline vs GEPA-optimized arms with CrewAI judge."""

from __future__ import annotations

import json
from typing import Any

import mlflow

from gepa_experiments.config_loader import ExperimentConfig
from gepa_experiments.crews.builder import get_seed_backstory, run_generator, run_judge
from gepa_experiments.data import load_dataset, split_dataset
from gepa_experiments.metrics import (
    JUDGE_DIMENSIONS,
    JUDGE_KEYS,
    parse_judge_scores,
    score_with_feedback,
)
from gepa_experiments.mlflow_utils import log_json_artifact, log_params_flat, setup_mlflow
from gepa_experiments.run_io import (
    create_evaluate_run,
    find_latest_optimized_prompt,
    safe_tag,
    style,
)


def _load_optimized_prompt(cfg: ExperimentConfig, model_tag: str) -> str | None:
    # Prefer newest per-run output
    from_run = find_latest_optimized_prompt(cfg.outputs_dir(), model_tag)
    if from_run:
        return from_run

    tag = safe_tag(model_tag)
    path = cfg.artifacts_dir() / f"optimized_prompt_{tag}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    candidate = data.get("best_candidate") or {}
    return candidate.get("system_prompt") or next(iter(candidate.values()), None)


def _arm_specs(
    cfg: ExperimentConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Build eval arms. Returns (specs, skipped_gepa) when optimized prompts are missing."""
    small = cfg.models.small
    large = cfg.models.large
    seed = get_seed_backstory(cfg.generator_crew_dir())

    specs = [
        {"arm": "baseline_small", "model": small, "prompt": seed, "prompt_source": "seed"},
        {"arm": "baseline_large", "model": large, "prompt": seed, "prompt_source": "seed"},
    ]
    skipped: list[dict[str, str]] = []

    opt_small = _load_optimized_prompt(cfg, small)
    opt_large = _load_optimized_prompt(cfg, large)
    if opt_small:
        specs.append(
            {
                "arm": "gepa_small",
                "model": small,
                "prompt": opt_small,
                "prompt_source": "gepa",
            }
        )
    else:
        skipped.append(
            {
                "arm": "gepa_small",
                "model": small,
                "reason": (
                    f"no optimized prompt for {small} "
                    f"(run: uv run gepa-optimize --model small)"
                ),
            }
        )
    if opt_large:
        specs.append(
            {
                "arm": "gepa_large",
                "model": large,
                "prompt": opt_large,
                "prompt_source": "gepa",
            }
        )
    else:
        skipped.append(
            {
                "arm": "gepa_large",
                "model": large,
                "reason": (
                    f"no optimized prompt for {large} "
                    f"(run: uv run gepa-optimize --model large)"
                ),
            }
        )
    return specs, skipped


def _normalize_arms(arms: list[str] | None) -> set[str] | None:
    if not arms:
        return None
    alias = {
        "baseline_4b": "baseline_small",
        "baseline_9b": "baseline_large",
        "gepa_4b": "gepa_small",
        "gepa_9b": "gepa_large",
    }
    return {alias.get(a, a) for a in arms}


def _ensure_gepa_prompts(
    cfg: ExperimentConfig,
    *,
    wanted: set[str] | None,
    optimize_missing: bool,
) -> list[dict[str, str]]:
    """
    Run GEPA optimize for missing prompts needed by evaluation.

    By default, gepa_small is required (H1/H2). gepa_large is optimized only when
    explicitly requested via --arms.
    """
    from gepa_experiments.optimize import run_optimization

    need_small = wanted is None or "gepa_small" in wanted
    need_large = wanted is not None and "gepa_large" in wanted
    ran: list[dict[str, str]] = []

    if not optimize_missing:
        return ran

    if need_small and _load_optimized_prompt(cfg, cfg.models.small) is None:
        print(
            f"{style.cyan('●')} GEPA optimize (missing gepa_small)  "
            f"task={style.bold(cfg.models.small)}  "
            f"reflect={style.bold(cfg.reflection_for('small'))}"
        )
        out = run_optimization(
            cfg,
            task_model=cfg.models.small,
            slot="small",
            reflection_model=cfg.reflection_for("small"),
        )
        ran.append(
            {
                "arm": "gepa_small",
                "model": cfg.models.small,
                "run_dir": str(out.get("run_dir") or ""),
            }
        )

    if need_large and _load_optimized_prompt(cfg, cfg.models.large) is None:
        print(
            f"{style.cyan('●')} GEPA optimize (missing gepa_large)  "
            f"task={style.bold(cfg.models.large)}  "
            f"reflect={style.bold(cfg.reflection_for('large'))}"
        )
        out = run_optimization(
            cfg,
            task_model=cfg.models.large,
            slot="large",
            reflection_model=cfg.reflection_for("large"),
        )
        ran.append(
            {
                "arm": "gepa_large",
                "model": cfg.models.large,
                "run_dir": str(out.get("run_dir") or ""),
            }
        )

    return ran


def evaluate_arms(
    cfg: ExperimentConfig,
    *,
    arms: list[str] | None = None,
    optimize_missing: bool = True,
) -> dict[str, Any]:
    setup_mlflow(cfg)
    dataset = load_dataset(cfg.dataset_path())
    _train, _val, test = split_dataset(dataset)
    if not test:
        raise RuntimeError("No test split found in dataset.")

    wanted = _normalize_arms(arms)
    optimized_now = _ensure_gepa_prompts(
        cfg, wanted=wanted, optimize_missing=optimize_missing
    )

    outputs_root = cfg.outputs_dir()
    outputs_root.mkdir(parents=True, exist_ok=True)
    run = create_evaluate_run(outputs_root)

    specs, skipped_gepa = _arm_specs(cfg)
    if wanted is not None:
        specs = [s for s in specs if s["arm"] in wanted]
        skipped_gepa = [s for s in skipped_gepa if s["arm"] in wanted]

    # gepa_small is required for the experiment hypotheses unless explicitly omitted
    missing_required = [
        s for s in skipped_gepa if s["arm"] == "gepa_small" and (wanted is None or "gepa_small" in wanted)
    ]
    if missing_required:
        reason = missing_required[0]["reason"]
        raise RuntimeError(
            f"Cannot evaluate hypotheses without gepa_small: {reason}. "
            "Re-run without --no-optimize-missing, or run gepa-optimize --model small first."
        )

    run.write_json(
        "meta.json",
        {
            "kind": "evaluate",
            "small_model": cfg.models.small,
            "large_model": cfg.models.large,
            "judge_model": cfg.models.judge,
            "num_test": len(test),
            "arms": [s["arm"] for s in specs],
            "skipped_gepa": skipped_gepa,
            "optimized_before_eval": optimized_now,
            "config": cfg.to_log_dict(),
        },
    )
    run.file_log(
        f"Starting evaluation | small={cfg.models.small} large={cfg.models.large} "
        f"judge={cfg.models.judge} | test={len(test)} | arms={[s['arm'] for s in specs]}"
        + (f" | skipped={[s['arm'] for s in skipped_gepa]}" if skipped_gepa else "")
        + (
            f" | optimized_now={[s['arm'] for s in optimized_now]}"
            if optimized_now
            else ""
        )
    )
    run.cli(
        f"{style.cyan('●')} evaluate  {style.dim(f'test={len(test)}')}"
    )
    run.cli(
        f"  {style.dim('models')}  "
        f"small={style.bold(cfg.models.small)}  "
        f"large={style.bold(cfg.models.large)}  "
        f"judge={style.bold(cfg.models.judge)}"
    )
    run.cli(f"  arms: {style.bold(', '.join(s['arm'] for s in specs))}")
    for done in optimized_now:
        run.cli(
            f"  {style.green('✓')} optimized {style.bold(done['arm'])}  "
            f"{style.dim(done.get('run_dir') or done['model'])}"
        )
    for skip in skipped_gepa:
        run.cli(
            f"  {style.yellow('⚠')} skipped {style.bold(skip['arm'])}  "
            f"{style.dim(skip['reason'])}"
        )
    run.cli(
        style.dim(
            "  H1: GEPA improves base/small vs seed  |  "
            "H2: GEPA-small matches/beats larger baseline"
        )
    )

    results: dict[str, Any] = {
        "arms": {},
        "per_example": [],
        "run_dir": str(run.root),
        "skipped_gepa": skipped_gepa,
        "optimized_before_eval": optimized_now,
    }

    try:
        with mlflow.start_run(run_name="evaluate"):
            log_params_flat(cfg.to_log_dict())
            log_params_flat(
                {
                    "judge_model": cfg.models.judge,
                    "small_model": cfg.models.small,
                    "large_model": cfg.models.large,
                    "num_test": len(test),
                    "run_dir": str(run.root),
                }
            )

            for spec in specs:
                arm = spec["arm"]
                run.file_log(
                    f"=== Arm {arm} | model={spec['model']} | source={spec['prompt_source']} ==="
                )
                run.file_block(f"{arm} prompt", spec["prompt"], max_chars=8000)
                run.cli(
                    f"{style.blue('▸')} {style.bold(arm)}  "
                    + style.dim(f"source={spec['prompt_source']}")
                )
                run.cli(
                    f"  {style.dim('models')}  "
                    f"generator={style.bold(spec['model'])}  "
                    f"judge={style.bold(cfg.models.judge)}"
                )
                prompt_title = (
                    f"{arm} prompt (seed / baseline)"
                    if spec["prompt_source"] == "seed"
                    else f"{arm} prompt (GEPA-mutated)"
                )
                run.cli_block(prompt_title, spec["prompt"])
                run.cli(
                    f"  {style.yellow('⚖')} judging generator outputs for this prompt with "
                    f"{style.bold(cfg.models.judge)}"
                )
                arm_dir = f"arms/{arm}"
                run.write_text(f"{arm_dir}/prompt.txt", spec["prompt"])
                run.write_json(
                    f"{arm_dir}/prompt.json",
                    {
                        "arm": arm,
                        "model": spec["model"],
                        "prompt_source": spec["prompt_source"],
                        "system_prompt": spec["prompt"],
                    },
                )

                arm_scores: list[float] = []
                checklist_scores: list[float] = []
                dim_sums = {k: 0.0 for k in JUDGE_KEYS}

                for ex_i, ex in enumerate(test):
                    ex_id = ex.get("id", f"ex{ex_i}")
                    run.file_log(f"[{arm}] {ex_i + 1}/{len(test)} id={ex_id} generating…")
                    run.cli(
                        f"  {style.cyan('→')} {style.bold(str(ex_id))}  "
                        f"{ex_i + 1}/{len(test)}  "
                        + style.dim(
                            f"models generator={spec['model']}  judge={cfg.models.judge}"
                        )
                    )
                    generated = run_generator(
                        cfg.generator_crew_dir(),
                        ex["brief"],
                        model_tag=spec["model"],
                        base_url=cfg.ollama_base_url,
                        temperature=cfg.llm.temperature,
                        max_tokens=cfg.llm.max_tokens,
                        prompt_override=spec["prompt"],
                        disable_thinking=cfg.llm.disable_thinking,
                    )
                    checklist, feedback = score_with_feedback(
                        generated, ex.get("checklist", []), ex.get("gold_requirements", "")
                    )
                    checklist_scores.append(checklist)
                    run.file_log(
                        f"[{arm}] id={ex_id} checklist={checklist:.4f} — {feedback}"
                    )

                    run.file_log(f"[{arm}] id={ex_id} judging with {cfg.models.judge}…")
                    raw_judge = run_judge(
                        cfg.judge_crew_dir(),
                        brief=ex["brief"],
                        gold_requirements=ex.get("gold_requirements", ""),
                        candidate_output=generated,
                        model_tag=cfg.models.judge,
                        base_url=cfg.ollama_base_url,
                        temperature=cfg.llm.judge_temperature,
                        max_tokens=cfg.llm.max_tokens,
                        disable_thinking=cfg.llm.disable_thinking,
                    )
                    judged = parse_judge_scores(raw_judge)
                    arm_scores.append(judged["overall"])
                    for k in dim_sums:
                        dim_sums[k] += float(judged[k])

                    example_record = {
                        "arm": arm,
                        "example_id": ex_id,
                        "checklist": checklist,
                        "checklist_feedback": feedback,
                        "judge": {k: judged[k] for k in JUDGE_KEYS},
                        "rationale": judged.get("rationale", ""),
                        "output": generated,
                        "brief": ex["brief"],
                    }
                    results["per_example"].append(
                        {
                            **example_record,
                            "output_preview": generated[:500],
                            "output": generated[:2000],
                        }
                    )
                    run.write_json(f"{arm_dir}/examples/{ex_id}.json", example_record)
                    run.append_score_row(
                        {
                            "arm": arm,
                            "example_id": ex_id,
                            "checklist": checklist,
                            **{k: judged[k] for k in JUDGE_KEYS},
                        }
                    )
                    run.file_log(
                        f"[{arm}] id={ex_id} judge overall={judged['overall']:.4f} "
                        f"rationale={judged.get('rationale', '')}"
                    )
                    run.cli(
                        f"  {style.bold(str(ex_id))}  "
                        f"checklist={style.score(checklist)}  "
                        f"overall={style.score(judged['overall'])}"
                    )
                    mid = (len(JUDGE_DIMENSIONS) + 1) // 2
                    run.cli(
                        "    "
                        + "  ".join(
                            f"{k}={style.score(judged[k])}"
                            for k in JUDGE_DIMENSIONS[:mid]
                        )
                    )
                    run.cli(
                        "    "
                        + "  ".join(
                            f"{k}={style.score(judged[k])}"
                            for k in JUDGE_DIMENSIONS[mid:]
                        )
                    )
                    run.cli(f"    {style.dim('checklist feedback:')} {feedback}")
                    if judged.get("rationale"):
                        run.cli(f"    {style.dim('judge rationale:')} {judged['rationale']}")
                    elif judged["overall"] == 0.0:
                        run.cli(
                            f"    {style.red('judge parse/empty response — raw:')} "
                            f"{style.dim(raw_judge[:300])}"
                        )

                n = max(len(test), 1)
                arm_summary = {
                    "model": spec["model"],
                    "prompt_source": spec["prompt_source"],
                    "mean_checklist": sum(checklist_scores) / n,
                    **{f"mean_{k}": dim_sums[k] / n for k in JUDGE_KEYS},
                }
                results["arms"][arm] = arm_summary
                run.write_json(f"{arm_dir}/scores.json", arm_summary)
                run.file_log(
                    f"[{arm}] DONE mean_overall={arm_summary['mean_overall']:.4f} "
                    f"mean_checklist={arm_summary['mean_checklist']:.4f}"
                )
                run.cli(
                    f"{style.green('✓')} {style.bold(arm)} metrics  "
                    f"overall={style.score(arm_summary['mean_overall'])}  "
                    f"checklist={style.score(arm_summary['mean_checklist'])}"
                )
                mid = (len(JUDGE_DIMENSIONS) + 1) // 2
                run.cli(
                    "    "
                    + "  ".join(
                        f"mean_{k}={style.score(arm_summary[f'mean_{k}'])}"
                        for k in JUDGE_DIMENSIONS[:mid]
                    )
                )
                run.cli(
                    "    "
                    + "  ".join(
                        f"mean_{k}={style.score(arm_summary[f'mean_{k}'])}"
                        for k in JUDGE_DIMENSIONS[mid:]
                    )
                )

                for key, value in arm_summary.items():
                    if isinstance(value, float):
                        mlflow.log_metric(f"{arm}.{key}", value)

            arms_out = results["arms"]
            hypotheses = _build_hypotheses(arms_out)
            results["hypotheses"] = hypotheses
            # Legacy single-key for older readers (H2)
            if "h2_gepa_small_vs_baseline_large" in hypotheses:
                results["hypothesis"] = hypotheses["h2_gepa_small_vs_baseline_large"]

            run.cli(f"{style.magenta('◆')} {style.bold('hypotheses')}")
            for key, hyp in hypotheses.items():
                mlflow.log_metric(hyp["mlflow_metric"], float(hyp["delta_overall"]))
                for dim, delta in (hyp.get("delta_dimensions") or {}).items():
                    mlflow.log_metric(f"{hyp['mlflow_metric']}.{dim}", float(delta))
                run.file_log(
                    f"{key}: {hyp['claim']} | Δ_overall={hyp['delta_overall']:.4f} | "
                    f"supported={hyp['supported']}"
                )
                badge = (
                    style.green("supported")
                    if hyp["supported"]
                    else style.red("not supported")
                )
                run.cli(
                    f"  {style.bold(hyp['id'])}  {hyp['claim']}  "
                    f"Δ_overall={style.score(float(hyp['delta_overall']), digits=3)}  "
                    f"{badge}"
                )
                if hyp.get("delta_dimensions"):
                    dims = list(hyp["delta_dimensions"].items())
                    mid = (len(dims) + 1) // 2
                    run.cli(
                        "    "
                        + "  ".join(
                            f"Δ_{k}={style.score(v, digits=3)}" for k, v in dims[:mid]
                        )
                    )
                    run.cli(
                        "    "
                        + "  ".join(
                            f"Δ_{k}={style.score(v, digits=3)}" for k, v in dims[mid:]
                        )
                    )

            if (
                "h1_gepa_improves_baseline_small" in hypotheses
                and "h2_gepa_small_vs_baseline_large" in hypotheses
            ):
                both = (
                    hypotheses["h1_gepa_improves_baseline_small"]["supported"]
                    and hypotheses["h2_gepa_small_vs_baseline_large"]["supported"]
                )
                results["hypotheses_summary"] = {
                    "both_supported": bool(both),
                    "h1_supported": hypotheses["h1_gepa_improves_baseline_small"][
                        "supported"
                    ],
                    "h2_supported": hypotheses["h2_gepa_small_vs_baseline_large"][
                        "supported"
                    ],
                }
                badge = style.green("YES") if both else style.red("NO")
                run.cli(f"  {style.bold('H1∧H2')} both supported?  {badge}")

            run.write_json("summary.json", results)
            run.write_json("scores_summary.json", results.get("arms", {}))
            run.write_json("hypotheses.json", hypotheses)
            log_json_artifact(results, "evaluation_results.json", cfg.artifacts_dir())

        out_path = cfg.artifacts_dir() / "evaluation_results.json"
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        run.cli(f"{style.green('✓')} done  {style.dim(str(run.root))}")
        return results
    finally:
        run.close()


def _dimension_deltas(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    """Per-judge-dimension mean deltas: left - right."""
    out: dict[str, float] = {}
    for dim in JUDGE_DIMENSIONS:
        lk, rk = f"mean_{dim}", f"mean_{dim}"
        if lk in left and rk in right:
            out[dim] = float(left[lk]) - float(right[rk])
    if "mean_checklist" in left and "mean_checklist" in right:
        out["checklist"] = float(left["mean_checklist"]) - float(right["mean_checklist"])
    return out


def _build_hypotheses(arms_out: dict[str, Any]) -> dict[str, Any]:
    """
    H1: GEPA on the small/base model improves its metric scores vs seed baseline.
    H2: GEPA-optimized small model matches or beats the larger baseline model.
    """
    hypotheses: dict[str, Any] = {}

    if "gepa_small" in arms_out and "baseline_small" in arms_out:
        delta = (
            float(arms_out["gepa_small"]["mean_overall"])
            - float(arms_out["baseline_small"]["mean_overall"])
        )
        hypotheses["h1_gepa_improves_baseline_small"] = {
            "id": "H1",
            "claim": "gepa_small > baseline_small (GEPA improves the base/small model)",
            "comparison": "gepa_small - baseline_small",
            "delta_overall": delta,
            "delta_dimensions": _dimension_deltas(
                arms_out["gepa_small"], arms_out["baseline_small"]
            ),
            "supported": delta > 0,
            "mlflow_metric": "delta.gepa_small_minus_baseline_small",
            "scores": {
                "gepa_small": arms_out["gepa_small"]["mean_overall"],
                "baseline_small": arms_out["baseline_small"]["mean_overall"],
            },
        }

    if "gepa_small" in arms_out and "baseline_large" in arms_out:
        delta = (
            float(arms_out["gepa_small"]["mean_overall"])
            - float(arms_out["baseline_large"]["mean_overall"])
        )
        hypotheses["h2_gepa_small_vs_baseline_large"] = {
            "id": "H2",
            "claim": "gepa_small >= baseline_large (GEPA small matches/beats larger model)",
            "comparison": "gepa_small - baseline_large",
            "delta_overall": delta,
            "delta_dimensions": _dimension_deltas(
                arms_out["gepa_small"], arms_out["baseline_large"]
            ),
            "supported": delta >= 0,
            "mlflow_metric": "delta.gepa_small_minus_baseline_large",
            "scores": {
                "gepa_small": arms_out["gepa_small"]["mean_overall"],
                "baseline_large": arms_out["baseline_large"]["mean_overall"],
            },
        }

    return hypotheses
