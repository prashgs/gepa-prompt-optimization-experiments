"""CLI: run GEPA prompt optimization for small and/or large task models."""

from __future__ import annotations

import argparse

from gepa_experiments.cli import add_common_args, config_from_args
from gepa_experiments.optimize import run_optimization
from gepa_experiments.run_io import style


def _print_result(result: dict) -> None:
    score = result.get("best_score_val")
    score_txt = style.score(float(score)) if score is not None else style.dim("n/a")
    slot = result.get("slot", "?")
    print(
        f"\n{style.bold(f'summary [{slot}]')}  "
        f"task={style.bold(str(result.get('task_model')))}  "
        f"reflection={style.bold(str(result.get('reflection_model')))}"
    )
    for row in result.get("candidate_scores") or []:
        mark = style.green("★") if row.get("is_best") else style.dim("·")
        kind = "seed" if row["index"] == 0 else f"mutated-{row['index']:03d}"
        parents = row.get("parents")
        print(
            f"  {mark} {kind}  val={style.score(float(row['val_score']))}  "
            + style.dim(f"parents={parents}")
        )
    print(
        f"  best={score_txt}  idx={result.get('best_idx')}  "
        f"calls={result.get('total_metric_calls')}"
    )
    print(style.dim(f"outputs  {result.get('run_dir')}"))
    print(style.dim(f"prompt   {result.get('run_dir')}/final/prompt.txt"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize CrewAI BRG generator prompts with GEPA. "
            "By default runs reflection for both small and large models."
        )
    )
    add_common_args(parser)
    parser.add_argument(
        "--model",
        choices=["small", "large", "task", "both"],
        default="both",
        help=(
            "Which model slot(s) to optimize with GEPA reflection. "
            "'both' (default) optimizes small then large. "
            "Overridden by --task-model when set (single run)."
        ),
    )
    args = parser.parse_args(argv)
    cfg = config_from_args(args)

    if args.task_model:
        # Honor explicit --task-model as a one-off task override
        results = [
            run_optimization(
                cfg,
                task_model=args.task_model,
                slot="task",
                reflection_model=cfg.reflection_for("task"),
            )
        ]
    elif args.model == "both":
        results = []
        for slot in ("small", "large"):
            print(
                f"\n{style.cyan('══')} GEPA reflection for "
                f"{style.bold(slot)} "
                f"({cfg.task_model_for(slot)}) "
                f"reflect={cfg.reflection_for(slot)}"
            )
            results.append(
                run_optimization(
                    cfg,
                    task_model=cfg.task_model_for(slot),
                    slot=slot,
                    reflection_model=cfg.reflection_for(slot),
                )
            )
    else:
        slot = args.model
        results = [
            run_optimization(
                cfg,
                task_model=cfg.task_model_for(slot),
                slot=slot,
                reflection_model=cfg.reflection_for(slot),
            )
        ]

    for result in results:
        _print_result(result)


if __name__ == "__main__":
    main()
