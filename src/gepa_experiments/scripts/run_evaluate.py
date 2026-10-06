"""CLI: evaluate baseline and GEPA arms with CrewAI judge."""

from __future__ import annotations

import argparse

from gepa_experiments.cli import add_common_args, config_from_args
from gepa_experiments.evaluate import evaluate_arms
from gepa_experiments.metrics import JUDGE_DIMENSIONS
from gepa_experiments.run_io import style


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate BRG generator arms with CrewAI judge (stronger model)"
    )
    add_common_args(parser)
    parser.add_argument(
        "--arms",
        default=None,
        help="Comma-separated arms to run "
        "(baseline_small,baseline_large,gepa_small,gepa_large "
        "or aliases baseline_4b,baseline_9b,gepa_4b,gepa_9b)",
    )
    parser.add_argument(
        "--no-optimize-missing",
        action="store_true",
        help=(
            "Do not auto-run GEPA when an optimized prompt is missing. "
            "By default, gepa-evaluate runs gepa-optimize --model small if needed."
        ),
    )
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    arm_list = [a.strip() for a in args.arms.split(",")] if args.arms else None

    results = evaluate_arms(
        cfg,
        arms=arm_list,
        optimize_missing=not args.no_optimize_missing,
    )

    print(f"\n{style.bold('summary — complete metrics')}")
    for arm, stats in (results.get("arms") or {}).items():
        print(f"  {style.bold(arm)}")
        print(
            f"    overall={style.score(stats['mean_overall'])}  "
            f"checklist={style.score(stats['mean_checklist'])}"
        )
        mid = (len(JUDGE_DIMENSIONS) + 1) // 2
        print(
            "    "
            + "  ".join(
                f"{k}={style.score(stats[f'mean_{k}'])}"
                for k in JUDGE_DIMENSIONS[:mid]
            )
        )
        print(
            "    "
            + "  ".join(
                f"{k}={style.score(stats[f'mean_{k}'])}"
                for k in JUDGE_DIMENSIONS[mid:]
            )
        )
        print(style.dim(f"    model={stats['model']}  prompt={stats['prompt_source']}"))

    skipped = results.get("skipped_gepa") or []
    if skipped:
        print(f"\n{style.bold('skipped GEPA arms')}")
        for skip in skipped:
            print(
                f"  {style.yellow('⚠')} {style.bold(skip['arm'])}  "
                f"{style.dim(skip['reason'])}"
            )

    print(f"\n{style.bold('hypotheses')}")
    print(
        style.dim(
            "  H1: GEPA improves the base/small model vs its seed prompt\n"
            "  H2: GEPA-optimized small model matches or beats the larger baseline"
        )
    )
    hypotheses = results.get("hypotheses") or {}
    if not hypotheses:
        print(
            style.dim(
                "  (no H1/H2 — need gepa_small metrics; run gepa-optimize for models.small first)"
            )
        )
    for hyp in hypotheses.values():
        badge = style.green("supported") if hyp.get("supported") else style.red("not supported")
        scores = hyp.get("scores") or {}
        score_bits = "  ".join(
            f"{k}={style.score(float(v))}" for k, v in scores.items()
        )
        print(
            f"  {style.bold(hyp['id'])}  {hyp['claim']}\n"
            f"    {score_bits}  Δ={style.score(float(hyp['delta_overall']), digits=3)}  "
            f"{badge}"
        )
    summary = results.get("hypotheses_summary")
    if summary:
        badge = style.green("YES") if summary.get("both_supported") else style.red("NO")
        print(f"  {style.bold('H1∧H2')} both supported?  {badge}")

    print(style.dim(f"\noutputs  {results.get('run_dir')}"))


if __name__ == "__main__":
    main()
