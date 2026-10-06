"""Shared CLI argument helpers for optimize/evaluate scripts."""

from __future__ import annotations

import argparse

from gepa_experiments.config_loader import DEFAULT_CONFIG_PATH, load_experiment_config


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to experiment.jsonc",
    )
    parser.add_argument(
        "--task-model",
        default=None,
        help="Ollama tag for the generator task model (e.g. qwen3.5:4b)",
    )
    parser.add_argument(
        "--reflection-model",
        default=None,
        help="Ollama tag for GEPA reflection LM (e.g. qwen3.5:9b)",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Ollama tag for the judge agent (e.g. qwen3.5:9b)",
    )
    parser.add_argument(
        "--small-model",
        default=None,
        help="Ollama tag treated as the small model arm (default: qwen3.5:4b)",
    )
    parser.add_argument(
        "--large-model",
        default=None,
        help="Ollama tag treated as the large model arm (default: qwen3.5:9b)",
    )
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=None,
        help="GEPA evaluation budget",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Generator LLM temperature",
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=None,
        help="Judge LLM temperature",
    )
    parser.add_argument(
        "--generator-crew",
        default=None,
        help="Directory containing generator crew.jsonc + agents/",
    )
    parser.add_argument(
        "--judge-crew",
        default=None,
        help="Directory containing judge crew.jsonc + agents/",
    )
    parser.add_argument(
        "--ollama-base-url",
        default=None,
        help="Ollama base URL",
    )


def config_from_args(args: argparse.Namespace):
    return load_experiment_config(
        config_path=args.config,
        task_model=getattr(args, "task_model", None),
        reflection_model=getattr(args, "reflection_model", None),
        judge_model=getattr(args, "judge_model", None),
        small_model=getattr(args, "small_model", None),
        large_model=getattr(args, "large_model", None),
        max_metric_calls=getattr(args, "max_metric_calls", None),
        temperature=getattr(args, "temperature", None),
        judge_temperature=getattr(args, "judge_temperature", None),
        generator_crew=getattr(args, "generator_crew", None),
        judge_crew=getattr(args, "judge_crew", None),
        ollama_base_url=getattr(args, "ollama_base_url", None),
    )
