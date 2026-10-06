"""JSONC loading and experiment configuration with CLI overrides."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import json5

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "experiment.jsonc"


def load_jsonc(path: str | Path) -> dict[str, Any]:
    """Load a JSONC file (comments + trailing commas allowed)."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    data = json5.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSONC object in {path}, got {type(data)}")
    return data


@dataclass
class ModelsConfig:
    small: str = "qwen3.5:4b"
    large: str = "qwen3.5:9b"
    task: str = "qwen3.5:4b"
    # Default GEPA reflection LM (used when per-slot override is unset)
    reflection: str = "qwen3.5:9b"
    # Reflection LMs used when optimizing the small / large task models
    reflection_small: str | None = None
    reflection_large: str | None = None
    judge: str = "qwen3.5:9b"


@dataclass
class LlmConfig:
    temperature: float = 0.2
    judge_temperature: float = 0.0
    max_tokens: int = 4096
    # Required for Qwen3.5 via Ollama+LiteLLM (otherwise content is empty)
    disable_thinking: bool = True


@dataclass
class GepaConfig:
    max_metric_calls: int = 30
    candidate_selection_strategy: str = "pareto"
    display_progress_bar: bool = True
    # When true, each GEPA eval (seed + mutated prompts) is also scored by the judge crew
    use_judge: bool = True
    # Blend: final = checklist_weight * checklist + (1 - checklist_weight) * judge.overall
    checklist_weight: float = 0.3


@dataclass
class MlflowConfig:
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "brg-gepa-qwen"


@dataclass
class PathsConfig:
    dataset: str = "data/brg_synthetic.json"
    generator_crew: str = "config/generator"
    judge_crew: str = "config/judge"
    artifacts_dir: str = "artifacts"
    outputs_dir: str = "outputs"


@dataclass
class ExperimentConfig:
    ollama_base_url: str = "http://localhost:11434"
    models: ModelsConfig = field(default_factory=ModelsConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    gepa: GepaConfig = field(default_factory=GepaConfig)
    mlflow: MlflowConfig = field(default_factory=MlflowConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    config_path: str = str(DEFAULT_CONFIG_PATH)

    def resolve_path(self, relative: str) -> Path:
        path = Path(relative)
        if path.is_absolute():
            return path
        return (PROJECT_ROOT / path).resolve()

    def dataset_path(self) -> Path:
        return self.resolve_path(self.paths.dataset)

    def generator_crew_dir(self) -> Path:
        return self.resolve_path(self.paths.generator_crew)

    def judge_crew_dir(self) -> Path:
        return self.resolve_path(self.paths.judge_crew)

    def artifacts_dir(self) -> Path:
        return self.resolve_path(self.paths.artifacts_dir)

    def outputs_dir(self) -> Path:
        return self.resolve_path(self.paths.outputs_dir)

    def to_log_dict(self) -> dict[str, Any]:
        return asdict(self)

    def ollama_model_id(self, tag: str) -> str:
        tag = tag.removeprefix("ollama/")
        return f"ollama/{tag}"

    def reflection_for(self, slot: str) -> str:
        """Resolve the reflection LM for a task slot: small | large | task."""
        if slot == "small":
            return self.models.reflection_small or self.models.reflection
        if slot == "large":
            return self.models.reflection_large or self.models.reflection
        return self.models.reflection

    def task_model_for(self, slot: str) -> str:
        if slot == "small":
            return self.models.small
        if slot == "large":
            return self.models.large
        return self.models.task


def _merge_dataclass(cls: type, data: dict[str, Any] | None) -> Any:
    if not data:
        return cls()
    valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in data.items() if k in valid})


def load_experiment_config(
    config_path: str | Path | None = None,
    *,
    task_model: str | None = None,
    reflection_model: str | None = None,
    judge_model: str | None = None,
    small_model: str | None = None,
    large_model: str | None = None,
    max_metric_calls: int | None = None,
    temperature: float | None = None,
    judge_temperature: float | None = None,
    generator_crew: str | None = None,
    judge_crew: str | None = None,
    ollama_base_url: str | None = None,
) -> ExperimentConfig:
    """Load experiment.jsonc and apply CLI overrides (CLI wins)."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()

    raw = load_jsonc(path) if path.exists() else {}

    cfg = ExperimentConfig(
        ollama_base_url=raw.get("ollama_base_url", ExperimentConfig.ollama_base_url),
        models=_merge_dataclass(ModelsConfig, raw.get("models")),
        llm=_merge_dataclass(LlmConfig, raw.get("llm")),
        gepa=_merge_dataclass(GepaConfig, raw.get("gepa")),
        mlflow=_merge_dataclass(MlflowConfig, raw.get("mlflow")),
        paths=_merge_dataclass(PathsConfig, raw.get("paths")),
        config_path=str(path),
    )

    # CLI --reflection-model overrides the shared default and both per-slot reflections
    reflection = reflection_model or cfg.models.reflection
    reflection_small = (
        reflection_model or cfg.models.reflection_small or reflection
    )
    reflection_large = (
        reflection_model or cfg.models.reflection_large or reflection
    )
    models = replace(
        cfg.models,
        task=task_model or cfg.models.task,
        reflection=reflection,
        reflection_small=reflection_small,
        reflection_large=reflection_large,
        judge=judge_model or cfg.models.judge,
        small=small_model or cfg.models.small,
        large=large_model or cfg.models.large,
    )
    llm = replace(
        cfg.llm,
        temperature=temperature if temperature is not None else cfg.llm.temperature,
        judge_temperature=(
            judge_temperature
            if judge_temperature is not None
            else cfg.llm.judge_temperature
        ),
    )
    gepa = replace(
        cfg.gepa,
        max_metric_calls=(
            max_metric_calls
            if max_metric_calls is not None
            else cfg.gepa.max_metric_calls
        ),
    )
    paths = replace(
        cfg.paths,
        generator_crew=generator_crew or cfg.paths.generator_crew,
        judge_crew=judge_crew or cfg.paths.judge_crew,
    )

    return replace(
        cfg,
        ollama_base_url=ollama_base_url or cfg.ollama_base_url,
        models=models,
        llm=llm,
        gepa=gepa,
        paths=paths,
    )
