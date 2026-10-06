"""MLflow helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow

from gepa_experiments.config_loader import ExperimentConfig


def resolve_mlflow_tracking_uri(cfg: ExperimentConfig) -> str:
    from gepa_experiments.config_loader import PROJECT_ROOT

    tracking = cfg.mlflow.tracking_uri
    if tracking.startswith("sqlite:///"):
        rest = tracking.removeprefix("sqlite:///")
        if rest and not rest.startswith("/") and "://" not in rest:
            abs_path = (PROJECT_ROOT / rest).resolve()
            return f"sqlite:///{abs_path}"
    if tracking.startswith("./") or tracking.startswith("../"):
        return str((PROJECT_ROOT / tracking).resolve())
    return tracking


def setup_mlflow(cfg: ExperimentConfig) -> None:
    tracking = resolve_mlflow_tracking_uri(cfg)
    mlflow.set_tracking_uri(tracking)
    mlflow.set_experiment(cfg.mlflow.experiment_name)


def log_params_flat(params: dict[str, Any], prefix: str = "") -> None:
    for key, value in params.items():
        name = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            log_params_flat(value, prefix=f"{name}.")
        else:
            mlflow.log_param(name, value)


def log_json_artifact(data: Any, filename: str, artifacts_dir: Path) -> Path:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / filename
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    mlflow.log_artifact(str(path))
    return path
