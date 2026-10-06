"""Build CrewAI crews from JSONC config directories with runtime overrides."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from crewai import Agent, Crew, LLM, Process, Task

from gepa_experiments.config_loader import load_jsonc


def _settings_from_agent(agent_cfg: dict[str, Any]) -> dict[str, Any]:
    settings = dict(agent_cfg.get("settings") or {})
    for key in ("verbose", "allow_delegation", "max_iter", "max_rpm", "memory", "cache"):
        if key in agent_cfg and key not in settings:
            settings[key] = agent_cfg[key]
    return settings


def load_crew_bundle(crew_dir: str | Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load crew.jsonc and agents/*.jsonc from a crew directory."""
    crew_dir = Path(crew_dir)
    crew_path = crew_dir / "crew.jsonc"
    if not crew_path.exists():
        crew_path = crew_dir / "crew.json"
    crew_cfg = load_jsonc(crew_path)

    agents_dir = crew_dir / "agents"
    agents: dict[str, dict[str, Any]] = {}
    for name in crew_cfg.get("agents", []):
        jsonc = agents_dir / f"{name}.jsonc"
        json_path = agents_dir / f"{name}.json"
        path = jsonc if jsonc.exists() else json_path
        if not path.exists():
            raise FileNotFoundError(f"Agent config not found: {jsonc} or {json_path}")
        agents[name] = load_jsonc(path)
    return crew_cfg, agents


def get_seed_backstory(crew_dir: str | Path, agent_name: str | None = None) -> str:
    _, agents = load_crew_bundle(crew_dir)
    if agent_name is None:
        agent_name = next(iter(agents))
    return str(agents[agent_name].get("backstory", ""))


def build_llm(
    model_tag: str,
    *,
    base_url: str,
    temperature: float,
    max_tokens: int,
    disable_thinking: bool = True,
) -> LLM:
    tag = model_tag.removeprefix("ollama/")
    kwargs: dict[str, Any] = {}
    if disable_thinking:
        # Qwen3.5 (and similar) thinking models return empty content via LiteLLM
        # unless thinking is disabled for Ollama.
        kwargs["extra_body"] = {"think": False}
    return LLM(
        model=f"ollama/{tag}",
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )


def build_crew_from_jsonc(
    crew_dir: str | Path,
    *,
    llm: LLM,
    prompt_override: str | None = None,
    evolvable_agent: str | None = None,
    evolvable_field: str = "backstory",
) -> Crew:
    """
    Construct a CrewAI Crew from JSONC configs.

    prompt_override replaces `evolvable_field` on the first (or named) agent —
    used by GEPA candidate evaluation and gepa_* evaluate arms.
    """
    crew_cfg, agents_cfg = load_crew_bundle(crew_dir)
    agents_cfg = deepcopy(agents_cfg)

    target = evolvable_agent or next(iter(agents_cfg))
    if prompt_override is not None:
        if target not in agents_cfg:
            raise KeyError(f"Unknown agent '{target}' in {crew_dir}")
        agents_cfg[target][evolvable_field] = prompt_override

    agents: dict[str, Agent] = {}
    for name, cfg in agents_cfg.items():
        settings = _settings_from_agent(cfg)
        agents[name] = Agent(
            role=cfg.get("role", name),
            goal=cfg.get("goal", ""),
            backstory=cfg.get("backstory", ""),
            llm=llm,
            verbose=bool(settings.get("verbose", False)),
            allow_delegation=bool(settings.get("allow_delegation", False)),
            max_iter=int(settings.get("max_iter", 5)),
        )

    process_name = str(crew_cfg.get("process", "sequential")).lower()
    process = Process.hierarchical if process_name == "hierarchical" else Process.sequential

    tasks: list[Task] = []
    task_by_name: dict[str, Task] = {}
    for task_cfg in crew_cfg.get("tasks", []):
        agent_name = task_cfg["agent"]
        if agent_name not in agents:
            raise KeyError(f"Task agent '{agent_name}' not found in agents")
        context_names = task_cfg.get("context") or []
        context = [task_by_name[n] for n in context_names if n in task_by_name]
        task = Task(
            description=task_cfg["description"],
            expected_output=task_cfg["expected_output"],
            agent=agents[agent_name],
            context=context or None,
        )
        tasks.append(task)
        if "name" in task_cfg:
            task_by_name[task_cfg["name"]] = task

    return Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=process,
        verbose=bool(crew_cfg.get("verbose", False)),
    )


def run_generator(
    crew_dir: str | Path,
    brief: str,
    *,
    model_tag: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    prompt_override: str | None = None,
    disable_thinking: bool = True,
) -> str:
    llm = build_llm(
        model_tag,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        disable_thinking=disable_thinking,
    )
    crew = build_crew_from_jsonc(crew_dir, llm=llm, prompt_override=prompt_override)
    result = crew.kickoff(inputs={"brief": brief})
    return str(result)


def run_judge(
    crew_dir: str | Path,
    *,
    brief: str,
    gold_requirements: str,
    candidate_output: str,
    model_tag: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    disable_thinking: bool = True,
) -> str:
    llm = build_llm(
        model_tag,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        disable_thinking=disable_thinking,
    )
    crew = build_crew_from_jsonc(crew_dir, llm=llm)
    result = crew.kickoff(
        inputs={
            "brief": brief,
            "gold_requirements": gold_requirements,
            "candidate_output": candidate_output,
        }
    )
    return str(result)
