from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TaskConfig:
    labels: list[str]
    model_path: str


@dataclass
class AppConfig:
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    device: str = "auto"
    tasks: dict[str, TaskConfig] = field(default_factory=dict)

    @staticmethod
    def from_dict(data: dict) -> AppConfig:
        tasks_cfg = {name: TaskConfig(**cfg) for name, cfg in (data.get("tasks") or {}).items()}
        return AppConfig(
            embedding_model=data.get("embedding_model") or "Qwen/Qwen3-Embedding-0.6B",
            device=data.get("device") or "auto",
            tasks=tasks_cfg,
        )


def load_config(config_path: str | None = None) -> AppConfig:
    path = (
        config_path
        or os.environ.get("AI_REVIEWER_CONFIG")
        or (Path("config.toml").resolve() if Path("config.toml").exists() else None)
    )
    if path and Path(path).exists():
        with Path(path).open("rb") as f:
            data = tomllib.load(f) or {}
        return AppConfig.from_dict(data)

    default_tasks = {
        "滑坡": TaskConfig(labels=["有", "无"], model_path=str(Path("models") / "slippery_slope.joblib")),
        "引战": TaskConfig(labels=["有", "无"], model_path=str(Path("models") / "incitement.joblib")),
        "拉踩": TaskConfig(labels=["有", "无"], model_path=str(Path("models") / "unfair_comparison.joblib")),
        "AI生成": TaskConfig(labels=["是", "否"], model_path=str(Path("models") / "ai_generation.joblib")),
    }
    return AppConfig(tasks=default_tasks)
