from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TaskConfig:
    labels: list[str]
    model_path: str
    classifier: str = "linear"  # 'linear' or 'lightgbm'
    threshold: float | None = None
    temperature: float | None = None


@dataclass
class AppConfig:
    # You can change to stronger Chinese embedding models, e.g., "BAAI/bge-large-zh-v1.5"
    embedding_model: str = "richinfoai/ritrieve_zh_v1"
    device: str = "auto"
    embed_batch_size: int = 32
    preprocess: bool = True
    tasks: dict[str, TaskConfig] = field(default_factory=dict)

    @staticmethod
    def from_dict(data: dict) -> AppConfig:
        tasks_cfg = {name: TaskConfig(**cfg) for name, cfg in (data.get("tasks") or {}).items()}
        return AppConfig(
            embedding_model=data.get("embedding_model") or "richinfoai/ritrieve_zh_v1",
            device=data.get("device") or "auto",
            embed_batch_size=int(data.get("embed_batch_size", 32)),
            preprocess=bool(data.get("preprocess", True)),
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
        "滑坡": TaskConfig(labels=["无", "有"], model_path=str(Path("models") / "slippery_slope.joblib")),
        "引战": TaskConfig(labels=["无", "有"], model_path=str(Path("models") / "incitement.joblib")),
        "拉踩": TaskConfig(labels=["无", "有"], model_path=str(Path("models") / "unfair_comparison.joblib")),
        "AI生成": TaskConfig(labels=["否", "是"], model_path=str(Path("models") / "ai_generation.joblib")),
    }
    return AppConfig(tasks=default_tasks)
