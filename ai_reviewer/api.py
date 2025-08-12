from __future__ import annotations

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import AppConfig, TaskConfig, load_config
from .embeddings import EmbeddingBackend
from .tasks import TaskRegistry


class PredictRequest(BaseModel):
    texts: list[str]
    tasks: list[str] | None = Field(
        default=None,
        description="要预测的任务名称列表；为空时对所有注册的任务进行预测",
    )


class UpdateRequest(BaseModel):
    texts: list[str]
    task: str
    labels: list[str]


class RegisterTaskRequest(BaseModel):
    name: str
    labels: list[str]
    model_path: str | None = None


class ConfigResponse(BaseModel):
    embedding_model: str
    device: str
    tasks: dict[str, dict[str, object]]


def create_app(config_path: str | None = None) -> FastAPI:
    cfg: AppConfig = load_config(config_path)
    embedder = EmbeddingBackend(cfg.embedding_model, cfg.device)
    registry = TaskRegistry()

    for name, tcfg in cfg.tasks.items():
        registry.ensure(name=name, labels=tcfg.labels, model_path=tcfg.model_path, embedder_encode=embedder.encode)

    app = FastAPI()

    @app.get("/config", response_model=ConfigResponse)
    async def get_config():
        return ConfigResponse(
            embedding_model=cfg.embedding_model,
            device=embedder.device,
            tasks={
                name: {"labels": task.labels, "model_path": task.model_path} for name, task in registry._tasks.items()
            },
        )

    @app.get("/tasks")
    async def list_tasks():
        return {"tasks": registry.list()}

    @app.post("/tasks/register")
    async def register_task(req: RegisterTaskRequest):
        name = req.name
        labels = req.labels
        if len(labels) < 2:
            raise HTTPException(status_code=400, detail="labels 至少包含两个类别")
        model_path = (
            req.model_path
            or cfg.tasks.get(name, TaskConfig(labels=labels, model_path=f"models/{name}.joblib")).model_path
        )
        task = registry.ensure(name=name, labels=labels, model_path=model_path, embedder_encode=embedder.encode)
        # also persist in memory config (not writing file for simplicity)
        cfg.tasks[name] = TaskConfig(labels=labels, model_path=task.model_path)
        return {"message": "task registered", "task": {"name": name, "labels": labels, "model_path": model_path}}

    @app.post("/predict")
    async def predict(req: PredictRequest):
        if not req.texts:
            raise HTTPException(status_code=400, detail="texts 不能为空")
        target_tasks = req.tasks or registry.list()
        missing = [t for t in target_tasks if t not in registry._tasks]
        if missing:
            raise HTTPException(status_code=404, detail=f"未找到任务: {missing}")
        embs = embedder.encode(req.texts)
        results = []
        for i in range(len(req.texts)):
            item_res = {}
            for tname in target_tasks:
                task = registry.get(tname)
                probs = task.clf.predict_proba(embs[[i]])[0]
                idx = int(np.argmax(probs))
                item_res[tname] = {
                    "label": task.labels[idx],
                    "confidence": float(probs[idx]),
                }
            results.append(item_res)
        return {"results": results}

    @app.post("/update")
    async def update(req: UpdateRequest):
        if req.task not in registry._tasks:
            raise HTTPException(status_code=404, detail=f"未找到任务: {req.task}")
        task = registry.get(req.task)
        if not all(label in task.labels for label in req.labels):
            raise HTTPException(status_code=400, detail="labels 包含未注册的类别")
        embs = embedder.encode(req.texts)
        y = [task.labels.index(label) for label in req.labels]
        registry.update(req.task, embs, y)
        return {"message": f"task {req.task} updated"}

    return app
