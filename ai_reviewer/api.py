from __future__ import annotations

import json
from pathlib import Path

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
    sample_weight: list[float] | None = Field(default=None, description="可选样本权重，与texts等长")


class RegisterTaskRequest(BaseModel):
    name: str
    labels: list[str]
    model_path: str | None = None


class ConfigResponse(BaseModel):
    embedding_model: str
    device: str
    tasks: dict[str, dict[str, object]]


class EvalRequest(BaseModel):
    texts: list[str]
    task: str
    labels: list[str]


class EvalResponse(BaseModel):
    task: str
    size: int
    accuracy: float
    per_label: dict[str, dict[str, float]]
    macro_precision: float | None = None
    macro_recall: float | None = None
    macro_f1: float | None = None
    hardest: list[dict[str, object]] | None = None


class ProbsRequest(BaseModel):
    texts: list[str]
    task: str


class ProbsResponse(BaseModel):
    task: str
    labels: list[str]
    probs: list[list[float]]


class CalibrateRequest(BaseModel):
    task: str
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    temperature: float | None = Field(default=None, gt=0.0)


def create_app(config_path: str | None = None) -> FastAPI:
    cfg: AppConfig = load_config(config_path)
    embedder = EmbeddingBackend(
        cfg.embedding_model,
        cfg.device,
        batch_size=getattr(cfg, "embed_batch_size", 32),
        enable_preprocess=getattr(cfg, "preprocess", True),
    )
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
                name: {
                    "labels": task.labels,
                    "model_path": task.model_path,
                    "threshold": task.threshold,
                    "temperature": task.temperature,
                }
                for name, task in registry._tasks.items()
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
                # Ensure label index is within persisted mapping range
                if idx >= len(task.labels):
                    # fallback to the last label to avoid index error; better surface warning in logs
                    idx = min(idx, len(task.labels) - 1)
                label = task.labels[idx]
                # temperature scaling (if available)
                if task.temperature is not None and task.temperature > 0:
                    # convert probabilities back to logits via log, then resoftmax with temperature
                    # add small eps to avoid log(0)
                    eps = 1e-12
                    logits = np.log(np.maximum(probs, eps))
                    logits = logits / float(task.temperature)
                    expv = np.exp(logits - np.max(logits))
                    probs = expv / np.sum(expv)
                    idx = int(np.argmax(probs))
                    label = task.labels[idx]
                conf = float(float(probs[idx]) if idx < len(probs) else 0.0)
                # Binary threshold support: if exactly 2 labels, use task.threshold when available
                if len(task.labels) == 2 and task.threshold is not None:
                    pos_idx = 1  # by our initialization, class indices are 0..K-1 in label order
                    pos_prob = float(probs[pos_idx]) if pos_idx < len(probs) else 0.0
                    label = task.labels[1] if pos_prob >= task.threshold else task.labels[0]
                    conf = pos_prob if label == task.labels[1] else 1.0 - pos_prob
                item_res[tname] = {"label": label, "confidence": conf}
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
        weights = None
        if req.sample_weight is not None:
            if len(req.sample_weight) != len(req.texts):
                raise HTTPException(status_code=400, detail="sample_weight 长度需与 texts 相同")
            weights = req.sample_weight
        registry.update(req.task, embs, y, sample_weight=weights)
        return {"message": f"task {req.task} updated"}

    @app.post("/eval", response_model=EvalResponse)
    async def eval_task(req: EvalRequest):
        if req.task not in registry._tasks:
            raise HTTPException(status_code=404, detail=f"未找到任务: {req.task}")
        task = registry.get(req.task)
        if not req.texts or not req.labels or len(req.texts) != len(req.labels):
            raise HTTPException(status_code=400, detail="texts/labels 数量需一致且非空")
        if not all(lb in task.labels for lb in req.labels):
            raise HTTPException(status_code=400, detail="labels 包含未注册的类别")
        embs = embedder.encode(req.texts)
        probs = task.clf.predict_proba(embs)
        # apply temperature for evaluation consistency if set
        if task.temperature is not None and task.temperature > 0:
            eps = 1e-12
            logits = np.log(np.maximum(probs, eps)) / float(task.temperature)
            expv = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = expv / np.sum(expv, axis=1, keepdims=True)
        preds = np.argmax(probs, axis=1)
        y_true = np.array([task.labels.index(lb) for lb in req.labels])
        acc = float((preds == y_true).mean()) if len(y_true) else 0.0
        # per-label metrics
        per_label: dict[str, dict[str, float]] = {}
        for i, lb in enumerate(task.labels):
            mask = y_true == i
            if mask.any():
                per_label[lb] = {
                    "support": float(mask.sum()),
                    "accuracy": float((preds[mask] == y_true[mask]).mean()),
                }
            else:
                per_label[lb] = {"support": 0.0, "accuracy": 0.0}
        # macro metrics
        try:
            from sklearn.metrics import precision_recall_fscore_support

            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, preds, labels=list(range(len(task.labels))), average="macro", zero_division=0
            )
            macro_precision = float(prec)
            macro_recall = float(rec)
            macro_f1 = float(f1)
        except Exception:
            macro_precision = macro_recall = macro_f1 = None

        # hardest samples: top-N with wrong prediction or low confidence margin
        hardest: list[dict[str, object]] = []
        try:
            margin = probs.max(axis=1) - np.partition(probs, -2, axis=1)[:, -2]
            for i in range(len(req.texts)):
                wrong = int(preds[i] != y_true[i])
                if wrong or margin[i] < 0.1:
                    hardest.append(
                        {
                            "text": req.texts[i],
                            "true": task.labels[y_true[i]],
                            "pred": task.labels[preds[i]],
                            "conf": float(probs[i, preds[i]]),
                            "margin": float(margin[i]),
                        }
                    )
            hardest = sorted(hardest, key=lambda d: (d.get("pred") == d.get("true"), d.get("margin", 0.0)))[:100]
        except Exception:
            hardest = []
        return EvalResponse(
            task=req.task,
            size=len(req.texts),
            accuracy=acc,
            per_label=per_label,
            macro_precision=macro_precision,
            macro_recall=macro_recall,
            macro_f1=macro_f1,
            hardest=hardest,
        )

    @app.post("/probs", response_model=ProbsResponse)
    async def get_probs(req: ProbsRequest):
        if req.task not in registry._tasks:
            raise HTTPException(status_code=404, detail=f"未找到任务: {req.task}")
        task = registry.get(req.task)
        if not req.texts:
            raise HTTPException(status_code=400, detail="texts 不能为空")
        embs = embedder.encode(req.texts)
        probs = task.clf.predict_proba(embs)
        # 注意：此端点返回未应用 temperature/threshold 的基础概率，便于做校准
        return ProbsResponse(task=req.task, labels=task.labels, probs=probs.tolist())

    @app.post("/tasks/calibrate")
    async def calibrate_task(req: CalibrateRequest):
        if req.task not in registry._tasks:
            raise HTTPException(status_code=404, detail=f"未找到任务: {req.task}")
        task = registry.get(req.task)
        updated = False
        if req.threshold is not None:
            if len(task.labels) != 2:
                raise HTTPException(status_code=400, detail="仅二分类任务支持 threshold")
            task.threshold = float(req.threshold)
            updated = True
        if req.temperature is not None:
            task.temperature = float(req.temperature)
            updated = True
        if not updated:
            return {"message": "no changes"}
        # 写回元数据
        try:
            meta_path = Path(task.model_path).with_suffix(Path(task.model_path).suffix + ".labels.json")
            meta: dict[str, object] = {"labels": task.labels}
            if task.threshold is not None:
                meta["threshold"] = task.threshold
            if task.temperature is not None:
                meta["temperature"] = task.temperature
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        return {"message": "calibrated", "task": req.task, "threshold": task.threshold, "temperature": task.temperature}

    return app
