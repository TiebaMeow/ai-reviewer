from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier


@dataclass
class Task:
    name: str
    labels: list[str]
    model_path: str
    clf: SGDClassifier
    threshold: float | None = None
    temperature: float | None = None


class TaskRegistry:
    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}

    def get(self, name: str) -> Task:
        return self._tasks[name]

    def list(self) -> list[str]:
        return list(self._tasks.keys())

    def ensure(self, name: str, labels: list[str], model_path: str, embedder_encode) -> Task:
        """
        Ensure a task exists. If a model already exists on disk, prefer the persisted
        label mapping to avoid index-order mismatches. Otherwise, create a new model,
        initialize classes deterministically (0..K-1), and persist both model and labels.
        """
        model_path_p = Path(model_path)
        model_path_p.parent.mkdir(parents=True, exist_ok=True)
        meta_path = model_path_p.with_suffix(model_path_p.suffix + ".labels.json")

        threshold: float | None = None
        temperature: float | None = None
        if model_path_p.exists():
            clf: SGDClassifier = joblib.load(model_path)
            # Load persisted labels if available; this guarantees stable mapping across restarts
            if meta_path.exists():
                try:
                    persisted = json.loads(meta_path.read_text(encoding="utf-8"))
                    persisted_labels = persisted.get("labels")
                    if isinstance(persisted_labels, list) and len(persisted_labels) == len(
                        getattr(clf, "classes_", [])
                    ):
                        labels = persisted_labels
                    thr = persisted.get("threshold")
                    if isinstance(thr, (int, float)):
                        threshold = float(thr)
                    temp = persisted.get("temperature")
                    if isinstance(temp, (int, float)):
                        temperature = float(temp)
                except Exception:
                    # Ignore meta read errors; fall back to provided labels
                    pass
            else:
                # No meta file: persist current labels to lock mapping going forward
                try:
                    meta: dict[str, object] = {"labels": labels}
                    if len(labels) == 2:
                        meta["threshold"] = 0.5
                    meta["temperature"] = 1.0
                    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    pass
        else:
            # Fresh model: initialize with stable class indices 0..K-1
            # Tuned for more stable online learning
            clf = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=1e-4,
                random_state=42,
                average=True,
            )
            templates = [f"{name}-{label}" for label in labels]
            x_init = embedder_encode(templates)
            y_init = list(range(len(labels)))
            clf.partial_fit(x_init, y_init, classes=np.arange(len(labels)))
            joblib.dump(clf, model_path)
            # Persist label order for this model
            try:
                # also persist a default threshold for binary tasks (0.5), can be updated later
                meta: dict[str, object] = {"labels": labels}
                if len(labels) == 2:
                    threshold = 0.5
                    meta["threshold"] = threshold
                temperature = 1.0
                meta["temperature"] = temperature
                meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

        task = Task(
            name=name,
            labels=labels,
            model_path=model_path,
            clf=clf,
            threshold=threshold,
            temperature=temperature,
        )
        self._tasks[name] = task
        return task

    def update(self, name: str, x, y, sample_weight=None):
        task = self._tasks[name]
        if sample_weight is not None:
            task.clf.partial_fit(x, y, sample_weight=sample_weight)
        else:
            task.clf.partial_fit(x, y)
        joblib.dump(task.clf, task.model_path)
        # also ensure labels meta persists (in case of initial missing meta)
        meta_path = Path(task.model_path).with_suffix(Path(task.model_path).suffix + ".labels.json")
        if not meta_path.exists():
            try:
                meta: dict[str, object] = {"labels": task.labels}
                if len(task.labels) == 2:
                    meta["threshold"] = task.threshold if task.threshold is not None else 0.5
                meta["temperature"] = task.temperature if task.temperature is not None else 1.0
                meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass
