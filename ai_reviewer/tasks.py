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
                except Exception:
                    # Ignore meta read errors; fall back to provided labels
                    pass
            else:
                # No meta file: persist current labels to lock mapping going forward
                try:
                    meta_path.write_text(json.dumps({"labels": labels}, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    pass
        else:
            # Fresh model: initialize with stable class indices 0..K-1
            clf = SGDClassifier(loss="log_loss", random_state=42)
            templates = [f"{name}-{label}" for label in labels]
            x_init = embedder_encode(templates)
            y_init = list(range(len(labels)))
            clf.partial_fit(x_init, y_init, classes=np.arange(len(labels)))
            joblib.dump(clf, model_path)
            # Persist label order for this model
            try:
                meta_path.write_text(json.dumps({"labels": labels}, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

        task = Task(name=name, labels=labels, model_path=model_path, clf=clf)
        self._tasks[name] = task
        return task

    def update(self, name: str, x, y):
        task = self._tasks[name]
        task.clf.partial_fit(x, y)
        joblib.dump(task.clf, task.model_path)
        # also ensure labels meta persists (in case of initial missing meta)
        meta_path = Path(task.model_path).with_suffix(Path(task.model_path).suffix + ".labels.json")
        if not meta_path.exists():
            try:
                meta_path.write_text(
                    json.dumps({"labels": task.labels}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass
