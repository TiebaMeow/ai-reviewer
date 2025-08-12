from __future__ import annotations

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
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        if Path(model_path).exists():
            clf: SGDClassifier = joblib.load(model_path)
        else:
            clf = SGDClassifier(loss="log_loss")
            # minimal bootstrap with two short samples per binary or uniform across labels
            # For N labels, create simple templates
            templates = [f"{name}-{label}" for label in labels]
            x_init = embedder_encode(templates)
            y_init = list(range(len(labels)))
            clf.partial_fit(x_init, y_init, classes=np.arange(len(labels)))
            joblib.dump(clf, model_path)
        task = Task(name=name, labels=labels, model_path=model_path, clf=clf)
        self._tasks[name] = task
        return task

    def update(self, name: str, x, y):
        task = self._tasks[name]
        task.clf.partial_fit(x, y)
        joblib.dump(task.clf, task.model_path)
