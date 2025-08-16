from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
from scipy import sparse as sp
from sklearn.linear_model import SGDClassifier


@dataclass
class Task:
    name: str
    labels: list[str]
    model_path: str
    classifier_type: str
    clf: Any  # SGDClassifier or lgb.LGBMClassifier
    threshold: float | None = None
    temperature: float | None = None
    vectorizer: Any | None = None  # sklearn TfidfVectorizer for LightGBM tasks


class TaskRegistry:
    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        # Shared embedder encode callback set on first ensure()
        self._encode = None

    def get(self, name: str) -> Task:
        return self._tasks[name]

    def list(self) -> list[str]:
        return list(self._tasks.keys())

    def ensure(
        self,
        name: str,
        labels: list[str],
        model_path: str,
        classifier_type: str,
        embedder_encode,
        threshold: float | None,
        temperature: float | None,
    ) -> Task:
        """
        Ensure a task exists. If a model already exists on disk, prefer the persisted
        label mapping to avoid index-order mismatches. Otherwise, create a new model,
        initialize classes deterministically (0..K-1), and persist both model and labels.
        """
        model_path_p = Path(model_path)
        model_path_p.parent.mkdir(parents=True, exist_ok=True)
        meta_path = model_path_p.with_suffix(model_path_p.suffix + ".labels.json")
        vec_path = model_path_p.with_suffix(model_path_p.suffix + ".tfidf.joblib")

        # Remember encode function
        if self._encode is None:
            self._encode = embedder_encode

        persisted_threshold: float | None = None
        persisted_temperature: float | None = None

        if model_path_p.exists():
            clf: Any = joblib.load(model_path)
            vec: Any | None = None
            if vec_path.exists():
                try:
                    vec = joblib.load(vec_path)
                except Exception:
                    vec = None
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
                        persisted_threshold = float(thr)
                    temp = persisted.get("temperature")
                    if isinstance(temp, (int, float)):
                        persisted_temperature = float(temp)
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
            # Fresh model
            if classifier_type == "lightgbm":
                print(f"✨ Initializing new LightGBM model for task '{name}'")
                clf = lgb.LGBMClassifier(random_state=42)
                # Create a tiny TF-IDF vectorizer to pair with the model so that API can build features later
                try:
                    from sklearn.feature_extraction.text import TfidfVectorizer

                    # Fit on simple templates just to initialize; should be overwritten by real training script
                    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1, max_features=1000)
                except Exception:
                    vec = None
            else:  # default to linear
                print(f"✨ Initializing new Linear model for task '{name}'")
                clf = SGDClassifier(
                    loss="log_loss",
                    penalty="l2",
                    alpha=1e-4,
                    random_state=42,
                    average=True,
                )
                vec = None

            # Cold start with template texts to establish classes
            templates = [f"{name}-{label}" for label in labels]
            x_emb = embedder_encode(templates)
            if vec is not None:
                try:
                    vec.fit(templates)
                    x_tfidf = vec.transform(templates)
                    x_init = sp.hstack([sp.csr_matrix(x_emb), x_tfidf], format="csr")
                except Exception:
                    x_init = x_emb
                    vec = None
            else:
                x_init = x_emb
            y_init = list(range(len(labels)))

            if hasattr(clf, "partial_fit"):
                clf.partial_fit(x_init, y_init, classes=np.arange(len(labels)))
            else:
                clf.fit(x_init, y_init)

            joblib.dump(clf, model_path)
            if vec is not None:
                try:
                    joblib.dump(vec, vec_path)
                except Exception:
                    pass
            # Persist label order for this model
            try:
                meta: dict[str, object] = {"labels": labels}
                if len(labels) == 2:
                    persisted_threshold = 0.5
                    meta["threshold"] = persisted_threshold
                persisted_temperature = 1.0
                meta["temperature"] = persisted_temperature
                meta["features"] = "emb|tfidf" if vec is not None else "emb"
                meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

        task = Task(
            name=name,
            labels=labels,
            model_path=model_path,
            classifier_type=classifier_type,
            clf=clf,
            threshold=threshold if threshold is not None else persisted_threshold,
            temperature=temperature if temperature is not None else persisted_temperature,
            vectorizer=locals().get("vec") if model_path_p.exists() else vec,
        )
        self._tasks[name] = task
        return task

    def update(self, name: str, x: np.ndarray, y: list[int]) -> None:
        task = self.get(name)
        if task.classifier_type != "linear":
            print(
                f"⚠️ WARNING: Online learning (/update) is only supported for 'linear' models. "
                f"Task '{name}' is '{task.classifier_type}'. Ignoring update."
            )
            return

        if hasattr(task.clf, "partial_fit"):
            task.clf.partial_fit(x, y)
            joblib.dump(task.clf, task.model_path)
        else:
            print(f"⚠️ WARNING: Classifier for task '{name}' does not support partial_fit. Online learning is disabled.")

    def _prepare_input_for_model(self, task: Task, x: Any) -> Any:
        """Pass-through: LightGBM accepts numpy arrays or scipy sparse. Avoid enforcing DataFrame.

        Kept for backward compatibility in linear models where x is ndarray.
        """
        return x

    def _build_features(self, task: Task, texts: list[str]) -> Any:
        """Build combined features: [embeddings | tfidf] -> hstack.

        Returns scipy.sparse.csr_matrix when TF-IDF exists; otherwise returns ndarray of embeddings.
        """
        if self._encode is None:
            raise RuntimeError("Embedding encoder not initialized in TaskRegistry")
        embs = self._encode(texts)
        if task.vectorizer is not None:
            try:
                x_tfidf = task.vectorizer.transform(texts)
                x = sp.hstack([sp.csr_matrix(embs), x_tfidf], format="csr")
                return x
            except Exception:
                # Fallback to embeddings only
                return embs
        return embs

    def predict(self, name: str, x: np.ndarray) -> list[dict[str, float]]:
        task = self.get(name)
        model_input = self._prepare_input_for_model(task, x)
        proba = task.clf.predict_proba(model_input)

        if task.threshold is not None and len(task.labels) == 2:
            # Binary classification with threshold
            scores = proba[:, 1]
            predictions = (scores >= task.threshold).astype(int)
            return [{"label": task.labels[p], "score": s} for p, s in zip(predictions, scores, strict=True)]
        else:
            # Multiclass classification
            return [dict(zip(task.labels, probs, strict=True)) for probs in proba]

    def predict_texts(self, name: str, texts: list[str]) -> list[dict[str, float]]:
        x = self._build_features(self.get(name), texts)
        return self.predict(name, x)

    def get_probs(self, name: str, x: np.ndarray) -> np.ndarray:
        """获取给定输入的原始概率分布（不应用温度或阈值）。"""
        task = self.get(name)
        model_input = self._prepare_input_for_model(task, x)
        proba = task.clf.predict_proba(model_input)
        return proba

    def get_probs_texts(self, name: str, texts: list[str]) -> np.ndarray:
        x = self._build_features(self.get(name), texts)
        return self.get_probs(name, x)
