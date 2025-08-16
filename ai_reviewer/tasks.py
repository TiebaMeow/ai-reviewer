from __future__ import annotations

import json
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import torch
import torch.nn.functional as nnf
from scipy import sparse as sp
from sklearn.linear_model import SGDClassifier
from transformers import AutoModelForSequenceClassification, AutoTokenizer


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
        # Cache for HF models keyed by model_path to avoid repeated loading
        self._hf_cache: dict[str, HFSequenceClassifier] = {}

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

        if model_path_p.exists() and classifier_type != "hf":
            # legacy/classical sklearn models
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
            if classifier_type == "hf":
                # HuggingFace fine-tuned model; must already exist at model_path
                print(f"✨ Loading HF Seq-Cls model for task '{name}' from '{model_path}'")
                if not model_path_p.exists():
                    raise FileNotFoundError(
                        f"HF model path not found for task '{name}': {model_path}. Please run fine-tuning first."
                    )
                # load or reuse from cache
                hf = self._hf_cache.get(model_path)
                if hf is None:
                    hf = HFSequenceClassifier(model_path)
                    self._hf_cache[model_path] = hf
                clf = hf  # store wrapper as clf for unified storage
                vec = None

                # Determine labels: prefer persisted meta, else HF config id2label, else provided
                resolved_labels = None
                meta_path_hf = model_path_p / "labels.json"
                if meta_path_hf.exists():
                    try:
                        persisted = json.loads(meta_path_hf.read_text(encoding="utf-8"))
                        persisted_labels = persisted.get("labels")
                        if isinstance(persisted_labels, list) and len(persisted_labels) == hf.num_labels:
                            resolved_labels = persisted_labels
                        thr = persisted.get("threshold")
                        if isinstance(thr, (int, float)):
                            persisted_threshold = float(thr)
                        temp = persisted.get("temperature")
                        if isinstance(temp, (int, float)):
                            persisted_temperature = float(temp)
                    except Exception:
                        pass
                if resolved_labels is None:
                    resolved_labels = hf.get_labels_from_config() or labels
                labels = resolved_labels

                # Persist meta if not present
                if not meta_path_hf.exists():
                    try:
                        meta: dict[str, object] = {"labels": labels}
                        if len(labels) == 2:
                            persisted_threshold = 0.5
                            meta["threshold"] = persisted_threshold
                        persisted_temperature = 1.0
                        meta["temperature"] = persisted_temperature
                        meta["features"] = "hf"
                        meta_path_hf.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                    except Exception:
                        pass

            elif classifier_type == "lightgbm":
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

            if classifier_type != "hf":
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
        if task.classifier_type == "hf":
            # HF models do not use external features here
            raise RuntimeError("_build_features called for 'hf' backend; use predict_texts/get_probs_texts directly")
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
        task = self.get(name)
        if task.classifier_type == "hf":
            hf: HFSequenceClassifier = task.clf  # type: ignore[assignment]
            proba = hf.predict_proba_texts(texts)
            if task.threshold is not None and len(task.labels) == 2:
                scores = proba[:, 1]
                predictions = (scores >= task.threshold).astype(int)
                return [{"label": task.labels[p], "score": float(s)} for p, s in zip(predictions, scores, strict=True)]
            else:
                return [dict(zip(task.labels, row.tolist(), strict=True)) for row in proba]
        x = self._build_features(task, texts)
        return self.predict(name, x)

    def get_probs(self, name: str, x: np.ndarray) -> np.ndarray:
        """获取给定输入的原始概率分布（不应用温度或阈值）。"""
        task = self.get(name)
        if task.classifier_type == "hf":
            raise RuntimeError("get_probs with raw features is not supported for 'hf' backend; use get_probs_texts")
        model_input = self._prepare_input_for_model(task, x)
        proba = task.clf.predict_proba(model_input)
        return proba

    def get_probs_texts(self, name: str, texts: list[str]) -> np.ndarray:
        task = self.get(name)
        if task.classifier_type == "hf":
            hf: HFSequenceClassifier = task.clf  # type: ignore[assignment]
            return hf.predict_proba_texts(texts)
        x = self._build_features(task, texts)
        return self.get_probs(name, x)


class HFSequenceClassifier:
    """Lightweight wrapper around a local HuggingFace sequence classification model.

    - Loads model/tokenizer from a directory (model_path)
    - Performs batched tokenization and inference
    - Returns softmax probabilities as numpy.ndarray (float32) of shape [B, K]
    """

    def __init__(self, model_path: str, device: str | None = None, max_length: int | None = None, batch_size: int = 32):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        # Determine label count
        self.num_labels = int(getattr(self.model.config, "num_labels", 0) or 0)
        # Reasonable default max_length if not provided
        if max_length is None:
            try:
                ml = int(getattr(self.tokenizer, "model_max_length", 512) or 512)
                # Some tokenizers use large sentinel value (e.g., 1000000000000000019884624838656)
                max_length = 512 if ml > 4096 else max(8, ml)
            except Exception:
                max_length = 512
        self.max_length = max_length
        self.batch_size = max(1, int(batch_size))

    def get_labels_from_config(self) -> list[str] | None:
        id2label = getattr(self.model.config, "id2label", None)
        if isinstance(id2label, dict) and id2label:
            try:
                # Keys may be strings; sort by integer id
                items = sorted(((int(k), v) for k, v in id2label.items()), key=operator.itemgetter(0))
                return [str(v) for _, v in items]
            except Exception:
                # Fallback: order by key as is
                return [str(v) for _, v in sorted(id2label.items(), key=operator.itemgetter(0))]
        return None

    @torch.no_grad()
    def predict_proba_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.num_labels), dtype=np.float32)
        outputs: list[np.ndarray] = []
        total = len(texts)
        bs = self.batch_size
        for i in range(0, total, bs):
            batch_texts = texts[i : i + bs]
            enc = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc).logits  # [B, K]
            probs = nnf.softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float32)
            outputs.append(probs)
        return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, self.num_labels), dtype=np.float32)
