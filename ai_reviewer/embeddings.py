from __future__ import annotations

import numpy as np
from sklearn.preprocessing import normalize


class EmbeddingBackend:
    """
    A small adapter that prefers SentenceTransformer for text embeddings and
    falls back to vanilla Transformers when needed.
    """

    def __init__(self, model_name: str, device: str = "auto") -> None:
        self.model_name = model_name
        self.device = device
        self._backend = None  # "sbert" | "hf"

        # Prefer SentenceTransformer if available
        try:
            import torch  # defer import to keep optional when unused
            from sentence_transformers import SentenceTransformer  # type: ignore

            resolved_device = (
                "cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device)
            )
            self._sbert = SentenceTransformer(model_name, device=resolved_device)
            self._backend = "sbert"
            self.device = resolved_device
        except Exception:
            # Fallback to Transformers (AutoModel/AutoTokenizer)
            import torch
            from transformers import AutoModel, AutoTokenizer  # type: ignore

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # trust_remote_code=True to support some embedding models exposing custom forward
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            if device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
            self.model.to(self.device)
            self.model.eval()
            self._backend = "hf"

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        if self._backend == "sbert":
            # Use SentenceTransformer encode
            # normalize afterwards to be explicit and version-agnostic
            embs = self._sbert.encode(texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=False)
            return normalize(embs)

        # Transformers fallback: mean-pool last_hidden_state
        import torch

        with torch.no_grad():
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            # mean pooling over sequence length (simple baseline)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return normalize(embeddings)
