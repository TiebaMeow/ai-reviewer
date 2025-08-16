from __future__ import annotations

import re
import unicodedata

import numpy as np
from sklearn.preprocessing import normalize


class EmbeddingBackend:
    """
    A small adapter that prefers SentenceTransformer for text embeddings and
    falls back to vanilla Transformers when needed.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        batch_size: int = 32,
        enable_preprocess: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.enable_preprocess = enable_preprocess
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

    @staticmethod
    def _normalize_texts(texts: list[str]) -> list[str]:
        """Lightweight normalization for Chinese text.

        - Unicode NFKC to unify full/half width
        - Remove URLs and @mentions
        - Collapse repeated punctuation/characters
        - Collapse whitespace
        - Lowercase ASCII letters
        """
        url_re = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
        # at_re = re.compile(r"@\S+")
        punct_re = re.compile(r"([!?。！，,．.、~…])\1{1,}")
        repeat_char_re = re.compile(r"(.)\1{2,}")
        ws_re = re.compile(r"\s+")

        normed: list[str] = []
        for t in texts:
            s = t if isinstance(t, str) else str(t)
            s = unicodedata.normalize("NFKC", s)
            s = url_re.sub(" ", s)
            # s = at_re.sub(" ", s)
            s = punct_re.sub(r"\1\1", s)  # limit long punctuation runs to 2
            s = repeat_char_re.sub(r"\1\1", s)  # limit long repeated chars to 2
            s = ws_re.sub(" ", s).strip()
            # lowercase latin letters only
            s = "".join(ch.lower() if ord(ch) < 128 else ch for ch in s)
            normed.append(s)
        return normed

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        proc_texts = self._normalize_texts(texts) if self.enable_preprocess else texts

        if self._backend == "sbert":
            # Use SentenceTransformer encode
            # normalize afterwards to be explicit and version-agnostic
            embs = self._sbert.encode(
                proc_texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            return normalize(embs)

        # Transformers fallback: mean-pool last_hidden_state
        import torch

        with torch.no_grad():
            inputs = self.tokenizer(proc_texts, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            # masked mean pooling to avoid padding influence
            last_hidden = outputs.last_hidden_state  # [B, L, H]
            attn_mask = inputs.get("attention_mask")
            if attn_mask is None:
                embeddings = last_hidden.mean(dim=1)
            else:
                mask = attn_mask.unsqueeze(-1).type_as(last_hidden)  # [B, L, 1]
                summed = (last_hidden * mask).sum(dim=1)  # [B, H]
                counts = mask.sum(dim=1).clamp(min=1e-9)  # [B, 1]
                embeddings = summed / counts
            embeddings = embeddings.cpu().numpy()
        return normalize(embeddings)
