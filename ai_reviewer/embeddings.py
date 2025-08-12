from __future__ import annotations

import torch
from sklearn.preprocessing import normalize
from transformers import AutoModel, AutoTokenizer


class EmbeddingBackend:
    def __init__(self, model_name: str, device: str = "auto") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: list[str]):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return normalize(embeddings)
