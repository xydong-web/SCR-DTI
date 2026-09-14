from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch


DEFAULT_ENCODERS = {
    "smiles": "ibm-research/MoLFormer-XL-both-10pct",
    "text": "laituan245/molt5-base",
    "protein": "facebook/esm2_t33_650M_UR50D",
}


def _mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
    numerator = (hidden * mask).sum(dim=1)
    denominator = mask.sum(dim=1).clamp_min(1.0)
    return numerator / denominator


def encode_texts(
    texts: Sequence[str],
    *,
    model_name: str,
    max_length: int,
    batch_size: int = 8,
    device: str | torch.device | None = None,
    trust_remote_code: bool = False,
) -> np.ndarray:
    """Encode strings with a Hugging Face encoder and mask-aware mean pooling."""

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised only with optional deps
        raise RuntimeError(
            "Representation extraction requires `pip install -e '.[encoders]'`."
        ) from exc
    device_obj = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code).to(device_obj)
    model.eval()
    outputs: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        encoded: dict[str, Any] = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device_obj) for key, value in encoded.items()}
        with torch.no_grad():
            result = model(**encoded)
            hidden = result.last_hidden_state
            pooled = _mean_pool(hidden, encoded["attention_mask"])
        outputs.append(pooled.cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, 0), dtype=np.float32)

