from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def fit_temperature(
    validation_logits: np.ndarray,
    validation_labels: np.ndarray,
    *,
    max_iter: int = 100,
) -> float:
    """Fit a positive scalar temperature on validation data only."""

    logits = torch.as_tensor(validation_logits, dtype=torch.float64).view(-1)
    labels = torch.as_tensor(validation_labels, dtype=torch.float64).view(-1)
    if logits.numel() == 0 or logits.shape != labels.shape:
        raise ValueError("validation logits and labels must be non-empty and aligned")
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.25, max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp_min(1e-6)
        loss = F.binary_cross_entropy_with_logits(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp_min(1e-6).item())


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    probabilities = np.empty_like(scaled)
    positive = scaled >= 0
    probabilities[positive] = 1.0 / (1.0 + np.exp(-scaled[positive]))
    exp_scaled = np.exp(scaled[~positive])
    probabilities[~positive] = exp_scaled / (1.0 + exp_scaled)
    eps = np.finfo(np.float64).eps
    return np.clip(probabilities, eps, 1.0 - eps)

