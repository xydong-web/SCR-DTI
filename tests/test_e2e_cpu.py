from __future__ import annotations

from pathlib import Path

import pandas as pd

from scrdti.config import load_config
from scrdti.data import load_feature_npz, make_synthetic_npz
from scrdti.evaluate import binary_metrics
from scrdti.training import load_release_checkpoint, predict_arrays, train_from_npz


def test_tiny_cpu_train_checkpoint_predict_evaluate(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/examples/tiny_cpu.yaml")
    data_path = make_synthetic_npz(tmp_path / "tiny.npz", n=72, seed=19)
    checkpoint_path = tmp_path / "model.pt"
    result = train_from_npz(config, data_path, checkpoint_path, device="cpu")
    assert checkpoint_path.is_file()
    assert result.best_epoch >= 1
    model, payload = load_release_checkpoint(checkpoint_path)
    assert payload["release"].startswith("SCR-DTI")
    predictions = predict_arrays(model, load_feature_npz(data_path), split_id=2, device="cpu")
    frame = pd.DataFrame(predictions)
    assert not frame.empty
    metrics = binary_metrics(frame["label"].to_numpy(), frame["probability"].to_numpy())
    assert 0.0 <= metrics["auprc"] <= 1.0

