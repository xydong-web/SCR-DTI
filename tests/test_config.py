from __future__ import annotations

from pathlib import Path

from scrdti.config import load_config


def test_all_release_configs_parse() -> None:
    root = Path(__file__).resolve().parents[1]
    configs = sorted((root / "configs").rglob("*.yaml"))
    assert len(configs) >= 10
    for path in configs:
        config = load_config(path)
        assert config.model.projection_dim > 0
        assert config.train.batch_size > 0

