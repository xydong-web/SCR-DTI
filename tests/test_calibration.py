from __future__ import annotations

import numpy as np

from scrdti.calibration import apply_temperature, fit_temperature
from scrdti.evaluate import negative_log_likelihood


def test_validation_only_temperature_scaling_is_well_formed() -> None:
    logits = np.array([-4.0, -2.0, -0.5, 0.5, 2.0, 4.0], dtype=np.float64)
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)
    temperature = fit_temperature(logits, labels, max_iter=30)
    assert temperature > 0
    calibrated = apply_temperature(logits, temperature)
    assert np.all((calibrated > 0) & (calibrated < 1))
    assert negative_log_likelihood(labels, calibrated) < 1.0

