"""Public SCR-DTI API."""

from scrdti.config import ModelConfig, ReleaseConfig, TrainConfig, load_config
from scrdti.model import SCRDTI, SCRDTIOutput

__version__ = "0.1.0"

__all__ = [
    "ModelConfig",
    "ReleaseConfig",
    "SCRDTI",
    "SCRDTIOutput",
    "TrainConfig",
    "load_config",
]

