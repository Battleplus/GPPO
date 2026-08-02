"""Accepted GPPO-v2 first-stage engineering baseline."""

from .paper_env import EVENT_NAMES, TASK_NAMES, PaperAlignedUAVEnv, PaperEnvConfig
from .paper_models import PaperHeteroActorCritic

__all__ = [
    "EVENT_NAMES",
    "PaperAlignedUAVEnv",
    "PaperEnvConfig",
    "PaperHeteroActorCritic",
    "TASK_NAMES",
]
