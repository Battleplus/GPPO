"""GPPO-v2 baseline and PCRL-v0 preference-control research prototype."""

from .paper_env import EVENT_NAMES, TASK_NAMES, PaperAlignedUAVEnv, PaperEnvConfig
from .paper_models import PaperHeteroActorCritic
from .pcrl_models import PreferenceConditionedPaperActorCritic
from .pcrl_v0 import PreferencePaperEnv

__all__ = [
    "EVENT_NAMES",
    "PaperAlignedUAVEnv",
    "PaperEnvConfig",
    "PaperHeteroActorCritic",
    "PreferenceConditionedPaperActorCritic",
    "PreferencePaperEnv",
    "TASK_NAMES",
]
