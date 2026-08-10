"""Paper-faithful GPPO for multi-UAV dynamic task assignment."""

from .paper_faithful_env import PAPER_SCALES, PaperFaithfulConfig, PaperFaithfulUAVEnv
from .paper_faithful_models import PaperFaithfulActorCritic
from .disturbances import (
    DisturbanceConfig,
    DisturbanceEvent,
    DisturbanceLogger,
    DisturbanceTape,
    SourceConfig,
)

__all__ = [
    "PAPER_SCALES",
    "PaperFaithfulActorCritic",
    "PaperFaithfulConfig",
    "PaperFaithfulUAVEnv",
    "DisturbanceConfig",
    "DisturbanceEvent",
    "DisturbanceLogger",
    "DisturbanceTape",
    "SourceConfig",
]
