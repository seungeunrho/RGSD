import torch
from abc import ABC, abstractmethod
from typing import Dict, Optional


class HierarchicalAdapter(ABC):
    """Adapter interface to connect hierarchical controller to a specific agent.

    Implementations should map agent-specific observation/latent plumbing to the
    controller-agnostic API used by the mixin.
    """

    @abstractmethod
    def latent_dim(self) -> int:
        """Return the latent dimension used by the agent/policy."""


