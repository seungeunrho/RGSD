import torch
from typing import Optional, Dict, Tuple

from protomotions.agents.rgsd.rgsd_diayn_agent import RGSD_DIAYN
from protomotions.agents.rgsd.base_agent import DiscoveryBase

from protomotions.agents.hierarchical.mixin import HierarchicalMixin


class RGSD_Hierarchical(HierarchicalMixin, RGSD_DIAYN):
# class RGSD_Hierarchical(HierarchicalMixin, DiscoveryBase):


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize hierarchical control with RGSD-specific adapter
        self._init_hl(
            cfg=self.config.hierarchical,
            num_envs=self.num_envs,
            device=self.device
        )

    def setup(self):
        """Setup with hierarchical control integration."""
        base_model, base_actor_opt, base_critic_opt, base_phi_opt = super().setup()
        self.model, self.actor_optimizer, self.critic_optimizer, self.phi_optimizer = self.fabric.setup(
            base_model, base_actor_opt, base_critic_opt, base_phi_opt
        )
        
        if hasattr(self.model, "_actor"):
            for p in self.model._actor.parameters():
                p.requires_grad = False
        if hasattr(self.model, "_critic"):
            for p in self.model._critic.parameters():
                p.requires_grad = False
        if hasattr(self.model, "_phi"):
            for p in self.model._phi.parameters():
                p.requires_grad = False
        if hasattr(self.model, "_phi_prime"):
            for p in self.model._phi_prime.parameters():
                p.requires_grad = False
        if hasattr(self.model, "_discriminator"):
            for p in self.model._discriminator.parameters():
                p.requires_grad = False

        self.model.mark_forward_method("act")
        self.model.mark_forward_method("get_action_and_value")
        return self.model, self.actor_optimizer, self.critic_optimizer, self.phi_optimizer

    # actor_step is handled by HierarchicalMixin.calculate_extra_actor_loss()
    # which uses self._hl_adapter.actor_loss_inputs() via the adapter pattern

    # get_state_dict and load_parameters are handled by HierarchicalMixin
    # which saves/loads _hl_net and _hl_logstd automatically

    # @torch.no_grad()
    # def calculate_extra_reward(self, hl_action):
    #     motion_ids = self.env.motion_manager.motion_ids
    #     target_latent = self.data_latents[motion_ids]

    #     action_diff = torch.norm(hl_action-target_latent, dim=-1)
    #     action_diff_squared = torch.square(action_diff)
    #     rew = torch.exp(-3*action_diff_squared)
    #     """No extra reward; only environment task reward (same as original RGSD)."""
    #     return rew
