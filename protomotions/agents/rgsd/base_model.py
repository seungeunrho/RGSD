# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
from torch import nn
from typing import List
from hydra.utils import instantiate
from protomotions.agents.common.mlp import MLP_WithNorm
from protomotions.agents.ppo.model import PPOModel
from protomotions.utils import model_utils
from torch.distributions.von_mises import VonMises
DISC_LOGIT_INIT_SCALE = 1.0
ENC_LOGIT_INIT_SCALE = 0.1


class SSDPhiEncoder(nn.Module):
    def __init__(self, config, num_in: int):
        super().__init__()
        self.config = config

        #self.config.contrastive_parameters.remove_phase_var

        # self.num_in = config.encoder.num_in

        self.trunk = MLP_WithNorm(config.trunk, num_in, config.trunk.num_out)
        self.trunk_output_activation = model_utils.get_activation_func(config.trunk.output_activation)
        
        self.encoder = torch.nn.Linear(config.trunk.num_out, config.encoder.num_out)
        torch.nn.init.uniform_(self.encoder.weight, -ENC_LOGIT_INIT_SCALE, ENC_LOGIT_INIT_SCALE)
        torch.nn.init.zeros_(self.encoder.bias)
        self.kappa = torch.as_tensor(self.config.vmf_kappa)  # Concentration parameter for Von Mises-Fisher distribution


    # def sample(self, input_dict: dict, num_samples: int) -> torch.Tensor:
    #     """Samples latent variables from a Von Mises distribution.

    #     Args:
    #         n_sample (int): Number of samples to draw.  
    #     """
    #     mu = self.mi_enc_forward(input_dict)  # (batch_size, n)
    #     kappa_t = torch.as_tensor(self.kappa, device=mu.device)
        
    #     dist = VonMises(mu, concentration=kappa_t)

        
    #     # sample() lets a leading sample shape; we want (..., num_samples, n)
    #     samples = dist.sample(sample_shape=(num_samples,))        # (num_samples, ..., n)
    #     # samples = samples.movedim(0, -2)     
    #     # angles = torch.arccos((X @ mu) .clamp(-1, 1))
    #     return samples
    
    def sample_around_mean(self, mu:torch.Tensor, num_samples: int) -> torch.Tensor:
        """Samples latent variables from a Von Mises distribution.

        Args:
            n_sample (int): Number of samples to draw.  
        """
        kappa_t = torch.as_tensor(self.kappa, device=mu.device)
        dist = VonMises(mu, concentration=kappa_t)
        samples = dist.sample(sample_shape=(num_samples,))        # (num_samples, ..., n)
        samples = torch.nn.functional.normalize(samples, dim=-1)

        # test = (samples *mu).sum(dim=-1)
        return samples
            


    def remove_phase(self, input_dict):
        if self.config.trunk.remove_phase_var:
            input_dict["historical_self_obs"] = input_dict["historical_self_obs"][:,1:]
        return input_dict
    
    def forward(self, input_dict: dict) -> torch.Tensor:
        """Forward pass through the discriminator network.

        Args:
            input_dict (dict): Dictionary containing input tensors, must contain 'historical_self_obs'.

        Returns:
            torch.Tensor: Discriminator output, sigmoid probability of being expert data. Shape (batch_size, 1).
        """
        # input_dict = self.remove_phase(input_dict)
        outs = self.trunk(input_dict)
        outs = self.trunk_output_activation(outs)
        outs = self.encoder(outs)
        outs = torch.nn.functional.normalize(outs, dim=-1)
        return outs
    
    def mi_enc_forward(self, input_dict: dict) -> torch.Tensor:
        """Forward pass through the encoder network for Mutual Information maximization.

        Args:
            input_dict (dict): Dictionary containing input tensors, must contain 'historical_self_obs'.

        Returns:
            torch.Tensor: Normalized encoder output. Shape (batch_size, encoder_output_dim).
        """
        outs = self.trunk(input_dict)
        outs = self.trunk_output_activation(outs)
        outs = self.encoder(outs)
        return torch.nn.functional.normalize(outs, dim=-1)
        
    def compute_mi_reward(self, obs, latents):
        """Computes the Mutual Information based reward.

        Args:
            obs (torch.Tensor): Observation tensor. Shape (batch_size, obs_dim).
            latents (torch.Tensor): Latent variable tensor. Shape (batch_size, latent_dim).

        Returns:
            torch.Tensor: Mutual Information reward tensor. Shape (batch_size, 1).
        """
        # input_dict = self.remove_phase(input_dict)

        obs = self.trunk(obs)
        obs = self.trunk_output_activation(obs)
        enc_pred = self.encoder(obs)
        enc_pred = torch.nn.functional.normalize(enc_pred, dim=-1)
        
        neg_err = -self.calc_von_mises_fisher_enc_error(enc_pred, latents)
        reward = neg_err
        reward = torch.clamp_min(neg_err, 0.0)
        # reward = (neg_err + 1) / 2

        # print("latent:", latents)
        # print("pred:", enc_pred)
        # print("reward:", neg_err.item())
        
        return reward
    
    def calc_von_mises_fisher_enc_error(self, enc_pred, latent):
        """Calculates the Von Mises-Fisher error between predicted and true latent vectors.

        Args:
            enc_pred (torch.Tensor): Predicted encoded latent vector. Shape (batch_size, latent_dim).
            latent (torch.Tensor): True latent vector. Shape (batch_size, latent_dim).

        Returns:
            torch.Tensor: Von Mises-Fisher error. Shape (batch_size, 1).
        """
        err = enc_pred * latent
        err = -torch.sum(err, dim=-1, keepdim=True)
        return err

    def all_weights(self):
        """Returns all weights of the discriminator encoder network.

        Returns:
            List[nn.Parameter]: List of weight parameters.
        """
        weights: list[nn.Parameter] = []
        for mod in self.trunk.mlp.modules():
            if isinstance(mod, nn.Linear):
                weights.append(mod.weight)
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                weights.append(mod.weight)
        return weights

    def all_discriminator_weights(self):
        """Returns all weights of the discriminator part of the network.

        Returns:
            List[nn.Parameter]: List of discriminator weight parameters.
        """
        weights: list[nn.Parameter] = []
        for mod in self.trunk.mlp.modules():
            if isinstance(mod, nn.Linear):
                weights.append(mod.weight)
        weights.append(self.logit_weights()[0])
        return weights

    def logit_weights(self) -> List[nn.Parameter]:
        """Returns the weights of the final discriminator layer.

        Returns:
            List[nn.Parameter]: List containing the weight parameter of the discriminator's linear layer.
        """
        return [self.discriminator.weight]
    
    def all_enc_weights(self):
        """Returns all weights of the encoder part of the network.

        Returns:
            List[nn.Parameter]: List of encoder weight parameters.
        """
        weights: list[nn.Parameter] = []
        for mod in self.trunk.mlp.modules():
            if isinstance(mod, nn.Linear):
                weights.append(mod.weight)
        weights.append(self.encoder.weight)
        return weights

    def enc_weights(self) -> List[nn.Parameter]:
        """Returns the weights of the final encoder layer.

        Returns:
            List[nn.Parameter]: List containing the weight parameter of the encoder's linear layer.
        """
        return [self.encoder.weight]


class SSDModel(PPOModel):
    def __init__(self, config):
        super().__init__(config)
        self._phi: SSDPhiEncoder = instantiate(
            self.config.ssd_phi_encoder,
        )

        self._phi_prime: SSDPhiEncoder = instantiate(
            self.config.ssd_phi_encoder,
        )

        self._lambda = torch.nn.Parameter(torch.tensor([30.0]), requires_grad=True)

