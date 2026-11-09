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
import logging

from typing import Tuple, Dict
from protomotions.agents.rgsd.phi_contrastive2 import ContrastiveLearning
import os
from protomotions.agents.utils.data_utils import swap_and_flatten01

from protomotions.agents.rgsd.base_agent import DiscoveryBase
from hydra.utils import instantiate


log = logging.getLogger(__name__)
torch.set_printoptions(precision=6, sci_mode=False)

class RGSD_DIAYN(DiscoveryBase):
    # -----------------------------
    # Initialization and Setup
    # -----------------------------

    def setup(self):
        model, actor_optimizer, critic_optimizer, phi_optimizer = super().setup()
        phi_prime_optimizer = instantiate(
            self.config.model.config.discriminator_optimizer,
            params=list(model._phi.parameters()),
        )

        (
            self.model,
            self.actor_optimizer,
            self.critic_optimizer,
            self.phi_optimizer,
            self.phi_prime_optimizer
        ) = self.fabric.setup(
            model, actor_optimizer, critic_optimizer, phi_optimizer, phi_prime_optimizer
        )

        self.model.mark_forward_method("act")
        self.model.mark_forward_method("get_action_and_value")

        if self.config.ssd_parameters.rgsd_on:
            self.ref_motion_tensor_lst = self._get_motion_ref_data()
            self.contrastive_algo = ContrastiveLearning(
                cfg=self.config.contrastive_parameters,
                motion_tensor_lst = self.ref_motion_tensor_lst,
                dt=self.env.dt,
                n_history=self.n_history,
                latent_dim=self.config.ssd_parameters.latent_dim,
                device=self.device,
            )

            if self.config.ssd_parameters.pretrain_phi:
                self._initial_contrastive_train(epochs = self.config.contrastive_parameters.epochs)
            self.data_latents = self.contrastive_algo.compute_ref_data_latents(self.model._phi, 
                                                                n_frames=self.n_history)
            
            self.model._phi_prime.load_state_dict(self.model._phi.state_dict())
            for param in self.model._phi.parameters():
                param.requires_grad = False

        return model, actor_optimizer, critic_optimizer, phi_optimizer

    # -----------------------------
    # Reward Calculation
    # -----------------------------
    @torch.no_grad()
    def calc_diayn_reward(self):
        """Calculates extra reward based on Mutual Information."""
        # mimic_r = self.calculate_mimic_reward()
        next_obs = {
            "historical_self_obs": self.experience_buffer.next_historical_self_obs
        }
        dones = self.experience_buffer.dones
        is_first_step = self.experience_buffer.is_first_step
        latents = self.experience_buffer.latents


        next_obs = self._adjust_phi_input(next_obs, key="historical_self_obs" )
        mi_r = self.model._phi_prime.compute_mi_reward(
            {
                "historical_self_obs": next_obs["historical_self_obs"],
            },
            latents=latents,
        ).squeeze(2)

        # self.experience_buffer.batch_update_data("mi_rewards", mi_r)
        # mask = torch.logical_or(dones, is_first_step)
        # mi_r[mask] = 0.0
  
        return mi_r
    

    def calc_mimic_reward(self):
        batch_size = len(self.experience_buffer.next_historical_self_obs)
        next_obs = self.experience_buffer.next_historical_self_obs

        dones = self.experience_buffer.dones
        is_first_step = self.experience_buffer.is_first_step
        latents = self.experience_buffer.latents

        motion_ids = self.experience_buffer.motion_id  
        next_obs = next_obs.view(batch_size,self.num_envs, self.n_history, -1)
        cur_motion_times = next_obs[:,:,0,0].flatten()

        expert_obs = self.env.self_obs_cb.build_self_obs_demo(
            motion_ids, cur_motion_times, self.n_history
        ).clone()
        expert_obs = expert_obs.view(batch_size,self.num_envs, -1)

        next_obs = {
            "historical_self_obs": self.experience_buffer.next_historical_self_obs
        }
        expert_obs = {
            "historical_self_obs": expert_obs
        }

        z_infer = self.model._phi.mi_enc_forward(next_obs)
        z_expert = self.model._phi.mi_enc_forward(expert_obs)

        rew = (z_infer * z_expert).sum(dim=-1)

        return rew


    @torch.no_grad()
    def calculate_extra_reward(self):
        #mimic rew
        mimic_rew = self.calc_mimic_reward()
        diayn_rew = self.calc_diayn_reward()

        is_ref = self.experience_buffer.ref_indices
        not_ref = torch.logical_not(is_ref)
        mimic_rew[not_ref] = 0.0
        diayn_rew[is_ref] = 0.0
        
        rew = mimic_rew + diayn_rew 

        return rew

    
    def _initial_contrastive_train(self, epochs = 1000):

        for epoch in range(epochs):
            total_loss = 0
            data_continues = True
            
            # if epoch%self.config.contrastive_parameters.draw_interval==0:
            if epoch in [0,1,5,10,100,500,1000,2000]:
            
                file_name = os.path.join(self.plot_path, f"my_plot_epoch_{epoch}")
                ax = self.contrastive_algo.draw_latent_figure(self.model._phi, 
                                                          str(self.current_epoch), 
                                                          file_name,
                                                          n_frames=self.n_history)
            
            while data_continues:
                self.phi_optimizer.zero_grad()
                loss, data_continues = self.contrastive_algo.compute_loss_once(self.model._phi)

                # enc_weight_params = self.model._phi.all_enc_weights()
                # total: Tensor = sum([p.pow(2).sum() for p in enc_weight_params])
                # print(total)
                # weight_decay_loss: Tensor = (
                #     total * 0.00001
                # )

                # loss += weight_decay_loss
                self.fabric.backward(loss)
                total_loss += loss.item()

                # Update weights
                self.phi_optimizer.step()

            # Print epoch loss
            avg_loss = total_loss / len(self.contrastive_algo.dataloader)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")
    
    def discriminator_step(self, batch_dict):
        """Performs a discriminator update step.

        Args:
            batch_dict (dict): Batch of data from the experience buffer.

        Returns:
            Tuple[Tensor, Dict]: Discriminator loss and logging dictionary.
        """

        # mask = torch.logical_or(batch_dict['is_first_step'], batch_dict['dones'])
        mask_ref_indices = batch_dict['ref_indices']
        mask_def_indices = torch.logical_not(mask_ref_indices)


        latents = batch_dict['latents']
        discriminator_log_dict = {}

        obs = self._adjust_phi_input(batch_dict, key="historical_self_obs" )

        mi_enc_pred = self.model._phi_prime.mi_enc_forward({
            'historical_self_obs' : obs['historical_self_obs']
        })

        mi_enc_err = self.model._phi_prime.calc_von_mises_fisher_enc_error(mi_enc_pred, latents)
        # mi_enc_err[mask] = 0.0
        mi_enc_err[mask_ref_indices] = 0.0
        mi_enc_loss = torch.mean(mi_enc_err) * 0.5
        phi_loss = mi_enc_loss

        with torch.no_grad():
            mi_enc_pred_frozen = self.model._phi.mi_enc_forward({
                'historical_self_obs' : obs['historical_self_obs']
            })
        mi_enc_err_frozen = self.model._phi.calc_von_mises_fisher_enc_error(mi_enc_pred, mi_enc_pred_frozen)
        # mi_enc_err_frozen[mask] = 0.0
        mi_enc_err_frozen[mask_def_indices] = 0.0
        mi_enc_loss_frozen = torch.mean(mi_enc_err_frozen) * 0.5
        phi_loss += mi_enc_loss_frozen
        
        discriminator_log_dict["loss/phi_mi_enc_loss_frozen"] = mi_enc_loss_frozen.detach()
        discriminator_log_dict["loss/phi_mi_enc_loss"] = mi_enc_loss.detach()

        return phi_loss, discriminator_log_dict


    def extra_optimization_steps(self, batch_dict, batch_idx: int):
        extra_opt_steps_dict = super().extra_optimization_steps(batch_dict, batch_idx)


        phi_loss, discriminator_loss_dict = self.discriminator_step(
            batch_dict
        )
        extra_opt_steps_dict.update(discriminator_loss_dict)
        

        self.phi_prime_optimizer.zero_grad()
        self.fabric.backward(phi_loss)

        discriminator_grad_clip_dict = self.handle_model_grad_clipping(
            self.model._phi,
            self.phi_optimizer,
            "_phi",
        )

        extra_opt_steps_dict.update(discriminator_grad_clip_dict)
        self.phi_prime_optimizer.step()

        return extra_opt_steps_dict

    # -----------------------------
    # Termination and Logging
    # -----------------------------
    def post_epoch_logging(self, training_log_dict: Dict):
        """Performs post epoch logging, including Mutual Information reward logging.

        Args:
            training_log_dict (Dict): Dictionary to update with logging information.
        """
        # training_log_dict["rewards/div_rewards"] = (
        #     self.experience_buffer.div_rewards.mean()
        # )
        super().post_epoch_logging(training_log_dict)

        if self.config.ssd_parameters.rgsd_on:
            self.update_data_latents()

            if self.current_epoch % 3000 == 0 and self.current_epoch > 0:
            
                file_name = os.path.join(self.plot_path, f"my_plot_epoch_{str(self.current_epoch)}")
                ax = self.contrastive_algo.draw_latent_figure(self.model._phi, 
                                                            str(self.current_epoch), 
                                                            file_name,
                                                            n_frames=self.n_history)
