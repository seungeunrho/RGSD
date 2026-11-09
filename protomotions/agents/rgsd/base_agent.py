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
import matplotlib.pyplot as plt
import time
from torch import Tensor
from hydra.utils import instantiate

from lightning.fabric import Fabric
from hydra.utils import instantiate
from protomotions.agents.common.common import weight_init

from protomotions.envs.base_env.env_utils.general import StepTracker
from protomotions.envs.base_env.env import BaseEnv
from typing import Tuple, Dict
from protomotions.agents.ppo.agent import PPO
from protomotions.agents.rgsd.base_model import SSDModel
from protomotions.agents.rgsd.phi_contrastive2 import ContrastiveLearning

from protomotions.envs.base_env.env_utils.humanoid_utils import (
    compute_humanoid_observations,
    compute_humanoid_observations_max,
)

import numpy as np


import os
from pathlib import Path
log = logging.getLogger(__name__)
torch.set_printoptions(precision=4, sci_mode=False)

class DiscoveryBase(PPO):
    # -----------------------------
    # Initialization and Setup
    # -----------------------------
    def __init__(self, fabric: Fabric, env: BaseEnv, config):
        super().__init__(fabric, env, config)
        self.latents = None
        self.latent_reset_steps = None
        self.latent_dim = self.config.ssd_parameters.latent_dim

        self.disc_loss_coef = self.config.ssd_parameters.disc_loss_coef
        self.lambda_loss_coef = self.config.ssd_parameters.lambda_loss_coef
        self.n_history = self.env.config.humanoid_obs.num_historical_steps

        self._adjust_encoder_num_in()
        self.kappa = torch.as_tensor(self.config.vmf_kappa) 


    def setup(self):
        model: SSDModel = instantiate(self.config.model)
        model.apply(weight_init)
        actor_optimizer = instantiate(
            self.config.model.config.actor_optimizer,
            params=list(model._actor.parameters()),
        )
        critic_optimizer = instantiate(
            self.config.model.config.critic_optimizer,
            params=list(model._critic.parameters()),
        )
        phi_optimizer = instantiate(
            self.config.model.config.discriminator_optimizer,
            params=list(model._phi.parameters()),
        )

        self.latents = torch.zeros(
            (self.num_envs, self.config.ssd_parameters.latent_dim),
            dtype=torch.float,
            device=self.device,
        )

        self.latent_reset_steps = StepTracker(
            self.num_envs,
            min_steps=self.config.ssd_parameters.latent_steps_min,
            max_steps=self.config.ssd_parameters.latent_steps_max,
            device=self.device,
        )

        self._set_save_path()


        self.latent_init_method = {
            "def_envs" : self.config.ssd_parameters.latent_init_method.default_envs,
            "ref_envs" : self.config.ssd_parameters.latent_init_method.ref_envs
        }

        self.motion_ids = torch.zeros(
            (self.num_envs, 1),
            dtype=torch.long,
            device=self.device,
        )

        return model, actor_optimizer, critic_optimizer, phi_optimizer
    
    def _adjust_encoder_num_in(self):

        if (not self.config.contrastive_parameters.attach_encoder_phase_var) and \
           self.env.config.humanoid_obs.attach_phase_variable:
            self.config.model.config.ssd_phi_encoder.num_in -= 1 * self.n_history

    def _set_save_path(self):
        if len(self.fabric.loggers) == 0:
            # means eval time
            save_dir = "results"
        else:
            path = self.fabric.loggers[0].log_dir
            root_dir = Path.cwd() / Path(self.fabric.loggers[0].root_dir)
            save_dir = Path.cwd() / Path(path)
        self.plot_path = os.path.join(save_dir, "plots")
        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path)
                
    def _get_motion_ref_data(self, use_max_coord_obs=True):
        results = []

        for motion_idx in range(self.env.motion_lib.num_motions()):
            length = self.env.motion_lib.get_motion_length(motion_idx)
            num_step = (length/self.env.dt).int()
            time_indices = torch.arange(num_step, device=self.device) *self.env.dt
            env_ids = torch.zeros_like(time_indices, dtype=torch.int, device=self.device) + motion_idx
            ref_state = self.env.motion_lib.get_motion_state(env_ids, time_indices)

            obs_ref = compute_humanoid_observations_max(
                ref_state.rigid_body_pos,
                ref_state.rigid_body_rot,
                ref_state.rigid_body_vel,
                ref_state.rigid_body_ang_vel,
                torch.zeros(num_step, 1, device=self.env.device),
                self.env.config.humanoid_obs.local_root_obs,
                self.env.config.humanoid_obs.root_height_obs,
                True,
            )

            # print(motion_idx, torch.abs(ref_state.rigid_body_vel).mean(dim=2).mean(dim=1))

            num_in = self.config.model.config.actor.config.mu_model.config.input_models.self_obs.num_in
            
            if self.config.contrastive_parameters.attach_encoder_phase_var:
                obs_ref = torch.cat((time_indices.unsqueeze(-1), obs_ref), dim=-1)
            # obs_ref = obs_ref[:,:num_in]

            results.append(obs_ref)

        return results


    # -----------------------------
    # Latent Management
    # -----------------------------
    def update_latents(self):
        """Updates latent variables based on latent reset steps."""
        self.latent_reset_steps.advance()
        reset_ids = self.latent_reset_steps.done_indices()

        if reset_ids.numel() > 0:
            self.reset_latents(reset_ids, methods="gaussian")
            self.latent_reset_steps.reset_steps(reset_ids)

    def reset_latents_fixed(self, env_ids=None):
        idx = self.config.ssd_parameters.latent_init_method.latent_fixed_idx
        latents = torch.zeros((len(env_ids), self.config.ssd_parameters.latent_dim),
                                dtype=torch.float,
                                device=self.device)
        latents[:, idx] = 1.0
        
        self.store_latents(latents, env_ids)

    def reset_latents_fixed_per_env(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        latents = torch.zeros((len(env_ids), self.config.ssd_parameters.latent_dim),
                                dtype=torch.float,
                                device=self.device)
        env_indices = torch.arange(len(env_ids), device=self.device)
        latent_id = torch.fmod(
                env_ids,
                self.config.ssd_parameters.latent_dim,
            )

        # print(latent_id)
        
        latents[env_indices, latent_id] = 1.0
        
        self.store_latents(latents, env_ids)

    def reset_latents_gaussian(self, env_ids=None):
        """Resets latent variables for specified environments or all environments if None.

        Args:
            env_ids (torch.Tensor, optional): Environment indices to reset latents for. Defaults to None (all envs).
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return None
        latents = self.sample_latents(len(env_ids))
        self.store_latents(latents, env_ids)

    def reset_latents_ref_data(self, env_ids=None, sampling=None):
        """Resets latent variables for specified environments or all environments if None.

        Args:
            env_ids (torch.Tensor, optional): Environment indices to reset latents for. Defaults to None (all envs).
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        if len(env_ids) == 0:
            return None

        # If reference data latents are unavailable (e.g., contrastive disabled), fallback to gaussian
        if not hasattr(self, "data_latents") or self.data_latents is None:
            self.reset_latents_gaussian(env_ids)
            return None
        
        if sampling is None:
            motion_ids = self.env.motion_manager.get_respawn_info(env_ids)[0]
            latents = self.data_latents[motion_ids]
        elif sampling == "uniform":
            # sample random ref_z out of ref motions.
            indices = torch.randint(0, len(self.data_latents), (len(env_ids),), device=self.device)
            latents = self.data_latents[indices]  # Shape: [1024, 32]  
        elif sampling == "vmf":
            motion_ids = self.env.motion_manager.get_respawn_info(env_ids)[0]
            latents = self.data_latents[motion_ids]
            sampled_latents = self.model._phi_prime.sample_around_mean(latents, num_samples=1)
            latents =  sampled_latents.squeeze(0)
        else:
            motion_ids = self.env.motion_manager.get_respawn_info(env_ids)[0]
            latents = self.data_latents[motion_ids]
        self.store_latents(latents, env_ids)

    def update_data_latents(self):
        self.data_latents = self.contrastive_algo.compute_ref_data_latents(self.model._phi, 
                                                               n_frames=self.n_history)    
    
    def store_latents(self, latents, env_ids):
        """Stores latent variables for specified environments.

        Args:
            latents (torch.Tensor): Latent variables to store. Shape (num_envs, latent_dim).
            env_ids (torch.Tensor): Environment indices to store latents for. Shape (num_envs,).
        """
        self.latents[env_ids] = latents

    def sample_latents(self, n):
        """Samples new latent variables uniformly on the unit-sphere.

        Args:
            n (int): Number of latent variables to sample.

        Returns:
            torch.Tensor: Sampled latent variables. Shape (n, latent_dim).
        """
        zeros = torch.zeros(
            [n, self.config.ssd_parameters.latent_dim], device=self.device
        )

        gaussian_sample = torch.normal(zeros)
        latents = torch.nn.functional.normalize(
            gaussian_sample, dim=-1
        )
        return latents

    # -----------------------------
    # Experience Buffer and Dataset Processing
    # -----------------------------
    def register_extra_experience_buffer_keys(self):
        super().register_extra_experience_buffer_keys()
        # self.experience_buffer.register_key("div_rewards")  # mi = mutual information
        self.experience_buffer.register_key(
            "latents", shape=(self.config.ssd_parameters.latent_dim,)
        )
        self.experience_buffer.register_key(
            "motion_id", shape=(1,), dtype=torch.long
        )

    def post_train_env_step(self, rewards, dones, done_indices, extras, step):
        """Post environment step processing during training. Updates experience buffer with latents and updates latents.

        Args:
            rewards (Tensor): Rewards from environment step. Shape (num_envs,).
            dones (Tensor): Done flags from environment step. Shape (num_envs,).
            done_indices (Tensor): Indices of environments that are done. Shape (num_done_envs,).
            extras (dict): Extra information from environment step.
            step (int): Current training step.
        """
        super().post_train_env_step(rewards, dones, done_indices, extras, step)
        self.experience_buffer.update_data("latents", step, self.latents)
        self.experience_buffer.update_data("motion_id", step, self.motion_ids)

        # self.update_latents()

    # -----------------------------
    # Environment Interaction
    # -----------------------------
    def handle_reset(self, done_indices=None):
        """Reset environment states and latents, then update the observation accordingly."""
        if self.force_full_restart:
            done_indices = None
            self.force_full_restart = False
        
        if done_indices is None:
            done_indices = torch.arange(self.num_envs, device=self.device, dtype=torch.long)



        obs, default_env_ids, ref_env_ids = self.env.reset(done_indices)
        obs["ref_indices"] = self.env.env_reset_ref.clone()

        if len(done_indices) == 0:
            obs["latents"] = self.latents.clone()
            return obs
        
        self.motion_ids[done_indices] = self.env.motion_manager.motion_ids[done_indices].unsqueeze(-1)       

        
        if done_indices is not None and len(done_indices) > 0:
            self.motion_ids[done_indices] = self.env.motion_manager.motion_ids[done_indices].unsqueeze(-1)       

        if self.env.state_init == self.env.StateInit.Default:
            # self.reset_latents(done_indices, "gaussian")
            self.reset_latents(done_indices, self.latent_init_method["def_envs"])
        elif self.env.state_init == self.env.StateInit.Data:
            self.reset_latents(done_indices, self.latent_init_method["ref_envs"])
        elif self.env.state_init == self.env.StateInit.Hybrid:
            self.reset_latents(default_env_ids, self.latent_init_method["def_envs"])
            self.reset_latents(ref_env_ids, "ref_data")
        elif self.env.state_init == self.env.StateInit.DataHybrid:
            self.reset_latents(default_env_ids, self.latent_init_method["def_envs"])
            self.reset_latents(ref_env_ids, "ref_data")
        else:
            raise ValueError("Invalid values for env state_init")
            
        obs["latents"] = self.latents.clone()
        
        return obs

    def reset_latents(self, indices, methods):
        if methods == "gaussian":
            self.reset_latents_gaussian(indices)
        elif methods == "ref_data":
            self.reset_latents_ref_data(env_ids=indices, sampling=None)
        elif methods == "ref_data_vmf":
            self.reset_latents_ref_data(env_ids=indices, sampling="vmf")
        elif methods == "fixed":
            self.reset_latents_fixed(env_ids = indices)
        elif methods == "fixed_per_env":
            self.reset_latents_fixed_per_env(env_ids = indices)
        else:
            raise ValueError("latent init method wrong")


        

    def env_step(self, actions):
        """Perform an environment step and inject current latents into observations."""
        obs, rewards, dones, terminated, extras = super().env_step(actions)
        obs["latents"] = self.latents.clone()
        return obs, rewards, dones, terminated, extras

    @torch.no_grad()
    def evaluate_policy(self):
        """Evaluates the policy in evaluation mode."""
        self.eval()
        self.setup_eval_data_dict()

        done_indices = None  # Force reset on first entry
        step = 0
        phi_outputs = []
        n_epi = 0
        rew = 0.0

        # Safely compute data latents only when RGSD is enabled and contrastive module is available
        if getattr(self.config.ssd_parameters, "rgsd_on", False) and hasattr(self, "contrastive_algo"):
            self.data_latents = self.contrastive_algo.compute_ref_data_latents(
                self.model._phi,
                n_frames=self.n_history,
            )

        while self.config.max_eval_steps is None or step < self.config.max_eval_steps:
            obs = self.handle_reset(done_indices)
            
            # Obtain actor predictions
            actions = self.model.act(obs)
            obs, rewards, dones, terminated, extras = self.env_step(actions)
            self.gather_eval_data(extras)
            self.show_eval_data()
            # extra_rew = self.calculate_extra_reward()
            rew += rewards
            # time.sleep(0.1) # se added

             # Collect phi_out
            phi_obs = self._adjust_phi_input(obs,key="historical_self_obs" )
            with torch.no_grad():
                phi_out = self.model._phi(phi_obs)  # [1, 3]
                phi_out = phi_out.squeeze(0).cpu()
            phi_outputs.append(phi_out)  # Save as [3] tensor

            all_done_indices = dones.nonzero(as_tuple=False)
            done_indices = all_done_indices.squeeze(-1)
            step += 1

            skill_vec = self.latents[0].cpu().numpy()
            # print(step, rew.item(), extra_rew.item(), skill_vec[:3], phi_out[:3])



            if dones[0] or (self.env.config.sync_motion and step % 150==0):
            # if step % 200==0:
                n_epi += self.num_envs
                print('--------------------"', n_epi)
                step, rew = 0, 0 

                self.draw_latent_plot(phi_outputs, n_epi)
                phi_outputs = []

                aa=1


    def draw_latent_plot(self, phi_outputs, n_epi):
        if not self.config.ssd_parameters.draw_latent_plot:
            return None
    
        skill_vec = self.latents[0].cpu().numpy()
        latent_dim = self.config.ssd_parameters.latent_dim 
        # latent_dim = 2
        


        # Stack and plot at the end
        phi_outputs = torch.stack(phi_outputs)[:,0,:].numpy()  # Shape: [T, 3]

        file_name = os.path.join(self.plot_path, "my_plot_epoch_eval_aft")
        if not hasattr(self, "contrastive_algo"):
            return None
        ax = self.contrastive_algo.draw_latent_figure(
            self.model._phi,
            "eval_aft",
            file_name,
            n_frames=self.n_history,
        )
        # fig = plt.figure(figsize=(10, 6))
        # ax = fig.add_subplot()
        # ax = fig.add_subplot(111, projection='3d')


        ax.set_title('Trajectory in Latent Space (phi_out)')
        ax.set_xlabel('Latent Dim 1')
        ax.set_ylabel('Latent Dim 2')

        if latent_dim == 2:
            ax.plot(phi_outputs[:, 0], phi_outputs[:, 1], marker='x')
            # ax.arrow(0, 0, skill_vec[0]*1, skill_vec[1]*1,
            #     head_width=3.0, head_length=3.0, fc='red', ec='red', length_includes_head=True)
            ax.arrow(0, 0, skill_vec[0]*1, skill_vec[1]*1, fc='red', ec='red')
            for i in range(0, len(phi_outputs[1:]), 5):
                x  = phi_outputs[i+1]
                ax.text(x[0], x[1], str(i), fontsize=8, color='blue', ha='right', va='bottom')

        else:
            ax.plot(phi_outputs[:, 0], phi_outputs[:, 1], phi_outputs[:, 2], marker='x')
            ax.quiver(0, 0, 0, skill_vec[0], skill_vec[1], skill_vec[2], 
                    length=0.2, color='black')
            for i in range(0, len(phi_outputs[1:]), 5):
                x  = phi_outputs[i+1]
                ax.text(x[0], x[1], x[2], str(i), fontsize=8, color='blue', ha='right', va='bottom')

        
        plt.grid(True)
        # ax.set_aspect('equal', adjustable='box')


        file_name = os.path.join(self.plot_path, f"latent_figure_{n_epi}.png")
        plt.savefig(file_name)

    # -----------------------------
    # Reward Calculation
    # -----------------------------
    @torch.no_grad()
    def calculate_extra_reward(self):
        """Calculates extra reward based on Mutual Information."""
        raise NotImplementedError(
                            f"evaluate_policy not implemented"
                        )
        

    def _adjust_phi_input(self, obs, key="historical_self_obs"):
        new_obs = {}
        if (not self.config.contrastive_parameters.attach_encoder_phase_var) and \
            self.env.config.humanoid_obs.attach_phase_variable:
            n_data =  obs[key].size()[0]

            if self.n_history==1:
                new_obs[key] = obs[key][:,1:]
            else:
                history_self = obs[key].view(n_data,self.n_history,-1)
                history_self= history_self[:,:,1:]
                new_obs[key] = history_self.reshape(n_data, -1)
        else:
            return obs
        return new_obs

        
    def calculate_extra_actor_loss(self, batch_dict, dist) -> Tuple[Tensor, Dict]:
        """Adds the diversity loss, if enabled.

        Args:
            batch_dict (dict): Batch of data from the experience buffer.
            dist (torch.distributions.Distribution): Action distribution.

        Returns:
            Tuple[Tensor, Dict]: Extra actor loss and logging dictionary.
        """
        extra_loss, extra_actor_log_dict = super().calculate_extra_actor_loss(
            batch_dict, dist
        )

        if self.config.ssd_parameters.diversity_bonus <= 0:
            return extra_loss, extra_actor_log_dict

        diversity_loss = self.diversity_loss(batch_dict) * self.config.ssd_parameters.diversity_bonus

        extra_actor_log_dict["actor/diversity_loss"] = diversity_loss.clone().detach()

        return (
            extra_loss
            + diversity_loss ,
            extra_actor_log_dict,
        )

    def diversity_loss(self, batch_dict):
        """Calculates the diversity loss to encourage latents to lead to diverse behaviors.

        Args:
            batch_dict (dict): Batch of data from the experience buffer.

        Returns:
            Tensor: Diversity loss.
        """
        prev_latents = batch_dict["latents"]
        new_latents = self.sample_latents(batch_dict["self_obs"].shape[0])
        batch_dict["latents"] = new_latents
        new_dist = self.model._actor(batch_dict)

        batch_dict["latents"] = prev_latents
        old_dist = self.model._actor(batch_dict)

        clipped_new_mu = torch.clamp(new_dist.mean, -1.0, 1.0)
        clipped_old_mu = torch.clamp(old_dist.mean, -1.0, 1.0)

        mu_diff = clipped_new_mu - clipped_old_mu
        mu_diff = torch.mean(torch.square(mu_diff), dim=-1)

        z_diff = new_latents * prev_latents
        z_diff = torch.sum(z_diff, dim=-1)
        z_diff = 0.5 - 0.5 * z_diff

        diversity_bonus = mu_diff / (z_diff + 1e-5)
        if "ref_indices" in batch_dict:
            ref_indices = batch_dict["ref_indices"]
            diversity_bonus[ref_indices] = 0.

        diversity_loss = torch.square(1. - diversity_bonus).mean()

        return diversity_loss

