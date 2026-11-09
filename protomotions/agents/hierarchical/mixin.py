import torch
from torch import nn
from torch.distributions import Normal
from typing import Dict, Optional, Tuple
from torch import Tensor
from protomotions.agents.utils.data_utils import DictDataset, ExperienceBuffer
import torch
import os
import logging
from protomotions.agents.ppo.agent import PPO

from torch import Tensor
from protomotions.agents.common.common import weight_init, get_params
from protomotions.utils.average_meter import AverageMeter, TensorAverageMeterDict

import time
from torch.distributions.von_mises import VonMises

from isaac_utils import torch_utils

from rich.progress import track
from protomotions.agents.ppo.utils import discount_values
from torch import distributions, nn


class HierarchicalMixin:
    """Reusable hierarchical control mixin for PPO/AMP-derived agents.

    Expects the concrete class to implement PPO-like hooks and to provide an adapter
    implementing HierarchicalAdapter.
    """
    def setup(self):
        """Setup with hierarchical control integration."""
        base_model, base_actor_opt, base_critic_opt = super().setup()
        self.model, self.actor_optimizer, self.critic_optimizer = self.fabric.setup(
            base_model, base_actor_opt, base_critic_opt
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

        if hasattr(self.model, "_motion_encoder"):
            for p in self.model._discriminator.parameters():
                p.requires_grad = False
                

        self.model.mark_forward_method("act")
        self.model.mark_forward_method("get_action_and_value")

        self.all_motion_encodings = self.model._motion_encoder(self.all_motions)
        return self.model, self.actor_optimizer, self.critic_optimizer
        
    @torch.no_grad()
    def evaluate_policy(self):
        self.setup_eval_data_dict()
        self.update_data_latents()
        self.eval()
        done_indices = None  # Force reset on first entry
        step = 0
        epoch = 0

        hl_prev_action = torch.zeros((self.num_envs, self._hl_output_dim), device=self.device)

        while self.config.max_eval_steps is None or step < self.config.max_eval_steps:
            low_obs = self.handle_reset(done_indices)
            # hl_obs = torch.cat([low_obs['self_obs'], low_obs['goal']], dim=-1)
            hl_obs = torch.cat([low_obs['self_obs'], low_obs['goal'], hl_prev_action], dim=-1)

            # hl_action = self._hl_dist(hl_obs)
            hl_action, hl_neglogp, hl_value = self.get_hl_action_and_value(hl_obs)
            hl_action_normalized = nn.functional.normalize(hl_action, dim=-1)

            motion_ids = self.env.motion_manager.motion_ids
            target_latent = self.data_latents[motion_ids]
            # hl_action_normalized = target_latent

            action_align = 0.5 + 0.5*(hl_action_normalized * target_latent).sum(dim=-1)
            # print(epoch, step, action_align)


            # Step the environment
            hl_rew = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            hl_dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            hl_terminates = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

            for i in range(self.hl_period):
                low_obs["latents"] = hl_action_normalized
                low_action = self.model.act(low_obs)
                low_obs, rewards, dones, terminated, extras = self.env_step(low_action)
                self.gather_eval_data(extras)
                self.show_eval_data()

                hl_rew += rewards * (1- hl_dones.float())
                hl_dones = torch.logical_or(hl_dones, dones)
                hl_terminates = torch.logical_or(hl_terminates, terminated)

            # hl_next_obs = torch.cat([low_obs['self_obs'], low_obs['goal']], dim=-1)

            hl_prev_action = hl_action

            all_done_indices = hl_dones.nonzero(as_tuple=False)
            done_indices = all_done_indices.squeeze(-1)

            if (hl_dones[0]):
                epoch += 1
                step = 0
                print(" ----- epi ends -------")

            step += 1

    def fit(self):
        # Setup experience buffer
        self.experience_buffer = ExperienceBuffer(self.num_envs, self.num_steps).to(
            self.device
        )
        self.experience_buffer.register_key(
            "self_obs", shape=(self._hl_input_dim,)
        )
        self.experience_buffer.register_key(
            "self_next_obs", shape=(self._hl_input_dim,)
        )
        self.experience_buffer.register_key(
            "actions", shape=(self._hl_output_dim,)
        )
        self.experience_buffer.register_key(
            "actions_executed", shape=(self._hl_output_dim,)
        )
        self.experience_buffer.register_key("rewards")
        self.experience_buffer.register_key("extra_rewards")
        self.experience_buffer.register_key("total_rewards")
        self.experience_buffer.register_key("dones", dtype=torch.long)
        self.experience_buffer.register_key("is_first_step", dtype=torch.long)

        self.experience_buffer.register_key("values")
        self.experience_buffer.register_key("next_values")
        self.experience_buffer.register_key("returns")
        self.experience_buffer.register_key("advantages")
        self.experience_buffer.register_key("neglogp")

        self.update_data_latents()

        # self.register_extra_experience_buffer_keys()

        done_indices = None
        if self.fit_start_time is None:
            self.fit_start_time = time.time()
        self.fabric.call("on_fit_start", self)

        prev_dones = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        all_idx = torch.arange(self.num_envs, device=self.device)
        hl_prev_action = torch.zeros((self.num_envs, self._hl_output_dim), device=self.device)

        
        while self.current_epoch < self.config.max_epochs:
            print("dist meter", self.env.dist_to_goal_meter.get_mean())
            self.epoch_start_time = time.time()

            # Set networks in eval mode so that normalizers are not updated
            self.eval()
            with torch.no_grad():
                self.fabric.call("before_play_steps", self)

                for step in track(
                    range(self.num_steps),
                    description=f"Epoch {self.current_epoch}, collecting data...",
                ):
                    low_obs = self.handle_reset(done_indices)
                    # hl_obs = torch.cat([low_obs['self_obs'], low_obs['goal']], dim=-1)
                    hl_obs = torch.cat([low_obs['self_obs'], low_obs['goal'], hl_prev_action.clone()], dim=-1)

                    # hl_action = self._hl_dist(hl_obs)
                    hl_action, hl_neglogp, hl_value = self.get_hl_action_and_value(hl_obs)

                    # motion_ids = self.env.motion_manager.motion_ids
                    # base_latent = self.data_latents[motion_ids]
                    # hl_action_executed = base_latent
                    # hl_action_executed = base_latent + 0.2*hl_action
                    # hl_action_executed = nn.functional.normalize(hl_action_executed, dim=-1)
                    # self.experience_buffer.update_data("actions_executed", step, hl_action_executed)

                    self.experience_buffer.update_data("self_obs", step, hl_obs)
                    hl_action_normalized = nn.functional.normalize(hl_action, dim=-1)
                    self.experience_buffer.update_data("actions_executed", step, hl_action_normalized)


                    self.experience_buffer.update_data("actions", step, hl_action)
                    self.experience_buffer.update_data("neglogp", step, hl_neglogp)
                    self.experience_buffer.update_data("values", step, hl_value.squeeze(-1))
                    self.experience_buffer.update_data("is_first_step", step, prev_dones)

                    # Step the environment
                    hl_rew = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                    hl_dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                    hl_terminates = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

                    for i in range(self.hl_period):
                        # low_obs["latents"] = hl_action
                        low_obs["latents"] = hl_action_normalized

                        low_action = self.model.act(low_obs)
                        low_obs, rewards, dones, terminated, extras = self.env_step(low_action)
                        hl_rew += rewards * (1- hl_dones.float())
                        hl_dones = torch.logical_or(hl_dones, dones)
                        hl_terminates = torch.logical_or(hl_terminates, terminated)

                    hl_prev_action = hl_action
                    

                    hl_next_obs = torch.cat([low_obs['self_obs'], low_obs['goal'], hl_action], dim=-1)
                    self.experience_buffer.update_data("self_next_obs", step, hl_next_obs)

                    all_done_indices = hl_dones.nonzero(as_tuple=False)
                    done_indices = all_done_indices.squeeze(-1)
                    prev_dones = torch.clone(hl_dones)

                    # Update logging metrics with the environment feedback
                    self.post_train_env_step(hl_rew, hl_dones, done_indices, extras, step)

                    self.experience_buffer.update_data("rewards", step, hl_rew/self.hl_period)
                    self.experience_buffer.update_data("dones", step, hl_dones)

                    next_value = self._hl_critic(hl_next_obs).flatten()
                    next_value = next_value * (1 - hl_terminates.float())
                    self.experience_buffer.update_data("next_values", step, next_value)

                    self.step_count += self.get_step_count_increment()

                

                # After data collection, compute rewards, advantages, and returns.
                rewards = self.experience_buffer.rewards
                extra_rewards = self.calculate_extra_reward()

                self.experience_buffer.batch_update_data("extra_rewards", extra_rewards)
                total_rewards = (self.task_reward_coeff*rewards + self.style_reward_coeff*extra_rewards) * 0.1
                # total_rewards = extra_rewards

                self.experience_buffer.batch_update_data("total_rewards", total_rewards)

                advantages = discount_values(
                    self.experience_buffer.dones,
                    self.experience_buffer.values,
                    total_rewards,
                    self.experience_buffer.next_values,
                    self.gamma,
                    self.tau,
                )
                returns = advantages + self.experience_buffer.values
                self.experience_buffer.batch_update_data("returns", returns)

                if self.config.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                self.experience_buffer.batch_update_data("advantages", advantages)

            training_log_dict = self.optimize_model()
            training_log_dict["epoch"] = self.current_epoch
            self.current_epoch += 1
            self.fabric.call("after_train", self)

            # Save model checkpoint at specified intervals before evaluation.
            if self.current_epoch % self.config.manual_save_every == 0:
                self.save(name=f"model_{self.current_epoch}.ckpt")

            if (
                self.config.eval_metrics_every is not None
                and self.current_epoch > 0
                and self.current_epoch % self.config.eval_metrics_every == 0
            ):
                eval_log_dict, evaluated_score = self.calc_eval_metrics()
                evaluated_score = self.fabric.broadcast(evaluated_score, src=0)
                if evaluated_score is not None:
                    if (
                        self.best_evaluated_score is None
                        or evaluated_score >= self.best_evaluated_score
                    ):
                        self.best_evaluated_score = evaluated_score
                        self.save(new_high_score=True)
                training_log_dict.update(eval_log_dict)

            self.post_epoch_logging(training_log_dict)
            self.env.on_epoch_end(self.current_epoch)

            if self.should_stop:
                self.save()
                return

        self.time_report.report()
        self.save()
        self.fabric.call("on_fit_end", self)

    def post_epoch_logging(self, training_log_dict):
        training_log_dict["rewards/hl_success_ratio"] = self.success_ratio_meter.get_mean().item()
        training_log_dict["rewards/hl_dist_to_goal"] = self.min_dist_to_goal_meter.get_mean().item()
        PPO.post_epoch_logging(self, training_log_dict)

    def post_train_env_step(self, rewards, dones, done_indices, extras, step):

        PPO.post_train_env_step(self, rewards, dones, done_indices, extras, step)

        if "success_ratio" in extras:
            self.success_ratio_meter.update(extras["success_ratio"][done_indices])
            self.min_dist_to_goal_meter.update(extras["dist_to_goal"][done_indices])

    def _init_hl(self, cfg, num_envs, device):
        self._hl_cfg = cfg
        self.task_reward_coeff = cfg.task_reward_coeff
        self.style_reward_coeff = cfg.style_reward_coeff

        self.hl_period = int(getattr(cfg, "period", getattr(cfg, "hl_period", 10)))
        self._hl_entropy_coef = float(getattr(cfg, "entropy_coef", 0.0))
        self._hl_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        
        self._hl_input_dim = int(self.env.config.robot.self_obs_size)+ self.env.goal_dim + self.latent_dim
        self._hl_output_dim = self.latent_dim
        self.kappa_t = torch.as_tensor(1000.0, device=device)

        hidden = int(getattr(cfg, "hidden", getattr(cfg, "hl_hidden", 256)))

        # High-level policy network
        self._hl_net = nn.Sequential(
            nn.Linear(self._hl_input_dim, hidden), 
            nn.ReLU(), 
            nn.Linear(hidden, hidden),
            nn.ReLU(), 
            nn.Linear(hidden, hidden),
            nn.ReLU(), 
            nn.Linear(hidden, hidden),
            nn.ReLU(), 
            nn.Linear(hidden, self._hl_output_dim)
        ).to(device)

        # High-level value function
        self._hl_critic = nn.Sequential(
            nn.Linear(self._hl_input_dim, hidden), 
            nn.ReLU(), 
            nn.Linear(hidden, hidden),
            nn.ReLU(), 
            nn.Linear(hidden, hidden),
            nn.ReLU(), 
            nn.Linear(hidden, hidden),
            nn.ReLU(), 
            nn.Linear(hidden, 1)
        ).to(device)

        self._hl_logstd = torch.full((self._hl_output_dim ,), -2.9, device=device)
        # self._hl_logstd = nn.Parameter(
        #     torch.ones(self._hl_output_dim, device=self.device) * -2.9,
        #     requires_grad=True
        # )


    def setup(self):
        out = super().setup()
        # add HL params to actor optimizer group
        if hasattr(self, "actor_optimizer"):
            self.actor_optimizer.add_param_group({"params": self._hl_net.parameters()})
            self.actor_optimizer.add_param_group({"params": [self._hl_logstd]})

        # add HL critic params to critic optimizer group
        if hasattr(self, "critic_optimizer"):
            self.critic_optimizer.add_param_group({"params": self._hl_critic.parameters()})
        # Optionally freeze low-level components

        self.success_ratio_meter = AverageMeter(1, 1000).to(self.device)
        self.min_dist_to_goal_meter = AverageMeter(1, 1000).to(self.device)
        
        
        return out

    def handle_reset(self, done_indices=None):
        obs = self.env.reset(done_indices)
        return obs

    # Save/load HL parameters similar to RGSD
    def get_state_dict(self, state_dict):
        state_dict = super().get_state_dict(state_dict)
        state_dict["high_level"] = self._hl_net.state_dict()
        state_dict["hl_critic"] = self._hl_critic.state_dict()
        state_dict["hl_logstd"] = self._hl_logstd
        return state_dict
    
    
    
    def get_hl_action_and_value(self, obs):
        dist = self._hl_dist(obs)
        action = dist.sample()
        # action_normalized = nn.functional.normalize(action, dim=-1)

        value = self._hl_critic(obs)
        # logstd = self._hl_logstd
        # std = torch.exp(logstd)

        neglogp = self.neglogp(action, dist.mean, self.kappa_t)
        return action, neglogp, value.flatten()
    
    # @staticmethod
    # def neglogp(self, x, mean, kappa_t):
    #     dist = VonMises(mean, concentration=kappa_t)
    #     return -dist.log_prob(x).sum(dim=-1)
    
    # def _hl_dist(self, x):
    #     mu = torch.nn.functional.normalize(self._hl_net(x), dim=-1)
    #     # mean = self._hl_net(x)
    #     return VonMises(mu, concentration=self.kappa_t)
    
   
    def neglogp(self, x, mean, kappa_t):
        std = torch.exp(self._hl_logstd)
        dist = distributions.Normal(mean, std)
        return -dist.log_prob(x).sum(dim=-1)
    
    
    def _hl_dist(self, x):
        mu = self._hl_net(x)
        std = torch.exp(self._hl_logstd)
        dist = distributions.Normal(mu, std)
        return dist



    def load_parameters(self, state_dict):
        super().load_parameters(state_dict)
        if "high_level" in state_dict:
            self._hl_net.load_state_dict(state_dict["high_level"], strict=False)
        if "hl_critic" in state_dict:
            self._hl_critic.load_state_dict(state_dict["hl_critic"], strict=False)
        if "hl_logstd" in state_dict:
            self._hl_logstd = state_dict["hl_logstd"].to(self.device)

    def actor_step(self, batch_dict) -> Tuple[Tensor, Dict]:

        dist = self._hl_dist(batch_dict["self_obs"])
        logstd = self._hl_logstd
        std = torch.exp(logstd)
        neglogp = self.neglogp(batch_dict["actions"], dist.mean, self.kappa_t)

        dones = batch_dict['dones']
        mask = torch.logical_or(batch_dict['is_first_step'], batch_dict['dones'])

        # Compute probability ratio between new and old policy.
        ratio = torch.exp(batch_dict["neglogp"] - neglogp)
        # print(ratio.max(), ratio.min(), ratio.mean())
        surr1 = batch_dict["advantages"] * ratio
        surr2 = batch_dict["advantages"] * torch.clamp(
            ratio, 1.0 - self.e_clip, 1.0 + self.e_clip
        )
        ppo_loss = torch.max(-surr1, -surr2)
        clipped = torch.abs(ratio - 1.0) > self.e_clip
        clipped = clipped.detach().float().mean()

        # ppo_loss[mask] = 0.0

        actor_ppo_loss = ppo_loss.mean()
        actor_loss = actor_ppo_loss 

        log_dict = {
            "actor/ppo_loss": actor_ppo_loss.detach(),
            "actor/clip_frac": clipped.detach(),
            "losses/actor_loss": actor_loss.detach(),
        }
        return actor_loss, log_dict


    
    def critic_step(self, batch_dict) -> Tuple[Tensor, Dict]:
        values = self._hl_critic(batch_dict["self_obs"]).flatten()
        if self.config.clip_critic_loss:
            critic_loss_unclipped = (values - batch_dict["returns"]).pow(2)
            v_clipped = batch_dict["values"] + torch.clamp(
                values - batch_dict["values"],
                -self.config.e_clip,
                self.config.e_clip,
            )
            critic_loss_clipped = (v_clipped - batch_dict["returns"]).pow(2)
            critic_loss_max = torch.max(critic_loss_unclipped, critic_loss_clipped)
            critic_loss = 0.5 * critic_loss_max.mean()
        else:
            critic_loss = 0.5 * (batch_dict["returns"] - values).pow(2).mean()
        log_dict = {"losses/critic_loss": critic_loss.detach()}
        return critic_loss, log_dict
    

    def handle_model_grad_clipping(self, model, optimizer, model_name):
        params = get_params(list(self._hl_net.parameters()))
        grad_norm_before_clip = torch_utils.grad_norm(params)
        if self.config.check_grad_mag:
            bad_grads = (
                torch.isnan(grad_norm_before_clip) or grad_norm_before_clip > 1000000.0
            )
        else:
            bad_grads = torch.isnan(grad_norm_before_clip)

        bad_grads_count = 0
        if bad_grads:
            if self.config.fail_on_bad_grads:
                all_params = torch.cat(
                    [p.grad.view(-1) for p in params if p.grad is not None],
                    dim=0,
                )
                raise ValueError(
                    f"NaN gradient in {model_name}"
                    + f" {all_params.isfinite().logical_not().float().mean().item()}"
                    + f" {all_params.abs().min().item()}"
                    + f" {all_params.abs().max().item()}"
                    + f" {grad_norm_before_clip.item()}"
                )
            else:
                bad_grads_count = 1
                for p in params:
                    if p.grad is not None:
                        p.grad.zero_()

        if self.config.gradient_clip_val > 0:
            self.fabric.clip_gradients(
                model,
                optimizer,
                max_norm=self.config.gradient_clip_val,
                error_if_nonfinite=True,
            )
        grad_norm_after_clip = torch_utils.grad_norm(params)
        clip_dict = {
            f"{model_name}/grad_norm_before_clip": grad_norm_before_clip.detach(),
            f"{model_name}/grad_norm_after_clip": grad_norm_after_clip.detach(),
            f"{model_name}/bad_grads_count": bad_grads_count,
        }
        return clip_dict
    
    # def calculate_extra_reward(self):
    #     hl_action_executed = self.experience_buffer.actions_executed
    #     motion_ids = self.env.motion_manager.motion_ids
    #     target_latent = self.data_latents[motion_ids]
    #     action_align = 0.5 + 0.5*(hl_action_executed * target_latent).sum(dim=-1)
    #     """No extra reward; only environment task reward (same as original RGSD)."""
    #     return action_align
    
    def calculate_extra_reward(self):
        hl_action = self.experience_buffer.actions_executed
        motion_ids = self.env.motion_manager.motion_ids
        target_latent = self.data_latents[motion_ids]
        cos_dist = 0.5 - 0.5*(hl_action * target_latent).sum(dim=-1)

        # squared_err = torch.norm(hl_action - target_latent, dim=-1)
        # squared_err = torch.norm(hl_action - target_latent, dim=-1).square()
        action_align = -1 + torch.exp(-8*cos_dist)
        return action_align
        # return -cos_dist
    
    # def calculate_extra_reward(self):
    #     hl_action = self.experience_buffer.actions
    #     norm_hl_action = torch.norm(hl_action, dim=-1, keepdim=False)
    #     return torch.exp(-4*norm_hl_action)
    
    def extra_optimization_steps(self, batch_dict, batch_idx: int):
        """No extra optimization steps for hierarchical training."""
        return {}
    
        
