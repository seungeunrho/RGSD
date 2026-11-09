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

from enum import Enum
from typing import Optional

import numpy as np
import torch
from hydra.utils import instantiate, get_class
from torch import Tensor
from isaac_utils import rotations, torch_utils

from protomotions.simulator.base_simulator.simulator import Simulator
from protomotions.simulator.base_simulator.config import MarkerConfig, VisualizationMarker, MarkerState, SimulatorConfig
from protomotions.simulator.base_simulator.robot_state import RobotState
from protomotions.envs.base_env.env_utils.humanoid_utils import (
    compute_humanoid_reset,
)
from protomotions.envs.base_env.env_utils.terrains.terrain import Terrain
from protomotions.envs.base_env.env_utils.terrains.terrain_config import TerrainConfig
from protomotions.envs.base_env.components.humanoid_obs import HumanoidObs
from protomotions.envs.base_env.components.terrain_obs import TerrainObs
from protomotions.envs.base_env.components.motion_manager import MotionManager

from protomotions.utils.motion_lib import MotionLib
from protomotions.utils.scene_lib import SceneLib

from protomotions.envs.ssd_env.env import SSDEnv
from protomotions.envs.base_env.env import BaseEnv
from protomotions.envs.goal_reaching.base_env import GoalReachingBaseEnv
from protomotions.utils.average_meter import AverageMeter, TensorAverageMeterDict




class WalkGoalReachingEnv(GoalReachingBaseEnv):
    def __init__(self, config, device: torch.device, *args, **kwargs):
        self.dist_to_goal_meter = AverageMeter(1, 1000).to(device)
        self.prev_dist = torch.zeros(config.num_envs, device=device)

        super().__init__(config, device, *args, **kwargs)

        
        # Goal sampling box parameters (in character's local frame)
        self.goal_box_min = torch.tensor(
            getattr(self.config, "goal_box_min", [5.0, -10.0, 0.8]), 
            device=self.device
        )
        self.goal_box_max = torch.tensor(
            getattr(self.config, "goal_box_max", [20, 10.0, 0.8]), 
            device=self.device
        )
        # self.goal_dim = 4  #  2 for root_pos_xy,  2 for goal_pos_xy
        self.goal_dim = 14  #  2 for root_pos_xy,  2 for goal_pos_xy

        self.goal_obs = torch.zeros(self.num_envs, self.goal_dim, device=self.device)
        
        # Initialize goal offsets for each environment (will be randomized)
        self.goal_offsets = torch.zeros(self.num_envs, 3, device=self.device)
        # This will store the calculated world-frame goal positions for each agent
        self.goal_positions = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.success_radius = self.config.goal_success_radius
        self.phase_reset_interval = self.config.phase_reset_interval
                    
        # Sample initial random goals
        self._sample_new_goals(torch.arange(self.num_envs, device=self.device))

        # Resolve hand indices robustly across assets (AMP uses "left_hand"/"right_hand", SMPL uses "L_Hand"/"R_Hand")
        name_pair = ["Pelvis"]

        self._update_goal_positions()

    def step(self, actions):
        """Before stepping, update goal positions based on current root state."""
        self._update_goal_positions()

        obs, rew, reset, extras = super().step(actions)
        self.simulator._draw_lines()

        if reset.sum() > 0:
            extras["success_ratio"] = self.goal_success.clone()
            extras["dist_to_goal"] = self.prev_dist.clone()
        return obs, rew, reset, extras

    def get_obs(self):
        obs = super().get_obs()
        # Provide goal as a local-frame offset in observations (per-environment)
        root_state = self.simulator.get_root_state()
        root_pos = root_state.root_pos
        root_rot = root_state.root_rot
        root_vel = root_state.root_vel
        root_ang_vel = root_state.root_ang_vel

        if self.config.humanoid_obs.attach_phase_variable:
            obs['self_obs'][:,0] = torch.fmod(
                    obs['self_obs'][:,0],
                    self.phase_reset_interval,
                )

        my_pos = root_pos[:,:2] - self.respawn_position[:,:2]
        goal_pos = self.goal_offsets.clone()
        # print("pos", my_pos[0], "goal", goal_pos[0,:2] )
        obs["goal"] = torch.concatenate([my_pos, root_rot, root_vel, root_ang_vel, goal_pos[:,:2]], dim=-1)
        # obs["goal"] = torch.concatenate([my_pos, goal_pos[:,:2]], dim=-1)

        return obs

    def _sample_new_goals(self, env_ids):
        """Sample new random goal positions within the reachable box for specified environments."""
        num_envs = len(env_ids)
        if num_envs == 0:
            return
            
        # Sample random positions within the box for each dimension
        random_offsets = torch.rand(num_envs, 3, device=self.device)
        
        # Scale to the goal box range
        box_size = self.goal_box_max - self.goal_box_min
        sampled_goals = self.goal_box_min + random_offsets * box_size

        self.dist_to_goal_meter.update(self.prev_dist[env_ids])
        
        # Update the goal offsets for the specified environments
        self.goal_offsets[env_ids] = sampled_goals
        self.prev_dist[env_ids] = torch.norm(self.goal_offsets[env_ids,:2], dim=-1)
        self.goal_success[env_ids] = False

        goal_pos = self.goal_offsets[0].cpu().numpy()
        self.simulator._clean_lines()
        self.simulator._draw_goal(goal_pos[0], goal_pos[1], 0.5)


    def _update_goal_positions(self):
        """Calculates world-frame goal positions from local offsets for each agent."""
        root_state = self.simulator.get_root_state()
        self.goal_positions = self.respawn_position + self.goal_offsets
        # The goal should be at least at a certain height
        self.goal_positions[..., 2] = torch.clamp(self.goal_positions[..., 2], min=0.2)


    def compute_reward(self):
        root_state = self.simulator.get_root_state()
        root_pos = root_state.root_pos

        dist_to_goal = torch.norm(root_pos[:,:2] - self.goal_positions[:,:2], dim=-1)
        dist_to_goal_squared = torch.square(dist_to_goal)
        dist_change = self.prev_dist - dist_to_goal
        self.prev_dist[:] = dist_to_goal[:]

        rew =  torch.exp(-0.25*dist_to_goal)
        self.goal_success = torch.logical_or(self.goal_success, 
                                             torch.lt(dist_to_goal, self.success_radius))
        # rew += self.goal_success.float() * 10.0
        
        self.rew_buf[:] = rew[:]

        

    def compute_reset(self):
        """Extend base resets with success termination when reaching goal."""
        super().compute_reset()
        self.reset_buf[:] = torch.logical_or(self.reset_buf, self.goal_success)
        self.terminate_buf[:] = torch.logical_or(self.terminate_buf, self.goal_success)
