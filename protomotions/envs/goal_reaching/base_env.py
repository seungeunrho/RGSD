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



class GoalReachingBaseEnv(SSDEnv):
    def __init__(self, config, device: torch.device, *args, **kwargs):
        super().__init__(config, device, *args, **kwargs)


    def reset(self, env_ids: Optional[Tensor] = None):
        """After reset, sample new goals and update goal positions for the new root states."""
        result, _, _ = super().reset(env_ids)
        
        # Sample new goals for reset environments
        if env_ids is not None and len(env_ids) > 0:
            self._sample_new_goals(env_ids)
        # else:
        #     # If no specific env_ids, reset all environments
        #     self._sample_new_goals(torch.arange(self.num_envs, device=self.device))
            
        self._update_goal_positions()
        return result

    def create_visualization_markers(self):
        if self.config.headless:
            return {}

        visualization_markers = super().create_visualization_markers()
        
        goal_markers = [MarkerConfig(size="regular")]
        goal_markers_cfg = VisualizationMarker(
            type="sphere",
            color=(1.0, 0.0, 0.0),
            markers=goal_markers
        )
        visualization_markers["goal_markers"] = goal_markers_cfg
        
        return visualization_markers
    
    def get_markers_state(self):
        if self.config.headless:
            return {}

        markers_state = super().get_markers_state()
        
        markers_state["goal_markers"] = MarkerState(
            translation=self.goal_positions.unsqueeze(1),
            orientation=torch.zeros(self.num_envs, 1, 4, device=self.device),
        )
        
        return markers_state
    
    def _sample_new_goals(self, env_ids):
        raise NotImplementedError

    def _update_goal_positions(self):
        raise NotImplementedError

    def compute_reward(self):
        raise NotImplementedError

    def compute_reset(self):
        super().compute_reset()
