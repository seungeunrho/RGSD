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
from typing import Optional, Tuple
import time
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

from protomotions.envs.base_env.env import BaseEnv

class SSDEnv(BaseEnv):
    class StateInit(Enum):
        Default = 0
        Data = 1
        Hybrid = 2
        DataHybrid = 3
        DataHybridDefault = 4

    def __init__(self, config, device: torch.device, *args, **kwargs):
        super().__init__(config, device, *args, **kwargs)
        self.prev_motion_ids = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        ) - 1

        self.env_reset_default = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.env_reset_ref = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if self.state_init in [self.StateInit.Hybrid, self.StateInit.DataHybrid, self.StateInit.DataHybridDefault]:
            self.num_ref_env = int(self.hybrid_init_prob * self.num_envs)
            ref_ids = env_ids[:self.num_ref_env]
            def_ids = env_ids[self.num_ref_env:]

            self.env_reset_default[def_ids] = True
            self.env_reset_ref[ref_ids] = True
        elif self.state_init == self.StateInit.Data:
            self.num_ref_env = self.num_envs
            self.env_reset_default[env_ids] = False
            self.env_reset_ref[env_ids] = True
        elif self.state_init ==  self.StateInit.Default:
            self.num_ref_env = 0
            self.env_reset_default[env_ids] = True
            self.env_reset_ref[env_ids] = False

        self.deviate_threshold_max_dict = None
        self.crawl_motion_idxs = None
        if self.config.terminate_ref_deviation and (self.env_reset_ref is not None):
            self._get_motion_idxs_for_deviate_ref()
            self.crawl_motion_idxs = self._get_crawl_motion_idxs()

        self.setup_eval_data()
        self.global_step_count = 0
        self.global_epi_count = 0

    def setup_eval_data(self):
        self.eval_data = {}
        if len(self.config.eval_metric_keys) > 0:
            self._get_task_idx()

            if "cartesian_err" in self.config.eval_metric_keys:
                self.eval_data["cartesian_err"] = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
            if "state_coverage" in self.config.eval_metric_keys or 'plot' in self.config.eval_metric_keys:
                self.eval_coverage_const ={
                    "cell_width" : {
                        "walk" : 0.2,
                        "run" : 0.2,
                        "side_step": 0.2,
                        "backward" : 0.2,
                        "punch" : 0.04,
                    },
                    "n_pelvis_cell" : 200,
                    "n_hand_cell" : 30
                }
                n_motion = self.motion_lib.num_motions()
                self.eval_coverage_n_finished_motions = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
                n_pelvis_cell, n_hand_cell = self.eval_coverage_const['n_pelvis_cell'], self.eval_coverage_const['n_hand_cell']
                self.eval_data["state_coverage_per_env"] = {
                    "Pelvis" : torch.zeros((self.num_envs, self.config.max_episode_length, 2), device=self.device, dtype=torch.float),
                    "R_Wrist" : torch.zeros((self.num_envs, self.config.max_episode_length, 3), device=self.device, dtype=torch.float),
                    "L_Wrist" : torch.zeros((self.num_envs, self.config.max_episode_length, 3), device=self.device, dtype=torch.float),
                }

                self.eval_data["state_coverage"] = {
                    "Pelvis" : torch.zeros((n_motion, n_pelvis_cell, n_pelvis_cell), device=self.device, dtype=torch.float),
                    "R_Wrist" : torch.zeros((n_motion, n_hand_cell, n_hand_cell, n_hand_cell), device=self.device, dtype=torch.float),
                    "L_Wrist" : torch.zeros((n_motion, n_hand_cell, n_hand_cell, n_hand_cell), device=self.device, dtype=torch.float),
                }
                self.eval_data["plot"] = {}
                for motion_idx in range(n_motion):
                    self.eval_data["plot"][motion_idx] = {}
                    for body_part in ["Pelvis", "R_Wrist", "L_Wrist"]:
                        self.eval_data["plot"][motion_idx][body_part] = []
                        
                self.cur_cell_size = torch.zeros((self.num_envs,1), device=self.device, dtype=torch.float)
                self._update_cur_cell_size()
            if "plot" in self.config.eval_metric_keys:
                self.ref_traj_dict = self._get_ref_traj()
    

    
    def _get_ref_traj(self):
        motion_lst = self.motion_lib.get_full_motion_state(self.dt, padding=False)
        ref_traj_dict = {}
        body_part_lst = ["Pelvis", "R_Wrist", "L_Wrist"]
        for motion_idx, (motion, _) in enumerate(motion_lst):
            ref_traj_dict[motion_idx] = {}

            gt, gr = (
                motion.rigid_body_pos,
                motion.rigid_body_rot,
            )
            gt2 = gt.clone()


            gt[:,1:] =gt[:,1:] - gt[:, 0:1]
            gt[:,0] =gt[:,0] - gt[0, 0]

            # rr = gr[:, 0]

            # inv_heading = torch_utils.calc_heading_quat_inv(rr, True)
            # # gt2 = self.rotate_pos_to_local(gt2, inv_heading)
            # gt2[:,1:] = gt2[:,1:] - gt2[:,0:1]

            for body_part in body_part_lst:
                body_part_idx = self.config.robot.body_names.index(body_part)
                body_part_traj = gt[:,body_part_idx,:]

                # if body_part == "Pelvis":
                #     body_part_traj = gt[:,body_part_idx,:]
                # else:
                #     body_part_traj = gt2[:,body_part_idx,:]
                    
                ref_traj_dict[motion_idx][body_part] = body_part_traj.cpu().numpy()

        return ref_traj_dict



    def _update_cur_cell_size(self):
        motion_ids = self.motion_manager.motion_ids
        for task in self._task_idx_dict:
            task_ids = self._task_idx_dict[task] 

            task_on = torch.isin(motion_ids, task_ids)
            self.cur_cell_size[task_on] = self.eval_coverage_const["cell_width"][task]
    
    def _motion_name_to_task(self, motion_name):
        motion_name_lst = motion_name.split("/")[3:]
        cur_motion_name = "".join(motion_name_lst)
        
        if "_Walk_turn_" in cur_motion_name:
            return "walk"
        elif "_side_step_" in cur_motion_name:
            return "side_step"
        elif "_run_turn_" in cur_motion_name:
            return "run"
        elif "_Crawl_" in cur_motion_name:
            return "crawl"
        elif "Punches" in cur_motion_name:
            return "punch"
        elif "_backwards_" in cur_motion_name:
            return "backward"
        else:
            return ""


    def _get_task_idx(self):
        self._task_idx_dict = {
            "walk" : [], "side_step": [], "run":[], "punch":[], "backward":[]
        }
        motion_names = self.motion_lib.state.motion_files
        
        for motion_idx in range(self.motion_lib.num_motions()):
            task_name = self._motion_name_to_task(motion_names[motion_idx])
            self._task_idx_dict[task_name].append(motion_idx)

        for task in self._task_idx_dict :
            idx_lst = self._task_idx_dict[task]
            self._task_idx_dict[task] = torch.tensor(idx_lst, dtype=torch.long, device=self.device)
        

    def _get_crawl_motion_idxs(self):
        keys = ["crawl", "Crawl"]
        motion_names = self.motion_lib.state.motion_files
        
        crawl_motion_idxs = []
        for i, motion_name in enumerate(motion_names):
            for key in keys:
                if key in motion_name:
                    crawl_motion_idxs.append(i)
        
        return torch.tensor(crawl_motion_idxs, device=self.device, dtype=torch.long)

    def _get_motion_idxs_for_deviate_ref(self):
        if isinstance(self.config.deviate_threshold_max, float):
            pass
        else:
            keys = self.config.deviate_threshold_max.keys()  
            assert len(keys) > 1, "At least two keys must be present in deviate_threshold_max_dict"

            key_dict = {}
            motion_names = self.motion_lib.state.motion_files
            
            for key in keys:
                assert isinstance(key, str), "Keys must be strings"
                if key=="Default":
                    continue

                for i, motion_name in enumerate(motion_names):
                    if key in motion_name:
                        key_dict[key] = [i] if key not in key_dict else key_dict[key] + [i]

            #convert key_dict's each values to tensor
            for key in key_dict:
                key_dict[key] = torch.tensor(key_dict[key], device=self.device, dtype=torch.long)

            self.deviate_threshold_max_dict = key_dict

    ###############################################################
    # Getters
    ###############################################################
    def get_obs(self):
        obs = self.self_obs_cb.get_obs()
        terrain_obs = self.terrain_obs_cb.get_obs()
        obs.update(terrain_obs)
        return obs

    def get_action_size(self):
        return self.simulator.num_act

    def get_motion_requires_scene(self, motion_ids):
        """
        Returns a boolean tensor indicating whether each motion requires a scene.
        By default, all motions don't require a scene.
        """
        requires_scene = torch.zeros_like(
            motion_ids, dtype=torch.bool, device=self.device
        )
        return requires_scene

    def get_markers_state(self):
        if self.config.headless:
            return {}
        
        markers_state = {}
        
        # Update terrain markers
        height_maps = self.terrain.get_height_maps(
            self.simulator.get_root_state(), None, return_all_dims=True
        ).view(self.num_envs, -1, 3)
        markers_state["terrain_markers"] = MarkerState(
            translation=height_maps,
            orientation=torch.zeros(self.num_envs, height_maps.shape[1], 4, device=self.device),
        )
        
        return markers_state

    ###############################################################
    # Environment step logic
    ###############################################################
    def step(self, actions):
        self.global_step_count += 1
        actions = self.pre_physics_step(actions)
        clamp_actions = self.config.robot.control.clamp_actions
        if clamp_actions is not None:
            actions = torch.clamp(actions, -clamp_actions, clamp_actions)
            self.log_dict["action_clamp_frac"] = (
                actions.abs() == clamp_actions
            ).sum() / actions.numel()

        markers_state = self.get_markers_state()
        self.simulator.step(actions, markers_state)
        self.post_physics_step()

        if self.simulator.user_requested_reset:
            self.user_reset()

        if self.global_step_count % 3 == 0 and self.global_epi_count > 1:
            self.simulator._draw_lines()

        # self.render()

        return self.get_obs(), self.rew_buf, self.reset_buf, self.extras


    def pre_physics_step(self, actions):
        if self.config.sync_motion:
            actions *= 0.0
            # actions *= 1 # se added
        
        return actions

    def on_epoch_end(self, current_epoch: int):
        pass

    def post_physics_step(self):
        self.progress_buf += 1
        self.self_obs_cb.post_physics_step()

        self.compute_observations()
        self.compute_reward()
        
        if not self.disable_reset:
            self.compute_reset()

        if len(self.config.eval_metric_keys) > 0:
            self.compute_eval_metrics()

        if self.config.sync_motion:
            self.sync_motion()


        self.log_dict["terminate_frac"] = self.terminate_buf.float().mean()

        self.extras["terminate"] = self.terminate_buf
        self.extras["to_log"] = self.log_dict

    @torch.no_grad()
    def compute_eval_metrics(self):
        if "cartesian_err" in self.config.eval_metric_keys:
            cur_cartesian_err = self.compute_cartesian_err().mean(-1)
            self.eval_data["cartesian_err"] += cur_cartesian_err

            done_indices = self.reset_buf.nonzero(as_tuple=False)

            if len(done_indices) > 0:
                mean_cartesian_err = self.eval_data["cartesian_err"][done_indices] / self.progress_buf[done_indices]
                motion_ids = self.motion_manager.motion_ids[done_indices]

                self.extras["cartesian_err"] = (done_indices, motion_ids, mean_cartesian_err)
                self.eval_data["cartesian_err"][done_indices] = 0.

        if "state_coverage" in self.config.eval_metric_keys or 'plot' in self.config.eval_metric_keys:
            self._update_cur_cell_size()

            current_state = self.simulator.get_bodies_state()
            body_pos = current_state.rigid_body_pos  # num_env, 24, 3
            body_pos[:,1:] = body_pos[:, 1:] -  body_pos[:, 0:1]

            local_gt = self._get_local_gt()


            motion_ids = self.motion_manager.motion_ids.clone()
            all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

            for body_part in self.eval_data["state_coverage"]:
                coverage_tensor_per_env = self.eval_data["state_coverage_per_env"][body_part]
                if body_part == "Pelvis":
                    # n_cell = self.eval_coverage_const["n_pelvis_cell"]
                    body_part_idx = self.config.robot.body_names.index(body_part)
                    body_part_pos_global = body_pos[:,0,:]
                    init_pos = self.respawn_position
                    body_part_pos_local = body_part_pos_global[:,:2] - init_pos[:,:2]
                    coverage_tensor_per_env[all_env_ids,self.progress_buf] = body_part_pos_local

                
                elif body_part in ["R_Wrist", "L_Wrist"]:
                    # n_cell = self.eval_coverage_const["n_hand_cell"]
                    body_part_idx = self.config.robot.body_names.index(body_part)
                    # body_part_pos_local = body_pos[:,body_part_idx,:]
                    body_part_pos_local = local_gt[:,body_part_idx,:]

                    coverage_tensor_per_env[all_env_ids,self.progress_buf] = body_part_pos_local
                else:
                    raise ValueError("Improper body_part name")
                

            done_indices = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if len(done_indices) > 0:
                time_over_ref = self.compute_ref_timeover()
                not_time_over = torch.logical_not(time_over_ref[done_indices].squeeze(-1))
                self.eval_coverage_n_finished_motions[done_indices] += 1
                motion_ids = motion_ids[done_indices].flatten().repeat_interleave(self.config.max_episode_length, dim=0)

                
                for body_part in self.eval_data["state_coverage_per_env"]:
                    coverage_tensor = self.eval_data["state_coverage"][body_part] 
                    body_part_pos_traj = self.eval_data["state_coverage_per_env"][body_part][done_indices].clone()

                    # if done, but not reached motions end, it means robot failed -> mask out
                    body_part_pos_traj[not_time_over] = 0. 
                    body_part_pos_traj_flat = body_part_pos_traj.view(len(done_indices)* self.config.max_episode_length, -1)

                    if body_part == "Pelvis":
                        n_cell = self.eval_coverage_const["n_pelvis_cell"]

                        idx = torch.div(body_part_pos_traj_flat, 0.2).long() + int(n_cell/2)
                        idx = torch.clamp(idx, min=0, max=n_cell-1)

                        coverage_tensor[motion_ids, idx[:,0], idx[:,1]] += 1
                    
                    elif body_part in ["R_Wrist", "L_Wrist"]:
                        n_cell = self.eval_coverage_const["n_hand_cell"]
                       
                        idx = torch.div(body_part_pos_traj_flat, 0.04).long() + int(n_cell/2)
                        
                        clamp_val = int(n_cell/2)
                        idx[:,0] = torch.clamp(idx[:,0], min=clamp_val, max=n_cell-1)
                        idx[:,2] = torch.clamp(idx[:,2], min=clamp_val, max=n_cell-1)
                        idx[:,1] = torch.clamp(idx[:,1], min=0, max=n_cell-1)

                        coverage_tensor[motion_ids, idx[:,0], idx[:,1], idx[:,2]] += 1

                    if 'plot' in self.config.eval_metric_keys:
                        for idx in done_indices:
                            motion_idx = self.motion_manager.motion_ids[idx].item()
                            self.eval_data["plot"][motion_idx][body_part].append(
                                self.eval_data["state_coverage_per_env"][body_part][idx].
                                clone().
                                squeeze(0).
                                cpu().
                                numpy()
                            )

                    self.eval_data["state_coverage_per_env"][body_part][done_indices] = 0.
            
            finished_envs = self.eval_coverage_n_finished_motions >= self.motion_lib.num_motions()
            if finished_envs.sum() >= self.num_envs:
                if "state_coverage" in self.config.eval_metric_keys:
                    self.extras["state_coverage"] = self.eval_data["state_coverage"]
                if 'plot' in self.config.eval_metric_keys:
                    self.extras["plot"] = self.eval_data["plot"]
                    self.extras["ref_traj"] = self.ref_traj_dict


    def user_reset(self):
        self.progress_buf[:] = 1e6

    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            is_reset= False
        else:
            is_reset= True

        self.self_obs_cb.compute_observations(env_ids, is_reset=is_reset)
        self.terrain_obs_cb.compute_observations(env_ids)

    def compute_reset(self):
        bodies_positions = self.simulator.get_bodies_state().rigid_body_pos
        bodies_contact_buf = self.self_obs_cb.body_contacts.clone()

        self.reset_buf[:], self.terminate_buf[:] = self.compute_humanoid_reset(
            self.reset_buf,
            self.progress_buf,
            bodies_contact_buf,
            self.non_termination_contact_body_ids,
            bodies_positions,
            self.config.max_episode_length,
            self.config.enable_height_termination,
            self.termination_heights
            + self.terrain.get_ground_heights(bodies_positions[:, 0]),
        )

        if self.config.terminate_ref_deviation and (self.env_reset_ref is not None):
            deviate_ref = self.compute_deviate_ref()  # se added
            # if self.env_reset_default is not None:
            #     deviate_ref[self.env_reset_default ] = False
            self.reset_buf[:] = torch.logical_or(self.reset_buf, deviate_ref)
            self.terminate_buf[:] = torch.logical_or(self.terminate_buf, deviate_ref)

        if self.config.terminate_ref_timeover and (self.env_reset_ref is not None):
            time_over_ref = self.compute_ref_timeover()
            time_over_ref = torch.logical_and(time_over_ref, self.env_reset_ref)
            # if self.env_reset_default is not None:
            #     time_out_ref[self.env_reset_default ] = False
            self.reset_buf[:] = torch.logical_or(self.reset_buf, time_over_ref)
            self.terminate_buf[:] = torch.logical_or(self.terminate_buf, time_over_ref)

        time_out_ref = self.compute_timeout_ref()
        self.reset_buf[:] = torch.logical_or(self.reset_buf, time_out_ref)
        self.terminate_buf[:] = torch.logical_or(self.terminate_buf, time_out_ref)

    
    def compute_timeout_ref(self):
        if self.config.humanoid_obs.attach_phase_variable:
            phase = self.self_obs_cb.humanoid_obs[:,0]
        else:
            phase = self.progress_buf * self.dt

        return phase > self.config.max_episode_length * self.dt

    def compute_ref_timeover(self):
        if self.config.humanoid_obs.attach_phase_variable:
            phase = self.self_obs_cb.humanoid_obs[:,0]
        else:
            phase = self.progress_buf * self.dt

        motion_len = self.motion_lib.get_motion_length(self.motion_manager.motion_ids)
        time_passed = phase > motion_len

        reset = torch.logical_and(time_passed, self.env_reset_ref)


        return reset
        

    def compute_reward(self):
        self.rew_buf[:] = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )

    def compute_deviate_ref(self):
        """
        Abbreviations:

        gt = global translation
        gr = global rotation
        rt = root translation
        rr = root rotation
        kb = key bodies
        dv = dof (degrees of freedom velocity)
        """
        if self.config.humanoid_obs.attach_phase_variable:
            phase = self.self_obs_cb.humanoid_obs[:,0]
        else:
            phase = self.progress_buf * self.dt

        cartesian_err = self.compute_cartesian_err()

        cartesian_err_max = cartesian_err.max(-1)[0]
        cartesian_mean = cartesian_err.mean(-1)

        if self.deviate_threshold_max_dict is not None:
            assert "Default" in self.config.deviate_threshold_max, "Default key must be present in deviate_threshold_max_dict"    
            deviate_ref_max = cartesian_err_max > self.config.deviate_threshold_max["Default"]
            
            for key in self.deviate_threshold_max_dict:
                cur_motion_ids = self.deviate_threshold_max_dict[key] 
                motion_ids = self.motion_manager.motion_ids

                indices = torch.isin(motion_ids, cur_motion_ids).nonzero(as_tuple=True)[0]
                deviate_ref_max[indices] = cartesian_err_max[indices] > self.config.deviate_threshold_max[key]  

        else:
            deviate_ref_max = cartesian_err_max > self.config.deviate_threshold_max

        # print("deviation", self.progress_buf[0].item(), cartesian_err_max)

        deviate_ref = cartesian_mean > self.config.deviate_threshold
        deviate_ref = torch.logical_or(deviate_ref, deviate_ref_max)
        motion_len = self.motion_lib.get_motion_length(self.motion_manager.motion_ids)
        time_passed = phase > motion_len
        # time_early = phase < 0.1


        deviate_ref[time_passed] = False  # we compute deviate_ref only before the episode ends.
        # deviate_ref[time_early] = False  # we compute deviate_ref only before the episode ends.
        reset = torch.logical_and(deviate_ref, self.env_reset_ref)

        return reset
    
    @torch.no_grad()
    def _get_local_gt(self):
        current_state = self.simulator.get_bodies_state()
        gt, gr = (
            current_state.rigid_body_pos,
            current_state.rigid_body_rot,
        )
        # first remove height based on current position
        # relative_to_data_gt = gt.clone()
        # relative_to_data_gt[:, :, -1:] -= self.terrain.get_ground_heights(gt[:, 0]).view(
        #     self.num_envs, 1, 1
        # )
        # # then remove offset to get back to the ground-truth data position
        # relative_to_data_gt[..., :2] -= self.respawn_offset_relative_to_data.clone()[
        #     ..., :2
        # ].view(self.num_envs, 1, 2)
        gt2 = gt.clone()

        rr = gr[:, 0]

        inv_heading = torch_utils.calc_heading_quat_inv(rr, True)
        # gt2 = self.rotate_pos_to_local(gt2, inv_heading)



        gt2[:,1:] = gt2[:,1:] - gt2[:,0:1]


        # local_gt[:,1:] = local_gt[:,1:] - local_gt[:,0:1]
        return gt2

    
    @torch.no_grad()
    def compute_cartesian_err(self):
        if self.config.humanoid_obs.attach_phase_variable:
            phase = self.self_obs_cb.humanoid_obs[:,0]
        else:
            phase = self.progress_buf * self.dt

        ref_state = self.motion_lib.get_motion_state(
            self.motion_manager.motion_ids,
            phase,
        )

        
        ref_gt = ref_state.rigid_body_pos
        ref_gr = ref_state.rigid_body_rot
        ref_lr = ref_state.local_rot
        ref_lr = ref_lr[:, self.simulator.get_dof_body_ids()]
        
        current_state = self.simulator.get_bodies_state()
        gt, gr = (
            current_state.rigid_body_pos,
            current_state.rigid_body_rot,
        )
        # first remove height based on current position
        relative_to_data_gt = gt.clone()
        relative_to_data_gt[:, :, -1:] -= self.terrain.get_ground_heights(gt[:, 0]).view(
            self.num_envs, 1, 1
        )
        # then remove offset to get back to the ground-truth data position
        relative_to_data_gt[..., :2] -= self.respawn_offset_relative_to_data.clone()[
            ..., :2
        ].view(self.num_envs, 1, 2)

        rr = gr[:, 0]
        ref_rr = ref_gr[:, 0]

        inv_heading = torch_utils.calc_heading_quat_inv(rr, True)
        ref_inv_heading = torch_utils.calc_heading_quat_inv(ref_rr, True)

        local_ref_gt = self.rotate_pos_to_local(ref_gt, ref_inv_heading)
        local_gt = self.rotate_pos_to_local(relative_to_data_gt, inv_heading)
        cartesian_err = (
            ((local_ref_gt - local_ref_gt[:, 0:1]) - (local_gt - local_gt[:, 0:1]))
            .pow(2)
            .sum(-1)
            .sqrt()
        )
        return cartesian_err

    def rotate_pos_to_local(self, pos: Tensor, heading: Optional[Tensor] = None):
        if heading is None:
            raise NotImplementedError("Heading is required for local rotation")
            # root_rot = self.rigid_body_rot[:, 0]
            root_rot = self.get_bodies_state().body_rot[:, 0]
            heading = torch_utils.calc_heading_quat_inv(root_rot, True)

        pos_num_dims = len(pos.shape)
        expanded_heading = heading.view(
            [heading.shape[0]] + [1] * (pos_num_dims - 2) + [heading.shape[1]]
        ).expand(pos.shape[:-1] + (4,))

        rotated = rotations.quat_rotate(
            expanded_heading.reshape(-1, 4), pos.reshape(-1, 3), True
        ).view(pos.shape)
        return rotated


    ###############################################################
    # Handle Resets
    ###############################################################
    def reset_default(self, env_ids):
        # Adjust root position
        default_state = self.default_state

        root_pos = default_state.root_pos[env_ids].clone()
        root_rot = default_state.root_rot[env_ids].clone()
        dof_pos = default_state.dof_pos[env_ids].clone()
        root_vel = default_state.root_vel[env_ids].clone()
        root_ang_vel = default_state.root_ang_vel[env_ids].clone()
        dof_vel = default_state.dof_vel[env_ids].clone()
        rigid_body_pos = default_state.rigid_body_pos[env_ids].clone()
        rigid_body_rot = default_state.rigid_body_rot[env_ids].clone()
        rigid_body_vel = default_state.rigid_body_vel[env_ids].clone()
        rigid_body_ang_vel = default_state.rigid_body_ang_vel[env_ids].clone()

        root_pos[:, :2] = 0
        root_pos[:, :3] += self.get_envs_respawn_position(
            env_ids,
            rigid_body_pos=rigid_body_pos,
            offset=0,
        )

        

        # Transfer entire body to the proper coordinates
        rigid_body_pos[:, :, :3] -= (
            rigid_body_pos[:, 0, :3].unsqueeze(1).clone()
        )
        rigid_body_pos[:, :, :3] += root_pos.unsqueeze(1)
        
        new_states = RobotState(
            root_pos=root_pos,
            root_rot=root_rot,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            rigid_body_pos=rigid_body_pos,
            rigid_body_rot=rigid_body_rot,
            rigid_body_vel=rigid_body_vel,
            rigid_body_ang_vel=rigid_body_ang_vel,
        )

        return new_states
    
    def reset_with_fixed_motion(self, env_ids=None, motion_id=None):
        if motion_id is None:
            raise ValueError("motion_id needed")
        
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        if len(env_ids) > 0:
            if isinstance(env_ids, list):
                env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
            env_ids = env_ids.to(self.device)

            new_states, motion_ids, motion_times = self.reset_ref_state_init(
                    env_ids, 
                    motion_ids = motion_id,
                    motion_times = 0.,
                )
            env_reset_default_env_ids, latent_reset_default_env_ids = [], []
            env_reset_ref_env_ids, latent_reset_default_env_ids = env_ids, env_ids
            reset_ref_motion_ids = motion_ids
            reset_ref_motion_times = motion_times

            self.simulator.reset_envs(new_states, env_ids)
            self.compute_observations(env_ids)

            self.self_obs_cb.reset_envs(
                env_ids,
                env_reset_default_env_ids,
                env_reset_ref_env_ids,
                reset_ref_motion_ids,
                reset_ref_motion_times,
            )

            self.progress_buf[env_ids] = 0
            self.reset_buf[env_ids] = 0
            self.terminate_buf[env_ids] = 0
        
        return self.get_obs()
        


    def reset(self, env_ids=None, return_motion_ids=False):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        #env_reset_default : s_0 = s_def, env_reset_ref : s_0 ~ M
        env_reset_default_env_ids, env_reset_ref_env_ids = [], []

        #latent_reset_default : z ~ N(0,I), latent_reset_ref : z ~ Uniform(Z_ref)
        latent_reset_default_env_ids, latent_reset_ref_env_ids = [], []

        if len(env_ids) > 0:
            if 0 in env_ids:
                self.global_epi_count += 1
                # print(self.global_epi_count, "episodes done.")
            if isinstance(env_ids, list):
                env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
            env_ids = env_ids.to(self.device)

            if self.motion_manager is not None:
                self.motion_manager.reset_envs(env_ids)

            if self.state_init == self.StateInit.Default:
                new_states = self.reset_default(env_ids)
                env_reset_default_env_ids, latent_reset_default_env_ids = env_ids, env_ids
                env_reset_ref_env_ids, latent_reset_default_env_ids = [], []
                reset_ref_motion_ids = []
                reset_ref_motion_times = []
                motion_times = 0.
            elif self.state_init == self.StateInit.Data:
                new_states, motion_ids, motion_times = self.reset_ref_state_init(
                    env_ids
                )
                env_reset_default_env_ids, latent_reset_default_env_ids = [], []
                env_reset_ref_env_ids, latent_reset_default_env_ids = env_ids, env_ids
                reset_ref_motion_ids = motion_ids
                reset_ref_motion_times = motion_times
            elif self.state_init == self.StateInit.Hybrid:
                latent_reset_ref_env_ids = env_ids[env_ids < self.num_ref_env]
                env_reset_ref_env_ids = env_ids[env_ids < self.num_ref_env]
                env_reset_default_env_ids = env_ids[env_ids >= self.num_ref_env]
                latent_reset_default_env_ids = env_ids[env_ids >= self.num_ref_env]


                new_states, motion_ids, motion_times = (
                    self.reset_hybrid_state_init(env_ids, reset_ref_env_ids, reset_default_env_ids)
                )

                reset_ref_motion_ids = motion_ids
                reset_ref_motion_times = motion_times

                # motion_times[reset_default_env_ids] = 0.

            elif self.state_init == self.StateInit.DataHybrid:
                new_states, motion_ids, motion_times = self.reset_ref_state_init(
                    env_ids
                )

                env_reset_ref_env_ids = env_ids[env_ids < self.num_ref_env]
                latent_reset_ref_env_ids = env_ids[env_ids < self.num_ref_env]

                env_reset_default_env_ids = env_ids[env_ids >= self.num_ref_env]
                latent_reset_default_env_ids = env_ids[env_ids >= self.num_ref_env]


                reset_ref_motion_ids = motion_ids
                reset_ref_motion_times = motion_times

            elif self.state_init == self.StateInit.DataHybridDefault:
                ratio_default_envs = 0.1
                num_real_ref_envs = self.num_envs * (1-ratio_default_envs)  #se added


                new_default_states ,env_reset_ref_new_states = None, None
                env_reset_ref_env_ids = env_ids[env_ids < num_real_ref_envs]
                if len(env_reset_ref_env_ids) > 0:
                    env_reset_ref_new_states, motion_ids, motion_times = self.reset_ref_state_init(
                        env_reset_ref_env_ids
                    )

                env_reset_default_env_ids = env_ids[env_ids >= num_real_ref_envs]
                if len(env_reset_default_env_ids) > 0:
                    new_default_states = self.reset_default(env_reset_default_env_ids)
                

                new_states = self.merge_two_states(env_reset_ref_new_states, env_reset_ref_env_ids, new_default_states, env_reset_default_env_ids)

                latent_reset_ref_env_ids = env_ids[env_ids < self.num_ref_env]
                latent_reset_default_env_ids = env_ids[env_ids >= self.num_ref_env]

                reset_ref_motion_ids = motion_ids
                reset_ref_motion_times = motion_times

            else:
                assert False, "Unsupported state initialization strategy: {:s}".format(
                    str(self.state_init)
                )

            self.simulator.reset_envs(new_states, env_ids)
            self.compute_observations(env_ids)

            self.self_obs_cb.reset_envs(
                env_ids,
                env_reset_default_env_ids,
                env_reset_ref_env_ids,
                reset_ref_motion_ids,
                reset_ref_motion_times,
            )

            self.progress_buf[env_ids] = 0
            self.reset_buf[env_ids] = 0
            self.terminate_buf[env_ids] = 0

            # print("hihi",  self.get_obs()['self_obs'][0,0:10])

        if return_motion_ids:
            if len(env_ids)==0:
                return self.get_obs(), []
            else:
                return self.get_obs(), motion_ids
    
        # if self.state_init in [self.StateInit.Hybrid, self.StateInit.DataHybrid, self.StateInit.DataHybridDefault]:
        #         return self.get_obs(), latent_reset_default_env_ids, latent_reset_ref_env_ids

        return self.get_obs(), latent_reset_default_env_ids, latent_reset_ref_env_ids

    def reset_ref_state_init(
        self,
        env_ids,
        motion_ids: Optional[Tensor] = None,
        motion_times: Optional[Tensor] = None,
    ):
        assert not (
            motion_ids is None and motion_times is not None
        ), "Motion times are set, but no corresponding motion ids provided."
        if motion_ids is not None:
            if motion_times is None:
                motion_times = self.motion_lib.sample_time(motion_ids)
            self.motion_manager.motion_ids[env_ids] = motion_ids
            self.motion_manager.motion_times[env_ids] = motion_times

        motion_ids, motion_times = self.motion_manager.get_respawn_info(env_ids)
        # print("motion_id", motion_ids[0].item(), "motion_times", motion_times[0].item())

        requires_scene = None
        if self.scene_lib is not None:
            # Sample scenes for each motion
            requires_scene = self.get_motion_requires_scene(motion_ids)

        ref_state = self.motion_lib.get_motion_state(motion_ids, motion_times)

        root_offset = ref_state.root_pos[:, :2].clone()

        # Ignore root offset from the data
        ref_state.root_pos[:, :2] = 0
        # Sample offset and fix spawn height
        ref_state.root_pos[:, :3] += self.get_envs_respawn_position(
            env_ids,
            rigid_body_pos=ref_state.rigid_body_pos,
            offset=root_offset,
            requires_scene=requires_scene,
        )

        

        # Transfer entire body to the proper coordinates
        ref_state.rigid_body_pos[:, :, :3] -= (
            ref_state.rigid_body_pos[:, 0, :3].unsqueeze(1).clone()
        )
        ref_state.rigid_body_pos[:, :, :3] += ref_state.root_pos.unsqueeze(1)

        return ref_state, motion_ids, motion_times
    


    def merge_two_states(self, states1, env_ids1, states2, env_ids2):
        default_state = self.default_state
        # Create placeholder for all states
        new_states = RobotState(
            root_pos=default_state.root_pos.clone(),
            root_rot=default_state.root_rot.clone(),
            dof_pos=default_state.dof_pos.clone(),
            root_vel=default_state.root_vel.clone(),
            root_ang_vel=default_state.root_ang_vel.clone(),
            dof_vel=default_state.dof_vel.clone(),
            rigid_body_pos=default_state.rigid_body_pos.clone(),
            rigid_body_rot=default_state.rigid_body_rot.clone(),
            rigid_body_vel=default_state.rigid_body_vel.clone(),
            rigid_body_ang_vel=default_state.rigid_body_ang_vel.clone(),
            # key_body_pos = torch.zeros(self.num_envs, 4, 3, device=self.device, dtype=torch.float),
            # local_rot = torch.zeros(self.num_envs, 24, 4, device=self.device, dtype=torch.float),
        )
        new_states = new_states.to_dict()

        # Each reset function will fill in the appropriate states for the reset ids
        if len(env_ids1) > 0:
            states1 = states1.to_dict()
            for key in new_states.keys():
                new_states[key][env_ids1] = states1[key]


        # default_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
        if len(env_ids2) > 0:
            states2 = states2.to_dict()
            for key in new_states.keys():
                new_states[key][env_ids2] = states2[key]


        tot_env_ids = torch.cat([env_ids1, env_ids2])

        # Only return the states for the envs that are being reset
        for key in new_states.keys():
            new_states[key] = new_states[key][tot_env_ids]

        fin_new_states = RobotState.from_dict(new_states)

        return fin_new_states


    def reset_hybrid_state_init(self, env_ids, ref_reset_ids, default_reset_ids):
        # num_envs = env_ids.shape[0]
        # ref_probs = torch_utils.to_torch(
        #     np.array([self.hybrid_init_prob] * num_envs), device=self.device
        # )
        # ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        # ref_reset_ids = env_ids[ref_init_mask]
        motion_ids = motion_times = []

        default_state = self.default_state
        # Create placeholder for all states
        new_states = RobotState(
            root_pos=default_state.root_pos.clone(),
            root_rot=default_state.root_rot.clone(),
            dof_pos=default_state.dof_pos.clone(),
            root_vel=default_state.root_vel.clone(),
            root_ang_vel=default_state.root_ang_vel.clone(),
            dof_vel=default_state.dof_vel.clone(),
            rigid_body_pos=default_state.rigid_body_pos.clone(),
            rigid_body_rot=default_state.rigid_body_rot.clone(),
            rigid_body_vel=default_state.rigid_body_vel.clone(),
            rigid_body_ang_vel=default_state.rigid_body_ang_vel.clone(),
            # key_body_pos = torch.zeros(self.num_envs, 4, 3, device=self.device, dtype=torch.float),
            # local_rot = torch.zeros(self.num_envs, 24, 4, device=self.device, dtype=torch.float),
        )
        new_states = new_states.to_dict()

        # Each reset function will fill in the appropriate states for the reset ids
        if len(ref_reset_ids) > 0:
            ref_states, motion_ids, motion_times = self.reset_ref_state_init(
                ref_reset_ids
            )
            ref_states = ref_states.to_dict()
            for key in new_states.keys():
                new_states[key][ref_reset_ids] = ref_states[key]


        # default_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
        if len(default_reset_ids) > 0:
            default_states = self.reset_default(default_reset_ids)
            default_states = default_states.to_dict()
            for key in new_states.keys():
                new_states[key][default_reset_ids] = default_states[key]

        # Only return the states for the envs that are being reset
        for key in new_states.keys():
            new_states[key] = new_states[key][env_ids]

        fin_new_states = RobotState.from_dict(new_states)

        return fin_new_states, motion_ids, motion_times

    ###############################################################
    # Terrain Helpers
    ###############################################################
    def create_terrain_and_scene_lib(self):
        terrain_config = TerrainConfig.from_dict(self.config.terrain.config)
        TerrainClass = get_class(self.config.terrain._target_)
        self.terrain: Terrain = TerrainClass(
            config=terrain_config,
            num_envs=self.num_envs,
            device=self.device,
        )

        self.scene_lib: SceneLib = None

    def create_motion_manager(self):
        self.motion_manager = MotionManager(self.config.motion_manager, self)

    ###############################################################
    # Helpers
    ###############################################################
    def create_visualization_markers(self):
        if self.config.headless:
            return {}
        
        visualization_markers = {}
        
        terrain_markers = []
        for _ in range(self.terrain.num_height_points):
            terrain_markers.append(MarkerConfig(size="small"))
        terrain_markers_cfg = VisualizationMarker(
            type="sphere",
            color=(0.008, 0.345, 0.224),
            markers=terrain_markers
        )
        visualization_markers["terrain_markers"] = terrain_markers_cfg
        
        return visualization_markers

    def build_termination_heights(self):
        head_term_height = self.config.head_termination_height
        termination_height = self.config.termination_height

        termination_heights = np.array([termination_height] * self.simulator.get_num_bodies())

        head_id = self.config.robot.body_names.index(self.config.robot.head_body_name)

        termination_heights[head_id] = max(
            head_term_height, termination_heights[head_id]
        )

        self.termination_heights = torch_utils.to_torch(
            termination_heights, device=self.device
        )

    def sync_motion(self):
        if self.sync_motion_just_reset.any():
            cur_ids = self.motion_manager.motion_ids[self.sync_motion_just_reset]
            motion_ids = torch.fmod(
                cur_ids + 1,
                self.motion_lib.num_motions(),
            )

            self.motion_manager.motion_ids[self.sync_motion_just_reset] = motion_ids

            self.sync_motion_times[self.sync_motion_just_reset] = 0
            self.sync_motion_just_reset[:] = False

        self.motion_manager.motion_times[:] = self.sync_motion_times

        ref_state = self.motion_lib.get_motion_state(
            self.motion_manager.motion_ids,
            self.motion_manager.motion_times,
        )

        # ref_state.root_vel *= 0
        # ref_state.root_ang_vel *= 0
        # ref_state.dof_vel *= 0
        # ref_state.rigid_body_vel *= 0
        # ref_state.rigid_body_ang_vel *= 0

        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)


        # TODO: for non-scene interactions, sample an initial random offset and then add terrain offset.
        # self.respawn_offset_relative_to_data[~has_scene, :] = 0

        self.simulator.reset_envs(ref_state, env_ids)

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.terminate_buf[env_ids] = 0

        self.sync_motion_times = self.sync_motion_times + self.sync_motion_dt
        # Check for motions that wrapped around
        wrapped_motions = (
            self.sync_motion_times
            >= self.motion_lib.state.motion_lengths[self.motion_manager.motion_ids]
        )
        # Set sync_motion_just_reset to True for wrapped motions
        self.sync_motion_just_reset[wrapped_motions] = True

    def instantiate_motion_lib(self):
        motion_lib: MotionLib = instantiate(
            self.config.motion_lib,
            dof_body_ids=self.simulator.get_dof_body_ids(),
            dof_offsets=self.simulator.get_dof_offsets(),
            key_body_ids=self.key_body_ids,
            device=self.device,
            skeleton_tree=None,
        )
        return motion_lib

    def get_state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

    def compute_humanoid_reset(self,
        reset_buf: Tensor,
        progress_buf: Tensor,
        contact_buf: Tensor,
        non_termination_contact_body_ids: Tensor,
        rigid_body_pos: Tensor,
        max_episode_length: float,
        enable_early_termination: bool,
        termination_heights: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        terminated = torch.zeros_like(reset_buf)

        if enable_early_termination:
            masked_contact_buf = contact_buf.clone()
            masked_contact_buf[:, non_termination_contact_body_ids, :] = 0
            fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
            fall_contact = torch.any(fall_contact, dim=-1)

            body_height = rigid_body_pos[..., 2]
            fall_height = body_height < termination_heights
            fall_height[:, non_termination_contact_body_ids] = False
            fall_height = torch.any(fall_height, dim=-1)

            # jump_height = body_height[:,0] > 1.1
            has_fallen = torch.logical_and(fall_contact, fall_height)
            # height_error = torch.logical_or(has_fallen, jump_height)
            height_error = has_fallen


            # first timestep can sometimes still have nonzero contact forces
            # so only check after first couple of steps
            height_error *= progress_buf > 1
            terminated = torch.where(height_error, torch.ones_like(reset_buf), terminated)

            # Don't terminate crawl motions
            # if (self.crawl_motion_idxs is not None) and len(self.crawl_motion_idxs) > 0:
            #     motion_ids = self.motion_manager.motion_ids
            #     crawl_indices = torch.isin(motion_ids, self.crawl_motion_idxs).nonzero(as_tuple=True)[0]
            #     terminated[crawl_indices] = False



        reset = torch.where(
            progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated
        )

        return reset, terminated