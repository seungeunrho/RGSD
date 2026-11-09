# RGSD

Official implementation of RGSD, built on top of [ProtoMotions](https://github.com/NVLabs/ProtoMotions).

> **Important:**</br>
> This codebase builds heavily on [Hydra](https://hydra.cc/) and [OmegaConf](https://omegaconf.readthedocs.io/).<br>
> It is recommended to familiarize yourself with these libraries and how config composition works.

# Installation

First run `git lfs fetch --all` to fetch all files stored in git-lfs.

1. Install [IsaacGym](https://developer.nvidia.com/isaac-gym)
```bash
wget https://developer.nvidia.com/isaac-gym-preview-4
tar -xvzf isaac-gym-preview-4
pip install -e isaacgym/python
```
2. From the repository root, install the package and its dependencies:
```bash
pip install -e .
pip install -r requirements_isaacgym.txt
pip install -e isaac_utils
pip install -e poselib
```

If you get python library errors:
```bash
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib/
```

# Data

The `motion_file` parameter receives a `.yaml` file listing the motions to train on.
The default training set is `data/yaml_files/all.yaml`.

1. Download the [SMPL](https://smpl.is.tue.mpg.de/) v1.1.0 parameters and place them in `data/smpl/`. Rename basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl, basicmodel_m_lbs_10_207_0_v1.1.0.pkl and basicmodel_f_lbs_10_207_0_v1.1.0.pkl to SMPL_NEUTRAL.pkl, SMPL_MALE.pkl and SMPL_FEMALE.pkl.
2. Download the [AMASS](https://amass.is.tue.mpg.de/) dataset.
3. Convert the motions to MotionLib format: `python data/scripts/convert_amass_to_isaac.py <path_to_AMASS_data>`.

Motions can be visualized via kinematic replay: `python protomotions/scripts/play_motion.py <motion file> isaacgym smpl`.

# Training

Train the RGSD low-level policy:
```bash
python protomotions/train_agent.py \
    +exp=rgsd/four_motions_mlp \
    +motion_file=data/yaml_files/all.yaml \
    +robot=smpl \
    +simulator=isaacgym \
    +experiment_name=<experiment_name>
```

`+robot` must be given before `+simulator`.
Results and checkpoints are stored in `results/<experiment_name>/`. Training with an existing experiment name resumes from the last checkpoint.
Add `+opt=wandb` to log to Weights and Biases.

Train a high-level policy for the goal-reaching task on top of a trained RGSD policy:
```bash
python protomotions/train_agent.py \
    +exp=rgsd/side_hierarchical_mlp \
    +env=walk_goal_reaching \
    +motion_file=data/yaml_files/all.yaml \
    +robot=smpl \
    +simulator=isaacgym \
    +experiment_name=<experiment_name> \
    ++num_envs=8192 \
    ++agent._target_=protomotions.agents.rgsd.hierarchical_agent_mixin.RGSD_Hierarchical \
    ++checkpoint=results/<rgsd_experiment>/lightning_logs/version_0/<model>.ckpt \
    ++load_checkpoint=True \
    +agent.config.load_only_model=True \
    +agent.config.reset_epoch_on_load=True
```
Use `rgsd/forward_hierarchical_mlp` or `rgsd/backward_hierarchical_mlp` for the other walking directions.

# Evaluation

Evaluate an RGSD policy (prints per-task Cartesian error):
```bash
python protomotions/eval_agent.py \
    +exp=rgsd/four_motions_mlp \
    +motion_file=data/yaml_files/synthetic_fid_eval.yaml \
    +robot=smpl \
    +simulator=isaacgym \
    +checkpoint=results/<experiment_name>/lightning_logs/version_0/<model>.ckpt \
    +load_checkpoint=True \
    +terrain=flat \
    +headless=True \
    ++num_envs=150
```

Evaluate a high-level policy (runs until stopped):
```bash
python protomotions/eval_agent.py \
    +exp=rgsd/backward_hierarchical_mlp \
    +motion_file=data/yaml_files/synthetic_fid_eval.yaml \
    +robot=smpl \
    +simulator=isaacgym \
    ++checkpoint=results/<experiment_name>/lightning_logs/version_0/<model>.ckpt \
    ++load_checkpoint=True \
    +headless=True \
    ++num_envs=500 \
    ++agent._target_=protomotions.agents.rgsd.hierarchical_agent_mixin.RGSD_Hierarchical
```

Remove `+headless=True` to open the viewer.

# Acknowledgements

This codebase is built on [ProtoMotions](https://github.com/NVLabs/ProtoMotions). Please adhere to the licensing in [LICENSE](LICENSE) and in the respective upstream repositories.

```bibtex
@misc{ProtoMotions,
  title = {ProtoMotions: Physics-based Character Animation},
  author = {Tessler, Chen and Juravsky, Jordan and Guo, Yunrong and Jiang, Yifeng and Coumans, Erwin and Peng, Xue Bin},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/NVLabs/ProtoMotions/}},
}
```

# Dependencies
* PyTorch, [LICENSE](https://github.com/pytorch/pytorch/blob/main/LICENSE)
* PyTorch Lightning, [LICENSE](https://github.com/Lightning-AI/pytorch-lightning/blob/master/LICENSE)
* IsaacGym, [LICENSE](https://developer.download.nvidia.com/isaac/NVIDIA_Isaac_Gym_Pre-Release_Evaluation_EULA_19Oct2020.pdf)
* SMPLSim, [LICENSE](https://github.com/ZhengyiLuo/SMPLSim/blob/0ec11c8dd3115792b8cf0bfeaef64e8c81be592a/LICENSE)
