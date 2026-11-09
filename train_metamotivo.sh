#!/bin/bash
#SBATCH --job-name train_metamotivo
#SBATCH --output logs/train_metamotivo.out
#SBATCH --error logs/train_metamotivo.err
#SBATCH --nodes 1
#SBATCH --qos short
#SBATCH -G a40:1
#SBATCH -c 8
#SBATCH -p ha-lab


source /nethome/atrinh31/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate isaacgym
cd /nethome/atrinh31/flash/repos/ImitateForDiscover
wandb login 531fce769e43363414f995984f1b8fe15c97a6e1

export LD_LIBRARY_PATH=/nethome/atrinh31/miniconda3/envs/isaacgym/lib:$LD_LIBRARY_PATH
export NCCL_IB_DISABLE=1

srun --ntasks=1 python -u -m protomotions.train_agent_mm +exp=metamotivo_mlp \
+motion_file=data/yaml_files/punch.yaml \
+robot=smpl +simulator=isaacgym +experiment_name=train_metamotivo +opt=wandb \
++num_envs=500