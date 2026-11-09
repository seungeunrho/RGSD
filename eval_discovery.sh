#!/bin/bash

DOC_FILE="eval_test.txt"
CMD_PREFIX="python protomotions/eval_agent.py \
                +headless=True \
                +robot=smpl \
                +simulator=isaacgym \
                +motion_file=data/yaml_files/all.yaml \
                +load_checkpoint=True \
                ++num_envs=100 \
                ++agent.config.eval_metric_keys=['cartesian_err','fid','state_coverage','plot'] \
                ++env.config.eval_metric_keys=['cartesian_err','fid','state_coverage','plot'] \
                ++env.config.enable_height_termination=False \
                ++env.config.terminate_ref_deviation=False \
                ++env.config.terminate_ref_timeover=True \
                ++agent.config.eval_doc_file=$DOC_FILE"

#-----------------------------------------------------------#

# RGSD_MODEL="rgsd_ang_fix_punch30"
# RGSD_VERSION=0
# RGSD_CKPT=21000

# # RGSD DISCOVER
# CMD="$CMD_PREFIX \
#         +exp=rgsd/four_motions_mlp \
#         +checkpoint=results/$RGSD_MODEL/lightning_logs/version_$RGSD_VERSION/model_$RGSD_CKPT.ckpt\
#         ++agent.config.vmf_kappa=50.0 \
#         ++agent.config.ssd_parameters.latent_init_method.ref_envs=ref_data_vmf"
# eval "$CMD"

# RGSD_MODEL="rgsd_ang_fix_punch28"
# RGSD_VERSION=0
# RGSD_CKPT=21000

# # RGSD DISCOVER
# CMD="$CMD_PREFIX \
#         +exp=rgsd/four_motions_mlp \
#         +checkpoint=results/$RGSD_MODEL/lightning_logs/version_$RGSD_VERSION/model_$RGSD_CKPT.ckpt\
#         ++agent.config.vmf_kappa=50.0 \
#         ++agent.config.ssd_parameters.latent_init_method.ref_envs=ref_data_vmf"
# eval "$CMD"

# RGSD_MODEL="rgsd_discoverratio_70"
# RGSD_VERSION=0
# RGSD_CKPT=21000

# # RGSD DISCOVER
# CMD="$CMD_PREFIX \
#         +exp=rgsd/four_motions_mlp \
#         +checkpoint=results/$RGSD_MODEL/lightning_logs/version_$RGSD_VERSION/model_$RGSD_CKPT.ckpt\
#         ++agent.config.vmf_kappa=50.0 \
#         ++agent.config.ssd_parameters.latent_init_method.ref_envs=ref_data_vmf"
# eval "$CMD"


#-----------------------------------------------------------#
# DIAYN_MODEL="train_diayn"
# DIAYN_VERSION=0
# DIAYN_CKPT=9000

# echo "Evaluating $DIAYN_MODEL , version $DIAYN_VERSION, ckpt $DIAYN_CKPT, mimic"
# CMD="$CMD_PREFIX \
#         +exp=diayn_mlp \
#         +checkpoint=results/$DIAYN_MODEL/lightning_logs/version_$DIAYN_VERSION/model_$DIAYN_CKPT.ckpt \
#         ++agent.config.ssd_parameters.latent_init_method.ref_envs=best_latent \
#         ++agent.config.ssd_parameters.latent_init_method.def_envs=best_latent"
# eval "$CMD"

# DIAYN_MODEL="train_diayn_rsi"
# DIAYN_VERSION=0
# DIAYN_CKPT=18000

# echo "Evaluating $DIAYN_MODEL , version $DIAYN_VERSION, ckpt $DIAYN_CKPT, mimic"
# CMD="$CMD_PREFIX \
#         +exp=diayn_mlp \
#         +checkpoint=results/$DIAYN_MODEL/lightning_logs/version_$DIAYN_VERSION/model_$DIAYN_CKPT.ckpt \
#         ++agent.config.ssd_parameters.latent_init_method.ref_envs=best_latent \
#         ++agent.config.ssd_parameters.latent_init_method.def_envs=best_latent"
# eval "$CMD"

# echo "Evaluating $DIAYN_MODEL , version $DIAYN_VERSION, ckpt $DIAYN_CKPT, discovery"
# CMD="$CMD_PREFIX \
#         +exp=diayn_mlp \
#         +checkpoint=results/$DIAYN_MODEL/lightning_logs/version_$DIAYN_VERSION/model_$DIAYN_CKPT.ckpt \
#         ++agent.config.vmf_kappa=30.0 \
#         ++agent.config.ssd_parameters.latent_init_method.ref_envs=best_latent_vmf \
#         ++agent.config.ssd_parameters.latent_init_method.def_envs=best_latent_vmf"
# eval "$CMD"

# #-----------------------------------------------------------#
# ASE_MODEL="ase_new_motions_phase"
# ASE_VERSION=0
# ASE_CKPT=15000

# # ASE MIMIC
# echo "Evaluating $ASE_MODEL , version $ASE_VERSION, ckpt $ASE_CKPT, mimic"
# CMD="$CMD_PREFIX \
#         +exp=ase_mlp \
#         +checkpoint=results/$ASE_MODEL/lightning_logs/version_$ASE_VERSION/model_$ASE_CKPT.ckpt \
#         ++agent.config.ase_parameters.latent_init_method=best_latent_vmf"
# eval "$CMD"

# # ASE DISCOVER
# echo "Evaluating $ASE_MODEL , version $ASE_VERSION, ckpt $ASE_CKPT, discovery"
# CMD="$CMD_PREFIX \
#         +exp=ase_mlp \
#         +checkpoint=results/$ASE_MODEL/lightning_logs/version_$ASE_VERSION/model_$ASE_CKPT.ckpt \
#         ++agent.config.vmf_kappa=30.0 \
#         ++agent.config.ase_parameters.latent_init_method=best_latent_vmf"
# eval "$CMD"

# #-----------------------------------------------------------#
CALM_MODEL="calm_new_motions_phase"
CALM_VERSION=0
CALM_CKPT=9000


echo "Evaluating $CALM_MODEL , version $CALM_VERSION, ckpt $CALM_CKPT, discovery"
CMD="$CMD_PREFIX \
        +exp=calm_mlp \
        +checkpoint=results/$CALM_MODEL/lightning_logs/version_$CALM_VERSION/model_$CALM_CKPT.ckpt \
        ++agent.config.vmf_kappa=50.0 \
        ++agent.config.calm_parameters.latent_init_method=ref_vmf"
eval "$CMD"


CALM_MODEL="calm_new_motions_phase_encfix"
CALM_VERSION=0
CALM_CKPT=12000


echo "Evaluating $CALM_MODEL , version $CALM_VERSION, ckpt $CALM_CKPT, discovery"
CMD="$CMD_PREFIX \
        +exp=calm_mlp \
        +checkpoint=results/$CALM_MODEL/lightning_logs/version_$CALM_VERSION/model_$CALM_CKPT.ckpt \
        ++agent.config.vmf_kappa=50.0 \
        ++agent.config.calm_parameters.latent_init_method=ref_vmf"
eval "$CMD"



#-----------------------------------------------------------#
# METRA_MODEL="train_metra"
# METRA_VERSION=0
# METRA_CKPT=24000

# echo "Evaluating $METRA_MODEL , version $METRA_VERSION, ckpt $METRA_CKPT, mimic"
# CMD="$CMD_PREFIX \
#         +exp=metra_mlp \
#         +checkpoint=results/$METRA_MODEL/lightning_logs/version_$METRA_VERSION/model_$METRA_CKPT.ckpt \
#         ++agent.config.ssd_parameters.latent_init_method.ref_envs=best_latent \
#         ++agent.config.ssd_parameters.latent_init_method.def_envs=best_latent"
# eval "$CMD"

# METRA_MODEL="train_metra_rsi"
# METRA_VERSION=0
# METRA_CKPT=24000

# echo "Evaluating $METRA_MODEL , version $METRA_VERSION, ckpt $METRA_CKPT, mimic"
# CMD="$CMD_PREFIX \
#         +exp=metra_mlp \
#         +checkpoint=results/$METRA_MODEL/lightning_logs/version_$METRA_VERSION/model_$METRA_CKPT.ckpt \
#         ++agent.config.ssd_parameters.latent_init_method.ref_envs=best_latent \
#         ++agent.config.ssd_parameters.latent_init_method.def_envs=best_latent"
# eval "$CMD"
