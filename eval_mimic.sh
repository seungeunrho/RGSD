#!/bin/bash

DOC_FILE="eval_test.txt"
CMD_PREFIX="python protomotions/eval_agent.py \
                +headless=True \
                +robot=smpl \
                +simulator=isaacgym \
                +motion_file=data/yaml_files/all.yaml \
                +load_checkpoint=True \
                ++num_envs=500 \
                ++agent.config.eval_metric_keys=['cartesian_err','fid','state_coverage, 'plot'] \
                ++env.config.eval_metric_keys=['cartesian_err','fid','state_coverage', 'plot'] \
                ++env.config.enable_height_termination=False \
                ++env.config.terminate_ref_deviation=False \
                ++env.config.terminate_ref_timeover=True \
                ++agent.config.eval_doc_file=$DOC_FILE"

#-----------------------------------------------------------#
RGSD_MODEL="rgsd_mimic_ourrew_termi12"
RGSD_VERSION=0
RGSD_CKPT=6000

CMD="$CMD_PREFIX \
        +exp=rgsd/four_motions_mlp \
        +checkpoint=results/$RGSD_MODEL/lightning_logs/version_$RGSD_VERSION/model_$RGSD_CKPT.ckpt \
        ++agent.config.ssd_parameters.latent_init_method.ref_envs=ref_data"
eval "$CMD"

RGSD_MODEL="rgsd_mimic_ourrew_termi20"
RGSD_VERSION=0
RGSD_CKPT=6000

CMD="$CMD_PREFIX \
        +exp=rgsd/four_motions_mlp \
        +checkpoint=results/$RGSD_MODEL/lightning_logs/version_$RGSD_VERSION/model_$RGSD_CKPT.ckpt \
        ++agent.config.ssd_parameters.latent_init_method.ref_envs=ref_data"
eval "$CMD"

RGSD_MODEL="rgsd_mimic_ourrew_termi30"
RGSD_VERSION=0
RGSD_CKPT=6000

CMD="$CMD_PREFIX \
        +exp=rgsd/four_motions_mlp \
        +checkpoint=results/$RGSD_MODEL/lightning_logs/version_$RGSD_VERSION/model_$RGSD_CKPT.ckpt \
        ++agent.config.ssd_parameters.latent_init_method.ref_envs=ref_data"
eval "$CMD"

RGSD_MODEL="rgsd_mimic_constrew_termi12"
RGSD_VERSION=0
RGSD_CKPT=6000

CMD="$CMD_PREFIX \
        +exp=rgsd/four_motions_mlp \
        +checkpoint=results/$RGSD_MODEL/lightning_logs/version_$RGSD_VERSION/model_$RGSD_CKPT.ckpt \
        ++agent.config.ssd_parameters.latent_init_method.ref_envs=ref_data"
eval "$CMD"

RGSD_MODEL="rgsd_mimic_constrew_termi20"
RGSD_VERSION=0
RGSD_CKPT=6000

CMD="$CMD_PREFIX \
        +exp=rgsd/four_motions_mlp \
        +checkpoint=results/$RGSD_MODEL/lightning_logs/version_$RGSD_VERSION/model_$RGSD_CKPT.ckpt \
        ++agent.config.ssd_parameters.latent_init_method.ref_envs=ref_data"
eval "$CMD"

RGSD_MODEL="rgsd_mimic_constrew_termi30"
RGSD_VERSION=0
RGSD_CKPT=6000

CMD="$CMD_PREFIX \
        +exp=rgsd/four_motions_mlp \
        +checkpoint=results/$RGSD_MODEL/lightning_logs/version_$RGSD_VERSION/model_$RGSD_CKPT.ckpt \
        ++agent.config.ssd_parameters.latent_init_method.ref_envs=ref_data"
eval "$CMD"


# RGSD_MODEL="rgsd_discoverratio_70"
# RGSD_VERSION=0
# RGSD_CKPT=12000

# CMD="$CMD_PREFIX \
#         +exp=rgsd/four_motions_mlp \
#         +checkpoint=results/$RGSD_MODEL/lightning_logs/version_$RGSD_VERSION/model_$RGSD_CKPT.ckpt \
#         ++agent.config.ssd_parameters.latent_init_method.ref_envs=ref_data"
# eval "$CMD"


# # RGSD MIMIC
# RGSD_MODEL="rgsd_puremimic_constrew"
# RGSD_VERSION=0
# RGSD_CKPT=12000

# echo "Evaluating $RGSD_MODEL , version $RGSD_VERSION, ckpt $RGSD_CKPT, mimic"
# CMD="$CMD_PREFIX \
#         +exp=rgsd/four_motions_mlp \
#         +checkpoint=results/$RGSD_MODEL/lightning_logs/version_$RGSD_VERSION/model_$RGSD_CKPT.ckpt \
#         ++agent.config.ssd_parameters.latent_init_method.ref_envs=ref_data"
# eval "$CMD"



# #-----------------------------------------------------------#
# DIAYN_MODEL="train_diayn"
# DIAYN_VERSION=0
# DIAYN_CKPT=9000

# CMD="$CMD_PREFIX \
#         +exp=diayn_mlp \
#         +checkpoint=results/$DIAYN_MODEL/lightning_logs/version_$DIAYN_VERSION/model_$DIAYN_CKPT.ckpt \
#         ++agent.config.ssd_parameters.latent_init_method.ref_envs=best_latent \
#         ++agent.config.ssd_parameters.latent_init_method.def_envs=best_latent"
# eval "$CMD"

# DIAYN_MODEL="train_diayn_rsi"
# DIAYN_VERSION=0
# DIAYN_CKPT=18000

# CMD="$CMD_PREFIX \
#         +exp=diayn_mlp \
#         +checkpoint=results/$DIAYN_MODEL/lightning_logs/version_$DIAYN_VERSION/model_$DIAYN_CKPT.ckpt \
#         ++agent.config.ssd_parameters.latent_init_method.ref_envs=best_latent \
#         ++agent.config.ssd_parameters.latent_init_method.def_envs=best_latent"
# eval "$CMD"


# #-----------------------------------------------------------#
# METRA_MODEL="train_metra"
# METRA_VERSION=0
# METRA_CKPT=24000

# CMD="$CMD_PREFIX \
#         +exp=metra_mlp \
#         +checkpoint=results/$METRA_MODEL/lightning_logs/version_$METRA_VERSION/model_$METRA_CKPT.ckpt \
#         ++agent.config.ssd_parameters.latent_init_method.ref_envs=best_latent \
#         ++agent.config.ssd_parameters.latent_init_method.def_envs=best_latent"
# eval "$CMD"

# METRA_MODEL="train_metra_rsi"
# METRA_VERSION=0
# METRA_CKPT=24000

# CMD="$CMD_PREFIX \
#         +exp=metra_mlp \
#         +checkpoint=results/$METRA_MODEL/lightning_logs/version_$METRA_VERSION/model_$METRA_CKPT.ckpt \
#         ++agent.config.ssd_parameters.latent_init_method.ref_envs=best_latent \
#         ++agent.config.ssd_parameters.latent_init_method.def_envs=best_latent"
# eval "$CMD"

# # #-----------------------------------------------------------#
# ASE_MODEL="ase_new_motions"
# ASE_VERSION=0
# ASE_CKPT=9000

# CMD="$CMD_PREFIX \
#         +exp=ase_mlp \
#         +checkpoint=results/$ASE_MODEL/lightning_logs/version_$ASE_VERSION/model_$ASE_CKPT.ckpt \
#         ++agent.config.ase_parameters.latent_init_method=best_latent"
# eval "$CMD"


# # #-----------------------------------------------------------#
# CALM_MODEL="calm_new_motions"
# CALM_VERSION=0
# CALM_CKPT=9000

# CMD="$CMD_PREFIX \
#         +exp=calm_mlp \
#         +checkpoint=results/$CALM_MODEL/lightning_logs/version_$CALM_VERSION/model_$CALM_CKPT.ckpt \
#         ++agent.config.calm_parameters.latent_init_method=default"
# eval "$CMD"



