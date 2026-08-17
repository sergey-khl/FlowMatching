# bash scripts/train_policy.sh train_dg_dagger dg_dagger_rgb wam_pandp-solo-cup_dmp_2cam_weighted_jp-ee 0001 0 0
# bash scripts/train_policy.sh train_cr_dagger cr_dagger_rgb wam_pandp-solo-cup_dmp_2cam_weighted_jp-ee 0001 0 0
# bash scripts/train_policy.sh train_flow_matching rtc_match_rgb wam_pandp-solo-cup_dmp_2cam_weighted_jp-ee 0001 0 0

# BEFORE RUNNING:
# collect data at 500 hz or at least 100. base willl run at 10hz and res at 50
# collect img timestamps
# torque is moving average with removed offset
# https://github.com/yifan-hou/cr-dagger/blob/00700d6513b646719f003a28b488aff593eb663f/PyriteML/diffusion_policy/workspace/train_residual_mlp_workspace.py#L51
# ^^ to load base policy checkpoint
# rc jp actoins should be diffs to match
#
#
# NOTES:
# pose9wrench6 is used. action dim 15
# https://github.com/yifan-hou/cr-dagger/blob/main/PyriteConfig/tasks/common/common_type_conversions.py#L454
# they have a virtual target estimator which puts the torques with params to get a stiffness estimate without MLP.
# not used in their online config
#
# they already are doing some diffusion with their residual idea: https://github.com/yifan-hou/cr-dagger/blob/main/PyriteML/diffusion_policy/policy/diffusion_transformer_timm_policy.py#L13
# and exporing how to use the latent space: https://github.com/yifan-hou/cr-dagger/blob/main/PyriteML/diffusion_policy/policy/bet_lowdim_policy.py
# https://github.com/yifan-hou/cr-dagger/blob/00700d6513b646719f003a28b488aff593eb663f/PyriteML/diffusion_policy/model/bet/latent_generators/mingpt.py
#
# ACTION:
    # last residual is used as diff with current 
    # find the closest policy command before or at the timestamp of the robot command
    # action: trim to the shortest length
    # action time is according to low_dim ts
# OBS
    # can include item poses but does not
    # includes policy with timestamps
    # includes target and stiffness in the non online version
    # paper says predicts 15 dim so i am guessing they used the online version in which case no target and no gripper
# training
    # done on corrections only
    # WeightedRandomSampler
# normalization
    # have an identity over rotmat. should consider if getting bad results 
    #

# TODO
# see how https://github.com/yifan-hou/cr-dagger/blob/main/PyriteML/diffusion_policy/policy/diffusion_unet_timm_mod1_policy.py#L30
# is different from the other diffusions. also how the transformer_obs_encoder





DEBUG=False
save_ckpt=True

workspace_name=${1}
alg_name=${2}
task_name=${3}
config_name=${alg_name}
addition_info=${4}
seed=${5}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"


# gpu_id=$(bash scripts/find_gpu.sh)
gpu_id=${6}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


if [ $DEBUG = True ]; then
    wandb_mode=offline
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

cd flow_policy


export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}
python diffusion_policy_3d/workspace/${workspace_name}.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt}
