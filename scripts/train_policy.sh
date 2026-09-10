# BASE:
# bash scripts/train_policy.sh train_flow_matching rtc_flow_match_rgb wam_pandp-apple_base_2cam 0001 0 0
# bash scripts/train_policy.sh train_flow_matching rtc_flow_match_rgb wam_wipe-board_base_2cam 0001 0 0
# bash scripts/train_policy.sh train_flow_matching rtc_flow_match_rgb wam_build-lego_base_2cam 0001 0 0
# bash scripts/train_policy.sh train_flow_matching rtc_flow_match_rgb wam_transfer-bag_base_2cam 0001 0 0
# CR
# bash scripts/train_policy.sh train_cr_dagger cr_dagger_rgb wam_pandp-apple_cr-dagger_2cam 0001 0 0
# bash scripts/train_policy.sh train_cr_dagger cr_dagger_rgb wam_wipe-board_cr-dagger_2cam 0001 0 0
# bash scripts/train_policy.sh train_cr_dagger cr_dagger_rgb wam_build-lego_cr-dagger_2cam 0001 0 0
# bash scripts/train_policy.sh train_cr_dagger cr_dagger_rgb wam_transfer-bag_cr-dagger_2cam 0001 0 0
# GCR
# bash scripts/train_policy.sh train_gcr_dagger gcr_dagger_rgb wam_pandp-apple_gcr-dagger_2cam 0001 0 0
# bash scripts/train_policy.sh train_gcr_dagger gcr_dagger_rgb wam_wipe-board_gcr-dagger_2cam 0001 0 0 DG bash scripts/train_policy.sh train_dg_dagger dg_dagger_rgb wam_pandp-apple_dg-dagger_2cam 0001 0 0
# DG
# bash scripts/train_policy.sh train_dg_dagger dg_dagger_rgb wam_pandp-appl_dg-dagger_2cam 0001 0 0
# bash scripts/train_policy.sh train_dg_dagger dg_dagger_rgb wam_wipe-board_dg-dagger_2cam 0001 0 0
#
# NEXT STEPS:
# check kernel for unet. test with kernel size 25
# remove losses for gate / scale. test both

# BEFORE RUNNING:
# DONE: collect data at 500 hz or at least 128. base willl run at 1hz and res at 50
# DONE: collect img timestamps. these are used as the base index for the sampler
# DONE: torque is ema at 500 windows with removed offset
# DONE: https://github.com/yifan-hou/cr-dagger/blob/00700d6513b646719f003a28b488aff593eb663f/PyriteML/diffusion_policy/workspace/train_residual_mlp_workspace.py#L51 << to load base policy checkpoint
# DONE: rc jp actoins should be diffs to match
#
#
# NOTES:
# pose9wrench6 is used. action dim 15
# https://github.com/yifan-hou/cr-dagger/blob/main/PyriteConfig/tasks/common/common_type_conversions.py#L454
# they have a virtual target estimator which puts the torques with params to get a stiffness estimate without MLP.
# not used in their online config. paper says stiffness is static.
#
# didnt see a base policy. will assume its just rtc flow matching without corrections
#
# num_of_new_episodes is 0 in https://github.com/yifan-hou/cr-dagger/blob/00700d6513b646719f003a28b488aff593eb663f/PyriteML/online_learning/configs/config_v1.py#L50
# can ignore new_episode_prob. it gets used when new episode data is available with load_new_episodes. it means there is a chance the new episode gets considered
# i think its equivalent to a uniform sample in the offline case
#
# they already are doing some diffusion with their residual idea: https://github.com/yifan-hou/cr-dagger/blob/main/PyriteML/diffusion_policy/policy/diffusion_transformer_timm_policy.py#L13
# and exporing how to use the latent space: https://github.com/yifan-hou/cr-dagger/blob/main/PyriteML/diffusion_policy/policy/bet_lowdim_policy.py
# https://github.com/yifan-hou/cr-dagger/blob/00700d6513b646719f003a28b488aff593eb663f/PyriteML/diffusion_policy/model/bet/latent_generators/mingpt.py
#
# the human is not interacting with the force torque sensor!! this makes our version way harder. yay!!
# but also that means they say a correction happens basically whenever there is a lot of ext torque or do they always use the button? not in correction_single_arm
#
# https://github.com/yifan-hou/cr-dagger/blob/00700d6513b646719f003a28b488aff593eb663f/PyriteEnvSuites/envs/task/manip_server_env.py#L71 
# they use force control in all axis' so no velocity control.
# find total wrench of 4 differenct wrenches, spring, force err, pid, damping. convert to velocity by solving with inertia
# convert that velocity to a pose cmd
# the spring wrench is where the ref pose is added
# https://docs.google.com/document/d/1OGB3R5M62lnUrIzuXUJB2nOg8ef5WupwzwgngtGalOU/edit?tab=t.0#heading=h.z272vf49q1uz
# as far as I understand they combine damped velocity, spring force of the pose err, and the err of the wrench with no gains.
# https://github.com/yifan-hou/force_control/blob/mainline/src/admittance_controller.cpp
#
# ACTION:
    # last residual is used as diff with current 
    # find the closest policy command before or at the timestamp of the robot command
    # do the diff with the robot state that is closest to policy time
    # action: trim to the shortest length
# OBS
    # can include item poses but does not
    # includes policy with timestamps but unused in non transformer version
    # includes target and stiffness in the non online version
    # paper says predicts 15 dim so i am guessing they used the online version in which case no target and no gripper
    # policy input is the base policy abs pose. In the paper this is 32 actions at 10hz so 3.2seconds of actions. In the sparse dataset we use 6 of the last 0.6 seconds as policy obs input
# training
    # done on corrections only
    # WeightedRandomSampler
    # given the idx of the current episode. begin with the time and idx of the first rgb for that episode. extrapolate the other items based on that time. see: sample_sequence
# normalization
    # have an identity norm over rotmat. should consider if getting bad results 
    # same thing with torque. in their case they do range norm with wrench 424: https://github.com/yifan-hou/cr-dagger/blob/main/PyriteML/diffusion_policy/dataset/dynamic_dataset.py

# Double Check
# see how https://github.com/yifan-hou/cr-dagger/blob/main/PyriteML/diffusion_policy/policy/diffusion_unet_timm_mod1_policy.py#L30
# is different from the other diffusions. also how the transformer_obs_encoder
# pretty sure virtual_target does nothign with static stiffness
#
# cr vs dg differences
# no ft sensor so force encoder input dim is 7.
# predicting ext_torque from joints
# using the traj_weight signal for determening correction threshold
# different controller but same concept





DEBUG=False
save_ckpt=True

workspace_name=${1}
alg_name=${2}
task_name=${3}
config_name=${alg_name}
addition_info=${4}
seed=${5}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="/mnt/10tb/serg/FlowMatching/outputs/${exp_name}_seed${seed}"


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
                            output_dir=${run_dir} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt}
