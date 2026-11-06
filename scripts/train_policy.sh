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
