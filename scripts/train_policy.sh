# bash scripts/train_policy.sh train_dg_dagger dg_dagger_rgb wam_pandp-solo-cup_dmp_2cam_weighted_jp-ee 0001 0 0



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
