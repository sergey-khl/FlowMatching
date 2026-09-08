# bash scripts/eval_policy.sh eval_flow_matching rtc_flow_match_rgb wam_plug-asus_base_2cam 0001 0 0
# bash scripts/eval_policy.sh eval_flow_matching rtc_flow_match_rgb wam_wipe-board_base_2cam 0001 0 0
# bash scripts/eval_policy.sh eval_flow_matching rtc_flow_match_rgb wam_build-lego_base_2cam 0001 0 0
# bash scripts/eval_policy.sh eval_cr_dagger cr_dagger_rgb wam_pandp-apple_cr-dagger_2cam 0001 0 0
# bash scripts/eval_policy.sh eval_cr_dagger cr_dagger_rgb wam_build-lego_cr-dagger_2cam 0001 0 0
# bash scripts/eval_policy.sh eval_gcr_dagger gcr_dagger_rgb wam_pandp-apple_gcr-dagger_2cam 0001 0 0
# bash scripts/eval_policy.sh eval_dg_dagger dg_dagger_rgb wam_pandp-apple_dg-dagger_2cam 0001 0 0



DEBUG=False

workspace_name=${1}
alg_name=${2}
task_name=${3}
config_name=${alg_name}
addition_info=${4}
seed=${5}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="/mnt/10tb/serg/FlowMatching/outputs/${exp_name}_seed${seed}"
# run_dir="/mnt/10tb/serg/FlowMatching/outputs/train_cr_dagger_wam_pandp-apple_cr-dagger_2cam"
# run_dir="/mnt/10tb/serg/FlowMatching/outputs/train_cr_dagger_wam_wipe-board_cr-dagger_2cam"
# run_dir="/mnt/10tb/serg/FlowMatching/outputs/train_rtc_flowmatch_wam_pandp-apple_base_2cam"
# run_dir="/mnt/10tb/serg/FlowMatching/outputs/train_rtc_flowmatch_wam_wipe-board_base_2cam"

gpu_id=${6}


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



                                
