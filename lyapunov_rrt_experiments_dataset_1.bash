#!/bin/bash


validate_task_ids=($(seq 0 95))
seeds=(30)

dataset="cross_dataset_test_level_1"

add_to_run_wandb_name=dataset_1_exps_1

for seed in "${seeds[@]}"
do
    for validate_task_id in "${validate_task_ids[@]}"
    do
        echo "------------------- EXP -------------------"
        echo "------------------------------------------"
        echo "validate_task_id: $validate_task_id"
        sh validate_lyapunov_rrt.sh $validate_task_id $seed $dataset $add_to_run_wandb_name
        echo "------------------------------------------"
        echo "------------------------------------------"
    done
done




"""
# HyperParams
#run=1
#start_run=9
# monitor
obs_lb_ub_koefs=(1 10)
n_levelss=(10 30)
n_range_est_samples=(10 30)
monitor_search_step_sizes=(0.01 0.1)
monitor_max_step_sizes=(1 5)
rrt_subgoal_safe_epss=(2.0 1.5)
# planner
seeds=(90 42 50 8 30)

# cycle for HyperParams tune
#for rrt_subgoal_safe_eps in "${rrt_subgoal_safe_epss[@]}"
#do
#    for seed in "${seeds[@]}"
#    do
#        run=$((run + 1))
#        result_dir_name="lyapunov_rrt_results_"$run
#        echo "result_dir_name: $result_dir_name"
#        if [ $run -ge $start_run ]; then
#            python ris_validate_polamp_env.py --exp_name lyapunov_ex_90 --seed $seed --results_dir $result_dir_name --rrt_subgoal_safe_eps $rrt_subgoal_safe_eps
#        fi
#    done
#done

#############-------------------------------
add_to_run_wandb_name=lyapunov_level_1_
#seeds=(30 50 90 42 8)
#seeds=(30 50 90)

# environment
start_val_task=0
validate_task_ids=($(seq 0 95))
dataset="cross_dataset_test_level_1"
#dataset="cross_dataset_test_level_2"
seeds=(30)

# lyapunov rrt
rrt_subgoal_safe_eps=3.0
monitor_search_step_size=0.5

echo ${validate_task_ids[@]}

get_video_validation_task="True"
for seed in "${seeds[@]}"
do
    for validate_task_id in "${validate_task_ids[@]}"
    do
        #run=$((run + 1))
        result_dir_name="lyapunov_rrt_results_"$run
        echo "validate_task_id: $validate_task_id"
        if [ $validate_task_id -ge $start_val_task ]; then
            python ris_validate_polamp_env.py --exp_name lyapunov_ex_90 \
                                              --seed $seed --validate_task_id $validate_task_id \
                                              --get_video_validation_task $get_video_validation_task \
                                              --rrt_subgoal_safe_eps $rrt_subgoal_safe_eps \
                                              --add_to_run_wandb_name $add_to_run_wandb_name"_"$seed"_" \
                                              --monitor_search_step_size $monitor_search_step_size \
                                              --dataset $dataset
        fi
    done
done
"""
