add_to_run_wandb_name=lyapunov_level_1_

# environment
dataset="cross_dataset_test_level_1"
#dataset="cross_dataset_test_level_2"
validate_task_id=0
seed=30

# lyapunov rrt
rrt_subgoal_safe_eps=3.0
monitor_search_step_size=0.5

# --save_v_table 
# --load_v_table
# --load_v_table_folder lyapunov_ex_1
# --static_env 
# --not_visual_validation 

python ris_validate_polamp_env.py --exp_name lyapunov_ex \
                                  --load --load_folder train_td3_4 \
                                  --load_v_table --load_v_table_folder lyapunov_ex_1 \
                                  --static_env \
                                  --seed $seed --validate_task_id $validate_task_id \
                                  --rrt_subgoal_safe_eps $rrt_subgoal_safe_eps \
                                  --add_to_run_wandb_name $add_to_run_wandb_name"_"$seed"_" \
                                  --monitor_search_step_size $monitor_search_step_size \
                                  --dataset $dataset