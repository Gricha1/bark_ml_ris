# $1 = val_task
# $2 = seed
# $3 = dataset
# $4 = wandb name
# $5 = max plan steps

if [ -z "$1" ]; then
    validate_task_id=0
else
    validate_task_id=$1
fi

if [ -z "$2" ]; then
    seed=30
else
    seed=$2
fi

if [ -z "$3" ]; then
    dataset="cross_dataset_test_level_1"
else
    dataset=$3
fi

if [ -z "$4" ]; then
    add_to_run_wandb_name="cross_dataset_test_level_1"
else
    add_to_run_wandb_name=$4
fi

if [ -z "$5" ]; then
    planner_max_iter=64000
else
    planner_max_iter=$5
fi


# lyapunov rrt
rrt_subgoal_safe_eps=3.0
monitor_search_step_size=0.5

# --save_v_table 
# --load_v_table
# --load_v_table_folder lyapunov_ex_1
# --static_env 
# --not_visual_validation 
# --with_dubins_curve

python ris_validate_polamp_env.py --exp_name lyapunov_ex \
                                  --load --load_folder train_td3_8 \
                                  --static_env \
                                  --planner_max_iter $planner_max_iter \
                                  --seed $seed --validate_task_id $validate_task_id \
                                  --rrt_subgoal_safe_eps $rrt_subgoal_safe_eps \
                                  --add_to_run_wandb_name $add_to_run_wandb_name"_"$seed"_" \
                                  --monitor_search_step_size $monitor_search_step_size \
                                  --dataset $dataset \
                                  --load_v_table --load_v_table_folder lyapunov_ex_14 \
                                  --with_dubins_curve \
                                  --not_visual_validation 