#!/bin/bash

# HyperParams
run=0
start_run=1
# monitor
obs_lb_ub_koefs=(1 10)
n_levelss=(10 30)
n_range_est_samples=(10 30)
monitor_search_step_sizes=(0.01 0.1)
monitor_max_step_sizes=(1 5)
# planner
rrt_subgoal_safe_epss=(1.5 2 3.0)
planner_max_iters=(9000 18000)

# cycle for HyperParams tune
for obs_lb_ub_koef in "${obs_lb_ub_koefs[@]}"
do
    for n_levels in "${n_levelss[@]}"
    do
        for n_range_est_sample in "${n_range_est_samples[@]}"
        do
            for monitor_search_step_size in "${monitor_search_step_sizes[@]}"
            do
                for monitor_max_step_size in "${monitor_max_step_sizes[@]}"
                do      
                    for rrt_subgoal_safe_eps in "${rrt_subgoal_safe_epss[@]}"
                    do  
                        for planner_max_iter in "${planner_max_iters[@]}"
                        do
                            run=$((run + 1))
                            echo "run: $run"
                            echo "exp_name: $exp_name"
                            if [ $run -ge $start_run ]; then
                                python ris_validate_polamp_env.py --exp_name lyapunov_ex_90 --obs_lb_ub_koef $obs_lb_ub_koef --n_levels $n_levels --n_range_est_sample $n_range_est_sample --monitor_search_step_size $monitor_search_step_size --monitor_max_step_size $monitor_max_step_size --rrt_subgoal_safe_eps $rrt_subgoal_safe_eps --planner_max_iter $planner_max_iter
                            fi
                        done
                    done
                done
            done
        done
    done
done

