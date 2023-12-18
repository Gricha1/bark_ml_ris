import os
import time
from pathlib import Path
import shutil
import pathlib
import json

import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
import wandb
import gym
from gym.envs.registration import register

from utils.logger import Logger
from polamp_RIS import RIS
from polamp_env.lib.utils_operations import generateDataSet
from ris_train_polamp_env import register_goal_polamp_env, get_config
from MFNLC_for_polamp_env.mfnlc.monitor.monitor import Monitor, LyapunovValueTable
from MFNLC_for_polamp_env.mfnlc.plan import Planner


def validate(args):
    # chech if hyperparams tuning
    if args.using_wandb:
        if type(args) == type(argparse.Namespace()):
            hyperparams_tune = False
            alg = ""
            safety = ""
            hierarch  = ""
            if args.train_td3:
               alg = "TD3"
            elif args.train_sac:
                alg = "SAC"
            else:
                alg = "RIS"
            if args.safety:
                safety = "SAC_L"
            elif args.lyapunov_rrt:
                safety = "lyapunov"
            if args.rrt:
                hierarch = "RRT"
            wandb.init(project=args.wandb_project, config=args, 
                    name=alg + ", " + safety + ", " + hierarch)
        else:
            hyperparams_tune = True
            wandb.init(config=args, name="hyperparams_tune_RIS")
            args = wandb.config
    
    print("**************")
    print("state_dim:", args.state_dim)
    print("max_timesteps:", args.max_timesteps)
    print("Lambda:", args.Lambda)
    print("alpha:", args.alpha)
    print("n_ensemble:", args.n_ensemble)

    assert args.dataset_curriculum == False, "didnt implement"

    register_goal_polamp_env(args)
    train_env_name = "polamp_env-v0"
    test_env_name = train_env_name

    env         = gym.make(train_env_name)
    test_env    = gym.make(test_env_name)
    vectorized = True
    action_dim = env.action_space.shape[0]
    env_obs_dim = env.observation_space["observation"].shape[0]
    print(f"env_obs_dim: {env_obs_dim}")
    if args.use_encoder:
        state_dim = args.state_dim 
    else:
        state_dim = env_obs_dim
    folder = "results/{}/RIS/{}/".format(args.env, args.exp_name)
    load_results = os.path.isdir(folder)

    # Create logger
    logger = None
    
    # Initialize policy
    env_state_bounds = {"x": 100, "y": 100, 
                        "theta": (-np.pi, np.pi),
                        "v": (env.environment.agent.dynamic_model.min_vel, 
                              env.environment.agent.dynamic_model.max_vel), 
                        "steer": (-env.environment.agent.dynamic_model.max_steer, 
                                 env.environment.agent.dynamic_model.max_steer)}
    R = env.environment.agent.dynamic_model.wheel_base / np.tan(env.environment.agent.dynamic_model.max_steer)
    curvature = 1 / R
    max_polamp_steps = env._max_episode_steps
    print(f"max_polamp_steps: {max_polamp_steps}")
    policy = RIS(state_dim=state_dim, action_dim=action_dim, 
                 alpha=args.alpha,
                 use_decoder=args.use_decoder,
                 use_encoder=args.use_encoder,
                 safety=args.safety,
                 n_critic=args.n_critic,
                 train_sac=args.train_sac,
                 train_td3=args.train_td3,
                 lyapunov_rrt=args.lyapunov_rrt,
                 use_dubins_filter=args.use_dubins_filter,
                 safety_add_to_high_policy=args.safety_add_to_high_policy,
                 cost_limit=args.cost_limit, update_lambda=args.update_lambda,
                 Lambda=args.Lambda, epsilon=args.epsilon,
                 h_lr=args.h_lr, q_lr=args.q_lr, pi_lr=args.pi_lr, 
                 n_ensemble=args.n_ensemble,
                 clip_v_function=args.clip_v_function, max_grad_norm=args.max_grad_norm, lambda_initialization=args.lambda_initialization,
                 max_episode_steps = max_polamp_steps,
                 device=args.device, logger=logger if args.log_loss else None, 
                 env_obs_dim=env_obs_dim, add_ppo_reward=env.add_ppo_reward,
                 add_obs_noise=args.add_obs_noise,
                 curriculum_high_policy=args.curriculum_high_policy,
                 vehicle_curvature=curvature,
                 env_state_bounds=env_state_bounds,
                 lidar_max_dist=env.environment.MAX_DIST_LIDAR,
                 train_env=env,
    )

    if args.lyapunov_rrt:
        add_monitor = False
        # lyapunov table initialization
        obs_lb = -np.ones(shape=env_obs_dim)
        obs_ub = np.ones(shape=env_obs_dim)
        #obs_lb = np.array([-1, -1, -1, -1, 9.8, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        #obs_ub = np.array([1, 1, 1, 1, 9.81, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        n_levels = 10 # default
        pgd_max_iter = 500
        pgd_lr = float(1e-3) # default
        n_range_est_sample = 10 # default
        n_radius_est_sample = 40 
        bound_cnst = float(100) # default
        lv_table = LyapunovValueTable(policy.tclf,
                                    obs_lb,
                                    obs_ub,
                                    n_levels=n_levels,
                                    pgd_max_iter=pgd_max_iter,
                                    pgd_lr=pgd_lr,
                                    n_range_est_sample=n_range_est_sample,
                                    n_radius_est_sample=n_radius_est_sample,
                                    bound_cnst=bound_cnst)
        # lyapunov monitor initialization
        monitor_max_step_size = float(0.2)
        monitor_search_step_size = float(0.01)
        if add_monitor:
            print("!!!!!!! building lyapunov table")
            lv_table.build()
            print("!!!!!!! building is complited")
            monitor = Monitor(lv_table, max_step_size=monitor_max_step_size, search_step_size=monitor_search_step_size)
        # planner initizlization
    rrt_data = {}
    if args.rrt or args.lyapunov_rrt:
        planner_max_iter = 18000
        planning_algo = "rrt*"
        #planning_algo = "rrt"
        planning_algo_kwargs = {}
        #planner = Planner(env, planning_algo)

        #env.reset(id=0, val_key="map0")
        #path = planner.plan(planner_max_iter, **planning_algo_kwargs)
        rrt_data["planning_algo_kwargs"] = planning_algo_kwargs
        rrt_data["planner_max_iter"] = planner_max_iter
        rrt_data["planning_algo"] = planning_algo
        rrt_data["add_monitor"] = False
        if add_monitor:
            rrt_data["add_monitor"] = True
            rrt_data["monitor"] = monitor

    if load_results and not hyperparams_tune:
        policy.load(folder)
        print("weights is loaded")
    else:
        print("WEIGHTS ISN'T LOADED")

    from ris_train_polamp_env import evalPolicy

    eval_distance, success_rate, eval_reward, \
    val_state, val_goal, \
    mean_actions, eval_episode_length, validation_info \
            = evalPolicy(policy, test_env, 
                        plot_full_env=True,
                        plot_subgoals=False,
                        plot_value_function=False,
                        render_env=False,
                        plot_only_agent_values=False, 
                        plot_decoder_agent_states=False,
                        plot_subgoal_dispertion=False,
                        plot_lidar_predictor=False,
                        data_to_plot={},
                        #video_validate_tasks = [("map0", 0), ("map0", 1), ("map1", 0), ("map1", 1)],
                        #video_validate_tasks = [("map0", 0), ("map0", 1), ("map0", 3), ("map0", 4)],
                        #video_validate_tasks = [("map0", 10), ("map0", 50), ("map0", 80), ("map0", 150), ("map0", 200)],
                        #video_validate_tasks = [("map0", 10), ("map0", 20), ("map0", 40), ("map0", 50), ("map0", 80), ("map0", 120), ("map0", 150), ("map0", 190), ("map0", 200), ("map0", 250), ("map0", 300)],
                        video_validate_tasks = [("map0", 190)],
                        full_validation = True,
                        #video_validate_tasks = [],
                        value_function_angles=["theta_agent", 0, -np.pi/2],
                        dataset_plot=True,
                        skip_not_video_tasks=True,
                        dataset_validation=args.dataset,
                        rrt=args.rrt,
                        rrt_data=rrt_data,
                        lyapunov_network_validation=args.lyapunov_rrt)
    wandb_log_dict = {}
    for val_key in validation_info:
        wandb_log_dict["validation/"+val_key] = validation_info[val_key]
    if args.using_wandb:
        for dict_ in val_state + val_goal:
            for key in dict_:
                wandb_log_dict[f"{key}"] = dict_[key]
        for map_name, task_indx, video in validation_info["videos"]:
            cur_step = 0
            wandb_log_dict["validation_video"+"_"+map_name+"_"+f"{task_indx}"] = \
                wandb.Video(video, fps=10, format="gif", caption=f"steps: {cur_step}")
        wandb.log(wandb_log_dict)
        del wandb_log_dict


if __name__ == "__main__":	
    args = get_config()
    #args.wandb_project = "validate_ris_sac_polamp"
    args.wandb_project = "validate_ris_polamp"
    #args.dataset = "cross_dataset_simplified"
    args.dataset = "without_obst_dataset"
    args.lyapunov_rrt = True
    #args.rrt = True
    args.rrt = True
    validate(args)


    