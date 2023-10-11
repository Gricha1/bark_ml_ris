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


if __name__ == "__main__":	
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",                  default="polamp_env")
    parser.add_argument("--test_env",             default="polamp_env")
    parser.add_argument("--dataset",              default="cross_dataset_simplified") # medium_dataset, hard_dataset, ris_easy_dataset, hard_dataset_simplified
    parser.add_argument("--dataset_curriculum",   default=False) # medium dataset -> hard dataset
    parser.add_argument("--dataset_curriculum_treshold", default=0.95, type=float) # medium dataset -> hard dataset
    parser.add_argument("--uniform_feasible_train_dataset", default=False)
    parser.add_argument("--random_train_dataset",           default=False)
    parser.add_argument("--train_sac",            default=True, type=bool)
    parser.add_argument("--validate_train_dataset",  default=True, type=bool)
    # ris
    parser.add_argument("--epsilon",            default=1e-16, type=float)
    parser.add_argument("--n_critic",           default=2, type=int) # 1
    parser.add_argument("--start_timesteps",    default=1e4, type=int) 
    parser.add_argument("--eval_freq",          default=int(3e4), type=int) # 3e4
    parser.add_argument("--max_timesteps",      default=5e6, type=int)
    parser.add_argument("--batch_size",         default=2048, type=int)
    parser.add_argument("--replay_buffer_size", default=5e5, type=int) # 5e5
    parser.add_argument("--n_eval",             default=5, type=int)
    parser.add_argument("--device",             default="cuda")
    parser.add_argument("--seed",               default=42, type=int) # 42
    parser.add_argument("--exp_name",           default="polamp_env_sac_8")
    parser.add_argument("--alpha",              default=0.1, type=float)
    parser.add_argument("--Lambda",             default=10, type=float) # 0.1
    parser.add_argument("--n_ensemble",         default=10, type=int) # 10
    parser.add_argument("--use_dubins_filter",  default=False, type=bool) # 10
    parser.add_argument("--h_lr",               default=1e-4, type=float)
    parser.add_argument("--q_lr",               default=1e-3, type=float)
    parser.add_argument("--pi_lr",              default=1e-4, type=float)
    parser.add_argument("--clip_v_function",    default=-368, type=float) # -368
    parser.add_argument("--add_obs_noise",           default=False, type=bool)
    parser.add_argument("--curriculum_alpha_val",        default=0, type=float)
    parser.add_argument("--curriculum_alpha_treshold",   default=500000, type=int) # 500000
    parser.add_argument("--curriculum_alpha",        default=False, type=bool)
    parser.add_argument("--curriculum_high_policy",  default=False, type=bool)
    # encoder
    parser.add_argument("--use_decoder",             default=True, type=bool)
    parser.add_argument("--use_encoder",             default=True, type=bool)
    parser.add_argument("--state_dim",               default=20, type=int) # 20
    # safety
    parser.add_argument("--safety_add_to_high_policy", default=False, type=bool)
    parser.add_argument("--safety",                    default=False, type=bool)
    parser.add_argument("--cost_limit",                default=0.5, type=float)
    parser.add_argument("--update_lambda",             default=1000, type=int)
    # logging
    parser.add_argument("--using_wandb",        default=True, type=bool)
    parser.add_argument("--wandb_project",      default="validate_ris_polamp", type=str)
    parser.add_argument('--log_loss', dest='log_loss', action='store_true')
    parser.add_argument('--no-log_loss', dest='log_loss', action='store_false')
    parser.set_defaults(log_loss=True)
    args = parser.parse_args()

    with open("goal_polamp_env/goal_environment_configs.json", 'r') as f:
        goal_our_env_config = json.load(f)

    with open("polamp_env/configs/train_configs.json", 'r') as f:
        train_config = json.load(f)

    with open("polamp_env/configs/environment_configs.json", 'r') as f:
        our_env_config = json.load(f)

    with open("polamp_env/configs/reward_weight_configs.json", 'r') as f:
        reward_config = json.load(f)

    with open("polamp_env/configs/car_configs.json", 'r') as f:
        car_config = json.load(f)

    if args.dataset == "medium_dataset":
        total_maps = 12
    elif args.dataset == "test_medium_dataset":
        total_maps = 3
    elif args.dataset == "hard_dataset_simplified_test":
        total_maps = 2
    else:
        total_maps = 1
    dataSet = generateDataSet(our_env_config, name_folder=args.dataset, total_maps=total_maps, dynamic=False)
    maps, trainTask, valTasks = dataSet["obstacles"]
    if args.validate_train_dataset:
        valTasks = trainTask
    goal_our_env_config["dataset"] = args.dataset
    goal_our_env_config["uniform_feasible_train_dataset"] = args.uniform_feasible_train_dataset
    goal_our_env_config["random_train_dataset"] = args.random_train_dataset
    if not goal_our_env_config["static_env"]:
        maps["map0"] = []

    args.evaluation = False
    environment_config = {
        'vehicle_config': car_config,
        'tasks': trainTask,
        'valTasks': valTasks,
        'maps': maps,
        'our_env_config' : our_env_config,
        'reward_config' : reward_config,
        'evaluation': args.evaluation,
        'goal_our_env_config' : goal_our_env_config,
    }
    args.other_keys = environment_config

    train_env_name = "polamp_env-v0"
    test_env_name = train_env_name

    # Set seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # register polamp env
    register(
        id=train_env_name,
        entry_point='goal_polamp_env.env:GCPOLAMPEnvironment',
        kwargs={'full_env_name': "polamp_env", "config": args.other_keys}
    )

    env         = gym.make(train_env_name)
    test_env    = gym.make(test_env_name)
    action_dim = env.action_space.shape[0]
    env_obs_dim = env.observation_space["observation"].shape[0]
    if args.use_encoder:
        state_dim = args.state_dim 
    else:
        state_dim = env_obs_dim

    folder = "results/{}/RIS/{}/".format(args.env, args.exp_name)
    load_results = os.path.isdir(folder)

    # Create logger
    # TODO: save_git_head_hash = True by default, change it if neccesary
    logger = None
    if args.using_wandb:
        run = wandb.init(project=args.wandb_project)
    
    # Initialize policy
    env_state_bounds = {"x": 100, "y": 100, 
                        "theta": (-np.pi, np.pi),
                        "v": (env.environment.agent.dynamic_model.min_vel, 
                              env.environment.agent.dynamic_model.max_vel), 
                        "steer": (-env.environment.agent.dynamic_model.max_steer, 
                                 env.environment.agent.dynamic_model.max_steer)}
    R = env.environment.agent.dynamic_model.wheel_base / np.tan(env.environment.agent.dynamic_model.max_steer)
    curvature = 1 / R
    policy = RIS(state_dim=state_dim, action_dim=action_dim, 
                 alpha=args.alpha,
                 use_decoder=args.use_decoder,
                 use_encoder=args.use_encoder,
                 safety=args.safety,
                 n_critic=args.n_critic,
                 train_sac=args.train_sac,
                 use_dubins_filter=args.use_dubins_filter,
                 safety_add_to_high_policy=args.safety_add_to_high_policy,
                 cost_limit=args.cost_limit, update_lambda=args.update_lambda,
                 Lambda=args.Lambda, epsilon=args.epsilon,
                 h_lr=args.h_lr, q_lr=args.q_lr, pi_lr=args.pi_lr, 
                 n_ensemble=args.n_ensemble,
                 clip_v_function=args.clip_v_function,
                 device=args.device, logger=logger if args.log_loss else None, 
                 env_obs_dim=env_obs_dim, add_ppo_reward=env.add_ppo_reward,
                 add_obs_noise=args.add_obs_noise,
                 curriculum_high_policy=args.curriculum_high_policy,
                 vehicle_curvature=curvature,
                 env_state_bounds=env_state_bounds,
                 lidar_max_dist=env.environment.MAX_DIST_LIDAR,
                 train_env=env,
    )

    if load_results:
        policy.load(folder, old_version=False)
        print("weights is loaded")
    else:
        print("WEIGHTS ISN'T LOADED")
        assert 1 == 0

    success_rate = 0
    fail_rate = 0
    num_val_tasks = 35
    val_keys = ["map0"]
    val_key = "map0"
    failed_tasks_idx = []

    # print(test_env.maps.keys())
    # print(len(test_env.valTasks[val_key]))
    # validate_tasks = [17, 24, 32, 62, 98, 102, 106, 138, 194, 212, 219]
    # validate_tasks = list(range(num_val_tasks))
    # validate_tasks = [1, 10, 25]

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
                                 plot_dubins_curve=False,
                                 # medium dataset
                                 #video_validate_tasks = [("map4", 8), ("map4", 13)],
                                 #video_validate_tasks = [("map6", 5), ("map6", 18)],
                                 #video_validate_tasks = [("map7", 19)],
                                 #video_validate_tasks = [("map5", 7)],
                                 #video_validate_tasks = [("map4", 8), ("map4", 13), ("map6", 5), ("map6", 18), ("map7", 19), ("map5", 7)],
                                 # hard dataset
                                 #video_validate_tasks = [("map0", 2), ("map0", 5), ("map0", 10), ("map0", 15)],
                                 #video_validate_tasks = [("map0", 2)],
                                 #video_validate_tasks = [("map0", 14), ("map0", 62), ("map0", 84), ("map0", 95), ("map0", 103), ("map0", 112), ("map0", 128), ("map0", 135)],
                                 # hard dataset simplified
                                 #video_validate_tasks = [("map0", 0), ("map0", 1), ("map0", 2)],
                                 #video_validate_tasks = [("map0", 0), ("map0", 1), ("map0", 2), ("map0", 3)] if not args.validate_train_dataset else [("map0", i) for i in range(600)],
                                 video_validate_tasks = [("map0", i) for i in range(600)],
                                 #video_validate_tasks = [("map0", 1), ("map1", 1)],
                                 value_function_angles=["theta_agent", 0, -np.pi/2],
                                 plot_decoder_agent_states=False,
                                 plot_subgoal_dispertion=True,
                                 plot_lidar_predictor=False,
                                 eval_strategy=None, # eval_strategy=action=[a, w], else None
                                 skip_not_video_tasks=True,
                                 plot_only_start_position=True)
    wandb_log_dict = {}
    for map_name, task_indx, video in validation_info["videos"]:
        wandb_log_dict["validation_video"+"_"+map_name+"_"+f"{task_indx}"] = wandb.Video(video, fps=10, format="gif")
    if args.using_wandb:
        run.log(wandb_log_dict)
    print("validation success rate:", success_rate)
    print("action info:", validation_info["action_info"])
    print([task[1] for task in validation_info if task[2] == "success"])

    