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
    
    parser.add_argument("--test_0_collision",   default=False, type=bool) # collision return to previous state & freeze
    parser.add_argument("--test_1_collision",   default=False, type=bool) # collision r = cur_step - max_step
    parser.add_argument("--test_2_collision",   default=False, type=bool) # collision = return to beggining of episode
    parser.add_argument("--test_3_collision",   default=False, type=bool) # collision return to 4 previous state & not freeze
    parser.add_argument("--test_4_collision",   default=True, type=bool) # collision r = -20, continue episode
    parser.add_argument("--her_corrections",    default=False, type=bool) # dont add collision states to HER
    parser.add_argument("--add_frame_stack",    default=False, type=bool) # add frame stack to goal&state
    parser.add_argument("--static_env",         default=True, type=bool)

    parser.add_argument("--env",                default="polamp_env")
    parser.add_argument("--test_env",           default="polamp_env")
    parser.add_argument("--epsilon",            default=1e-16, type=float)
    parser.add_argument("--distance_threshold", default=0.5, type=float)
    parser.add_argument("--start_timesteps",    default=1e4, type=int) 
    parser.add_argument("--eval_freq",          default=int(2e3), type=int)
    parser.add_argument("--max_timesteps",      default=5e6, type=int)
    parser.add_argument("--batch_size",         default=2048, type=int)
    parser.add_argument("--replay_buffer_size", default=1e6, type=int)
    parser.add_argument("--n_eval",             default=5, type=int)
    parser.add_argument("--device",             default="cuda")
    parser.add_argument("--seed",               default=42, type=int)
    parser.add_argument("--exp_name",           default="RIS_ant")
    parser.add_argument("--alpha",              default=0.1, type=float)
    parser.add_argument("--Lambda",             default=0.1, type=float)
    parser.add_argument("--h_lr",               default=1e-4, type=float)
    parser.add_argument("--q_lr",               default=1e-3, type=float)
    parser.add_argument("--pi_lr",              default=1e-3, type=float)

    parser.add_argument("--state_dim",          default=5, type=int)
    parser.add_argument("--using_wandb",        default=True, type=bool)
    parser.add_argument("--wandb_project",      default="validate_ris_sac_polamp", type=str)
    parser.add_argument('--log_loss',           dest='log_loss', action='store_true')
    parser.add_argument('--no-log_loss',        dest='log_loss', action='store_false')
    parser.set_defaults(log_loss=True)
    args = parser.parse_args()

    with open("polamp_env/configs/train_configs.json", 'r') as f:
        train_config = json.load(f)

    with open("polamp_env/configs/environment_configs.json", 'r') as f:
        our_env_config = json.load(f)

    with open("polamp_env/configs/reward_weight_configs.json", 'r') as f:
        reward_config = json.load(f)

    with open("polamp_env/configs/car_configs.json", 'r') as f:
        car_config = json.load(f)

    dataSet = generateDataSet(our_env_config, name_folder="maps", total_maps=1, dynamic=False)
    #maps, trainTask, valTasks = dataSet["empty"]
    maps, trainTask, valTasks = dataSet["obstacles"]
    if not args.static_env:
        maps["map0"] = []

    # dataset info
    print("dataset:", len(dataSet["obstacles"][0]["map0"]), #4 maps
                      len(dataSet["obstacles"][1]["map0"]), #70 train tasks
                      len(dataSet["obstacles"][2]["map0"])  #35 val tasks
         )

    args.evaluation = False
    environment_config = {
        'vehicle_config': car_config,
        'tasks': trainTask,
        'valTasks': valTasks,
        'maps': maps,
        'our_env_config' : our_env_config,
        'reward_config' : reward_config,
        'evaluation': args.evaluation,
        "static_env": args.static_env,
        "test_0_collision": args.test_0_collision,
        "test_1_collision": args.test_1_collision,
        "test_2_collision": args.test_2_collision,
        "test_3_collision": args.test_3_collision,
        "test_4_collision": args.test_4_collision,
        "add_frame_stack": args.add_frame_stack,
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
    state_dim = args.state_dim 

    folder = "results/{}/RIS/{}/".format(args.env, args.exp_name)
    load_results = os.path.isdir(folder)

    # Create logger
    # TODO: save_git_head_hash = True by default, change it if neccesary
    logger = None
    if args.using_wandb:
        run = wandb.init(project=args.wandb_project)
    
    # Initialize policy
    policy = RIS(state_dim=state_dim, action_dim=action_dim, alpha=args.alpha, Lambda=args.Lambda, h_lr=args.h_lr, q_lr=args.q_lr, pi_lr=1e-3, device=args.device, logger=logger if args.log_loss else None)

    if load_results:
        policy.load(folder)
        print("weights is loaded")

    success_rate = 0
    fail_rate = 0
    num_val_tasks = 35
    val_keys = ["map0"]
    val_key = "map0"
    failed_tasks_idx = []

    #validate_tasks = [17, 24, 32, 62, 98, 102, 106, 138, 194, 212, 219]
    #validate_tasks = list(range(num_val_tasks))
    #validate_tasks = [1, 10, 25]

    from ris_train_polamp_env import evalPolicy

    eval_distance, success_rate, eval_reward, \
    val_state, val_goal, \
    mean_actions, eval_episode_length, images, validation_info \
                    = evalPolicy(policy, test_env, 
                                 plot_subgoals=True, 
                                 render_env=False, 
                                 plot_only_agent_values=True,
                                 plot_obstacles=args.static_env, 
                                 video_task_id=12, eval_strategy=None) # 18, 12
    wandb_log_dict = {}
    wandb_log_dict["validation_video"] = wandb.Video(images, fps=10, format="gif")
    run.log(wandb_log_dict)
    print("validation success rate:", success_rate)
    print("action info:", validation_info["action_info"])
    print([task[1] for task in validation_info if task[2] == "success"])

    