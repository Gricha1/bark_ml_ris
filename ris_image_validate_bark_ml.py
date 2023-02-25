import os
import time

import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse

import gym
from gym.envs.registration import register

from utils.logger import Logger
from custom_RIS import RIS
from HER import HERReplayBuffer, PathBuilder
import wandb

from pathlib import Path
import shutil
import pathlib

if __name__ == "__main__":	
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",                default="Image84SawyerPushAndReachArenaTrainEnvBig-v0")
    parser.add_argument("--test_env",           default="Image84SawyerPushAndReachArenaTestEnvBig-v1")
    parser.add_argument("--epsilon",            default=1e-4, type=float)
    parser.add_argument("--replay_buffer_goals",default=0.5, type=float)
    parser.add_argument("--distance_threshold", default=0.05, type=float)
    parser.add_argument("--start_timesteps",    default=1e4, type=int) 
    parser.add_argument("--eval_freq",          default=1e3, type=int)
    parser.add_argument("--max_timesteps",      default=5e6, type=int)
    parser.add_argument("--max_episode_length", default=100, type=int)
    parser.add_argument("--batch_size",         default=256, type=int)
    parser.add_argument("--replay_buffer_size", default=1e5, type=int)
    parser.add_argument("--n_eval",             default=20, type=int)
    parser.add_argument("--device",             default="cuda")
    parser.add_argument("--seed",               default=42, type=int)
    parser.add_argument("--exp_name",           default="RIS_sawyer")
    parser.add_argument("--alpha",              default=0.1, type=float)
    parser.add_argument("--Lambda",             default=0.1, type=float)
    parser.add_argument("--h_lr",               default=1e-4, type=float)
    parser.add_argument("--q_lr",               default=1e-3, type=float)
    parser.add_argument("--pi_lr",              default=1e-4, type=float)
    parser.add_argument("--enc_lr",             default=1e-4, type=float)
    parser.add_argument("--state_dim",          default=16, type=int)

    parser.add_argument('--log_loss', dest='log_loss', action='store_true')
    parser.add_argument('--no-log_loss', dest='log_loss', action='store_false')
    parser.set_defaults(log_loss=True)
    args = parser.parse_args()
    print(args)

    # set path for experement video
    working_dir_name = str(pathlib.Path().resolve())
    if os.path.isdir(working_dir_name + "/video_validation/"):
        shutil.rmtree(working_dir_name + "/video_validation/")
    observation_dir_name = working_dir_name + f"/video_validation/obs_pngs"
    if os.path.isdir(observation_dir_name):
        shutil.rmtree(observation_dir_name)
    path = Path(observation_dir_name)
    path.mkdir(parents=True, exist_ok=True)

    video_name = working_dir_name + "/video_validation/pngs"


    train_env_name = "parking-v0"
    test_env_name = train_env_name

    # Set seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    register(
        id=train_env_name,
        entry_point='custom_bark_gym_env.custom_gym_bark_ml_env:GCContinuousParkingGym'
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
    run = wandb.init(project='RIS_bark_ml_validate')
    
    # Initialize policy
    policy = RIS(state_dim=state_dim, action_dim=action_dim, 
                 image_env=True, alpha=args.alpha, 
                 Lambda=args.Lambda, epsilon=args.epsilon, 
                 h_lr=args.h_lr, q_lr=args.q_lr, pi_lr=args.pi_lr, 
                 enc_lr=args.enc_lr, device=args.device, 
                 logger=logger if args.log_loss else None, max_env_steps=args.max_episode_length)
    if load_results:
        policy.load(folder)
        print("weights is loaded")


    epoches = 10
    for epoch in range(epoches):
        # Initialize environment
        obs = env.reset()
        done = False
        state = obs["image_observation"]
        goal = obs["image_desired_goal"]
        t = 0

        while not done:
            t += 1

            # debug
            print("step:", t, end=" ")
            
            # Select action
            if t < args.start_timesteps:
                action = env.action_space.sample()
            else:
                action = policy.select_action(state, goal)

            # debug
            #action = [0.5, 0]
            print("action:", action, end=" ")

            # Perform action
            next_obs, reward, done, info = env.step(action) 
            print("reward:", reward, 
                "dist to goal:", info["dist_to_goal"], 
                "agent theta:", info["agent_state"][2],
                "agent steer:", info["agent_state"][4],
                "episode ends:", done)
            done = done or t >= int(args.max_timesteps)

            next_state = next_obs["image_observation"]

            state = next_state
            obs = next_obs

            print()
            print("agent:", info["obs_to_validation"][3, 0, :5])
            print("goal:", info["obs_to_validation"][3, 1, :5])
            print()

            save_obs = False

            if save_obs:
                observed_next_state = info["obs_to_validation"]
                observed_next_state = np.concatenate([observed_next_state[0:1, :, :] + observed_next_state[1:2, :, :],
                                                    observed_next_state[2:3, :, :],   
                                                    observed_next_state[4:5, :, :]], axis=0)
                observed_next_state = np.moveaxis(observed_next_state, 0, -1)
                plt.imshow(observed_next_state)
                plt.savefig(observation_dir_name + '/' + f'fig{t}.png')
        
            if done:
                if save_obs:
                    from utilite_video_generator import generate_video
                    generate_video(env=False, obs=True, run=run)
                    print('save obs video', end=" ")
                    shutil.rmtree(observation_dir_name)
                    path = Path(observation_dir_name)
                    path.mkdir(parents=True, exist_ok=True)

                from utilite_video_generator import generate_video
                generate_video(env=True, obs=False, run=run)
                print('save env video', end=" ")
                shutil.rmtree(video_name)
                path = Path(video_name)
                path.mkdir(parents=True, exist_ok=True)

                print("episode is finished")
                break


