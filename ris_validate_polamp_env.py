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
from custom_second_RIS import RIS, get_img_feat_from_rollout
from HER import HERReplayBuffer, PathBuilder
import wandb

# polamp depends
from polamp_env.lib.utils_operations import generateDataSet
import json


from pathlib import Path
import shutil
import pathlib

if __name__ == "__main__":	
    parser = argparse.ArgumentParser()

    """
    parser.add_argument("--env",                default="Image84SawyerPushAndReachArenaTrainEnvBig-v0")
    parser.add_argument("--test_env",           default="Image84SawyerPushAndReachArenaTestEnvBig-v1")
    parser.add_argument("--epsilon",            default=1e-4, type=float)
    parser.add_argument("--replay_buffer_goals",default=0.5, type=float)
    parser.add_argument("--distance_threshold", default=0.05, type=float)
    parser.add_argument("--start_timesteps",    default=1e4, type=int) 
    parser.add_argument("--eval_freq",          default=1e3, type=int)
    parser.add_argument("--max_timesteps",      default=5e6, type=int)
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
    parser.add_argument("--state_dim",          default=5, type=int)
    """

    parser.add_argument("--env",                default="polamp_env")
    parser.add_argument("--test_env",           default="polamp_env")
    parser.add_argument("--epsilon",            default=1e-4, type=float)
    parser.add_argument("--replay_buffer_goals",default=0.5, type=float)
    parser.add_argument("--distance_threshold", default=0.05, type=float)
    parser.add_argument("--start_timesteps",    default=0, type=int) 
    parser.add_argument("--eval_freq",          default=int(2e3), type=int)
    parser.add_argument("--max_timesteps",      default=5e6, type=int)
    parser.add_argument("--batch_size",         default=1024, type=int)
    parser.add_argument("--replay_buffer_size", default=5e5, type=int)
    parser.add_argument("--n_eval",             default=10, type=int)
    parser.add_argument("--device",             default="cuda")
    parser.add_argument("--seed",               default=42, type=int)
    parser.add_argument("--exp_name",           default="RIS_sawyer")
    parser.add_argument("--alpha",              default=0.1, type=float)
    parser.add_argument("--Lambda",             default=0.1, type=float)

    parser.add_argument("--h_lr",               default=1e-4, type=float)
    parser.add_argument("--q_lr",               default=1e-3, type=float)
    parser.add_argument("--pi_lr",              default=1e-3, type=float)
    parser.add_argument("--state_dim",          default=5, type=int)

    parser.add_argument("--no_video",           default=False, type=bool)
    parser.add_argument("--validate_bad_cases", default=False, type=bool)
    parser.add_argument("--wandb_project",      default="RIS_polamp_env_validate", type=str)

    parser.add_argument('--log_loss', dest='log_loss', action='store_true')
    parser.add_argument('--no-log_loss', dest='log_loss', action='store_false')
    parser.set_defaults(log_loss=True)
    args = parser.parse_args()
    print(args)

    """
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
    """

    with open("polamp_env/configs/train_configs.json", 'r') as f:
        train_config = json.load(f)

    with open("polamp_env/configs/environment_configs.json", 'r') as f:
        our_env_config = json.load(f)
        # print(our_env_config)

    with open("polamp_env/configs/reward_weight_configs.json", 'r') as f:
        reward_config = json.load(f)

    with open("polamp_env/configs/car_configs.json", 'r') as f:
        car_config = json.load(f)

    dataSet = generateDataSet(our_env_config, name_folder="maps", total_maps=1, dynamic=False)
    #maps, trainTask, valTasks = dataSet["empty"]
    maps, trainTask, valTasks = dataSet["obstacles"]
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
    }
    args.other_keys = environment_config

    train_env_name = "polamp_env-v0"
    test_env_name = train_env_name

    # Set seed
    #np.random.seed(args.seed)
    #random.seed(args.seed)
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
    run = wandb.init(project='RIS_polamp_env_validate')
    
    # Initialize policy
    '''
    policy = RIS(state_dim=state_dim, action_dim=action_dim, 
                 image_env=True, alpha=args.alpha, 
                 Lambda=args.Lambda, epsilon=args.epsilon, 
                 h_lr=args.h_lr, q_lr=args.q_lr, pi_lr=args.pi_lr, 
                 enc_lr=args.enc_lr, device=args.device, 
                 logger=logger if args.log_loss else None, max_env_steps=args.max_episode_length)
    '''
    #policy = RIS(state_dim=state_dim, action_dim=action_dim, image_env=True, alpha=args.alpha, Lambda=args.Lambda, epsilon=args.epsilon, h_lr=args.h_lr, q_lr=args.q_lr, pi_lr=args.pi_lr, enc_lr=args.enc_lr, device=args.device, logger=logger if args.log_loss else None)
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

    validate_tasks = [1, 10, 25]

    for task_id in validate_tasks:
        # Initialize environment
        #obs = env.reset(val_scenario_idx=task_id)
        obs = env.reset(id=task_id, val_key=val_key)

        # debug
        print("task id:", task_id, "obs_x:", obs["observation"][0])
        if not args.no_video:
            images = []

        done = False
        state = obs["observation"]
        goal = obs["desired_goal"]
        t = 0

        # debug subgoals
        with torch.no_grad():
            #changed
            encoded_state = torch.FloatTensor(state).to(args.device).unsqueeze(0)
            encoded_goal = torch.FloatTensor(goal).to(args.device).unsqueeze(0)
            #x_state, x_f_state = get_img_feat_from_rollout(encoded_state, c_count=1)
            #x_goal, x_f_goal = get_img_feat_from_rollout(encoded_goal, c_count=1)
            #encoded_state = policy.encoder(x_state, x_f_state)
            #encoded_goal = policy.encoder(x_goal, x_f_goal)
            subgoal_distribution = policy.subgoal_net(encoded_state, encoded_goal)
            subgoal = subgoal_distribution.rsample()

            fig = plt.figure()
            fig.add_subplot(111)
            #x_f_state_to_draw = x_f_state.cpu()
            #x_f_goal_to_draw = x_f_goal.cpu()
            x_f_state_to_draw = encoded_state.cpu()
            x_f_goal_to_draw = encoded_goal.cpu()
            plt.scatter([x_f_state_to_draw[0][0], x_f_goal_to_draw[0][0]], 
                        [x_f_state_to_draw[0][1], x_f_goal_to_draw[0][1]])

            dx = x_f_state_to_draw[0][0] - x_f_goal_to_draw[0][0]
            dy = x_f_state_to_draw[0][1] - x_f_goal_to_draw[0][1]
            euc_dist = np.sqrt(dx ** 2 - dy ** 2)
            K =  euc_dist / policy.value(encoded_state, encoded_goal)
            euc_state_to_subgoal = policy.value(encoded_state, subgoal) * K
            euc_subgoal_to_goal = policy.value(subgoal, encoded_goal) * K
            circle_state_subgoal = plt.Circle((x_f_state_to_draw[0][0], x_f_state_to_draw[0][1]), 
                                              euc_state_to_subgoal, color='b', fill=False)
            circle_goal_subgoal = plt.Circle((x_f_goal_to_draw[0][0], x_f_goal_to_draw[0][1]), 
                                             euc_subgoal_to_goal, color='g', fill=False)
            ax = fig.gca()
            ax.add_patch(circle_state_subgoal)
            ax.add_patch(circle_goal_subgoal)

            plt.scatter([subgoal.cpu()[0][0]], [subgoal.cpu()[0][1]])
            
            if not args.no_video:
                fig.canvas.draw()
                data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                img = wandb.Image(data)
                run.log({f"agent_state": img})


        while not done:
            t += 1
            # debug
            print("step:", t, end=" ")
        
            action = policy.select_action(state, goal)
            # debug
            #action = np.array([1, -0.5])
            print("action:", action, end=" ")

            # Perform action
            next_obs, reward, done, info = env.step(action) 
            print("reward:", reward, 
                  "dist to goal:", info["dist_to_goal"], 
                  "agent theta:", info["agent_state"][2],
                  "agent steer:", info["agent_state"][4],
                  "episode ends:", done)
            print("current step in env:", info["last_step_num"])

            #done = done or t >= int(args.max_timesteps)
            if done:
                if info["geometirc_goal_achieved"]: success_rate += 1
                else: 
                    fail_rate += 1
                    failed_tasks_idx.append(task_id)
                run.log({f"success_rate": success_rate / len(validate_tasks)})
                run.log({f"fail_rate": fail_rate / len(validate_tasks)})
                run.log({f"task count": len(validate_tasks)})
                
            next_state = next_obs["observation"]

            state = next_state
            obs = next_obs
            
            #with torch.no_grad():
                #changed
                #encoded_state = torch.FloatTensor(state).to(args.device).unsqueeze(0)
                #encoded_goal = torch.FloatTensor(goal).to(args.device).unsqueeze(0)
                #x_state, x_f_state = get_img_feat_from_rollout(encoded_state, c_count=1)
                #x_goal, x_f_goal = get_img_feat_from_rollout(encoded_goal, c_count=1)
                #encoded_state = policy.encoder(x_state, x_f_state)
                #encoded_goal = policy.encoder(x_goal, x_f_goal)
                #run.log({f"V state to subgoal{epoch}": policy.value(encoded_state, subgoal),
                #         f"V subgoal to goal{epoch}": policy.value(subgoal, encoded_goal),
                #         f"V state to goal{epoch}": policy.value(encoded_state, encoded_goal)})
                #print()
            

            if not args.no_video:
                image = env.render()
                images.append(image)
                #from utilite_video_generator import create_video_from_imgs
                #video = create_video_from_imgs(images)
                
        if not args.no_video:    
            from utilite_video_generator import create_video_from_imgs
            video = create_video_from_imgs(images)    
            run.log({"validate_video": wandb.Video(video, fps=20)})
            """
            # save episode observations
            save_obs = False
            if save_obs:
                observed_next_state = info["obs_to_validation"]
                observed_next_state = np.concatenate([observed_next_state[0:1, :, :] + observed_next_state[1:2, :, :],
                                                    observed_next_state[2:3, :, :],   
                                                    observed_next_state[4:5, :, :]], axis=0)
                observed_next_state = np.moveaxis(observed_next_state, 0, -1)
                plt.imshow(observed_next_state)
                plt.savefig(observation_dir_name + '/' + f'fig{t}.png')
            
            # save episode video
            if not args.no_video:
                if args.validate_bad_cases and (not info["terminal_state"] and done) or \
                        not args.validate_bad_cases and done:
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
            """

    print("failed tasks idxs:", failed_tasks_idx)




