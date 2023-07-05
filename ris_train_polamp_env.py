import os
import time
import json

import torch
import numpy as np
import random
import argparse
import wandb
import gym
from gym.envs.registration import register
import matplotlib.pyplot as plt

from utils.logger import Logger
from polamp_RIS import RIS
from polamp_RIS import normalize_state
from polamp_HER import HERReplayBuffer, PathBuilder
from polamp_env.lib.utils_operations import generateDataSet


def evalPolicy(policy, env, 
               plot_full_env=True, plot_subgoals=False, plot_value_function=False, 
               plot_only_agent_values=False, plot_actions=False, render_env=False, 
               plot_obstacles=False, 
               video_task_id=0, video_task_map="map0", 
               data_to_plot={}, show_data_to_plot=True, 
               eval_strategy=None,
               validate_one_task=False):
    plot_obstacles = env.static_env
    assert (plot_subgoals and policy.use_encoder and policy.use_decoder) or not plot_subgoals
    #assert (policy.use_encoder and not plot_subgoals) or not policy.use_encoder, "cant plot subgoals with encoder"
    assert plot_full_env != render_env, "only show subgoals video or render env"
    validation_info = {}
    if render_env:
        images = []
        images.append(env.render())    
    if plot_full_env:
        images = []
        if plot_value_function:
            if plot_only_agent_values:
                fig = plt.figure(figsize=[6.4 * 2, 4.8])
                ax_states = fig.add_subplot(121)
                ax_values_agent = fig.add_subplot(122)
                ax_values_s = [ax_values_agent]
            else:
                fig = plt.figure(figsize=[6.4*2, 4.8*3])
                ax_states = fig.add_subplot(321)
                ax_values_agent = fig.add_subplot(322)
                ax_values_left = fig.add_subplot(323)
                ax_values_right = fig.add_subplot(324)
                ax_values_down = fig.add_subplot(325)
                ax_values_up = fig.add_subplot(326)
                ax_values_s = [ax_values_agent, 
                            ax_values_left, ax_values_right, 
                            ax_values_down, ax_values_up]
        else:
            fig = plt.figure(figsize=[6.4, 4.8])
            ax_states = fig.add_subplot(111)

    state_distrs = {"x": [], "start_x": [], "y": [], "theta": [], "v": [], "steer": []}
    goal_dists = {"goal_x": [], "goal_y": [], "goal_theta": [], "goal_v": [], "goal_steer": []}
    max_state_vals = {}
    min_state_vals = {}
    max_goal_vals = {}
    min_goal_vals = {}
    action_info = {"max_linear_acc": None, "max_steer_rate": None, "min_linear_acc": None, "min_steer_rate": None}
    mean_actions = {"a": [], "v_s": []}
    final_distances = []
    successes = [] 
    acc_rewards = []
    episode_lengths = []
    task_statuses = []

    for val_key in env.maps.keys():
        eval_tasks = len(env.valTasks[val_key])
        for task_id in range(eval_tasks):  
            if validate_one_task and (task_id != video_task_id or val_key != video_task_map):
                continue
            obs = env.reset(id=task_id, val_key=val_key)
            info = {}
            agent = env.environment.agent.current_state
            goal = env.environment.agent.goal_state
            info["agent_state"] = [agent.x, agent.y, agent.theta, agent.v, agent.steer]
            info["goal_state"] = [goal.x, goal.y, goal.theta, goal.v, goal.steer]
            done = False
            state = obs["observation"]
            goal = obs["desired_goal"]
            t = 0
            acc_reward = 0
            state_distrs["start_x"].append(state[0])

            while not done:
                if (plot_full_env or plot_value_function) and task_id == video_task_id and val_key == video_task_map:
                    env_min_x = env.dataset_info["min_x"]
                    env_max_x = env.dataset_info["max_x"]
                    env_min_y = env.dataset_info["min_y"]
                    env_max_y = env.dataset_info["max_y"]
                    grid_resolution_x = 20
                    grid_resolution_y = 20

                    with torch.no_grad():
                        to_torch_state = torch.FloatTensor(state).to(policy.device).unsqueeze(0)
                        to_torch_goal = torch.FloatTensor(goal).to(policy.device).unsqueeze(0)
                        if policy.use_encoder:
                            encoded_state = policy.encoder(to_torch_state)
                            encoded_goal = policy.encoder(to_torch_goal)
                        else:
                            encoded_state = to_torch_state
                            encoded_goal = to_torch_goal
                        subgoal_distribution = policy.subgoal_net(encoded_state, encoded_goal)
                        subgoal = subgoal_distribution.loc
                        if plot_subgoals:
                            def generate_subgoals(encoded_state, encoded_goal, subgoals, K=2, add_to_end=True):
                                if K == 0:
                                    return
                                subgoal_distribution = policy.subgoal_net(encoded_state, encoded_goal)
                                subgoal = subgoal_distribution.loc
                                if add_to_end:
                                    subgoals.append(subgoal)
                                else:
                                    subgoals.insert(0, subgoal)
                                generate_subgoals(encoded_state, subgoal, subgoals, K-1, add_to_end=False)
                                generate_subgoals(subgoal, encoded_goal, subgoals, K-1, add_to_end=True)
                            subgoals = []
                            generate_subgoals(encoded_state, encoded_goal, subgoals, K=2)
                        
                        x_agent = info["agent_state"][0]
                        y_agent = info["agent_state"][1]
                        theta_agent = info["agent_state"][2]
                        x_goal = info["goal_state"][0]
                        y_goal = info["goal_state"][1]
                        theta_goal = info["goal_state"][2]
                        car_length = 2

                        current_state = env.environment.agent.current_state
                        center_state = env.environment.agent.dynamic_model.shift_state(current_state)
                        agentBB = env.environment.getBB(center_state, ego=True)
                        ax_states.scatter(np.linspace(agentBB[0][0].x, agentBB[0][1].x, 500), 
                                        np.linspace(agentBB[0][0].y, agentBB[0][1].y, 500), 
                                        color="green", s=1)
                        ax_states.scatter(np.linspace(agentBB[1][0].x, agentBB[1][1].x, 500), 
                                        np.linspace(agentBB[1][0].y, agentBB[1][1].y, 500), 
                                        color="green", s=1)
                        ax_states.scatter(np.linspace(agentBB[2][0].x, agentBB[2][1].x, 500), 
                                        np.linspace(agentBB[2][0].y, agentBB[2][1].y, 500), 
                                        color="green", s=1)
                        ax_states.scatter(np.linspace(agentBB[3][0].x, agentBB[3][1].x, 500), 
                                        np.linspace(agentBB[3][0].y, agentBB[3][1].y, 500), 
                                        color="green", s=1)
                    
                        ax_states.set_ylim(bottom=env_min_y, top=env_max_y)
                        ax_states.set_xlim(left=env_min_x, right=env_max_x)
                        ax_states.scatter([x_agent], [y_agent], color="green", s=50)
                        ax_states.text(x_agent + 0.05, y_agent + 0.05, "agent")
                        ax_states.scatter([np.linspace(x_agent, x_agent + car_length*np.cos(theta_agent), 100)], 
                                        [np.linspace(y_agent, y_agent + car_length*np.sin(theta_agent), 100)], 
                                        color="green", s=5)
                        ax_states.scatter([x_goal], [y_goal], color="yellow", s=50)
                        ax_states.text(x_goal + 0.05, y_goal + 0.05, "goal")
                        ax_states.scatter([np.linspace(x_goal, x_goal + car_length*np.cos(theta_goal), 100)], 
                                        [np.linspace(y_goal, y_goal + car_length*np.sin(theta_goal), 100)], 
                                        color="yellow", s=5)
                        if plot_subgoals:
                            for ind, subgoal in enumerate(subgoals):
                                if env.PPO_agent_observation:
                                    assert 1 == 0, "didnt implement"
                                    x_subgoal = subgoal.cpu()[0][0]
                                    y_subgoal = subgoal.cpu()[0][1]
                                    theta_subgoal = subgoal.cpu()[0][2]
                                else:
                                    decoded_subgoal = policy.encoder.decoder(subgoal).cpu()
                                    x_subgoal = decoded_subgoal[0][0]
                                    y_subgoal = decoded_subgoal[0][1]
                                    theta_subgoal = decoded_subgoal[0][2]
                                ax_states.scatter([x_subgoal], [y_subgoal], color="orange", s=50)
                                ax_states.scatter([np.linspace(x_subgoal, x_subgoal + car_length*np.cos(theta_subgoal), 100)], 
                                                [np.linspace(y_subgoal, y_subgoal + car_length*np.sin(theta_subgoal), 100)], 
                                                color="orange", s=5)
                        if plot_obstacles:
                            for obstacle in env.environment.obstacle_segments:
                                ax_states.scatter(np.linspace(obstacle[0][0].x, obstacle[0][1].x, 500), 
                                                np.linspace(obstacle[0][0].y, obstacle[0][1].y, 500), 
                                                color="blue", s=1)
                                ax_states.scatter(np.linspace(obstacle[1][0].x, obstacle[1][1].x, 500), 
                                                np.linspace(obstacle[1][0].y, obstacle[1][1].y, 500), 
                                                color="blue", s=1)
                                ax_states.scatter(np.linspace(obstacle[2][0].x, obstacle[2][1].x, 500), 
                                                np.linspace(obstacle[2][0].y, obstacle[2][1].y, 500), 
                                                color="blue", s=1)
                                ax_states.scatter(np.linspace(obstacle[3][0].x, obstacle[3][1].x, 500), 
                                                np.linspace(obstacle[3][0].y, obstacle[3][1].y, 500), 
                                                color="blue", s=1)
                        if len(data_to_plot) != 0 and show_data_to_plot:
                            if "dataset_x" in data_to_plot and "dataset_y" in data_to_plot:
                                ax_states.scatter(data_to_plot["dataset_x"], 
                                                data_to_plot["dataset_y"], 
                                                color="red", s=5)
                            if "train_step_x" in data_to_plot and "train_step_y" in data_to_plot:
                                ax_states.scatter(data_to_plot["train_step_x"], 
                                                data_to_plot["train_step_y"], 
                                                color="red", s=3)
                        #if plot_subgoals:
                        #    ax_states.text(subgoal.cpu()[0][0] + 0.05, subgoal.cpu()[0][1] + 0.05, f"{ind + 1}")
                        ax_states.text(env_max_x - 10, env_max_y - 1.5, f"R:{acc_reward}")
                        ax_states.text(env_max_x - 3.5, env_max_y - 1.5, f"t:{t}")

                        # values plot
                        if plot_value_function:
                            def plot_values(ax_values, theta):
                                ax_values.set_ylim(bottom=env_min_y, top=env_max_y)
                                ax_values.set_xlim(left=env_min_x, right=env_max_x)
                                max_state_value = 1  
                                grid_states = []              
                                grid_goals = []
                                grid_dx = (env_max_x - env_min_x) / grid_resolution_x
                                grid_dy = (env_max_y - env_min_y) / grid_resolution_y
                                for grid_state_y in np.linspace(env_min_y + grid_dy/2, env_max_y - grid_dy/2, grid_resolution_y):
                                    for grid_state_x in np.linspace(env_min_x + grid_dx/2, env_max_x - grid_dx/2, grid_resolution_x):
                                        if env.add_frame_stack:
                                            agent_state = [grid_state_x, grid_state_y, theta]
                                            agent_state.extend([0 for _ in range(env.agent_state_len - 3)])
                                            grid_state = []
                                            for _ in range(env.frame_stack):
                                                grid_state.extend(agent_state)
                                        else:
                                            grid_state = [grid_state_x, grid_state_y]
                                            grid_state.extend([theta])
                                            grid_state.extend([0 for _ in range(len(state) - 3)])
                                        grid_states.append(grid_state)
                                grid_states = torch.FloatTensor(np.array(grid_states)).to(policy.device)
                                assert type(grid_states) == type(encoded_state), f"{type(grid_states)} == {type(encoded_state)}"                
                                grid_goals = torch.FloatTensor([goal for _ in range(grid_resolution_x * grid_resolution_y)]).to(policy.device)
                                assert grid_goals.shape == grid_states.shape
                                grid_vs = policy.value(grid_states, grid_goals)
                                grid_vs = grid_vs.detach().cpu().numpy().reshape(grid_resolution_x, grid_resolution_y)[::-1]                
                                img = ax_values.imshow(grid_vs, extent=[env_min_x,env_max_x, env_min_y,env_max_y])
                                cb = fig.colorbar(img)
                                ax_values.scatter([np.linspace(env_max_x - 3.5, env_max_x - 3.5 + car_length*np.cos(theta), 100)], 
                                                [np.linspace(env_max_y - 1.5, env_max_y - 1.5 + car_length*np.sin(theta), 100)], 
                                                color="black", s=5)
                                ax_values.scatter([env_max_x - 3.5], [env_max_y - 1.5], color="black", s=40)
                                ax_values.scatter([x_agent], [y_agent], color="green", s=100)
                                ax_values.scatter([np.linspace(x_agent, x_agent + car_length*np.cos(theta_agent), 100)], 
                                                [np.linspace(y_agent, y_agent + car_length*np.sin(theta_agent), 100)], 
                                                color="black", s=5)
                                if plot_subgoals and plot_value_function:
                                    assert 1 == 0, "didnt implement for decoder"
                                    for ind, subgoal in enumerate(subgoals):
                                        ax_values.scatter([subgoal.cpu()[0][0]], [subgoal.cpu()[0][1]], color="orange", s=100)
                                        ax_values.text(subgoal.cpu()[0][0] + 0.05, subgoal.cpu()[0][1] + 0.05, f"{ind + 1}")
                                        ax_values.scatter([np.linspace(subgoal.cpu()[0][0], subgoal.cpu()[0][0] + car_length*np.cos(subgoal.cpu()[0][2]), 100)], 
                                                    [np.linspace(subgoal.cpu()[0][1], subgoal.cpu()[0][1] + car_length*np.sin(subgoal.cpu()[0][2]), 100)], 
                                                    color="orange", s=5)
                                ax_values.scatter([x_goal], [y_goal], color="yellow", s=100)
                                ax_values.scatter([np.linspace(x_goal, x_goal + car_length*np.cos(theta_goal), 100)], 
                                                [np.linspace(y_goal, y_goal + car_length*np.sin(theta_goal), 100)], 
                                                color="black", s=5)

                                return cb
                            cbs = [plot_values(ax, theta=theta) for ax, theta in zip(ax_values_s, [theta_agent, np.pi, 0, np.pi/2, -np.pi/2])]
                        fig.canvas.draw()
                        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                        images.append(data)
                        if plot_value_function:
                            for cb in cbs:
                                cb.remove()
                            for ax_values in ax_values_s:
                                ax_values.clear()
                        ax_states.clear()

                state_distrs["x"].append(state[0])
                state_distrs["y"].append(state[1])
                state_distrs["theta"].append(state[2])
                state_distrs["v"].append(state[3])
                state_distrs["steer"].append(state[4])

                goal_dists["goal_x"].append(goal[0])
                goal_dists["goal_y"].append(goal[1])
                goal_dists["goal_theta"].append(goal[2])
                goal_dists["goal_v"].append(goal[3])
                goal_dists["goal_steer"].append(goal[4])
                
                if eval_strategy is None:
                    action = policy.select_action(state, goal)
                else:
                    print("EVAL ACTION = ", eval_strategy)
                    action = eval_strategy
                if action_info["max_linear_acc"] is None:
                    action_info["max_linear_acc"] = action[0]
                    action_info["max_steer_rate"] = action[1]
                    action_info["min_linear_acc"] = action[0]
                    action_info["min_steer_rate"] = action[1]
                else:
                    action_info["max_linear_acc"] = max(action_info["max_linear_acc"], action[0])
                    action_info["max_steer_rate"] = max(action_info["max_steer_rate"], action[1])
                    action_info["min_linear_acc"] = min(action_info["min_linear_acc"], action[0])
                    action_info["min_steer_rate"] = min(action_info["min_steer_rate"], action[1])
                mean_actions["a"].append(action[0])
                mean_actions["v_s"].append(action[1])

                next_obs, reward, done, info = env.step(action) 

                acc_reward += reward
                
                next_state = next_obs["observation"]
                state = next_state

                if render_env and task_id == video_task_id:
                    images.append(env.render())
                t += 1

            final_distances.append(info["dist_to_goal"])
            success = 1.0 * info["geometirc_goal_achieved"]
            task_status = "success"
            if env.static_env: 
                if "Collision" in info:
                    task_status = "collision"
                    success = 0.0
            successes.append(success)
            episode_lengths.append(info["last_step_num"])
            acc_rewards.append(acc_reward)
            task_statuses.append((val_key, task_id, task_status))

    eval_distance = np.mean(final_distances) 
    success_rate = np.mean(successes)
    eval_reward = np.mean(acc_rewards)
    eval_episode_length = np.mean(episode_lengths)

    for key in state_distrs:
        max_state_vals["max_" + str(key)] = np.max(state_distrs[key])
        min_state_vals["min_" + str(key)] = np.min(state_distrs[key])
        state_distrs[key] = np.mean(state_distrs[key])
    for key in goal_dists:
        max_goal_vals["max_" + str(key)] = np.max(goal_dists[key])
        min_goal_vals["min_" + str(key)] = np.min(goal_dists[key])
        goal_dists[key] = np.mean(goal_dists[key])

    env.close()
    if plot_full_env:
        plt.close()
    if plot_full_env or render_env:
        images = np.transpose(np.array(images), axes=[0, 3, 1, 2])
    validation_info["task_statuses"] = task_statuses
    validation_info["images"] = images
    validation_info["action_info"] = action_info

    return eval_distance, success_rate, eval_reward, \
           [state_distrs, max_state_vals, min_state_vals], \
           [goal_dists, max_goal_vals, min_goal_vals], mean_actions, eval_episode_length, images, validation_info


def sample_and_preprocess_batch(replay_buffer, batch_size=256, device=torch.device("cuda")):
    # Extract 
    batch = replay_buffer.random_batch(batch_size)
    state_batch         = batch["observations"]
    action_batch        = batch["actions"]
    next_state_batch    = batch["next_observations"]
    goal_batch          = batch["resampled_goals"]
    reward_batch        = batch["rewards"]
    done_batch          = batch["terminals"]
    agent_state_batch   = batch["state_observation"]
    goal_state_batch    = batch["state_desired_goal"]
    if env.static_env and env.add_collision_reward: 
        current_step_batch  = batch["current_step"] 
        collision_batch     = batch["collision"]
        if env.add_frame_stack:
            current_step_batch  = current_step_batch[:, 0:1]
            collision_batch     = collision_batch[:, 0:1]
    
    # Compute sparse rewards: -1 for all actions until the goal is reached
    if env.PPO_agent_observation:
        assert 1 == 0, "didnt implement ppo agent obs"
        reward_batch = np.sqrt(np.power(np.array(agent_state_batch - goal_state_batch)[:, :2], 2).sum(-1, keepdims=True)) # distance: next_state to goal
    else:
        reward_batch = np.sqrt(np.power(np.array(next_state_batch - goal_batch)[:, :2], 2).sum(-1, keepdims=True)) # distance: next_state to goal

    if env.static_env and env.collision_reward_to_episode_end:
        done_batch   = 1.0 * ( (1.0 * (reward_batch < env.SOFT_EPS) + collision_batch) >= 1.0)
        reward_batch = (- np.ones_like(done_batch) * env.abs_time_step_reward) * (1.0 - collision_batch) \
                    + (current_step_batch - env._max_episode_steps) * collision_batch
    elif env.static_env:
        if env.add_ppo_reward:
            assert 1 == 0, "didnt implement add ppo reward"
            done_batch   = 1.0 * (reward_batch < env.SOFT_EPS) # terminal condition
            reward_batch = env.HER_reward(state=state_batch, action=action_batch, 
                                          next_state=next_state_batch, goal=goal_batch, 
                                          collision=collision_batch, goal_was_reached=done_batch, 
                                          step_counter=current_step_batch)
        else:
            done_batch   = 1.0 * (reward_batch < env.SOFT_EPS) # terminal condition
            reward_batch = (- np.ones_like(done_batch) * env.abs_time_step_reward) * (1.0 - collision_batch) \
                            + (env.collision_reward) * collision_batch
    else:
        done_batch   = 1.0 * (reward_batch < env.SOFT_EPS) # terminal condition
        reward_batch = - np.ones_like(done_batch) * env.abs_time_step_reward
    
    # Convert to Pytorch
    state_batch         = torch.FloatTensor(state_batch).to(device)
    action_batch        = torch.FloatTensor(action_batch).to(device)
    reward_batch        = torch.FloatTensor(reward_batch).to(device)
    next_state_batch    = torch.FloatTensor(next_state_batch).to(device)
    done_batch          = torch.FloatTensor(done_batch).to(device)
    goal_batch          = torch.FloatTensor(goal_batch).to(device)

    return state_batch, action_batch, reward_batch, next_state_batch, done_batch, goal_batch

if __name__ == "__main__":	
    parser = argparse.ArgumentParser()

    parser.add_argument("--env",                  default="polamp_env")
    parser.add_argument("--test_env",             default="polamp_env")
    parser.add_argument("--dataset",              default="medium_dataset") # test_medium_dataset, medium_dataset, safety_dataset, ris_easy_dataset
    parser.add_argument("--uniform_feasible_train_dataset", default=False)
    parser.add_argument("--random_train_dataset",           default=False)

    parser.add_argument("--epsilon",            default=1e-16, type=float)
    parser.add_argument("--start_timesteps",    default=1e4, type=int) 
    parser.add_argument("--eval_freq",          default=int(3e4), type=int) # 2e4
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
    parser.add_argument("--pi_lr",              default=1e-4, type=float)
    
    parser.add_argument("--use_decoder",        default=True, type=bool)
    parser.add_argument("--use_encoder",        default=True, type=bool)
    parser.add_argument("--state_dim",          default=20, type=int)
    parser.add_argument("--using_wandb",        default=True, type=bool)
    parser.add_argument("--wandb_project",      default="train_ris_sac_polamp", type=str)
    parser.add_argument('--log_loss', dest='log_loss', action='store_true')
    parser.add_argument('--no-log_loss', dest='log_loss', action='store_false')
    parser.set_defaults(log_loss=True)
    args = parser.parse_args()
    print(args)

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
    else:
        total_maps = 1
    dataSet = generateDataSet(our_env_config, name_folder=args.dataset, total_maps=total_maps, dynamic=False)
    maps, trainTask, valTasks = dataSet["obstacles"]
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
    vectorized = True
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
    logger = Logger(vars(args), save_git_head_hash=False)
    if args.using_wandb:
        run = wandb.init(project=args.wandb_project)
    
    # Initialize policy
    env_state_bounds = {"x": 100, "y": 100, "theta": 3.14,
                        "v": 2.778, "steer": 0.7854}
    policy = RIS(state_dim=state_dim, action_dim=action_dim, 
                 alpha=args.alpha,
                 use_decoder=args.use_decoder,
                 use_encoder=args.use_encoder,
                 Lambda=args.Lambda, epsilon=args.epsilon,
                 h_lr=args.h_lr, q_lr=args.q_lr, pi_lr=args.pi_lr, 
                 device=args.device, logger=logger if args.log_loss else None, 
                 env_state_bounds=env_state_bounds,
                 env_obs_dim=env_obs_dim, add_ppo_reward=env.add_ppo_reward)

    # Initialize replay buffer and path_builder
    replay_buffer = HERReplayBuffer(
        max_size=args.replay_buffer_size,
        env=env,
        fraction_goals_are_rollout_goals = 0.2,
        fraction_resampled_goals_are_env_goals = 0.0,
        fraction_resampled_goals_are_replay_buffer_goals = 0.5,
        ob_keys_to_save     =["state_observation", "state_achieved_goal", "state_desired_goal", "current_step", "collision"],
        desired_goal_keys   =["desired_goal", "state_desired_goal"],
        observation_key     = 'observation',
        desired_goal_key    = 'desired_goal',
        achieved_goal_key   = 'achieved_goal',
        vectorized          = vectorized 
    )
    path_builder = PathBuilder()

    if load_results:
        policy.load(folder)

    # Initialize environment
    obs = env.reset()
    done = False
    state = obs["observation"]
    goal = obs["desired_goal"]
    episode_timesteps = 0
    episode_num = 0 
    old_success_rate = None
    save_policy_count = 0 

    assert args.eval_freq > env._max_episode_steps, "logger is erased after each eval"
    logger.store(train_step_x = state[0])
    logger.store(train_step_y = state[1])
    buffer_size = 0 # for args.her_corrections

    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1
        print("step:", t, end=" ")

        # Select action
        start_action_time = time.time()
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.select_action(state, goal)
        action_time = time.time() - start_action_time
        logger.store(action_time = action_time)

        # Perform action
        start_step_time = time.time()
        next_obs, reward, done, _ = env.step(action) 
        step_time = time.time() - start_step_time
        logger.store(step_time = step_time)

        next_state = next_obs["observation"]
        next_agent_state = next_obs["state_observation"]

        transition_added = False # for args.her_corrections
        if env.her_corrections:
            if next_obs["collision_happend_on_trajectory"] == 0.0:
                buffer_size += 1 # for args.her_corrections
                transition_added = True # for args.her_corrections
                path_builder.add_all(
                    observations=obs,
                    actions=action,
                    rewards=reward,
                    next_observations=next_obs,
                    terminals=[1.0*done]
                )
        else:
            buffer_size += 1 # for args.her_corrections
            transition_added = True # for args.her_corrections
            path_builder.add_all(
                observations=obs,
                actions=action,
                rewards=reward,
                next_observations=next_obs,
                terminals=[1.0*done]
            )

        if env.static_env and env.her_corrections:
            print("collission:", next_obs["collision_happend_on_trajectory"], end=" ")
        
        state = next_state
        agent_state = next_agent_state
        obs = next_obs
        logger.store(train_step_x = agent_state[0])
        logger.store(train_step_y = agent_state[1])

        # Train agent after collecting enough data
        if t >= args.batch_size and t >= args.start_timesteps and buffer_size >= args.start_timesteps and transition_added: # for args.her_corrections
            state_batch, action_batch, reward_batch, next_state_batch, done_batch, goal_batch = sample_and_preprocess_batch(
                replay_buffer, 
                batch_size=args.batch_size,
                device=args.device
            )
            # Sample subgoal candidates uniformly in the replay buffer
            subgoal_batch = torch.FloatTensor(replay_buffer.random_state_batch(args.batch_size)).to(args.device)

            start_train_batch_time = time.time()
            policy.train(state_batch, action_batch, reward_batch, next_state_batch, done_batch, goal_batch, subgoal_batch)
            train_batch_time = time.time() - start_train_batch_time
            logger.store(train_batch_time = train_batch_time)
            print("train", args.exp_name, end=" ")

        if done: 
            # Add path to replay buffer and reset path builder
            replay_buffer.add_path(path_builder.get_all_stacked())
            path_builder = PathBuilder()
            logger.store(t=t, reward=reward)		

            # Reset environment
            obs = env.reset()
            done = False
            state = obs["observation"]
            goal = obs["desired_goal"]
            episode_timesteps = 0
            episode_num += 1 
            logger.store(dataset_x = state[0])
            logger.store(dataset_y = state[1])

        if (t + 1) % args.eval_freq == 0 and t >= args.start_timesteps:
            # Eval policy
            eval_distance, success_rate, eval_reward, \
            val_state, val_goal, \
            mean_actions, eval_episode_length, images, validation_info \
                    = evalPolicy(policy, test_env, 
                                plot_full_env=True,
                                plot_subgoals=False,
                                plot_value_function=False,
                                render_env=False,
                                plot_only_agent_values=True, 
                                data_to_plot={
                                              "train_step_x": logger.data["train_step_x"], 
                                              "train_step_y": logger.data["train_step_y"],
                                              },
                                video_task_id=len(dataSet["obstacles"][2]["map0"])-1,
                                show_data_to_plot=True)

            wandb_log_dict = {
                    'steps': logger.data["t"][-1],

                     # train logging
                     'train_adv': sum(logger.data["adv"][-args.eval_freq:]) / args.eval_freq,    
                     'train_D_KL': sum(logger.data["D_KL"][-args.eval_freq:]) / args.eval_freq,
                     'subgoal_loss': sum(logger.data["subgoal_loss"][-args.eval_freq:]) / args.eval_freq,
                     'train_critic_loss': sum(logger.data["critic_loss"][-args.eval_freq:]) / args.eval_freq,
                     'critic_value': sum(logger.data["critic_value"][-args.eval_freq:]) / args.eval_freq,
                     'target_value': sum(logger.data["target_value"][-args.eval_freq:]) / args.eval_freq,
                     'actor_loss': sum(logger.data["actor_loss"][-args.eval_freq:]) / args.eval_freq,

                     # validate logging
                     f'val_distance({args.n_eval} episodes)': eval_distance,
                     f'eval_reward({args.n_eval} episodes)': eval_reward,
                     f'val_rate({args.n_eval} episodes)': success_rate,
                     "val_episode_length": eval_episode_length,

                     # batch state
                     'train_state_x_max': sum(logger.data["train_state_x_max"][-args.eval_freq:]) / args.eval_freq,
                     'train_state_x_mean': sum(logger.data["train_state_x_mean"][-args.eval_freq:]) / args.eval_freq,
                     'train_state_x_min': sum(logger.data["train_state_x_min"][-args.eval_freq:]) / args.eval_freq,
                     'train_state_y_max': sum(logger.data["train_state_y_max"][-args.eval_freq:]) / args.eval_freq,
                     'train_state_y_mean': sum(logger.data["train_state_y_mean"][-args.eval_freq:]) / args.eval_freq,
                     'train_state_y_min': sum(logger.data["train_state_y_min"][-args.eval_freq:]) / args.eval_freq,

                     # batch sampled subgoal
                     'train_subgoal_x_max': sum(logger.data["train_subgoal_x_max"][-args.eval_freq:]) / args.eval_freq,
                     'train_subgoal_x_min': sum(logger.data["train_subgoal_x_min"][-args.eval_freq:]) / args.eval_freq,
                     'train_subgoal_x_mean': sum(logger.data["train_subgoal_x_mean"][-args.eval_freq:]) / args.eval_freq,
                     'train_subgoal_y_max': sum(logger.data["train_subgoal_y_max"][-args.eval_freq:]) / args.eval_freq,
                     'train_subgoal_y_min': sum(logger.data["train_subgoal_y_min"][-args.eval_freq:]) / args.eval_freq,
                     'train_subgoal_y_mean': sum(logger.data["train_subgoal_y_mean"][-args.eval_freq:]) / args.eval_freq,

                     # batch sampled goal
                     'train_goal_x_max': sum(logger.data["train_goal_x_max"][-args.eval_freq:]) / args.eval_freq,
                     'train_goal_x_min': sum(logger.data["train_goal_x_min"][-args.eval_freq:]) / args.eval_freq,
                     'train_goal_x_mean': sum(logger.data["train_goal_x_mean"][-args.eval_freq:]) / args.eval_freq,
                     'train_goal_y_max': sum(logger.data["train_goal_y_max"][-args.eval_freq:]) / args.eval_freq,
                     'train_goal_y_min': sum(logger.data["train_goal_y_min"][-args.eval_freq:]) / args.eval_freq,
                     'train_goal_y_mean': sum(logger.data["train_goal_y_mean"][-args.eval_freq:]) / args.eval_freq,

                     # batch sampled reward
                     'train_reward_max': sum(logger.data["train_reward_max"][-args.eval_freq:]) / args.eval_freq,
                     'train_reward_min': sum(logger.data["train_reward_min"][-args.eval_freq:]) / args.eval_freq,
                     'train_reward_mean': sum(logger.data["train_reward_mean"][-args.eval_freq:]) / args.eval_freq,

                     # batch sampled goal
                     'train_subgoal_data_x_max': sum(logger.data["train_subgoal_data_x_max"][-args.eval_freq:]) / args.eval_freq,
                     'train_subgoal_data_x_min': sum(logger.data["train_subgoal_data_x_min"][-args.eval_freq:]) / args.eval_freq,
                     'train_subgoal_data_x_mean': sum(logger.data["train_subgoal_data_x_mean"][-args.eval_freq:]) / args.eval_freq,
                     'train_subgoal_data_y_max': sum(logger.data["train_subgoal_data_y_max"][-args.eval_freq:]) / args.eval_freq,
                     'train_subgoal_data_y_min': sum(logger.data["train_subgoal_data_y_min"][-args.eval_freq:]) / args.eval_freq,
                     'train_subgoal_data_y_mean': sum(logger.data["train_subgoal_data_y_mean"][-args.eval_freq:]) / args.eval_freq,
                    }
            if args.using_wandb:
                for dict_ in val_state + val_goal:
                    for key in dict_:
                        wandb_log_dict[f"{key}"] = dict_[key]
                wandb_log_dict["validation_video"] = wandb.Video(images, fps=10, format="gif")
                run.log(wandb_log_dict)
     
            # Save (best) results
            if old_success_rate is None or success_rate >= old_success_rate:
                save_policy_count += 1
                folder = "results/{}/RIS/{}/".format(args.env, args.exp_name)
                if not os.path.exists(folder):
                    os.makedirs(folder)
                logger.save(folder + "log.pkl")
                policy.save(folder)
                run.log({"save_policy_count": save_policy_count})
            old_success_rate = success_rate

            # clean log buffer
            logger.data = dict()
            print("eval", end=" ")
        
        print()
