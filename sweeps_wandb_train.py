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
import matplotlib.patches as patches
import copy
from math import pi

from polamp_env.lib.utils_operations import normalizeAngle
from utils.logger import Logger
from polamp_RIS import RIS
from polamp_RIS import normalize_state
from polamp_HER import HERReplayBuffer, PathBuilder
from polamp_env.lib.utils_operations import generateDataSet
from polamp_env.lib.structures import State
from ris_train_polamp_env import train
#from PythonRobotics.PathPlanning.DubinsPath import dubins_path_planner


def evalPolicy(policy, env, 
               plot_full_env=True, plot_subgoals=False, plot_value_function=False,
               value_function_angles=["theta_agent", np.pi, 0, np.pi/2, -np.pi/2],
               plot_safe_bound=False, 
               plot_lidar_predictor=False,
               plot_decoder_agent_states=False,
               plot_subgoal_dispertion=False,
               plot_agent_lidar_data_encoder=False,
               plot_subgoal_lidar_data=False,
               plot_dubins_curve=False,
               add_text=False,
               plot_only_agent_values=False, plot_actions=False, render_env=False, 
               video_validate_tasks = [],                
               data_to_plot={}, dataset_plot=True, 
               eval_strategy=None,
               validate_one_task=False):
    """
        medium dataset: video_validate_tasks = [("map4", 8), ("map4", 13), ("map6", 5), ("map6", 18), ("map7", 19), ("map5", 7)]
        hard dataset: video_validate_tasks = [("map0", 2), ("map0", 5), ("map0", 10), ("map0", 15)]
    """
    assert 1.0 * plot_full_env + 1.0 * render_env >= 1, "didnt implement other"
    assert type(video_validate_tasks) == type(list())
    assert (plot_subgoals and policy.use_encoder and policy.use_decoder) or (plot_subgoals and not policy.use_encoder)
    assert (plot_decoder_agent_states and policy.use_decoder) or not plot_decoder_agent_states
    assert not plot_lidar_predictor or (plot_lidar_predictor and policy.use_lidar_predictor)
    print()

    plot_obstacles = env.static_env
    validation_info = {}
    if render_env:
        videos = []   
    if plot_full_env:
        videos = []
        if plot_value_function:
            if plot_only_agent_values:
                fig = plt.figure(figsize=[6.4 * 2, 4.8])
                ax_states = fig.add_subplot(121)
                ax_values_agent = fig.add_subplot(122)
                ax_values_s = [ax_values_agent]
                value_function_angles=["theta_agent"]
            else:
                if len(value_function_angles) == 2:
                    fig = plt.figure(figsize=[6.4, 4.8*3])
                    ax_states = fig.add_subplot(311)
                    ax_values_right = fig.add_subplot(312)
                    ax_values_down = fig.add_subplot(313)
                    ax_values_s = [ax_values_right, ax_values_down]
                elif len(value_function_angles) == 3:
                    fig = plt.figure(figsize=[6.4, 4.8*4])
                    ax_states = fig.add_subplot(411)
                    ax_values_agent = fig.add_subplot(412)
                    ax_values_right = fig.add_subplot(413)
                    ax_values_down = fig.add_subplot(414)
                    ax_values_s = [ax_values_agent, ax_values_right, ax_values_down] 
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
    acc_costs = []
    acc_collisions = []
    episode_lengths = []
    task_statuses = []

    for val_key in env.maps.keys():
        eval_tasks = len(env.valTasks[val_key])
        for task_id in range(eval_tasks):  
            need_to_plot_task = (plot_full_env or plot_value_function or render_env) \
                                and (val_key, task_id) in video_validate_tasks
            if validate_one_task and not need_to_plot_task:
                continue
            if need_to_plot_task:
                images = []
            print(f"map={val_key}", f"task={task_id}", end=" ")
            if need_to_plot_task:
                print("DO VIDEO", end=" ")
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
            acc_cost = 0
            acc_collision = 0
            state_distrs["start_x"].append(state[0])

            while not done:
                if plot_full_env and need_to_plot_task:
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
                        if plot_subgoals:
                            def generate_subgoals(encoded_state, encoded_goal, subgoals, stds, K=2, add_to_end=True):
                                if K == 0:
                                    return
                                subgoal_distribution = policy.subgoal_net(encoded_state, encoded_goal)
                                subgoal = subgoal_distribution.loc
                                if policy.high_level_without_frame:
                                    subgoal = subgoal.repeat(1, 4)
                                if policy.use_lidar_predictor:
                                    subgoal = policy.add_lidar_data_to_subgoals(subgoal, encoded_state, encoded_goal)
                                std = subgoal_distribution.scale
                                if add_to_end:
                                    subgoals.append(subgoal)
                                    stds.append(std)
                                else:
                                    subgoals.insert(0, subgoal)
                                    stds.insert(0, std)
                                generate_subgoals(encoded_state, subgoal, subgoals, stds, K-1, add_to_end=False)
                                generate_subgoals(subgoal, encoded_goal, subgoals, stds, K-1, add_to_end=True)
                            subgoals = []
                            stds = []
                            generate_subgoals(encoded_state, encoded_goal, subgoals, stds, K=2)
                        
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
                        ax_states.scatter([np.linspace(x_agent, x_agent + car_length*np.cos(theta_agent), 100)], 
                                        [np.linspace(y_agent, y_agent + car_length*np.sin(theta_agent), 100)], 
                                        color="green", s=5)
                        ax_states.scatter([x_goal], [y_goal], color="yellow", s=50)
                        ax_states.scatter([np.linspace(x_goal, x_goal + car_length*np.cos(theta_goal), 100)], 
                                        [np.linspace(y_goal, y_goal + car_length*np.sin(theta_goal), 100)], 
                                        color="yellow", s=5)
                        if plot_lidar_predictor:
                            lidar_data = policy.lidar_predictor(to_torch_state[:, 0:policy.subgoal_dim], to_torch_state, to_torch_goal).cpu().squeeze()
                            for angle, d in zip(env.environment.angle_space, lidar_data):
                                color_lidar = "green"
                                obst_x = x_agent + d.item() * np.cos(theta_agent + angle)
                                obst_y = y_agent + d.item() * np.sin(theta_agent + angle)
                                ax_states.scatter([obst_x], [obst_y], color=color_lidar, s=5)
                        if add_text:
                            ax_states.text(x_agent + 0.05, y_agent + 0.05, "agent")
                            ax_states.text(x_goal + 0.05, y_goal + 0.05, "goal")
                        if plot_subgoals:
                            for ind, (subgoal, std) in enumerate(zip(subgoals, stds)):
                                if policy.use_encoder:
                                    cuda_decoded_subgoal = policy.encoder.decoder(subgoal)
                                    decoded_subgoal = cuda_decoded_subgoal.cpu()
                                else:
                                    cuda_decoded_subgoal = subgoal
                                    decoded_subgoal = subgoal.cpu()
                                x_subgoal = decoded_subgoal[0][0].item()
                                y_subgoal = decoded_subgoal[0][1].item()
                                theta_subgoal = decoded_subgoal[0][2].item()
                                lidar_data = decoded_subgoal[0][5:44]
                                if policy.use_lidar_predictor:
                                    x_subgoal_1 = decoded_subgoal[0][policy.subgoal_dim+policy.lidar_data_dim].item()
                                    y_subgoal_1 = decoded_subgoal[0][policy.subgoal_dim+policy.lidar_data_dim+1].item()
                                    x_subgoal_2 = decoded_subgoal[0][2*(policy.subgoal_dim+policy.lidar_data_dim)].item()
                                    y_subgoal_2 = decoded_subgoal[0][2*(policy.subgoal_dim+policy.lidar_data_dim)+1].item()
                                    x_subgoal_3 = decoded_subgoal[0][3*(policy.subgoal_dim+policy.lidar_data_dim)].item()
                                    y_subgoal_3 = decoded_subgoal[0][3*(policy.subgoal_dim+policy.lidar_data_dim)+1].item()
                                if plot_lidar_predictor and ind == len(subgoals) // 2:
                                    lidar_data = policy.lidar_predictor(cuda_decoded_subgoal[:, 0:policy.subgoal_dim], to_torch_state, to_torch_goal).cpu().squeeze()
                                    for angle, d in zip(env.environment.angle_space, lidar_data):
                                        color_lidar = "red"
                                        obst_x = x_subgoal + d.item() * np.cos(theta_subgoal + angle)
                                        obst_y = y_subgoal + d.item() * np.sin(theta_subgoal + angle)
                                        ax_states.scatter([obst_x], [obst_y], color=color_lidar, s=10)
                                if plot_subgoal_lidar_data and ind == len(subgoals) // 2:
                                    for angle, d in zip(env.environment.angle_space, lidar_data):
                                        color_lidar = "orange"
                                        obst_x = x_subgoal + d * np.cos(theta_subgoal + angle)
                                        obst_y = y_subgoal + d * np.sin(theta_subgoal + angle)
                                        ax_states.scatter([obst_x], [obst_y], color=color_lidar, s=10)
                                ax_states.scatter([x_subgoal], [y_subgoal], color="orange", s=50)
                                if policy.use_lidar_predictor and ind == len(subgoals) // 2:
                                    ax_states.scatter([x_subgoal_1], [y_subgoal_1], color="orange", s=15)
                                    ax_states.text(x_subgoal_1, y_subgoal_1, "1")
                                    ax_states.scatter([x_subgoal_2], [y_subgoal_2], color="orange", s=15)
                                    ax_states.text(x_subgoal_2, y_subgoal_2, "2")
                                    ax_states.scatter([x_subgoal_3], [y_subgoal_3], color="orange", s=15)
                                    ax_states.text(x_subgoal_3, y_subgoal_3, "3")
                                ax_states.scatter([np.linspace(x_subgoal, x_subgoal + car_length*np.cos(theta_subgoal), 100)], 
                                                [np.linspace(y_subgoal, y_subgoal + car_length*np.sin(theta_subgoal), 100)], 
                                                color="orange", s=5)
                                if plot_subgoal_dispertion:
                                    if ind == len(subgoals) // 2:   
                                        x_std = std[0][0].item()
                                        y_std = std[0][1].item()
                                        rect = patches.Rectangle((x_subgoal - x_std/2, y_subgoal - y_std/2), x_std, y_std, linewidth=1, edgecolor='r', facecolor='none')
                                        ax_states.add_patch(rect)

                                if plot_dubins_curve and ind == len(subgoals) // 2:
                                    R = env.environment.agent.dynamic_model.wheel_base / np.tan(env.environment.agent.dynamic_model.max_steer)
                                    curvature = 1 / R
                                    path_x, path_y, path_yaw, mode, _ = dubins_path_planner.plan_dubins_path(
                                            x_agent, y_agent, theta_agent, x_subgoal, y_subgoal, theta_subgoal, curvature)
                                    ax_states.plot(path_x, path_y)
                                    path_x, path_y, path_yaw, mode, _ = dubins_path_planner.plan_dubins_path(
                                            x_subgoal, y_subgoal, theta_subgoal, x_goal, y_goal, theta_goal, curvature)
                                    ax_states.plot(path_x, path_y)

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
                        if len(data_to_plot) != 0 and dataset_plot:
                            if "dataset_x" in data_to_plot and "dataset_y" in data_to_plot:
                                ax_states.scatter(data_to_plot["dataset_x"], 
                                                data_to_plot["dataset_y"], 
                                                color="red", s=5)
                            if "train_step_x" in data_to_plot and "train_step_y" in data_to_plot:
                                ax_states.scatter(data_to_plot["train_step_x"], 
                                                data_to_plot["train_step_y"], 
                                                color="red", s=3)
                        ax_states.text(env_max_x - 46, env_max_y - 2, f"a:{round(env.environment.agent.action[0], 2)}")
                        ax_states.text(env_max_x - 39, env_max_y - 2, f"w:{round(env.environment.agent.action[1], 2)}")
                        ax_states.text(env_max_x - 32, env_max_y - 2, f"v:{round(env.environment.agent.current_state.v, 2)}")
                        ax_states.text(env_max_x - 25, env_max_y - 2, f"st:{round(env.environment.agent.current_state.steer * 180 / pi, 1)}")
                        ax_states.text(env_max_x - 18, env_max_y - 2, f"R:{int(acc_reward*10)/10}")
                        ax_states.text(env_max_x - 11, env_max_y - 2, f"C:{int(acc_cost*10)/10}")
                        ax_states.text(env_max_x - 4, env_max_y - 2, f"t:{t}")

                        if plot_decoder_agent_states:
                            decoded_state = policy.encoder.decoder(encoded_state).cpu()
                            x_decoded_state = decoded_state[0][0]
                            y_decoded_state = decoded_state[0][1]
                            theta_decoded_state = decoded_state[0][2]
                            lidar_data = decoded_state[0][5:44]
                            if plot_agent_lidar_data_encoder:
                                for angle, d in zip(env.environment.angle_space, lidar_data):
                                    color_lidar = "green"
                                    obst_x = x_decoded_state + d * np.cos(theta_decoded_state + angle)
                                    obst_y = y_decoded_state + d * np.sin(theta_decoded_state + angle)
                                    ax_states.scatter([obst_x], [obst_y], color=color_lidar, s=10)
                                ax_states.scatter([x_decoded_state], [y_decoded_state], color="red", s=50)
                                ax_states.scatter([np.linspace(x_decoded_state, x_decoded_state + car_length*np.cos(theta_decoded_state), 100)], 
                                                [np.linspace(y_decoded_state, y_decoded_state + car_length*np.sin(theta_decoded_state), 100)], 
                                                color="red", s=5)

                        # values plot
                        if plot_value_function:
                            def plot_values(ax_values, theta, set_lim_xy=True, return_cb=True):
                                if set_lim_xy:
                                    ax_values.set_ylim(bottom=env_min_y, top=env_max_y)
                                    ax_values.set_xlim(left=env_min_x, right=env_max_x)
                                grid_states = []              
                                grid_goals = []
                                grid_dx = (env_max_x - env_min_x) / grid_resolution_x
                                grid_dy = (env_max_y - env_min_y) / grid_resolution_y
                                for grid_state_y in np.linspace(env_min_y + grid_dy/2, env_max_y - grid_dy/2, grid_resolution_y):
                                    for grid_state_x in np.linspace(env_min_x + grid_dx/2, env_max_x - grid_dx/2, grid_resolution_x):
                                        if env.add_frame_stack:
                                            agent_state = [grid_state_x, grid_state_y, theta, 0, 0]
                                            grid_state = []
                                            if env.use_lidar_data:
                                                grid_state_struct = copy.deepcopy(env.environment.agent.current_state)
                                                grid_state_struct.x = grid_state_x
                                                grid_state_struct.y = grid_state_y
                                                grid_state_struct.theta = theta
                                                grid_state_struct.steer = 0
                                                grid_state_struct.v = 0
                                                beams_observation = env.environment.get_observation_without_env_change(grid_state_struct)
                                                agent_state.extend(beams_observation.tolist())
                                            for _ in range(env.frame_stack):
                                                grid_state.extend(agent_state)
                                        else:
                                            assert 1 == 0, "something incorrect here"
                                            grid_state = [grid_state_x, grid_state_y]
                                            grid_state.extend([theta])
                                            grid_state.extend([0 for _ in range(5 - 3)])
                                        grid_states.append(grid_state)
                                grid_states = torch.FloatTensor(np.array(grid_states)).to(policy.device)
                                assert type(grid_states) == type(encoded_state), f"{type(grid_states)} == {type(encoded_state)}"                
                                numpy_encoded_goal = encoded_goal.cpu().squeeze().numpy()
                                assert type(goal) == type(numpy_encoded_goal)
                                numpy_grid_goals = np.array([numpy_encoded_goal if policy.use_encoder else goal 
                                                for _ in range(grid_resolution_x * grid_resolution_y)])
                                grid_goals = torch.FloatTensor(numpy_grid_goals).to(policy.device)
                                if policy.use_encoder:
                                    grid_states = policy.encoder(grid_states)
                                assert grid_goals.shape == grid_states.shape, \
                                       f"doesnt equal state shape: {grid_states.shape}   to goal shape: {grid_goals.shape} "
                                grid_vs = policy.value(grid_states, grid_goals)
                                grid_vs = grid_vs.detach().cpu().numpy().reshape(grid_resolution_x, grid_resolution_y)[::-1]                
                                img = ax_values.imshow(grid_vs, extent=[env_min_x,env_max_x, env_min_y,env_max_y])
                                if return_cb:
                                    cb = fig.colorbar(img)
                                else:
                                    cb = None
                                ax_values.scatter([np.linspace(env_max_x - 3.5, env_max_x - 3.5 + car_length*np.cos(theta), 100)], 
                                                [np.linspace(env_max_y - 1.5, env_max_y - 1.5 + car_length*np.sin(theta), 100)], 
                                                color="black", s=5)
                                ax_values.scatter([env_max_x - 3.5], [env_max_y - 1.5], color="black", s=40)
                                ax_values.scatter([x_agent], [y_agent], color="green", s=100)
                                ax_values.scatter([np.linspace(x_agent, x_agent + car_length*np.cos(theta_agent), 100)], 
                                                [np.linspace(y_agent, y_agent + car_length*np.sin(theta_agent), 100)], 
                                                color="black", s=5)
                                if plot_subgoals:
                                    for ind, (subgoal, std) in enumerate(zip(subgoals, stds)):
                                        if policy.use_encoder:
                                            cuda_decoded_subgoal = policy.encoder.decoder(subgoal)
                                            decoded_subgoal = cuda_decoded_subgoal.cpu()
                                        else:
                                            cuda_decoded_subgoal = subgoal
                                            decoded_subgoal = subgoal.cpu()
                                        x_subgoal = decoded_subgoal[0][0]
                                        y_subgoal = decoded_subgoal[0][1]
                                        theta_subgoal = decoded_subgoal[0][2]
                                        ax_values.scatter([x_subgoal], [y_subgoal], color="orange", s=50)
                                        ax_values.scatter([np.linspace(x_subgoal, x_subgoal + car_length*np.cos(theta_subgoal), 100)], 
                                                        [np.linspace(y_subgoal, y_subgoal + car_length*np.sin(theta_subgoal), 100)], 
                                                        color="orange", s=5)
                                        if plot_subgoal_dispertion:
                                            if ind == len(subgoals) // 2:   
                                                x_std = std[0][0].item()
                                                y_std = std[0][1].item()
                                                rect = patches.Rectangle((x_subgoal - x_std/2, y_subgoal - y_std/2), x_std, y_std, linewidth=1, edgecolor='r', facecolor='none')
                                                ax_values.add_patch(rect)
                                ax_values.scatter([x_goal], [y_goal], color="yellow", s=100)
                                ax_values.scatter([np.linspace(x_goal, x_goal + car_length*np.cos(theta_goal), 100)], 
                                                [np.linspace(y_goal, y_goal + car_length*np.sin(theta_goal), 100)], 
                                                color="black", s=5)

                                return cb

                        if plot_value_function:
                            cbs = [plot_values(ax, theta=theta_agent) if theta=="theta_agent" else plot_values(ax, theta=theta) 
                                    for ax, theta in zip(ax_values_s, value_function_angles)]
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
                    action = policy.select_deterministic_action(state, goal)
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
                acc_cost += info["cost"]
                acc_collision += 1.0 * ("Collision" in info)
                
                next_state = next_obs["observation"]
                state = next_state

                if render_env and need_to_plot_task:
                    images.append(env.render())
                t += 1

            if need_to_plot_task:
                if plot_full_env and render_env:
                    images_full_env = [image for ind, image in enumerate(images) if (ind+1) % 2 != 0]
                    images_render = [image for ind, image in enumerate(images) if (ind+1) % 2 == 0]
                    videos.append((val_key+"_full_env", task_id, images_full_env))
                    videos.append((val_key+"_render", task_id, images_render))
                else:
                    videos.append((val_key, task_id, images))
            final_distances.append(info["EuclideanDistance"])
            success = 1.0 * info["goal_achieved"]
            task_status = "success"
            if env.static_env: 
                if "Collision" in info:
                    task_status = "collision"
                    success = 0.0
            successes.append(success)
            episode_lengths.append(info["last_step_num"])
            acc_rewards.append(acc_reward)
            acc_collisions.append(acc_collision)
            acc_costs.append(acc_cost)
            task_statuses.append((val_key, task_id, task_status))

    eval_distance = np.mean(final_distances) 
    success_rate = np.mean(successes)
    eval_reward = np.mean(acc_rewards)
    eval_cost = np.mean(acc_costs)
    eval_collisions = np.mean(acc_collisions)
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
        videos = [(map_name, task_indx, np.transpose(np.array(video), axes=[0, 3, 1, 2])) for map_name, task_indx, video in videos]
    validation_info["task_statuses"] = task_statuses
    validation_info["videos"] = videos
    validation_info["action_info"] = action_info
    validation_info["eval_cost"] = eval_cost
    validation_info["eval_collisions"] = eval_collisions

    return eval_distance, success_rate, eval_reward, \
           [state_distrs, max_state_vals, min_state_vals], \
           [goal_dists, max_goal_vals, min_goal_vals], mean_actions, eval_episode_length, validation_info


def sample_and_preprocess_batch(replay_buffer, env, batch_size=256, device=torch.device("cuda")):
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
    if env.static_env:
        clearance_is_enough_batch     = batch["clearance_is_enough"]
    if env.static_env and env.add_collision_reward: 
        current_step_batch  = batch["current_step"] 
        collision_batch     = batch["collision"]        
        if env.add_frame_stack:
            current_step_batch  = current_step_batch[:, 0:1]
            collision_batch     = collision_batch[:, 0:1]
    
    # Compute sparse rewards: -1 for all actions until the goal is reached
    reward_batch = np.sqrt(np.power(np.array(next_state_batch - goal_batch)[:, :2], 2).sum(-1, keepdims=True)) # distance: next_state to goal
    angle_batch = abs(np.vectorize(normalizeAngle)(abs(np.array(next_state_batch - goal_batch)[:, 2:3])))
    if env.static_env:
        cost_batch = (np.ones_like(done_batch) * next_state_batch[:, 3:4]) * (1.0 - clearance_is_enough_batch)
    else:
        cost_batch = (- np.ones_like(done_batch) * 0)
    if env.static_env and env.add_collision_reward:
        if env.add_ppo_reward:
            assert 1 == 0, "didnt implement add ppo reward"
            done_batch   = 1.0 * (reward_batch < env.SOFT_EPS) # terminal condition
            reward_batch = env.HER_reward(state=state_batch, action=action_batch, 
                                          next_state=next_state_batch, goal=goal_batch, 
                                          collision=collision_batch, goal_was_reached=done_batch, 
                                          step_counter=current_step_batch)
        else:
            done_batch   = 1.0 * env.is_terminal_dist * (reward_batch < env.SOFT_EPS) \
                         + 1.0 * env.is_terminal_angle * (angle_batch < env.ANGLE_EPS) # terminal condition
            done_batch = done_batch // (1.0 * env.is_terminal_dist + 1.0 * env.is_terminal_angle)
            reward_batch = (- np.ones_like(done_batch) * env.abs_time_step_reward) * (1.0 - collision_batch) \
                            + (env.collision_reward) * collision_batch
    else:
        done_batch   = 1.0 * (reward_batch < env.SOFT_EPS) # terminal condition
        reward_batch = - np.ones_like(done_batch) * env.abs_time_step_reward
    
    # Convert to Pytorch
    state_batch         = torch.FloatTensor(state_batch).to(device)
    action_batch        = torch.FloatTensor(action_batch).to(device)
    reward_batch        = torch.FloatTensor(reward_batch).to(device)
    cost_batch        = torch.FloatTensor(cost_batch).to(device)
    next_state_batch    = torch.FloatTensor(next_state_batch).to(device)
    done_batch          = torch.FloatTensor(done_batch).to(device)
    goal_batch          = torch.FloatTensor(goal_batch).to(device)

    return state_batch, action_batch, reward_batch, cost_batch, next_state_batch, done_batch, goal_batch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument("--env",                  default="polamp_env")
    parser.add_argument("--test_env",             default="polamp_env")
    parser.add_argument("--dataset",              default="cross_dataset_simplified") # medium_dataset, hard_dataset, ris_easy_dataset, hard_dataset_simplified
    parser.add_argument("--dataset_curriculum",   default=False) # medium dataset -> hard dataset
    parser.add_argument("--dataset_curriculum_treshold", default=0.95, type=float) # medium dataset -> hard dataset
    parser.add_argument("--uniform_feasible_train_dataset", default=False)
    parser.add_argument("--random_train_dataset",           default=False)
    parser.add_argument("--train_sac",            default=False, type=bool)
    # ris
    parser.add_argument("--epsilon",            default=1e-16, type=float)
    parser.add_argument("--n_critic",           default=2, type=int) # 1
    parser.add_argument("--start_timesteps",    default=1e4, type=int) 
    parser.add_argument("--eval_freq",          default=int(3e4), type=int) # 3e4
    parser.add_argument("--max_timesteps",      default=800000, type=int)
    parser.add_argument("--batch_size",         default=2048, type=int)
    parser.add_argument("--replay_buffer_size", default=5e5, type=int) # 5e5
    parser.add_argument("--n_eval",             default=5, type=int)
    parser.add_argument("--device",             default="cuda")
    parser.add_argument("--seed",               default=42, type=int) # 42
    parser.add_argument("--exp_name",           default="RIS_ant")
    parser.add_argument("--alpha",              default=0.1, type=float)
    parser.add_argument("--Lambda",             default=0.1, type=float) # 0.1
    parser.add_argument("--n_ensemble",         default=20, type=int) # 10
    parser.add_argument("--use_dubins_filter",  default=False, type=bool) # 10
    parser.add_argument("--h_lr",               default=1e-4, type=float)
    parser.add_argument("--q_lr",               default=1e-3, type=float)
    parser.add_argument("--pi_lr",              default=1e-4, type=float)
    parser.add_argument("--clip_v_function",    default=-150, type=float) # -368
    parser.add_argument("--add_obs_noise",           default=False, type=bool)
    parser.add_argument("--curriculum_alpha_val",        default=0, type=float)
    parser.add_argument("--curriculum_alpha_treshold",   default=500000, type=int) # 500000
    parser.add_argument("--curriculum_alpha",        default=False, type=bool)
    parser.add_argument("--curriculum_high_policy",  default=False, type=bool)
    # her
    parser.add_argument("--fraction_goals_are_rollout_goals",  default=0.2, type=float) 
    parser.add_argument("--fraction_resampled_goals_are_env_goals",  default=0.0, type=float) 
    parser.add_argument("--fraction_resampled_goals_are_replay_buffer_goals",  default=0.5, type=float) # 20
    # encoder
    parser.add_argument("--use_decoder",             default=True, type=bool)
    parser.add_argument("--use_encoder",             default=True, type=bool)
    parser.add_argument("--state_dim",               default=40, type=int) # 40
    # safety
    parser.add_argument("--safety_add_to_high_policy", default=False, type=bool)
    parser.add_argument("--safety",                    default=False, type=bool)
    parser.add_argument("--cost_limit",                default=0.5, type=float)
    parser.add_argument("--update_lambda",             default=1000, type=int)
    # logging
    parser.add_argument("--using_wandb",        default=True, type=bool)
    parser.add_argument("--wandb_project",      default="sweep_train_ris_sac_polamp_1", type=str)
    parser.add_argument('--log_loss', dest='log_loss', action='store_true')
    parser.add_argument('--no-log_loss', dest='log_loss', action='store_false')
    parser.set_defaults(log_loss=True)
    args = parser.parse_args()
    
    # agrparse to dictionary params
    parse_parameters_dict = vars(args)
    parameters_dict = {}
    for key_ in parse_parameters_dict:
        parameters_dict[key_] = {"value": parse_parameters_dict[key_]}
    parameters_dict.update({
        # ris
        #'state_dim': {
        #    'values': [20, 40]
        #},
        'alpha': {
            #'values': [1, 0.1, 0.01, 0.001, 0.0001]
            'distribution': 'uniform',
            'min': 0.0001,
            'max': 5
        },
        'Lambda': {
            #'values': [1, 0.1, 0.01, 0.001, 0.0001]
            'distribution': 'uniform',
            'min': 0.001,
            'max': 10
        },
        #'n_ensemble': {
        #    'values': [10, 20]},
    })
    # set sweep config
    sweep_config = {
        'method': 'bayes'
    }
    metric = {
        'name': 'train_rate',
        'goal': 'maximize'   
    }
    sweep_config['metric'] = metric
    sweep_config['parameters'] = parameters_dict

    import pprint
    pprint.pprint(sweep_config) 

    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
    wandb.agent(sweep_id, train, count=60)

