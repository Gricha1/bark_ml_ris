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


def evalPolicy(policy, env, save_subgoal_image=True, render_env=False, plot_obstacles=False, video_task_id=12):
    assert save_subgoal_image != render_env, "only show subgoals video or render env"
    if render_env:
        images = []
        images.append(env.render())    
    if save_subgoal_image:
        images = []
        env_min_x = -5
        env_max_x = 40.
        env_min_y = -5
        env_max_y = 36.
        fig = plt.figure(figsize=[6.4*2, 4.8])
        ax_states = fig.add_subplot(121)
        ax_values = fig.add_subplot(122)
        state_values = []
        subgoal_values = []

    state_distrs = {"x": [], "start_x": [], "y": [], "theta": [], "v": [], "steer": []}
    goal_dists = {"goal_x": [], "goal_y": [], "goal_theta": [], "goal_v": [], "goal_steer": []}
    max_state_vals = {}
    min_state_vals = {}
    max_goal_vals = {}
    min_goal_vals = {}
    mean_actions = {"a": [], "v_s": []}
    final_distances = []
    successes = [] 
    acc_rewards = []
    subgoal_max_dists = []
    episode_lengths = []
    val_key = "map0"
    eval_tasks = len(env.valTasks[val_key])
    for task_id in range(eval_tasks):    
        obs = env.reset(id=task_id, val_key=val_key)
        done = False
        state = obs["observation"]
        goal = obs["desired_goal"]
        t = 0
        acc_reward = 0
        state_distrs["start_x"].append(state[0])

        with torch.no_grad():
            encoded_state = torch.FloatTensor(state).to(policy.device).unsqueeze(0)
            encoded_goal = torch.FloatTensor(goal).to(policy.device).unsqueeze(0)
            subgoal_distribution = policy.subgoal_net(encoded_state, encoded_goal)
            subgoal = subgoal_distribution.rsample()
            sub_x = subgoal.cpu()[0][0]
            sub_y = subgoal.cpu()[0][1]
            dist_to_state = np.sqrt((sub_x - encoded_state.cpu()[0][0]) ** 2 + \
                                    (sub_y - encoded_state.cpu()[0][1]))
            dist_to_goal = np.sqrt((sub_x - encoded_goal.cpu()[0][0]) ** 2 + \
                                    (sub_y - encoded_goal.cpu()[0][1]))
            subgoal_max_dists.append(max(dist_to_state, dist_to_goal))

        while not done:

            # normalize states
            #state = normalize_state(state, env_state_bounds, validate=True)
            #goal = normalize_state(goal, env_state_bounds, validate=True)
            if save_subgoal_image and task_id == video_task_id:
                with torch.no_grad():
                    encoded_state = torch.FloatTensor(state).to(policy.device).unsqueeze(0)
                    encoded_goal = torch.FloatTensor(goal).to(policy.device).unsqueeze(0)
                    subgoal_distribution = policy.subgoal_net(encoded_state, encoded_goal)
                    subgoal = subgoal_distribution.loc
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
                    
                    x_agent = encoded_state.cpu()[0][0]
                    y_agent = encoded_state.cpu()[0][1]
                    theta_agent = encoded_state.cpu()[0][2]
                    x_goal = encoded_goal.cpu()[0][0]
                    y_goal = encoded_goal.cpu()[0][1]
                    theta_goal = encoded_goal.cpu()[0][2]
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
                    for ind, subgoal in enumerate(subgoals):
                        ax_states.scatter([subgoal.cpu()[0][0]], [subgoal.cpu()[0][1]], color="orange", s=50)
                        ax_states.text(subgoal.cpu()[0][0] + 0.05, subgoal.cpu()[0][1] + 0.05, f"{ind + 1}")
                    
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
           
                    ax_values.set_ylim(bottom=env_min_y, top=env_max_y)
                    ax_values.set_xlim(left=env_min_x, right=env_max_x)
                    max_state_value = 1  
                    grid_states = []              
                    grid_goals = []
                    grid_resolution_x = 10
                    grid_resolution_y = 10
                    grid_dx = (env_max_x - env_min_x) / grid_resolution_x
                    grid_dy = (env_max_y - env_min_y) / grid_resolution_y
                    for grid_state_y in np.linspace(env_min_y + grid_dy/2, env_max_y - grid_dy/2, grid_resolution_y):
                        for grid_state_x in np.linspace(env_min_x + grid_dx/2, env_max_x - grid_dx/2, grid_resolution_x):
                            grid_state = [grid_state_x, grid_state_y]
                            grid_state.extend([0 for _ in range(len(state) - 2)])
                            grid_states.append(grid_state)
                    grid_states = torch.FloatTensor(np.array(grid_states)).to(policy.device)
                    assert type(grid_states) == type(encoded_state), f"{type(grid_states)} == {type(encoded_state)}"                
                    grid_goals = torch.FloatTensor([goal for _ in range(grid_resolution_x * grid_resolution_y)]).to(policy.device)
                    assert grid_goals.shape == grid_states.shape
                    grid_vs = policy.value(grid_states, grid_goals)
                    grid_vs = grid_vs.detach().cpu().numpy().reshape(grid_resolution_x, grid_resolution_y)[::-1]                
                    img = ax_values.imshow(grid_vs, extent=[env_min_x,env_max_x, env_min_y,env_max_y])
                    cb = fig.colorbar(img)
                    ax_values.scatter([x_agent], [y_agent], color="green", s=100)
                    ax_values.scatter([np.linspace(x_agent, x_agent + car_length*np.cos(theta_agent), 100)], 
                                      [np.linspace(y_agent, y_agent + car_length*np.sin(theta_agent), 100)], 
                                      color="black", s=5)
                    for ind, subgoal in enumerate(subgoals):
                        ax_values.scatter([subgoal.cpu()[0][0]], [subgoal.cpu()[0][1]], color="orange", s=100)
                        ax_values.text(subgoal.cpu()[0][0] + 0.05, subgoal.cpu()[0][1] + 0.05, f"{ind + 1}")
                    ax_values.scatter([x_goal], [y_goal], color="yellow", s=100)
                    ax_values.scatter([np.linspace(x_goal, x_goal + car_length*np.cos(theta_goal), 100)], 
                                      [np.linspace(y_goal, y_goal + car_length*np.sin(theta_goal), 100)], 
                                      color="black", s=5)

                    fig.canvas.draw()
                    
                    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    images.append(data)
                    cb.remove()
                    ax_states.clear()
                    ax_values.clear()

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
            
            action = policy.select_action(state, goal)
            mean_actions["a"].append(action[0])
            mean_actions["v_s"].append(action[1])

            # produce mean action
            #    converted_state = torch.FloatTensor(state).to(args.device).unsqueeze(0)
            #    converted_goal = torch.FloatTensor(goal).to(args.device).unsqueeze(0)
            #    _, _, action = policy.actor.sample(converted_state, converted_goal)
                #action = action.cpu().data.numpy().flatten()
            next_obs, reward, done, info = env.step(action) 

            acc_reward += reward
            
            next_state = next_obs["observation"]
            state = next_state

            if render_env and task_id == video_task_id:
                images.append(env.render())
            t += 1

        final_distances.append(info["dist_to_goal"])
        successes.append(1.0 * info["geometirc_goal_achieved"])
        episode_lengths.append(info["last_step_num"])
        acc_rewards.append(acc_reward)        
  
    eval_distance = np.mean(final_distances) 
    success_rate = np.mean(successes)
    eval_reward = np.mean(acc_rewards)
    eval_subgoal_dist = np.mean(subgoal_max_dists)
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
    if save_subgoal_image:
        plt.close()
    if save_subgoal_image or render_env:
        images = np.transpose(np.array(images), axes=[0, 3, 1, 2])

    return eval_distance, success_rate, eval_reward, \
           eval_subgoal_dist, [state_distrs, max_state_vals, min_state_vals], \
           [goal_dists, max_goal_vals, min_goal_vals], mean_actions, eval_episode_length, images


def sample_and_preprocess_batch(replay_buffer, batch_size=256, distance_threshold=0.05, device=torch.device("cuda")):
    # Extract 
    batch = replay_buffer.random_batch(batch_size)
    state_batch         = batch["observations"]
    action_batch        = batch["actions"]
    next_state_batch    = batch["next_observations"]
    goal_batch          = batch["resampled_goals"]
    reward_batch        = batch["rewards"]
    done_batch          = batch["terminals"] 

    # Compute sparse rewards: -1 for all actions until the goal is reached
    reward_batch = np.sqrt(np.power(np.array(next_state_batch - goal_batch)[:, :2], 2).sum(-1, keepdims=True)) # distance to goal
    done_batch   = 1.0 * (reward_batch < env.SOFT_EPS) # terminal condition
    reward_batch = - np.ones_like(done_batch) * env.reward_scale

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
    parser.add_argument("--train_static_env",   default=True, type=bool)

    parser.add_argument("--env",                default="polamp_env")
    parser.add_argument("--test_env",           default="polamp_env")
    parser.add_argument("--epsilon",            default=1e-16, type=float)
    parser.add_argument("--distance_threshold", default=0.5, type=float)
    parser.add_argument("--start_timesteps",    default=1e4, type=int) 
    parser.add_argument("--eval_freq",          default=int(2e4), type=int)
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
    parser.add_argument("--wandb_project",      default="train_ris_sac_polamp", type=str)
    parser.add_argument('--log_loss', dest='log_loss', action='store_true')
    parser.add_argument('--no-log_loss', dest='log_loss', action='store_false')
    parser.set_defaults(log_loss=True)
    args = parser.parse_args()
    print(args)

    with open("polamp_env/configs/train_configs.json", 'r') as f:
        train_config = json.load(f)

    with open("polamp_env/configs/environment_configs.json", 'r') as f:
        our_env_config = json.load(f)

    with open("polamp_env/configs/reward_weight_configs.json", 'r') as f:
        reward_config = json.load(f)

    with open("polamp_env/configs/car_configs.json", 'r') as f:
        car_config = json.load(f)

    dataSet = generateDataSet(our_env_config, name_folder="maps", total_maps=1)
    # maps, trainTask, valTasks = dataSet["empty"]
    maps, trainTask, valTasks = dataSet["obstacles"]
    if not args.train_static_env:
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
        "train_static_env": args.train_static_env,
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
    state_dim = args.state_dim 

    folder = "results/{}/RIS/{}/".format(args.env, args.exp_name)
    load_results = os.path.isdir(folder)

    # Create logger
    # TODO: save_git_head_hash = True by default, change it if neccesary
    logger = Logger(vars(args), save_git_head_hash=False)
    if args.using_wandb:
        run = wandb.init(project=args.wandb_project)
    
    # Initialize policy
    # normalize state
    env_state_bounds = {"x": 100, "y": 100, "theta": 3.14,
                        "v": 2.778, "steer": 0.7854}
    image_env = False
    policy = RIS(state_dim=state_dim, action_dim=action_dim, alpha=args.alpha,
                 image_env=image_env,
                 Lambda=args.Lambda, epsilon=args.epsilon,
                 h_lr=args.h_lr, q_lr=args.q_lr, pi_lr=args.pi_lr, 
                 device=args.device, logger=logger if args.log_loss else None, 
                 env_state_bounds=env_state_bounds)

    # Initialize replay buffer and path_builder
    replay_buffer = HERReplayBuffer(
        max_size=args.replay_buffer_size,
        env=env,
        fraction_goals_are_rollout_goals = 0.2,
        fraction_resampled_goals_are_env_goals = 0.0,
        fraction_resampled_goals_are_replay_buffer_goals = 0.5,
        ob_keys_to_save     =["state_achieved_goal", "state_desired_goal"],
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

        path_builder.add_all(
            observations=obs,
            actions=action,
            rewards=reward,
            next_observations=next_obs,
            terminals=[1.0*done]
        )

        state = next_state
        obs = next_obs

        # Train agent after collecting enough data
        if t >= args.batch_size and t >= args.start_timesteps:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch, goal_batch = sample_and_preprocess_batch(
                replay_buffer, 
                batch_size=args.batch_size, 
                distance_threshold=args.distance_threshold, 
                device=args.device
            )
            # Sample subgoal candidates uniformly in the replay buffer
            subgoal_batch = torch.FloatTensor(replay_buffer.random_state_batch(args.batch_size)).to(args.device)

            start_train_batch_time = time.time()
            policy.train(state_batch, action_batch, reward_batch, next_state_batch, done_batch, goal_batch, subgoal_batch)
            train_batch_time = time.time() - start_train_batch_time
            logger.store(train_batch_time = train_batch_time)
            # debug
            print("train", end=" ")

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

        if (t + 1) % args.eval_freq == 0 and t >= args.start_timesteps:
            # Eval policy
            eval_distance, success_rate, eval_reward, \
            eval_subgoal_dist, val_state, val_goal, \
            mean_actions, eval_episode_length, images \
                    = evalPolicy(policy, test_env, plot_obstacles=args.train_static_env)

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
            # debug
            print("eval", end=" ")
        
        # debug
        print()
