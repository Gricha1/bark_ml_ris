# from EnvLib.ObstGeomEnv import *
from time import sleep
import gym
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import wandb
from polamp_HER import HERReplayBuffer, PathBuilder

class HighPolicyMemory:
    def __init__(self, args, env):
        self.max_size = args.replay_buffer_high_policy_size
        self.current_size = 0
        self.batch_size = args.high_policy_batch_size
        self.obs = torch.zeros((self.max_size, env.observation_space["observation"].shape[0]), dtype=torch.float32)
        self.goal = torch.zeros((self.max_size, env.observation_space["desired_goal"].shape[0]), dtype=torch.float32)

    def add(self, obs, goal):
        self.obs[self.current_size % self.max_size] = obs
        self.goal[self.current_size % self.max_size] = goal
        self.current_size += 1
    
    def _sample_indices(self):
        return np.random.choice(list(range(min(self.current_size, self.max_size))), self.batch_size)

    def sample_batch(self):
        batch_indexes = self._sample_indices()
        return self.obs[batch_indexes], self.goal[batch_indexes]

    def random_state_batch(self):
        batch_indexes = self._sample_indices()
        return self.obs[batch_indexes]


class Memory:
    def __init__(self, args, env, device):
        #self.obs = torch.zeros((args.batch_size, env.observation_space.shape[0]), dtype=torch.float32).to(device)
        self.obs = torch.zeros((args.batch_size, env.observation_space["observation"].shape[0]), dtype=torch.float32).to(device)
        self.goal = torch.zeros((args.batch_size, env.observation_space["desired_goal"].shape[0]), dtype=torch.float32).to(device)
        self.actions = torch.zeros((args.batch_size, env.action_space.shape[0]), dtype=torch.float32).to(device)
        self.logprobs = torch.zeros(args.batch_size, dtype=torch.float32).to(device)
        self.rewards = torch.zeros(args.batch_size, dtype=torch.float32).to(device)
        self.constrained_costs = torch.zeros(args.batch_size, dtype=torch.float32).to(device)
        self.dones = torch.zeros(args.batch_size, dtype=torch.bool).to(device)
        self.num_steps = args.batch_size
        self.device = device
    
    def rewards_monte_carlo(self, gamma):
        torch_rewards = torch.zeros(self.num_steps, dtype=torch.float32).to(self.device)
        torch__constrained_cost = torch.zeros(self.num_steps, dtype=torch.float32).to(self.device)
        discounted_reward = 0
        discounted_constrained_cost = 0
        for t in reversed(range(self.num_steps)):
            is_terminal = self.dones[t]
            reward = self.rewards[t]
            constrained_cost = self.constrained_costs[t]
            if is_terminal:
                discounted_reward = 0
                discounted_constrained_cost = 0
            discounted_reward = reward + gamma * discounted_reward
            discounted_constrained_cost = constrained_cost + gamma * discounted_constrained_cost
            
            torch_rewards[t] = discounted_reward
            torch__constrained_cost[t] = discounted_constrained_cost

        self.rewards = torch_rewards
        self.constrained_costs = torch__constrained_cost

    def shuffle(self):
        b_inds = np.arange(self.num_steps)
        np.random.shuffle(b_inds)
        self.obs = self.obs[b_inds]
        self.goal = self.goal[b_inds]
        self.actions = self.actions[b_inds]
        self.logprobs = self.logprobs[b_inds]
        self.rewards = self.rewards[b_inds]
        self.constrained_costs = self.constrained_costs[b_inds]
        self.dones = self.dones[b_inds]


def validate(env, agent, max_steps, save_image=False, id=None, val_key=None, run=None, save_subgoal_image=False, save_subgoal_first_image=False):
    if id is None or val_key is None:
        return
    agent.eval()
    state = env.reset(id=id, val_key=val_key)
    goal = state["desired_goal"]
    state = state["observation"]
    if save_image or save_subgoal_image:
        images = []
    if save_image:
        images.append(env.render())
    if save_subgoal_image:
        fig = plt.figure()
        fig.add_subplot(111)
    isDone = False
    t = 0
    sum_reward = 0
    episode_constrained = []
    episode_min_beam = []

    # debug subgoals
    if save_subgoal_first_image:
        with torch.no_grad():
            encoded_state = torch.FloatTensor(state).to(agent.device).unsqueeze(0)
            encoded_goal = torch.FloatTensor(goal).to(agent.device).unsqueeze(0)
            subgoal_distribution = agent.subgoal_net(encoded_state, encoded_goal)
            #subgoal = subgoal_distribution.rsample()
            subgoal = subgoal_distribution.loc
            #fig = plt.figure()
            #fig.add_subplot(111)
            x_f_state_to_draw = encoded_state.cpu()
            x_agent = x_f_state_to_draw[0][0]
            y_agent = x_f_state_to_draw[0][1]
            theta_agent = x_f_state_to_draw[0][2]
            x_f_goal_to_draw = encoded_goal.cpu()
            x_goal = x_f_goal_to_draw[0][0]
            y_goal = x_f_goal_to_draw[0][1]
            theta_goal = x_f_goal_to_draw[0][2]
            car_length = 0.5
            plt.plot([x_agent, x_agent + np.cos(theta_agent) * car_length], 
                    [y_agent, y_agent + np.sin(theta_agent) * car_length], color="green", linewidth=3)
            plt.scatter([x_agent], [y_agent], color="green", s=100)
            plt.text(x_agent + 0.05, y_agent + 0.05, "agent")

            plt.plot([x_goal, x_goal + np.cos(theta_goal) * car_length], 
                    [y_goal, y_goal + np.sin(theta_goal) * car_length], color="yellow", linewidth=3)
            plt.scatter([x_goal], [y_goal], color="yellow", s=100)
            plt.text(x_goal + 0.05, y_goal + 0.05, "goal")

            plt.scatter([subgoal.cpu()[0][0]], [subgoal.cpu()[0][1]], color="orange", s=100)
            plt.text(subgoal.cpu()[0][0] + 0.05, subgoal.cpu()[0][1] + 0.05, "subgoal")

            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = wandb.Image(data)
            run.log({f"agent_state": img})

    while not isDone and t < max_steps:

        # debug subgoals
        if save_subgoal_image:
            with torch.no_grad():
                encoded_state = torch.FloatTensor(state).to(agent.device).unsqueeze(0)
                encoded_goal = torch.FloatTensor(goal).to(agent.device).unsqueeze(0)
                subgoal_distribution = agent.subgoal_net(encoded_state, encoded_goal)
                #subgoal = subgoal_distribution.rsample()
                subgoal = subgoal_distribution.loc
                x_f_state_to_draw = encoded_state.cpu()
                x_agent = x_f_state_to_draw[0][0]
                y_agent = x_f_state_to_draw[0][1]
                theta_agent = x_f_state_to_draw[0][2]
                x_f_goal_to_draw = encoded_goal.cpu()
                x_goal = x_f_goal_to_draw[0][0]
                y_goal = x_f_goal_to_draw[0][1]
                theta_goal = x_f_goal_to_draw[0][2]
                car_length = 0.5
                plt.plot([x_agent, x_agent + np.cos(theta_agent) * car_length], 
                        [y_agent, y_agent + np.sin(theta_agent) * car_length], color="green", linewidth=3)
                plt.scatter([x_agent], [y_agent], color="green", s=100)
                plt.text(x_agent + 0.05, y_agent + 0.05, "agent")

                plt.plot([x_goal, x_goal + np.cos(theta_goal) * car_length], 
                        [y_goal, y_goal + np.sin(theta_goal) * car_length], color="yellow", linewidth=3)
                plt.scatter([x_goal], [y_goal], color="yellow", s=100)
                plt.text(x_goal + 0.05, y_goal + 0.05, "goal")

                plt.scatter([subgoal.cpu()[0][0]], [subgoal.cpu()[0][1]], color="orange", s=100)
                plt.text(subgoal.cpu()[0][0] + 0.05, subgoal.cpu()[0][1] + 0.05, "subgoal")

                fig.canvas.draw()
                data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                images.append(data)
                fig.clear()


        #action = agent.act(state, False)
        # start_time = time.time()
        action = agent.get_action(state, goal, deterministic=True)
        # goal_time = time.time()
        # print(f"------ network time ----: {(goal_time - start_time) * 1e3}")
        # print("action: ", action)
        # start_time = time.time()
        state, reward, isDone, info = env.step(action)
        goal = state["desired_goal"]
        state = state["observation"]
        # goal_time = time.time()
        # print(f"------ env time ----: {(goal_time - start_time) * 1e3}")
        # print("info: ", info)
        sum_reward += reward
        episode_constrained.append(info.get('cost', 0))
        episode_min_beam.append(env.environment.min_beam)
        if save_image:
            images.append(env.render())
        t += 1
        
    env.close()
    if save_subgoal_image:
        plt.close()
    if save_image or save_subgoal_image:
        images = np.transpose(np.array(images), axes=[0, 3, 1, 2])
    return sum_reward if not save_image and not save_subgoal_image else images, isDone, info, np.mean(episode_constrained), np.min(episode_min_beam) 

    
def ppo_batch_train(env, test_env, agent, args, wandb=None, saveImage=True):
    update_timestep = args.batch_size
    log_interval = args.log_interval
    max_episodes = args.max_episodes
    max_timesteps = args.max_time_steps
    max_episode_timesteps = args.max_episode_timesteps
    name_save = args.name_save
    cost_limit = args.cost_limit
    cost_limit_truncated = args.cost_limit_truncated
    max_available_collision = args.max_available_collision
    running_reward = 0
    constraint_cost = 0
    timestep = 0
    counter_done = 0
    counter_success_done = 0
    counter_collision = 0
    total_distance = 0
    constraint_violation = 0
    
    #high_policy_memory = HighPolicyMemory(args, env)
    
    vectorized = True
    high_policy_memory = HERReplayBuffer(
        max_size=args.replay_buffer_high_policy_size,
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
    

    memory = Memory(args, env, agent.device)

    batch_time = 0
    # цикл обучения
    episodes = 0
    penalizing_ppo = True
    for i_episode in range(1, max_episodes + 1):
        if timestep >= max_timesteps:
            break  
        lst_states = []
        start_time = time.time()
        observation = env.reset()
        full_obs = observation # to HER
        goal = observation["desired_goal"]
        observation = observation["observation"]
        goal_time = time.time()
        batch_time += goal_time - start_time
        # env.render(save_image=False)
        min_distance = float('inf')
        # print("i: ", i_episode)
        episodes += 1
        for t in range(max_episode_timesteps):

            #action, log_prob = agent.policy_old.act(observation)
            action, log_prob = agent.policy_old.act(np.concatenate([observation, goal]))

            # lst_states.append(state)
            #high_policy_memory.add(torch.Tensor(observation), torch.Tensor(goal))

            index = timestep % update_timestep
            memory.obs[index] = torch.Tensor(observation).to("cuda")
            memory.goal[index] = torch.Tensor(goal).to("cuda")
            memory.actions[index] = action
            memory.logprobs[index] = log_prob
            action_step = action.cpu().numpy()
            start_time = time.time()

            observation, reward, done, info = env.step(action_step)
            full_next_obs = observation # to HER
            goal = observation["desired_goal"]
            observation = observation["observation"]
            goal_time = time.time()
            batch_time += goal_time - start_time

            
            path_builder.add_all(
                observations=full_obs,
                actions=action_step,
                rewards=reward,
                next_observations=full_next_obs,
                terminals=[1.0*done]
            )
            
            full_obs = full_next_obs
            
            memory.rewards[index] = reward
            memory.dones[index] = done
            # adding cost
            cost = info.get('cost', 0)
            memory.constrained_costs[index] = cost
            
            if "EuclideanDistance" in info:
                if min_distance >= info["EuclideanDistance"]:
                    min_distance = info["EuclideanDistance"]
            
            running_reward += reward
            constraint_cost += cost

            timestep += 1

            # train high level policy
            if timestep >= args.high_policy_start_timesteps and (timestep % update_timestep) % 100 == 0:
                print("update high level")
                stast_high_policy = agent.update_high_level_policy(high_policy_memory)

            # выполняем обновление
            if timestep % update_timestep == 0:
                print(f"------ updating {args.name_save}------")

                # train high level policy
                #if timestep >= args.high_policy_start_timesteps:
                #    stast_high_policy = agent.update_high_level_policy(high_policy_memory)
                
                # train low level policy
                if timestep >= args.high_policy_start_timesteps:
                    stast = agent.update_low_level_policy(memory, penalizing_ppo)

                if not wandb is None and timestep >= args.high_policy_start_timesteps:  
                    try:
                        stast_high_policy # is defined                 
                        wandb_log = { 
                                    # high level policy
                                    'H_train_adv': stast_high_policy["adv"],
                                    'H_subgoal_loss': stast_high_policy["subgoal_loss"],
                                    'H_sampled_subgoal_v': stast_high_policy["high_policy_v"],
                                    'H_target_subgoal_v': stast_high_policy["high_v"],

                                    'H_subgoal_x_max': stast_high_policy["H_subgoal_x_max"],
                                    'H_subgoal_x_mean': stast_high_policy["H_subgoal_x_mean"],
                                    'H_subgoal_x_min': stast_high_policy["H_subgoal_x_min"],

                                    'H_sampled_subgoal_x_max': stast_high_policy["H_sampled_subgoal_x_max"],
                                    'H_sampled_subgoal_x_mean': stast_high_policy["H_sampled_subgoal_x_mean"],
                                    'H_sampled_subgoal_x_min': stast_high_policy["H_sampled_subgoal_x_min"],
                                    }
                    except:
                        wandb_log = {}

                    for key in stast:
                        assert not(key in wandb_log), ""
                        wandb_log[key] = stast[key]

                    wandb.log(wandb_log, step = timestep)

                elif wandb is None:
                    print(f"dist_entropy: {stast['dist_entropy']}")
                    print(f"loss_penalty: {stast['penalty_loss']}")
                    print(f"cost_mean: {stast['constrained_costs']}")
                    print(f"lyambda: {stast['lagrange_multiplier']}")
                    print(f"action_means_linear_acc: {stast['action_means'][0]}")
                    print(f"action_mean_steering_vel: {stast['action_means'][1]}")
                    print(f"std_linear_acc: {np.exp(stast['logstd'][0])}")
                    print(f"std_steering_vel: {np.exp(stast['logstd'][1])}")
                    print(f"total_loss: {stast['total_loss']}")
                    print(f"policy_loss: {stast['policy_loss']}")
                    print(f"mse_loss: {stast['mse_loss']}")
                    print(f"penalty_surr_loss: {stast['penalty_surr_loss']}")
                    print(f"fps: {update_timestep / batch_time}")
                    print(f"batch_time: {batch_time}")
                    print(f"timestep: {timestep}")
                batch_time = 0
                agent.save(os.path.join('./' + name_save))

                # логирование
                if timestep % (update_timestep * log_interval) == 0:
                    running_reward /= episodes
                    constraint_cost /= episodes
                    counter_done /= episodes
                    counter_success_done /= episodes
                    counter_done *= 100
                    counter_success_done *= 100
                    total_distance /= episodes
                    counter_collision /= episodes
                    counter_collision *= 100
                    constraint_violation /= episodes
                    constraint_violation *= 100
                    if counter_collision >= max_available_collision:
                        penalizing_ppo = True
                    else:
                        penalizing_ppo = False

                    if not wandb is None:
                        wandb.log({'running_reward': running_reward,
                                   'constraint_cost': constraint_cost,
                                   #'success_rate': counter_done,
                                   'success_rate': counter_success_done,
                                   'min_distance' : total_distance,
                                   'collision_rate': counter_collision,
                                   'constraint_violation_rate': constraint_violation,
                                   'penalizing_ppo': float(penalizing_ppo)}
                                   ,step = timestep)
                    else:
                        print("------- wandb ------")
                        print(f"running_reward: {running_reward}")
                        print(f"constraint_cost: {constraint_cost}")
                        print(f"counter_done: {counter_done}")
                        print(f"total_distance: {total_distance}")
                        print(f"counter_collision: {counter_collision}")
                        print(f"constraint_violation: {constraint_violation}")
                        print(f"timestep: {timestep}")
                        print(f"episodes: {episodes}")
                    running_reward = 0
                    constraint_cost = 0
                    counter_done = 0
                    counter_success_done = 0
                    total_distance = 0
                    counter_collision = 0
                    constraint_violation = 0
                    episodes = 1
            
            """
            if "SoftEps" in info or "Collision" in info:
                total_distance += min_distance
                if "Collision" in info:
                    counter_collision += 1
                break
            """

            if done:
                
                high_policy_memory.add_path(path_builder.get_all_stacked())
                path_builder = PathBuilder()
                

                total_distance += min_distance
                counter_done += 1
                if info["geometirc_goal_achieved"]:
                    counter_success_done += 1
                break
            
            """
            if cost_limit_truncated:
                if cost >= cost_limit:
                    total_distance += min_distance
                    constraint_violation += 1
                    break
            """

            if timestep % args.eval_freq == 0:
                print(f"------ validating------")
                # get random validation task
                val_key = "map0"
                run = wandb
                id = np.random.choice(list(range(len(test_env.valTasks[val_key]))))
                images, isDone, info, episode_cost, min_beam = validate(test_env, agent, test_env._max_episode_steps, save_image=args.save_image, id=id, val_key=val_key, run=run, save_subgoal_image=args.save_subgoals_image, save_subgoal_first_image=args.save_subgoal_first_image)
                run.log({f"validation_video": wandb.Video(images, fps=10, format="gif")})
