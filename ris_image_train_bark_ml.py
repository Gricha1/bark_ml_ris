import os

import torch
import numpy as np
import random
import argparse

import gym
from gym.envs.registration import register
#from multiworld.envs.mujoco import register_custom_envs as register_mujoco_envs

import time

from utils.logger import Logger
from custom_second_RIS import RIS
from HER import HERReplayBuffer, PathBuilder
import wandb


'''
# set path for experement video
working_dir_name = str(pathlib.Path().resolve())
observation_dir_name = working_dir_name + f"/video_train/obs_pngs"
if os.path.isdir(observation_dir_name):
    shutil.rmtree(observation_dir_name)
path = Path(observation_dir_name)
path.mkdir(parents=True, exist_ok=True)
'''

def evalPolicy(policy, env, N=100, Tmax=100, distance_threshold=0.05, logger=None):
    final_distances = []
    successes = [] 
    acc_rewards = []
    subgoal_max_dists = []

    for _ in range(N):
        obs = env.reset()
        done = False
        #changed
        if image_env:
            state = obs["image_observation"]
            goal = obs["image_desired_goal"]
        else:
            state = obs["observation"]
            goal = obs["desired_goal"]
        t = 0
        acc_reward = 0

        # debug
        if not image_env:
            with torch.no_grad():
                encoded_state = torch.FloatTensor(state).to(args.device).unsqueeze(0)
                encoded_goal = torch.FloatTensor(goal).to(args.device).unsqueeze(0)
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
            #changed
            action = policy.select_action(state, goal)
            #    converted_state = torch.FloatTensor(state).to(args.device).unsqueeze(0)
            #    converted_goal = torch.FloatTensor(goal).to(args.device).unsqueeze(0)
            #    _, _, action = policy.actor.sample(converted_state, converted_goal)
                #action = action.cpu().data.numpy().flatten()
            next_obs, reward, done, info = env.step(action) 

            acc_reward += reward
            
            if image_env:
                next_state = next_obs["image_observation"]
            else:
                next_state = next_obs["observation"]
            state = next_state

            done = done or t >= Tmax

            t += 1

        final_distances.append(info["dist_to_goal"])
        successes.append(1.0 * info["geometirc_goal_achieved"])
        acc_rewards.append(acc_reward)        

        
    # debug acc rewards
    print("eval final distances:", final_distances)
    eval_distance, success_rate = np.mean(final_distances), np.mean(successes)
    eval_reward = np.mean(acc_rewards)
    eval_subgoal_dist = np.mean(subgoal_max_dists)

    if logger is not None:
        logger.store(eval_distance=eval_distance, success_rate=success_rate, 
                     eval_reward=eval_reward, eval_subgoal_dist=eval_subgoal_dist)

    return eval_distance, success_rate, eval_reward, eval_subgoal_dist


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
    # debug
    #reward_batch = np.ones_like(done_batch) * (-1)
    reward_batch = np.ones_like(done_batch) * (-0.1)

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

    #register_mujoco_envs()
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
    vectorized = True
    action_dim = env.action_space.shape[0]
    state_dim = args.state_dim 

    folder = "results/{}/RIS/{}/".format(args.env, args.exp_name)
    load_results = os.path.isdir(folder)

    # Create logger
    # TODO: save_git_head_hash = True by default, change it if neccesary
    logger = Logger(vars(args), save_git_head_hash=False)
    run = wandb.init(project='RIS_bark_ml_train')
    
    # Initialize policy
    image_env = False
    
    policy = RIS(state_dim=state_dim, action_dim=action_dim, 
                 image_env=image_env,alpha=args.alpha, 
                 Lambda=args.Lambda, target_update_interval=1, 
                 h_lr=1e-4, q_lr=args.q_lr, pi_lr=1e-3, 
                 device=args.device, logger=logger if args.log_loss else None)
    
    '''
    policy = RIS(state_dim=state_dim, action_dim=action_dim, 
                 image_env=image_env,alpha=args.alpha, 
                 Lambda=args.Lambda,  gamma=0.95, tau=0.05, target_update_interval=1, 
                 h_lr=1e-4, q_lr=args.q_lr, pi_lr=1e-3, 
                 device=args.device, logger=logger if args.log_loss else None)
    #policy = RIS(state_dim=state_dim, action_dim=action_dim, alpha=args.alpha, Lambda=args.Lambda, gamma=0.95, tau=0.05, target_update_interval=1, h_lr=1e-4, q_lr=args.q_lr, pi_lr=1e-3, device=args.device, logger=logger if args.log_loss else None)
    '''
    if load_results:
        policy.load(folder)

    # Initialize replay buffer and path_builder
    if image_env:
        
        replay_buffer = HERReplayBuffer(
            max_size=args.replay_buffer_size,
            env=env,
            fraction_resampled_goals_are_replay_buffer_goals = args.replay_buffer_goals, 
            #ob_keys_to_save     =["state_achieved_goal", "state_desired_goal"],
            #desired_goal_keys   =["image_desired_goal", "state_desired_goal"],
            observation_key     = 'image_observation',
            desired_goal_key    = 'image_desired_goal',
            achieved_goal_key   = 'image_achieved_goal',
            vectorized          = True
        )
        
    else:
        replay_buffer = HERReplayBuffer(
            max_size=args.replay_buffer_size,
            env=env,
            fraction_goals_are_rollout_goals = 0.2,
            fraction_resampled_goals_are_env_goals = 0.0,
            fraction_resampled_goals_are_replay_buffer_goals = 0.5,
            #ob_keys_to_save     =["state_achieved_goal", "state_desired_goal"],
            #desired_goal_keys   =["desired_goal", "state_desired_goal"],
            observation_key     = 'observation',
            desired_goal_key    = 'desired_goal',
            achieved_goal_key   = 'achieved_goal',
            vectorized          = vectorized 
        )
    path_builder = PathBuilder()

    # Initialize environment
    obs = env.reset()
    done = False
    #changed
    if image_env:
        state = obs["image_observation"]
        goal = obs["image_desired_goal"]
    else:
        state = obs["observation"]
        goal = obs["desired_goal"]
    episode_timesteps = 0
    episode_num = 0 
    old_success_rate = None
    save_policy_count = 0 

    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1

        # debug
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

        #changed
        if image_env:
            next_state = next_obs["image_observation"]
        else:
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

            
        if done or episode_timesteps >= args.max_episode_length: 
            # Add path to replay buffer and reset path builder
            replay_buffer.add_path(path_builder.get_all_stacked())
            path_builder = PathBuilder()
            logger.store(t=t, reward=reward)		

            # Reset environment
            obs = env.reset()
            done = False
            #changed
            if image_env:
                state = obs["image_observation"]
                goal = obs["image_desired_goal"]
            else:
                state = obs["observation"]
                goal = obs["desired_goal"]
            episode_timesteps = 0
            episode_num += 1 

        if (t + 1) % args.eval_freq == 0 and t >= args.start_timesteps:
            # Eval policy
            eval_distance, success_rate, eval_reward, eval_subgoal_dist = evalPolicy(
                policy, test_env, 
                N=args.n_eval,
                Tmax=args.max_episode_length, 
                distance_threshold=args.distance_threshold,
                logger = logger
            )
            print("RIS t={} | {}".format(t+1, logger))
            run.log({
                     'train_adv': sum(logger.data["adv"][-args.eval_freq:]) / args.eval_freq, 
                     'critic_value': sum(logger.data["critic_value"][-args.eval_freq:]) / args.eval_freq,  
                     'target_value': sum(logger.data["target_value"][-args.eval_freq:]) / args.eval_freq,  
                     'train_D_KL': sum(logger.data["D_KL"][-args.eval_freq:]) / args.eval_freq,
                     'entropy_1': sum(logger.data["entropy_1"][-args.eval_freq:]) / args.eval_freq,
                     'entropy_2': sum(logger.data["entropy_2"][-args.eval_freq:]) / args.eval_freq,
                     'steps': logger.data["t"][-1],
                     'subgoal_loss': sum(logger.data["subgoal_loss"][-args.eval_freq:]) / args.eval_freq,
                     'train_critic_loss': sum(logger.data["critic_loss"][-args.eval_freq:]) / args.eval_freq,
                     'actor_loss': sum(logger.data["actor_loss"][-args.eval_freq:]) / args.eval_freq,
                     f'val_distance({args.n_eval} episodes)': eval_distance,
                     f'eval_reward({args.n_eval} episodes)': eval_reward,
                     f'val_rate({args.n_eval} episodes)': success_rate,
                     f'val_subgoal_dist({args.n_eval} episodes)': eval_subgoal_dist,
                     'critic_grad_1': sum(logger.data["critic_grad_1"][-args.eval_freq:]) / args.eval_freq,
                     'critic_grad_2': sum(logger.data["critic_grad_2"][-args.eval_freq:]) / args.eval_freq,
                     'actor_grad': sum(logger.data["actor_grad"][-args.eval_freq:]) / args.eval_freq,
                     'subgoal_grad': sum(logger.data["subgoal_grad"][-args.eval_freq:]) / args.eval_freq
                    })
            

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

            # debug
            print("eval", end=" ")

            # debug
            #if (success_rate >= 0.9):
            #    print("finish training", f"succes rate:{success_rate}")
            #    break
        
        # debug
        print()
