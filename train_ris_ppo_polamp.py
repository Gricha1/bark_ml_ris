import numpy as np
import random
import torch
import json
import wandb
import argparse
import gym
from gym.envs.registration import register

from polamp_env.lib.envs import POLAMPEnvironment
from polamp_env.lib.utils_operations import generateDataSet
from custom_algos.lagrangian_ppo.ris_ppo_train import ppo_batch_train, validate
from custom_algos.lagrangian_ppo.ris_ppo import RIS_PPO


with open("polamp_env/configs/environment_configs.json", 'r') as f:
    our_env_config = json.load(f)

with open("polamp_env/configs/reward_weight_configs.json", 'r') as f:
    reward_config = json.load(f)

with open("polamp_env/configs/car_configs.json", 'r') as f:
    car_config = json.load(f)

dataSet = generateDataSet(our_env_config, name_folder="polamp_env/safety", total_maps=1, dynamic=False)

maps, trainTask, valTasks = dataSet["obstacles"]
maps["map0"] = []
environment_config = {
    'vehicle_config': car_config,
    'tasks': trainTask,
    'valTasks': valTasks,
    'maps': maps,
    'our_env_config' : our_env_config,
    'reward_config' : reward_config,
    'evaluation': {},
}

train_env_name = "polamp_env-v0"
test_env_name = train_env_name

# Set seed
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

# register polamp env
register(
    id=train_env_name,
    entry_point='goal_polamp_env.env:GCPOLAMPEnvironment',
    kwargs={'full_env_name': "polamp_env", "config": environment_config}
)

env         = gym.make(train_env_name)
test_env    = gym.make(test_env_name)
#env = POLAMPEnvironment("polamp", environment_config) 

#project_name = "train_ris_ppo_polamp"
#state_dim = env.observation_space.shape[0]
state_dim = env.observation_space["observation"].shape[0]
goal_dim = env.observation_space["achieved_goal"].shape[0]
action_dim = env.action_space.shape[0] 

parser = argparse.ArgumentParser()
parser.add_argument("--env",                        default="polamp_env")
parser.add_argument("--replay_buffer_high_policy_size", default=int(1e6), type=int)
parser.add_argument("--high_policy_batch_size",     default=2048, type=int)
parser.add_argument("--high_policy_start_timesteps",default=int(1e4), type=int) 
parser.add_argument("--batch_size",                 default=2048, type=int)
parser.add_argument("--max_episodes",               default=15000000, type=int)
parser.add_argument("--max_time_steps",             default=15000000, type=int)
parser.add_argument("--max_episode_timesteps",      default=250, type=int)
parser.add_argument("--val_time_steps",             default=1000000, type=int)
parser.add_argument("--log_interval",               default=5, type=int)
parser.add_argument("--cost_limit_truncated",       default=0, type=bool)
parser.add_argument("--name_save",                  default="LPPOExp")
parser.add_argument("--name_val",                   default="LPPOExp")
parser.add_argument("--curriculum",                 default=0, type=bool)
parser.add_argument("--curriculum_name",            default="LPPOExp")
parser.add_argument("--hidden_size",                default=256, type=int)
parser.add_argument("--gamma",                      default=0.99, type=float)
parser.add_argument("--eps_clip",                   default=0.2, type=float)
parser.add_argument("--lr",                         default=1e-4, type=float)
parser.add_argument("--K_epochs",                   default=10, type=int)
parser.add_argument("--constrained_ppo",            default=0, type=bool)
parser.add_argument("--max_available_collision",    default=20, type=float)
parser.add_argument("--cost_limit",                 default=0.5, type=float)
parser.add_argument("--penalty_init",               default=0.1, type=float)
parser.add_argument("--penalty_lr",                 default=5e-4, type=float)
parser.add_argument("--max_penalty",                default=1.0, type=float)
parser.add_argument("--penalty_clip",               default=0.6, type=float)
parser.add_argument("--adaptive_std",                default=1, type=bool)
parser.add_argument("--using_wandb",                default=1, type=bool)
parser.add_argument("--training",                   default=1, type=bool)
parser.add_argument("--save_image",                 default=0, type=bool)
parser.add_argument("--save_subgoals_image",        default=1, type=bool)
parser.add_argument("--save_subgoal_first_image",   default=0, type=bool)
parser.add_argument("--eval_freq",    default=2e5, type=float)
args = parser.parse_args()
print(f"args: {args}")

agent = RIS_PPO(state_dim, goal_dim, action_dim, args, env.action_space.high, save_path=args.name_save, high_policy_batch_size=args.high_policy_batch_size)
if args.training:
    if args.using_wandb:
        project_name = "train_ris_ppo_polamp"
        wandb.init(config=args, project=project_name)
        config = wandb.config
    else:
        wandb = None
    if args.curriculum:
        print(f"train_config['name_save']: {args.name_save}")
        agent.load("{}{}".format("./", args.curriculum_name))

    ppo_batch_train(env, test_env, agent, args, wandb=wandb)
else:
    if args.using_wandb:
        project_name = "validate_ris_ppo_polamp"
        run = wandb.init(config=environment_config, project=project_name)
    else:
        run = None
    name_val = args.name_val
    agent.load("{}{}".format("./", name_val))
    collision_tasks = 0
    successed_tasks = 0
    total_tasks = 0
    constrained_cost = []
    lst_min_beam = []
    task_collision = {  "map0": [7, 19],
                        "map1": [18],
                        "map2": [3],
                        "map3": [0],
                        "map4": [0],
                        "map6": [14],
                        "map7": [4, 16],
                        "map8": [10],
                        "map9": [12],
                        }
    faled_tasks = []

    #val_keys = env.valTasks
    val_keys = ["map0"]

    for val_key in val_keys:
        val_task_ids = list(range(len(env.valTasks[val_key])))
        #val_task_ids = [7, 9]
        eval_tasks = len(val_task_ids)
        total_tasks += eval_tasks
        
        print(f"val_key: {val_key}")
        print(f"eval_tasks: {eval_tasks}")
        # for id in range(eval_tasks):
        if args.save_image or args.save_subgoals_image:
            assert args.save_image != args.save_subgoals_image, "only 1 type of image"
            print("ok")
            # if val_key!="map11":
            #     continue
            # if not val_key in task_collision:
            #     continue
            # for id in task_collision[val_key]:
            for id in val_task_ids:
                print(id)
                #if (id % 5) != 0:
                #    continue
            # for id in range(5, 6):
                # obs = env.reset(id=id, val_key=val_key)
                images, isDone, info, episode_cost, min_beam = validate(env, agent, env._max_episode_steps, save_image=args.save_image, id=id, val_key=val_key, run=run, save_subgoal_image=args.save_subgoals_image, save_subgoal_first_image=args.save_subgoal_first_image)
                if isDone:
                    #total_distance += min_distance
                    #counter_done += 1
                    if info["geometirc_goal_achieved"]:
                        successed_tasks += 1
                    else:
                        faled_tasks.append((val_key, id))
                    """
                    if "Collision" in info:
                        # collision = True
                        # isDone = False
                        print("$$ Collision $$")
                        print(f"val_key: {val_key}")
                        print(f"id: {id}")
                        collision_tasks += 1
                    elif "SoftEps" in info:
                        print("$$ SoftEps $$")
                    else:
                        successed_tasks += 1
                    """

                constrained_cost.append(episode_cost)
                lst_min_beam.append(min_beam)
                if args.save_image or args.save_subgoals_image:
                    #wandb.init(config=environment_config, project="validation_custom_ppo")

                    #print("type:", type(images[0]))
                    #print("shape:", images[0].shape)
                    #assert 1 == 0
                    
                    run.log({f"random_task": wandb.Video(images, fps=10, format="gif")})
        else:
            for id in range(eval_tasks):
                # obs = env.reset(id=id, val_key=val_key)
                images, isDone, info, episode_cost, min_beam = validate(env, agent, env._max_episode_steps, save_image=args.save_image, id=id, val_key=val_key)
                print(f"episode_cost: {episode_cost}")
                constrained_cost.append(episode_cost)
                lst_min_beam.append(min_beam)
                if isDone:
                    if "Collision" in info:
                        # collision = True
                        # isDone = False
                        print("$$ Collision $$")
                        print(f"val_key: {val_key}")
                        print(f"id: {id}")
                        collision_tasks += 1
                    elif "SoftEps" in info:
                        print("$$ SoftEps $$")
                    else:
                        successed_tasks += 1

    #success_rate = successed_tasks / total_tasks * 100
    #collision_rate = collision_tasks / total_tasks * 100
    print(f"successed_tasks: {successed_tasks}")
    print(f"total_tasks: {total_tasks}")
    print(f"failed tasks:", faled_tasks)
    #print(f"collision_rate: {collision_rate}")
    #print(f"mean constrained_cost: {np.mean(constrained_cost)}")
    #print(f"max constrained_cost: {np.max(constrained_cost)}")
    #print(f"min constrained_cost: {np.min(constrained_cost)}")
    #print(f"mean lst_min_beam: {np.mean(lst_min_beam)}")
    #print(f"max lst_min_beam: {np.max(lst_min_beam)}")
    #print(f"min lst_min_beam: {np.min(lst_min_beam)}")
