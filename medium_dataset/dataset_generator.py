import math
import gym
import numpy as np
from lib.envs import POLAMPEnvironment
from lib.utils_operations import generateDataSet
from custom_ppo.train import ppo_batch_train, validate
from custom_ppo.ppo import PPO
from time import sleep
import json
import matplotlib.pyplot as plt
import wandb

# with open("configs/train_configs.json", 'r') as f:
#     train_config = json.load(f)

with open("configs/environment_configs.json", 'r') as f:
    our_env_config = json.load(f)
    # print(our_env_config)

with open("configs/reward_weight_configs.json", 'r') as f:
    reward_config = json.load(f)

with open("configs/car_configs.json", 'r') as f:
    car_config = json.load(f)

dataSet = generateDataSet(our_env_config)

maps, trainTask, valTasks = dataSet["obstacles"]
environment_config = {
        'vehicle_config': car_config,
        'tasks': trainTask,
        'valTasks': valTasks,
        'maps': maps,
        'our_env_config' : our_env_config,
        'reward_config' : reward_config,
        'evaluation': {},
    }

model_config = {
    "critic": [256, 256],
    "actor": [256, 256],
    "gamma": 0.99,
    "eps_clip": 0.2,
    "K_epochs": 10,
    "lr": 1e-4,
    "betas": [0.9, 0.999],
    "constrained_ppo": 1,
    "cost_limit": 0.2,
    "penalty_init": 1.0,
    "penalty_lr": 5e-4,
    "max_penalty": 1.0,
    "penalty_clip": 0.6,
}

env = POLAMPEnvironment("polamp", environment_config) 

obstacles = [
            [50, 15, 0 * math.pi/2, 5, 30], 
            [25, 35, 0 * math.pi/2, 5, 15],
            [75, 35, 0 * math.pi/2, 5, 15],
            [55, 55, 0 * math.pi/2, 5, 25],
            ]
for obst in obstacles:
    env.environment.getBB(obst, width=2.0, length=3.8, ego=True)
    env.drawObstacles()
    plt.show()