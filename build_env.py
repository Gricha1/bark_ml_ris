import json

import gym
from gym.envs.registration import register
from envs.unicycle_env import UnicycleEnv
from envs.simulated_cars_env import SimulatedCarsEnv
from envs.pvtol_env import PvtolEnv
# POLAMP
from envs.goal_polamp_env.polamp_env.lib.utils_operations import generateDataSet
from envs.polamp_env_wrapper import FlatKinematicsWrapper

"""
This file includes a function that simply returns one of the supported environments. 
"""


def build_env(env_name, obs_config='default', rand_init=False):
    """Build our custom gym environment."""

    if env_name == 'Unicycle':
        return UnicycleEnv(obs_config, rand_init=rand_init)
    elif env_name == 'SimulatedCars':
        return SimulatedCarsEnv()
    elif env_name == 'Pvtol':
        return PvtolEnv(obs_config, rand_init=rand_init)
    elif env_name == 'Polamp':
        # TODO
        with open("envs/goal_polamp_env/goal_environment_configs.json", 'r') as f:
            goal_our_env_config = json.load(f)
        with open("envs/goal_polamp_env/polamp_env/configs/train_configs.json", 'r') as f:
            train_config = json.load(f)
        with open("envs/goal_polamp_env/polamp_env/configs/environment_configs.json", 'r') as f:
            our_env_config = json.load(f)
        with open("envs/goal_polamp_env/polamp_env/configs/reward_weight_configs.json", 'r') as f:
            reward_config = json.load(f)
        with open("envs/goal_polamp_env/polamp_env/configs/car_configs.json", 'r') as f:
            car_config = json.load(f)

        dataset_name = "envs/goal_polamp_env/cross_dataset_test_level_1"
        total_maps = 1
        dataSet = generateDataSet(our_env_config, name_folder=dataset_name, total_maps=total_maps, dynamic=False)
        maps, trainTask, valTasks = dataSet["obstacles"]
        goal_our_env_config["dataset"] = dataset_name
        goal_our_env_config["uniform_feasible_train_dataset"] = False
        goal_our_env_config["random_train_dataset"] = False
        if not goal_our_env_config["static_env"]:
            maps["map0"] = []

        environment_config = {
            'vehicle_config': car_config,
            'tasks': trainTask,
            'valTasks': valTasks,
            'maps': maps,
            'our_env_config' : our_env_config,
            'reward_config' : reward_config,
            'evaluation': False,
            'goal_our_env_config' : goal_our_env_config,
        }

        train_env_name = "polamp_env-v0"

        # Set seed
        #np.random.seed(args.seed)
        #random.seed(args.seed)
        #torch.manual_seed(args.seed)

        # register polamp env
        """
        env_dict = gym.envs.registration.registry.env_specs.copy()
        for env in env_dict:
            if train_env_name in env:
                print("Remove {} from registry".format(env))
                del gym.envs.registration.registry.env_specs[env]
        """
        register(
            id=train_env_name,
            entry_point='envs.goal_polamp_env.env:GCPOLAMPEnvironment',
            kwargs={'full_env_name': "polamp_env", "config": environment_config}
        )

        env         = gym.make(train_env_name)
        return FlatKinematicsWrapper(env)
    else:
        raise Exception('Env {} not supported!'.format(env_name))
