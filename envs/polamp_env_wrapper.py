import gym
from gym.spaces import *
import numpy as np

class PolampEnv():
    def __init__(self, goal_env):
        self.env = goal_env
    
    def observation_space(self):
        return self.env.observation_space()

    def action_space(self):
        return self.env.action_space()

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)



class FlatKinematicsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
        self.dynamics_mode = 'Polamp'
        # Определяем новое пространство наблюдений: кинематика агента + цель
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(10,),  # [x, y, theta, v, steer, goal_x, goal_y, goal_theta, goal_v, goal_steer]
            dtype=np.float32
        )
        
        # Пространство действий остается таким же, как в исходной среде
        self.action_space = env.action_space
        self.safe_action_space = env.action_space #spaces.Box(low=-2.5, high=2.5, shape=(2,))
        #self.max_episode_steps = env.max_episode_steps()

    def seed(self, seed):
        pass

    def reset(self, **kwargs):
        # Получаем исходное наблюдение (словарь)
        obs_dict = self.env.reset(**kwargs)
        
        # Извлекаем кинематику агента и цели
        agent_kinematics = obs_dict['observation'][:5]  # x, y, theta, v, steer
        goal_kinematics = obs_dict['desired_goal'][:5]  # goal_x, goal_y, goal_theta, goal_v, goal_steer
        
        # Объединяем в один flat-вектор
        flat_obs = np.concatenate([agent_kinematics, goal_kinematics])
        
        return flat_obs, {}

    def step(self, action):
        # Выполняем шаг в исходной среде
        obs_dict, reward, done, info = self.env.step(action)
        
        # Аналогично извлекаем и объединяем наблюдения
        agent_kinematics = obs_dict['observation'][:5]
        goal_kinematics = obs_dict['desired_goal'][:5]
        flat_obs = np.concatenate([agent_kinematics, goal_kinematics])
        
        return flat_obs, reward, done, info

    def get_obstacles(self):
        return self.maps[self.map_key]

