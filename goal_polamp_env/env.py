import gym
import numpy as np
from gym.spaces import *
from polamp_env.lib.envs import POLAMPEnvironment


class GCPOLAMPEnvironment(POLAMPEnvironment):
  def __init__(self, full_env_name, config):
    POLAMPEnvironment.__init__(self, full_env_name, config)

    # change observation space
    observation = Box(-np.inf, np.inf, (5,), np.float32) 
    desired_goal = Box(-np.inf, np.inf, (5,), np.float32) 
    achieved_goal = Box(-np.inf, np.inf, (5,), np.float32)

    state_observation = Box(-0.20000000298023224, 0.699999988079071, (4,), np.float32) 
    state_desired_goal = Box(-0.20000000298023224, 0.699999988079071, (4,), np.float32) 
    state_achieved_goal = Box(-0.20000000298023224, 0.699999988079071, (4,), np.float32) 
    proprio_observation = Box(-0.20000000298023224, 0.699999988079071, (2,), np.float32) 
    proprio_desired_goal = Box(-0.20000000298023224, 0.699999988079071, (2,), np.float32) 
    proprio_achieved_goal = Box(-0.20000000298023224, 0.699999988079071, (2,), np.float32) 

    #changed 
    image_observation = Box(-np.inf, np.inf, (5,), np.float32) 
    image_desired_goal = Box(-np.inf, np.inf, (5,), np.float32) 
    image_achieved_goal = Box(-np.inf, np.inf, (5,), np.float32)

    image_proprio_observation = Box(-0.20000000298023224, 1.0, (21170,), np.float32) 
    image_proprio_desired_goal = Box(-0.20000000298023224, 1.0, (21170,), np.float32) 
    image_proprio_achieved_goal = Box(-0.20000000298023224, 1.0, (21170,), np.float32)

    obs_dict = {"observation" : observation,
                "desired_goal" : desired_goal,
                "achieved_goal" : achieved_goal,
                "state_observation" : state_observation,
                "state_desired_goal" : state_desired_goal, 
                "state_achieved_goal" : state_achieved_goal,
                "proprio_observation" : proprio_observation,
                "proprio_desired_goal" : proprio_desired_goal, 
                "proprio_achieved_goal" : proprio_achieved_goal,
                "image_observation" :image_observation,
                "image_desired_goal" :image_desired_goal,
                "image_achieved_goal" :image_desired_goal, 
                "image_proprio_observation" :image_proprio_observation, 
                "image_proprio_desired_goal" :image_proprio_desired_goal,
                "image_proprio_achieved_goal" :image_proprio_achieved_goal
              } 


    self.observation_space = gym.spaces.dict.Dict(obs_dict)

  # TODO: create compute_rewards function
  def compute_rewards(self, new_actions, new_next_obs_dict):
    #return np.zeros(new_actions.shape)
    return np.zeros((new_actions.shape[0], 1))

  def reset(self, **kwargs):

    observed_state = POLAMPEnvironment.reset(self, **kwargs)

    agent = self.environment.agent.current_state
    goal = self.environment.agent.goal_state
    #print("agent state:", self.environment.agent.current_state)
    #print("goal state:", self.environment.agent.goal_state)


    #changed
    observation = np.array([agent.x, agent.y, agent.theta, agent.v, agent.steer])
    desired_goal = np.array([goal.x, goal.y, goal.theta, goal.v, goal.steer])
    achieved_goal = np.zeros(5)

    state_observation = np.zeros(4)
    state_desired_goal = np.zeros(4)
    state_achieved_goal = np.zeros(4)
    proprio_observation = np.zeros(2)
    proprio_desired_goal = np.zeros(2)
    proprio_achieved_goal = np.zeros(2)

    #changed
    image_observation = observation
    image_desired_goal = image_observation
    image_achieved_goal = image_desired_goal
    
    image_proprio_observation = np.zeros(21170)
    image_proprio_desired_goal = np.zeros(21170)
    image_proprio_achieved_goal = np.zeros(21170)

    obs_dict = {"observation" : observation,
                "desired_goal" : desired_goal,
                "achieved_goal" : achieved_goal,
                "state_observation" : state_observation,
                "state_desired_goal" : state_desired_goal, 
                "state_achieved_goal" : state_achieved_goal,
                "proprio_observation" : proprio_observation,
                "proprio_desired_goal" : proprio_desired_goal, 
                "proprio_achieved_goal" : proprio_achieved_goal,
                "image_observation" :image_observation,
                "image_desired_goal" :image_desired_goal,
                "image_achieved_goal" :image_desired_goal, 
                "image_proprio_observation" :image_proprio_observation, 
                "image_proprio_desired_goal" :image_proprio_desired_goal,
                "image_proprio_achieved_goal" :image_proprio_achieved_goal
              } 

    return obs_dict

    
  def step(self, action, **kwargs):

    # normalize actions action = [-1:1, -1:1]
    agent = self.environment.agent
    normalized_action = [action[0] * agent.dynamic_model.max_acc, 
                         action[1] * agent.dynamic_model.max_ang_vel]
    
    observed_state, reward, isDone, info = POLAMPEnvironment.step(self, normalized_action, **kwargs)

    agent = self.environment.agent.current_state
    goal = self.environment.agent.goal_state

    #changed
    observation = np.array([agent.x, agent.y, agent.theta, agent.v, agent.steer])
    desired_goal = np.array([goal.x, goal.y, goal.theta, goal.v, goal.steer])
    achieved_goal = np.zeros(5)

    state_observation = np.zeros(4)
    state_desired_goal = np.zeros(4)
    state_achieved_goal = np.zeros(4)
    proprio_observation = np.zeros(2)
    proprio_desired_goal = np.zeros(2)
    proprio_achieved_goal = np.zeros(2)

    #changed
    image_observation = observation
    image_desired_goal = image_observation
    image_achieved_goal = image_desired_goal
    
    image_proprio_observation = np.zeros(21170)
    image_proprio_desired_goal = np.zeros(21170)
    image_proprio_achieved_goal = np.zeros(21170)

    obs_dict = {"observation" : observation,
                "desired_goal" : desired_goal,
                "achieved_goal" : achieved_goal,
                "state_observation" : state_observation,
                "state_desired_goal" : state_desired_goal, 
                "state_achieved_goal" : state_achieved_goal,
                "proprio_observation" : proprio_observation,
                "proprio_desired_goal" : proprio_desired_goal, 
                "proprio_achieved_goal" : proprio_achieved_goal,
                "image_observation" :image_observation,
                "image_desired_goal" :image_desired_goal,
                "image_achieved_goal" :image_desired_goal, 
                "image_proprio_observation" :image_proprio_observation, 
                "image_proprio_desired_goal" :image_proprio_desired_goal,
                "image_proprio_achieved_goal" :image_proprio_achieved_goal
              } 

    info["dist_to_goal"] = info["EuclideanDistance"]
    info["last_step_num"] = self.step_counter

    isDone = False
    distance_to_goal = self.environment.get_goal_distance()
    if distance_to_goal < self.SOFT_EPS:
      info["geometirc_goal_achieved"] = True
    else:
      info["geometirc_goal_achieved"] = False
    
    if info["geometirc_goal_achieved"] or self._max_episode_steps == self.step_counter:
      isDone = True
    
    #reward = -0.1 * (not info["geometirc_goal_achieved"])
    reward = -1.0 * (not info["geometirc_goal_achieved"])


    """
    if isDone:
      print("steps in env:", self.step_counter)
      if not (self._max_episode_steps == self.step_counter):
        info["geometirc_goal_achieved"] = True
      else:
        distance_to_goal = self.environment.get_goal_distance()
        if distance_to_goal < self.SOFT_EPS:
          info["geometirc_goal_achieved"] = True
        else:
          info["geometirc_goal_achieved"] = False
    """

    info["agent_state"] = observation
    return obs_dict, reward, isDone, info

    
   
