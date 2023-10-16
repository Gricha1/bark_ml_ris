import gym
import numpy as np
import math
import matplotlib.pyplot as plt
from gym.spaces import *
from polamp_env.lib.envs import POLAMPEnvironment
from polamp_env.lib.utils_operations import normalizeAngle
from polamp_env.lib.structures import State


class GCPOLAMPEnvironment(POLAMPEnvironment):
  def __init__(self, full_env_name, config):
    POLAMPEnvironment.__init__(self, full_env_name, config)
    goal_env_config = config["goal_our_env_config"]
    self.abs_time_step_reward = self.reward_config["timeStep"]
    self.collision_reward = -self.reward_config["collision"]
    self.goal_reward = self.reward_config["goal"]
    self.static_env = goal_env_config["static_env"]  
    self.dataset = goal_env_config["dataset"]
    self.uniform_feasible_train_dataset = goal_env_config["uniform_feasible_train_dataset"]
    self.random_train_dataset = goal_env_config["random_train_dataset"]
    self.use_lidar_data = goal_env_config["use_lidar_data"]    
    self.inside_obstacles_movement = goal_env_config["inside_obstacles_movement"]
    self.add_frame_stack = goal_env_config["add_frame_stack"]
    self.teleport_back_on_collision = goal_env_config["teleport_back_on_collision"]
    self.teleport_back_steps = goal_env_config["teleport_back_steps"]
    self.add_ppo_reward = goal_env_config["add_ppo_reward"]
    self.add_collision_reward = goal_env_config["add_collision_reward"]
    self.is_terminal_dist = goal_env_config["is_terminal_dist"]
    self.is_terminal_angle = goal_env_config["is_terminal_angle"]
    assert self.UPDATE_SPARSE == 1, "need for correct cost count"
    assert self.add_ppo_reward == 0, "didnt implement"
    assert self.dataset == "medium_dataset" \
           or self.dataset == "hard_dataset" \
           or self.dataset == "safety_dataset" \
           or self.dataset == "ris_easy_dataset" \
           or self.dataset == "test_medium_dataset" \
           or self.dataset == "hard_dataset_simplified" \
           or self.dataset == "hard_dataset_simplified_v2" \
           or self.dataset == "hard_dataset_simplified_turns" \
           or self.dataset == "hard_dataset_simplified_expanded" \
           or self.dataset == "hard_dataset_simplified_test" \
           or self.dataset == "cross_dataset_simplified" \
           ,"not impemented other datasets for random sampling"
    assert self.reward_config["clearance"] \
           == self.reward_config["reverse"] \
           == self.reward_config["overSpeeding"] \
           == self.reward_config["overSteering"] \
           == self.reward_config["goal"] \
           == self.reward_config["distance"] \
           == 0.0, "didnt implement these rewards"
    assert (self.static_env and self.random_train_dataset == self.inside_obstacles_movement) or not self.static_env
    assert not self.static_env or \
          (self.static_env and 
              (self.inside_obstacles_movement != self.teleport_back_on_collision) or
              (not self.inside_obstacles_movement and not self.teleport_back_on_collision)
          )
    self.goal_observation = None
    self.polamp_features_size = 10 # dx, dy, dtheta, dv, dsteer, theta, v, steer, action[0], action[1]
    observation_size = 5
    if self.add_frame_stack:
      observation = Box(-np.inf, np.inf, (observation_size * self.frame_stack if not self.use_lidar_data else self.frame_stack * (observation_size + self.n_beams),), np.float32) 
      desired_goal = Box(-np.inf, np.inf, (observation_size * self.frame_stack if not self.use_lidar_data else self.frame_stack * (observation_size + self.n_beams),), np.float32) 
      achieved_goal = Box(-np.inf, np.inf, (observation_size * self.frame_stack if not self.use_lidar_data else self.frame_stack * (observation_size + self.n_beams),), np.float32) 
      current_step = Box(0.0, np.inf, (1 * self.frame_stack,), np.float32)
    else:
      observation = Box(-np.inf, np.inf, (observation_size if not self.use_lidar_data else observation_size + self.n_beams,), np.float32) 
      desired_goal = Box(-np.inf, np.inf, (observation_size if not self.use_lidar_data else observation_size + self.n_beams,), np.float32) 
      achieved_goal = Box(-np.inf, np.inf, (observation_size if not self.use_lidar_data else observation_size + self.n_beams,), np.float32)
      current_step = Box(0.0, np.inf, (1,), np.float32)
    collision = Box(0.0, 1.0, (1,), np.float32)
    clearance_is_enough = Box(0.0, 1.0, (1,), np.float32)
    collision_happend_on_trajectory = Box(0.0, 1.0, (1,), np.float32)
    state_observation = observation 
    state_desired_goal = desired_goal 
    state_achieved_goal = state_observation

    obs_dict = {"observation" : observation,
                "desired_goal" : desired_goal,
                "achieved_goal" : achieved_goal,
                "state_observation" : state_observation,
                "state_desired_goal" : state_desired_goal, 
                "state_achieved_goal" : state_achieved_goal,
                "current_step" : current_step,
                "collision" : collision,
                "clearance_is_enough": clearance_is_enough,
                "collision_happend_on_trajectory": collision_happend_on_trajectory,
              } 
    self.observation_space = gym.spaces.dict.Dict(obs_dict)

    self.action_space = gym.spaces.Box(
                                        low=np.array([-1.0, -1.0], dtype=np.float32),\
                                        high=np.array([1.0, 1.0], dtype=np.float32)
                                        )

  def compute_rewards(self, new_actions, new_next_obs_dict):    
    return -1.0 * self.abs_time_step_reward * np.ones((new_actions.shape[0], 1))


  def reset_goal_env(self, **kwargs):
    self.dataset_info = {}
    if self.dataset == "safety_dataset" or self.dataset == "hard_dataset" or self.dataset == "hard_dataset_simplified" or self.dataset == "hard_dataset_simplified_v2" or self.dataset == "hard_dataset_simplified_turns" or self.dataset == "hard_dataset_simplified_expanded" or self.dataset == "hard_dataset_simplified_test" or self.dataset == "cross_dataset_simplified" or self.dataset == "ris_easy_dataset" or not self.static_env:
      self.dataset_info["min_x"] = -5
      self.dataset_info["max_x"] = 40
      self.dataset_info["min_y"] = -5
      self.dataset_info["max_y"] = 36
    elif self.dataset == "medium_dataset" or self.dataset == "test_medium_dataset":
      self.dataset_info["min_x"] = -5
      self.dataset_info["max_x"] = 50
      self.dataset_info["min_y"] = -5
      self.dataset_info["max_y"] = 47
    else:
      assert 1 == 0

    if not self.uniform_feasible_train_dataset and not self.random_train_dataset:
      observed_state = POLAMPEnvironment.reset(self, **kwargs)
    else:
      observed_state = self.uniform_train_data_reset(self, **kwargs)

  def uniform_train_data_reset(self, task=None, grid_map=None, id=None, val_key=None):
        self.hardGoalReached = False
        self.step_counter = 0
        self.last_observations = []
        self.dynamic_obstacles = []
        self.collision_time = 0
        if id is None or val_key is None:
            if task is None or grid_map is None:
                self.map_key = self.lst_keys[np.random.randint(len(self.lst_keys))]
                polygon_map = self.maps[self.map_key]
                tasks = self.trainTasks[self.map_key]
                id = np.random.randint(len(tasks))
                current_task = {}
                current_task["start"] = tasks[id][0]
                current_task["goal"] = tasks[id][1]

                if not self.static_env or self.random_train_dataset:
                  # random sample state&goal
                  polygon_map = []
                  env_boundaries = {"x": (-5 + 2, 40 - 2), "y": (-5 + 2, 36 - 2), # random dataset
                                    "theta": (-1.5707963267948966, 1.5707963267948966), 
                                    "v": (0, 0), "steer": (0, 0)}
                  dataset_info = {}
                  dataset_info["boundaries"] = env_boundaries
                  boundaries = [dataset_info["boundaries"]["x"], 
                                dataset_info["boundaries"]["y"],
                                dataset_info["boundaries"]["theta"],
                                dataset_info["boundaries"]["v"],
                                dataset_info["boundaries"]["steer"]]
                  current_task["start"] = [(boundary[1] - boundary[0]) * x + boundary[0] 
                                          for x, boundary in zip(np.random.random(5), boundaries)] 
                  current_task["goal"] = [(boundary[1] - boundary[0]) * x + boundary[0] 
                                          for x, boundary in zip(np.random.random(5), boundaries)] 
                else:
                  if self.uniform_feasible_train_dataset:
                    if self.dataset == "safety_dataset" or self.dataset == "hard_dataset":
                      def get_random_sampled_state():
                        dataset_info = {}
                        env_boundaries = {"v": (0, 0), "steer": (0, 0)}
                        dataset_info["boundaries"] = env_boundaries
                        agent_horizontal_orientation = np.random.choice([False, True])
                        agent_left_to_right_down_to_up = np.random.choice([False, True])
                        line = np.random.choice([1, 2, 3])
                        if agent_horizontal_orientation:
                          env_boundaries["x"] = (-2, 35)
                          if agent_left_to_right_down_to_up:
                            env_boundaries["theta"] = (-0.35, 0.35)
                          else:
                            env_boundaries["theta"] = (np.pi - 0.35, np.pi + 0.35)
                          if line == 1: env_boundaries["y"] = (33, 35)
                          elif line == 2: env_boundaries["y"] = (16.5, 17.5)
                          elif line == 3: env_boundaries["y"] = (-4, 0.5)
                        else:
                          env_boundaries["y"] = (-3, 33)
                          if agent_left_to_right_down_to_up:
                            env_boundaries["theta"] = (np.pi/2 - 0.35, np.pi/2 + 0.35)
                          else:
                            env_boundaries["theta"] = (-np.pi/2 - 0.35, -np.pi/2 + 0.35)
                          if line == 1: env_boundaries["x"] = (-5, -2)
                          elif line == 2: env_boundaries["x"] = (17.5, 18.5)
                          elif line == 3: env_boundaries["x"] = (38, 40)
                        boundaries = [dataset_info["boundaries"]["x"], 
                                      dataset_info["boundaries"]["y"],
                                      dataset_info["boundaries"]["theta"],
                                      dataset_info["boundaries"]["v"],
                                      dataset_info["boundaries"]["steer"]]
                        task = [(boundary[1] - boundary[0]) * x + boundary[0] 
                                for x, boundary in zip(np.random.random(5), boundaries)] 

                        return task

                    elif self.dataset == "ris_easy_dataset":
                      # theta = 0, np.pi
                      # case1: x = -2, 35; y = 26.5, 35
                      # case2: x = 27, 35; y = 4.5, 26.5
                      # case3: x = -2, 35; y = -3, 4.5
                      # case4: x = -2, 6; y = 4.5, 26.5

                      # theta = -np.pi/2, np.pi/2
                      # case1: x = -2, 35; y = 29, 35  ++++
                      # case2: x = 23.5, 35; y = 2, 29  ++++
                      # case3: x = -2, 35; y = -3, 2   ++++
                      # case4: x = -2, 11; y = 2, 29 ++++


                      def get_random_sampled_state():
                        dataset_info = {}
                        env_boundaries = {"v": (0, 0), "steer": (0, 0)}
                        dataset_info["boundaries"] = env_boundaries
                        geometric_case = np.random.choice([1, 2, 3, 4])
                        agent_horizontal_orientation = np.random.choice([False, True])
                        if agent_horizontal_orientation:
                          env_boundaries["theta"] = np.random.choice([-np.pi, 0, np.pi])
                          env_boundaries["theta"] = (env_boundaries["theta"], env_boundaries["theta"])
                          if geometric_case == 1:
                            env_boundaries["x"] = (-2, 35)    
                            env_boundaries["y"] = (26.5, 35)
                          elif geometric_case == 2:
                            env_boundaries["x"] = (27, 35)    
                            env_boundaries["y"] = (4.5, 26.5)
                          elif geometric_case == 3:
                            env_boundaries["x"] = (-2, 35)      
                            env_boundaries["y"] = (-3, 4.5)
                          elif geometric_case == 4:
                            env_boundaries["x"] = (-2, 6)      
                            env_boundaries["y"] = (4.5, 26.5)
                          else: assert 1 == 0
                        else:
                          env_boundaries["theta"] = np.random.choice([-np.pi/2, np.pi/2])
                          env_boundaries["theta"] = (env_boundaries["theta"], env_boundaries["theta"])
                          if geometric_case == 1:
                            env_boundaries["x"] = (-2, 35)    
                            env_boundaries["y"] = (29, 35)
                          elif geometric_case == 2:
                            env_boundaries["x"] = (23.5, 35)    
                            env_boundaries["y"] = (2, 29)
                          elif geometric_case == 3:
                            env_boundaries["x"] = (-2, 35)      
                            env_boundaries["y"] = (-3, 2)
                          elif geometric_case == 4:
                            env_boundaries["x"] = (-2, 11)      
                            env_boundaries["y"] = (2, 29)
                          else: assert 1 == 0
                        boundaries = [dataset_info["boundaries"]["x"], 
                                      dataset_info["boundaries"]["y"],
                                      dataset_info["boundaries"]["theta"],
                                      dataset_info["boundaries"]["v"],
                                      dataset_info["boundaries"]["steer"]]
                        task = [(boundary[1] - boundary[0]) * x + boundary[0] 
                                for x, boundary in zip(np.random.random(5), boundaries)] 
                        return task

                    elif self.dataset == "medium_dataset":
                      assert 1 == 0
                      def get_random_sampled_state():
                        return [0, 0, 0, 0, 0]

                    else:
                      assert 1 == 0, "doesnt exist random tasks for this dataset"

                    current_task["start"] = get_random_sampled_state() 
                    current_task["goal"] = get_random_sampled_state() 
                    # change angle to [-np.pi, np.pi]
                    current_task["start"][2] = normalizeAngle(current_task["start"][2])
                    current_task["goal"][2] = normalizeAngle(current_task["goal"][2])

                  else:
                    assert 1 == 0
          
                # Checking if the task is correct
                while not self.environment.set_polygon_task(current_task, polygon_map):
                    print("------one more time------")
                    assert 1 == 0, "didnt load task, initial state is collided"
                    self.map_key = self.lst_keys[np.random.randint(len(self.lst_keys))]
                    polygon_map = self.maps[self.map_key]
                    tasks = self.trainTasks[self.map_key]
                    id = np.random.randint(len(tasks))
                    current_task = {}
                    current_task["start"] = tasks[id][0]
                    current_task["goal"] = tasks[id][1]
                # print(f"polygon_map: {polygon_map}")
                # print(f"current_task: {current_task}")
            else:
                self.environment.set_task(task, grid_map)
        else:
            print("---------- Validating ----------")
            self.map_key = val_key
            polygon_map = self.maps[self.map_key]
            tasks = self.valTasks[self.map_key]
            current_task = {}
            current_task["start"] = tasks[id][0]
            current_task["goal"] = tasks[id][1]
            self.environment.set_polygon_task(current_task, polygon_map)

        self.environment.reset(self.environment.occupancy_grid.resolution)
        # We need to include debug for inference?
        self.environment.agent.action = [0., 0.]
        beams_observation = self.environment.get_observation(self.environment.agent.current_state)
        extra_observation = self.environment.agent.getDiff()
        
        if len(self.last_observations) == 0:
            for _ in range(self.frame_stack - 1):
                self.last_observations.extend(beams_observation)
                self.last_observations.extend(extra_observation)
        observation = list(self.last_observations)
        observation.extend(beams_observation)
        observation.extend(extra_observation)

        return np.array(observation, dtype=np.float32)

  def reset(self, **kwargs):

    self.reset_goal_env(**kwargs)
    agent = self.environment.agent.current_state
    self.goal = self.environment.agent.goal_state
    assert -np.pi <= self.goal.theta <= np.pi, "incorrect dataset"
    agent_state = [agent.x, agent.y, agent.theta, agent.v, agent.steer]
    goal_state = [self.goal.x, self.goal.y, self.goal.theta, self.goal.v, self.goal.steer]
    if self.use_lidar_data:
        beams_observation = self.environment.get_observation(agent)
        agent_state.extend(beams_observation.tolist())
        beams_observation = self.environment.get_observation(self.goal)
        goal_state.extend(beams_observation.tolist())
    if self.add_frame_stack:
      observation = []
      desired_goal = []
      current_step = [0.0 for _ in range(self.frame_stack)]
      for _ in range(self.frame_stack):
        observation.extend(agent_state)
        desired_goal.extend(goal_state)
      observation = np.array(observation)
      desired_goal = np.array(desired_goal)
      achieved_goal = observation
      current_step = np.array([current_step])
    else:
      observation = np.array(agent_state)
      desired_goal = np.array(goal_state)
      achieved_goal = observation
      current_step = np.array([0.0])
    collision = np.array([0.0])
    clearance_is_enough = self.environment.clearance_is_enough
    self.collision_happend_on_trajectory = False
    state_observation = observation 
    state_desired_goal = desired_goal 
    state_achieved_goal = state_observation
    self.goal_observation = desired_goal
    obs_dict = {"observation" : observation,
                "desired_goal" : desired_goal,
                "achieved_goal" : achieved_goal,
                "state_observation" : state_observation,
                "state_desired_goal" : state_desired_goal, 
                "state_achieved_goal" : state_achieved_goal,
                "current_step" : current_step,
                "collision" : collision,
                "clearance_is_enough": clearance_is_enough,
                "collision_happend_on_trajectory": self.collision_happend_on_trajectory,
              } 
    
    self.previous_agent_state = self.environment.agent.current_state
    self.previous_agent_states = [self.previous_agent_state]
    self.previous_agent_observation = agent_state
    if self.add_frame_stack:
      self.previous_agent_collisions = [self.collision_happend_on_trajectory for _ in range(self.frame_stack)]
      self.previous_agent_observations = [self.previous_agent_observation for _ in range(self.frame_stack)]
    else:
      self.previous_agent_observations = [self.previous_agent_observation] 
    self.max_acc = self.environment.agent.dynamic_model.max_acc
    self.max_ang_vel = self.environment.agent.dynamic_model.max_ang_vel

    return obs_dict

  def step(self, action, **kwargs):
    # action = [-1:1, -1:1]
    assert len(action) == 2
    action = [action[0] * self.max_acc, 
              action[1] * self.max_ang_vel]
    observed_state, reward, isDone, info = POLAMPEnvironment.step(self, action, **kwargs)
    # CMDP get clearance for HER buffer (dont change!!!)
    clearance_is_enough = self.environment.clearance_is_enough

    agent = self.environment.agent.current_state
    # goal = self.environment.agent.goal_state
    assert 1 == self.environment.agent.resolution, "not sure if this more than 1"
    # Checking the bounds of map0
    lower_x = 0
    upper_x = 35
    lower_y = -5
    upper_y = 36
    if agent.x < lower_x or agent.x > upper_x or agent.y < lower_y or agent.y > upper_y:
      info["Collision"] = True
      isDone = True

    info["last_step_num"] = self.step_counter

    isDone = False
    distance_to_goal = info["EuclideanDistance"]
    angle_to_goal = abs(normalizeAngle(abs(agent.theta - self.goal.theta)))
    goal_achieved = 1.0 * self.is_terminal_dist * (distance_to_goal < self.SOFT_EPS) \
                  + 1.0 * self.is_terminal_angle * (angle_to_goal < self.ANGLE_EPS)
    goal_achieved = bool(goal_achieved // (1.0 * self.is_terminal_dist + 1.0 * self.is_terminal_angle))
    info["goal_achieved"] = goal_achieved

    if info["goal_achieved"] or self._max_episode_steps == self.step_counter:
      isDone = True
    
    polamp_reward = reward
    if not info["goal_achieved"]:
      reward = self.compute_rewards(np.array([1]), None).item()
    else:
      reward = 0.0
    if self.add_ppo_reward:
      assert 1 == 0, "incorrect count with collision"
      reward = polamp_reward

    if self.static_env and "Collision" in info:
      self.collision_happend_on_trajectory = True
      if self.teleport_back_on_collision:
        reward += self.collision_reward
        steer_angle_when_collide = self.environment.agent.current_state.steer
        self.environment.agent.current_state = self.previous_agent_states[-self.teleport_back_steps]
        self.environment.agent.current_state.v = 0
        self.environment.agent.current_state.steer = steer_angle_when_collide
        self.previous_agent_states = self.previous_agent_states[:-self.teleport_back_steps]
        self.previous_agent_observations = self.previous_agent_observations[:-self.teleport_back_steps]
      elif self.inside_obstacles_movement:
        if not self.add_ppo_reward:
          reward += self.collision_reward
      else:
        isDone = True
        reward += self.collision_reward
    
    agent = self.environment.agent.current_state
    # goal = self.environment.agent.goal_state
    agent_state = [agent.x, agent.y, agent.theta, agent.v, agent.steer]
    # goal_state = [goal.x, goal.y, goal.theta, goal.v, goal.steer]
    if self.use_lidar_data:
        # beams_observation = self.environment.get_observation(agent)
        agent_state.extend(self.beams_observation.tolist())
        # beams_observation = self.environment.get_observation(goal)
        # goal_state.extend(beams_observation.tolist())
    if self.add_frame_stack:
      observation = agent_state.copy()
      # goal_state = goal_state.copy()
      # desired_goal = []
      current_step = [1.0 * self.step_counter for _ in range(self.frame_stack)] # for last state
      for i in range(1, self.frame_stack):
        observation.extend(self.previous_agent_observations[-i])
      # for _ in range(self.frame_stack):
      #   desired_goal.extend(goal_state)
      observation = np.array(observation)
      # desired_goal = np.array(desired_goal)
      achieved_goal = observation
      current_step = np.array([current_step])
    else:
      observation = np.array(agent_state)
      # desired_goal = np.array(goal_state)
      achieved_goal = observation
      current_step = np.array([1.0 * self.step_counter])
    collision = np.array([1.0 * ("Collision" in info)])
    state_observation = observation 
    desired_goal = self.goal_observation 
    state_desired_goal = desired_goal
    state_achieved_goal = state_observation
    obs_dict = {"observation" : observation,
                "desired_goal" : desired_goal,
                "achieved_goal" : achieved_goal,
                "state_observation" : state_observation,
                "state_desired_goal" : state_desired_goal, 
                "state_achieved_goal" : state_achieved_goal,
                "current_step" : current_step,
                "collision" : collision,
                "clearance_is_enough": clearance_is_enough,
                "collision_happend_on_trajectory": 1.0 * self.collision_happend_on_trajectory,
              } 
    info["agent_state"] = [agent.x, agent.y, agent.theta, agent.v, agent.steer]
    info["goal_state"] = [self.goal.x, self.goal.y, self.goal.theta, self.goal.v, self.goal.steer]
    self.previous_agent_state = agent
    if self.teleport_back_on_collision:
      self.previous_agent_states.append(self.previous_agent_state)
    self.previous_agent_observation = agent_state
    self.previous_agent_observations.append(self.previous_agent_observation)
    self.previous_agent_observations = self.previous_agent_observations[1:]
    
    return obs_dict, reward, isDone, info

  def HER_reward(self, state, action, next_state, goal, collision, goal_was_reached, step_counter):
    
    assert self.reward_config["collision"] == abs(self.collision_reward) \
           and self.reward_config["timeStep"] == self.abs_time_step_reward \
           and self.reward_config["goal"] == self.goal_reward
    
    #previous_delta = self.agent.get_previous_goal_distance()
    #new_delta = self.agent.get_goal_distance()
    #reward = []
    #reward.append(-1 if collision else 0)

    # reward collision & goal reached
    reward = goal_was_reached * self.goal_reward
    reward += collision * self.collision_reward
    # distance reward
    previous_delta = np.hypot(state[:, 0:1] - goal[:, 0:1], state[:, 1:2] - goal[:, 1:2])
    new_delta = np.hypot(next_state[:, 0:1] - goal[:, 0:1], next_state[:, 1:2] - goal[:, 1:2])
    new_delta[new_delta < 0.5] = 0.5
    reward += 1.0 * (((step_counter + 1) % self.environment.UPDATE_SPARSE) == 0) \
                  * ((previous_delta - new_delta) * self.environment.occupancy_grid.resolution)
    # timestep reward
    reward += -1.0 * (((step_counter + 1) % self.environment.UPDATE_SPARSE) == 0) \
                  * self.abs_time_step_reward

    #if goal_was_reached:
    #    reward.append(1)
    #else:
    #    reward.append(0)

    #if self.environment.clearance_is_enough:
    #    reward.append(0)
    #else:
    #    reward.append(-1)
    """
    if not ((step_counter + 1) % self.UPDATE_SPARSE):
        if (new_delta < 0.5):
            new_delta = 0.5
        reward.append(-1)
        # if self.with_potential:
        #     reward.append((previous_delta - new_delta) / new_delta)
        # else:
        # print(f"previous_delta - new_delta: {previous_delta - new_delta}")
        reward.append((previous_delta - new_delta) * self.occupancy_grid.resolution)
        reward.append(0 if self.agent.current_state.v >= 0 else -1)
        reward.append(-1 if self.agent.overSpeeding else 0)
        reward.append(-1 if self.agent.overSteering else 0)
        #Penalizing only for acceleration
        # reward.append(0)
        # Updating the current state
        self.agent.old_state = self.agent.current_state
    else:
        reward.append(0)
        reward.append(0)
        reward.append(0)
        reward.append(0)
        reward.append(0)

    # print(f"self.reward_weights: {self.reward_weights}")
    # print(f"reward: {reward}")
    return np.matmul(self.reward_weights, reward)
    """
    return reward

  # overload
  def render(self, mode="human", save_image=True):
        
        fig, ax = plt.subplots(figsize=(10, 8))

        # for axis in ['top','bottom','left','right']:
        #     ax.spines[axis].set_linewidth(4)
        resolution = self.environment.occupancy_grid.resolution
        # self.MAX_DIST_LIDAR = 20
        x_delta = self.MAX_DIST_LIDAR / resolution
        y_delta = self.MAX_DIST_LIDAR / resolution
        # print(f"x_delta: {x_delta}")

        x_min = self.environment.agent.current_state.x - x_delta
        x_max = self.environment.agent.current_state.x + x_delta
        ax.set_xlim(x_min, x_max)

        y_min = self.environment.agent.current_state.y - y_delta
        y_max = self.environment.agent.current_state.y + y_delta
        ax.set_ylim(y_min, y_max)

        current_state = self.environment.agent.current_state
        center_state = self.environment.agent.dynamic_model.shift_state(current_state)
        agentBB = self.environment.getBB(center_state, ego=True)
        self.drawObstacles(agentBB)
        ax.arrow(current_state.x, current_state.y, 3 * math.cos(current_state.theta), 3 * math.sin(current_state.theta), head_width=0.5,
                 color='red', linewidth=4)

        center_state = self.environment.agent.dynamic_model.shift_state(self.environment.agent.goal_state)
        agentBB = self.environment.getBB(center_state, ego=True)
        self.drawObstacles(agentBB, "-g")
        ax.arrow(self.environment.agent.goal_state.x, self.environment.agent.goal_state.y, 3 * math.cos(self.environment.agent.goal_state.theta), 3 * math.sin(self.environment.agent.goal_state.theta), head_width=0.5,
                 color='green', linewidth=4)

        ax.plot([current_state.x, self.environment.agent.goal_state.x], [current_state.y, self.environment.agent.goal_state.y], '--r')
        
        tuple_rays = self.environment.get_observation(current_state, debug=True)
        
        counter = 0
        for distance, goal in zip(tuple_rays[0], tuple_rays[1]):
            if counter == 0:
                ax.plot([current_state.x, goal[0]], [current_state.y, goal[1]], '-', linewidth = 4, color='green')
                counter += 1
                continue
            ax.plot([current_state.x, goal[0]], [current_state.y, goal[1]], '-', linewidth = 2, color='orange')
        
        counter = 0
        for distance, angle in zip(self.environment.agent.safe_vehicle_array, self.environment.angle_space):
            if counter == 0:
                ax.plot([current_state.x, current_state.x + distance * math.cos(angle + current_state.theta)],\
                        [current_state.y, current_state.y + distance * math.sin(angle + current_state.theta)],\
                        '-', linewidth = 4, color='red')
                counter += 1
                continue
            ax.plot([current_state.x, current_state.x + distance * math.cos(angle + current_state.theta)],\
                    [current_state.y, current_state.y + distance * math.sin(angle + current_state.theta)],\
                    '-', linewidth = 2, color='cyan')

        if self.is_polygon_env:
            for obstacle_segment in self.environment.obstacle_segments:
                self.drawObstacles(obstacle_segment, "-b")
        else:
            print(f"self.environment.occupancy_grid.resolution: {self.environment.occupancy_grid.resolution}")
            for id, cell in enumerate(self.environment.occupancy_grid.map):
                if cell >= 80:
                    # print(f"cell: {cell}")
                    x = id % self.environment.occupancy_grid.width
                    y = id // self.environment.occupancy_grid.height
                    if x >= x_min and x <= x_max and y >= y_min and y <= y_max:
                        ax.plot(x, y, ".k")
            
            contour = self.environment.get_grid_robot()
            for id, point in enumerate(contour):
                ax.plot(point[0], point[1], "Hg")
        
        action = self.environment.agent.action
        linear_velocity = current_state = self.environment.agent.current_state.v
        steering_angle = current_state = self.environment.agent.current_state.steer * 180 / math.pi

        ax.set_title('lin-acc: {:.2f}, ang-vel: {:.2f}, lin-vel: {:.2f}, steer: {:.1f}, t: {:.0f}'.format(action[0], action[1], linear_velocity, steering_angle, self.step_counter))
          
        if save_image:
            fig.canvas.draw()  # draw the canvas, cache the renderer
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close('all')
            return image
        else:
            plt.show()

    
   