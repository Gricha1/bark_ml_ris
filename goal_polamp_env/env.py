import gym
import numpy as np
import math
import matplotlib.pyplot as plt
from gym.spaces import *
from polamp_env.lib.envs import POLAMPEnvironment

from polamp_env.lib.structures import State


class GCPOLAMPEnvironment(POLAMPEnvironment):
  def __init__(self, full_env_name, config):
    POLAMPEnvironment.__init__(self, full_env_name, config)

    self.reward_scale = 1
    self.dataset_info = {}
    self.static_env = config["static_env"]    
    self.test_0_collision = config["test_0_collision"]
    self.test_1_collision = config["test_1_collision"]
    self.test_2_collision = config["test_2_collision"] 
    assert 1.0 * self.test_0_collision + 1.0 * self.test_1_collision + 1.0 * self.test_2_collision == 1.0
    env_boundaries = {"x": (-5 + 2, 40 - 2), "y": (-5 + 2, 36 - 2), # add -2 for visualization purpuse 
                      "theta": (-1.5707963267948966, 1.5707963267948966), 
                      "v": (0, 0), "steer": (0, 0)}
    self.dataset_info["boundaries"] = env_boundaries

    observation = Box(-np.inf, np.inf, (5,), np.float32) 
    desired_goal = Box(-np.inf, np.inf, (5,), np.float32) 
    achieved_goal = Box(-np.inf, np.inf, (5,), np.float32)
    state_observation = observation 
    state_desired_goal = desired_goal 
    state_achieved_goal = achieved_goal 
    current_step = Box(0.0, np.inf, (1,), np.float32)
    collision = Box(0.0, 1.0, (1,), np.float32)

    obs_dict = {"observation" : observation,
                "desired_goal" : desired_goal,
                "achieved_goal" : achieved_goal,
                "state_observation" : state_observation,
                "state_desired_goal" : state_desired_goal, 
                "state_achieved_goal" : state_achieved_goal,
                "current_step" : current_step,
                "collision" : collision,
              } 


    self.observation_space = gym.spaces.dict.Dict(obs_dict)

  def compute_rewards(self, new_actions, new_next_obs_dict):    
    return -1.0 * self.reward_scale * np.ones((new_actions.shape[0], 1))


  def random_data_reset(self, task=None, grid_map=None, id=None, val_key=None, static_obsts=False):
        # self.maps = dict(self.maps_init)
        self.hardGoalReached = False
        # if not grid_map is None:
            # print(f"grid_map.resolution: {grid_map.resolution}")
            # self.environment.reset(grid_map.resolution)

        self.step_counter = 0
        self.last_observations = []
        # self.last_action = [0., 0.]
        # self.obstacle_segments = []
        # self.dyn_obstacle_segments = []
        self.dynamic_obstacles = []
        # self.dyn_acc = 0
        # self.dyn_ang_vel = 0
        self.collision_time = 0
        # print(f"self.step_counter: {self.step_counter}")
        # set new tasks
        if id is None or val_key is None:
            if task is None or grid_map is None:
                self.map_key = self.lst_keys[np.random.randint(len(self.lst_keys))]
                polygon_map = self.maps[self.map_key]
                tasks = self.trainTasks[self.map_key]
                id = np.random.randint(len(tasks))
                current_task = {}
                current_task["start"] = tasks[id][0]
                current_task["goal"] = tasks[id][1]

                if not static_obsts:
                  # random sample state&goal
                  polygon_map = []
                  boundaries = [self.dataset_info["boundaries"]["x"], 
                                self.dataset_info["boundaries"]["y"],
                                self.dataset_info["boundaries"]["theta"],
                                self.dataset_info["boundaries"]["v"],
                                self.dataset_info["boundaries"]["steer"]]
                  current_task["start"] = [(boundary[1] - boundary[0]) * x + boundary[0] 
                                          for x, boundary in zip(np.random.random(5), boundaries)] 
                  current_task["goal"] = [(boundary[1] - boundary[0]) * x + boundary[0] 
                                          for x, boundary in zip(np.random.random(5), boundaries)] 
                else:
                  # left - right 1
                  # x = (-2)<=(+35), y = (33)<=(35), theta = (-0.35)<=(+0.35)
                  # left - right 2
                  # x = (-2)<=(+35), y = (+18)<=(+18.4), theta = (-0.35)<=(+0.1)
                  # x = (-2)<=(+35), y = (+16)<=(+18), theta = (-0.35)<=(+0.35)
                  # x = (-2)<=(+35), y = (+15.6)<=(+16), theta = (-0.1)<=(+0.35)
                  # left - right 3
                  # x = (-2)<=(+35), y = (-4)<=(0.5), theta = (-0.35)<=(+0.35)
                  
                  # up - down 1
                  # x = (-5)<=(-2), y = (-3)<=(+33), theta = (np.pi/2 - 0.1)<=(np.pi/2 + 0.35)
                  # up - down 2
                  # x = (+18.5)<=(+19), y = (-3)<=(+33), theta = (np.pi/2 - 0.1)<=(np.pi/2 + 0.35)
                  # x = (+17.5)<=(+18.5), y = (-3)<=(+33), theta = (np.pi/2 - 0.35)<=(np.pi/2 + 0.35)
                  # x = (+17)<=(+17.5), y = (-3)<=(+33), theta = (np.pi/2 - 0.35)<=(np.pi/2 + 0.1)
                  # up - down 3
                  # x = (38)<=(40), y = (-3)<=(+33), theta = (np.pi/2 - 0.35)<=(np.pi/2 + 0.35)

                  def get_random_sampled_state():
                    dataset_info = {}
                    env_boundaries = {"v": (0, 0), "steer": (0, 0)}
                    dataset_info["boundaries"] = env_boundaries
                    agent_horizontal_orientation = np.random.choice([False, True])
                    agent_left_to_right_down_to_up = np.random.choice([False, True])
                    goal_left_to_right = np.random.choice([False, True])
                    line = np.random.choice([1, 2, 3])
                    if agent_horizontal_orientation:
                      env_boundaries["x"] = (-2, 35)
                      env_boundaries["theta"] = (-0.35, 0.35)
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

                  current_task["start"] = get_random_sampled_state() 
                  current_task["goal"] = get_random_sampled_state() 
          
                # Checking if the task is correct
                while not self.environment.set_polygon_task(current_task, polygon_map):
                    print("------one more time------")
                    assert 1 == 0, "didnt load task"
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

    #observed_state = POLAMPEnvironment.reset(self, **kwargs)
    observed_state = self.random_data_reset(self, **kwargs, static_obsts=self.static_env)

    agent = self.environment.agent.current_state
    goal = self.environment.agent.goal_state

    observation = np.array([agent.x, agent.y, agent.theta, agent.v, agent.steer])
    desired_goal = np.array([goal.x, goal.y, goal.theta, goal.v, goal.steer])
    achieved_goal = observation
    state_observation = observation
    state_desired_goal = desired_goal
    state_achieved_goal = achieved_goal
    current_step = np.array([0.0])
    collision = np.array([0.0])

    obs_dict = {"observation" : observation,
                "desired_goal" : desired_goal,
                "achieved_goal" : achieved_goal,
                "state_observation" : state_observation,
                "state_desired_goal" : state_desired_goal, 
                "state_achieved_goal" : state_achieved_goal,
                "current_step" : current_step,
                "collision" : collision,
              } 
    
    self.previous_agent_state = np.array([agent.x, agent.y, agent.theta, agent.v, agent.steer])
    if self.static_env and self.test_0_collision:
      self.not_collision_state = None
    if self.static_env and self.test_2_collision:
      self.start_state = np.array([agent.x, agent.y, agent.theta, agent.v, agent.steer])

    return obs_dict

    
  def step(self, action, **kwargs):
    # normalized actions = [-1:1, -1:1]
    agent = self.environment.agent
    #normalized_action = [action[0] * agent.dynamic_model.max_acc, 
    #                     action[1] * agent.dynamic_model.max_ang_vel]
    #observed_state, reward, isDone, info = POLAMPEnvironment.step(self, normalized_action, **kwargs)
    observed_state, reward, isDone, info = POLAMPEnvironment.step(self, action, **kwargs)

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
    
    polamp_reward = reward
    if not info["geometirc_goal_achieved"]:
      reward = self.compute_rewards(np.array([1]), None).item()
    else:
      reward = 0.0
    #reward = polamp_reward

    if self.static_env and "Collision" in info:
      if self.test_1_collision:
        isDone = True
        reward = self.step_counter - self._max_episode_steps
      elif self.test_0_collision:
        if self.not_collision_state is None:
          self.not_collision_state = State(self.previous_agent_state)
          self.not_collision_state.v = 0
      elif self.test_2_collision:
        self.environment.agent.current_state = State(self.start_state)
    if self.static_env and self.test_0_collision:
      if self.not_collision_state is not None:
        self.environment.agent.current_state = self.not_collision_state

    agent = self.environment.agent.current_state
    goal = self.environment.agent.goal_state
    observation = np.array([agent.x, agent.y, agent.theta, agent.v, agent.steer])
    desired_goal = np.array([goal.x, goal.y, goal.theta, goal.v, goal.steer])
    achieved_goal = observation
    state_observation = observation
    state_desired_goal = desired_goal
    state_achieved_goal = achieved_goal
    current_step = np.array([1.0 * self.step_counter])
    collision = np.array([1.0 * ("Collision" in info)])
    obs_dict = {"observation" : observation,
                "desired_goal" : desired_goal,
                "achieved_goal" : achieved_goal,
                "state_observation" : state_observation,
                "state_desired_goal" : state_desired_goal, 
                "state_achieved_goal" : state_achieved_goal,
                "current_step" : current_step,
                "collision" : collision,
              } 
    info["agent_state"] = observation
    self.previous_agent_state = np.array([agent.x, agent.y, agent.theta, agent.v, agent.steer])

    return obs_dict, reward, isDone, info

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
            # print(f"self.environment.occupancy_grid.width: {self.environment.occupancy_grid.width}")
            # print(f"self.environment.occupancy_grid.height: {self.environment.occupancy_grid.height}")
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
        # dx = self.goal.x - self.current_state.x
        # dy = self.goal.y - self.current_state.y
        # theta = radToDeg(self.current_state.theta)
        # v = mToKm(self.current_state.v)
        # delta = radToDeg(self.current_state.steer)
        
        ax.set_title('lin-acc: {:.2f}, ang-vel: {:.2f}, lin-vel: {:.2f}, steer: {:.1f}, t: {:.0f}'.format(action[0], action[1], linear_velocity, steering_angle, self.step_counter))
          
        if save_image:
            fig.canvas.draw()  # draw the canvas, cache the renderer
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close('all')
            return image
        else:
            # plt.pause(0.1)
            plt.show()
            # plt.pause(0.1)
            # plt.close('all')
            # plt.close('all')

    
   