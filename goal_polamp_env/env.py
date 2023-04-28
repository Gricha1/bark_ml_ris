import gym
import numpy as np
import math
import matplotlib.pyplot as plt
from gym.spaces import *
from polamp_env.lib.envs import POLAMPEnvironment


class GCPOLAMPEnvironment(POLAMPEnvironment):
  def __init__(self, full_env_name, config):
    POLAMPEnvironment.__init__(self, full_env_name, config)

    self.reward_scale = 1

    observation = Box(-np.inf, np.inf, (5,), np.float32) 
    desired_goal = Box(-np.inf, np.inf, (5,), np.float32) 
    achieved_goal = Box(-np.inf, np.inf, (5,), np.float32)
    state_observation = observation 
    state_desired_goal = desired_goal 
    state_achieved_goal = achieved_goal 

    obs_dict = {"observation" : observation,
                "desired_goal" : desired_goal,
                "achieved_goal" : achieved_goal,
                "state_observation" : state_observation,
                "state_desired_goal" : state_desired_goal, 
                "state_achieved_goal" : state_achieved_goal
              } 


    self.observation_space = gym.spaces.dict.Dict(obs_dict)

  def compute_rewards(self, new_actions, new_next_obs_dict):    
    return -1.0 * self.reward_scale * np.ones((new_actions.shape[0], 1))

  def reset(self, **kwargs):

    observed_state = POLAMPEnvironment.reset(self, **kwargs)

    agent = self.environment.agent.current_state
    goal = self.environment.agent.goal_state

    observation = np.array([agent.x, agent.y, agent.theta, agent.v, agent.steer])
    desired_goal = np.array([goal.x, goal.y, goal.theta, goal.v, goal.steer])
    achieved_goal = observation
    state_observation = observation
    state_desired_goal = desired_goal
    state_achieved_goal = achieved_goal

    obs_dict = {"observation" : observation,
                "desired_goal" : desired_goal,
                "achieved_goal" : achieved_goal,
                "state_observation" : state_observation,
                "state_desired_goal" : state_desired_goal, 
                "state_achieved_goal" : state_achieved_goal
              } 

    return obs_dict

    
  def step(self, action, **kwargs):
    # normalized actions = [-1:1, -1:1]
    agent = self.environment.agent
    #normalized_action = [action[0] * agent.dynamic_model.max_acc, 
    #                     action[1] * agent.dynamic_model.max_ang_vel]
    #observed_state, reward, isDone, info = POLAMPEnvironment.step(self, normalized_action, **kwargs)
    observed_state, reward, isDone, info = POLAMPEnvironment.step(self, action, **kwargs)

    agent = self.environment.agent.current_state
    goal = self.environment.agent.goal_state

    observation = np.array([agent.x, agent.y, agent.theta, agent.v, agent.steer])
    desired_goal = np.array([goal.x, goal.y, goal.theta, goal.v, goal.steer])
    achieved_goal = observation
    state_observation = observation
    state_desired_goal = desired_goal
    state_achieved_goal = achieved_goal

    obs_dict = {"observation" : observation,
                "desired_goal" : desired_goal,
                "achieved_goal" : achieved_goal,
                "state_observation" : state_observation,
                "state_desired_goal" : state_desired_goal, 
                "state_achieved_goal" : state_achieved_goal
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
    
    if not info["geometirc_goal_achieved"]:
      reward = self.compute_rewards(np.array([1]), None).item()
    else:
      reward = 0.0

    info["agent_state"] = observation
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

    
   