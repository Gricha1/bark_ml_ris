from bark.runtime.scenario import Scenario
from bark.runtime.scenario.scenario_generation import ScenarioGeneration
from bark.core.world.agent import *
from bark.core.models.behavior import *
from bark.core.world import *
from bark.core.world.goal_definition import *
from bark.core.world.map import *
from bark.core.models.dynamic import *
from bark.core.models.execution import *
from bark.core.geometry import *
from bark.core.geometry.standard_shapes import *
from bark.core.world.opendrive import *

import numpy as np
import shutil
import os
from pathlib import Path
import pathlib

VALIDATE_ENV = False
video_name = ""

if VALIDATE_ENV:
  # set display param and create folders -  ./pngs/ ./gifs/

  os.system('Xvfb :1 -screen 0 1600x1200x16  &')    # create virtual display with size 1600x1200 and 16 bit color. Color can be changed to 24 or 8
  os.environ['DISPLAY']=':1.0'    # tell X clients to use our virtual DISPLAY :1.0

  run_number = 1
  run_name = "run_" + f"{run_number}"
  working_dir_name = str(pathlib.Path().resolve())
  video_name = working_dir_name + "/video_validation/pngs/" + f"{run_name}"
  #gif_name = working_dir_name + "/video_validation/gifs/" + f"{run_name}"

  if os.path.isdir(video_name):
    shutil.rmtree(video_name)
  path = Path(video_name)
  path.mkdir(parents=True, exist_ok=True)

  #if os.path.isdir(gif_name):
  #  shutil.rmtree(gif_name)
  #path = Path(gif_name)
  #path.mkdir(parents=True, exist_ok=True)

  assert os.path.isdir(video_name),\
        "folder for pngs or gifs wasnt created"


class CustomBox:
  def __init__(self, left_top, left_bottom, right_bottom, right_top):
    self.left_top = left_top
    self.left_bottom = left_bottom
    self.right_bottom = right_bottom
    self.right_top = right_top
  
  def heading(self):
    EPS = 0.001
    dx = self.left_top.x - self.left_bottom.x
    dy = self.left_top.y - self.left_bottom.y
    if dx - EPS < 0 and dx + EPS > 0:
      if dy > 0:
        return np.pi / 2
      else:
        return -(np.pi / 2)
    tg = dy / dx
    return np.arctan(tg)

class AgentLaneCorridorConfig:
  """This class enables the configuration of a single LaneCorridor
     It assigns all models for the agents, determines their positions and more.
     Additionally, it can be chosen which agents should be controlled,
     what goal and model they should have.
  """
  def __init__(self,
               params=None,
               **kwargs):
    self._road_corridor = None
    self._params = params
    self._current_s = None
    self._lane_corridor = None

    # set these params
    self._road_ids = kwargs.pop("road_ids", None)
    self._lane_corridor_id = kwargs.pop("lane_corridor_id", None)
    self._s_min = kwargs.pop("s_min", 0.)
    self._s_max = kwargs.pop("s_max", 60.)
    self._ds_min = kwargs.pop("ds_min", 10.)
    self._ds_max = kwargs.pop("ds_max", 20.)
    self._min_vel = kwargs.pop("min_vel", 8.)
    self._max_vel = kwargs.pop("max_vel", 10.)
    self._source_pos = kwargs.pop("source_pos", None)
    self._sink_pos = kwargs.pop("sink_pos", None)
    self._behavior_model = \
      kwargs.pop("behavior_model", BehaviorIDMClassic(self._params))
    self._controlled_behavior_model = \
      kwargs.pop("controlled_behavior_model", None)
    self._controlled_ids = kwargs.pop("controlled_ids", None)
    self._wb = kwargs.pop("wb", 3) # wheelbase
    self._crad = kwargs.pop("crad", 1) # collision radius
    self.num_spawned_agents = 0

  def InferRoadIdsAndLaneCorr(self, world):
    goal_polygon = Polygon2d([0, 0, 0],
                             [Point2d(-1,0),
                              Point2d(-1,1),
                              Point2d(1,1),
                              Point2d(1,0)])
    start_point = Point2d(self._source_pos[0], self._source_pos[1])
    end_point = Point2d(self._sink_pos[0], self._sink_pos[1])
    goal_polygon = goal_polygon.Translate(end_point)
    self._road_corridor = world.map.GenerateRoadCorridor(
      start_point, goal_polygon)
    self._road_ids = self._road_corridor.road_ids
    self._lane_corridor = self._road_corridor.GetCurrentLaneCorridor(
      start_point)

  def state(self, world):
    """
    Returns a state of the agent
    Arguments:
        world {bark.core.world}
    Returns:
        np.array -- time, x, y, theta, velocity
    """
    pose = self.position(world)
    if pose is None:
      return None
    velocity = self.velocity()
    # old pose (near to junction)
    # return np.array([0, 80, -3, 0, 2])
    agent_x = -30
    agent_y = -3
    return np.array([0, agent_x, agent_y, 0, 2])

  def ds(self):
    """Increment for placing the agents
    Keyword Arguments:
        s_min {double} -- Min. lon. distance (default: {5.})
        s_max {double} -- Max. lon. distance (default: {10.})
    Returns:
        double -- delta s-value
    """
    return np.random.uniform(self._ds_min, self._ds_max)

  def position(self, world):
    """Using the defined LaneCorridor it finds positions for the agents
    Arguments:
        world {bark.core.world} -- BARK world
    Keyword Arguments:
        min_s {double} -- Min. lon. value (default: {0.})
        max_s {double} -- Max. lon. value (default: {100.})
    Returns:
        tuple -- (x, y, theta)
    """
    if self._road_corridor == None:
      world.map.GenerateRoadCorridor(
        self._road_ids, XodrDrivingDirection.forward)
      self._road_corridor = world.map.GetRoadCorridor(
        self._road_ids, XodrDrivingDirection.forward)
    if self._road_corridor is None:
      return None
    if self._lane_corridor is not None:
      lane_corr = self._lane_corridor
    else:
      lane_corr = self._road_corridor.lane_corridors[self._lane_corridor_id]
    if lane_corr is None:
      return None
    centerline = lane_corr.center_line
    if self._current_s == None:
      self._current_s = np.random.uniform(0, self._ds_max)
    xy_point = GetPointAtS(centerline, self._current_s)
    angle = GetTangentAngleAtS(centerline, self._current_s)
    if self._current_s > self._s_max:
      return None
    self._current_s += self.ds()

    # test
    self._current_s = self._s_max + 1
    return (xy_point.x(), xy_point.y(), angle)

  def velocity(self):
    return np.random.uniform(low=self._min_vel, high=self._max_vel)

  def behavior_model(self, world):
    """Returns behavior model
    """
    return self._behavior_model

  @property
  def execution_model(self):
    """Returns exec. model
    """
    return ExecutionModelInterpolate(self._params)

  @property
  def dynamic_model(self):
    """Returns dyn. model
    """
    return SingleTrackModel(self._params)

  @property
  def shape(self):
    return GenerateCarRectangle(self._wb, self._crad)

  def goal(self, world):
    # TODO: create correct goal definition 
    # for the agent which has geometric and kinematic goals
                                     
    #old agent pose: np.array([0, 80, -3, 0, 2])
    #left_bottom = Point2d(85, -13)
    #left_top = Point2d(85, -7)
    #right_top = Point2d(89, -7)
    #right_bottom = Point2d(89, -13)

    agent_x = -30
    agent_y = -3
   
    # test
    #left_bottom = Point2d(agent_x + 8, agent_y - 10)
    #left_top = Point2d(agent_x + 8, agent_y - 4)
    #right_top = Point2d(agent_x + 12, agent_y - 4)
    #right_bottom = Point2d(agent_x + 12, agent_y - 10)

    #left_bottom = Point2d(agent_x + 8, agent_y - 10)
    #left_top = Point2d(agent_x + 8, agent_y + 5)
    #right_top = Point2d(agent_x + 15, agent_y + 5)
    #right_bottom = Point2d(agent_x + 15, agent_y - 10)

    left_bottom = Point2d(agent_x + 25, agent_y + 2)
    left_top = Point2d(agent_x + 25, agent_y + 7)
    right_top = Point2d(agent_x + 32, agent_y + 7)
    right_bottom = Point2d(agent_x + 32, agent_y + 2)

    goal_polygon = Polygon2d([0, 0, 0],
                             [left_top,
                              left_bottom,
                              right_bottom,
                              right_top])

    return GoalDefinitionPolygon(goal_polygon)
    #goal_x = (left_top.x() + right_bottom.x()) / 2
    #goal_y = (left_top.y() + right_bottom.y()) / 2
    #goal_theta = 0
    #goal_v = 0
    #goal_steer = 0

    #return CustomGoalDefinition(goal_polygon, goal_x, goal_y, goal_theta, goal_v, goal_steer)

  def controlled_ids(self, agent_list):
    """Returns an ID-List of controlled agents
    """
    if self._controlled_ids is None:
      return []
    random_int = [agent_list[np.random.randint(0, len(agent_list))]]
    return random_int

  def controlled_goal(self, world):
    """Goal for the controlled agent
    Arguments:
        world {bark.core.world} -- BARK world
    Returns:
        GoalDefinition -- Goal for the controlled agent
    """
    return self.goal(world)

  def controlled_behavior_model(self, world):
    """Behavior model for controlled agent
    Returns:
        BehaviorModel -- BARK behavior model
    """
    if self._controlled_behavior_model is None:
      return self.behavior_model(world)

  def reset(self):
    """Resets the LaneCorridorConfig
    """
    self._current_s = None



class CustomConfigWithEase(ScenarioGeneration):
  """Configure your scenarios with ease
     &copy; Patrick Hart
  """
  def __init__(self,
               num_scenarios,
               map_file_name=None,
               params=None,
               random_seed=None,
               lane_corridor_configs=None,
               observer_model=None,
               map_interface=None):
    self._map_file_name = map_file_name
    self._lane_corridor_configs = lane_corridor_configs or []
    self._map_interface = map_interface or None
    self._observer_model = observer_model
    super(CustomConfigWithEase, self).__init__(params, num_scenarios)
    self.initialize_params(params)

  def create_scenarios(self, params, num_scenarios):
    """see baseclass
    """
    scenario_list = []
    for scenario_idx in range(0, num_scenarios):
      scenario = self.create_single_scenario()
      scenario_list.append(scenario)
    return scenario_list

  def create_single_scenario(self):
    """Creates one scenario using the defined LaneCorridorConfig

    Returns:
        Scenario -- Returns a BARK scenario
    """
    scenario = Scenario(map_file_name=self._map_file_name,
                        json_params=self._params.ConvertToDict(),
                        observer_model=self._observer_model)
    # as we always use the same world, we can create the MapIntf. once
    if self._map_interface is None:
      scenario.CreateMapInterface(self._map_file_name)
    else:
      scenario.map_interface = self._map_interface
    self._map_interface = scenario.map_interface
    world = scenario.GetWorldState()
    map_interface = world.map
    # fill agent list of the BARK world and set agents that are controlled
    scenario._agent_list = []
    scenario._eval_agent_ids = []
    agent_id = 0

    # debug 
    #print("lane corridors:", _lane_corridor_configs)

    for lc_config in self._lane_corridor_configs:
      agent_state = True
      lc_agents = []
      if lc_config._source_pos is not None and lc_config._sink_pos is not None:
        lc_config.InferRoadIdsAndLaneCorr(world)
      while agent_state is not None:
        agent_state = lc_config.state(world)
        if agent_state is not None:
          agent_behavior = lc_config.behavior_model(world)
          agent_dyn = lc_config.dynamic_model
          agent_exec = lc_config.execution_model
          agent_polygon = lc_config.shape
          agent_params = self._params.AddChild("agent")
          agent_goal = lc_config.goal(world)
          new_agent = Agent(
            agent_state,
            agent_behavior,
            agent_dyn,
            agent_exec,
            agent_polygon,
            agent_params,
            agent_goal,
            map_interface)

          new_agent.road_corridor = lc_config._road_corridor
          lc_agents.append(new_agent)
          new_agent.SetAgentId(agent_id)

        agent_id += 1
        # set the road corridor

      # handle controlled agents
      controlled_agent_ids = []
      for controlled_agent in lc_config.controlled_ids(lc_agents):

        controlled_agent.goal_definition = lc_config.controlled_goal(world)
        controlled_agent.behavior_model = \
          lc_config.controlled_behavior_model(world)
        controlled_agent_ids.append(controlled_agent.id)

      scenario._eval_agent_ids.extend(controlled_agent_ids)
      scenario._agent_list.extend(lc_agents)
      lc_config.reset()
      world.UpdateAgentRTree()

    return scenario















import os
import numpy as np
import time
import matplotlib.pyplot as plt

# BARK imports
from bark.core.models.behavior import *
from bark.runtime.commons.parameters import ParameterServer
from bark.runtime.viewer.matplotlib_viewer import MPViewer
from bark.runtime.viewer.buffered_mp_viewer import BufferedMPViewer
from bark.runtime.viewer.video_renderer import VideoRenderer
from bark.runtime.viewer.buffered_mp_viewer import BufferedMPViewer
from bark.runtime.scenario.scenario_generation.config_with_ease import \
  LaneCorridorConfig, ConfigWithEase
from bark.core.world.opendrive import XodrDrivingDirection
from bark.core.world.goal_definition import GoalDefinitionStateLimitsFrenet, GoalDefinitionPolygon
from bark.runtime.scenario.scenario_generation.configurable_scenario_generation import ConfigurableScenarioGeneration
from bark.core.world.evaluation import \
  BaseEvaluator

# BARK-ML imports
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.observers.nearest_state_observer import NearestAgentsObserver
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint
from bark_ml.environments.blueprints import ContinuousMergingBlueprint
from bark_ml.evaluators import *
from bark_ml.environments.blueprints.blueprint import Blueprint
from bark_ml.behaviors.cont_behavior import BehaviorContinuousML
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteMacroActionsML
from bark_ml.observers.nearest_state_observer import NearestAgentsObserver
from gym.spaces import *


class CustomGoalDefinition(GoalDefinitionPolygon):

  def __init__(self, goal_polygon, goal_x, goal_y, goal_theta, goal_v, goal_steer):
    self.goal_x = goal_x
    self.goal_y = goal_y
    self.goal_theta = goal_theta
    self.goal_v = goal_v
    self.goal_steer = goal_steer
    GoalDefinitionPolygon.__init__(self, goal_polygon)

  
class TestBlueprint(Blueprint):

  def __init__(self,
               params=None,
               num_scenarios=250,
               random_seed=0,
               dt=0.1,
               ml_behavior=None,
               viewer=None):

    '''
    left_lane = LaneCorridorConfig(params=params,
                                   road_ids=[0,1],
                                   min_vel=0,
                                   max_vel=3,
                                   lane_corridor_id=0,
                                   ds_min=10,
                                   ds_max=10,
                                   s_max=40)
    right_lane = LaneCorridorConfig(params=params,
                                    road_ids=[1,0],
                                    lane_corridor_id=1,
                                    min_vel=0,
                                    max_vel=3,
                                    controlled_ids=True)
    
    lane_corridors = []
    lane_corridors.append(left_lane)
    lane_corridors.append(right_lane)
    
    scenario_generation = \
      ConfigWithEase(
          num_scenarios=5,
          map_file_name="/content/highway_simple.xodr",
          random_seed=0,
          params=params,
          lane_corridor_configs=lane_corridors)
    '''
    
    # create intersection scenario
    lane_corridors = []
    
    '''
    lane_corridors.append(
      LaneCorridorConfig(params=params,
                        #road_ids = [0],
                        #lane_corridor_id=-1,
                        source_pos=[170,3],
                        sink_pos=[97,-70],
                        min_vel=2,
                        max_vel=3,
                        ds_min=0,
                        ds_max=3,
                        #ds_max=0,
                        s_max=40,
                        #s_max=0
                        #sink_pos=[97,-50],
                        ))
    '''

    lane_corridors.append(
      AgentLaneCorridorConfig(params=params,
                        #road_ids = [1],
                        #lane_corridor_id=1,
                        #source_pos=[30,-3],
                        source_pos=[13.63924827385749,-1.7491607197737886],
                        sink_pos=[103,70],
                        s_min=0,
                        s_max=30,
                        ds_min=0,
                        ds_max=10,
                        min_vel=0,
                        max_vel=6,
                        controlled_ids=True
                        ))
    scenario_generation = \
      CustomConfigWithEase(
          num_scenarios=num_scenarios,
          map_file_name="./map/testing_intersection.xodr",
          random_seed=0,
          params=params,
          lane_corridor_configs=lane_corridors)
    
    Blueprint.__init__(
      self,
      scenario_generation=scenario_generation,
      dt=dt,
      ml_behavior=ml_behavior)


class LinearAccSteerAccBehaviorContinuousML(BehaviorContinuousML):
  def __init__(self, params, dt):
    self.custom_time = dt    
    BehaviorContinuousML.__init__(self, params)
    #self._lower_bounds = params["ML"]["BehaviorContinuousML"][
    #  "ActionsLowerBound",
    #  "Lower-bound for actions.",
    #  [-1, -1]]
    #self._lower_bounds = [-1, -1]
    self._lower_bounds = [-5, -5]
    #self._upper_bounds = params["ML"]["BehaviorContinuousML"][
    #  "ActionsUpperBound",
    #  "Upper-bound for actions.",
    #  [1, 1]]
    #self._upper_bounds = [1, 1]
    self._upper_bounds = [5, 5]

    # TODO: implement get steer angle in behavior model
    self.steer_angle = 0
    self.max_acc = 5
    self.max_ang_acc = self._upper_bounds[1]
    self.max_ang_vel = 1

  # change behavior module to produce 
  # action: [linear accleration, steering acceleration]
  # origin action: [linear acceleration, steering rate] 
  def ActionToBehavior(self, action):
    
    a, w = action
    # clip actions
    a = np.clip(a, -self.max_acc, self.max_acc)
    w = np.clip(w, -self.max_ang_vel, self.max_ang_vel)
    action = [a, w]

    """
    a = np.clip(a, -self.max_acc, self.max_acc)
    Eps = np.clip(Eps, -self.max_ang_acc, self.max_ang_acc)
    _, last_v_s = self.GetAction()
    v_s = last_v_s + Eps * self.custom_time
    action = [a, v_s]

    last_steer_angle = self.steer_angle
    self.steer_angle = last_steer_angle + v_s * self.custom_time
    """

    BehaviorContinuousML.ActionToBehavior(self, action)


class ContinuousParkingBlueprint(TestBlueprint):
  def __init__(self,
               params=None,
               num_scenarios=25,
               random_seed=0,
               dt=0.1,
               viewer=None):
    ml_behavior = LinearAccSteerAccBehaviorContinuousML(params, dt)
    TestBlueprint.__init__(self,
                           params=params,
                           num_scenarios=num_scenarios,
                           random_seed=random_seed,
                           dt=dt,
                           ml_behavior=ml_behavior,
                           viewer=viewer)


class KinematicEvaluatorGoalReached(BaseEvaluator):

  def __init__(self):
    pass

  def Evaluate(self, observed_world):
    return True


class TestEvaluator:

  def __init__(self,
               params=ParameterServer(),
               eval_agent=None,
               bark_eval_fns=None,
               bark_ml_eval_fns=None):
    self._eval_agent = eval_agent
    self._params = params["ML"]["GeneralEvaluator"]
    self._bark_eval_fns = bark_eval_fns
    self._bark_ml_eval_fns = bark_ml_eval_fns

  def Evaluate(self, observed_world, action):
    """Returns information about the current world state."""
    # evaluate geometric
    eval_results = observed_world.Evaluate()
    reward = 0.
    scheduleTerminate = False

    for _, eval_fn in self._bark_ml_eval_fns.items():
      t, r, i = eval_fn(observed_world, action, eval_results)
      eval_results = {**eval_results, **i} # merge info
      reward += r # accumulate reward
      if t: # if any of the t are True -> terminal
        scheduleTerminate = True

    # evaluate kinematic
    ego_agent = observed_world.ego_agent
    goal_definition = ego_agent.goal_definition
    goal = goal_definition.goal_shape.ToArray()
    right_top = Point(goal[1][0], goal[1][1])
    right_bottom = Point(goal[2][0], goal[2][1])
    left_bottom = Point(goal[3][0], goal[3][1])
    left_top = Point(goal[4][0], goal[4][1])
    goal = CustomBox(left_top, left_bottom, right_bottom, right_top)


    # TODO: get goal info from goal_definition
    L_EPS = 0.5
    THETA_EPS = 5
    V_EPS = 0.05
    STEER_EPS = 20

    goal_x = (goal.left_top.x + goal.right_bottom.x) / 2
    goal_y = (goal.left_top.y + goal.right_bottom.y) / 2
    goal_theta = goal.heading()
    goal_v = 0
    goal_steer = 0
    goal_kinematic_state = np.array([goal_x, goal_y, goal_theta, goal_v, goal_steer])

    # TODO: get correct agent steer angle
    agent_kinematic_state = ego_agent.state[1:].tolist()
    agent_steer = 0
    agent_kinematic_state.append(agent_steer)
    agent_kinematic_state = np.array(agent_kinematic_state)
    
    dx = goal_kinematic_state[0] - agent_kinematic_state[0]
    dy = goal_kinematic_state[1] - agent_kinematic_state[1]
    dl = np.sqrt(dx ** 2 + dy ** 2)
    dtheta = goal_kinematic_state[2] - agent_kinematic_state[2]
    dv = goal_kinematic_state[3] - agent_kinematic_state[3]
    dsteer = goal_kinematic_state[4] - agent_kinematic_state[4]
    
    dl = 1 if abs(dl) <= L_EPS else 0
    dtheta = 1 if abs(dtheta) <= (180 / np.pi) * THETA_EPS else 0
    dv = 1 if abs(dv) <= V_EPS else 0
    dsteer = 1 if abs(dsteer) <= (180 / np.pi) * STEER_EPS else 0

    kinematic_goal_reached = False
    if dl + dtheta + dv + dsteer == 4:
      kinematic_goal_reached = True

    eval_results["kinematic_goal_reached"] = kinematic_goal_reached
    eval_results["dist_to_goal"] = dl

    return reward, scheduleTerminate, eval_results


  def Reset(self, world):
    world.ClearEvaluators()
    for eval_name, eval_fn in self._bark_eval_fns.items():
      world.AddEvaluator(eval_name, eval_fn())
    for _, eval_func in self._bark_ml_eval_fns.items():
      eval_func.Reset()
    return world

  def SetViewer(self, viewer):
    self._viewer = viewer

class TestMLEvaluator(TestEvaluator):
  def __init__(self, params):
    self._params = params["ML"]["RewardShapingEvaluator"]
    bark_ml_eval_fns = {}
    bark_eval_fns = {
        "goal_reached" : lambda: EvaluatorGoalReached(),
        "step_count" : lambda: EvaluatorStepCount(),
        "drivable_area" : lambda: EvaluatorDrivableArea()
    }
    super().__init__(
      params=self._params,
      bark_ml_eval_fns={}, 
      bark_eval_fns=bark_eval_fns)
    

from gym import spaces
import numpy as np
from bark.core.models.dynamic import StateDefinition
from bark.runtime.commons.parameters import ParameterServer
import operator
import cv2 as cv

from bark_ml.observers.observer import BaseObserver

class Point:
  def __init__(self, x, y):
      self.x = x
      self.y = y

class ImageObserver(BaseObserver):

  def __init__(self, params=ParameterServer()):
    BaseObserver.__init__(self, params)
    self._state_definition = [int(StateDefinition.X_POSITION),
                              int(StateDefinition.Y_POSITION),
                              int(StateDefinition.THETA_POSITION),
                              int(StateDefinition.VEL_POSITION)]
    self._max_distance_other_agents = \
      self._params["ML"]["NearestAgentsObserver"]["MaxOtherDistance",
      "Agents further than this distance are not observed; if not max" + \
      "other agents are seen, remaining concatenation state is set to zero",
      100]
    self.adding_ego_features = True
    self.adding_dynamic_features = False
    # test
    #self.gridCount = 4
    self.gridCount = 5 # static, dynamic, agent, adding_features, goal
    self.grid_resolution = 4
    self.grid_shape = (120, 120)
    state_min_box = [[[-np.inf for j in range(self.grid_shape[1])] 
            for i in range(self.grid_shape[0])] for _ in range(self.gridCount)]
    state_max_box = [[[np.inf for j in range(self.grid_shape[1])] 
            for i in range(self.grid_shape[0])] for _ in range(self.gridCount)]
    self.obs_min_box = np.array(state_min_box)
    self.obs_max_box = np.array(state_max_box)
  
  def Reset(self, world):
    self.last_images = []
    self.first_obs = True
    return super().Reset(world)

  def Observe(self, observed_world):
    # obs params
    adding_ego_features = self.adding_ego_features
    adding_dynamic_features = self.adding_dynamic_features
    gridCount = self.gridCount
    grid_resolution = self.grid_resolution
    grid_shape = self.grid_shape
    fake_static_obstacles = False
    frame_stack = 1
    assert grid_shape[0] % grid_resolution == 0 \
                   and grid_shape[1] % grid_resolution == 0, \
                   "incorrect grid shape"

    # get agent and its goal
    ego_agent = observed_world.ego_agent
    goal_definition = ego_agent.goal_definition
    goal_shape = goal_definition.goal_shape.ToArray()
    right_top = Point(goal_shape[1][0], goal_shape[1][1])
    right_bottom = Point(goal_shape[2][0], goal_shape[2][1])
    left_bottom = Point(goal_shape[3][0], goal_shape[3][1])
    left_top = Point(goal_shape[4][0], goal_shape[4][1])
    goal = CustomBox(left_top, left_bottom, right_bottom, right_top)

    """
    # get static obsts
    obstacle_segments = []
    # bottom left
    obst = []
    right_top = Point(goal.left_top.x, goal.left_top.y)
    right_bottom = Point(goal.left_bottom.x, goal.left_bottom.y)
    left_bottom = Point(goal.left_bottom.x - 12, goal.left_bottom.y)
    left_top = Point(goal.left_top.x - 12, goal.left_top.y)
    obst.append(left_top)
    obst.append(left_bottom)
    obst.append(right_bottom)
    obst.append(right_top)
    obstacle_segments.append(obst)
    # bottom right
    obst = []
    right_top = Point(goal.right_top.x + 12, goal.right_top.y)
    right_bottom = Point(goal.right_bottom.x + 12, goal.right_bottom.y)
    left_bottom = Point(goal.right_bottom.x, goal.right_bottom.y)
    left_top = Point(goal.right_top.x, goal.right_top.y)
    obst.append(left_top)
    obst.append(left_bottom)
    obst.append(right_bottom)
    obst.append(right_top)
    obstacle_segments.append(obst)
    # bottom (under parking place)
    obst = []
    right_top = Point(goal.right_bottom.x, goal.right_bottom.y)
    right_bottom = Point(goal.right_bottom.x, goal.right_bottom.y - 2)
    left_bottom = Point(goal.left_bottom.x, goal.left_bottom.y - 2)
    left_top = Point(goal.left_bottom.x, goal.left_bottom.y)
    obst.append(left_top)
    obst.append(left_bottom)
    obst.append(right_bottom)
    obst.append(right_top)
    obstacle_segments.append(obst)
    # top
    obst = []
    right_top = Point(goal.right_top.x + 12, goal.right_top.y + 8 + 4)
    right_bottom = Point(goal.right_top.x + 12, goal.right_top.y + 8)
    left_bottom = Point(goal.left_top.x - 12, goal.left_top.y + 8)
    left_top = Point(goal.left_top.x - 12, goal.left_top.y + 8 + 4)
    obst.append(left_top)
    obst.append(left_bottom)
    obst.append(right_bottom)
    obst.append(right_top)
    obstacle_segments.append(obst)
    """

    # debug
    #print("stats obsts:")
    #for box_ in obstacle_segments:
    #  print("box:")
    #  for point_ in box_:
    #    print("x:", point_.x, "y:", point_.y,)


    # get lane from observation
    #lane_corridor = observed_world.lane_corridor
    #x_s = [p[0] for p in lane_corridor.polygon.ToArray()]
    #y_s = [p[1] for p in lane_corridor.polygon.ToArray()]
    #obstacle_segments = []
    #obst = []    
    #left_bottom = Point(x_s[0], y_s[0])
    #left_top = Point(x_s[0], y_s[0])
    #obst.append((Point(x_s[3], y_s[3]), Point(0, 0)))
    #obst.append((Point(x_s[2], y_s[2]), Point(0, 0)))
    #obst.append((Point(x_s[1], y_s[1]), Point(0, 0)))
    #obst.append((Point(x_s[0], y_s[0]), Point(0, 0)))
    #obstacle_segments.append(obst)

    # TODO: get correct goal box
    grid_resolution = grid_resolution
    grid_static_obst = np.zeros(grid_shape)
    grid_dynamic_obst = np.zeros(grid_shape)
    grid_agent = np.zeros(grid_shape)
    grid_goal = np.zeros(grid_shape)
    grid_with_adding_features = np.zeros(grid_shape)
    
    agent_kinematic_state = ego_agent.state[1:].tolist()
    agent_steer = 0
    agent_kinematic_state.append(agent_steer)
    agent_kinematic_state = np.array(agent_kinematic_state)

    x_min = agent_kinematic_state[0] - self.grid_shape[0] / (2 * self.grid_resolution) 
    y_min = agent_kinematic_state[1] - self.grid_shape[1] / (2 * self.grid_resolution)

    normalized_x_init = x_min
    normalized_y_init = y_min

    
    # get normalized static boxes    
    normalized_static_boxes = []
    """
    for obstacle in obstacle_segments:
        #normalized_static_boxes.append(
        #    [Point(pair_[0].x - normalized_x_init, 
        #      pair_[0].y - normalized_y_init) 
        #      for pair_ in obstacle])
        normalized_static_boxes.append(
            [Point(point_.x - normalized_x_init, 
              point_.y - normalized_y_init) 
              for point_ in obstacle])
    """

    # get normalized dynamic boxes
    vehicles = []
    for agent_id, agent in observed_world.other_agents.items():
      vehicle = []
      vehicle_nparray = agent.GetPolygonFromState(agent.state).ToArray()
      x_s = [p[0] for p in vehicle_nparray]
      y_s = [p[1] for p in vehicle_nparray]
      vehicle.append((Point(x_s[3], y_s[3]), Point(0, 0)))
      vehicle.append((Point(x_s[2], y_s[2]), Point(0, 0)))
      vehicle.append((Point(x_s[1], y_s[1]), Point(0, 0)))
      vehicle.append((Point(x_s[0], y_s[0]), Point(0, 0)))
      vehicles.append(vehicle)

    normalized_dynamic_boxes = []
    for vehicle in vehicles:
        vertices = vehicle
        normalized_dynamic_boxes.append(
                [Point(max(0, pair_[0].x - normalized_x_init), 
                  max(0, pair_[0].y - normalized_y_init)) 
                  for pair_ in vertices])

    # get normalized agent box
    vehicle = []
    vehicle_nparray = ego_agent.GetPolygonFromState(ego_agent.state).ToArray()
    x_s = [p[0] for p in vehicle_nparray]
    y_s = [p[1] for p in vehicle_nparray]
    vehicle.append((Point(x_s[1], y_s[1]), Point(0, 0)))
    vehicle.append((Point(x_s[0], y_s[0]), Point(0, 0)))
    vehicle.append((Point(x_s[3], y_s[3]), Point(0, 0)))
    vehicle.append((Point(x_s[2], y_s[2]), Point(0, 0)))
    vertices = vehicle

    normalized_agent_box = [Point(max(0, pair_[0].x - normalized_x_init), 
                                  max(0, pair_[0].y - normalized_y_init)) 
                                  for pair_ in vertices]
    
    # get normalized goal box
    vertices = []
    vertices.append(goal.left_bottom)
    vertices.append(goal.left_top)
    vertices.append(goal.right_top)
    vertices.append(goal.right_bottom)
    normalized_goal_box = [Point(max(0, vert.x - normalized_x_init), 
                                 max(0, vert.y - normalized_y_init)) 
                                 for vert in vertices]

    
    # choice grid indexes
    all_normilized_boxes = normalized_static_boxes.copy()
    all_normilized_boxes.extend(normalized_dynamic_boxes)
    all_normilized_boxes.append(normalized_agent_box)

    # debug 
    all_normilized_boxes.append(normalized_goal_box)

    x_shape, y_shape = grid_static_obst.shape
    cv_index_boxes = []
    for box_ in all_normilized_boxes:
        box_cv_indexes = []
        for i in range(len(box_)):
            prev_x, prev_y = box_[i - 1].x, box_[i - 1].y
            curr_x, curr_y = box_[i].x, box_[i].y
            next_x, next_y = box_[(i + 1) % len(box_)].x, \
                              box_[(i + 1) % len(box_)].y
            x_f, x_ceil = np.modf(curr_x)
            y_f, y_ceil = np.modf(curr_y)
            one_x_one = (int(x_ceil * grid_resolution), 
                         int(y_ceil * grid_resolution))
            one_x_one_x_ind = 0
            one_x_one_y_ind = 0
            rx, lx, ry, ly = 1.0, 0.0, 1.0, 0.0
            curr_ind_add = grid_resolution
            while rx - lx > 1 / grid_resolution:
                curr_ind_add = curr_ind_add // 2
                mx = (lx + rx) / 2
                if x_f < mx:
                    rx = mx
                else:
                    lx = mx
                    one_x_one_x_ind += curr_ind_add
                my = (ly + ry) / 2
                if y_f < my:
                    ry = my
                else:
                    ly = my
                    one_x_one_y_ind += curr_ind_add
            if x_f == 0:
                if prev_x <= curr_x and next_x <= curr_x:
                    x_ceil -= 1
                    one_x_one = (int(x_ceil * grid_resolution), 
                                  int(y_ceil * grid_resolution))
                    one_x_one_x_ind = grid_resolution - 1
            if y_f == 0:
                if prev_y <= curr_y and next_y <= curr_y:
                    y_ceil -= 1
                    one_x_one = (int(x_ceil * grid_resolution), 
                                  int(y_ceil * grid_resolution))
                    one_x_one_y_ind = grid_resolution - 1
            index_grid_rev_x = one_x_one[0] + one_x_one_x_ind
            index_grid_rev_y = one_x_one[1] + one_x_one_y_ind
            cv_index_x = index_grid_rev_x
            cv_index_y = y_shape - index_grid_rev_y 
            box_cv_indexes.append(Point(cv_index_x, cv_index_y))
        cv_index_boxes.append(box_cv_indexes)

    # get cv index boxes in reverse order (static, dynamic, agent, goal)
    cv_index_goal_box = cv_index_boxes.pop(-1)
    cv_index_agent_box = cv_index_boxes.pop(-1)
    
    # draw on numpy matrices
    for ind_box, cv_box in enumerate(cv_index_boxes):
        contours = np.array([[cv_box[3].x, cv_box[3].y], 
                             [cv_box[2].x, cv_box[2].y], 
                             [cv_box[1].x, cv_box[1].y], 
                             [cv_box[0].x, cv_box[0].y]])
        color = 1
        if ind_box >= len(normalized_static_boxes):
            grid_dynamic_obst = cv.fillPoly(grid_dynamic_obst, 
                                                  pts = [contours], color=color)    
        grid_static_obst = cv.fillPoly(grid_static_obst, 
                                            pts = [contours], color=color)

    cv_box = cv_index_agent_box
    contours = np.array([[cv_box[3].x, cv_box[3].y], [cv_box[2].x, cv_box[2].y], 
                        [cv_box[1].x, cv_box[1].y], [cv_box[0].x, cv_box[0].y]])
    grid_agent = cv.fillPoly(grid_agent, pts = [contours], color=1)

    cv_box = cv_index_goal_box
    contours = np.array([[cv_box[3].x, cv_box[3].y], [cv_box[2].x, cv_box[2].y], 
                        [cv_box[1].x, cv_box[1].y], [cv_box[0].x, cv_box[0].y]])
    grid_goal = cv.fillPoly(grid_goal, pts = [contours], color=1)

    # TODO: set correct adding ego features, - make the row from image and features
    if not adding_ego_features:
      grid_with_adding_features = np.zeros(grid_shape)

    else:
        grid_with_adding_features = np.zeros(grid_shape)

        # TODO: get agent goal from agent.goal_definition
        # goal kinematic state
        goal_x = (goal.left_top.x + goal.right_bottom.x) / 2
        goal_y = (goal.left_top.y + goal.right_bottom.y) / 2
        goal_theta = goal.heading()
        goal_v = 0
        goal_steer = 0
        goal_kinematic_state = np.array([goal_x, goal_y, goal_theta, goal_v, goal_steer])
        
        # TODO: get correct agent steer angle
        agent_kinematic_state = ego_agent.state[1:].tolist()
        agent_steer = 0
        agent_kinematic_state.append(agent_steer)
        agent_kinematic_state = np.array(agent_kinematic_state)

        assert len(agent_kinematic_state) == len(goal_kinematic_state), \
               "goal and agent state should be the same"

        grid_with_adding_features[0, 0:len(agent_kinematic_state)] = agent_kinematic_state
        grid_with_adding_features[1, 0:len(goal_kinematic_state)] = goal_kinematic_state

    if fake_static_obstacles:
        grid_static_obst = np.zeros(grid_shape)
    dim_images = []
    dim_images.append(np.expand_dims(grid_static_obst, 0))
    dim_images.append(np.expand_dims(grid_dynamic_obst, 0))
    dim_images.append(np.expand_dims(grid_agent, 0))
    dim_images.append(np.expand_dims(grid_with_adding_features, 0))
    dim_images.append(np.expand_dims(grid_goal, 0))
    
    image = np.concatenate(dim_images, axis = 0)
    self.last_images.append(image)
    if self.first_obs:
        assert len(self.last_images) == 1, "incorrect init images"
        for _ in range(frame_stack - 1):
            self.last_images.append(image)
        self.first_obs = False
    else:
        self.last_images.pop(0)

    frames_images = np.concatenate(self.last_images, axis = 0)
 
    if fake_static_obstacles:
        self.obstacle_map = []
        self.obstacle_segments = []
 
    return frames_images
    
  @property
  def observation_space(self):
    '''
    return gym.spaces.Box(low=self.obs_min_box, 
                          high=self.obs_max_box, 
                          dtype=np.float32)
    '''
    # TODO: set correct observation for goal, and agent obs and reshape them to ONE ROW

    observation = Box(0.0, 1.0, (21168,), np.float32) 
    desired_goal = Box(0.0, 1.0, (21168,), np.float32) 
    achieved_goal = Box(0.0, 1.0, (21168,), np.float32) 
    state_observation = Box(-0.20000000298023224, 0.699999988079071, (4,), np.float32) 
    state_desired_goal = Box(-0.20000000298023224, 0.699999988079071, (4,), np.float32) 
    state_achieved_goal = Box(-0.20000000298023224, 0.699999988079071, (4,), np.float32) 
    proprio_observation = Box(-0.20000000298023224, 0.699999988079071, (2,), np.float32) 
    proprio_desired_goal = Box(-0.20000000298023224, 0.699999988079071, (2,), np.float32) 
    proprio_achieved_goal = Box(-0.20000000298023224, 0.699999988079071, (2,), np.float32) 
    image_observation = Box(-np.inf, np.inf, (28805,), np.float32) 
    image_desired_goal = Box(-np.inf, np.inf, (14405,), np.float32) 
    image_achieved_goal = Box(-np.inf, np.inf, (14405,), np.float32) 
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

    return gym.spaces.dict.Dict(obs_dict)

  def _norm(self, agent_state):
    if not self._normalization_enabled:
        return agent_state
    agent_state[int(StateDefinition.X_POSITION)] = \
      self._norm_to_range(agent_state[int(StateDefinition.X_POSITION)],
                          self._world_x_range)
    agent_state[int(StateDefinition.Y_POSITION)] = \
      self._norm_to_range(agent_state[int(StateDefinition.Y_POSITION)],
                          self._world_y_range)
    agent_state[int(StateDefinition.THETA_POSITION)] = \
      self._norm_to_range(agent_state[int(StateDefinition.THETA_POSITION)],
                          self._theta_range)
    agent_state[int(StateDefinition.VEL_POSITION)] = \
      self._norm_to_range(agent_state[int(StateDefinition.VEL_POSITION)],
                          self._velocity_range)
    return agent_state

  @staticmethod
  def _norm_to_range(value, range):
    return (value - range[0])/(range[1]-range[0])

  def _calculate_relative_agent_state(self, ego_agent_state, agent_state):
    return agent_state

  @property
  def _len_relative_agent_state(self):
    return len(self._state_definition)

  @property
  def _len_ego_state(self):
    return len(self._state_definition)




import time
import gym
from gym.envs.registration import register

params = ParameterServer()

params["Visualization"]["Agents"]["DrawAgentId", "Draw id of each agent", False]
params["Visualization"]["Agents"][
            "DrawBehaviorPlanEvalAgent", "Draw behavior plan of evalauted agent", True]
params["Visualization"]["Agents"]["DrawReference",
      "Draw reference lines with alpha trace for each agent", False]
params["Visualization"]["Agents"]["DrawRoute",
                      "Draw Route of each agent", False]
params["Visualization"]["Agents"]["Color"][
            "UseColormapForOtherAgents", "Flag to enable color map for other agents", False]
params["Visualization"]["Agents"]["Color"]["Controlled"]["Face",
                        "Color of controlled, evaluated agents", (0, 0, 0)]
params["Visualization"]["Agents"]["DrawHistory",
    "Draw history with alpha trace for each agent", False]
params["Visualization"]["Map"]["Plane"]["Color",
                  "Color of the background plane", (0, 0, 0, 0)]
params["Visualization"]["Agents"]["Alpha"]["Controlled",
                        "Alpha of evalagents", 1]
params["Visualization"]["Agents"]["Alpha"]["Other",
                             "Alpha of other agents", 1]
params["Visualization"]["Agents"]["EvalGoalColor",
               "Color of eval agent goals", (255, 255, 0)]

params["BehaviorIDMLaneTracking"]["CrosstrackErrorGain"] = 2.5
params["BehaviorIDMClassic"]["DesiredVelocity"] = 0.1
params["BehaviorIDMClassic"]["MaxLatDifferenceToBeFront"] = 0.0
params["BehaviorIDMClassic"]["MaxAngleDifferenceToBeFront"] = 3.1
params["BehaviorIDMClassic"]["MaxLonDifferenceToBeFront"] = 0.0

params["BehaviorIDMClassic"]["BrakeForLaneEnd"] = True
params["BehaviorIDMClassic"]["BrakeForLaneEndEnabledDistance"] = 100.
params["BehaviorIDMClassic"]["BrakeForLaneEndDistanceOffset"] = 30.
params["BehaviorIDMClassic"]["DesiredVelocity"] = 3
params["World"]["remove_agents_out_of_map"] = False

params["ML"]["RewardShapingEvaluator"]["PotentialVelocityFunctor"][
      "DesiredVel", "Desired velocity for the ego agent.", 20]

params["ML"]["BehaviorContinuousML"]["ActionsLowerBound",
      "Lower-bound for actions.",
      [-5, -1]]
params["ML"]["BehaviorContinuousML"][
      "ActionsUpperBound",
      "Upper-bound for actions.",
      [5, 1]]


class ContinuousParkingGym(SingleAgentRuntime, gym.Env):

  def __init__(self):
    #params = ParameterServer(filename=
    #  os.path.join(os.path.dirname(__file__),
    #  "../environments/blueprints/visualization_params.json"))
    cont_parking_bp = ContinuousParkingBlueprint(params,
                                              num_scenarios=1,
                                              dt=0.1,
                                              random_seed=0)

    #observer = NearestAgentsObserver(params)
    observer = ImageObserver(params)
    evaluator = TestMLEvaluator(params)

    renderer = MPViewer(params,
                        x_range=[-20, 20],
                        y_range=[-30, 30],
                        use_world_bounds=False,
                        enforce_x_length=True,
                        enforce_y_length=True,
                        follow_agent_id=False)

    viewer = VideoRenderer(renderer=renderer, 
                           world_step_time=cont_parking_bp._dt, 
                           video_name=video_name)
    SingleAgentRuntime.__init__(self,
                                blueprint=cont_parking_bp,
                                observer=observer,
                                viewer=viewer if VALIDATE_ENV else False,
                                evaluator=evaluator,
                                render=True if VALIDATE_ENV else False
                                )

  def reset(self, seed=None):
    info = {}
    # reset eval agent behavior model. Cuz it containes agent steer angle 
    reset_runtime = SingleAgentRuntime.reset(self)

    return reset_runtime, info
  
  def step(self, action):

    observed_state, reward, done, info = SingleAgentRuntime.step(self, action)

    # TODO: implement goal check function (kinematic characteristic)
    if info["goal_reached"]:
      done = True
      reward = 0
    else:
      reward = -1

    truncated = False
    return observed_state, reward, done, truncated, info


class GCContinuousParkingGym(ContinuousParkingGym):

  def __init__(self):
    ContinuousParkingGym.__init__(self)

  # TODO: create compute_rewards function
  def compute_rewards(self, new_actions, new_next_obs_dict):
    return np.zeros(new_actions.shape)

  def reset(self):

    # TODO: set correct observation for goal, and agent obs and reshape them to ONE ROW
    observed_state, info = ContinuousParkingGym.reset(self)

    observation = np.zeros(21168)
    desired_goal = np.zeros(21168)
    achieved_goal = np.zeros(21168)
    state_observation = np.zeros(4)
    state_desired_goal = np.zeros(4)
    state_achieved_goal = np.zeros(4)
    proprio_observation = np.zeros(2)
    proprio_desired_goal = np.zeros(2)
    proprio_achieved_goal = np.zeros(2)

    image_observation = observed_state[:2, :, :].reshape(2 * observed_state.shape[1] * observed_state.shape[2])
    image_observation = np.concatenate((image_observation, observed_state[3, 0, :5]), axis=0)

    image_desired_goal = observed_state[-1, :, :].reshape(1 * observed_state.shape[1] * observed_state.shape[2])
    image_desired_goal = np.zeros_like(image_desired_goal)
    image_desired_goal = np.concatenate((image_desired_goal, observed_state[3, 1, :5]), axis=0)

    image_achieved_goal = np.zeros(14405)
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
    
  def step(self, action):
    
    observed_state, reward, done, truncated, info = ContinuousParkingGym.step(self, action)

    # TODO: set correct observation for goal, and agent obs and reshape them to ONE ROW

    observation = np.zeros(21168)
    desired_goal = np.zeros(21168)
    achieved_goal = np.zeros(21168)
    state_observation = np.zeros(4)
    state_desired_goal = np.zeros(4)
    state_achieved_goal = np.zeros(4)
    proprio_observation = np.zeros(2)
    proprio_desired_goal = np.zeros(2)
    proprio_achieved_goal = np.zeros(2)

    image_observation = observed_state[:2, :, :].reshape(2 * observed_state.shape[1] * observed_state.shape[2])
    image_observation = np.concatenate((image_observation, observed_state[3, 0, :5]), axis=0)

    # TODO: get correct goal box
    image_desired_goal = observed_state[4:5, :, :].reshape(1 * observed_state.shape[1] * observed_state.shape[2])
    image_desired_goal = np.zeros_like(image_desired_goal)
    image_desired_goal = np.concatenate((image_desired_goal, observed_state[3, 1, :5]), axis=0)

    image_achieved_goal = np.zeros(14405)
    image_proprio_observation = np.zeros(21170)
    image_proprio_desired_goal = np.zeros(21170)
    image_proprio_achieved_goal = np.zeros(21170)

    obs_dict = {
                "observation" : observation,
                "desired_goal" : desired_goal,
                "achieved_goal" : achieved_goal,
                "state_observation" : state_observation,
                "state_desired_goal" : state_desired_goal, 
                "state_achieved_goal" : state_achieved_goal,
                "proprio_observation" : proprio_observation,
                "proprio_desired_goal" : proprio_desired_goal, 
                "proprio_achieved_goal" : proprio_achieved_goal,
                "image_observation" : image_observation,
                "image_desired_goal" : image_desired_goal,
                "image_achieved_goal" : image_desired_goal, 
                "image_proprio_observation" : image_proprio_observation, 
                "image_proprio_desired_goal" : image_proprio_desired_goal,
                "image_proprio_achieved_goal" : image_proprio_achieved_goal
              } 
    
    next_obs = obs_dict
    status = {}

    # TODO: implemet get correct xy-distance (terminal distance)
    status["xy-distance"] = [info["dist_to_goal"]]
    status["done"] = done

    # TODO: implement correct reward 
    achieved_goals = state_achieved_goal
    desired_goals = state_desired_goal
    #reward = desired_goals - achieved_goals

    # debug
    info["obs_to_validation"] = observed_state

    return next_obs, reward, done, info
