import torch
from torch import nn
import numpy as np

""" Actor """
class GaussianPolicy(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dims=[256, 256], goal_state_diff_obs=False, train_td3=False):
		super(GaussianPolicy, self).__init__()
		self.goal_state_diff_obs = goal_state_diff_obs
		self.train_td3 = train_td3
		fc = [nn.Linear(2*state_dim, hidden_dims[0]), nn.ReLU()]
		for hidden_dim_in, hidden_dim_out in zip(hidden_dims[:-1], hidden_dims[1:]):
			fc += [nn.Linear(hidden_dim_in, hidden_dim_out), nn.ReLU()]
		self.fc = nn.Sequential(*fc)
		self.mean_linear = nn.Linear(hidden_dims[-1], action_dim)
		if not self.train_td3:
			self.logstd_linear = nn.Linear(hidden_dims[-1], action_dim)

			self.LOG_SIG_MIN, self.LOG_SIG_MAX = -20, 2

	def forward(self, state, goal):
		if self.goal_state_diff_obs:
			x = self.fc(goal - state)
		else:
			x = self.fc(torch.cat([state, goal], -1))
		if self.train_td3:
			mean = self.mean_linear(x)
			mean = torch.tanh(mean)
			return mean
		else:
			mean = self.mean_linear(x)
			log_std = self.logstd_linear(x)
			std = torch.clamp(log_std, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX).exp()
			normal = torch.distributions.Normal(mean, std)
			return normal

	def sample(self, state, goal):
		if self.train_td3:
			mean = self.forward(state, goal)
			return mean, None, mean
		else:
			normal = self.forward(state, goal)
			x_t = normal.rsample()
			action = torch.tanh(x_t)
			log_prob = normal.log_prob(x_t)
			log_prob -= torch.log((1 - action.pow(2)) + 1e-6)
			log_prob = log_prob.sum(-1, keepdim=True)
			mean = torch.tanh(normal.mean)
			return action, log_prob, mean

""" Critic """
class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dims=[256, 256], goal_state_diff_obs=False):
		super(Critic, self).__init__()
		fc = [nn.Linear(2*state_dim + action_dim, hidden_dims[0]), nn.ReLU()]
		for hidden_dim1, hidden_dim2 in zip(hidden_dims[:-1], hidden_dims[1:]):
			fc += [nn.Linear(hidden_dim1, hidden_dim2), nn.ReLU()]
		fc += [nn.Linear(hidden_dims[-1], 1)]
		self.fc = nn.Sequential(*fc)
		self.goal_state_diff_obs = goal_state_diff_obs

	def forward(self, state, action, goal):
		if self.goal_state_diff_obs:
			x = torch.cat([goal - state, action], -1)
		else:
			x = torch.cat([state, action, goal], -1)
		return self.fc(x)


class EnsembleCritic(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dims=[256, 256], n_Q=2, goal_state_diff_obs=False):
		super(EnsembleCritic, self).__init__()
		ensemble_Q = [Critic(state_dim=state_dim, action_dim=action_dim, hidden_dims=hidden_dims, goal_state_diff_obs=goal_state_diff_obs) for _ in range(n_Q)]			
		self.ensemble_Q = nn.ModuleList(ensemble_Q)
		self.n_Q = n_Q

	def forward(self, state, action, goal):
		Q = [self.ensemble_Q[i](state, action, goal) for i in range(self.n_Q)]
		Q = torch.cat(Q, dim=-1)
		return Q

""" High-level policy """
class LaplacePolicy(nn.Module):	
	def __init__(self, state_dim, goal_dim, hidden_dims=[256, 256]):	
		super(LaplacePolicy, self).__init__()	
		fc = [nn.Linear(2*state_dim, hidden_dims[0]), nn.ReLU()]
		for hidden_dim_in, hidden_dim_out in zip(hidden_dims[:-1], hidden_dims[1:]):
			fc += [nn.Linear(hidden_dim_in, hidden_dim_out), nn.ReLU()]
		self.fc = nn.Sequential(*fc)

		#self.mean = nn.Linear(hidden_dims[-1], state_dim)	
		#self.log_scale = nn.Linear(hidden_dims[-1], state_dim)	
		self.mean = nn.Linear(hidden_dims[-1], goal_dim)	
		self.log_scale = nn.Linear(hidden_dims[-1], goal_dim)	
		self.LOG_SCALE_MIN = -20	
		self.LOG_SCALE_MAX = 2	

	def forward(self, state, goal):	
		h = self.fc( torch.cat([state, goal], -1) )	
		mean = self.mean(h)
		scale = self.log_scale(h).clamp(min=self.LOG_SCALE_MIN, max=self.LOG_SCALE_MAX).exp()	
		distribution = torch.distributions.laplace.Laplace(mean, scale)
		return distribution

""" Encoder """
def weights_init_encoder(m):
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		m.bias.data.fill_(0.0)
	elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
		assert m.weight.size(2) == m.weight.size(3)
		m.weight.data.fill_(0.0)
		m.bias.data.fill_(0.0)
		mid = m.weight.size(2) // 2
		gain = nn.init.calculate_gain('relu')
		nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

class Encoder(nn.Module):
	def __init__(self, input_dim, n_channels=3, state_dim=16, use_decoder=False):
		super(Encoder, self).__init__()
		self.encoder = nn.Sequential(
			nn.Linear(input_dim, 256), nn.ReLU(),
			nn.Linear(256, 256), nn.ReLU(),
			nn.Linear(256, state_dim)
		)
		self.use_decoder = use_decoder
		if self.use_decoder:
			self.decoder = nn.Sequential(
				nn.Linear(state_dim, 256), nn.ReLU(),
				nn.Linear(256, 256), nn.ReLU(),
				nn.Linear(256, input_dim)
			)
		self.apply(weights_init_encoder)

	def forward(self, x):
		state = self.encoder(x)
		return state

	def autoencoder_forward(self, x):
		if self.use_decoder:
			state = self.encoder(x)
			y = self.decoder(state)
			return y
		else:
			assert 1 == 0, "didnt initialize decoder"
			return

""" High-level policy: lidar predictor """
class LidarPredictor(nn.Module):
	def __init__(self, subgoal_dim=5, agent_state_dim=176, lidar_data_dim=39, lidar_max_dist=None, without_state_goal=False):
		super(LidarPredictor, self).__init__()
		self.lidar_max_dist = lidar_max_dist
		self.without_state_goal = without_state_goal
		if without_state_goal:
			self.predictor = nn.Sequential(
				nn.Linear(subgoal_dim, 256), nn.ReLU(),
				nn.Linear(256, 256), nn.ReLU(),
				nn.Linear(256, lidar_data_dim)
			)
		else:
			self.predictor = nn.Sequential(
				nn.Linear(subgoal_dim + 2*agent_state_dim, 256), nn.ReLU(),
				nn.Linear(256, 256), nn.ReLU(),
				nn.Linear(256, lidar_data_dim)
			)
		self.apply(weights_init_encoder)

	def forward(self, subgoal, state, goal):
		if self.without_state_goal:
			x = subgoal
		else:
			x = torch.cat([subgoal, state, goal], -1)
		return torch.clamp(self.predictor(x), min=0, max=self.lidar_max_dist)

