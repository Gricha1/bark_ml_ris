import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from polamp_Models import GaussianPolicy, EnsembleCritic, LaplacePolicy, Encoder
from utils.data_aug import random_translate

def normalize_state(new_subgoal, env_state_bounds, validate=False):
	if not validate:
		new_subgoal[:, 0] = new_subgoal[:, 0] / env_state_bounds["x"]
		new_subgoal[:, 1] = new_subgoal[:, 1] / env_state_bounds["y"]
		new_subgoal[:, 2] = new_subgoal[:, 2] / env_state_bounds["theta"]
		new_subgoal[:, 3] = new_subgoal[:, 3] / env_state_bounds["v"]
		new_subgoal[:, 4] = new_subgoal[:, 4] / env_state_bounds["steer"]
	else:
		new_subgoal[0] = new_subgoal[0] / env_state_bounds["x"]
		new_subgoal[1] = new_subgoal[1] / env_state_bounds["y"]
		new_subgoal[2] = new_subgoal[2] / env_state_bounds["theta"]
		new_subgoal[3] = new_subgoal[3] / env_state_bounds["v"]
		new_subgoal[4] = new_subgoal[4] / env_state_bounds["steer"]
	return new_subgoal

class RIS(object):
	def __init__(self, state_dim, action_dim, alpha=0.1, Lambda=0.1, image_env=False, n_ensemble=10, gamma=0.99, tau=0.005, target_update_interval=1, h_lr=1e-4, q_lr=1e-3, pi_lr=1e-4, enc_lr=1e-4, epsilon=1e-16, logger=None, device=torch.device("cuda"), env_state_bounds={}):		

		# normalize states
		self.env_state_bounds = env_state_bounds

		# Actor
		self.actor = GaussianPolicy(state_dim, action_dim).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=pi_lr)
		self.actor_target = GaussianPolicy(state_dim, action_dim).to(device)
		self.actor_target.load_state_dict(self.actor.state_dict())

		# Critic
		self.critic 		= EnsembleCritic(state_dim, action_dim).to(device)
		self.critic_target 	= EnsembleCritic(state_dim, action_dim).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=q_lr)

		# Subgoal policy 
		self.subgoal_net = LaplacePolicy(state_dim).to(device)
		self.subgoal_optimizer = torch.optim.Adam(self.subgoal_net.parameters(), lr=h_lr)

		# Encoder (for vision-based envs)
		self.image_env = image_env
		if self.image_env:
			self.encoder = Encoder(state_dim=state_dim).to(device)
			self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=enc_lr)

		# Actor-Critic Hyperparameters
		self.tau = tau
		self.target_update_interval = target_update_interval
		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon

		# High-level policy hyperparameters
		self.Lambda = Lambda
		self.n_ensemble = n_ensemble

		# Utils
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.device = device
		self.logger = logger
		self.total_it = 0

	def save(self, folder, save_optims=False):
		torch.save(self.actor.state_dict(),		 folder + "actor.pth")
		torch.save(self.critic.state_dict(),		folder + "critic.pth")
		torch.save(self.subgoal_net.state_dict(),   folder + "subgoal_net.pth")
		if self.image_env:
			torch.save(self.encoder.state_dict(), folder + "encoder.pth")
		if save_optims:
			torch.save(self.actor_optimizer.state_dict(), 	folder + "actor_opti.pth")
			torch.save(self.critic_optimizer.state_dict(), 	folder + "critic_opti.pth")
			torch.save(self.subgoal_optimizer.state_dict(), folder + "subgoal_opti.pth")
			if self.image_env:
				torch.save(self.encoder_optimizer.state_dict(), folder + "encoder_opti")

	def load(self, folder):
		self.actor.load_state_dict(torch.load(folder+"actor.pth", map_location=self.device))
		self.critic.load_state_dict(torch.load(folder+"critic.pth", map_location=self.device))
		self.subgoal_net.load_state_dict(torch.load(folder+"subgoal_net.pth", map_location=self.device))
		if self.image_env:
			self.encoder.load_state_dict(torch.load(folder+"encoder.pth", map_location=self.device))

	def select_action(self, state, goal):
		with torch.no_grad():
			state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
			goal = torch.FloatTensor(goal).to(self.device).unsqueeze(0)
			if self.image_env:
				state = state.view(1, 3, 84, 84)
				goal = goal.view(1, 3, 84, 84)
				state = self.encoder(state)
				goal = self.encoder(goal)
			action, _, _ = self.actor.sample(state, goal)
		return action.cpu().data.numpy().flatten()
		
	def value(self, state, goal):
		_, _, action = self.actor.sample(state, goal)
		V = self.critic(state, action, goal).min(-1, keepdim=True)[0]
		return V

	def sample_subgoal(self, state, goal):
		subgoal_distribution = self.subgoal_net(state, goal)
		subgoal = subgoal_distribution.rsample((self.n_ensemble,))
		subgoal = torch.transpose(subgoal, 0, 1)
		return subgoal

	def sample_action_and_KL(self, state, goal):
		batch_size = state.size(0)
		# Sample action, subgoals and KL-divergence
		action_dist = self.actor(state, goal)
		action = action_dist.rsample()

		with torch.no_grad():
			subgoal = self.sample_subgoal(state, goal)
		
		prior_action_dist = self.actor_target(state.unsqueeze(1).expand(batch_size, subgoal.size(1), self.state_dim), subgoal)
		prior_prob = prior_action_dist.log_prob(action.unsqueeze(1).expand(batch_size, subgoal.size(1), self.action_dim)).sum(-1, keepdim=True).exp()
		prior_log_prob = torch.log(prior_prob.mean(1) + self.epsilon)
		D_KL = action_dist.log_prob(action).sum(-1, keepdim=True) - prior_log_prob

		action = torch.tanh(action)
		return action, D_KL

	def train_highlevel_policy(self, state, goal, subgoal):
		# Compute subgoal distribution 
		subgoal_distribution = self.subgoal_net(state, goal)

		with torch.no_grad():
			# Compute target value
			new_subgoal = subgoal_distribution.loc
			policy_v_1 = self.value(state, new_subgoal)
			policy_v_2 = self.value(new_subgoal, goal)
			policy_v = torch.cat([policy_v_1, policy_v_2], -1).clamp(min=-100.0, max=0.0).abs().max(-1)[0]

			# Compute subgoal distance loss
			v_1 = self.value(state, subgoal)
			v_2 = self.value(subgoal, goal)
			v = torch.cat([v_1, v_2], -1).clamp(min=-100.0, max=0.0).abs().max(-1)[0]
			adv = - (v - policy_v)
			weight = F.softmax(adv/self.Lambda, dim=0)

		log_prob = subgoal_distribution.log_prob(subgoal).sum(-1)
		subgoal_loss = - (log_prob * weight).mean()

		# Update network
		self.subgoal_optimizer.zero_grad()
		subgoal_loss.backward()
		self.subgoal_optimizer.step()
		

		# debug subgoal
		train_subgoal = {"x_max": new_subgoal[:, 0].max().item(),
						 "x_mean": new_subgoal[:, 0].mean().item(), 
						 "x_min": new_subgoal[:, 0].min().item(),
						 "y_max": new_subgoal[:, 1].max().item(),
						 "y_mean": new_subgoal[:, 1].mean().item(), 
						 "y_min": new_subgoal[:, 1].min().item(),
						 }
		#train_subgoal_x = new_subgoal[:, 0].mean().item()
		#train_subgoal_y = new_subgoal[:, 1].mean().item()
		if self.logger is not None:
			self.logger.store(
				train_subgoal_x_max = train_subgoal["x_max"],
				train_subgoal_x_mean = train_subgoal["x_mean"],
				train_subgoal_x_min = train_subgoal["x_min"],
				train_subgoal_y_max = train_subgoal["y_max"],
				train_subgoal_y_mean = train_subgoal["y_mean"],
				train_subgoal_y_min = train_subgoal["y_min"],
			)

		# Log variables
		if self.logger is not None:
			self.logger.store(
				adv = adv.mean().item(),
				ratio_adv = adv.ge(0.0).float().mean().item(),
				subgoal_loss = subgoal_loss.item(),
				high_policy_v = policy_v.mean().item(),
				high_v = v.mean().item()
			)

	def train(self, state, action, reward, next_state, done, goal, subgoal):
		""" Encode images (if vision-based environment), use data augmentation """

		# normalize state, goal, next_state, subgoal
		#state = normalize_state(state, self.env_state_bounds)
		#next_state = normalize_state(next_state, self.env_state_bounds)
		#goal = normalize_state(goal, self.env_state_bounds)
		#subgoal = normalize_state(subgoal, self.env_state_bounds)
		
		if self.image_env:
			state = state.view(-1, 3, 84, 84)
			next_state = next_state.view(-1, 3, 84, 84)
			goal = goal.view(-1, 3, 84, 84)
			subgoal = subgoal.view(-1, 3, 84, 84)

			# Data augmentation
			state = random_translate(state, pad=8)
			next_state = random_translate(next_state, pad=8)
			goal = random_translate(goal, pad=8)
			subgoal = random_translate(subgoal, pad=8)

			# Stop gradient for subgoal goal and next state
			state = self.encoder(state)
			with torch.no_grad():
				goal = self.encoder(goal)
				next_state = self.encoder(next_state)
				subgoal = self.encoder(subgoal)

		""" Critic """
		# Compute target Q
		with torch.no_grad():
			next_action, _, _ = self.actor.sample(next_state, goal)
			target_Q = self.critic_target(next_state, next_action, goal)
			target_Q = torch.min(target_Q, -1, keepdim=True)[0]
			target_Q = reward + (1.0-done) * self.gamma*target_Q

		# Compute critic loss
		Q = self.critic(state, action, goal)
		critic_loss = 0.5 * (Q - target_Q).pow(2).sum(-1).mean()

		# Optimize the critic
		if self.image_env: self.encoder_optimizer.zero_grad()
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		if self.image_env: self.encoder_optimizer.step()

		if self.logger is not None:
			self.logger.store(
				critic_value   = Q.mean().item(),
				target_value  = target_Q.mean().item()
			)

		self.critic_optimizer.step()

		# Stop backpropagation to encoder
		if self.image_env:
			state = state.detach()
			goal = goal.detach()
			subgoal = subgoal.detach()

		""" High-level policy learning """
		self.train_highlevel_policy(state, goal, subgoal)

		""" Actor """
		# Sample action
		action, D_KL = self.sample_action_and_KL(state, goal)

		# Compute actor loss
		Q = self.critic(state, action, goal)
		Q = torch.min(Q, -1, keepdim=True)[0]
		actor_loss = (self.alpha*D_KL - Q).mean()

		# Optimize the actor 
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		# Update target networks
		self.total_it += 1
		if self.total_it % self.target_update_interval == 0:
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


		# debug
		train_state = {"x_max": state[:, 0].max().item(),
					   "x_mean": state[:, 0].mean().item(), 
					   "x_min": state[:, 0].min().item(),
					   "y_max": state[:, 1].max().item(),
					   "y_mean": state[:, 1].mean().item(), 
					   "y_min": state[:, 1].min().item(),
					  }

		train_goal = {"x_max": goal[:, 0].max().item(),
					   "x_mean": goal[:, 0].mean().item(), 
					   "x_min": goal[:, 0].min().item(),
					   "y_max": goal[:, 1].max().item(),
					   "y_mean": goal[:, 1].mean().item(), 
					   "y_min": goal[:, 1].min().item(),
					  }
		train_reward = {"max": reward[:, 0].max().item(),
					  "mean": reward[:, 0].mean().item(), 
					  "min": reward[:, 0].min().item(),
					  }
		
		train_subgoal_data = {"x_max": subgoal[:, 0].max().item(),
							  "x_mean": subgoal[:, 0].mean().item(), 
							  "x_min": subgoal[:, 0].min().item(),
							  "y_max": subgoal[:, 1].max().item(),
							  "y_mean": subgoal[:, 1].mean().item(), 
							  "y_min": subgoal[:, 1].min().item(),
							 }

		if self.logger is not None:
			self.logger.store(
				train_state_x_max = train_state["x_max"],
				train_state_x_mean = train_state["x_mean"],
				train_state_x_min = train_state["x_min"],
				train_state_y_max = train_state["y_max"],
				train_state_y_mean = train_state["y_mean"],
				train_state_y_min = train_state["y_min"],
			)
		
		if self.logger is not None:
			self.logger.store(
				train_goal_x_max = train_goal["x_max"],
				train_goal_x_mean = train_goal["x_mean"],
				train_goal_x_min = train_goal["x_min"],
				train_goal_y_max = train_goal["y_max"],
				train_goal_y_mean = train_goal["y_mean"],
				train_goal_y_min = train_goal["y_min"],
			)

		if self.logger is not None:
			self.logger.store(
				train_reward_max = train_reward["max"],
				train_reward_mean = train_reward["mean"],
				train_reward_min = train_reward["min"],
			)
		
		if self.logger is not None:
			self.logger.store(
				train_subgoal_data_x_max = train_subgoal_data["x_max"],
				train_subgoal_data_x_mean = train_subgoal_data["x_mean"],
				train_subgoal_data_x_min = train_subgoal_data["x_min"],
				train_subgoal_data_y_max = train_subgoal_data["y_max"],
				train_subgoal_data_y_mean = train_subgoal_data["y_mean"],
				train_subgoal_data_y_min = train_subgoal_data["y_min"],
			)

		# Log variables
		if self.logger is not None:
			self.logger.store(
				actor_loss   = actor_loss.item(),
				critic_loss  = critic_loss.item(),
				D_KL		 = D_KL.mean().item()		

			)
