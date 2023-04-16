import numpy as np
import torch
import torch.nn.functional as F

from custom_Models import GaussianPolicy, EnsembleCritic, LaplacePolicy, goal_Encoder

from utils.data_aug import random_translate


import torch.nn as nn

# changed
def transform_subgoal(new_subgoal, env_state_bounds, was_normalized=True):
	if not was_normalized:
		new_subgoal[:, 0] = new_subgoal[:, 0].clamp(-env_state_bounds["x"], env_state_bounds["x"])
		new_subgoal[:, 1] = new_subgoal[:, 1].clamp(-env_state_bounds["y"], env_state_bounds["y"])
		new_subgoal[:, 2] = new_subgoal[:, 2].clamp(-env_state_bounds["theta"], env_state_bounds["theta"])
		new_subgoal[:, 3] = new_subgoal[:, 3].clamp(0, env_state_bounds["v"])
		new_subgoal[:, 4] = new_subgoal[:, 4].clamp(-env_state_bounds["steer"], env_state_bounds["steer"])
	else:
		new_subgoal[:, 0] = new_subgoal[:, 0].clamp(-1, 1)
		new_subgoal[:, 1] = new_subgoal[:, 1].clamp(-1, 1)
		new_subgoal[:, 2] = new_subgoal[:, 2].clamp(-1, 1)
		new_subgoal[:, 3] = new_subgoal[:, 3].clamp(0, 1)
		new_subgoal[:, 4] = new_subgoal[:, 4].clamp(-1, 1)
	return new_subgoal

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

# changed
def get_img_feat_from_rollout(state, c_count=2, batch_size=1):
	x = state[:, :-5]
	x_f = state[:, -5:]
	#x = x.view(batch_size, c_count, 120, 120)
	x = x.view(batch_size, c_count, 40, 40)
	x_f = x_f.view(batch_size, -1)
	return x, x_f

class RIS(object):
	#def __init__(self, state_dim, action_dim, alpha=0.1, Lambda=0.1, image_env=False, n_ensemble=10, gamma=0.99, tau=0.005, target_update_interval=1, h_lr=1e-4, q_lr=1e-3, pi_lr=1e-4, enc_lr=1e-4, epsilon=1e-16, logger=None, device=torch.device("cuda")):		
	def __init__(self, state_dim, action_dim, alpha=0.1, Lambda=0.1, image_env=False, n_ensemble=10, gamma=0.99, tau=0.005, target_update_interval=1, h_lr=1e-4, q_lr=1e-3, pi_lr=1e-4, enc_lr=1e-4, epsilon=1e-16, logger=None, device=torch.device("cuda"), env_state_bounds={}):		

		# changed
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
		# changed
		self.is_normalize_state = True

		self.image_env = image_env
		if self.image_env:
			self.encoder = goal_Encoder(state_dim=state_dim).to(device)
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
				# changed
				x, x_f = get_img_feat_from_rollout(state, c_count=1)
				x_goal, x_f_goal = get_img_feat_from_rollout(goal, c_count=1)
				#changed
				state = self.encoder(x, x_f)
				goal = self.encoder(x_goal, x_f_goal)
				state = x_f
				goal = x_f_goal
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
			# changed
			subgoal = transform_subgoal(subgoal, self.env_state_bounds)

		prior_action_dist = self.actor_target(state.unsqueeze(1).expand(batch_size, subgoal.size(1), self.state_dim), subgoal)
		prior_prob = prior_action_dist.log_prob(action.unsqueeze(1).expand(batch_size, subgoal.size(1), self.action_dim)).sum(-1, keepdim=True).exp()
		prior_log_prob = torch.log(prior_prob.mean(1) + self.epsilon)
		D_KL = action_dist.log_prob(action).sum(-1, keepdim=True) - prior_log_prob			

		# debug
		with torch.no_grad():
			entropy_1 = action_dist.entropy()[:, 0].mean().item()
			entropy_2 = action_dist.entropy()[:, 1].mean().item()
		# Log variables
		if self.logger is not None:
			self.logger.store(
				entropy_1   = entropy_1,
				entropy_2   = entropy_2				
			)

		action = torch.tanh(action)
		
		return action, D_KL

	def train_highlevel_policy(self, state, goal, subgoal):
		# Compute subgoal distribution 
		subgoal_distribution = self.subgoal_net(state, goal)

		with torch.no_grad():
			# Compute target value
			new_subgoal = subgoal_distribution.loc

			# changed
			new_subgoal = transform_subgoal(new_subgoal, self.env_state_bounds)

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

		# changed
		#torch.nn.utils.clip_grad_norm_(self.subgoal_net.parameters(), 0.5)

		"""
		#changed
		with torch.no_grad():
			total_norm = 0
			parameters = [p for p in self.subgoal_net.parameters() if p.grad is not None and p.requires_grad]
			for p in parameters:			
				param_norm = p.grad.detach().data.norm(2)
				total_norm += param_norm.item() ** 2
			total_norm = total_norm ** 0.5
		"""
		self.subgoal_optimizer.step()
		
		# Log variables
		#changed
		if self.logger is not None:
			self.logger.store(
				adv = adv.mean().item(),
				ratio_adv = adv.ge(0.0).float().mean().item(),
				subgoal_loss = subgoal_loss.item(),
				#subgoal_grad = total_norm,
				high_policy_v = policy_v.mean().item(),
				high_v = v.mean().item(),
				sub_goal_log_prob = log_prob.mean().item()
			)

	def train(self, state, action, reward, next_state, done, goal, subgoal):
		""" Encode images (if vision-based environment), use data augmentation """

		# changed
		if self.is_normalize_state:
			state = normalize_state(state, self.env_state_bounds)
			next_state = normalize_state(next_state, self.env_state_bounds)
			goal = normalize_state(goal, self.env_state_bounds)
			subgoal = normalize_state(subgoal, self.env_state_bounds)

		if self.image_env:
			#changed
			x_state, x_f_state = get_img_feat_from_rollout(state, c_count=1, batch_size=state.shape[0])
			x_next_state, x_f_next_state = get_img_feat_from_rollout(next_state, c_count=1, batch_size=next_state.shape[0])
			x_goal, x_f_goal = get_img_feat_from_rollout(goal, c_count=1, batch_size=goal.shape[0])
			x_sub_goal, x_f_sub_goal = get_img_feat_from_rollout(subgoal, c_count=1, batch_size=subgoal.shape[0])

			# Data augmentation
			#changed
			x_state = random_translate(x_state, pad=8)
			x_next_state = random_translate(x_next_state, pad=8)
			x_goal = random_translate(x_goal, pad=8)
			x_sub_goal = random_translate(x_sub_goal, pad=8)

			# Stop gradient for subgoal goal and next state
			#changed
			state = self.encoder(x_state, x_f_state)
			with torch.no_grad():
				goal = self.encoder(x_goal, x_f_goal)
				next_state = self.encoder(x_next_state, x_f_next_state)
				subgoal = self.encoder(x_sub_goal, x_f_sub_goal)
			#state = x_f_state
			#goal = x_f_goal
			#next_state = x_f_next_state
			#subgoal = x_f_sub_goal

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
		
		#changed
		# torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
		"""
		with torch.no_grad():
			total_norm_1 = 0
			parameters = [p for p in self.critic.ensemble_Q[0].parameters() if p.grad is not None and p.requires_grad]
			for p in parameters:			
				param_norm = p.grad.detach().data.norm(2)
				total_norm_1 += param_norm.item() ** 2
			total_norm_1 = total_norm_1 ** 0.5	

			total_norm_2 = 0
			parameters = [p for p in self.critic.ensemble_Q[1].parameters() if p.grad is not None and p.requires_grad]
			for p in parameters:			
				param_norm = p.grad.detach().data.norm(2)
				total_norm_2 += param_norm.item() ** 2
			total_norm_2 = total_norm_2 ** 0.5	

		if self.logger is not None:
			self.logger.store(
				critic_value   = Q.mean().item(),
				target_value  = target_Q.mean().item(),
				critic_grad_1   = total_norm_1,
				critic_grad_2   = total_norm_2
			)
		"""
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

		# changed
		#torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
		"""
		with torch.no_grad():
			total_norm = 0
			parameters = [p for p in self.actor.parameters() if p.grad is not None and p.requires_grad]
			for p in parameters:			
				param_norm = p.grad.detach().data.norm(2)
				total_norm += param_norm.item() ** 2
			total_norm = total_norm ** 0.5		
		if self.logger is not None:
			self.logger.store(
				actor_grad   = total_norm,
			)
		"""

		self.actor_optimizer.step()

		# Update target networks
		self.total_it += 1
		if self.total_it % self.target_update_interval == 0:
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


		# Log variables
		if self.logger is not None:
			self.logger.store(
				actor_loss   = actor_loss.item(),
				critic_loss  = critic_loss.item(),
				D_KL		 = D_KL.mean().item()				
			)
