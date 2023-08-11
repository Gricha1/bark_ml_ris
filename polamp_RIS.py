import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from polamp_Models import GaussianPolicy, EnsembleCritic, LaplacePolicy, Encoder
from utils.data_aug import random_translate
from utils.data_aug import NormalNoise

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
	def __init__(self, state_dim, action_dim, alpha=0.1, Lambda=0.1, 
				 use_decoder=False, use_encoder=False, 
				 safety=False, safety_add_to_high_policy=False, cost_limit=0.5, update_lambda=1000, 
				 n_ensemble=10, gamma=0.99, tau=0.005, target_update_interval=1, 
				 h_lr=1e-4, q_lr=1e-3, pi_lr=1e-4, enc_lr=1e-4, epsilon=1e-16, 
				 clip_v_function=-100,
				 logger=None, device=torch.device("cuda"), env_state_bounds={}, 
				 env_obs_dim=None, add_ppo_reward=False, add_obs_noise=False, 
				 curriculum_high_policy=False):		

		assert not (use_decoder and not use_encoder), 'cant use decoder without encoder'
		assert add_ppo_reward == False, "didnt implement PPO reward for high level policy"
		assert not safety_add_to_high_policy or (safety_add_to_high_policy and safety)
		# normalize states
		self.env_state_bounds = env_state_bounds
		self.safety = safety
		self.safety_add_to_high_policy = safety_add_to_high_policy
		self.curriculum_high_policy = curriculum_high_policy
		self.stop_train_high_policy = False

		# Actor
		self.actor = GaussianPolicy(state_dim, action_dim).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=pi_lr)
		self.actor_target = GaussianPolicy(state_dim, action_dim).to(device)
		self.actor_target.load_state_dict(self.actor.state_dict())

		# Critic
		self.critic 		= EnsembleCritic(state_dim, action_dim, n_Q=1).to(device)
		self.critic_target 	= EnsembleCritic(state_dim, action_dim, n_Q=1).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=q_lr)

		# Safety Critic
		if self.safety:
			self.test_case_soft_critic = False
			self.test_max_critic = True
			self.critic_cost 		= EnsembleCritic(state_dim, action_dim, n_Q=2).to(device)
			self.critic_cost_target = EnsembleCritic(state_dim, action_dim, n_Q=2).to(device)
			self.critic_cost_target.load_state_dict(self.critic_cost.state_dict())
			self.critic_cost_optimizer = torch.optim.Adam(self.critic_cost.parameters(), lr=q_lr)

			self.cost_limit         = cost_limit
			self.update_lambda      = update_lambda
			self.lambda_coefficient = torch.tensor(1.0, requires_grad=True)
			self.lambda_optimizer = torch.optim.Adam([self.lambda_coefficient], lr=5e-4)
			
		# Subgoal policy 
		self.subgoal_net = LaplacePolicy(state_dim).to(device)
		self.subgoal_optimizer = torch.optim.Adam(self.subgoal_net.parameters(), lr=h_lr)

		# Encoder
		self.add_obs_noise = add_obs_noise
		if self.add_obs_noise:
			obs_noise_x = NormalNoise(sigma=1)
			obs_noise_y = NormalNoise(sigma=1)
			obs_noise_theta = NormalNoise(sigma=0.15)
			obs_noise_v = NormalNoise(sigma=1)
			obs_noise_steer = NormalNoise(sigma=0.1)
		self.use_encoder = use_encoder
		self.use_decoder = use_decoder
		if self.use_encoder:
			if not self.use_decoder:
				self.encoder = Encoder(input_dim=env_obs_dim, state_dim=state_dim).to(device)
				self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=enc_lr)
			else:
				# AutoEncoder
				self.encoder = Encoder(input_dim=env_obs_dim, state_dim=state_dim, use_decoder=True).to(device)
				self.encoder_optimizer = torch.optim.Adam(self.encoder.encoder.parameters(), lr=enc_lr)
				self.autoencoder_criterion = nn.MSELoss()
				self.autoencoder_optimizer = torch.optim.Adam(self.encoder.decoder.parameters(), lr=enc_lr)

		# Actor-Critic Hyperparameters
		self.tau = tau
		self.target_update_interval = target_update_interval
		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon

		# High-level policy hyperparameters
		self.clip_v_function = clip_v_function
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
		if self.safety:
			torch.save(self.critic_cost.state_dict(),		folder + "critic_cost.pth")
		torch.save(self.subgoal_net.state_dict(),   folder + "subgoal_net.pth")
		if self.use_encoder:
			torch.save(self.encoder.state_dict(), folder + "encoder.pth")
		if save_optims:
			torch.save(self.actor_optimizer.state_dict(), 	folder + "actor_opti.pth")
			torch.save(self.critic_optimizer.state_dict(), 	folder + "critic_opti.pth")
			torch.save(self.subgoal_optimizer.state_dict(), folder + "subgoal_opti.pth")
			if self.use_encoder:
				torch.save(self.encoder_optimizer.state_dict(), folder + "encoder_opti")

	def load(self, folder):
		self.actor.load_state_dict(torch.load(folder+"actor.pth", map_location=self.device))
		self.critic.load_state_dict(torch.load(folder+"critic.pth", map_location=self.device))
		if self.safety:
			self.critic_cost.load_state_dict(torch.load(folder+"critic_cost.pth", map_location=self.device))
		self.subgoal_net.load_state_dict(torch.load(folder+"subgoal_net.pth", map_location=self.device))
		if self.use_encoder:
			self.encoder.load_state_dict(torch.load(folder+"encoder.pth", map_location=self.device))

	def select_action(self, state, goal):
		with torch.no_grad():
			state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
			goal = torch.FloatTensor(goal).to(self.device).unsqueeze(0)
			if self.use_encoder:
				state = self.encoder(state)
				goal = self.encoder(goal)
			action, _, _ = self.actor.sample(state, goal)
		return action.cpu().data.numpy().flatten()
		
	def value(self, state, goal):
		_, _, action = self.actor.sample(state, goal)
		V = self.critic(state, action, goal).min(-1, keepdim=True)[0]
		return V

	# if self.safety
	def safety_value(self, state, goal):
		_, _, action = self.actor.sample(state, goal)
		V = self.critic_cost(state, action, goal).min(-1, keepdim=True)[0]
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
			policy_v = torch.cat([policy_v_1, policy_v_2], -1).clamp(min=self.clip_v_function, max=0.0).abs().max(-1)[0]
			if self.safety_add_to_high_policy:
				policy_sefety_v_1 = self.safety_value(state, new_subgoal)
				policy_sefety_v_2 = self.safety_value(new_subgoal, goal)
				policy_sefety_v = torch.cat([policy_sefety_v_1, policy_sefety_v_2], -1).max(-1)[0]
				policy_v += policy_sefety_v

			# Compute subgoal distance loss
			v_1 = self.value(state, subgoal)
			v_2 = self.value(subgoal, goal)
			v = torch.cat([v_1, v_2], -1).clamp(min=self.clip_v_function, max=0.0).abs().max(-1)[0]
			if self.safety_add_to_high_policy:
				safety_v_1 = self.safety_value(state, subgoal)
				safety_v_2 = self.safety_value(subgoal, goal)
				safety_v = torch.cat([safety_v_1, safety_v_2], -1).max(-1)[0]
				v += safety_v
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

	# if self.safety
	def train_lagrangian(self, state, action, goal):
		if self.use_encoder:
			with torch.no_grad():
				state = self.encoder(state)
				goal = self.encoder(goal)
		Q_cost = self.critic_cost(state, action, goal)
		Q_cost = torch.min(Q_cost, -1, keepdim=True)[0]
		violation = Q_cost - self.cost_limit
		lambda_loss =  self.lambda_coefficient * violation.detach()
		#lambda_loss = -lambda_loss.sum(dim=-1)
		lambda_loss = -lambda_loss.mean()
		self.lambda_optimizer.zero_grad()
		lambda_loss.backward()
		self.lambda_optimizer.step()

		if self.logger is not None:
			self.logger.store(
				lambda_coef   = self.lambda_coefficient.item(),
			)

	def train(self, state, action, reward, cost, next_state, done, goal, subgoal):
		assert cost.min().item() >= 0, f"batch cost:{cost.min().item()}, cant be negative"
		assert done.min().item() >= 0, f"done{done.min().item()}"
		""" Encode images (if vision-based environment), use data augmentation """
		if self.use_encoder:
			if self.add_obs_noise:
				assert 1 == 0, "didnt implement"
				state = obs_noise_x.perturb_action(state, min_action=-np.inf, max_action=np.inf)
				obs_noise_y = NormalNoise(sigma=1)
				obs_noise_theta = NormalNoise(sigma=0.15)
				obs_noise_v = NormalNoise(sigma=1)
				obs_noise_steer = NormalNoise(sigma=0.1)
				#ctrl_noise = utils.NormalNoise(sigma=args.ctrl_noise_sigma)
				#action = controller_policy.select_action(state, subgoal)
        		#action = ctrl_noise.perturb_action(action, -max_action, max_action)

			# Stop gradient for subgoal goal and next state
			if self.use_decoder:
				environment_state = state.clone().detach()
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
			if self.safety:
				target_Q_cost = self.critic_cost_target(next_state, next_action, goal)
				if self.test_max_critic:
					target_Q_cost = torch.max(target_Q_cost, -1, keepdim=True)[0]
				else:
					target_Q_cost = torch.min(target_Q_cost, -1, keepdim=True)[0]
				target_Q_cost = cost + (1.0-done) * self.gamma*target_Q_cost

		# Compute critic loss
		Q = self.critic(state, action, goal)
		critic_loss = 0.5 * (Q - target_Q).pow(2).sum(-1).mean()
		if self.safety and self.test_case_soft_critic:
			Q_cost = self.critic_cost(state, action, goal)
			critic_cost_loss = 0.5 * (Q_cost - target_Q_cost).pow(2).sum(-1).mean()
			critic_loss += critic_cost_loss

		# Optimize the critic
		if self.use_encoder: self.encoder_optimizer.zero_grad()
		self.critic_optimizer.zero_grad()
		if self.safety and self.test_case_soft_critic: self.critic_cost_optimizer.zero_grad()
		critic_loss.backward()
		if self.use_encoder: self.encoder_optimizer.step()
		self.critic_optimizer.step()
		if self.safety and self.test_case_soft_critic: self.critic_cost_optimizer.step()

		# Optimize autoencoder
		if self.use_decoder:
			y = self.encoder.autoencoder_forward(environment_state)
			autoencoder_loss = self.autoencoder_criterion(environment_state, y)
			self.autoencoder_optimizer.zero_grad()
			autoencoder_loss.backward()
			self.autoencoder_optimizer.step()

		if self.logger is not None:
			self.logger.store(
				critic_value   = Q.mean().item(),
				target_value  = target_Q.mean().item()
			)

		# Stop backpropagation to encoder
		if self.use_encoder:
			state = state.detach()
			goal = goal.detach()
			subgoal = subgoal.detach()

		""" Safety Critic """
		if self.safety and not self.test_case_soft_critic:
			# Compute safety critic loss
			Q_cost = self.critic_cost(state, action, goal)
			critic_cost_loss = 0.5 * (Q_cost - target_Q_cost).pow(2).sum(-1).mean()

			# Optimize the safety critic
			self.critic_cost_optimizer.zero_grad()
			critic_cost_loss.backward()
			self.critic_cost_optimizer.step()

		if self.safety:
			if self.logger is not None:
				self.logger.store(
					safety_critic_value   = Q_cost.mean().item(),
					safety_target_value   = target_Q_cost.mean().item(),
					critic_cost_loss      = critic_cost_loss.item(),
				)

		""" High-level policy learning """
		if self.curriculum_high_policy:
			if self.stop_train_high_policy:
				if self.logger is not None:
					self.logger.store(
						train_subgoal_x_max = 0,
						train_subgoal_x_mean = 0,
						train_subgoal_x_min = 0,
						train_subgoal_y_max = 0,
						train_subgoal_y_mean = 0,
						train_subgoal_y_min = 0,
					)
					if self.logger is not None:
						self.logger.data["adv"] = [0]
						self.logger.data["ratio_adv"] = [0]
						self.logger.data["subgoal_loss"] = [0]
						self.logger.data["high_policy_v"] = [0]
						self.logger.data["high_v"] = [0]
			else:
				self.train_highlevel_policy(state, goal, subgoal)
		else:
			self.train_highlevel_policy(state, goal, subgoal)

		""" Actor """
		# Sample action
		action, D_KL = self.sample_action_and_KL(state, goal)

		if self.safety:
			# Compute actor loss + safety
			Q = self.critic(state, action, goal)
			Q = torch.min(Q, -1, keepdim=True)[0]
			lambda_multiplier = torch.nn.functional.softplus(self.lambda_coefficient)
			Q_cost = self.critic_cost(state, action, goal)
			if self.test_max_critic:
				Q_cost = lambda_multiplier * torch.max(Q_cost, -1, keepdim=True)[0]
			else:
				Q_cost = lambda_multiplier * torch.min(Q_cost, -1, keepdim=True)[0]
			actor_loss = (self.alpha*D_KL - Q + Q_cost).mean()
		else:
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
			if self.safety:
				for param, target_param in zip(self.critic_cost.parameters(), self.critic_cost_target.parameters()):
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
				D_KL		 = D_KL.mean().item(),
				alpha        = self.alpha,		

			)
