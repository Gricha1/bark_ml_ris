import numpy as np
import torch
import torch.nn.functional as F

from custom_Models import GaussianPolicy, EnsembleCritic, LaplacePolicy, Encoder, obs_Encoder, goal_Encoder

from utils.data_aug import random_translate


import torch.nn as nn

def get_img_feat_from_rollout(state, c_count=2, batch_size=1):
	x = state[:, :-5]
	x_f = state[:, -5:]
	x = x.view(batch_size, c_count, 120, 120)
	x_f = x_f.view(batch_size, -1)
	return x, x_f


class RIS(object):
	def __init__(self, state_dim, action_dim, alpha=0.1, 
				 Lambda=0.1, image_env=False, n_ensemble=10, 
				 gamma=0.99, tau=0.005, target_update_interval=1, 
				 h_lr=1e-4, q_lr=1e-3, pi_lr=1e-4, enc_lr=1e-4, 
				 epsilon=1e-16, logger=None, device=torch.device("cuda"), max_env_steps=700.0):		
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
			#self.obs_Encoder = obs_Encoder(state_dim=state_dim).to(device)
			self.goal_Encoder = goal_Encoder(state_dim=state_dim).to(device)
			#self.obs_encoder_optimizer = torch.optim.Adam(self.obs_Encoder.parameters(), lr=enc_lr)
			self.goal_encoder_optimizer = torch.optim.Adam(self.goal_Encoder.parameters(), lr=enc_lr)

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
		self.max_env_steps = max_env_steps

	def save(self, folder, save_optims=False):
		torch.save(self.actor.state_dict(),		 folder + "actor.pth")
		torch.save(self.critic.state_dict(),		folder + "critic.pth")
		torch.save(self.subgoal_net.state_dict(),   folder + "subgoal_net.pth")
		if self.image_env:
			#torch.save(self.obs_Encoder.state_dict(), folder + "obs_encoder.pth")
			torch.save(self.goal_Encoder.state_dict(), folder + "goal_encoder.pth")
		if save_optims:
			torch.save(self.actor_optimizer.state_dict(), 	folder + "actor_opti.pth")
			torch.save(self.critic_optimizer.state_dict(), 	folder + "critic_opti.pth")
			torch.save(self.subgoal_optimizer.state_dict(), folder + "subgoal_opti.pth")
			if self.image_env:
				#torch.save(self.obs_encoder_optimizer.state_dict(), folder + "obs_encoder_opti")
				torch.save(self.goal_encoder_optimizer.state_dict(), folder + "goal_encoder_opti")

	def load(self, folder):
		self.actor.load_state_dict(torch.load(folder+"actor.pth", map_location=self.device))
		self.critic.load_state_dict(torch.load(folder+"critic.pth", map_location=self.device))
		self.subgoal_net.load_state_dict(torch.load(folder+"subgoal_net.pth", map_location=self.device))
		if self.image_env:
			#self.obs_Encoder.load_state_dict(torch.load(folder+"obs_encoder.pth", map_location=self.device))
			self.goal_Encoder.load_state_dict(torch.load(folder+"goal_encoder.pth", map_location=self.device))

	def select_action(self, state, goal):

		with torch.no_grad():
			state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
			goal = torch.FloatTensor(goal).to(self.device).unsqueeze(0)
			if self.image_env:
				#x, x_f = get_img_feat_from_rollout(state, c_count=2)
				x, x_f = get_img_feat_from_rollout(state, c_count=1)
				#state = self.obs_Encoder(x, x_f)
				state = self.goal_Encoder(x, x_f)

				x_goal, x_f_goal = get_img_feat_from_rollout(goal, c_count=1)
				goal = self.goal_Encoder(x_goal, x_f_goal)

			# debug
			action, _, _ = self.actor.sample(state, goal)
			#_, _, mean = self.actor.sample(state, goal)

		return action.cpu().data.numpy().flatten()
		#return mean.cpu().data.numpy().flatten()
		
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

			# debug
			#policy_v = torch.cat([policy_v_1, policy_v_2], -1).clamp(min=-100.0, max=0.0).abs().max(-1)[0]
			policy_v = torch.cat([policy_v_1, policy_v_2], -1).clamp(min=-self.max_env_steps, max=0.0).abs().max(-1)[0]

			# Compute subgoal distance loss
			v_1 = self.value(state, subgoal)
			v_2 = self.value(subgoal, goal)

			# debug
			#v = torch.cat([v_1, v_2], -1).clamp(min=-100.0, max=0.0).abs().max(-1)[0]
			v = torch.cat([v_1, v_2], -1).clamp(min=-self.max_env_steps, max=0.0).abs().max(-1)[0]

			adv = - (v - policy_v)
			weight = F.softmax(adv/self.Lambda, dim=0)
		
		# debug
		#print("subgoal v_1(state-sub): ", v_1)
		#print("gen subgoal v_1(state-sub):", policy_v_1)

		log_prob = subgoal_distribution.log_prob(subgoal).sum(-1)
		subgoal_loss = - (log_prob * weight).mean()

		# Update network
		self.subgoal_optimizer.zero_grad()
		subgoal_loss.backward()
		self.subgoal_optimizer.step()
		
		# Log variables
		if self.logger is not None:
			self.logger.store(
				adv = adv.mean().item(),
				ratio_adv = adv.ge(0.0).float().mean().item(),
			)

	def train(self, state, action, reward, next_state, done, goal, subgoal):
		""" Encode images (if vision-based environment), use data augmentation """
		
		if self.image_env:

			#x_state, x_f_state = get_img_feat_from_rollout(state, c_count=2, batch_size=state.shape[0])
			#x_next_state, x_f_next_state = get_img_feat_from_rollout(next_state, c_count=2, batch_size=next_state.shape[0])
			x_state, x_f_state = get_img_feat_from_rollout(state, c_count=1, batch_size=state.shape[0])
			x_next_state, x_f_next_state = get_img_feat_from_rollout(next_state, c_count=1, batch_size=next_state.shape[0])
			x_goal, x_f_goal = get_img_feat_from_rollout(goal, c_count=1, batch_size=goal.shape[0])
			#x_sub_goal, x_f_sub_goal = get_img_feat_from_rollout(subgoal, c_count=2, batch_size=subgoal.shape[0])
			x_sub_goal, x_f_sub_goal = get_img_feat_from_rollout(subgoal, c_count=1, batch_size=subgoal.shape[0])

			# Data augmentation
			#state = random_translate(state, pad=8)
			#next_state = random_translate(next_state, pad=8)
			#goal = random_translate(goal, pad=8)
			#subgoal = random_translate(subgoal, pad=8)

			# Stop gradient for subgoal goal and next state
			#state = self.obs_Encoder(x_state, x_f_state)
			state = self.goal_Encoder(x_state, x_f_state)
			with torch.no_grad():
				goal = self.goal_Encoder(x_goal, x_f_goal)
				#next_state = self.obs_Encoder(x_next_state, x_f_next_state)
				next_state = self.goal_Encoder(x_next_state, x_f_next_state)
				#subgoal = self.obs_Encoder(x_sub_goal, x_f_sub_goal)
				subgoal = self.goal_Encoder(x_sub_goal, x_f_sub_goal)
				


		# debug
		#print("state:", state[0])
		#print("next_state:", next_state[0])
		#print("goal:", goal[0])
		#print("subgoal:", subgoal[0])

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

		# debug
		#print("reward batch:", reward)


		# debug
		if self.logger is not None:
			self.logger.store(
				critic_value   = Q.mean().item(),
				target_value  = target_Q.mean().item()
			)
		#print("critic mean value:", Q.mean().item())
		#print("target mean value:", Q.mean().item())

		# Optimize the critic
		if self.image_env:
			#self.obs_encoder_optimizer.zero_grad()
			self.goal_encoder_optimizer.zero_grad()
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		if self.image_env:
			#self.obs_encoder_optimizer.zero_grad()
			self.goal_encoder_optimizer.zero_grad()
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

		# Log variables
		if self.logger is not None:
			self.logger.store(
				actor_loss   = actor_loss.item(),
				critic_loss  = critic_loss.item(),
				D_KL		 = D_KL.mean().item()				
			)
