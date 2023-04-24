import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.distributions import MultivariateNormal
import gym
import numpy as np
import os
from .models import ActorCritic, Lyambda


class RIS_PPO:
    def __init__(self, state_dim, goal_dim, action_dim, args, action_space_high):
        self.lr = args.lr
        self.gamma = args.gamma
        self.eps_clip = args.eps_clip
        self.K_epochs = args.K_epochs
        self.constrained_ppo = args.constrained_ppo
        self.cost_limit = args.cost_limit
        self.penalty_init = args.penalty_init
        self.penalty_lr = args.penalty_lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_penalty = torch.tensor(args.max_penalty, dtype=torch.float32).to(self.device)
        self.min_penalty = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        self.penalty_clip = torch.tensor(-1, dtype=torch.float32).to(self.device)
        
        #self.policy = ActorCritic(state_dim, action_dim, args, self.device, action_space_high)
        self.policy = ActorCritic(state_dim + goal_dim, action_dim, args, self.device, action_space_high)
        self.actor_optimizer = torch.optim.Adam(self.policy.action_layer.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.policy.value_layer.parameters(), lr=self.lr)
        #self.policy_old = ActorCritic(state_dim, action_dim, args, self.device, action_space_high)
        self.policy_old = ActorCritic(state_dim + goal_dim, action_dim, args, self.device, action_space_high)
        self.policy_old.action_layer.load_state_dict(self.policy.action_layer.state_dict())
        self.policy_old.value_layer.load_state_dict(self.policy.value_layer.state_dict())
        self.MseLoss = nn.MSELoss()

        if self.constrained_ppo:
            self.const_critic_optimizer = torch.optim.Adam(self.policy.const_value_layer.parameters(), lr=self.lr)
            self.policy_old.const_value_layer.load_state_dict(self.policy.const_value_layer.state_dict())
            self.lagrange = Lyambda(self.penalty_init, self.device).to(self.device)
            self.lagrange_optimizer = torch.optim.Adam(self.lagrange.parameters(), lr=self.penalty_lr)

    def update(self, memory, penalizing_ppo):
        memory.rewards_monte_carlo(self.gamma)
        memory.shuffle()
        rewards = memory.rewards
        rewards = (rewards - rewards.mean()) / (rewards + 1.0 ** -5).std()
        constrained_costs = memory.constrained_costs
        centered_constrained_costs = constrained_costs - constrained_costs.mean()
        old_obs = memory.obs.detach()
        old_goal = memory.goal.detach()
        old_actions = memory.actions.detach()
        old_logprobs = memory.logprobs.detach()

        total_loss = 0
        total_policy_loss = 0
        total_mse_loss = 0
        total_cmse_loss = 0
        total_dist_entropy = 0
        total_penalty_loss = 0
        c_mse = 0
        stats_log = {}
        # Adding gradient ascent for langrange multiplier
        penalty = 0.0
        if self.constrained_ppo:
            penalty_loss = -self.lagrange.penalty_param * (constrained_costs.mean() - self.cost_limit)
            self.lagrange_optimizer.zero_grad()
            penalty_loss.mean().backward()
            self.lagrange_optimizer.step()
            total_penalty_loss = penalty_loss

        lst_actions_mean = []
        lst_std = []
        total_surr_penalty_loss = 0.
        surr_const_loss = 0.
        
        for _ in range(self.K_epochs):
            #logprobs, state_values, const_state_value, dist_entropy, action_stats = self.policy.evaluate(old_obs, old_actions)
            logprobs, state_values, const_state_value, dist_entropy, action_stats = self.policy.evaluate(torch.cat((old_obs, old_goal), 1), old_actions)
            lst_actions_mean.append(action_stats["action_mean"].detach().numpy())
            lst_std.append(action_stats["logstd"].detach().numpy())
            
            mse = 0.5 * self.MseLoss(state_values, rewards)
            self.critic_optimizer.zero_grad()
            mse.mean().backward()
            self.critic_optimizer.step()
            
            if self.constrained_ppo:
                c_advantages = centered_constrained_costs - const_state_value.detach()
                c_mse = 0.5 * self.MseLoss(const_state_value, centered_constrained_costs)
                self.const_critic_optimizer.zero_grad()
                c_mse.mean().backward()
                self.const_critic_optimizer.step()

            ratios =  torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            clamp_ratios = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
            surr2 = clamp_ratios * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            loss = policy_loss - 0.01 * dist_entropy

            if self.constrained_ppo:
                penalty = self.lagrange.penalty()
                if penalizing_ppo:
                    penalty = torch.clamp(penalty, self.min_penalty, self.max_penalty)
                    surr_const_loss = (clamp_ratios * c_advantages).mean()
                    surr_const_loss = penalty * surr_const_loss
                    loss += surr_const_loss
                    loss /= (1 + penalty)
            
            self.actor_optimizer.zero_grad()
            loss.mean().backward()
            self.actor_optimizer.step()

            total_loss += loss
            total_mse_loss += mse
            total_cmse_loss += c_mse
            total_policy_loss += policy_loss
            total_dist_entropy += dist_entropy
            total_surr_penalty_loss += surr_const_loss

        # копируем веса
        self.policy_old.value_layer.load_state_dict(self.policy.value_layer.state_dict())
        self.policy_old.action_layer.load_state_dict(self.policy.action_layer.state_dict())
        if self.constrained_ppo:
            self.policy_old.const_value_layer.load_state_dict(self.policy.const_value_layer.state_dict())

        stats_log["total_loss"] = total_loss / self.K_epochs
        stats_log["policy_loss"] = total_policy_loss / self.K_epochs
        stats_log["mse_loss"] = total_mse_loss / self.K_epochs
        stats_log["cmse_loss"] = total_cmse_loss / self.K_epochs
        stats_log["dist_entropy"] = total_dist_entropy / self.K_epochs
        stats_log["penalty_surr_loss"] = total_surr_penalty_loss / self.K_epochs
        
        stats_log["penalty_loss"] = total_penalty_loss
        stats_log["constrained_costs"] = constrained_costs.mean()
        stats_log["lagrange_multiplier"] = penalty
        stats_log["action_means"] = np.mean(lst_actions_mean, 0)
        stats_log["logstd"] = np.mean(lst_std, 0)
        
        return stats_log
        #, np.mean(np.array(lst_dist_entropy))

    def get_action(self, state, deterministic=False):
        #state = torch.FloatTensor(state).to(device)
        self.eval()
        action, _ = self.policy.act(state, deterministic=deterministic)

        return action.detach().cpu().numpy()
    
    def eval(self):
        self.policy.eval()
        self.policy_old.eval()

    def save(self, folder_path):
        if folder_path is not None:
            folder_path = os.path.join('custom_train_dir', folder_path)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            self.policy.save(folder_path)
            
    def load(self, folder_path):
        if folder_path is not None:
            folder_path = os.path.join('custom_train_dir', folder_path)
            self.policy.load(folder_path)