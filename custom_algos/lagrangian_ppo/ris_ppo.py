import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.distributions import MultivariateNormal
import gym
import numpy as np
import os
import torch.nn.functional as F
from .goal_models import ActorCritic, Lyambda, LaplacePolicy


class RIS_PPO:
    def __init__(self, state_dim, goal_dim, action_dim, args, action_space_high, h_lr=1e-4, alpha=0.1, Lambda=0.1, n_ensemble=10, epsilon=1e-16, save_path=None):
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
        self.policy_old_temp = ActorCritic(state_dim + goal_dim, action_dim, args, self.device, action_space_high)
        self.policy_old_temp.action_layer.load_state_dict(self.policy.action_layer.state_dict())
        self.policy_old_temp.value_layer.load_state_dict(self.policy.value_layer.state_dict())

        self.MseLoss = nn.MSELoss()

        if self.constrained_ppo:
            self.const_critic_optimizer = torch.optim.Adam(self.policy.const_value_layer.parameters(), lr=self.lr)
            self.policy_old.const_value_layer.load_state_dict(self.policy.const_value_layer.state_dict())
            self.lagrange = Lyambda(self.penalty_init, self.device).to(self.device)
            self.lagrange_optimizer = torch.optim.Adam(self.lagrange.parameters(), lr=self.penalty_lr)

        self.subgoal_net = LaplacePolicy(state_dim).to(self.device)
        self.subgoal_optimizer = torch.optim.Adam(self.subgoal_net.parameters(), lr=h_lr)

        self.epsilon = epsilon
        
        # High-level policy hyperparameters
        self.alpha = alpha
        self.Lambda = Lambda
        self.n_ensemble = n_ensemble

        self.save_path = save_path

    def sample_subgoal(self, state, goal):
        subgoal_distribution = self.subgoal_net(state, goal)
        subgoal = subgoal_distribution.rsample((self.n_ensemble,))
        subgoal = torch.transpose(subgoal, 0, 1)
        return subgoal

    def evluate_and_sample_DL(self, old_obs, old_goal, old_actions):
        logprobs, state_values, const_state_value, dist_entropy, action_stats = self.policy.evaluate(torch.cat((old_obs, old_goal), 1), old_actions)

        #new_action, _ = self.policy.act(torch.cat((old_obs, old_goal), 1), to_device=False)
        #assert new_action.shape[1] == 4, "should be batch_size/2 x 4"
        #new_action = torch.cat((new_action[:, 0:2], new_action[:, 2:]), 0)
        #assert new_action.shape == old_actions.shape
        #new_logprobs, _, _, _, _ = self.policy.evaluate(torch.cat((old_obs, old_goal), 1), new_action)
        with torch.no_grad():
            subgoal = self.sample_subgoal(old_obs, old_goal)
        sum_old_probs = torch.zeros((subgoal.size(0))).to(self.device)
        for j in range(self.n_ensemble):
            subgoal_j = subgoal[:, j, :]
            old_logprobs, _, _, _, _ = self.policy_old.evaluate(torch.cat((old_obs, subgoal_j), 1), old_actions)
            #old_logprobs, _, _, _, _ = self.policy_old.evaluate(torch.cat((old_obs, subgoal_j), 1), new_action)
            #old_logprobs, _, _, _, _ = self.policy_old_temp.evaluate(torch.cat((old_obs, subgoal_j), 1), old_actions)
            old_probs = old_logprobs.exp()
            sum_old_probs += old_probs
        sum_old_probs /= self.n_ensemble
        sum_old_logprobs = torch.log(sum_old_probs + self.epsilon)
        D_KL = logprobs - sum_old_logprobs
        #D_KL = new_logprobs - sum_old_logprobs

        
        if abs(logprobs.mean().item()) > 1000 or abs(sum_old_logprobs.mean().item()) > 1000:

            print("#########################")
            print("BUG WITH LOGPPROB:")

            print("logprobs mean:", logprobs.mean().item())
            print("logprobs max:", logprobs.max().item())
            print("logprobs min:", logprobs.min().item())
            
            print("old logprobs mean:", sum_old_logprobs.mean().item())
            print("old logprobs max:", sum_old_logprobs.max().item())
            print("old logprobs min:", sum_old_logprobs.min().item())

            print("old action max acc:", old_actions[:, 0].max().item())
            print("old action min acc:", old_actions[:, 0].min().item())
            print("old action max steer:", old_actions[:, 1].max().item())
            print("old action min steer:", old_actions[:, 1].min().item())
            #print("action stats:", action_stats["action_dist"][logprobs.argmin().item(), :])
            print("action dist means:", action_stats["action_dist"].loc[logprobs.argmin().item(), :])
            print("action dist std:", action_stats["action_dist"].scale[logprobs.argmin().item(), :])

            print("argmin logprob state:", old_obs[logprobs.argmin().item(), :])
            print("argmin logprob goal:", old_goal[logprobs.argmin().item(), :])
            print("argmin logprob action:", old_actions[logprobs.argmin().item(), :])
            print("argmin logprob subgoal:", subgoal[logprobs.argmin().item(), :][0])
                
            #if not(self.save_path is None):
            #    self.save(self.save_path)
            #else:
            self.save("./" + "temp_checkpoint_debug_1")

            assert 1 == 0
            
        
        #action_stats["ppo_actor_new_logprobs"] = new_logprobs.mean().item()
        action_stats["ppo_actor_new_logprobs"] = 0
        action_stats["ppo_actor_logprobs"] = logprobs.mean().item()
        action_stats["ppo_actor_logprobs_min"] = logprobs.min().item()
        action_stats["ppo_oldactor_logprobs"] = sum_old_logprobs.mean().item()


        # копируем веса
        #self.policy_old_temp.value_layer.load_state_dict(self.policy.value_layer.state_dict())
        #self.policy_old_temp.action_layer.load_state_dict(self.policy.action_layer.state_dict())

        return logprobs, state_values, const_state_value, dist_entropy, action_stats, D_KL

    def update_high_level_policy(self, memory):
        stats_log = {}
        
        obs, goal = memory.sample_batch()
        subgoal = memory.random_state_batch()
        #obs = memory.obs.detach()
        #goal = memory.goal.detach()
        #subgoal = obs[torch.randperm(obs.size()[0])]

        obs = obs.to(self.device)
        goal = goal.to(self.device)
        subgoal = subgoal.to(self.device)

        subgoal_distribution = self.subgoal_net(obs, goal)

        with torch.no_grad():
            # Compute target value
            new_subgoal = subgoal_distribution.loc

            policy_v_1 = self.policy.value_layer(torch.cat((obs, new_subgoal), -1))
            policy_v_2 = self.policy.value_layer(torch.cat((new_subgoal, goal), -1))
            policy_v = torch.cat([policy_v_1, policy_v_2], -1).clamp(min=-100.0, max=0.0).abs().max(-1)[0]

			# Compute subgoal distance loss
            v_1 = self.policy.value_layer(torch.cat((obs, subgoal), -1))
            v_2 = self.policy.value_layer(torch.cat((subgoal, goal), -1))
            v = torch.cat([v_1, v_2], -1).clamp(min=-100.0, max=0.0).abs().max(-1)[0]
            adv = - (v - policy_v)
            weight = F.softmax(adv/self.Lambda, dim=0)

        log_prob = subgoal_distribution.log_prob(subgoal).sum(-1)
        subgoal_loss = - (log_prob * weight).mean()

        # Update network
        self.subgoal_optimizer.zero_grad()
        subgoal_loss.backward()
        self.subgoal_optimizer.step()

        adv = adv.mean().item()
        subgoal_loss = subgoal_loss.item()
        high_policy_v = policy_v.mean().item()
        high_v = v.mean().item()
  
        stats_log["adv"] = adv
        stats_log["subgoal_loss"] = subgoal_loss
        stats_log["high_policy_v"] = high_policy_v
        stats_log["high_v"] = high_v

        train_subgoal = {"x_max": new_subgoal[:, 0].max().item(),
						 "x_mean": new_subgoal[:, 0].mean().item(), 
						 "x_min": new_subgoal[:, 0].min().item(),
						 "y_max": new_subgoal[:, 1].max().item(),
						 "y_mean": new_subgoal[:, 1].mean().item(), 
						 "y_min": new_subgoal[:, 1].min().item(),
						 }

        train_subgoal_data = {"x_max": subgoal[:, 0].max().item(),
							  "x_mean": subgoal[:, 0].mean().item(), 
							  "x_min": subgoal[:, 0].min().item(),
							  "y_max": subgoal[:, 1].max().item(),
							  "y_mean": subgoal[:, 1].mean().item(), 
							  "y_min": subgoal[:, 1].min().item(),
							 }

        stats_log["H_subgoal_x_max"] = train_subgoal["x_max"]
        stats_log["H_subgoal_x_mean"] = train_subgoal["x_mean"]
        stats_log["H_subgoal_x_min"] = train_subgoal["x_min"]
        stats_log["H_sampled_subgoal_x_max"] = train_subgoal_data["x_max"]
        stats_log["H_sampled_subgoal_x_mean"] = train_subgoal_data["x_mean"]
        stats_log["H_sampled_subgoal_x_min"] = train_subgoal_data["x_min"]

        return stats_log

    def update_low_level_policy(self, memory, penalizing_ppo):
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
        total_state_values = 0
        total_D_KL = 0
        actor_new_logprobs = 0
        actor_new_logprobs_min = 0
        actor_logprobs = 0
        oldactor_logprobs = 0
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
            logprobs, state_values, const_state_value, dist_entropy, action_stats, D_KL = self.evluate_and_sample_DL(old_obs, old_goal, old_actions)
            #logprobs, state_values, const_state_value, dist_entropy, action_stats = self.policy.evaluate(torch.cat((old_obs, old_goal), 1), old_actions)
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
            #policy_loss = -torch.min(surr1, surr2).mean()
            policy_loss = -torch.min(surr1, surr2)
            policy_loss += self.alpha*D_KL
            policy_loss = policy_loss.mean()
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

            # low level policy grad normlization
            #torch.nn.utils.clip_grad_norm_(self.policy.value_layer.parameters(), 5)

            self.actor_optimizer.step()

            total_loss += loss
            total_mse_loss += mse
            total_cmse_loss += c_mse
            total_policy_loss += policy_loss
            total_dist_entropy += dist_entropy
            total_surr_penalty_loss += surr_const_loss
            total_state_values += state_values.mean().item()
            total_D_KL += D_KL.mean().item()

            actor_new_logprobs += action_stats["ppo_actor_new_logprobs"]
            actor_new_logprobs_min += action_stats["ppo_actor_logprobs_min"]
            actor_logprobs += action_stats["ppo_actor_logprobs"]
            oldactor_logprobs += action_stats["ppo_oldactor_logprobs"]

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
        stats_log["state_values"] = total_state_values / self.K_epochs
        stats_log["D_KL"] = total_D_KL / self.K_epochs
        stats_log["actor_new_logprobs"] = actor_new_logprobs / self.K_epochs
        stats_log["actor_new_logprobs_min"] = actor_new_logprobs_min / self.K_epochs
        stats_log["actor_logprobs"] = actor_logprobs / self.K_epochs
        stats_log["oldactor_logprobs"] = oldactor_logprobs / self.K_epochs
        
        stats_log["penalty_loss"] = total_penalty_loss
        stats_log["constrained_costs"] = constrained_costs.mean()
        stats_log["lagrange_multiplier"] = penalty
        stats_log["action_means"] = np.mean(lst_actions_mean, 0)
        stats_log["logstd"] = np.mean(lst_std, 0)
        
        return stats_log
        #, np.mean(np.array(lst_dist_entropy))

    def get_action(self, state, goal, deterministic=False):
        #state = torch.FloatTensor(state).to(device)
        self.eval()
        state = np.concatenate([state, goal], axis=0)
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