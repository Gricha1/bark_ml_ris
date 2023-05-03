import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
import copy


""" High-level policy """

class LaplacePolicy(nn.Module):	
	def __init__(self, state_dim, hidden_dims=[256, 256]):	
		super(LaplacePolicy, self).__init__()	
		fc = [nn.Linear(2*state_dim, hidden_dims[0]), nn.ReLU()]
		for hidden_dim_in, hidden_dim_out in zip(hidden_dims[:-1], hidden_dims[1:]):
			fc += [nn.Linear(hidden_dim_in, hidden_dim_out), nn.ReLU()]
		self.fc = nn.Sequential(*fc)

		self.mean = nn.Linear(hidden_dims[-1], state_dim)	
		self.log_scale = nn.Linear(hidden_dims[-1], state_dim)	
		self.LOG_SCALE_MIN = -20	
		self.LOG_SCALE_MAX = 2	

	def forward(self, state, goal):	
		h = self.fc( torch.cat([state, goal], -1) )	
		mean = self.mean(h)
		scale = self.log_scale(h).clamp(min=self.LOG_SCALE_MIN, max=self.LOG_SCALE_MAX).exp()	
		distribution = torch.distributions.laplace.Laplace(mean, scale)
		return distribution


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, device, adaptive_std):
        super(Actor, self).__init__()
        self.device = device
        self.adaptive_std = adaptive_std
        if self.adaptive_std:
            action_dim *= 2
            action_dim = int(action_dim)
            self.stddev_min = 1e-5
            self.stddev_max = 1e1
        else:
            std_init = torch.zeros(action_dim, dtype=torch.float32)
            # # std_init[0] = np.log(action_space_high[0] / 3.)
            # # std_init[1] = np.log(action_space_high[1] / 3.)
            self.logstd = nn.Parameter(
                torch.tensor(std_init, dtype=torch.float32, device=self.device)
            )

        self.action_layer = nn.Sequential(
                            layer_init(nn.Linear(state_dim, hidden_size)),
                            nn.Tanh(),
                            layer_init(nn.Linear(hidden_size, hidden_size)),
                            nn.Tanh(),
                            layer_init(nn.Linear(hidden_size, action_dim), std=0.01),
                            )

    def forward(self, state):
        return self.action_layer(state)

    def get_corrected_std(self, action_logstd):
        stddevs = torch.exp(action_logstd)
        if self.adaptive_std:
            stddevs = torch.clamp(stddevs, self.stddev_min, self.stddev_max)
        
        return stddevs
        
    def evaluate(self, state, action):
        action_mean = self.action_layer(state)
        if not self.adaptive_std:
            action_logstd = self.logstd.expand_as(action_mean)
        else:
            action_mean, action_logstd = action_mean.chunk(2, dim=1)

        action_std = self.get_corrected_std(action_logstd)
        dist = Normal(action_mean, action_std)
        
        action_logprobs = dist.log_prob(action).sum(1)
        dist_entropy = dist.entropy().sum(1)


        # debug normalize actions to -1, +1
        #action_mean = torch.tanh(action_mean)

        return action_mean, action_logprobs, dist_entropy, action_std
    
    def act(self, state, deterministic=False, to_device=True):
        if to_device:
            state = torch.FloatTensor(state).to(self.device)
        action_mean = self.action_layer(state)
        if not self.adaptive_std:
            action_logstd = self.logstd
        else:
            action_mean, action_logstd = action_mean.chunk(2)
        action_std = self.get_corrected_std(action_logstd)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        # debug normalize actions to -1, +1
        #x_t = dist.sample()
        #action = torch.tanh(x_t)
        #log_prob = dist.log_prob(x_t)
        #log_prob -= torch.log((1 - action.pow(2)) + 1e-6)
        #log_prob = log_prob.sum(-1, keepdim=True)
        #action_mean = torch.tanh(action_mean)

        return action.detach() if not deterministic else action_mean.detach(), log_prob.detach()
    
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size, device):
        super(Critic, self).__init__()
        self.device = device        
        # critic
        self.value_layer =  nn.Sequential(
                            layer_init(nn.Linear(state_dim, hidden_size)),
                            nn.Tanh(),
                            layer_init(nn.Linear(hidden_size, hidden_size)),
                            nn.Tanh(),
                            layer_init(nn.Linear(hidden_size, 1), std=1.0)
                            )

    def forward(self, obs_act):
        return self.value_layer(obs_act)
    
class ActorCritic():
    def __init__(self, state_dim, action_dim, args, device, action_space_high):
        self.device = device
        self.constrained_ppo = args.constrained_ppo
        hidden_size = args.hidden_size
        adaptive_std = args.adaptive_std
        # actor
        self.action_layer = Actor(state_dim, action_dim, hidden_size, device, adaptive_std)
        self.action_layer.to(device=device)
        # critic
        self.value_layer =  Critic(state_dim, hidden_size, device)
        self.value_layer.to(device=device)
        # critic cost
        if self.constrained_ppo:
            self.const_value_layer = copy.deepcopy(self.value_layer)  
            self.const_value_layer.to(device=device)
                
    def forward(self):
        raise NotImplementedError

    def act(self, state, deterministic=False, to_device=True):
        return self.action_layer.act(state, deterministic, to_device=to_device)
        
    def evaluate(self, state, action):
        action_mean, action_logprobs, dist_entropy, action_std = self.action_layer.evaluate(state, action)
        state_value = self.value_layer(state)

        const_state_value = torch.zeros(1)
        if self.constrained_ppo:
            const_state_value = self.const_value_layer(state)

        action_stats = {}
        action_stats["action_mean"] = torch.mean(action_mean, 0).cpu()
        action_stats["logstd"] = torch.mean(action_std, 0).cpu()

        # debug RIS
        action_stats["action_dist"] = Normal(action_mean, action_std)

        return action_logprobs, torch.squeeze(state_value), torch.squeeze(const_state_value), dist_entropy.mean(), action_stats

    def save(self, folder_path):
        torch.save(self.value_layer.state_dict(), f'{folder_path}/critic.pkl')
        torch.save(self.action_layer.state_dict(), f'{folder_path}/actor.pkl')
        if self.constrained_ppo:
            torch.save(self.const_value_layer.state_dict(), f'{folder_path}/constraint_critic.pkl')
    
    def load(self, folder_path):
        self.value_layer.load_state_dict(torch.load(f'{folder_path}/critic.pkl'))
        self.action_layer.load_state_dict(torch.load(f'{folder_path}/actor.pkl'))
        if self.constrained_ppo:
            self.const_value_layer.load_state_dict(torch.load(f'{folder_path}/constraint_critic.pkl'))

    def eval(self):
        self.value_layer.eval()
        self.action_layer.eval()
        if self.constrained_ppo:
            self.const_value_layer.eval()

class Lyambda(nn.Module):
    def __init__(self, penalty_init, device):
        super(Lyambda, self).__init__()
        self.device = device
        penalty_init = penalty_init
        param_init = np.log(max(np.exp(penalty_init)-1, 1e-8))
        self.activation = nn.Softplus()
        self.penalty_param = nn.Parameter(torch.as_tensor(param_init))
        
    def penalty(self):
        penalty = self.activation(self.penalty_param)
        
        return penalty