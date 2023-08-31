import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import json

class Buffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.wins = []
        self.outcomes = []

    def append(self, state, action, reward, next_state, done, win, outcome):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.wins.append(win)
        self.outcomes.append(outcome)
    
    def numpy(self):
        return np.array(self.states), np.array(self.actions), np.array(self.rewards), np.array(self.next_states)
    
    def sample(self, batch_size):
        if batch_size > len(self.states):
            self.numpy()
        else:
            inds = np.random.choice(len(self.states), batch_size, replace = False)
            return np.array([self.states[idx] for idx in inds]), np.array([self.actions[idx] for idx in inds]), \
                np.array([self.rewards[idx] for idx in inds]), np.array([self.next_states[idx] for idx in inds]), np.array([self.dones[idx] for idx in inds], dtype=bool)

class Agent:
    def __init__(self, rng, name):
        self.rng = rng
        self.name = name
        self.context_dim = None
        self.step = 0

    def newdata(self, s, a, r, s_, done, win, outcome):
        '''
        This function is called at every step.

        Inputs
        s[numpy array] : state   a[float] : action  r[float] : reward
        s_[numpy array] : next state    done[bool] : the episode is done
        win[float; zero or one] : the agent won the auction at the step
        outcome[int; zero or one] : the conversion event occured at the step(this can be 1 even if the agent lost the auction)

        This function returns nothing
        '''
        raise NotImplementedError

    def bid(self, state):
        '''
        This function is called at every step, even if the agent has no budget.

        Input
        s[numpy array] : state

        Output : bidding[float] (bidding will be clipped in main.py not to exceed current budget)
        '''
        raise NotImplementedError
    
    def update(self):
        '''
        This function is called at the end of episodes.
        '''
        raise NotImplementedError

    
    
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, scale):
        super().__init__()
        self.input_dim = in_features
        self.output_dim = out_features
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scale = scale
        
        self.weight_mu = nn.Parameter(torch.empty((out_features, in_features)))
        self.weight_rho = nn.Parameter(torch.empty((out_features, in_features)))
        
        self.bias_mu = nn.Parameter(torch.empty((out_features,)))
        self.bias_rho = nn.Parameter(torch.empty((out_features,)))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mu)
        nn.init.uniform_(self.weight_rho, -0.02, 0.02)
        nn.init.uniform_(self.bias_mu, -np.sqrt(3/self.weight_mu.size(1)), np.sqrt(3/self.weight_mu.size(1)))
        nn.init.uniform_(self.bias_rho, -0.02, 0.02)
        
    def forward(self, input, sample=False, pre_sampled=False):
        if sample:
            weight_sigma = torch.log1p(torch.exp(self.weight_rho))
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu)
            bias = self.bias_mu + bias_sigma * torch.randn_like(self.bias_mu)
        elif pre_sampled:
            weight = self.weight_
            bias = self.bias_
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(input, weight, bias)
    
    def get_uncertainty(self):
        with torch.no_grad():
            weight_sigma = torch.log1p(torch.exp(self.weight_rho)).numpy(force=True)
            bias_sigma = torch.log1p(torch.exp(self.bias_rho)).numpy(force=True)
        return np.concatenate([weight_sigma.reshape(-1), bias_sigma.reshape(-1)])
    
    def sample_weight(self):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        self.weight_ = (self.weight_mu + self.scale * weight_sigma * torch.randn_like(self.weight_mu)).detach().clone()
        self.bias_ = (self.bias_mu + self.scale * bias_sigma * torch.randn_like(self.bias_mu)).detach().clone()

class NoisyCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, context_dim, var_scale):
        super().__init__()
        self.context_dim = context_dim
        self.fc1 = BayesianLinear(input_dim, hidden_dim-3, var_scale)
        self.fc2 = BayesianLinear(hidden_dim, hidden_dim-3, var_scale)
        self.fc3 = BayesianLinear(hidden_dim, 1, var_scale)

        self.fc4 = BayesianLinear(input_dim, hidden_dim-3, var_scale)
        self.fc5 = BayesianLinear(hidden_dim, hidden_dim-3, var_scale)
        self.fc6 = BayesianLinear(hidden_dim, 1, var_scale)
    
    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x, sample=True):
        r = x[:, self.context_dim:]
        
        q1 = self.swish(self.fc1(x, sample))
        q1 = self.swish(self.fc2(torch.cat([q1, r], dim=1), sample))
        q1 = self.fc3(torch.cat([q1, r], dim=1), sample)

        q2 = self.swish(self.fc4(x, sample))
        q2 = self.swish(self.fc5(torch.cat([q2, r], dim=1), sample))
        q2 = self.fc6(torch.cat([q2, r], dim=1), sample)

        return q1, q2
    
    def Q1(self, x, sample = False, pre_sampled=False):
        r = x[:, self.context_dim:]
        q1 = self.swish(self.fc1(x, sample, pre_sampled))
        q1 = self.swish(self.fc2(torch.cat([q1, r], dim=1), sample, pre_sampled))
        return self.fc3(torch.cat([q1, r], dim=1), sample, pre_sampled)
    
    
    def get_uncertainty(self):
        u = np.concatenate([self.fc1.get_uncertainty(), self.fc2.get_uncertainty(), self.fc3.get_uncertainty()])
        return np.mean(u)
    
    def sample_net(self):
        self.fc1.sample_weight()
        self.fc2.sample_weight()
        self.fc3.sample_weight()
        self.fc4.sample_weight()
        self.fc5.sample_weight()
        self.fc6.sample_weight()

class Shin(Agent):
    def __init__(self, rng, name, context_dim):
        super().__init__(rng, name)
        self.context_dim = context_dim
        self.buffer = Buffer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open('src/agents/Shin/config.json') as f:
            config = json.load(f)
        # -------------------교체---
        # self.local_network = Critic(context_dim + 2 + 1, config['hidden_dim'], self.context_dim).to(self.device)
        # self.target_network = copy.deepcopy(self.local_network)
        # self.eps_init = self.eps = config['eps_init']
        # self.eps_min = config['eps_min']
        # self.eps_decay = config['eps_decay']

        self.local_network = NoisyCritic(context_dim + 2 + 1, config['hidden_dim'], self.context_dim, config['var_scale']).to(self.device)
        self.target_network = copy.deepcopy(self.local_network)
        # --------------------------

        self.optimizer = optim.Adam(self.local_network.parameters(), lr = config['lr'])
        self.batch_size = config['batch_size']
        self.tau = config['tau']
        self.num_grad_steps = config['num_grad_steps']
        self.episode = 0

    def set_exploration_param(self, ep):
        self.ep = ep
        self.local_network.sample_net()
    
    def newdata(self, s, a, r, s_, done, win, outcome):
        self.buffer.append(s,a,r,s_,done,win,outcome)

    def bid(self, state):
        self.step += 1
        n_values_search = 10
        b_grid = np.linspace(0, 1, n_values_search)
        x = torch.Tensor(np.concatenate([np.tile(state, (n_values_search, 1)),b_grid.reshape(-1,1)], axis=1)).to(self.device)
        with torch.no_grad():
        # ----------------교체---------
        #     if self.rng.uniform(0, 1) < self.eps:
        #         bidding = self.rng.random()
        #     else:
        #         index = np.argmax(self.local_network.Q1(x).numpy(force=True))
        #         bidding = b_grid[index]
        # return bidding
        
            index = np.argmax(self.local_network.Q1(x, sample=True, pre_sampled=False).numpy(force=True))
            bidding = b_grid[index]
        return bidding
        # -----------------------------
    def update(self):
        self.episode += 1
        
        if len(self.buffer.states)<self.batch_size:
            return
        
        criterion = nn.MSELoss()
        self.local_network.train()
        self.target_network.eval()

        for i in range(self.num_grad_steps):
            states, biddings, rewards, next_states, dones = self.buffer.sample(self.batch_size)
            q1, q2 = self.local_network(torch.Tensor(np.hstack([states, biddings.reshape(-1, 1)])).to(self.device))
            with torch.no_grad():
                n_values_search = 10
                b_grid = np.linspace(0, 1, n_values_search)
                x = torch.Tensor(np.concatenate([np.tile(next_states, (1, n_values_search)).reshape(-1,self.context_dim+2),
                                np.tile(b_grid.reshape(-1,1), (self.batch_size,1))], axis=1)).to(self.device)
                # next_q1, next_q2  = self.target_network(x)
                next_q1, next_q2  = self.target_network(x, sample=False)
                target_q1 = torch.max(next_q1.reshape(self.batch_size,-1), dim=1, keepdim=False).values
                target_q2 = torch.max(next_q2.reshape(self.batch_size,-1), dim=1, keepdim=False).values
                target_q = torch.Tensor(rewards).to(self.device) + torch.Tensor(1.0-dones).to(self.device) * torch.min(target_q1, target_q2)
                            
            loss = criterion(q1.squeeze(), target_q) + criterion(q2.squeeze(), target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.episode%2 == 0:
            for target_param, local_param in zip(self.target_network.parameters(), self.local_network.parameters()):
                target_param.data.copy_(self.tau * local_param + (1-self.tau) * target_param)
    
    def get_uncertainty(self, len):
        return self.local_network.get_uncertainty()