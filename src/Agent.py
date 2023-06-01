import numpy as np
import torch.optim as optim
import copy

from models import *
from Allocator import *
from Bidder import *

def parse_kwargs(kwargs):
    parsed = ','.join([f'{key}={value}' for key, value in kwargs.items()])
    return ',' + parsed if parsed else ''

class Agent:
    def __init__(self, rng, name, item_features, item_values, context_dim, buffer):
        self.rng = rng
        self.name = name
        self.items = item_features
        self.item_values = item_values

        self.num_items = item_features.shape[0]
        self.feature_dim = item_features.shape[1]
        self.context_dim = context_dim

        self.buffer = buffer
        self.clock = 0

        self.auction = None

    def select_item(self, context):
        pass

    def bid(self, context):
        pass

class Bandit(Agent):
    ''' A bandit-style agent '''

    def __init__(self, rng, name, item_features, item_values, context_dim, buffer, config):
        super().__init__(rng, name, item_features, item_values, context_dim, buffer)

        self.allocator = eval(f"{config['allocator']['type']}(rng=rng, item_features=item_features, context_dim=context_dim{parse_kwargs(config['allocator']['kwargs'])})")
        self.bidder = eval(f"{config['bidder']['type']}(rng=rng,  context_dim=context_dim{parse_kwargs(config['bidder']['kwargs'])})")

        self.exploration_length = config['exploration_length']
    
    def should_random_bid(self):
        return self.clock < self.exploration_length
        return self.rng.random() < 1.0 - self.clock / (self.exploration_length+1e-2)

    def select_item(self, context):
        # Estimate CTR for all items
        if not isinstance(self.allocator, OracleAllocator) and self.allocator.mode=='UCB':
            estim_CTRs, uncertainty = self.allocator.estimate_CTR(context)
            estim_CTRs = estim_CTRs + self.allocator.c * uncertainty
        elif not isinstance(self.allocator, OracleAllocator) and self.allocator.mode=='TS':
            estim_CTRs, uncertainty = self.allocator.estimate_CTR(context)
        else:
            estim_CTRs = self.allocator.estimate_CTR(context)
            uncertainty = np.zeros_like(estim_CTRs)
        
        # Select the item with the highest expected reward
        estim_values = estim_CTRs * self.item_values
        best_item = np.argmax(estim_values)

        if not isinstance(self.allocator, OracleAllocator) and self.allocator.mode=='Epsilon-greedy':
            if self.rng.uniform(0,1)<self.allocator.eps:
                best_item = self.rng.choice(self.num_items, 1).item()

        return best_item, estim_CTRs[best_item], uncertainty[best_item]

    def bid(self, state, t, value=None, prob_win=None, b_grid=None):
        self.clock += 1
        context = state[:self.context_dim]
        item, estimated_CTR, uncertainty = self.select_item(context)
        optimistic_CTR = estimated_CTR
        value = self.item_values[item]

        if self.should_random_bid():
            bid = value * self.rng.random()
        elif isinstance(self.bidder, OracleBidder):
            n_values_search = int(value*100)
            b_grid = np.linspace(0.1*value, 1.5*value, n_values_search)
            prob_win = self.auction.winrate_model(context, b_grid)
            bid = self.bidder.bid(value, estimated_CTR, prob_win, b_grid)
        else:
            if not isinstance(self.allocator, OracleAllocator) and self.allocator.mode=='UCB':
                mean_CTR, _ = self.allocator.estimate_CTR(context)
                estimated_CTR = mean_CTR[item]
                bid = self.bidder.bid(value, context, estimated_CTR)
            elif not isinstance(self.allocator, OracleAllocator) and self.allocator.mode=='TS':
                mean_CTR, _ = self.allocator.estimate_CTR(context)
                estimated_CTR = mean_CTR[item]
                bid = self.bidder.bid(value, context, estimated_CTR)
            else:
                bid = self.bidder.bid(value, context, estimated_CTR)
        return item, np.clip(bid, 0.0, state[-2])

    def update(self, t):
        # Update response model with data from winning bids
        states, items, biddings, rewards, next_states, dones, wins, outcomes = self.buffer.numpy()
        contexts = states[:, :self.context_dim]
        self.allocator.update(contexts[wins], items[wins], outcomes[wins], t)

        # Update bidding model with all data
        self.bidder.update(contexts, biddings, wins)
    
    def get_uncertainty(self, len):
        if isinstance(self.allocator, OracleAllocator):
            return 0.0
        else:
            return np.mean(self.allocator.uncertainty[-len:])

class DQN(Agent):
    ''' A MDP style agent '''

    def __init__(self, rng, name, item_features, item_values, context_dim, buffer, config):
        super().__init__(rng, name, item_features, item_values, context_dim, buffer)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.exploration_strategy = config['exploration_strategy']
        if self.exploration_strategy=='Eps-Greedy':
            self.local_network = Critic(context_dim + 2 + self.feature_dim + 1, config['hidden_dim']).to(self.device)
            self.target_network = copy.deepcopy(self.local_network)
            self.eps_init = self.eps = config['eps_init']
            self.eps_min = config['eps_min']
            self.eps_decay = config['eps_decay']
        elif self.exploration_strategy=='Noise Injection':
            self.local_network = Critic(context_dim + 2 + self.feature_dim + 1, config['hidden_dim']).to(self.device)
            self.target_network = copy.deepcopy(self.local_network)
            self.noise_init = self.noise = config['noise_init']
            self.noise_min = config['noise_min']
            self.noise_decay = config['noise_decay']
        elif self.exploration_strategy=='Noisy Network':
            self.local_network = NoisyCritic(context_dim + 2 + self.feature_dim + 1, config['hidden_dim'], config['var_scale']).to(self.device)
            self.target_network = copy.deepcopy(self.local_network)

        self.optimizer = optim.Adam(self.local_network.parameters(), lr = config['lr'])
        self.batch_size = config['batch_size']
        self.tau = config['tau']
        self.num_grad_steps = config['num_grad_steps']
    
    def bid(self, state, t):
        self.clock += 1
        n_values_search = int(100*np.max(self.item_values))
        b_grid = np.linspace(0, 1.5*np.max(self.item_values), n_values_search)
        x = torch.Tensor(np.hstack([np.tile(state, (n_values_search * self.num_items, 1)), np.tile(self.items, (n_values_search, 1)), \
                       np.transpose(np.tile(b_grid, (self.num_items, 1))).reshape(-1, 1)])).to(self.device)
        with torch.no_grad():
            if self.exploration_strategy=='Eps-Greedy':
                if self.rng.uniform(0, 1) < self.eps:
                    item = self.rng.choice(self.num_items, 1).item()
                    bid = self.item_values[item] * self.rng.random()
                else:
                    index = np.argmax(self.local_network.Q1(x).numpy(force=True))
                    item = index % self.num_items
                    bid = b_grid[int(index / self.num_items)]
                self.eps = np.maximum(self.eps*self.eps_decay, self.eps_min)

            elif self.exploration_strategy=='Noise Injection':
                raise NotImplementedError
                param_dict = {}
                for key, param in self.local_network.parameters().items():
                    param_dict[key] = param.copy()
                    param.data.copy_(param + torch.randn_like(param)*self.noise)
                index = np.argmax(self.local_network.Q1(x).numpy(force=True))
                item = index % self.num_items
                bid = b_grid[int(index / self.num_items)]
                for key, param in self.local_network.parameters().items():
                    param.data.copy_(param_dict[key])
                self.noise = np.maximum(self.noise*self.noise_decay, self.noise_min)

            elif self.exploration_strategy=='Noisy Network':
                index = np.argmax(self.local_network.Q1(x).numpy(force=True))
                item = index % self.num_items
                bid = b_grid[int(index / self.num_items)]
        
        return item, np.clip(bid, 0.0, state[-2])

    def update(self, t):
        if len(self.buffer.states)<self.batch_size:
            return
        # Update response model with data from winning bids
        criterion = nn.MSELoss()
        self.local_network.train()
        self.target_network.eval()

        for i in range(self.num_grad_steps):
            states, item_inds, biddings, rewards, next_states, dones = self.buffer.sample(self.batch_size)
            q1, q2 = self.local_network(torch.Tensor(np.hstack([states, self.items[item_inds], biddings.reshape(-1, 1)])).to(self.device))
            with torch.no_grad():
                n_values_search = int(100*np.max(self.item_values))
                b_grid = np.linspace(0, 1.5*np.max(self.item_values), n_values_search)
                tmp = np.hstack([np.tile(self.items, (n_values_search, 1)), np.transpose(np.tile(b_grid, (self.num_items, 1))).reshape(-1, 1)])
                x = torch.Tensor(np.hstack([np.tile(next_states, (1, n_values_search * self.num_items)).reshape(-1, self.context_dim+2),\
                                        np.tile(tmp, (self.batch_size, 1))])).to(self.device)
                if self.exploration_strategy=='Noisy Network':
                    next_q1, next_q2  = self.target_network(x, sample=False)
                else:
                    next_q1, next_q2  = self.target_network(x)
                target_q1 = torch.max(next_q1.reshape(self.batch_size,-1), dim=1, keepdim=False).values
                target_q2 = torch.max(next_q2.reshape(self.batch_size,-1), dim=1, keepdim=False).values
                target_q = torch.Tensor(rewards).to(self.device) + torch.Tensor(1.0-dones).to(self.device) * torch.min(target_q1, target_q2)
            loss = criterion(q1.squeeze(), target_q) + criterion(q2.squeeze(), target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if t%2 == 0:
            for target_param, local_param in zip(self.target_network.parameters(), self.local_network.parameters()):
                target_param.data.copy_(self.tau * local_param + (1-self.tau) * target_param)
    
    def get_uncertainty(self, len):
        if self.exploration_strategy=='Noisy Network':
            return self.local_network.get_uncertainty()
        else:
            return 0.0

class QBid(Agent):
    ''' CVR estimator + DQN bidder '''

    def __init__(self, rng, name, item_features, item_values, context_dim, buffer, config):
        super().__init__(rng, name, item_features, item_values, context_dim, buffer)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.allocator = eval(f"{config['allocator']['type']}(rng=rng, item_features=item_features, context_dim=context_dim{parse_kwargs(config['allocator']['kwargs'])})")

        self.exploration_strategy = config['exploration_strategy']
        if self.exploration_strategy=='Eps-Greedy':
            self.local_network = Critic(context_dim + 2 + self.feature_dim + 1, config['hidden_dim']).to(self.device)
            self.target_network = copy.deepcopy(self.local_network)
            self.eps_init = config['eps_init']
            self.eps_min = config['eps_min']
            self.eps_decay = config['eps_decay']
        elif self.exploration_strategy=='Noise Injection':
            self.local_network = Critic(context_dim + 2 + self.feature_dim + 1, config['hidden_dim']).to(self.device)
            self.target_network = copy.deepcopy(self.local_network)
            self.noise_init = config['noise_init']
            self.noise_min = config['noise_min']
            self.noise_decay = config['noise_decay']
        elif self.exploration_strategy=='Noisy Network':
            self.local_network = NoisyCritic(context_dim + 2 + self.feature_dim + 1, config['hidden_dim']).to(self.device)
            self.target_network = copy.deepcopy(self.local_network)

        self.optimizer = optim.Adam(self.local_network.parameters(), lr = config['lr'])
        self.batch_size = config['batch_size']
        self.num_grad_steps = config['num_grad_steps']
        self.tau = config['tau']
    
    def select_item(self, context):
        # Estimate CTR for all items
        if not isinstance(self.allocator, OracleAllocator) and self.allocator.mode=='UCB':
            estim_CTRs, uncertainty = self.allocator.estimate_CTR(context, UCB=True)
            estim_CTRs = estim_CTRs + self.allocator.c * uncertainty
        elif not isinstance(self.allocator, OracleAllocator) and self.allocator.mode=='TS':
            estim_CTRs, uncertainty = self.allocator.estimate_CTR(context, TS=True)
        else:
            estim_CTRs = self.allocator.estimate_CTR(context)
            uncertainty = np.zeros_like(estim_CTRs)
        
        # Select the item with the highest expected reward
        estim_values = estim_CTRs * self.item_values
        best_item = np.argmax(estim_values)

        if not isinstance(self.allocator, OracleAllocator) and self.allocator.mode=='Epsilon-greedy':
            if self.rng.uniform(0,1)<self.allocator.eps:
                best_item = self.rng.choice(self.num_items, 1).item()

        return best_item, estim_CTRs[best_item], uncertainty[best_item]
    
    def bid(self, state, t):
        self.clock += 1
        context = state[:self.context_dim]
        item, estimated_CTR, uncertainty = self.select_item(context)
        value = self.item_values[item]

        n_values_search = int(100*value)
        b_grid = np.linspace(0, 1.5*value, n_values_search)
        x = torch.Tensor(np.hstack([np.tile(state, (n_values_search, 1)), np.tile(self.items[item], (n_values_search, 1)), \
                    b_grid.reshape(-1, 1)])).to(self.device)

        with torch.no_grad():
            if self.exploration_strategy=='Eps-Greedy':
                if self.rng.uniform(0, 1) < self.eps:
                    bid = self.item_values[item] * self.rng.random()
                else:
                    index = np.argmax(self.local_network.Q1(x).numpy(force=True))
                    bid = b_grid[index]
                self.eps = np.minimum(self.eps*self.eps_decay, self.eps_min)

            elif self.exploration_strategy=='Noise Injection':
                param_dict = {}
                for key, param in self.local_network.parameters().items():
                    param_dict[key] = param.copy()
                    param.data.copy_(param + torch.randn_like(param)*self.noise)
                index = np.argmax(self.local_network.Q1(x).numpy(force=True))
                bid = b_grid[index]
                for key, param in self.local_network.parameters().items():
                    param.data.copy_(param_dict[key])
                self.noise = np.minimum(self.noise*self.noise_decay, self.noise_min)

            elif self.exploration_strategy=='Noisy Network':
                index = np.argmax(self.local_network.Q1(x).numpy(force=True))
                bid = b_grid[index]
        
        return item, np.clip(bid, 0.0, state[-2])

    def update(self, t):
        states, item_inds, biddings, rewards, next_states, dones, wins, outcomes = self.buffer.numpy()
        contexts = states[:, :self.context_dim]
        self.allocator.update(contexts[wins], item_inds[wins], outcomes[wins], t)

        if len(self.buffer.states)<self.batch_size:
            return
        
        criterion = nn.MSELoss()
        self.local_network.train()
        self.target_network.eval()

        for i in range(self.num_grad_steps):
            states, item_inds, biddings, rewards, next_states, dones = self.buffer.sample(self.batch_size)
            q1, q2 = self.local_network(torch.Tensor(np.hstack([states, self.items[item_inds], biddings.reshape(-1, 1)])).to(self.device))
            with torch.no_grad():
                n_values_search = int(100*np.max(self.item_values))
                b_grid = np.linspace(0, 1.5*np.max(self.item_values), n_values_search)
                tmp = np.hstack([np.tile(self.items, (n_values_search, 1)), np.transpose(np.tile(b_grid, (self.num_items, 1))).reshape(-1, 1)])
                x = torch.Tensor(np.hstack([np.tile(next_states, (1, n_values_search * self.num_items)).reshape(-1, self.context_dim+2),\
                                        np.tile(tmp, (self.batch_size, 1))])).to(self.device)
                if self.exploration_strategy=='Noisy Network':
                    next_q1, next_q2  = self.target_network(x, sample=False)
                else:
                    next_q1, next_q2  = self.target_network(x)
                target_q1 = torch.max(next_q1.reshape(self.batch_size,-1), dim=1, keepdim=False).values
                target_q2 = torch.max(next_q2.reshape(self.batch_size,-1), dim=1, keepdim=False).values
                target_q = torch.Tensor(rewards).to(self.device) + torch.Tensor(1.0-dones).to(self.device) * torch.min(target_q1, target_q2)
            loss = criterion(q1.squeeze(), target_q) + criterion(q2.squeeze(), target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        if t%2 == 0:
            for target_param, local_param in zip(self.target_network.parameters(), self.local_network.parameters()):
                target_param.data.copy_(self.tau * local_param + (1-self.tau) * target_param)
    
    def get_uncertainty(self, len):
        if isinstance(self.allocator, OracleAllocator):
            return 0.0
        else:
            return np.mean(self.allocator.uncertainty[-len:])


class TD3Q(Agent):
    def __init__(self, rng, name, item_features, item_values, context_dim, buffer, config):
        super().__init__(rng, name, item_features, item_values, context_dim, buffer)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.allocator = eval(f"{config['allocator']['type']}(rng=rng, item_features=item_features, context_dim=context_dim{parse_kwargs(config['allocator']['kwargs'])})")

        self.critic = Critic(self.context_dim+2+self.feature_dim+1, config['hidden_dim']).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr = config['lr'])

        self.exploration_strategy = config['exploration_strategy']
        if self.exploration_strategy=='Eps-Greedy':
            self.actor = Actor(self.context_dim+2+self.feature_dim+1, config['hidden_dim']).to(self.device)
            self.actor_target = copy.deepcopy(self.actor)
            self.eps_init = self.eps = config['eps_init']
            self.eps_min = config['eps_min']
            self.eps_decay = config['eps_decay']
        elif self.exploration_strategy=='Gaussian Noise':
            self.actor = Actor(self.context_dim+2+self.feature_dim+1, config['hidden_dim']).to(self.device)
            self.actor_target = copy.deepcopy(self.actor)
            self.noise_init = self.noise = config['noise_init']
            self.noise_min = config['noise_min']
            self.noise_decay = config['noise_decay']
        elif self.exploration_strategy=='Noisy Network':
            self.actor = NoisyActor(self.context_dim+2+self.feature_dim+1, config['hidden_dim']).to(self.device)
            self.actor_target = copy.deepcopy(self.actor)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr = config['lr'])
        self.batch_size = config['batch_size']
        self.num_grad_steps = config['num_grad_steps']
        self.tau = config['tau']

        self.winrate = eval(f"{config['winrate']['type']}(rng=rng, context_dim=context_dim{parse_kwargs(config['winrate']['kwargs'])})")
        self.simulation_steps = config['simulation_steps']

        self.Gram = np.eye(self.context_dim)

    def select_item(self, context, t):
        # Estimate CTR for all items
        if not isinstance(self.allocator, OracleAllocator) and self.allocator.mode=='UCB':
            estim_CTRs, uncertainty = self.allocator.estimate_CTR(context)
            estim_CTRs = estim_CTRs + self.allocator.c * uncertainty
        elif not isinstance(self.allocator, OracleAllocator) and self.allocator.mode=='TS':
            estim_CTRs, uncertainty = self.allocator.estimate_CTR(context)
        else:
            estim_CTRs = self.allocator.estimate_CTR(context)
            uncertainty = np.zeros_like(estim_CTRs)
        
        # Select the item with the highest expected reward
        estim_values = estim_CTRs * self.item_values
        best_item = np.argmax(estim_values)

        if self.allocator.mode=='Epsilon-greedy':
            if self.rng.uniform(0,1)<self.allocator.eps:
                best_item = self.rng.choice(self.num_items, 1).item()
        
        return best_item, estim_CTRs[best_item], uncertainty[best_item]
    
    def bid(self, state, t):
        self.clock += 1
        context = state[:self.context_dim]
        item, estimated_CTR, uncertainty = self.select_item(context, t)
        value = self.item_values[item]

        if self.exploration_strategy=='Eps-Greedy':
            if self.rng.uniform(0, 1) < self.eps:
                    bid = self.item_values[item] * self.rng.random()
            else:
                x = torch.Tensor(np.concatenate([state, self.items[item], np.array([estimated_CTR])])).to(self.device)
                bid = np.clip(value * self.actor(x).item(), 0.0, value)
            self.eps = np.maximum(self.eps*self.eps_decay, self.eps_min)
        elif self.exploration_strategy=='Gaussian Noise':
            x = torch.Tensor(np.concatenate([state, self.items[item], np.array([estimated_CTR])])).to(self.device)
            bid = np.clip(value * self.actor(x).item() + self.noise * self.rng.normal(), 0.0, value)
            self.noise = np.maximum(self.noise*self.noise_decay, self.noise_min)
        elif self.exploration_strategy=='Noisy Network':
            x = torch.Tensor(np.concatenate([state, self.items[item], np.array([estimated_CTR])])).to(self.device)
            bid = np.clip(value * self.actor(x).item(), 0.0, value)

        return item,np.clip(bid, 0.0, state[-2])

    def update(self, t):
        # update CVR estimator and win rate estimator
        states, item_inds, biddings, rewards, next_states, dones, wins, outcomes = self.buffer.numpy()
        contexts = states[:, :self.context_dim]
        self.allocator.update(contexts[wins], item_inds[wins], outcomes[wins], t)
        self.winrate.update(contexts, biddings, wins)

        if len(self.buffer.states)<self.batch_size:
            return
        
        criterion = nn.MSELoss()
        self.critic.train()
        self.critic_target.eval()

        # update actor and critic using real data
        for i in range(self.num_grad_steps):
            states, item_inds, biddings, rewards, next_states, dones = self.buffer.sample(self.batch_size)
            contexts = states[:, :self.context_dim]
            estimated_CTRs = np.max(self.allocator.estimate_CTR_batched(contexts), axis=1).reshape(-1,1)
            x = torch.Tensor(np.concatenate([states, self.items[item_inds], biddings.reshape(-1,1)], axis=1)).to(self.device)
            q1, q2 = self.critic(x)

            with torch.no_grad():
                n_values_search = int(50*np.max(self.item_values))
                b_grid = np.linspace(0, 1.5*np.max(self.item_values), n_values_search)
                tmp = np.hstack([np.tile(self.items, (n_values_search, 1)), np.transpose(np.tile(b_grid, (self.num_items, 1))).reshape(-1, 1)])
                x = torch.Tensor(np.hstack([np.tile(next_states, (1, n_values_search * self.num_items)).reshape(-1, self.context_dim+2),\
                                        np.tile(tmp, (self.batch_size, 1))])).to(self.device)
                if self.exploration_strategy=='Noisy Network':
                    next_q1, next_q2  = self.critic_target(x, sample=False)
                else:
                    next_q1, next_q2  = self.critic_target(x)
                target_q1 = torch.max(next_q1.reshape(self.batch_size,-1), dim=1, keepdim=False).values
                target_q2 = torch.max(next_q2.reshape(self.batch_size,-1), dim=1, keepdim=False).values
                target_q = torch.Tensor(rewards).to(self.device) + torch.Tensor(1.0-dones).to(self.device) * torch.min(target_q1, target_q2)
            loss = criterion(q1.squeeze(), target_q) + criterion(q2.squeeze(), target_q)
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()

            if t % 2 == 0:
                x = torch.Tensor(np.concatenate([states, self.items[item_inds], estimated_CTRs], axis=1)).to(self.device)
                biddings = self.actor(x)
                x = torch.Tensor(np.concatenate([states, self.items[item_inds]], axis=1)).to(self.device)
                x = torch.concat([x, biddings], dim=1)
                loss = -self.critic.Q1(x).mean()
                self.actor_optim.zero_grad()
                loss.backward()
                self.actor_optim.step()

        # update actor and critic using simulated experiences
        for i in range(self.simulation_steps):
            states, _, _, _, _, _ = self.buffer.sample(self.batch_size)

            contexts = states[:, :self.context_dim]
            item_inds = self.rng.choice(self.num_items, size=(self.batch_size,))
            estimated_CTRs = np.max(self.allocator.estimate_CTR_batched(contexts), axis=1).reshape(-1,1)
            x = torch.Tensor(np.concatenate([states, self.items[item_inds], estimated_CTRs], axis=1)).to(self.device)
            biddings = self.actor(x).numpy(force=True)
            
            x = torch.Tensor(np.concatenate([contexts, biddings], axis=1)).to(self.device)
            if isinstance(self.winrate, OracleBidder):
                prob_win = []
                for j in range(self.batch_size):
                    prob_win.append(self.auction.winrate_model(contexts[j], biddings[j]))
                prob_win = np.array(prob_win).reshape(-1)
            else:
                prob_win = self.winrate.winrate_model(x).numpy(force=True).reshape(-1)
            wins = self.rng.binomial(1, prob_win)
            outcomes = self.rng.binomial(1, estimated_CTRs.reshape(-1))
            rewards = self.item_values[item_inds] * wins * outcomes
            assert len(rewards.shape)==1
            
            x = torch.Tensor(np.concatenate([states, self.items[item_inds], biddings.reshape(-1,1)], axis=1)).to(self.device)
            q1, q2 = self.critic(x)

            with torch.no_grad():
                n_values_search = int(50*np.max(self.item_values))
                b_grid = np.linspace(0, 1.5*np.max(self.item_values), n_values_search)
                tmp = np.hstack([np.tile(self.items, (n_values_search, 1)), np.transpose(np.tile(b_grid, (self.num_items, 1))).reshape(-1, 1)])
                x = torch.Tensor(np.hstack([np.tile(next_states, (1, n_values_search * self.num_items)).reshape(-1, self.context_dim+2),\
                                        np.tile(tmp, (self.batch_size, 1))])).to(self.device)
                if self.exploration_strategy=='Noisy Network':
                    next_q1, next_q2  = self.critic_target(x, sample=False)
                else:
                    next_q1, next_q2  = self.critic_target(x)
                target_q1 = torch.max(next_q1.reshape(self.batch_size,-1), dim=1, keepdim=False).values
                target_q2 = torch.max(next_q2.reshape(self.batch_size,-1), dim=1, keepdim=False).values
                target_q = torch.Tensor(rewards).to(self.device) + torch.Tensor(1.0-dones).to(self.device) * torch.min(target_q1, target_q2)
            loss = criterion(q1.squeeze(), target_q) + criterion(q2.squeeze(), target_q)
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()

            if t % 2 == 0:
                x = torch.Tensor(np.concatenate([states, self.items[item_inds], biddings], axis=1)).to(self.device)
                loss = -self.critic.Q1(x).mean()
                self.actor_optim.zero_grad()
                loss.backward()
                self.actor_optim.step()
        
        # polyak average network weights
        if t % 2 == 0:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
    def get_uncertainty(self, len):
        if isinstance(self.allocator, OracleAllocator):
            return 0.0
        else:
            return np.mean(self.allocator.uncertainty[-len:])


class TD3(Agent):
    def __init__(self, rng, name, item_features, item_values, context_dim, buffer, config):
        super().__init__(rng, name, item_features, item_values, context_dim, buffer)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.allocator = eval(f"{config['allocator']['type']}(rng=rng, item_features=item_features, context_dim=context_dim{parse_kwargs(config['allocator']['kwargs'])})")

        self.critic = Critic(self.context_dim+2+self.feature_dim+1, config['hidden_dim'], self.context_dim+self.feature_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr = config['lr'])

        self.exploration_strategy = config['exploration_strategy']
        if self.exploration_strategy=='Eps-Greedy':
            self.actor = Actor(self.context_dim+2+self.feature_dim+1, config['hidden_dim']).to(self.device)
            self.actor_target = copy.deepcopy(self.actor)
            self.eps_init = self.eps = config['eps_init']
            self.eps_min = config['eps_min']
            self.eps_decay = config['eps_decay']
        elif self.exploration_strategy=='Gaussian Noise':
            self.actor = Actor(self.context_dim+2+self.feature_dim+1, config['hidden_dim']).to(self.device)
            self.actor_target = copy.deepcopy(self.actor)
            self.noise_init = self.noise = config['noise_init']
            self.noise_min = config['noise_min']
            self.noise_decay = config['noise_decay']
        elif self.exploration_strategy=='Noisy Network':
            self.actor = NoisyActor(self.context_dim+2+self.feature_dim+1, config['hidden_dim']).to(self.device)
            self.actor_target = copy.deepcopy(self.actor)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr = config['lr'])
        self.batch_size = config['batch_size']
        self.num_grad_steps = config['num_grad_steps']
        self.tau = config['tau']

        self.winrate = eval(f"{config['winrate']['type']}(rng=rng, context_dim=context_dim{parse_kwargs(config['winrate']['kwargs'])})")
        self.simulation_steps = config['simulation_steps']

        self.Gram = np.eye(self.context_dim)

    def select_item(self, context, t):
        # Estimate CTR for all items
        if not isinstance(self.allocator, OracleAllocator) and self.allocator.mode=='UCB':
            estim_CTRs, uncertainty = self.allocator.estimate_CTR(context)
            estim_CTRs = estim_CTRs + self.allocator.c * uncertainty
        elif not isinstance(self.allocator, OracleAllocator) and self.allocator.mode=='TS':
            estim_CTRs, uncertainty = self.allocator.estimate_CTR(context)
        else:
            estim_CTRs = self.allocator.estimate_CTR(context)
            uncertainty = np.zeros_like(estim_CTRs)
        
        # Select the item with the highest expected reward
        estim_values = estim_CTRs * self.item_values
        best_item = np.argmax(estim_values)

        if self.allocator.mode=='Epsilon-greedy':
            if self.rng.uniform(0,1)<self.allocator.eps:
                best_item = self.rng.choice(self.num_items, 1).item()
        
        return best_item, estim_CTRs[best_item], uncertainty[best_item]
    
    def bid(self, state, t):
        self.clock += 1
        context = state[:self.context_dim]
        resource = state[self.context_dim:]
        item, estimated_CTR, uncertainty = self.select_item(context, t)
        value = self.item_values[item]

        if self.exploration_strategy=='Eps-Greedy':
            if self.rng.uniform(0, 1) < self.eps:
                    bid = self.item_values[item] * self.rng.random()
            else:
                x = torch.Tensor(np.concatenate([context, self.items[item], np.array([estimated_CTR]), resource])).to(self.device)
                bid = np.clip(value * self.actor(x).item(), 0.0, value)
            self.eps = np.maximum(self.eps*self.eps_decay, self.eps_min)
        elif self.exploration_strategy=='Gaussian Noise':
            x = torch.Tensor(np.concatenate([context, self.items[item], np.array([estimated_CTR]), resource])).to(self.device)
            bid = np.clip(value * self.actor(x).item() + self.noise * self.rng.normal(), 0.0, value)
            self.noise = np.maximum(self.noise*self.noise_decay, self.noise_min)
        elif self.exploration_strategy=='Noisy Network':
            x = torch.Tensor(np.concatenate([context, self.items[item], np.array([estimated_CTR]), resource])).to(self.device)
            bid = np.clip(value * self.actor(x).item(), 0.0, value)

        return item,np.clip(bid, 0.0, state[-2])

    def update(self, t):
        # update CVR estimator and win rate estimator
        states, item_inds, biddings, rewards, next_states, dones, wins, outcomes = self.buffer.numpy()
        contexts = states[:, :self.context_dim]
        self.allocator.update(contexts[wins], item_inds[wins], outcomes[wins], t)
        self.winrate.update(contexts, biddings, wins)

        if len(self.buffer.states)<self.batch_size:
            return
        
        criterion = nn.MSELoss()
        
        # update actor and critic using real data
        for i in range(self.num_grad_steps):
            states, item_inds, biddings, rewards, next_states, dones = self.buffer.sample(self.batch_size)
            contexts = states[:, :self.context_dim]
            resources = states[:,self.context_dim:]
            estimated_CTRs = np.max(self.allocator.estimate_CTR_batched(contexts), axis=1).reshape(-1,1)
            x = torch.Tensor(np.concatenate([contexts, self.items[item_inds], resources, biddings.reshape(-1,1)], axis=1)).to(self.device)
            q1, q2 = self.critic(x)

            with torch.no_grad():
                next_contexts = next_states[:, :self.context_dim]
                next_resources = next_states[:, self.context_dim:]
                next_estimated_CTRs = self.allocator.estimate_CTR_batched(next_contexts)
                next_items = np.argmax(next_estimated_CTRs, axis=1)
                next_estimated_CTRs = np.max(next_estimated_CTRs, axis=1).reshape(-1,1)

                x = torch.Tensor(np.concatenate([next_contexts, self.items[next_items], next_estimated_CTRs, next_resources], axis=1)).to(self.device)
                next_biddings = self.actor_target(x).numpy(force=True).reshape(-1,1)

                x = torch.Tensor(np.concatenate([next_contexts, self.items[next_items], next_resources, next_biddings], axis=1)).to(self.device)
                target_q1, target_q2 = self.critic_target(x)
                target_q = torch.min(target_q1, target_q2)
                target_q = torch.Tensor(rewards).to(self.device) + torch.Tensor(1.0-dones).to(self.device) * target_q.squeeze()
            loss = criterion(q1.squeeze(), target_q) + criterion(q2.squeeze(), target_q)
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()

            if t % 2 == 0:
                x = torch.Tensor(np.concatenate([contexts, self.items[item_inds], estimated_CTRs, resources], axis=1)).to(self.device)
                biddings = self.actor(x)
                x = torch.Tensor(np.concatenate([contexts, self.items[item_inds], resources], axis=1)).to(self.device)
                x = torch.concat([x, biddings], dim=1)
                loss = -self.critic.Q1(x).mean()
                self.actor_optim.zero_grad()
                loss.backward()
                self.actor_optim.step()

        # update actor and critic using simulated experiences
        for i in range(self.simulation_steps):
            states, _, _, _, _, _ = self.buffer.sample(self.batch_size)

            contexts = states[:, :self.context_dim]
            resources = states[:,self.context_dim:]
            item_inds = self.rng.choice(self.num_items, size=(self.batch_size,))
            estimated_CTRs = np.max(self.allocator.estimate_CTR_batched(contexts), axis=1).reshape(-1,1)
            x = torch.Tensor(np.concatenate([contexts, self.items[item_inds], estimated_CTRs, resources], axis=1)).to(self.device)
            biddings = self.actor(x).numpy(force=True)
            
            x = torch.Tensor(np.concatenate([contexts, biddings], axis=1)).to(self.device)
            if isinstance(self.winrate, OracleBidder):
                prob_win = []
                for j in range(self.batch_size):
                    prob_win.append(self.auction.winrate_model(contexts[j], biddings[j]))
                prob_win = np.array(prob_win).reshape(-1)
            else:
                prob_win = self.winrate.winrate_model(x).numpy(force=True).reshape(-1)
            wins = self.rng.binomial(1, prob_win)
            outcomes = self.rng.binomial(1, estimated_CTRs.reshape(-1))
            rewards = self.item_values[item_inds] * wins * outcomes
            assert len(rewards.shape)==1
            
            x = torch.Tensor(np.concatenate([contexts, self.items[item_inds], resources, biddings.reshape(-1,1)], axis=1)).to(self.device)
            q1, q2 = self.critic(x)

            with torch.no_grad():
                next_contexts = next_states[:, :self.context_dim]
                next_resources = next_states[:, self.context_dim:]
                next_estimated_CTRs = self.allocator.estimate_CTR_batched(next_contexts)
                next_items = np.argmax(next_estimated_CTRs, axis=1)
                next_estimated_CTRs = np.max(next_estimated_CTRs, axis=1).reshape(-1,1)

                x = torch.Tensor(np.concatenate([next_contexts, self.items[next_items], next_estimated_CTRs, next_resources], axis=1)).to(self.device)
                next_biddings = self.actor_target(x).numpy(force=True).reshape(-1,1)

                x = torch.Tensor(np.concatenate([next_contexts, self.items[next_items], next_resources, next_biddings], axis=1)).to(self.device)
                target_q1, target_q2 = self.critic_target(x)
                target_q = torch.min(target_q1, target_q2)
                target_q = torch.Tensor(rewards).to(self.device) + torch.Tensor(1.0-dones).to(self.device) * target_q.squeeze()
            loss = criterion(q1.squeeze(), target_q) + criterion(q2.squeeze(), target_q)
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()

            if t % 2 == 0:
                x = torch.Tensor(np.concatenate([contexts, self.items[item_inds], resources, biddings], axis=1)).to(self.device)
                loss = -self.critic.Q1(x).mean()
                self.actor_optim.zero_grad()
                loss.backward()
                self.actor_optim.step()
        
        # polyak average network weights
        if t % 2 == 0:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
    def get_uncertainty(self, len):
        if isinstance(self.allocator, OracleAllocator):
            return 0.0
        else:
            return np.mean(self.allocator.uncertainty[-len:])