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
        

class A3C(Agent):
    def __init__(self, rng, name, item_features, item_values, context_dim, buffer, config):
        super().__init__(rng, name, item_features, item_values, context_dim, buffer)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.allocator = eval(f"{config['allocator']['type']}(rng=rng, item_features=item_features, context_dim=context_dim{parse_kwargs(config['allocator']['kwargs'])})")

        self.net = Net(self.context_dim+2, config['hidden_dim'], action_dim=10).to(self.device)
        self.optim = optim.Adam(self.net.parameters(), lr = config['lr'])
        self.dist = torch.distributions.Categorical

        self.exploration_strategy = config['exploration_strategy']
        if self.exploration_strategy=='Eps-Greedy':
            self.eps_init = self.eps = config['eps_init']
            self.eps_min = config['eps_min']
            self.eps_decay = config['eps_decay']
        elif self.exploration_strategy=='Gaussian Noise':
            self.noise_init = self.noise = config['noise_init']
            self.noise_min = config['noise_min']
            self.noise_decay = config['noise_decay']

        self.batch_size = config['batch_size']
        self.num_grad_steps = config['num_grad_steps']

        self.winrate = eval(f"{config['winrate']['type']}(rng=rng, context_dim=context_dim{parse_kwargs(config['winrate']['kwargs'])})")
        self.simulation_steps = config['simulation_steps']

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

        x = torch.Tensor(state).to(self.device)
        if self.exploration_strategy=='Eps-Greedy':
            if self.rng.uniform(0, 1) < self.eps:
                    bid = self.item_values[item] * self.rng.random()
            else:
                bid = 0.1 * self.net.act(x) + 0.1
                bid = np.clip(value * bid, 0.0, value)
            self.eps = np.maximum(self.eps*self.eps_decay, self.eps_min)
        elif self.exploration_strategy=='Gaussian Noise':
            bid = 0.1 * self.net.act(x) + 0.1
            bid = np.clip(value * bid + self.noise * self.rng.normal(), 0.0, value)
            self.noise = np.maximum(self.noise*self.noise_decay, self.noise_min)
        elif self.exploration_strategy=='Noisy Network':
            bid = 0.1 * self.net.act(x) + 0.1
            bid = np.clip(value * bid, 0.0, value)

        return item,np.clip(bid, 0.0, state[-2])

    def update(self, t):
        # update CVR estimator and win rate estimator
        states, item_inds, biddings, rewards, next_states, dones, wins, outcomes = self.buffer.numpy()
        contexts = states[:, :self.context_dim]
        self.allocator.update(contexts[wins], item_inds[wins], outcomes[wins], t)
        self.winrate.update(contexts, biddings, wins)

        if len(self.buffer.states)<self.batch_size:
            return
        
        # update actor and critic using real data
        for i in range(self.num_grad_steps):
            states, item_inds, biddings, rewards, next_states, dones = self.buffer.sample_recent(self.batch_size)
            s = torch.Tensor(states).to(self.device)
            s_ = torch.Tensor(next_states).to(self.device)
            v = self.net.v(s).squeeze()
            v_next = self.net.v(s_).squeeze()
            v_target = torch.Tensor(rewards).to(self.device) + torch.Tensor(1.0-dones).to(self.device) * v_next.detach()

            critic_loss = (v_target-v)**2
            logits = self.net.pi(s)
            probs = F.softmax(logits, dim=1)
            m = self.dist(probs)
            biddings = np.clip(biddings, 0.1, 1.0)
            action = torch.LongTensor(np.floor(biddings/0.1-1)).to(self.device)
            actor_loss = -m.log_prob(action) * (v_target-v).detach()

            loss = (critic_loss + actor_loss).mean()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()