import numpy as np
import torch.optim as optim

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

    def select_item(self, context):
        pass

    def bid(self, context):
        pass

class Bandit(Agent):
    ''' A bandit-style agent '''

    def __init__(self, rng, name, item_features, item_values, context_dim, buffer, config):
        super().__init__(rng, name, item_features, item_values, context_dim, buffer)

        self.allocator = eval(f"{config['allocator']['type']}(rng=rng, item_features=item_features,  num_items=self.num_items, context_dim=context_dim{parse_kwargs(config['allocator']['kwargs'])})")
        self.bidder = eval(f"{config['bidder']['type']}(rng=rng,  context_dim=context_dim{parse_kwargs(config['bidder']['kwargs'])})")

        self.exploration_length = config['exploration_length']

    def select_item(self, context):
        # Estimate CTR for all items
        if not isinstance(self.allocator, OracleAllocator) and self.allocator.mode=='UCB':
            estim_CTRs = self.allocator.estimate_CTR(context, UCB=True)
        elif not isinstance(self.allocator, OracleAllocator) and self.allocator.mode=='TS':
            estim_CTRs = self.allocator.estimate_CTR(context, TS=True)
        else:
            estim_CTRs = self.allocator.estimate_CTR(context)
        # Compute value if clicked
        estim_values = estim_CTRs * self.item_values
        best_item = np.argmax(estim_values)
        if not isinstance(self.allocator, OracleAllocator) and self.allocator.mode=='Epsilon-greedy':
            if self.rng.uniform(0,1)<self.allocator.eps:
                best_item = self.rng.choice(self.num_items, 1).item()

        return best_item, estim_CTRs[best_item]

    def bid(self, state, value=None, prob_win=None, b_grid=None):
        self.clock += 1
        context = state[:self.context_dim]
        item, estimated_CTR = self.select_item(context)
        optimistic_CTR = estimated_CTR
        value = self.item_values[item]

        if self.clock < self.exploration_length:
            bid = value
        elif isinstance(self.bidder, OracleBidder):
            bid = self.bidder.bid(value, estimated_CTR, prob_win, b_grid)
        else:
            if not isinstance(self.allocator, OracleAllocator) and self.allocator.mode=='UCB':
                mean_CTR = self.allocator.estimate_CTR(context, UCB=False)
                estimated_CTR = mean_CTR[item]
                bid = self.bidder.bid(value, context, optimistic_CTR)
            elif not isinstance(self.allocator, OracleAllocator) and self.allocator.mode=='TS':
                mean_CTR = self.allocator.estimate_CTR(context, TS=False)
                estimated_CTR = mean_CTR[item]
                bid = self.bidder.bid(value, context, optimistic_CTR)
            else:
                bid = self.bidder.bid(value, context, estimated_CTR)
        return item, bid

    def update(self):
        # Update response model with data from winning bids
        states, items, biddings, rewards, next_states, dones, wins, outcomes = self.buffer.numpy()
        contexts = states[:, :self.context_dim]
        self.allocator.update(contexts[wins], items[wins], outcomes[wins])

        # Update bidding model with all data
        self.bidder.update(contexts, biddings, wins)

class DQN(Agent):
    ''' A MDP style agent '''

    def __init__(self, rng, name, item_features, item_values, context_dim, buffer, config):
        super().__init__(rng, name, item_features, item_values, context_dim, buffer)

        self.local_network = QNet(context_dim + 2 + self.feature_dim + 1, config['fc1_size'], config['fc2_size'])
        self.target_network = QNet(context_dim + 2 + self.feature_dim + 1, config['fc1_size'], config['fc2_size'])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.exploration_length = config['exploration_length']
        self.optimizer = optim.Adam(self.local_network.parameters(), lr = 5e-4)
        self.update_nums = config['update_nums']
        self.epsilon_initial = config['epsilon_initial']
        self.epsilon_decay = config['epsilon_decay']
        self.epsilon_final = config['epsilon_final']
        self.batch_size = config['batch_size']
        self.epsilon = self.epsilon_initial
        self.tau = config['tau']
    
    def bid(self, state):
        self.clock += 1
        n_values_search = int(100*np.max(self.item_values))
        b_grid = np.linspace(0, 1.5*np.max(self.item_values), n_values_search)
        x = torch.Tensor(np.hstack([np.tile(state, (n_values_search * self.num_items, 1)), np.tile(self.items, (n_values_search, 1)), \
                       np.transpose(np.tile(b_grid, (self.num_items, 1))).reshape(-1, 1)])).to(self.device)
        index = np.argmax(self.local_network(x).detach().numpy())
        item = index % self.num_items
        bid = b_grid[int(index / self.num_items)]
        if self.rng.uniform(0, 1) < self.epsilon:
            item = self.rng.choice(self.num_items, 1).item()
        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay
        if self.clock < self.exploration_length:
            bid = self.item_values[item]
        return item, bid

    def update(self):
        # Update response model with data from winning bids
        for i in range(self.update_nums):
            criterion = nn.MSELoss()
            self.local_network.train()
            self.target_network.eval()
            states, item_inds, biddings, rewards, next_states, dones = self.buffer.sample(self.batch_size)
            predicted_targets = self.local_network(torch.Tensor(np.hstack([states, self.items[item_inds], biddings.reshape(-1, 1)])).to(self.device)).squeeze()
            with torch.no_grad():
                n_values_search = int(100*np.max(self.item_values))
                b_grid = np.linspace(0, 1.5*np.max(self.item_values), n_values_search)
                tmp = np.hstack([np.tile(self.items, (n_values_search, 1)), np.transpose(np.tile(b_grid, (self.num_items, 1))).reshape(-1, 1)])
                x = torch.Tensor(np.hstack([np.tile(next_states, (1, n_values_search * self.num_items)).reshape(-1, self.context_dim+2),\
                                np.tile(tmp, (self.batch_size, 1))])).to(self.device)
                labels = torch.tensor(rewards, dtype=torch.float32).to(self.device) \
                    + torch.max(self.target_network(x).reshape(self.batch_size, -1), dim=1, keepdim=False).values.to(self.device)
            loss = criterion(predicted_targets, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        for target_param, local_param in zip(self.target_network.parameters(), self.local_network.parameters()):
            target_param.data.copy_(self.tau * local_param + (1-self.tau) * target_param)