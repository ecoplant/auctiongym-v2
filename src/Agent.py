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
            estim_CTRs = self.allocator.estimate_CTR(context, UCB=True)
        elif not isinstance(self.allocator, OracleAllocator) and self.allocator.mode=='TS':
            estim_CTRs = self.allocator.estimate_CTR(context, TS=True)
        else:
            estim_CTRs = self.allocator.estimate_CTR(context)
        # Compute value if clicked
        estim_values = estim_CTRs * self.item_values
        best_item = np.argmax(estim_values)
        if not isinstance(self.allocator, OracleAllocator) and self.allocator.mode=='Epsilon-greedy':
            if self.rng.uniform(0,1)<self.allocator.eps(self.exploration_length, self.clock):
                best_item = self.rng.choice(self.num_items, 1).item()

        return best_item, estim_CTRs[best_item]

    def bid(self, state, value=None, prob_win=None, b_grid=None):
        self.clock += 1
        context = state[:self.context_dim]
        item, estimated_CTR = self.select_item(context)
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.local_network = QNet(context_dim + 2 + self.feature_dim + 1, config['fc1_size'], config['fc2_size']).to(self.device)
        self.target_network = QNet(context_dim + 2 + self.feature_dim + 1, config['fc1_size'], config['fc2_size']).to(self.device)

        self.exploration_length = config['exploration_length']
        self.optimizer = optim.Adam(self.local_network.parameters(), lr = 5e-4)
        self.batch_size = config['batch_size']
        self.tau = config['tau']
        self.num_grad_steps = config['num_grad_steps']
    
    def bid(self, state):
        self.clock += 1
        n_values_search = int(100*np.max(self.item_values))
        b_grid = np.linspace(0, 1.5*np.max(self.item_values), n_values_search)
        x = torch.Tensor(np.hstack([np.tile(state, (n_values_search * self.num_items, 1)), np.tile(self.items, (n_values_search, 1)), \
                       np.transpose(np.tile(b_grid, (self.num_items, 1))).reshape(-1, 1)])).to(self.device)
        index = np.argmax(self.local_network(x).numpy(force=True))
        item = index % self.num_items
        bid = b_grid[int(index / self.num_items)]
        if self.clock < self.exploration_length:
            bid = self.item_values[item] * self.rng.random()
        return item, bid

    def update(self):
        if len(self.buffer.states)<self.batch_size:
            return
        # Update response model with data from winning bids
        criterion = nn.MSELoss()
        self.local_network.train()
        self.target_network.eval()
        for i in range(self.num_grad_steps):
            states, item_inds, biddings, rewards, next_states, dones = self.buffer.sample(self.batch_size)
            predicted_targets = self.local_network(torch.Tensor(np.hstack([states, self.items[item_inds], biddings.reshape(-1, 1)])).to(self.device)).squeeze()
            with torch.no_grad():
                n_values_search = int(100*np.max(self.item_values))
                b_grid = np.linspace(0, 1.5*np.max(self.item_values), n_values_search)
                tmp = np.hstack([np.tile(self.items, (n_values_search, 1)), np.transpose(np.tile(b_grid, (self.num_items, 1))).reshape(-1, 1)])
                x = torch.Tensor(np.hstack([np.tile(next_states, (1, n_values_search * self.num_items)).reshape(-1, self.context_dim+2),\
                                            np.tile(tmp, (self.batch_size, 1))])).to(self.device)
                labels = torch.Tensor(rewards).to(self.device) + torch.max(self.target_network(x).reshape(self.batch_size,-1), dim=1, keepdim=False).values
            loss = criterion(predicted_targets, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        for target_param, local_param in zip(self.target_network.parameters(), self.local_network.parameters()):
            target_param.data.copy_(self.tau * local_param + (1-self.tau) * target_param)

class QBid(Agent):
    ''' CVR estimator + DQN bidder '''

    def __init__(self, rng, name, item_features, item_values, context_dim, buffer, config):
        super().__init__(rng, name, item_features, item_values, context_dim, buffer)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.allocator = eval(f"{config['allocator']['type']}(rng=rng, item_features=item_features, context_dim=context_dim{parse_kwargs(config['allocator']['kwargs'])})")

        self.local_network = QNet(context_dim + 2 + self.feature_dim + 1, config['fc1_size'], config['fc2_size']).to(self.device)
        self.target_network = QNet(context_dim + 2 + self.feature_dim + 1, config['fc1_size'], config['fc2_size']).to(self.device)

        self.exploration_length = config['exploration_length']
        self.optimizer = optim.Adam(self.local_network.parameters(), lr = config['lr'])
        self.batch_size = config['batch_size']
        self.num_grad_steps = config['num_grad_steps']
        self.tau = config['tau']

        self.estimated_CTRs = []
    
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
            if self.rng.uniform(0,1)<self.allocator.eps(self.exploration_length, self.clock):
                best_item = self.rng.choice(self.num_items, 1).item()

        return best_item, estim_CTRs[best_item]
    
    def bid(self, state):
        self.clock += 1
        context = state[:self.context_dim]
        item, estimated_CTR = self.select_item(context)
        self.estimated_CTRs.append(estimated_CTR)
        value = self.item_values[item]

        if self.clock < self.exploration_length:
            bid = value
        else:
            n_values_search = int(100*value)
            b_grid = np.linspace(0, 1.5*value, n_values_search)
            with torch.no_grad():
                x = torch.Tensor(np.hstack([np.tile(state, (n_values_search, 1)), np.tile(self.items[item], (n_values_search, 1)), \
                            b_grid.reshape(-1, 1)])).to(self.device)
                index = np.argmax(self.local_network(x).numpy(force=True))
            bid = b_grid[index]
        
        return item, bid

    def update(self):
        states, item_inds, biddings, rewards, next_states, dones, wins, outcomes = self.buffer.numpy()
        contexts = states[:, :self.context_dim]
        self.allocator.update(contexts[wins], item_inds[wins], outcomes[wins])

        if len(self.buffer.states)<self.batch_size:
            return
        
        criterion = nn.MSELoss()
        self.local_network.train()
        self.target_network.eval()

        for i in range(self.num_grad_steps):
            states, item_inds, biddings, rewards, next_states, dones = self.buffer.sample(self.batch_size)
            predicted_targets = self.local_network(torch.Tensor(np.hstack([states, self.items[item_inds], biddings.reshape(-1, 1)])).to(self.device)).squeeze()
            with torch.no_grad():
                n_values_search = int(100*np.max(self.item_values))
                b_grid = np.linspace(0, 1.5*np.max(self.item_values), n_values_search)
                tmp = np.hstack([np.tile(self.items, (n_values_search, 1)), np.transpose(np.tile(b_grid, (self.num_items, 1))).reshape(-1, 1)])
                x = torch.Tensor(np.hstack([np.tile(next_states, (1, n_values_search * self.num_items)).reshape(-1, self.context_dim+2),\
                                        np.tile(tmp, (self.batch_size, 1))])).to(self.device)
                labels = torch.Tensor(rewards).to(self.device) + torch.max(self.target_network(x).reshape(self.batch_size,-1), dim=1, keepdim=False).values
            loss = criterion(predicted_targets, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        for target_param, local_param in zip(self.target_network.parameters(), self.local_network.parameters()):
            target_param.data.copy_(self.tau * local_param + (1-self.tau) * target_param)


class TD3(Agent):
    def __init__(self, rng, name, item_features, item_values, context_dim, buffer, config):
        super().__init__(rng, name, item_features, item_values, context_dim, buffer)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.allocator = eval(f"{config['allocator']['type']}(rng=rng, item_features=item_features, context_dim=context_dim{parse_kwargs(config['allocator']['kwargs'])})")

        self.actor = Actor(self.context_dim+2+self.feature_dim+1, config['hidden_dim']).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr = config['lr'])

        self.critic = Critic(self.context_dim+2+self.feature_dim+2, config['hidden_dim']).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr = config['lr'])

        self.batch_size = config['batch_size']
        self.num_grad_steps = config['num_grad_steps']
        self.tau = config['tau']
        self.noise = config['noise']

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
            if self.rng.uniform(0,1)<self.allocator.eps(self.exploration_length, self.clock):
                best_item = self.rng.choice(self.num_items, 1).item()

        return best_item, estim_CTRs[best_item]
    
    def bid(self, state):
        self.clock += 1
        context = state[:self.context_dim]
        item, estimated_CTR = self.select_item(context)
        value = self.item_values[item]

        if self.clock < self.exploration_length:
            bid = value * self.rng.random()
        else:
            x = torch.Tensor(np.concatenate([state, self.items[item], np.array([estimated_CTR])])).to(self.device)
            bid = value * self.actor(x).item()
            bid = np.clip(bid+self.rng.normal(0.0,self.noise), 0.0, value)

        return item, bid

    def update(self):
        states, item_inds, biddings, rewards, next_states, dones, wins, outcomes = self.buffer.numpy()
        contexts = states[:, :self.context_dim]
        self.allocator.update(contexts[wins], item_inds[wins], outcomes[wins])

        if len(self.buffer.states)<self.batch_size:
            return
        
        criterion = nn.MSELoss()
        self.critic.train()
        self.critic_target.eval()

        for i in range(self.num_grad_steps):
            states, item_inds, biddings, rewards, next_states, dones = self.buffer.sample(self.batch_size)
            with torch.no_grad():
                next_contexts = next_states[:, :self.context_dim]
                next_estimated_CTRs = self.allocator.estimate_CTR_batched(next_contexts)
                next_items = np.argmax(next_estimated_CTRs, axis=1)
                next_estimated_CTRs = np.max(next_estimated_CTRs, axis=1).reshape(-1,1)

                x = torch.Tensor(np.concatenate([next_states, self.items[next_items], next_estimated_CTRs], axis=1)).to(self.device)
                next_biddings = self.actor_target(x).numpy(force=True).reshape(-1,1)

                x = torch.Tensor(np.concatenate([next_states, self.items[next_items], next_estimated_CTRs, next_biddings], axis=1)).to(self.device)
                target_q1, target_q2 = self.critic_target(x)
                target_q = torch.min(target_q1, target_q2)
                target_q = torch.Tensor(rewards).to(self.device) + torch.Tensor(1.0-dones).to(self.device) * target_q.squeeze()
            
            contexts = states[:, :self.context_dim]
            estimated_CTRs = np.max(self.allocator.estimate_CTR_batched(contexts), axis=1).reshape(-1,1)
            x = torch.Tensor(np.concatenate([states, self.items[item_inds], estimated_CTRs, biddings.reshape(-1,1)], axis=1)).to(self.device)
            q1, q2 = self.critic(x)
            loss = criterion(q1.squeeze(), target_q) + criterion(q2.squeeze(), target_q)
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()
        
            if i % 2 == 0:
                x = torch.Tensor(np.concatenate([states, self.items[item_inds], estimated_CTRs], axis=1)).to(self.device)
                biddings = self.actor(x)
                x = torch.concat([x, biddings], dim=1)
                loss = -self.critic.Q1(x).mean()
                self.actor_optim.zero_grad()
                loss.backward()
                self.actor_optim.step()

                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)