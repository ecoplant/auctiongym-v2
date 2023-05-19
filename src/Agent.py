import numpy as np

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

class Bandit:
    ''' A bandit-style agent '''

    def __init__(self, rng, name, item_features, item_values, context_dim, config):
        super.__init__(rng, name, item_features, item_values, context_dim)

        self.allocator = eval(f"{config['allocator']['type']}(rng=rng, item_features=item_features, context_dim=context_dim{parse_kwargs(config['allocator']['kwargs'])})")
        self.bidder = eval(f"{config['bidder']['type']}(rng=rng,  context_dim=context_dim, num_items=self.num_items{parse_kwargs(config['bidder']['kwargs'])})")

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

    def bid(self, context, value=None, prob_win=None, b_grid=None):
        self.clock += 1
        item, estimated_CTR = self.select_item(context)
        optimistic_CTR = estimated_CTR
        value = self.item_values[item]

        if isinstance(self.bidder, OracleBidder):
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
        return item, bid, estimated_CTR, optimistic_CTR

    def update(self):
        # Update response model with data from winning bids
        states, items, biddings, rewards, next_states, dones, wins, outcomes = self.buffer.numpy()
        contexts = states[:, :self.context_dim]
        self.allocator.update(contexts[wins], items[wins], outcomes[wins])

        # Update bidding model with all data
        self.bidder.update(contexts, biddings, wins)

