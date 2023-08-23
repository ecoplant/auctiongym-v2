import numpy as np
import gym
from gym.spaces import Dict, Box, Discrete, MultiBinary

CONTEXT_LOW = -5.0
CONTEXT_HIGH = 5.0
VALUE_LOW = 0.0
VALUE_HIGH = 1.0
BID_LOW = 0.0
BID_HIGH = 2.0


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class Auction(gym.Env):
    def __init__(self, rng, agent, CTR_model, winrate_model, item_features, item_values, context_dim, context_dist, horizon, budget):
        super().__init__()
        self.rng = rng
        self.agent = agent

        self.CTR_model = CTR_model
        self.winrate_model = winrate_model
        self.item_features = item_features
        self.item_values = item_values
        self.context_dim = item_features.shape[1]
        self.num_items = item_features.shape[0]

        self.context_low = CONTEXT_LOW * np.ones(self.context_dim)
        self.context_high = CONTEXT_HIGH * np.ones(self.context_dim)
        self.observation_space = Dict({
            'context' : Box(low=self.context_low, high=self.context_high),
        })
        self.action_space = Dict({
            'item' : Discrete(self.item_features.shape[0]),
            'bid' : Box(low=BID_LOW, high=BID_HIGH)
        })
        self.context_dim = context_dim
        self.context_dist = context_dist # Gaussian, Bernoulli, Uniform
        self.gaussian_var = 1.0
        self.bernoulli_p = 0.5

        self.horizon = horizon
        self.budget = budget
    
    def generate_context(self):
        if self.context_dist=='Gaussian':
            context = self.rng.normal(0.0, 1.0, size=self.context_dim)
        elif self.context_dist=='Bernoulli':
            context = self.rng.binomial(1, self.bernoulli_p, size=self.context_dim)
        else:
            context = self.rng.uniform(-1.0, 1.0, size=self.context_dim)
        return np.clip(context, CONTEXT_LOW, CONTEXT_HIGH)
    
    def reset(self):
        self.context = self.generate_context()
        self.remaining_steps = self.horizon
        self.remaining_budget = self.budget
        return np.concatenate([self.context, np.array([self.remaining_budget]), np.array([self.remaining_steps])]), {}

    def step(self, action):
        item = action['item']
        bid = action['bid']
        winrate = self.winrate_model(self.context, bid)
        win = self.rng.binomial(1, winrate)
        CTR = self.CTR_model(self.context)
        outcome = self.rng.binomial(1, CTR[item])
        reward = self.item_values[item] * outcome * win

        info = {
            'win' : win,
            'outcome' : outcome,
            'optimal_selection' : item==np.argmax(self.item_values * CTR)
        }
        self.remaining_budget -= win * bid.item()
        self.remaining_steps -= 1
        self.context = self.generate_context()
        return np.concatenate([self.context, np.array([self.remaining_budget]), np.array([self.remaining_steps])]), reward, self.remaining_budget<1e-3, self.remaining_steps==0, info