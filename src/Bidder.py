import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from models import *

class Bidder:
    """ Bidder base class"""
    def __init__(self, rng):
        self.rng = rng

    def update(self, context, bid, won):
        pass
    
class TruthfulBidder(Bidder):
    def __init__(self, rng, noise=0.1):
        super().__init__(rng)
        self.noise = noise

    def bid(self, value, context, estimated_CTR):
        bid = value * (estimated_CTR + self.rng.normal(0,self.noise,1))
        return bid.item()

class OracleBidder(Bidder):
    def __init__(self, rng, context_dim):
        super().__init__(rng)

    def bid(self, value, estimated_CTR, prob_win, b_grid):
        expected_value = value * estimated_CTR
    
        estimated_utility = prob_win * (expected_value - b_grid)
        bid = b_grid[np.argmax(estimated_utility)]

        return bid

class DefaultBidder(Bidder):
    def __init__(self, rng, lr, context_dim, noise=0.0, num_grad_steps=10, batch_size=512):
        super().__init__(rng)
        self.lr = lr
        self.context_dim = context_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.winrate_model = NeuralWinRateEstimator(context_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.winrate_model.parameters(), lr=self.lr)
        self.noise = noise
        self.num_grad_steps = num_grad_steps
        self.batch_size = batch_size
        self.initialize()
    
    def initialize(self):
        X = []
        y = []
        for i in range(500):
            context = self.rng.normal(0.0, 1.0, size=self.context_dim)
            b = self.rng.uniform(0.0, 1.5, 10).reshape(-1,1)
            y.append((1 + np.exp(-10*(b-1.0)))**(-1))
            X.append(np.concatenate([np.tile(context/np.sqrt(np.sum(context**2)), (10,1)), b], axis=-1))
        self.X_init = np.concatenate(X)
        self.y_init = np.concatenate(y)
        X = torch.Tensor(self.X_init).to(self.device)
        y = torch.Tensor(self.y_init).to(self.device)

        epochs = 10000
        optimizer = torch.optim.Adam(self.winrate_model.parameters(), lr=self.lr, weight_decay=1e-6, amsgrad=True)
        MSE = nn.MSELoss()
        self.winrate_model.train()
        for epoch in tqdm(range(epochs), desc='initializing winrate estimators'):
            optimizer.zero_grad()
            y_pred = self.winrate_model(X)
            loss = MSE(y_pred, y)
            loss.backward()
            optimizer.step()
        self.winrate_model.eval()

    def bid(self, value, context, estimated_CTR):
        # Compute the bid as expected value
        expected_value = value * estimated_CTR
        # Grid search over gamma
        n_values_search = int(value*100)
        b_grid = np.linspace(0.1*value, 1.5*value, n_values_search)
        x = torch.Tensor(np.hstack([np.tile(context, ((n_values_search, 1))), b_grid.reshape(-1,1)])).to(self.device)

        prob_win = self.winrate_model(x).numpy(force=True).ravel()

        estimated_utility = prob_win * (expected_value - b_grid)
        bid = b_grid[np.argmax(estimated_utility)]
        
        bid = np.clip(bid+self.rng.normal(0,self.noise)*value, 0.0, 1.0*value)
        return bid

    def update(self, context, bid, won):
        N = context.shape[0]
        batch_size = min(N, self.batch_size)

        self.winrate_model.train()
        for _ in range(self.num_grad_steps):
            ind = self.rng.choice(N, size=batch_size, replace=False)
            X = np.hstack((context[ind].reshape(-1,self.context_dim), bid[ind].reshape(-1, 1)))
            X = torch.Tensor(X).to(self.device)
            y = won[ind].astype(np.float32).reshape(-1,1)
            y = torch.Tensor(y).to(self.device)
            self.optimizer.zero_grad()
            loss = self.winrate_model.loss(X, y)
            loss.backward()
            self.optimizer.step()
        self.winrate_model.eval()


