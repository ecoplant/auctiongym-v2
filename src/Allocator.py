import numpy as np

from models import *

class Allocator:
    """ Base class for an allocator """

    def __init__(self, rng, item_features):
        self.rng = rng
        self.item_features = item_features
        self.feature_dim = item_features.shape[1]
        self.K = item_features.shape[0]

    def update(self, contexts, items, outcomes):
        pass
    

class OracleAllocator(Allocator):
    """ An allocator that acts based on the true P(click)"""

    def __init__(self, rng, item_features, context_dim):
        super().__init__(rng, item_features)
        self.context_dim = context_dim

    def set_CTR_model(self, M):
        self.M = M

    def estimate_CTR(self, context):
        return sigmoid(self.item_features @ self.M.T @ context / np.sqrt(context.shape[0]))
    
    def get_uncertainty(self):
        return np.array([0])


class LogisticAllocator(Allocator):
    def __init__(self, rng, item_features, lr, context_dim, mode, c=0.0, eps=0.1, nu=0.0):
        super().__init__(rng, item_features)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        self.lr = lr

        self.d = context_dim
        self.c = c
        self.eps = eps
        self.nu = nu
        if self.mode=='UCB':
            self.model = LogisticRegression(self.d, self.item_features, self.mode, self.rng, self.lr, c=self.c).to(self.device)
        elif self.mode=='TS':
            self.model = LogisticRegression(self.d, self.item_features, self.mode, self.rng, self.lr, nu=self.nu).to(self.device)
        else:
            self.model = LogisticRegression(self.d, self.item_features, self.mode, self.rng, self.lr).to(self.device)

    def update(self, contexts, items, outcomes):
        self.model.update(contexts, items, outcomes)

    def estimate_CTR(self, context, UCB=False, TS=False):
        return self.model.estimate_CTR(context, UCB, TS)

    def get_uncertainty(self):
        return self.model.get_uncertainty()

class NeuralAllocator(Allocator):
    def __init__(self, rng, item_features, lr, batch_size, weight_decay, latent_dim, num_epochs, context_dim, mode, eps_max=None, eps_min=None, prior_var=None):
        super().__init__(rng, item_features)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        self.num_epochs = num_epochs
        self.context_dim = context_dim
        if self.mode=='Epsilon-greedy':
            self.net = NeuralRegression(context_dim+self.feature_dim, latent_dim).to(self.device)
            self.eps_max = eps_max
            self.eps_min = eps_min
        else:
            raise NotImplementedError
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad=True)
    
    def eps(self, l, t):
        return np.maximum(self.eps_max + t/(l+1e-2)*(self.eps_min-self.eps_max), self.eps_min)

    def update(self, contexts, items, outcomes):
        N = contexts.shape[0]
        if N<10:
            return

        self.net.train()
        batch_size = min(N, self.batch_size)

        for epoch in range(int(self.num_epochs)):
            ind = self.rng.choice(N, size=batch_size, replace=False)
            X = np.concatenate([contexts[ind], self.item_features[items[ind]]], axis=1)
            X, y = torch.Tensor(X).to(self.device), torch.Tensor(outcomes[ind]).to(self.device)
            self.optimizer.zero_grad()
            if self.mode=='Epsilon-greedy':
                loss = self.net.loss(self.net(X).squeeze(), y)
            else:
                loss = self.net.loss(self.net(X).squeeze(), y, N)
            loss.backward()
            self.optimizer.step()
        self.net.eval()

    def estimate_CTR(self, context, TS=False):
        if TS:
            X = torch.Tensor(np.concatenate([np.tile(context.reshape(1,-1),(self.K, 1)), self.item_features],axis=1)).to(self.device)
            y = []
            for i in range(self.num_samples):
                y.append(self.net(X, MAP=False).numpy(force=True).reshape(-1))
            y = np.stack(y)
            std = np.std(y, axis=0)
            self.uncertainty.append(np.mean(std))
            return y[0], std
        else:
            X = torch.Tensor(np.concatenate([np.tile(context.reshape(1,-1),(self.K, 1)), self.item_features],axis=1)).to(self.device)
            if self.mode=='TS':
                return self.net(X, MAP=True).numpy(force=True).reshape(-1)
            else:
                return self.net(X).numpy(force=True).reshape(-1)
    
    def estimate_CTR_batched(self, context):
        X = torch.Tensor(np.concatenate([np.tile(context, (1,self.K)).reshape(-1,self.context_dim), np.tile(self.item_features, (context.shape[0],1))],axis=1)).to(self.device)
        return self.net(X).numpy(force=True).reshape(context.shape[0], self.K)
