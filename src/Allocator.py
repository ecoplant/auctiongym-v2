import numpy as np

from models import *

class Allocator:
    """ Base class for an allocator """

    def __init__(self, rng, item_features):
        self.rng = rng
        self.item_features = item_features
        self.feature_dim = item_features.shape[1]
        self.K = item_features.shape[0]

    def update(self, contexts, items, outcomes, t):
        pass
    

class OracleAllocator(Allocator):
    """ An allocator that acts based on the true P(click)"""

    def __init__(self, rng, item_features, context_dim):
        super().__init__(rng, item_features)
        self.context_dim = context_dim
        self.mode = 'Epsilon-greedy'
        self.eps = 0.0

    def set_CTR_model(self, M):
        self.M = M

    def estimate_CTR(self, context):
        return sigmoid(self.item_features @ self.M.T @ context / np.sqrt(context.shape[0]))
    
    def estimate_CTR_batched(self, context):
        y = []
        for i in range(context.shape[0]):
            y.append(sigmoid(self.item_features @ self.M.T @ context[i] / np.sqrt(context.shape[1])))
        return np.stack(y)
    
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
        
        self.uncertainty = self.model.uncertainty

    def update(self, contexts, items, outcomes, t):
        self.model.update(contexts, items, outcomes, t)

    def estimate_CTR(self, context):
        return self.model.estimate_CTR(context)
    
    def estimate_CTR_batched(self, context):
        return self.model.estimate_CTR_batched(context).reshape(context.shape[0], self.K)

    def get_uncertainty(self):
        return self.model.get_uncertainty()

class NeuralAllocator(Allocator):
    def __init__(self, rng, item_features, lr, batch_size, weight_decay, latent_dim, num_epochs, context_dim, eps):
        super().__init__(rng, item_features)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = 'Epsilon-greedy'
        self.num_epochs = num_epochs
        self.context_dim = context_dim

        self.net = NeuralRegression(context_dim+self.feature_dim, latent_dim).to(self.device)
        self.eps = eps
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.uncertainty = []
    
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
        X = torch.Tensor(np.concatenate([np.tile(context.reshape(1,-1),(self.K, 1)), self.item_features],axis=1)).to(self.device)
        self.uncertainty.append(0.0)
        return self.net(X).numpy(force=True).reshape(-1)
    
    def estimate_CTR_batched(self, context):
        X = torch.Tensor(np.concatenate([np.tile(context, (1,self.K)).reshape(-1,self.context_dim), np.tile(self.item_features, (context.shape[0],1))],axis=1)).to(self.device)
        return self.net(X).numpy(force=True).reshape(context.shape[0], self.K)

class BootstrapAllocator(Allocator):
    def __init__(self, rng, item_features, lr, batch_size, weight_decay, latent_dim, num_nets, num_epochs, context_dim, mode, c=None, nu=None):
        super().__init__(rng, item_features)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        if self.mode=='UCB':
            self.c = c
        elif self.mode=='TS':
            self.nu = nu
        self.num_epochs = num_epochs
        self.context_dim = context_dim
        
        self.num_nets = num_nets
        self.nets = [NeuralRegression(context_dim+self.feature_dim, latent_dim).to(self.device) for _ in range(num_nets)]

        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.optimizers = [torch.optim.Adam(self.nets[i].parameters(), lr=self.lr, weight_decay=self.weight_decay) for i in range(num_nets)]

        self.uncertainty = []
    
    def eps(self, l, t):
        return np.maximum(self.eps_max + t/(l+1e-2)*(self.eps_min-self.eps_max), self.eps_min)

    def update(self, contexts, items, outcomes):
        N = contexts.shape[0]
        if N<10:
            return

        batch_size = min(N, self.batch_size)

        for epoch in range(int(self.num_epochs)):
            for i in range(self.num_nets):
                ind = self.rng.choice(N, size=batch_size, replace=False)
                X = np.concatenate([contexts[ind], self.item_features[items[ind]]], axis=1)
                X, y = torch.Tensor(X).to(self.device), torch.Tensor(outcomes[ind]).to(self.device)
                self.optimizers[i].zero_grad()
                loss = self.nets[i].loss(self.nets[i](X).squeeze(), y)
                loss.backward()
                self.optimizers[i].step()

    def estimate_CTR(self, context):
        if self.mode=='UCB':
            X = torch.Tensor(np.concatenate([np.tile(context.reshape(1,-1),(self.K, 1)), self.item_features],axis=1)).to(self.device)
            y = []
            for i in range(self.num_nets):
                y.append(self.nets[i](X).numpy(force=True).reshape(-1))
            y = np.stack(y)
            mean = np.mean(y, axis=0)
            std = np.std(y, axis=0)
            self.uncertainty.append(np.mean(std))
            return mean, std
        
        elif self.mode=='TS':
            X = torch.Tensor(np.concatenate([np.tile(context.reshape(1,-1),(self.K, 1)), self.item_features],axis=1)).to(self.device)
            y = []
            for i in range(self.num_nets):
                y.append(self.nets[i](X).numpy(force=True).reshape(-1))
            y = np.stack(y)
            std = np.std(y, axis=0)
            self.uncertainty.append(np.mean(std))
            return y[self.rng.choice(self.num_nets)], std
    
    def estimate_CTR_batched(self, context):
        X = torch.Tensor(np.concatenate([np.tile(context, (1,self.K)).reshape(-1,self.context_dim), np.tile(self.item_features, (context.shape[0],1))],axis=1)).to(self.device)
        if self.mode=='UCB':
            y = []
            for i in range(self.num_nets):
                y.append(self.nets[i](X).numpy(force=True).reshape(context.shape[0], self.K))
            y = np.stack(y)
            return np.mean(y, axis=0)
        
        elif self.mode=='TS':
            y = []
            for i in range(self.num_nets):
                y.append(self.nets[i](X).numpy(force=True).reshape(context.shape[0], self.K))
            y = np.stack(y)
            return y[self.rng.choice(self.num_nets)]