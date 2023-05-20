import numpy as np

from models import *

class Allocator:
    """ Base class for an allocator """

    def __init__(self, rng, item_features):
        self.rng = rng
        self.item_features = item_features
        self.feature_dim = item_features.shape[1]
        self.K = item_features.shape[0]

    def update(self, contexts, items, outcomes, name):
        pass
    

class OracleAllocator(Allocator):
    """ An allocator that acts based on the true P(click)"""

    def __init__(self, rng, item_features):
        super(OracleAllocator, self).__init__(rng, item_features)

    def set_CTR_model(self, M):
        self.M = M

    def estimate_CTR(self, context):
        return sigmoid(self.item_features @ self.M.T @ context / np.sqrt(context.shape[0]*self.item_features.shape[1]))
    
    def get_uncertainty(self):
        return np.array([0])


class LogisticAllocator(Allocator):
    def __init__(self, rng, item_features, lr, context_dim, num_items, mode, c=0.0, eps=0.1, nu=0.0):
        super().__init__(rng, item_features)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        self.lr = lr

        self.K = num_items
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
    def __init__(self, rng, item_features, lr, batch_size, weight_decay, latent_dim, num_epochs, context_dim, mode, eps=None, prior_var=None):
        super().__init__(rng, item_features)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        self.num_epochs = num_epochs
        self.context_dim = context_dim
        if self.mode=='Epsilon-greedy':
            self.net = NeuralRegression(context_dim+self.feature_dim, latent_dim).to(self.device)
            self.eps = eps
        else:
            raise NotImplementedError
        self.count = 0
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad=True)

    def update(self, contexts, items, outcomes):
        self.count += 1

        X = np.concatenate([contexts, self.item_features[items]], axis=1)
        X, y = torch.Tensor(X).to(self.device), torch.Tensor(outcomes).to(self.device)
        N = X.shape[0]
        if N<10:
            return

        self.net.train()
        batch_size = min(N, self.batch_size)

        for epoch in range(int(self.num_epochs)):
            shuffled_ind = self.rng.choice(N, size=N, replace=False)
            epoch_loss = 0
            for i in range(int(N/batch_size)):
                self.optimizer.zero_grad()
                ind = shuffled_ind[i*batch_size:(i+1)*batch_size]
                X_ = X[ind]
                y_ = y[ind]
                if self.mode=='Epsilon-greedy':
                    loss = self.net.loss(self.net(X_).squeeze(), y_)
                else:
                    loss = self.net.loss(self.net(X_).squeeze(), y_, N)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
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

