import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from scipy.stats import norm

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, scale):
        super().__init__()
        self.input_dim = in_features
        self.output_dim = out_features
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scale = scale
        
        self.weight_mu = nn.Parameter(torch.empty((out_features, in_features)))
        self.weight_rho = nn.Parameter(torch.empty((out_features, in_features)))
        
        self.bias_mu = nn.Parameter(torch.empty((out_features,)))
        self.bias_rho = nn.Parameter(torch.empty((out_features,)))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mu)
        nn.init.uniform_(self.weight_rho, -0.02, 0.02)
        nn.init.uniform_(self.bias_mu, -np.sqrt(3/self.weight_mu.size(1)), np.sqrt(3/self.weight_mu.size(1)))
        nn.init.uniform_(self.bias_rho, -0.02, 0.02)
        
    def forward(self, input, sample=True):
        if self.training or sample:
            weight_sigma = torch.log1p(torch.exp(self.weight_rho))
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            weight = self.weight_mu + self.scale * weight_sigma * torch.randn_like(self.weight_mu)
            bias = self.bias_mu + self.scale * bias_sigma * torch.randn_like(self.bias_mu)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(input, weight, bias)
    
    def get_uncertainty(self):
        with torch.no_grad():
            weight_sigma = torch.log1p(torch.exp(self.weight_rho)).numpy(force=True)
            bias_sigma = torch.log1p(torch.exp(self.bias_rho)).numpy(force=True)
        return np.concatenate([weight_sigma.reshape(-1), bias_sigma.reshape(-1)])

class Logistic:
    def __init__(self, param):
        self.w = param
    
    def __call__(self, x):
        return sigmoid(x @ self.w / np.sqrt(len(x)))

class MLP:
    def __init__(self, param):
        self.w1, self.b1, self.w2, self.b2 = param
    
    def __call__(self, x):
        x = sigmoid(x @ self.w1 + self.b1)
        return sigmoid(x @ self.w2 + self.b2)
    
class Bilinear:
    def __init__(self, param):
        self.M = param
    
    def __call__(self, context, features):
        return sigmoid(features @ self.M.T @ context / np.sqrt(len(context))).reshape(-1)

class Winrate:
    def __init__(self, mode, context_dim, param=None, CTR_model=None):
        self.mode = mode
        self.context_dim = context_dim
        if mode=='simulation':
            self.param = param
            self.num_competitors = len(param)
            self.CTR_models = []
            for i in range(self.num_competitors):
                ctr = copy.deepcopy(CTR_model)
                ctr.item_features = param[i]
                self.CTR_models.append(ctr)
        elif mode=='logistic':
            self.model = Logistic(param)
        elif mode=='MLP':
            self.model = MLP(param)
    
    def __call__(self, context, bid):
        if self.mode=='simulation':
            if len(bid)==1:
                prob = 1.0
                for i in range(self.num_competitors):
                    mean = np.max(self.CTR_models[i](context) * 0.6)
                    prob *= norm.cdf(bid.item(), loc=mean, scale=0.2)
                return prob
            else:
                prob_list = []
                for j in range(len(bid)):
                    prob = 1.0
                    for i in range(self.num_competitors):
                        mean = np.max(self.CTR_models[i](context) * 0.6)
                        prob *= norm.cdf(bid[j], loc=mean, scale=0.2)
                    prob_list.append(prob)
                return np.array(prob_list)
        else:
            if len(bid)==1:
                return self.model(np.concatenate([context, bid]))
            else:
                x =np.concatenate([
                    np.tile(context.reshape(1,-1), (len(bid),1)),
                    bid.reshape(-1,1)
                ], axis=1)
                return self.model(x)
        
class CTR:
    def __init__(self, mode, context_dim, item_features, param):
        self.mode = mode
        self.d = context_dim
        self.item_features = item_features
        self.K = item_features.shape[0]
        self.h = item_features.shape[1]
        if mode=='bilinear':
            self.model = Bilinear(param)
        elif mode=='MLP':
            pass
    
    def __call__(self, context):
        if self.mode=='bilinear':
            return self.model(context, self.item_features)
        elif self.mode=='MLP':
            pass

class LogisticRegression(nn.Module):
    def __init__(self, context_dim, items, mode, rng, lr, c=1.0, nu=1.0):
        super().__init__()
        self.rng = rng
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr

        self.items_np = items
        self.items = torch.Tensor(items).to(self.device)
        self.K = items.shape[0] # number of items
        self.d = context_dim
        self.h = items.shape[1] # item feature dimension
        self.c = c
        self.nu = nu

        self.M = nn.Parameter(torch.Tensor(self.d, self.h)) # CTR = sigmoid(context @ M @ item_feature)
        nn.init.kaiming_uniform_(self.M)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.BCE = torch.nn.BCELoss(reduction='sum')
        self.uncertainty = []
        self.S0_inv = torch.Tensor(np.eye(self.h*self.d)).to(self.device)
        self.S_inv = np.eye(self.h*self.d)
        self.S = torch.Tensor(np.eye(self.h*self.d)).to(self.device)
        self.sqrt_S = torch.Tensor(np.eye(self.h*self.d)).to(self.device)

        self.uncertainty = []

    def forward(self, X, A):
        return torch.sigmoid(torch.sum(F.linear(X, self.M.T)*self.items[A], dim=1))
    
    def update(self, contexts, items, outcomes, t):
        X = torch.Tensor(contexts).to(self.device)
        A = torch.LongTensor(items).to(self.device)
        y = torch.Tensor(outcomes).to(self.device)
        N = X.size(0)

        epochs = 10
        for epoch in range(int(epochs)):
            self.optimizer.zero_grad()
            loss = self.loss(X, A, y)
            loss.backward()
            self.optimizer.step()

        # epochs = 10
        # batch_size = min(N, 256)
        # for epoch in range(int(epochs)):
        #     ind = self.rng.choice(N, size=batch_size, replace=False)
        #     X_, A_, y_ = X[ind], A[ind], y[ind]
        #     self.optimizer.zero_grad()
        #     loss = self.loss(X_, A_, y_)
        #     loss.backward()
        #     self.optimizer.step()

        if t%5==0:
            y = self(X, A).numpy(force=True)
            y = y * (1 - y)
            contexts = contexts.reshape(-1,self.d)
            self.S_inv = self.S0_inv.numpy(force=True)
            for i in range(contexts.shape[0]):
                context = contexts[i]
                item_feature = self.items_np[A[i]]
                phi = np.outer(context, item_feature).reshape(-1)
                self.S_inv += y[i] * np.outer(phi, phi)
            self.S = torch.Tensor(np.diag(np.diag(self.S_inv)**(-1))).to(self.device)
            self.sqrt_S = torch.Tensor(np.diag(np.sqrt(np.diag(self.S_inv)+1e-6)**(-1))).to(self.device)

    def loss(self, X, A, y):
        y_pred = self(X, A)
        m = self.flatten(self.M)
        return self.BCE(y_pred, y) + torch.sum(m.T @ self.S0_inv @ m / 2)
    
    def estimate_CTR(self, context):
        # context @ M @ item_feature = M * outer(context, item_feature)
        X = []
        context = context.reshape(-1)
        for i in range(self.K):
            X.append(np.outer(context, self.items_np[i]).reshape(-1))
        X = torch.Tensor(np.stack(X)).to(self.device)
        with torch.no_grad():
            if self.mode=='UCB':
                m = self.flatten(self.M)
                uncertainty = self.c * torch.sum((X @ self.S) * X, dim=1).numpy(force=True).reshape(-1)
                mean = torch.sigmoid(X @ m).numpy(force=True).reshape(-1)
                self.uncertainty.append(np.mean(uncertainty))
                return mean, uncertainty
            elif self.mode=='TS':
                m = self.flatten(self.M)
                y = []
                for i in range(5):
                    m_ = m + self.nu * self.sqrt_S @ torch.Tensor(self.rng.normal(0,1,self.d*self.h).reshape(-1,1)).to(self.device)
                    y.append(torch.sigmoid(X @ m_).numpy(force=True).reshape(-1))
                y = np.stack(y)
                uncertainty = np.std(y, axis=0)
                self.uncertainty.append(np.mean(uncertainty))
                return y[self.rng.choice(5)], uncertainty
            else:
                m = self.flatten(self.M)
                return torch.sigmoid(X @ m).numpy(force=True).reshape(-1)
    
    def estimate_CTR_batched(self, context):
        X = []
        for i in range(context.shape[0]):
            for j in range(self.K):
                X.append(np.outer(context[i], self.items_np[j]).reshape(-1))
        X = torch.Tensor(np.stack(X)).to(self.device)
        m = self.flatten(self.M)
        return torch.sigmoid(X @ m).numpy(force=True)

    # def get_uncertainty(self):
    #     S_ = self.S.numpy(force=True)
    #     eigvals = np.linalg.eigvals(S_).reshape(-1)
    #     return eigvals.real

    def flatten(self, tensor):
        return torch.reshape(tensor, (tensor.shape[0]*tensor.shape[1], -1))
    
    
class NeuralRegression(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d = input_dim
        self.h = latent_dim
        self.feature = nn.Linear(self.d, self.h)
        self.head = nn.Linear(self.h, 1)
        self.BCE = nn.BCELoss()
        self.eval()

    def forward(self, x):
        x = torch.relu(self.feature(x))
        return torch.sigmoid(self.head(x))

    def loss(self, predictions, labels):
        return self.BCE(predictions, labels)


# ==========winrate estimators==========

class NeuralWinRateEstimator(nn.Module):
    def __init__(self, context_dim, skip_connection=True):
        super().__init__()
        self.skip_connection = skip_connection
        self.H = 32
        if self.skip_connection:
            self.linear1 = nn.Linear(context_dim, self.H-1)
        else:
            self.linear1 = nn.Linear(context_dim+1, self.H)
        self.linear2 = nn.Linear(self.H, 1)
        self.BCE = nn.BCEWithLogitsLoss()
        self.eval()

    def forward(self, x, sample=False):
        if self.skip_connection:
            context = x[:,:-1]
            gamma = x[:,-1].reshape(-1,1)
            hidden = torch.relu(self.linear1(context))
            hidden_ = torch.concat([hidden, gamma], dim=-1)
            return torch.sigmoid(self.linear2(hidden_))
        else:
            hidden = torch.relu(self.linear1(x))
            return torch.sigmoid(self.linear2(hidden))
    
    def loss(self, x, y):
        if self.skip_connection:
            context = x[:,:-1]
            gamma = x[:,-1].reshape(-1,1)
            hidden = torch.relu(self.linear1(context))
            hidden_ = torch.concat([hidden, gamma], dim=-1)
            logit = self.linear2(hidden_)
        else:
            hidden = torch.relu(self.linear1(x))
            logit = self.linear2(hidden)
        return self.BCE(logit, y)


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super().__init__()
        self.c1 = nn.Linear(input_dim, hidden_dim)
        self.c2 = nn.Linear(hidden_dim, 1)

        self.a1 = nn.Linear(input_dim, hidden_dim)
        self.a2 = nn.Linear(hidden_dim, action_dim)

        self.dist = torch.distributions.Categorical
    
    def v(self, x):
        x = F.relu(self.c1(x))
        return self.c2(x)
    
    def pi(self, x):
        x = F.relu(self.a1(x))
        return self.a2(x)

    def act(self, x):
        self.eval()
        logits = self.pi(x)
        prob = F.softmax(logits, -1).data
        m = self.dist(prob)
        return m.sample().item()