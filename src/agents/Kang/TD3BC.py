import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import json

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def parse_kwargs(kwargs):
    parsed = ','.join([f'{key}={value}' for key, value in kwargs.items()])
    return ',' + parsed if parsed else ''

class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, context_dim):
        super().__init__()
        self.context_dim = context_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim-3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim-3)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        if x.dim()>1:
            r = x[:,self.context_dim:]
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(torch.concat([x, r], dim=1)))
            return torch.sigmoid(self.fc3(torch.concat([x, r], dim=1)))
        else:
            r = x[self.context_dim:]
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(torch.concat([x, r])))
            return torch.sigmoid(self.fc3(torch.concat([x, r])))

class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim, context_dim):
        super().__init__()
        self.context_dim = context_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim-3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim-3)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.fc4 = nn.Linear(input_dim, hidden_dim-3)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim-3)
        self.fc6 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        r = x[:,self.context_dim:]
        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(torch.concat([q1, r], dim=1)))
        q1 = self.fc3(torch.concat([q1, r], dim=1))

        q2 = F.relu(self.fc4(x))
        q2 = F.relu(self.fc5(torch.concat([q2, r], dim=1)))
        q2 = self.fc6(torch.concat([q2, r], dim=1))

        return q1, q2
    
    def Q1(self, x):
        r = x[:,self.context_dim:]
        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(torch.concat([q1, r], dim=1)))
        return self.fc3(torch.concat([q1, r], dim=1))

class Allocator:
    """ Base class for an allocator """

    def __init__(self, rng, item_features):
        self.rng = rng
        self.item_features = item_features
        self.feature_dim = item_features.shape[1]
        self.K = item_features.shape[0]

    def update(self, contexts, items, outcomes, t):
        pass

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

        if N<1000:
            epochs = 10
        else:
            epochs = 1
        for epoch in range(int(epochs)):
            self.optimizer.zero_grad()
            loss = self.loss(X, A, y)
            loss.backward()
            self.optimizer.step()

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

    def flatten(self, tensor):
        return torch.reshape(tensor, (tensor.shape[0]*tensor.shape[1], -1))

class Buffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.wins = []
        self.outcomes = []

    def append(self, state, action, reward, next_state, done, win, outcome):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.wins.append(win)
        self.outcomes.append(outcome)
    
    def numpy(self):
        return np.array(self.states), np.array(self.actions), np.array(self.rewards), np.array(self.next_states), \
                np.array(self.dones), np.array(self.wins), np.array(self.outcomes)
    
    def sample(self, batch_size):
        if batch_size > len(self.states):
            return np.array(self.states), np.array(self.actions), np.array(self.rewards), np.array(self.next_states), \
                np.array(self.dones)
        else:
            inds = np.random.choice(len(self.states), batch_size, replace = False)
            return np.array([self.states[idx] for idx in inds]), np.array([self.actions[idx] for idx in inds]), \
                np.array([self.rewards[idx] for idx in inds]), np.array([self.next_states[idx] for idx in inds]), np.array([self.dones[idx] for idx in inds], dtype=bool)

class Agent:
    def __init__(self, rng, name):
        self.rng = rng
        self.name = name
        self.context_dim = None
        self.step = 0

    def newdata(self, s, a, r, s_, done, win, outcome):
        '''
        This function is called at every step.

        Inputs
        s[numpy array] : state   a[float] : action  r[float] : reward
        s_[numpy array] : next state    done[bool] : the episode is done
        win[float; zero or one] : the agent won the auction at the step
        outcome[int; zero or one] : the conversion event occured at the step(this can be 1 even if the agent lost the auction)

        This function returns nothing
        '''
        raise NotImplementedError

    def bid(self, state):
        '''
        This function is called at every step, even if the agent has no budget.

        Input
        s[numpy array] : state

        Output : bidding[float] (bidding will be clipped in main.py not to exceed current budget)
        '''
        raise NotImplementedError
    
    def update(self):
        '''
        This function is called at the end of episodes.
        '''
        raise NotImplementedError
    

class TD3BC(Agent):
    def __init__(self, rng, name, item_features, context_dim):
        super().__init__(rng, name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.context_dim = context_dim
        self.item_features = item_features
        with open('src/agents/Kang/config.json') as f:
            config = json.load(f)

        self.buffer = Buffer()

        self.allocator = eval(f"{config['allocator']['type']}(rng=rng, item_features=item_features, context_dim=context_dim{parse_kwargs(config['allocator']['kwargs'])})")
        
        self.critic = Critic(self.context_dim+2+1, config['hidden_dim'], self.context_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr = config['lr'])
        self.alpha = config['alpha']

        self.exploration_strategy = config['exploration_strategy']
        self.episodic_exploration = config['episodic_exploration']
        if self.exploration_strategy=='Eps-Greedy':
            self.actor = Actor(self.context_dim+2+1, config['hidden_dim'], self.context_dim).to(self.device)
            self.actor_target = copy.deepcopy(self.actor)
            self.eps_init = self.eps = config['eps_init']
            self.eps_min = config['eps_min']
            self.eps_decay = config['eps_decay']
        elif self.exploration_strategy=='Gaussian Noise':
            self.actor = Actor(self.context_dim+2+1, config['hidden_dim'], self.context_dim).to(self.device)
            self.actor_target = copy.deepcopy(self.actor)
            self.noise_init = self.noise = config['noise_init']
            self.noise_min = config['noise_min']
            self.noise_decay = config['noise_decay']
            if self.episodic_exploration:
                self.mean_noise_var = self.mean_noise = config['mean_noise_init']
                self.mean_noise_decay = config['mean_noise_decay']
            else:
                self.mean_noise = 0.0
        elif self.exploration_strategy=='Noise Injection':
            self.actor = Actor(self.context_dim+2+1, config['hidden_dim'], self.context_dim).to(self.device)
            self.actor_target = copy.deepcopy(self.actor)
            self.noise_init = self.noise = config['noise_init']
            self.noise_min = config['noise_min']
            self.noise_decay = config['noise_decay']

        self.actor_optim = optim.Adam(self.actor.parameters(), lr = config['lr'])
        self.batch_size = config['batch_size']
        self.num_grad_steps = config['num_grad_steps']
        self.tau = config['tau']

        self.episode = 0

    def newdata(self, s, a, r, s_, done, win, outcome):
        self.buffer.append(s,a,r,s_,done,win,outcome)
    
    def set_exploration_param(self):
        if self.exploration_strategy=='Eps-Greedy':
            self.eps = np.maximum(self.eps*self.eps_decay, self.eps_min)
        
        elif self.exploration_strategy=='Gaussian Noise':
            if self.episodic_exploration:
                self.mean_noise = self.rng.normal(0, self.mean_noise_var)
                self.mean_noise_var = self.mean_noise_var*self.mean_noise_decay
            self.noise = np.maximum(self.noise*self.noise_decay, self.noise_min)

        elif self.exploration_strategy=='Noise Injection':
            if self.episodic_exploration:
                self.clone_actor = copy.deepcopy(self.actor)
                state_dict = self.clone_actor.state_dict()
                for name, param in state_dict.items():
                    transformed_param = param + param*torch.randn_like(param)*self.noise
                    param.copy_(transformed_param)
            self.noise = np.maximum(self.noise*self.noise_decay, self.noise_min)
    
    def bid(self, state):
        context = state[:self.context_dim]
        resource = state[self.context_dim:]
        estimated_CTR, _ = self.allocator.estimate_CTR(context)
        estimated_CTR = estimated_CTR[0]

        if self.exploration_strategy=='Eps-Greedy':
            if self.rng.uniform(0, 1) < self.eps:
                bid = self.rng.random()
            else:
                x = torch.Tensor(np.concatenate([context, np.array([estimated_CTR]), resource])).to(self.device)
                bid = self.actor(x).item()

        elif self.exploration_strategy=='Gaussian Noise':
            x = torch.Tensor(np.concatenate([context, np.array([estimated_CTR]), resource])).to(self.device)
            bid = np.clip(self.actor(x).item(), 0.0, 1.0)
            bid = np.clip(bid + self.mean_noise + self.noise * self.rng.normal(), 0.0, 1.0)

        elif self.exploration_strategy=='Noise Injection':
            x = torch.Tensor(np.concatenate([context, np.array([estimated_CTR]), resource])).to(self.device)
            if not self.episodic_exploration:
                self.clone_actor = copy.deepcopy(self.actor)
                state_dict = self.clone_actor.state_dict()
                for name, param in state_dict.items():
                    transformed_param = param + param*torch.randn_like(param)*self.noise
                    param.copy_(transformed_param)
            bid = np.clip(self.clone_actor(x).item(), 0.0, 1.0)

        return np.clip(bid, 0.0, state[-2])

    def update(self):
        self.episode += 1
        self.set_exploration_param()
        
        states, biddings, rewards, next_states, dones, wins, outcomes = self.buffer.numpy()
        contexts = states[:, :self.context_dim]
        self.allocator.update(contexts[wins==1], np.zeros((int(wins.sum()))), outcomes[wins==1], 0)

        if len(self.buffer.states)<self.batch_size:
            return
        
        criterion = nn.MSELoss()
        
        # update actor and critic using real data
        for i in range(self.num_grad_steps):
            states, biddings, rewards, next_states, dones = self.buffer.sample(self.batch_size)
            contexts = states[:, :self.context_dim]
            resources = states[:,self.context_dim:]
            estimated_CTRs = np.max(self.allocator.estimate_CTR_batched(contexts), axis=1).reshape(-1,1)
            x = torch.Tensor(np.concatenate([states, biddings.reshape(-1,1)], axis=1)).to(self.device)
            q1, q2 = self.critic(x)

            with torch.no_grad():
                next_contexts = next_states[:, :self.context_dim]
                next_resources = next_states[:, self.context_dim:]
                next_estimated_CTRs = self.allocator.estimate_CTR_batched(next_contexts)
                next_items = np.argmax(next_estimated_CTRs, axis=1)
                next_estimated_CTRs = np.max(next_estimated_CTRs, axis=1).reshape(-1,1)

                x = torch.Tensor(np.concatenate([next_contexts, next_estimated_CTRs, next_resources], axis=1)).to(self.device)
                next_biddings = self.actor_target(x).numpy(force=True).reshape(-1,1)

                x = torch.Tensor(np.concatenate([next_contexts, next_resources, next_biddings], axis=1)).to(self.device)
                target_q1, target_q2 = self.critic_target(x)
                target_q = torch.min(target_q1, target_q2)
                target_q = torch.Tensor(rewards).to(self.device) + torch.Tensor(1.0-dones).to(self.device) * target_q.squeeze()
            #compute critic loss
            loss = criterion(q1.squeeze(), target_q) + criterion(q2.squeeze(), target_q)
            #optimize the critic
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()
            #delayed policy updates
            if self.episode % 2 == 0:
                x = torch.Tensor(np.concatenate([contexts, estimated_CTRs, resources], axis=1)).to(self.device)
                biddings = self.actor(x)
                x = torch.concat([torch.Tensor(states).to(self.device), biddings], dim=1)
                #compute actor loss
                pi = self.actor(x)
                x = torch.concat([torch.Tensor(states).to(self.device), pi], dim=1)
                Q = self.critic.Q1(x)
                lmbda = self.alpha/Q.abs().mean().detach()
                loss = -lmbda * Q.mean() + criterion(pi, biddings)

                #optimize the actor
                self.actor_optim.zero_grad()
                loss.backward()
                self.actor_optim.step()
        
        # polyak average network weights
        if self.episode % 2 == 0:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
