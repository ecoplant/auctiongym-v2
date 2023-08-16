import numpy as np

CONTEXT_LOW = -5.0
CONTEXT_HIGH = 5.0

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class Auction:
    def __init__(self, rng, agents, CTR_param, item_features, context_dim, context_dist, horizon, budget):
        super().__init__()
        self.rng = rng
        self.agents = agents
        self.num_agents = len(agents)

        self.CTR_param = CTR_param
        self.item_features = item_features
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
        self.remaining_steps = self.horizon * np.ones((self.num_agents))
        self.remaining_budget = self.budget * np.ones((self.num_agents))

        return np.concatenate([np.tile(self.context,(self.num_agents,1)), self.remaining_budget.reshape(-1,1), self.remaining_steps.reshape(-1,1)],axis=1), {}

    def step(self, actions):
        CTR = []
        for j in range(self.num_agents):
            CTR.append(sigmoid(self.context @ self.CTR_param @ self.item_features[j] / np.sqrt(self.context_dim)).item())
        outcome = self.rng.binomial(1, p=CTR)

        win = np.zeros((self.num_agents))
        win[np.argmax(actions)] = 1
        reward =  outcome * win

        info = {
            'win' : win,
            'outcome' : outcome
        }
        self.remaining_budget -= win * np.array(actions)
        self.remaining_steps -= 1
        self.context = self.generate_context()
        return np.concatenate([np.tile(self.context,(self.num_agents,1)), self.remaining_budget.reshape(-1,1), self.remaining_steps.reshape(-1,1)],axis=1), \
                                reward, np.logical_or(self.remaining_budget<1e-3, self.remaining_steps==0), info