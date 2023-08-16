import numpy as np
import torch.optim as optim
import copy


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
        return np.array(self.states), np.array(self.actions), np.array(self.rewards), np.array(self.next_states)
    
    def sample(self, batch_size):
        if batch_size > len(self.states):
            self.numpy()
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
    

class Random(Agent):
    def __init__(self, rng, name, context_dim):
        super().__init__(rng, name)
        self.context_dim = context_dim
        self.buffer = Buffer()

    def newdata(self, s, a, r, s_, done, win, outcome):
        self.buffer.append(s,a,r,s_,done,win,outcome)

    def bid(self, state):
        bidding = self.rng.uniform(0,state[-2])
        return bidding

    def update(self):
        pass