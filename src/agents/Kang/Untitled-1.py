class TD3BC(Agent):
    def __init__(self, rng, name, context_dim):
        super().__init__(rng,name)
        self.context_dim = context_dim
        self.buffer = Buffer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open('src/agents/TD3BC/config.json') as f:
            config = json.load(f)
        
        self.critic = Critic(self.context_dim+2+1, config['hidden_dim'], self.context_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr = config['lr'])
        self.alpha = config['alpha']
        self.actor = Actor(self.context_dim+2+1, config['hidden_dim'], self.context_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.noise_init = config['noise_init']
        self.noise_min = config['eps_min']
        self.noise_decay = config['eps_decay']
        self.mean_noise = 0.0

        self.actor_optim = optim.Adam(self.actor.parameters(), lr = config['lr'])
        self.batch_size = config['batch_size']
        self.num_grad_steps = config['num_grad_steps']
        self.tau = config['tau']
        self.episode = 0
    
    def newdata(self, s, a, r, s_, done, win, outcome):
        self.buffer.append(s,a,r,s_,done,win,outcome)

    def bid(self,state):
        self.step += 1
        context = state[:self.context_dim]
        resource = state[self.context_dim:]

        x = torch.Tensor(np.concatenate([context, np.array([estimated_CTR]), resource])).to(self.device)
        with torch.no_grad():
            bidding = np.clip()

        best_item = self.rng.choice(self.num_items, 1).item()
        value = self.item_values[best_item]




