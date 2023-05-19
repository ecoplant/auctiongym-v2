import argparse
import json
import numpy as np
import os
import shutil
from copy import deepcopy
from tqdm import tqdm
import time

import numpy as np
from gym.spaces import Dict, Box

from Agent import Agent
from Allocator import *
from Auction import Auction
from Bidder import * 
from plot import *

class Buffer:
    def __init__(self):
        self.states = []
        self.items = []
        self.biddings = []
        self.rewards = []
        self.next_states = []
        self.d = []
        self.wins = []
        self.outcomes = []

    def append(self, state, item, bidding, reward, next_state, done, win, outcome):
        self.states.append(state)
        self.items.append(item)
        self.biddings.append(bidding)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.d.append(done)
        self.wins.append(win)
        self.outcomes.append(outcome)
    
    def numpy(self):
        return np.array(self.states), np.array(self.items), np.array(self.biddings), np.array(self.rewards),\
              np.array(self.next_states), np.array(self.d, dtype=bool), np.array(self.wins, dtype=bool), np.array(self.outcomes, dtype=bool)

def draw_features(rng, num_runs, feature_dim, agent_configs):
    run2item_features = {}
    run2item_values = {}

    for run in range(num_runs):
        temp = []
        for k in range(agent_config['num_items']):
            feature = rng.normal(0.0, 1.0, size=feature_dim)
            temp.append(feature)
        run2item_features[run] = np.stack(temp)

        run2item_values[run] = np.ones((agent_config['num_items'],))

    return run2item_features, run2item_values

def set_model_params(rng, CTR_mode, winrate_mode, context_dim, feature_dim):
    if CTR_mode=='bilinear':
        CTR_param = rng.normal(0.0, 1.0, size=(context_dim, feature_dim))
    elif CTR_mode=='MLP':
        d = context_dim * feature_dim
        w1 = rng.normal(0.0, 1.0, size=(d, d))
        b1 = rng.normal(0.0, 1.0, size=(d, 1))
        w2 = rng.normal(0.0, 1.0, size=(d, 1))
        b2 = rng.normal(0.0, 1.0, size=(1, 1))
        CTR_param = (w1, b1, w2, b2)
    
    if winrate_mode=='simulation':
        raise NotImplementedError
    elif winrate_mode=='logistic':
        winrate_param = rng.normal(0.0, 1.0, size=(context_dim+1,))
    elif winrate_mode=='MLP':
        d = context_dim + 1
        w1 = rng.normal(0.0, 1.0, size=(d, d))
        b1 = rng.normal(0.0, 1.0, size=(d, 1))
        w2 = rng.normal(0.0, 1.0, size=(d, 1))
        b2 = rng.normal(0.0, 1.0, size=(1, 1))
        winrate_param = (w1, b1, w2, b2)

    return CTR_param, winrate_param

if __name__ == '__main__':
    # Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to experiment configuration file')
    parser.add_argument('--cuda', type=str, default='0')
    args = parser.parse_args()

    with open('config/training_config.json') as f:
        training_config = json.load(f)
    
    with open(args.config) as f:
        agent_config = json.load(f)

    # Set up Random Generator
    rng = np.random.default_rng(training_config['random_seed'])
    np.random.seed(training_config['random_seed'])

    # training loop config
    num_runs = training_config['num_runs']
    num_episodes  = training_config['num_episodes']
    record_interval = training_config['record_interval']
    update_interval = training_config['update_interval']
    horizon = training_config['horizon']
    budget = training_config['budget']

    # context, item feature config
    context_dim = training_config['context_dim']
    feature_dim = training_config['feature_dim']
    context_dist = training_config['context_distribution']

    # CTR, winrate model
    CTR_mode = training_config['CTR_mode']
    winrate_mode = training_config['winrate_mode']

    # allocator, bidder type
    allocator_type = agent_config['allocator']['type']
    bidder_type = agent_config['bidder']['type']

    os.environ["CUDA_VISIBLE_DEVICES"]= args.cuda
    print("running in {}".format('cuda' if torch.cuda.is_available() else 'cpu'))

    # Parse configuration file
    output_dir = agent_config['output_dir']
    output_dir = output_dir + time.strftime('%y%m%d-%H%M%S')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    shutil.copy(args.config, os.path.join(output_dir, 'agent_config.json'))
    shutil.copy('config/training_config.json', os.path.join(output_dir, 'training_config.json'))

    # memory recording statistics
    reward = np.zeros((num_runs, num_episodes))
    win_rate = np.zeros((num_runs, num_episodes))
    # optimal_bandit_reward = np.zeros((num_runs, num_episodes))
    # optimal_bandit_win_rate = np.zeros((num_runs, num_episodes))
    optimal_selection = np.zeros((num_runs, num_episodes))

    # estimated_CTR = np.zeros((num_runs, num_episodes))

    # bidding_error_buffer = np.zeros((record_interval))

    # num_records = int(num_episodes / record_interval)
    # CTR_RMSE = np.zeros((num_runs, num_records))

    run2item_features, run2item_values = draw_features(rng, num_runs, context_dim, feature_dim, agent_config)

    for run in range(num_runs):
        item_features = run2item_features[run]
        item_values = run2item_values[run]

        CTR_param, winrate_param = set_model_params(rng, CTR_mode, winrate_mode, context_dim, feature_dim)
        CTR_model = CTR(CTR_mode, context_dim, item_features, CTR_param)
        winrate_model = Winrate(winrate_mode, context_dim, winrate_param)

        buffer = Buffer()

        agent = Agent(rng, agent_config['name'], item_features, item_values, context_dim, buffer, agent_config)
        auction = Auction(rng, agent, CTR_model, winrate_model, item_features, item_values, context_dim, context_dist, horizon, budget)

        t = 0
        for i in tqdm(range(num_episodes), desc=f'run {run}'):
            state, _ = auction.reset()
            done = truncated = False
            start = t
            episode_reward = 0
            episode_win = 0
            episode_optimal_selection = 0
            while not (done or truncated):
                item, bidding = agent.bid(state)
                action = {'item' : item, 'bid' : np.array([bidding])}
                next_state, reward, done, truncated, info = auction.step(action)
                buffer.append(state, item, bidding, reward, next_state, (done or truncated), info['win'], info['outcome'])
                t += 1
                episode_reward += reward
                episode_win += float(info['win'])
                episode_optimal_selection += float(info['optimal_selection'])
            
            if (i+1)%update_interval==0:
                agent.update()

            reward[run,i] = episode_reward
            win_rate[run,i] = episode_win / (t - start)
            optimal_selection = episode_optimal_selection / (t - start)

    experiment_id = f"CTR:{CTR_mode}_WR:{winrate_mode}_alloc:{allocator_type}_bid:{bidder_type}"

    reward = average(reward, record_interval)
    optimal_selection = average(optimal_selection.astype(float), record_interval)
    prob_win = average(win_rate, record_interval)

    cumulative_reward = np.cumsum(reward, axis=1)

    reward_df = numpy2df(reward, 'Reward')
    reward_df.to_csv(experiment_id + '/reward.csv', index=False)
    plot_measure(reward_df, 'Reward', record_interval, experiment_id)

    cumulative_reward_df = numpy2df(cumulative_reward, 'Cumulative Reward')
    plot_measure(cumulative_reward_df, 'Cumulative Reward', record_interval, experiment_id)

    optimal_selection_df = numpy2df(optimal_selection, 'Optimal Selection Rate')
    optimal_selection_df.to_csv(experiment_id + '/optimal_selection_rate.csv', index=False)
    plot_measure(optimal_selection_df, 'Optimal Selection Rate', record_interval, experiment_id)

    prob_win_df = numpy2df(prob_win, 'Probability of Winning')
    prob_win_df.to_csv(experiment_id + '/prob_win.csv', index=False)
    plot_measure(prob_win_df, 'Probability of Winning', record_interval, experiment_id)