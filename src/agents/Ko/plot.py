import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FIGSIZE = (8, 5)
FONTSIZE = 14

def average(array, window):
    runs = array.shape[0]
    len = array.shape[1]
    num_agents = array.shape[2]
    comp_len = int(len/window)
    comp_array = np.empty((runs, comp_len,num_agents))
    for r in range(runs):
        for i in range(comp_len):
            comp_array[r,i] = np.mean(array[r,window*i:window*(i+1)],axis=0)
    return comp_array

def numpy2df(array, agents, measure_name):
    df_rows = {'Run': [], 'Step': [], 'Agent':[], measure_name: []}
    for (run,step,agent), measure in np.ndenumerate(array):
        df_rows['Run'].append(run)
        df_rows['Step'].append(step)
        df_rows['Agent'].append(agents[agent].name)
        df_rows[measure_name].append(measure)
    return pd.DataFrame(df_rows)

def list_of_numpy2df(dict, measure_name):
    df_rows = {'Run': [], 'Step': [], 'Index': [], measure_name: []}
    for run, list in dict.item():
        for step, array in list:
            for (ind), measure in np.ndenumerate(array):
                df_rows['Run'].append(run)
                df_rows['Step'].append(step)
                df_rows['Index'].append(ind)
                df_rows[measure_name].append(measure)
        return pd.DataFrame(df_rows)

def plot_measure(df, measure_name, window_size, output_dir):
    fig, axes = plt.subplots(figsize=FIGSIZE)
    plt.title(f'{measure_name} Over Time', fontsize=FONTSIZE + 2)
    sns.lineplot(data=df, x="Step", y=measure_name, hue="Agent", ax=axes)
    min_measure = min(0.0, np.min(df[measure_name]))
    max_measure = max(0.0, np.max(df[measure_name]))
    max_step = max(df['Step']) + 1

    plt.xticks(ticks=np.linspace(0,max_step,11).astype(int),
               labels=np.linspace(0,max_step*window_size,11).astype(int),fontsize=FONTSIZE-2)
    plt.ylabel(f'{measure_name}', fontsize=FONTSIZE)
    plt.xlabel("Episode", fontsize=FONTSIZE)
    factor = 1.1 if min_measure < 0 else 0.9
    plt.ylim(min_measure * factor, max_measure * 1.1)
    plt.yticks(fontsize=FONTSIZE - 2)
    plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{measure_name.replace(' ', '_')}.png", bbox_inches='tight')

def boxplot(df, measure_name, window_size, output_dir):
    max_step = max(df['Step']) + 1

    fig, axes = plt.subplots(figsize=FIGSIZE)
    plt.title(f'{measure_name} Distribution', fontsize=FONTSIZE + 2)
    sns.boxplot(data=df, x="Step", y=measure_name, ax=axes)
    plt.xticks(ticks=np.linspace(0,max_step,11).astype(int),
               labels=np.linspace(0,max_step*window_size,11).astype(int),fontsize=FONTSIZE-2)
    plt.ylabel(measure_name, fontsize=FONTSIZE)
    plt.xlabel("Time", fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE - 2)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{measure_name.replace(' ', '_')}.png", bbox_inches='tight')