import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_scores(scores, rolling_window=10, save_fig=False):
    """Plot scores and optional rolling mean using specified window."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.title(f'scores')
    plt.xlabel('Episode #')
    plt.ylabel('Score')
    plt.plot(pd.Series(scores).rolling(rolling_window).mean());

    if save_fig:
        plt.savefig(f'scores.png', bbox_inches='tight', pad_inches=0)