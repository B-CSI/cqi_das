import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import numpy as np


def view_event(data, vmin=-0.5, vmax=0.5, decimation=False, title='', freq=50, figsize=(8,5), subplot_ax=None):
    if subplot_ax is not None:
        subplot_ax.tick_params(axis='x', labelsize=15)
        subplot_ax.tick_params(axis='y', labelsize=15)
        #subplot_ax.xticks(ticks)
        subplot_ax.set_xlabel('Time (s))', fontsize=18);
        subplot_ax.set_ylabel('Channel number', fontsize=18)
        if len(title) > 0:
            subplot_ax.set_title(title, fontsize=22)
        first_ch = data.columns[0]
        last_ch = data.columns[-1]
        first_sec = 0 
        if decimation is True:
            last_sec = (data.index[-1] - data.index[0]) / 50
        else:
            last_sec = (data.index[-1] - data.index[0]) / freq
        subplot_ax.imshow(data.T,vmin=vmin, vmax=vmax, aspect = 'auto', cmap='seismic',extent=[first_sec,last_sec,last_ch,first_ch]);
    else:
        fig, ax = plt.subplots(1,1, figsize=figsize)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.set_xlabel('Time (s))', fontsize=12);
        ax.set_ylabel('Channel number', fontsize=12)
        if len(title) > 0:
            ax.set_title(title)
        first_ch = data.columns[0]
        last_ch = data.columns[-1]
        first_sec = 0 
        if decimation is True:
            last_sec = (data.index[-1] - data.index[0]) / 50
        else:
            last_sec = (data.index[-1] - data.index[0]) / freq
        plt.imshow(data.T,vmin=vmin, vmax=vmax, aspect = 'auto', cmap='seismic',extent=[first_sec,last_sec,last_ch,first_ch]);
        plt.show()
        plt.close(fig)

def max_hist(data):
    channel_maxima = data.apply(lambda x: np.argmax(x))
    plt.hist(channel_maxima, bins=100);
    plt.title('Histogram of maximum of each channel in timesteps')

def plot_beta(data, bigbeta_list, title=''):
    # Plot beta against data

    fig, (ax2, ax3) = plt.subplots(1,2, figsize=(25,12))

    x = np.arange(0, len(bigbeta_list),1)

    ax2.plot(bigbeta_list)
    ax2.set_title('Reliability indicator ' + r'$\beta$ in channel order')
    ax2.set_xticks(range(415, len(bigbeta_list)+415, 200), x[::200], rotation = 45)
    ax2.set_xlabel('Channel number')
    ax2.grid()

    ax3.set_title('Overall plot')
    ax3.imshow(data.T,vmin=-1,vmax=1, aspect = 'auto', cmap='seismic');
    ax3.set_xlabel('Time (s)');
    ax3.set_ylabel('Channel number')
    for ax in (ax2, ax3):
        for item in (ax.get_xticklabels() + ax.get_yticklabels() + [ax.title, ax.xaxis.label, ax.yaxis.label]):
            item.set_fontsize(25)
        
    plt.tight_layout()

    plt.show()
    plt.close(fig)

def compare_bad_channels(data, phase_bad_channels):
    #Visualize bad channels on data

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(35,12))

    first_ch = data.columns[0]
    last_ch = data.columns[-1]
    first_sec = data.index[0]
    last_sec = data.index[-1]
    
    ax1.set_title('Overall plot')
    ax1.imshow(data.T,vmin=-1,vmax=1, aspect = 'auto', cmap='seismic',extent=[first_sec,last_sec,last_ch,first_ch]);
    ax1.set_xlabel('Time (s))');
    ax1.set_ylabel('Channel number')

    ax2.set_title('With bad channels')
    ax2.imshow(data.T,vmin=-1,vmax=1, aspect = 'auto', cmap='seismic',extent=[first_sec,last_sec,last_ch,first_ch]);
    ax2.set_xlabel('Time (s)');
    ax2.set_ylabel('Channel number')
    for channel in phase_bad_channels:
        plt.axhline(y=int(channel), color='b', linestyle='-')
    plt.show()
    plt.close(fig)

def compare_good_channels(data, good_channels):
    #Visualize bad channels on data

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(35,12))

    ax1.set_title('Overall plot')
    ax1.imshow(data.T,vmin=-1,vmax=1, aspect = 'auto', cmap='seismic');
    ax1.set_xlabel('Time (s))');
    ax1.set_ylabel('Channel number')

    ax2.set_title('With good channels')
    ax2.imshow(data.T,vmin=-1,vmax=1, aspect = 'auto', cmap='seismic');
    ax2.set_xlabel('Time (s)');
    ax2.set_ylabel('Channel number')
    not_good_channels = [ch for ch in np.arange(data.shape[1]) if ch not in good_channels]
    for channel in not_good_channels:
        plt.axhline(y=int(channel), color='w', linestyle='-')
    plt.show()
    plt.close(fig)

