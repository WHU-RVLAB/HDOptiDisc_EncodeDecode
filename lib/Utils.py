import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import numpy as np
import torch

def plot_altogether(X, Ys, title, xlabel, ylabel, xtick_interval=None, ytick_interval=None):
    for Y in Ys:
        plt.plot(X, Y['data'], label=Y['label'], color=Y['color'], linestyle=Y.get('linestyle', '-')) 
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True)
    plt.legend()
    if xtick_interval is not None:
        plt.xticks(np.arange(min(X), max(X) + xtick_interval, xtick_interval))
    if ytick_interval is not None:
        plt.yticks(np.arange(min(np.min(Y['data']) for Y in Ys), max(np.max(Y['data']) for Y in Ys) + ytick_interval, ytick_interval))
    plt.show()

def plot_separated(Xs, Ys, titles, xlabels, ylabels, Xtick_intervals=[None], Ytick_intervals=[None]):
    cols = 3 if len(Ys) > 5 else 2 if len(Ys) > 3 else 1
    rows = len(Ys) // cols if len(Ys) % cols == 0 else len(Ys) // cols + 1

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()
    
    if Xtick_intervals == [None]:
        Xtick_intervals = [None] * len(Xs)
    if Ytick_intervals == [None]:
        Ytick_intervals = [None] * len(Ys)

    for i, (X, Y, xtick_interval, ytick_interval) in enumerate(zip(Xs, Ys, Xtick_intervals, Ytick_intervals)):
        ax = axes[i]
        if Y['label'] == 'binary Sequence':
            ax.stem(X, Y['data'], label=Y['label'], basefmt=" ")
        else:
            ax.plot(X, Y['data'], label=Y['label'], color=Y['color'], linestyle=Y.get('linestyle', '-'))
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_title(titles[i] if i < len(titles) else titles[0])
        ax.set_xlabel(xlabels[i] if i < len(xlabels) else xlabels[0])
        ax.set_ylabel(ylabels[i] if i < len(ylabels) else ylabels[0])
        ax.grid(True)
        ax.legend()
        if xtick_interval is not None:
            ax.set_xticks(np.arange(min(X), max(X) + xtick_interval, xtick_interval))
        if ytick_interval is not None:
            ax.set_yticks(np.arange(min(Y), max(Y) + ytick_interval, ytick_interval))

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def plot_eye_diagram(signal, samples_truncation, title, xlabel, ylabel, smooth_factor=10):
    time_original = np.linspace(0, 2, samples_truncation*2)
    time_smooth = np.linspace(0, 2, samples_truncation*2*smooth_factor)
    plt.figure()
    for i in range(0, len(signal['data'])-2*samples_truncation, samples_truncation):
        signal_truncation = signal['data'][i:i+2*samples_truncation]
        cs = CubicSpline(time_original, signal_truncation)
        smoothed_signal = cs(time_smooth)
        plt.plot(time_smooth, smoothed_signal, color=signal['color'], linestyle=signal.get('linestyle', '-'))   
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    plt.axvline(1, color='red', linestyle=':', linewidth=1.2, alpha=0.6) 
    plt.title(title, fontsize=12, pad=20)
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.xlim(0.25, 1.75) 
    plt.tight_layout()
    plt.show()
            
def Fourier_Analysis(signal, sample_periods, T_L, downsample_factor = 1):
    signal = signal[::downsample_factor]

    omega = np.fft.rfftfreq(sample_periods, T_L)
    fft_magnitude = T_L*np.abs(np.fft.rfft(signal, n=sample_periods))

    return T_L*omega, fft_magnitude

def codeword_threshold(x):
    # torch tensor: > 0.5 = 1; <= 0.5 = 0
    x[x > 0.5] = 1
    x[x <= 0.5] = 0
    return x
 
# find one idx for the matched sequence
def find_index(all_array, element):
    all_array = all_array.tolist()
    element = element.tolist()
    if element in all_array:
        return all_array.index(element)
    
def sliding_shape(x, input_size):
    '''
    Input: (1, length) numpy array
    Output: (input_size, length) numpy array
    Mapping: sliding window for each time step
    '''
    with torch.no_grad():
        batch_size, time_step = x.shape
        zero_padding_len = input_size - 1
        
        x = np.concatenate((np.zeros((batch_size, zero_padding_len)), x), axis=1)
        y = np.zeros((batch_size, time_step, input_size))
        
        for bt in range(batch_size):
            for time in range(time_step):
                y[bt, time, :] = x[bt, time:time+input_size]
    
    return y.astype(np.float16)
