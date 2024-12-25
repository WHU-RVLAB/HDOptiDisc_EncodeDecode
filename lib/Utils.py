import matplotlib.pyplot as plt
import numpy as np

def plot_altogether(X, Ys, title, xlabel, ylabel, xtick_interval=None, ytick_interval=None):
    for Y in Ys:
        plt.plot(X, Y['data'], label=Y['label'], color=Y['color'])
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
    cols = 2 if len(Ys) > 3 else 1
    rows = len(Ys) // cols if len(Ys) % cols == 0 else len(Ys) // cols + 1

    fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 4))
    axes = axes.flatten()
    
    if Xtick_intervals == [None]:
        Xtick_intervals *= len(Xs)
    if Ytick_intervals == [None]:
        Ytick_intervals *= len(Xs)

    for i, (X, Y, xtick_interval, ytick_interval) in enumerate(zip(Xs, Ys, Xtick_intervals, Ytick_intervals)):
        ax = axes[i]
        if Y['label'] == 'binary Sequence':
            ax.stem(X, Y['data'], label=Y['label'], basefmt=" ", use_line_collection=True)
        else:
            ax.plot(X, Y['data'], label=Y['label'], color=Y['color'])
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
            ax.set_yticks(np.arange(min(np.min(Y['data']) for Y in Ys), max(np.max(Y['data']) for Y in Ys) + ytick_interval, ytick_interval))

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def Fourier_Analysis(signal, sample_periods, T_L, downsample_factor = 1):
    signal = signal[::downsample_factor]

    omega = np.fft.rfftfreq(sample_periods, T_L)
    fft_magnitude = T_L*np.abs(np.fft.rfft(signal, n=sample_periods))

    return T_L*omega, fft_magnitude
  
# find one idx for the matched sequence
def find_index(all_array, element):
    all_array = all_array.tolist()
    element = element.tolist()
    if element in all_array:
        return all_array.index(element)