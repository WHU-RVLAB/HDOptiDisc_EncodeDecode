import matplotlib.pyplot as plt
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

    fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 4))
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
    batch_size, time_step = x.shape
    zero_padding_len = input_size - 1
    
    x = np.concatenate((np.zeros((batch_size, zero_padding_len)), x), axis=1)
    y = np.zeros((batch_size, time_step, input_size))
    
    for bt in range(batch_size):
        for time in range(time_step):
            y[bt, time, :] = x[bt, time:time+input_size]
    
    return y.astype(np.float32)

def evaluation(eval_length, data_eval, model, device):
    dec = torch.zeros((1, 0)).float().to(device)
    for idx in range(data_eval.shape[0]):
        truncation_in = data_eval[idx:idx + 1, : , :]
        with torch.no_grad():
            dec_block = codeword_threshold(model(truncation_in)[:, :eval_length])
        # concatenate the decoding codeword
        dec = torch.cat((dec, dec_block), 1)
        
    return dec.cpu().numpy()

class Dictionary(object):
    def __init__(self, bd_di_coef, tap_bd_num):
        binary_combinations = [list(format(i, f'0{tap_bd_num}b')) for i in range(2 ** tap_bd_num)]
        convolution_results = []

        for binary in binary_combinations:
            binary_array = np.array([int(b) for b in binary]) 
            conv_result = np.sum(binary_array * bd_di_coef) 
            convolution_results.append(conv_result)

        self.lut = np.array(convolution_results)
        print("\nrf signal look up table is:")
        print(self.lut)

    def signal2idx(self, signals):
        idxs = []
        for signal in signals[0, :]:
            differences = np.abs(self.lut - signal)
            idxs.append(np.argmin(differences) + 4)
        return np.array(idxs).reshape(1, -1)
    
    def idx2signal(self, idxs):
        signals = []
        for idx in idxs[0, :]:
            signals.append(self.lut[idx - 4])
        return np.array(signals).reshape(1, -1)

def subsequent_mask(bt_size, size):
    "Mask out subsequent positions."
    attn_shape = (bt_size, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return subsequent_mask == 0

# default src domain
# 4-67: info

# default target domain
# 0-1: info

# 2: start
# 3: stop
# 68 pad
def convert2transformer(data, label=None, num_heads=4, start=2, stop=3, pad=68):
    src = torch.cat([torch.full((data.shape[0], 1), start, dtype=data.dtype), data, torch.full((data.shape[0], 1), stop, dtype=data.dtype)], dim=1)
    src = src.long().contiguous()
    
    if label is None:
        return src
    
    target = torch.cat([torch.full((label.shape[0], 1), start, dtype=label.dtype), label, torch.full((label.shape[0], 1), stop, dtype=label.dtype)], dim=1)
    target_input = target[:, :-1].long().contiguous()
    target_pred = target[:, 1:].long().contiguous()
    
    bsz, tgt_len = target_input.shape
    target_mask  = (target_input != pad).unsqueeze(-2) & subsequent_mask(bsz, tgt_len)
    target_mask = target_mask.repeat(num_heads, 1, 1)
    target_mask = target_mask.view(bsz * num_heads, tgt_len, tgt_len).bool().contiguous()
    
    return src, target_input, target_pred, target_mask
