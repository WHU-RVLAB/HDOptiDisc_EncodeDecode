import sys
import os
import numpy as np
sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)))
from Disk_Response import BD_symbol_response, HDDVD_symbol_response
from Utils import plot_altogether, plot_separated, Fourier_Analysis
sys.path.pop()

def sinc(x):
    return np.sinc(x / np.pi)

def partial_response(PR_coefs, bit_periods = 10, bits_freq = 132e6, upsample_factor = 10):
    T = 1 / bits_freq
    t = np.linspace(-bit_periods * T, bit_periods * T, 2*upsample_factor*bit_periods + 1)
    
    target_pr = np.zeros_like(t)
    for n in range(len(PR_coefs)):
        target_pr += PR_coefs[n] * sinc((t - n * T) * np.pi / T)
    
    return t/T, target_pr
    
if __name__ == '__main__':
    BD_bits_freq = 132e6
    upsample_factor = 10
    Normalized_t1, symbol_response = BD_symbol_response(bit_periods = 150, bits_freq = BD_bits_freq, upsample_factor = upsample_factor)
    PR1_coefs = [1, 2, 2, 2, 1]
    Normalized_t2, target_pr1 = partial_response(PR_coefs = PR1_coefs, bit_periods = 10, bits_freq = BD_bits_freq, upsample_factor = upsample_factor)
    PR2_coefs = [1, 1, 1, 1]
    Normalized_t3, target_pr2 = partial_response(PR_coefs = PR2_coefs, bit_periods = 10, bits_freq = BD_bits_freq, upsample_factor = upsample_factor)
    PR3_coefs = [1, 2, 3, 4, 3, 2, 1]
    Normalized_t4, target_pr3 = partial_response(PR_coefs = PR3_coefs, bit_periods = 10, bits_freq = BD_bits_freq, upsample_factor = upsample_factor)
    Xs = [
        Normalized_t1,
        Normalized_t2,
        Normalized_t3,
        Normalized_t4
    ]
    Ys = [
    {'data': symbol_response, 'label': 'BDs Symbol Response', 'color': 'blue'},
    {'data': target_pr1, 'label': 'PR [1 2 2 2 1] Response', 'color': 'red'},
    {'data': target_pr2, 'label': 'PR [1 1 1 1] Response', 'color': 'green'},
    {'data': target_pr3, 'label': 'PR [1 2 3 4 3 2 1] Response', 'color': 'black'},
    ]
    titles = [
        'BDs Symbol Response',
        'Target PR [1 2 2 2 1] Response',
        'Target PR [1 1 1 1] Response',
        'Target PR [1 2 3 4 3 2 1] Response',
    ]
    xlabels = ["Time (t/T)"]
    ylabels = ["Amplitude"]
    plot_separated(
        Xs=Xs, 
        Ys=Ys, 
        titles=titles,     
        xlabels=xlabels, 
        ylabels=ylabels
    )
    bit_periods = max((len(symbol_response)-1)//2, (len(target_pr1)-1)//2)
    downsample_factor = 10
    freqs, BD_symbol_response_fft_magnitude = Fourier_Analysis(symbol_response, bit_periods = bit_periods, bits_freq = BD_bits_freq, downsample_factor = downsample_factor)
    _, target_pr1_fft_magnitude = Fourier_Analysis(target_pr1, bit_periods = bit_periods, bits_freq = BD_bits_freq, downsample_factor = downsample_factor)
    _, target_pr2_fft_magnitude = Fourier_Analysis(target_pr2, bit_periods = bit_periods, bits_freq = BD_bits_freq, downsample_factor = downsample_factor)
    _, target_pr3_fft_magnitude = Fourier_Analysis(target_pr3, bit_periods = bit_periods, bits_freq = BD_bits_freq, downsample_factor = downsample_factor)
    Ys = [
    {'data': BD_symbol_response_fft_magnitude / max(BD_symbol_response_fft_magnitude), 'label': 'BDs Symbol Response', 'color': 'blue'},
    {'data': target_pr1_fft_magnitude / max(target_pr1_fft_magnitude), 'label': 'PR [1 2 2 2 1] Response', 'color': 'red'},
    {'data': target_pr2_fft_magnitude / max(target_pr2_fft_magnitude), 'label': 'PR [1 1 1 1] Response', 'color': 'green'},
    {'data': target_pr3_fft_magnitude / max(target_pr3_fft_magnitude), 'label': 'PR [1 2 3 4 3 2 1] Response', 'color': 'black'},
    ]
    Normalized_f = freqs/max(freqs)
    plot_altogether(
        X=Normalized_f, 
        Ys=Ys, 
        title='Frequency Spectrum of Symbol Responses and Target PR Responses for BDs',     
        xlabel="Normalized Frequency", 
        ylabel="Normalized Amplitude",
        xtick_interval=0.1,
        ytick_interval=0.1
    )
    # HDDVD_bits_freq = 64.8e6
    HDDVD_bits_freq = 3.2e6 
    upsample_factor = 10
    # looks like closer to expectations
    Normalized_t1, symbol_response = HDDVD_symbol_response(bit_periods = 10, bits_freq = HDDVD_bits_freq, upsample_factor = upsample_factor)
    PR1_coefs = [1, 2, 2, 2, 1]
    Normalized_t2, target_pr1 = partial_response(PR_coefs = PR1_coefs, bit_periods = 10, bits_freq = HDDVD_bits_freq, upsample_factor = upsample_factor)
    PR2_coefs = [1, 1, 1, 1]
    Normalized_t3, target_pr2 = partial_response(PR_coefs = PR2_coefs, bit_periods = 10, bits_freq = HDDVD_bits_freq, upsample_factor = upsample_factor)
    PR3_coefs = [1, 2, 3, 4, 3, 2, 1]
    Normalized_t4, target_pr3 = partial_response(PR_coefs = PR3_coefs, bit_periods = 10, bits_freq = HDDVD_bits_freq, upsample_factor = upsample_factor)
    Xs = [
        Normalized_t1,
        Normalized_t2,
        Normalized_t3,
        Normalized_t4
    ]
    Ys = [
    {'data': symbol_response, 'label': 'HDDVDs Symbol Response', 'color': 'purple'},
    {'data': target_pr1, 'label': 'PR [1 2 2 2 1] Response', 'color': 'red'},
    {'data': target_pr2, 'label': 'PR [1 1 1 1] Response', 'color': 'green'},
    {'data': target_pr3, 'label': 'PR [1 2 3 4 3 2 1] Response', 'color': 'black'},
    ]
    titles = [
        'HDDVDs Symbol Response',
        'Target PR [1 2 2 2 1] Response',
        'Target PR [1 1 1 1] Response',
        'Target PR [1 2 3 4 3 2 1] Response',
    ]
    xlabels = ["Time (t/T)"]
    ylabels = ["Amplitude"]
    plot_separated(
        Xs=Xs, 
        Ys=Ys, 
        titles=titles,     
        xlabels=xlabels, 
        ylabels=ylabels
    )
    bit_periods = max((len(symbol_response)-1)//2, (len(target_pr1)-1)//2)
    downsample_factor = 10
    freqs, HDDVD_symbol_response_fft_magnitude = Fourier_Analysis(symbol_response, bit_periods = bit_periods, bits_freq = HDDVD_bits_freq, downsample_factor = downsample_factor)
    _, target_pr1_fft_magnitude = Fourier_Analysis(target_pr1, bit_periods = bit_periods, bits_freq = HDDVD_bits_freq, downsample_factor = downsample_factor)
    _, target_pr2_fft_magnitude = Fourier_Analysis(target_pr2, bit_periods = bit_periods, bits_freq = HDDVD_bits_freq, downsample_factor = downsample_factor)
    _, target_pr3_fft_magnitude = Fourier_Analysis(target_pr3, bit_periods = bit_periods, bits_freq = HDDVD_bits_freq, downsample_factor = downsample_factor)
    Ys = [
    {'data': HDDVD_symbol_response_fft_magnitude / max(HDDVD_symbol_response_fft_magnitude), 'label': 'HDDVD Symbol Response', 'color': 'purple'},
    {'data': target_pr1_fft_magnitude / max(target_pr1_fft_magnitude), 'label': 'PR [1 2 2 2 1] Response', 'color': 'red'},
    {'data': target_pr2_fft_magnitude / max(target_pr2_fft_magnitude), 'label': 'PR [1 1 1 1] Response', 'color': 'green'},
    {'data': target_pr3_fft_magnitude / max(target_pr3_fft_magnitude), 'label': 'PR [1 2 3 4 3 2 1] Response', 'color': 'black'},
    ]
    Normalized_f = freqs/max(freqs)
    plot_altogether(
        X=Normalized_f, 
        Ys=Ys, 
        title='Frequency Spectrum of Symbol Responses and Target PR Responses for HDDVDs',     
        xlabel="Normalized Frequency", 
        ylabel="Normalized Amplitude",
        xtick_interval=0.1,
        ytick_interval=0.1
    )