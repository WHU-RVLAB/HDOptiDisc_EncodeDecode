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

def partial_response(PR_coefs, bit_periods, T_L = 74.5e-9, upsample_factor = 1):
    t = np.linspace(-bit_periods * T_L, bit_periods * T_L, int(2*upsample_factor*bit_periods + 1))
    
    target_pr = np.zeros_like(t)
    for n in range(len(PR_coefs)):
        target_pr += PR_coefs[n] * sinc((t - n * T_L) * np.pi / T_L)
    
    return t/T_L, target_pr

if __name__ == '__main__':
    
    bit_periods = 10
    BD_T_L = 74.5e-9
    upsample_factor = 10
    Normalized_t1, symbol_response = BD_symbol_response(bit_periods = bit_periods, upsample_factor = upsample_factor)
    PR1_coefs = [1, 2, 2, 1]
    Normalized_t2, target_pr1 = partial_response(PR_coefs = PR1_coefs, bit_periods = bit_periods, T_L = BD_T_L, upsample_factor = 10)
    PR2_coefs = [1, 3, 3, 1]
    Normalized_t3, target_pr2 = partial_response(PR_coefs = PR2_coefs, bit_periods = bit_periods, T_L = BD_T_L, upsample_factor = 10)
    PR3_coefs = [1, 2, 2, 2, 1]
    Normalized_t4, target_pr3 = partial_response(PR_coefs = PR3_coefs, bit_periods = bit_periods, T_L = BD_T_L, upsample_factor = 10)
    PR4_coefs = [1, 2, 3, 3, 2, 1]
    Normalized_t5, target_pr4 = partial_response(PR_coefs = PR4_coefs, bit_periods = bit_periods, T_L = BD_T_L, upsample_factor = 10)
    Xs = [
        Normalized_t1,
        Normalized_t2,
        Normalized_t3,
        Normalized_t4,
        Normalized_t5
    ]
    Ys = [
    {'data': symbol_response, 'label': 'BDs Symbol Response', 'color': 'black'},
    {'data': target_pr1, 'label': 'PR [1 2 2 1] Response', 'color': 'red'},
    {'data': target_pr2, 'label': 'PR [1 3 3 1] Response', 'color': 'springgreen'},
    {'data': target_pr3, 'label': 'PR [1 2 2 2 1] Response', 'color': 'deeppink'},
    {'data': target_pr4, 'label': 'PR [1 2 3 3 2 1] Response', 'color': 'blue', 'linestyle':'--'},
    ]
    titles = [
        'BDs Symbol Response',
        'Target PR [1 2 2 1] Response',
        'Target PR [1 3 3 1] Response',
        'Target PR [1 2 2 2 1] Response',
        'Target PR [1 2 3 3 2 1] Response',
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
    sample_periods = 300
    downsample_factor = 10
    Normalized_f, symbol_response_fft_magnitude = Fourier_Analysis(symbol_response, sample_periods = sample_periods, T_L = BD_T_L, downsample_factor = downsample_factor)
    _, target_pr1_fft_magnitude = Fourier_Analysis(target_pr1, sample_periods = sample_periods, T_L = BD_T_L, downsample_factor = downsample_factor)
    _, target_pr2_fft_magnitude = Fourier_Analysis(target_pr2, sample_periods = sample_periods, T_L = BD_T_L, downsample_factor = downsample_factor)
    _, target_pr3_fft_magnitude = Fourier_Analysis(target_pr3, sample_periods = sample_periods, T_L = BD_T_L, downsample_factor = downsample_factor)
    _, target_pr4_fft_magnitude = Fourier_Analysis(target_pr4, sample_periods = sample_periods, T_L = BD_T_L, downsample_factor = downsample_factor)
    Ys = [
    {'data': symbol_response_fft_magnitude / max(symbol_response_fft_magnitude), 'label': 'BDs Symbol Response', 'color': 'black'},
    {'data': target_pr1_fft_magnitude / max(target_pr1_fft_magnitude), 'label': 'PR [1 2 2 1] Response', 'color': 'red'},
    {'data': target_pr2_fft_magnitude / max(target_pr2_fft_magnitude), 'label': 'PR [1 3 3 1] Response', 'color': 'springgreen'},
    {'data': target_pr3_fft_magnitude / max(target_pr3_fft_magnitude), 'label': 'PR [1 2 2 2 1] Response', 'color': 'deeppink'},
    {'data': target_pr4_fft_magnitude / max(target_pr4_fft_magnitude), 'label': 'PR [1 2 3 3 2 1] Response', 'color': 'blue', 'linestyle':'--'},
    ]
    plot_altogether(
        X=Normalized_f, 
        Ys=Ys, 
        title='Frequency Spectrum of Symbol Responses and Target PR Responses for BDs',     
        xlabel="Normalized Frequency", 
        ylabel="Normalized Amplitude",
        xtick_interval=0.1,
        ytick_interval=0.1
    )
    
    bit_periods = 10
    HDDVD_T_L = 0.102e-6
    upsample_factor = 10
    Normalized_t1, symbol_response = HDDVD_symbol_response(bit_periods = bit_periods, upsample_factor = upsample_factor)
    PR1_coefs = [1, 2, 2, 1]
    Normalized_t2, target_pr1 = partial_response(PR_coefs = PR1_coefs, bit_periods = bit_periods, T_L = HDDVD_T_L, upsample_factor = 10)
    PR2_coefs = [1, 3, 3, 1]
    Normalized_t3, target_pr2 = partial_response(PR_coefs = PR2_coefs, bit_periods = bit_periods, T_L = HDDVD_T_L, upsample_factor = 10)
    PR3_coefs = [1, 2, 2, 2, 1]
    Normalized_t4, target_pr3 = partial_response(PR_coefs = PR3_coefs, bit_periods = bit_periods, T_L = HDDVD_T_L, upsample_factor = 10)
    PR4_coefs = [1, 2, 3, 3, 2, 1]
    Normalized_t5, target_pr4 = partial_response(PR_coefs = PR4_coefs, bit_periods = bit_periods, T_L = HDDVD_T_L, upsample_factor = 10)
    Xs = [
        Normalized_t1,
        Normalized_t2,
        Normalized_t3,
        Normalized_t4,
        Normalized_t5
    ]
    Ys = [
    {'data': symbol_response, 'label': 'BDs Symbol Response', 'color': 'black'},
    {'data': target_pr1, 'label': 'PR [1 2 2 1] Response', 'color': 'red'},
    {'data': target_pr2, 'label': 'PR [1 3 3 1] Response', 'color': 'springgreen'},
    {'data': target_pr3, 'label': 'PR [1 2 2 2 1] Response', 'color': 'deeppink'},
    {'data': target_pr4, 'label': 'PR [1 2 3 3 2 1] Response', 'color': 'blue', 'linestyle':'--'},
    ]
    titles = [
        'HDDVDs Symbol Response',
        'Target PR [1 2 2 1] Response',
        'Target PR [1 3 3 1] Response',
        'Target PR [1 2 2 2 1] Response',
        'Target PR [1 2 3 3 2 1] Response',
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
    sample_periods = 300
    downsample_factor = 10
    Normalized_f, symbol_response_fft_magnitude = Fourier_Analysis(symbol_response, sample_periods = sample_periods, T_L = HDDVD_T_L, downsample_factor = downsample_factor)
    _, target_pr1_fft_magnitude = Fourier_Analysis(target_pr1, sample_periods = sample_periods, T_L = HDDVD_T_L, downsample_factor = downsample_factor)
    _, target_pr2_fft_magnitude = Fourier_Analysis(target_pr2, sample_periods = sample_periods, T_L = HDDVD_T_L, downsample_factor = downsample_factor)
    _, target_pr3_fft_magnitude = Fourier_Analysis(target_pr3, sample_periods = sample_periods, T_L = HDDVD_T_L, downsample_factor = downsample_factor)
    _, target_pr4_fft_magnitude = Fourier_Analysis(target_pr4, sample_periods = sample_periods, T_L = HDDVD_T_L, downsample_factor = downsample_factor)
    Ys = [
    {'data': symbol_response_fft_magnitude / max(symbol_response_fft_magnitude), 'label': 'HDDVDs Symbol Response', 'color': 'black'},
    {'data': target_pr1_fft_magnitude / max(target_pr1_fft_magnitude), 'label': 'PR [1 2 2 1] Response', 'color': 'red'},
    {'data': target_pr2_fft_magnitude / max(target_pr2_fft_magnitude), 'label': 'PR [1 3 3 1] Response', 'color': 'springgreen'},
    {'data': target_pr3_fft_magnitude / max(target_pr3_fft_magnitude), 'label': 'PR [1 2 2 2 1] Response', 'color': 'deeppink'},
    {'data': target_pr4_fft_magnitude / max(target_pr4_fft_magnitude), 'label': 'PR [1 2 3 3 2 1] Response', 'color': 'blue', 'linestyle':'--'},
    ]
    plot_altogether(
        X=Normalized_f, 
        Ys=Ys, 
        title='Frequency Spectrum of Symbol Responses and Target PR Responses for HDDVDs',     
        xlabel="Normalized Frequency", 
        ylabel="Normalized Amplitude",
        xtick_interval=0.1,
        ytick_interval=0.1
    )