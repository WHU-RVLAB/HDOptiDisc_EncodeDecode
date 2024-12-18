import sys
import os
import numpy as np
from scipy.special import erf
sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)))
from Utils import plot_separated, Fourier_Analysis
sys.path.pop()

def disk_impulse_response(wavelength, na, bit_periods, bits_freq, upsample_factor):
    T_0 = 0.86 * wavelength / na
    T_S = 1 / bits_freq
    t = np.linspace(-bit_periods * T_S, bit_periods * T_S, 2*upsample_factor*bit_periods + 1)
    
    impulse_response = (2 / (T_0 * np.sqrt(np.pi))) * np.exp(-(2 * t / T_0) ** 2)
    
    return t/T_S, impulse_response

def disk_symbol_response(wavelength, na, bit_periods, bits_freq, upsample_factor):
    T_0 = 0.86 * wavelength / na
    T_S = 1 / bits_freq
    t = np.linspace(-bit_periods * T_S, bit_periods * T_S, 2*upsample_factor*bit_periods + 1)

    symbol_response = 0.5 * (erf(t / T_0) - erf((t - T_S) / T_0))
    
    return t/T_S, symbol_response

def BD_impulse_response(bit_periods, bits_freq = 132e6, upsample_factor = 1):
    wavelength = 405e-9
    na = 0.85
    return disk_impulse_response(wavelength, na, bit_periods, bits_freq, upsample_factor)

def BD_symbol_response(bit_periods, bits_freq = 132e6, upsample_factor = 1):
    wavelength = 405e-9
    na = 0.85
    return disk_symbol_response(wavelength, na, bit_periods, bits_freq, upsample_factor)

def HDDVD_impulse_response(bit_periods, bits_freq = 64.8e6, upsample_factor = 1):
    wavelength = 405e-9
    na = 0.65
    return disk_impulse_response(wavelength, na, bit_periods, bits_freq, upsample_factor)

def HDDVD_symbol_response(bit_periods, bits_freq = 64.8e6, upsample_factor = 1):
    wavelength = 405e-9
    na = 0.65
    return disk_symbol_response(wavelength, na, bit_periods, bits_freq, upsample_factor)
       
if __name__ == '__main__':
    BD_bits_freq = 132e6
    upsample_factor = 1
    Normalized_t1, impulse_response = BD_impulse_response(bit_periods = 10, bits_freq = BD_bits_freq, upsample_factor = upsample_factor)
    Normalized_t2, symbol_response = BD_symbol_response(bit_periods = 40, bits_freq = BD_bits_freq, upsample_factor = upsample_factor)
    
    Xs = [
    Normalized_t1,
    Normalized_t2
    ]
    Ys = [
    {'data': impulse_response / max(impulse_response), 'label': 'BDs Impulse Response', 'color': 'red'},
    {'data': symbol_response / max(symbol_response), 'label': 'BDs Symbol Response', 'color': 'blue'},
    ]
    titles = [
    'BDs Impulse Response',
    'BDs Symbol Response',
    ]
    xlabels = ["Time (t/T_S)"]
    ylabels = ["Normalized Amplitude"]
    Xtick_intervals = [
        2,
        10
    ]
    Ytick_intervals = [
        0.1,
        0.1
    ]
    plot_separated(
        Xs=Xs, 
        Ys=Ys, 
        titles=titles,     
        xlabels=xlabels, 
        ylabels=ylabels,
        Xtick_intervals=Xtick_intervals,
        Ytick_intervals=Ytick_intervals
    )
    
    bit_periods = 150
    downsample_factor = 1
    freqs, impulse_response_fft_magnitude = Fourier_Analysis(impulse_response, bit_periods = bit_periods, bits_freq = BD_bits_freq, downsample_factor = downsample_factor)
    _, symbol_response_fft_magnitude = Fourier_Analysis(symbol_response, bit_periods = bit_periods, bits_freq = BD_bits_freq, downsample_factor = downsample_factor)

    Normalized_f = freqs/max(freqs)
    Xs = [
    Normalized_f,
    Normalized_f
    ]
    Ys = [
    {'data': impulse_response_fft_magnitude / max(impulse_response_fft_magnitude), 'label': 'BDs Impulse Response', 'color': 'red'},
    {'data': symbol_response_fft_magnitude / max(symbol_response_fft_magnitude), 'label': 'BDs Symbol Response', 'color': 'blue'}
    ]
    titles = [
    'Frequency Spectrum of BDs Impulse Response',
    'Frequency Spectrum of BDs Symbol Response',
    ]
    xlabels = ["Normalized Frequency"]
    ylabels = ["Normalized Amplitude"]
    Xtick_intervals = [
        0.1,
        0.05
    ]
    Ytick_intervals = [
        0.1,
        0.1
    ]
    plot_separated(
        Xs=Xs, 
        Ys=Ys, 
        titles=titles,     
        xlabels=xlabels, 
        ylabels=ylabels,
        Xtick_intervals=Xtick_intervals,
        Ytick_intervals=Ytick_intervals
    )
    
    HDDVD_bits_freq = 10e6
    # HDDVD_bits_freq = 1e6
    upsample_factor = 10
    Normalized_t1, impulse_response = HDDVD_impulse_response(bit_periods = 10, bits_freq = HDDVD_bits_freq, upsample_factor = upsample_factor)
    Normalized_t2, symbol_response = HDDVD_symbol_response(bit_periods = 40, bits_freq = HDDVD_bits_freq, upsample_factor = upsample_factor)

    Xs = [
    Normalized_t1,
    Normalized_t2
    ]
    Ys = [
    {'data': impulse_response / max(impulse_response), 'label': 'HDDVDs Impulse Response', 'color': 'red'},
    {'data': symbol_response / max(symbol_response), 'label': 'HDDVDs Symbol Response', 'color': 'blue'},
    ]
    titles = [
    'HDDVDs Impulse Response',
    'HDDVDs Symbol Response',
    ]
    xlabels = ["Time (t/T_S)"]
    ylabels = ["Normalized Amplitude"]
    Xtick_intervals = [
        2,
        10
    ]
    Ytick_intervals = [
        0.1,
        0.1
    ]
    plot_separated(
        Xs=Xs, 
        Ys=Ys, 
        titles=titles,     
        xlabels=xlabels, 
        ylabels=ylabels,
        Xtick_intervals=Xtick_intervals,
        Ytick_intervals=Ytick_intervals
    )
    
    bit_periods = 150
    downsample_factor = 10
    freqs, impulse_response_fft_magnitude = Fourier_Analysis(impulse_response, bit_periods = bit_periods, bits_freq = HDDVD_bits_freq, downsample_factor = downsample_factor)
    _, symbol_response_fft_magnitude = Fourier_Analysis(symbol_response, bit_periods = bit_periods, bits_freq = HDDVD_bits_freq, downsample_factor = downsample_factor)

    Normalized_f = freqs/max(freqs)
    Xs = [
    Normalized_f,
    Normalized_f
    ]
    Ys = [
    {'data': impulse_response_fft_magnitude / max(impulse_response_fft_magnitude), 'label': 'HDDVDs Impulse Response', 'color': 'red'},
    {'data': symbol_response_fft_magnitude / max(symbol_response_fft_magnitude), 'label': 'HDDVDs Symbol Response', 'color': 'blue'}
    ]
    titles = [
    'Frequency Spectrum of HDDVDs Impulse Response',
    'Frequency Spectrum of HDDVDs Symbol Response',
    ]
    xlabels = ["Normalized Frequency"]
    ylabels = ["Normalized Amplitude"]
    Xtick_intervals = [
        0.1,
        0.05
    ]
    Ytick_intervals = [
        0.1,
        0.1
    ]
    plot_separated(
        Xs=Xs, 
        Ys=Ys, 
        titles=titles,     
        xlabels=xlabels, 
        ylabels=ylabels,
        Xtick_intervals=Xtick_intervals,
        Ytick_intervals=Ytick_intervals
    )