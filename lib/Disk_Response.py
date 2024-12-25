import sys
import os
import numpy as np
from scipy.special import erf
sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)))
from Utils import plot_separated, Fourier_Analysis
sys.path.pop()

def disk_impulse_response(wavelength, na, T_L, bit_periods, upsample_factor):
    T_0 = 0.86 * wavelength / na
    t = np.linspace(-bit_periods * T_L, bit_periods * T_L, int(2*upsample_factor*bit_periods + 1))
    
    impulse_response = (2 / (T_0 * np.sqrt(np.pi))) * np.exp(-(2 * t / T_0) ** 2)
    
    return t/T_L, impulse_response

def disk_symbol_response(wavelength, na, T_L, bit_periods, upsample_factor):
    T_0 = 0.86 * wavelength / na
    t = np.linspace(-bit_periods * T_L, bit_periods * T_L, int(2*upsample_factor*bit_periods + 1))

    symbol_response = 0.5 * (erf(2*(t + 0.5*T_L) / T_0) - erf(2*(t - 0.5*T_L) / T_0))
    
    return t/T_L, symbol_response

def BD_impulse_response(bit_periods, upsample_factor = 1):
    wavelength = 405e-9
    na = 0.85
    T_L = 74.5e-9
    return disk_impulse_response(wavelength, na, T_L, bit_periods, upsample_factor)

def BD_symbol_response(bit_periods, upsample_factor = 1):
    wavelength = 405e-9
    na = 0.85
    T_L = 74.5e-9
    return disk_symbol_response(wavelength, na, T_L, bit_periods, upsample_factor)

def HDDVD_impulse_response(bit_periods, upsample_factor = 1):
    wavelength = 405e-9
    na = 0.65
    T_L = 0.102e-6
    return disk_impulse_response(wavelength, na, T_L, bit_periods, upsample_factor)

def HDDVD_symbol_response(bit_periods, upsample_factor = 1):
    wavelength = 405e-9
    na = 0.65
    T_L = 0.102e-6
    return disk_symbol_response(wavelength, na, T_L, bit_periods, upsample_factor)
       
if __name__ == '__main__':
    
    bit_periods = 10
    BD_T_L = 74.5e-9
    upsample_factor = 10
    Normalized_t1, impulse_response = BD_impulse_response(bit_periods = bit_periods, upsample_factor = upsample_factor)
    Normalized_t2, symbol_response = BD_symbol_response(bit_periods = bit_periods, upsample_factor = upsample_factor)
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
    xlabels = ["Time (t/T)"]
    ylabels = ["Normalized Amplitude"]
    plot_separated(
        Xs=Xs, 
        Ys=Ys, 
        titles=titles,     
        xlabels=xlabels, 
        ylabels=ylabels
    )
    sample_periods = 300
    downsample_factor = 10
    Normalized_f, impulse_response_fft_magnitude = Fourier_Analysis(impulse_response, sample_periods = sample_periods, T_L = BD_T_L, downsample_factor = downsample_factor)
    _, symbol_response_fft_magnitude = Fourier_Analysis(symbol_response, sample_periods = sample_periods, T_L = BD_T_L, downsample_factor = downsample_factor)
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
    plot_separated(
        Xs=Xs, 
        Ys=Ys, 
        titles=titles,     
        xlabels=xlabels, 
        ylabels=ylabels
    ) 
    
    
    bit_periods = 10
    HDDVD_T_L = 0.102e-6
    upsample_factor = 10
    Normalized_t1, impulse_response = HDDVD_impulse_response(bit_periods = bit_periods, upsample_factor = upsample_factor)
    Normalized_t2, symbol_response = HDDVD_symbol_response(bit_periods = bit_periods, upsample_factor = upsample_factor)
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
    xlabels = ["Time (t/T)"]
    ylabels = ["Normalized Amplitude"]
    plot_separated(
        Xs=Xs, 
        Ys=Ys, 
        titles=titles,     
        xlabels=xlabels, 
        ylabels=ylabels
    )
    sample_periods = 300
    downsample_factor = 10
    Normalized_f, impulse_response_fft_magnitude = Fourier_Analysis(impulse_response, sample_periods = sample_periods, T_L = HDDVD_T_L, downsample_factor = downsample_factor)
    _, symbol_response_fft_magnitude = Fourier_Analysis(symbol_response, sample_periods = sample_periods, T_L = HDDVD_T_L, downsample_factor = downsample_factor)
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
    plot_separated(
        Xs=Xs, 
        Ys=Ys, 
        titles=titles,     
        xlabels=xlabels, 
        ylabels=ylabels
    )