import sys
import os
import numpy as np
sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)))
from Const import RLL_state_machine
from Channel_Modulator import RLL_Modulator
from Channel_Converter import NRZI_Converter
from Disk_Response import BD_symbol_response
from Utils import plot_separated, plot_eye_diagram
from Params import Params
sys.path.pop()
    
class Disk_Read_Channel(object):
    
    def __init__(self, params:Params):
        self.params = params
        upsample_factor = params.upsample_factor
        _, bd_di_coef = BD_symbol_response(bit_periods = 10, upsample_factor=upsample_factor)
        mid_idx = len(bd_di_coef)//2
        self.bd_di_coef = bd_di_coef[mid_idx : mid_idx + upsample_factor*self.params.tap_bd_num].reshape(1,-1)
        
        print('\nThe dipulse bd coefficient is')
        print(bd_di_coef)
        print(f"bd_di_coef.shape: {bd_di_coef.shape}")
        print('\nTap bd coefficient is')
        print(self.bd_di_coef)
        print(f"self.bd_di_coef.shape: {self.bd_di_coef.shape}")
    
    def RF_signal_jitter(self, codeword):
        params = self.params
        signal_ideal = codeword.reshape(-1)
        signal_ideal_pad = np.concatenate([[0], signal_ideal])
        signal_ideal_diff = np.diff(signal_ideal_pad)
        
        upsample_factor = params.upsample_factor
        signal_upsample_ideal = np.repeat(signal_ideal, upsample_factor)
        
        max_jcl, min_jcl = upsample_factor*params.jcl_stop, upsample_factor*params.jcl_start
        miu = (max_jcl + min_jcl)/2
        sigma = (max_jcl - miu)/3
        upsample_jitter = np.zeros(len(signal_ideal)).astype(int)
        for i in range(1, len(signal_ideal)):# not consider the first signal
            if signal_ideal_diff[i]:
                random_jitter = np.random.normal(miu, sigma)*np.random.choice([-1, 1])
                upsample_jitter[i] = np.round(random_jitter).astype(int)
                upsample_jitter[i-1] = -upsample_jitter[i]
        
        # print(np.mean(upsample_jitter))
        upsample_jitter += upsample_factor
        
        signal_upsample_jittered = np.repeat(signal_ideal, upsample_jitter).reshape(1, -1)

        downsample_factor = upsample_factor
        bd_di_coef_sum = sum(self.bd_di_coef[0, :][::downsample_factor])
        bd_di_coef_upsample_sum = sum(self.bd_di_coef[0, :])
        if not params.signal_norm:
            bd_di_coef_upsample_sum /= bd_di_coef_sum
            bd_di_coef_sum = 1
        
        rf_signal_ideal = (np.convolve(self.bd_di_coef[0, :][::downsample_factor], codeword[0, :])
               [:-(self.params.tap_bd_num - 1)].reshape(codeword.shape))/bd_di_coef_sum
        
        rf_signal = (np.convolve(self.bd_di_coef[0, :], signal_upsample_jittered[0, :])
               [:-(upsample_factor*self.params.tap_bd_num - 1)].reshape(signal_upsample_jittered.shape))/bd_di_coef_upsample_sum
        
        rf_signal = rf_signal[:, ::downsample_factor]
        
        return signal_upsample_ideal, signal_upsample_jittered, rf_signal_ideal, rf_signal

    def addsin(self, x):
        amplitude = 0.03
        frequency = 0.001
        t = np.arange(len(x.reshape(-1)))
        sine_wave = amplitude * np.sin(2 * np.pi * frequency * t)
        x_noise = x + sine_wave
        return x_noise

    def multiplysin(self, x):
        amplitude = 0.03
        frequency = 0.001
        t = np.arange(len(x.reshape(-1)))
        sine_wave = amplitude * np.sin(2 * np.pi * frequency * t)
        x_noise = x * (1 + sine_wave)
        return x_noise

    def awgn(self, x, snr):
        E_b = np.mean(np.square(x[0, :self.params.truncation4energy]))
        sigma = np.sqrt(0.5 * E_b * 10 ** (- snr * 1.0 / 10))
        x_noise = x + sigma * np.random.normal(0, 1, x.shape)
        return x_noise   
    
if __name__ == '__main__':
    
    # constant and input paras
    params = Params()
    encoder_dict, encoder_definite = RLL_state_machine()
    RLL_modulator = RLL_Modulator(encoder_dict, encoder_definite)
    NRZI_converter = NRZI_Converter()
    disk_read_channel = Disk_Read_Channel(params)
    
    # rate for constrained code
    num_sym_in_constrain = encoder_dict[1]['input'].shape[1]
    num_sym_out_constrain = encoder_dict[1]['output'].shape[1]
    rate_constrain = num_sym_in_constrain / num_sym_out_constrain
    codeword_len = int(params.equalizer_train_len/rate_constrain)
    
    Normalized_t = np.linspace(0, int(params.module_test_len/rate_constrain) - 1, int(params.module_test_len/rate_constrain))
    Normalized_t_upsample = np.linspace(0, int(params.module_test_len/rate_constrain) - 1/params.upsample_factor, params.upsample_factor*int(params.module_test_len/rate_constrain))

    snr = 25
    info = np.random.randint(2, size=(1, params.module_test_len))
    codeword = NRZI_converter.forward_coding(RLL_modulator.forward_coding(info))
    signal_upsample_ideal, signal_upsample_jittered, rf_signal_ideal, rf_signal = disk_read_channel.RF_signal_jitter(codeword)
    signal_diff = signal_upsample_ideal.reshape(-1) - signal_upsample_jittered.reshape(-1)
    rf_signal_awgn = disk_read_channel.awgn(rf_signal, snr)
    rf_signal_awgn_addsin = disk_read_channel.addsin(rf_signal_awgn)
    rf_signal_awgn_multiplysin = disk_read_channel.multiplysin(rf_signal_awgn)
    signal_addsin_diff = rf_signal_awgn_addsin.reshape(-1) - rf_signal_awgn.reshape(-1)
    signal_multisin_diff = rf_signal_awgn_multiplysin.reshape(-1) - rf_signal_awgn.reshape(-1)
    signal_sine_diff = rf_signal_awgn_multiplysin.reshape(-1) - rf_signal_awgn_addsin.reshape(-1)

    #jitter and sine
    Xs = [
        Normalized_t,
        Normalized_t,
        Normalized_t,
        Normalized_t,
        Normalized_t,
        Normalized_t
    ]
    Ys = [
    {'data': rf_signal_awgn.reshape(-1), 'label': 'rf_signal_awgn', 'color': 'red'},
    {'data': rf_signal_awgn_addsin.reshape(-1), 'label': 'rf_signal_awgn_addsin', 'color': 'red'},
    {'data': rf_signal_awgn_multiplysin.reshape(-1), 'label': 'rf_signal_awgn_multiplysin', 'color': 'red'},
    {'data': signal_addsin_diff.reshape(-1), 'label': 'signal_addsin_diff', 'color': 'red'},
    {'data': signal_multisin_diff.reshape(-1), 'label': 'signal_multisin_diff', 'color': 'red'},
    {'data': signal_sine_diff.reshape(-1), 'label': 'signal_sine_diff', 'color': 'red'}
    ]
    titles = [
        'rf_signal_awgn',
        'rf_signal_awgn_addsin',
        'rf_signal_awgn_multiplysin',
        'signal_addsin_diff',
        'signal_multisin_diff',
        'signal_sine_diff'
    ]
    xlabels = ["Time (t/T)"]
    ylabels = [
        "Amplitude",
        "Amplitude",
        "Amplitude",
        "Amplitude",
        "Amplitude",
        "Amplitude"
    ]
    plot_separated(
        Xs=Xs,
        Ys=Ys,
        titles=titles,
        xlabels=xlabels,
        ylabels=ylabels
    )

    #diff of ideal signal and jitter signal
    Xs = [
        Normalized_t_upsample,
        Normalized_t_upsample,
        Normalized_t_upsample
    ]
    Ys = [
    {'data': signal_upsample_ideal.reshape(-1), 'label': 'signal_upsample_ideal', 'color': 'red'},
    {'data': signal_upsample_jittered.reshape(-1), 'label': 'signal_upsample_jittered', 'color': 'red'},
    {'data': signal_diff.reshape(-1), 'label': 'diff', 'color': 'red'}
    ]
    titles = [
        'signal_upsample_ideal',
        'signal_upsample_jittered',
        'diff'
    ]
    xlabels = ["Time (t/T)"]
    ylabels = [
        "Binary",
        "Binary",
        "Binary"
    ]
    plot_separated(
        Xs=Xs,
        Ys=Ys,
        titles=titles,
        xlabels=xlabels,
        ylabels=ylabels
    )

    #jitter and sine
    Xs = [
        Normalized_t,
        Normalized_t,
        Normalized_t,
        Normalized_t,
        Normalized_t
    ]
    Ys = [
    {'data': rf_signal_ideal.reshape(-1), 'label': 'rf_signal_ideal', 'color': 'red'},
    {'data': rf_signal.reshape(-1), 'label': 'rf_signal', 'color': 'red'},
    {'data': rf_signal_awgn.reshape(-1), 'label': 'rf_signal_awgn', 'color': 'red'},
    {'data': rf_signal_awgn_addsin.reshape(-1), 'label': 'rf_signal_awgn_addsin', 'color': 'red'},
    {'data': rf_signal_awgn_multiplysin.reshape(-1), 'label': 'rf_signal_awgn_multiplysin', 'color': 'red'}
    ]
    titles = [
        'rf_signal_ideal',
        'rf_signal',
        f'rf_signal_awgn_snr{snr}',
        'rf_signal_awgn_addsin',
        'rf_signal_awgn_multiplysin'
    ]
    xlabels = ["Time (t/T)"]
    ylabels = [
        "Amplitude",
        "Amplitude",
        "Amplitude",
        "Amplitude",
        "Amplitude"
    ]
    plot_separated(
        Xs=Xs,
        Ys=Ys,
        titles=titles,
        xlabels=xlabels,
        ylabels=ylabels
    )
    
    signal = {'data': rf_signal_ideal.reshape(-1), 'label': 'rf_signal_ideal', 'color': 'black'}
    title = 'rf_signal_ideal eyes diagram'
    xlabel = "Time (t/T)"
    ylabel = "Amplitude"
    plot_eye_diagram(
        signal=signal,
        samples_truncation=params.eye_diagram_truncation,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel
    )

    signal = {'data': rf_signal.reshape(-1), 'label': 'rf_signal', 'color': 'black'}
    title = 'rf_signal eyes diagram'
    xlabel = "Time (t/T)"
    ylabel = "Amplitude"
    plot_eye_diagram(
        signal=signal,
        samples_truncation=params.eye_diagram_truncation,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel
    )

    signal = {'data': rf_signal_awgn.reshape(-1), 'label': f'rf_signal_awgn_snr{snr}', 'color': 'black'}
    title = f'rf_signal_awgn_snr{snr} eyes diagram'
    xlabel = "Time (t/T)"
    ylabel = "Amplitude"
    plot_eye_diagram(
        signal=signal,
        samples_truncation=params.eye_diagram_truncation,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel
    )
    
    signal = {'data': rf_signal_awgn_addsin.reshape(-1), 'label': 'rf_signal_awgn_addsin', 'color': 'black'}
    title = 'rf_signal_awgn_addsin eyes diagram'
    xlabel = "Time (t/T)"
    ylabel = "Amplitude"
    plot_eye_diagram(
        signal=signal,
        samples_truncation=params.eye_diagram_truncation,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel
    )
    
    signal = {'data': rf_signal_awgn_multiplysin.reshape(-1), 'label': f'rf_signal_awgn_multiplysin', 'color': 'black'}
    title = 'rf_signal_awgn_multiplysin eyes diagram'
    xlabel = "Time (t/T)"
    ylabel = "Amplitude"
    plot_eye_diagram(
        signal=signal,
        samples_truncation=params.eye_diagram_truncation,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel
    )