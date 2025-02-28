import sys
import os
import numpy as np
from scipy.interpolate import interp1d
sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)))
from Const import RLL_state_machine
from Channel_Modulator import RLL_Modulator
from Channel_Converter import NRZI_Converter
from Utils import plot_separated, plot_eye_diagram
from Params import Params
sys.path.pop()
    
class Target_PR_Channel(object):
    
    def __init__(self, params:Params):
        self.params = params
        self.PR_coefs = np.array(params.PR_coefs).reshape(1,-1)
        
        print('\nTarget Channel is')
        print(self.PR_coefs)
    
    def target_channel(self, codeword):
        target_channel_signal = (np.convolve(self.PR_coefs[0, :], codeword[0, :])
               [:-(self.PR_coefs.shape[1] - 1)].reshape(codeword.shape))
        
        return target_channel_signal
    
    def jitter(self, x):
        params = self.params
        signal_ideal = x.reshape(-1)
        t_ideal = np.linspace(0, len(signal_ideal), len(signal_ideal))
        
        # random jcl
        miu = (params.jcl_start + params.jcl_stop)/2
        sigma = (params.jcl_stop - miu)/2
        jitter = np.random.normal(miu, sigma, len(signal_ideal))
        t_jittered = t_ideal + jitter

        f_interp_ideal = interp1d(t_ideal, signal_ideal, kind='linear', fill_value='extrapolate')
        signal_jittered = f_interp_ideal(t_jittered)
        
        return signal_jittered.reshape(1, -1)
    
    def awgn(self, x, snr):
        E_b = np.mean(np.square(x[0, :self.params.truncation4energy]))
        sigma = np.sqrt(0.5 * E_b * 10 ** (- snr * 1.0 / 10))
        x_noise = x + sigma * np.random.normal(0, 1, x.shape)
        return x_noise  
    
if __name__ == '__main__':
    
    # constant and input paras
    params = Params()
    encoder_dict, encoder_definite = RLL_state_machine()

    # rate for constrained code
    num_sym_in_constrain = encoder_dict[1]['input'].shape[1]
    num_sym_out_constrain = encoder_dict[1]['output'].shape[1]
    rate_constrain = num_sym_in_constrain / num_sym_out_constrain
    codeword_len = int(params.equalizer_train_len/rate_constrain)
    
    RLL_modulator = RLL_Modulator(encoder_dict, encoder_definite)
    NRZI_converter = NRZI_Converter()
    target_pr_channel = Target_PR_Channel(params)
    
    params.snr_step = (params.snr_stop-params.snr_start)/(params.num_plots - 1)
    num_ber = int((params.snr_stop-params.snr_start)/params.snr_step + 1)
    
    Normalized_t = np.linspace(1, int(params.module_test_len/rate_constrain), int(params.module_test_len/rate_constrain))
    
    for idx in np.arange(0, num_ber):
        snr = params.snr_start+idx*params.snr_step
        
        info = np.random.randint(2, size = (1, params.module_test_len))
        codeword = NRZI_converter.forward_coding(RLL_modulator.forward_coding(info))
        pr_signal = target_pr_channel.target_channel(codeword)
        pr_signal_jitter = target_pr_channel.jitter(pr_signal)
        pr_signal_noise = target_pr_channel.awgn(pr_signal_jitter, snr)
        
        Xs = [
            Normalized_t,
            Normalized_t,
            Normalized_t,
            Normalized_t
        ]
        Ys = [
            {'data': codeword.reshape(-1), 'label': 'binary Sequence'}, 
            {'data': pr_signal.reshape(-1), 'label': 'pr_signal', 'color': 'red'},
            {'data': pr_signal_jitter.reshape(-1), 'label': 'pr_signal_jitter', 'color': 'red'},
            {'data': pr_signal_noise.reshape(-1), 'label': f'pr_signal_noise{snr}', 'color': 'red'},
        ]
        titles = [
            'Binary Sequence',
            'pr_signal',
            'pr_signal_jitter',
            f'pr_signal_noise{snr}'
        ]
        xlabels = ["Time (t/T)"]
        ylabels = [
            "Binary",
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
        
        signal = {'data': pr_signal.reshape(-1), 'label': 'pr_signal', 'color': 'red'}
        title = 'pr_signal eyes diagram'
        xlabel = "Time (t/T)"
        ylabel = "Amplitude"
        plot_eye_diagram(
            signal=signal,
            samples_truncation=params.eye_diagram_truncation, 
            title=title,     
            xlabel=xlabel, 
            ylabel=ylabel
        )
        
        signal = {'data': pr_signal_jitter.reshape(-1), 'label': 'pr_signal_jitter', 'color': 'red'}
        title = 'pr_signal_jitter eyes diagram'
        xlabel = "Time (t/T)"
        ylabel = "Amplitude"
        plot_eye_diagram(
            signal=signal,
            samples_truncation=params.eye_diagram_truncation, 
            title=title,     
            xlabel=xlabel, 
            ylabel=ylabel
        )
        
        signal = {'data': pr_signal_noise.reshape(-1), 'label': f'pr_signal_noise{snr}', 'color': 'red'}
        title = f'pr_signal_noise{snr} eyes diagram'
        xlabel = "Time (t/T)"
        ylabel = "Amplitude"
        plot_eye_diagram(
            signal=signal,
            samples_truncation=params.eye_diagram_truncation, 
            title=title,     
            xlabel=xlabel, 
            ylabel=ylabel
        )