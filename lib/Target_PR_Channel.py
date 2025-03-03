import sys
import os
import numpy as np
sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)))
from Const import RLL_state_machine
from Channel_Modulator import RLL_Modulator
from Channel_Converter import NRZI_Converter
from Target_PR_Response import partial_response
from Utils import plot_separated, plot_eye_diagram
from Params import Params
sys.path.pop()
    
class Target_PR_Channel(object):
    
    def __init__(self, params:Params):      
        self.params = params
        upsample_factor = params.upsample_factor
        _, PR_coefs = partial_response(PR_coefs=params.PR_coefs, bit_periods = 10, upsample_factor=upsample_factor)
        self.PR_coefs = PR_coefs.reshape(1,-1)
        mid_idx = len(PR_coefs)//2
        self.PR_coefs = PR_coefs[mid_idx : mid_idx + upsample_factor*len(params.PR_coefs)].reshape(1,-1)
        
        print('\nTarget Channel coefficient is')
        print(PR_coefs)
        print(f"PR_coefs.shape: {PR_coefs.shape}")
        print('\nTap target Channel coefficient is')
        print(self.PR_coefs)
        print(f"self.PR_coefs.shape: {self.PR_coefs.shape}")
    
    def target_channel_jitter(self, codeword):
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
        PR_coefs_sum = sum(self.PR_coefs[0, :][::downsample_factor])
        PR_coefs_upsample_sum = sum(self.PR_coefs[0, :])
        if not params.signal_norm:
            PR_coefs_upsample_sum /= PR_coefs_sum
            PR_coefs_sum = 1
        
        pr_signal_ideal = (np.convolve(self.PR_coefs[0, :][::downsample_factor], codeword[0, :])
               [:-(len(self.params.PR_coefs) - 1)].reshape(codeword.shape))/PR_coefs_sum
        
        pr_signal_real = (np.convolve(self.PR_coefs[0, :], signal_upsample_jittered[0, :])
               [:-(upsample_factor*len(self.params.PR_coefs) - 1)].reshape(signal_upsample_jittered.shape))/PR_coefs_upsample_sum
        
        pr_signal_real = pr_signal_real[:, ::downsample_factor]
        
        return signal_upsample_ideal, signal_upsample_jittered, pr_signal_ideal, pr_signal_real
    
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
    
    Normalized_t = np.linspace(0, int(params.module_test_len/rate_constrain) - 1, int(params.module_test_len/rate_constrain))
    Normalized_t_upsample = np.linspace(0, int(params.module_test_len/rate_constrain) - 1/params.upsample_factor, params.upsample_factor*int(params.module_test_len/rate_constrain))
    
    for idx in np.arange(0, num_ber):
        snr = params.snr_start+idx*params.snr_step
        
        info = np.random.randint(2, size = (1, params.module_test_len))
        codeword = NRZI_converter.forward_coding(RLL_modulator.forward_coding(info))
        signal_upsample_ideal, signal_upsample_jittered, pr_signal_ideal, pr_signal_real = target_pr_channel.target_channel_jitter(codeword)
        pr_signal_noise = target_pr_channel.awgn(pr_signal_real, snr)
        
        Xs = [
            Normalized_t_upsample,
            Normalized_t_upsample
        ]
        Ys = [
        {'data': signal_upsample_ideal.reshape(-1), 'label': 'signal_upsample_ideal', 'color': 'red'}, 
        {'data': signal_upsample_jittered.reshape(-1), 'label': 'signal_upsample_jittered', 'color': 'red'}
        ]
        titles = [
            'signal_upsample_ideal',
            'signal_upsample_jittered'
        ]
        xlabels = ["Time (t/T)"]
        ylabels = [
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

        Xs = [
            Normalized_t,
            Normalized_t,
            Normalized_t
        ]
        Ys = [
            {'data': pr_signal_ideal.reshape(-1), 'label': 'pr_signal_ideal', 'color': 'red'}, 
            {'data': pr_signal_real.reshape(-1), 'label': 'pr_signal_real', 'color': 'red'},
            {'data': pr_signal_noise.reshape(-1), 'label': f'pr_signal_noise{snr}', 'color': 'red'},
        ]
        titles = [
            'pr_signal_ideal',
            'pr_signal_real',
            f'pr_signal_noise{snr}'
        ]
        xlabels = ["Time (t/T)"]
        ylabels = [
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
        
        signal = {'data': pr_signal_ideal.reshape(-1), 'label': 'pr_signal_ideal', 'color': 'black'}
        title = 'pr_signal_ideal eyes diagram'
        xlabel = "Time (t/T)"
        ylabel = "Amplitude"
        plot_eye_diagram(
            signal=signal,
            samples_truncation=params.eye_diagram_truncation, 
            title=title,     
            xlabel=xlabel, 
            ylabel=ylabel
        )
        
        signal = {'data': pr_signal_real.reshape(-1), 'label': 'pr_signal_real', 'color': 'black'}
        title = 'pr_signal_real eyes diagram'
        xlabel = "Time (t/T)"
        ylabel = "Amplitude"
        plot_eye_diagram(
            signal=signal,
            samples_truncation=params.eye_diagram_truncation, 
            title=title,     
            xlabel=xlabel, 
            ylabel=ylabel
        )

        signal = {'data': pr_signal_noise.reshape(-1), 'label': f'pr_signal_noise{snr}', 'color': 'black'}
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