import numpy as np
import sys
import os
np.set_printoptions(threshold=sys.maxsize)

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__))))
from lib.Const import RLL_state_machine, Target_channel_state_machine
from lib.Params import Params
from lib.Channel_Modulator import RLL_Modulator
from lib.Channel_Converter import NRZI_Converter
from lib.Disk_Read_Channel import Disk_Read_Channel
from lib.Target_PR_Channel import Target_PR_Channel
from lib.Adaptive_Equalizer import Adaptive_Equalizer
from lib.Utils import plot_separated, find_index
sys.path.pop()

np.random.seed(12345)

def realistic_sys(params:Params):
    
    # constant and input paras
    encoder_dict, encoder_definite = RLL_state_machine()
    channel_dict = Target_channel_state_machine()

    # Initial metric 
    ini_metric = 1000 * np.ones((channel_dict['num_state'], 1))
    ini_metric[0, 0] = 0
    ini_metric_pr = ini_metric
    
    # rate for constrained code
    num_sym_in_constrain = encoder_dict[1]['input'].shape[1]
    num_sym_out_constrain = encoder_dict[1]['output'].shape[1]
    rate_constrain = num_sym_in_constrain / num_sym_out_constrain
    dummy_len = int(params.overlap_length * num_sym_in_constrain 
                 / num_sym_out_constrain)
    
    # class
    RLL_modulator = RLL_Modulator(encoder_dict, encoder_definite)
    NRZI_converter = NRZI_Converter()
    disk_read_channel = Disk_Read_Channel(params)
    target_pr_channel = Target_PR_Channel(params)
    viterbi_detector = Viterbi(params, channel_dict, ini_metric)
    
    pr_adaptive_equalizer = Adaptive_Equalizer(        
        equalizer_input  = None,
        reference_signal = None,
        taps_num = 15,
        mu = 0.01
    )
    pr_adaptive_equalizer.equalizer_coeffs = np.loadtxt(params.equalizer_coeffs_file).reshape(1, -1)
    print(f"\nload equalizer_coeffs from txt files:{params.equalizer_coeffs_file}")
    print(f"\nequalizer_coeffs are {pr_adaptive_equalizer.equalizer_coeffs}")
    
    # define ber
    num_ber = int((params.snr_stop-params.snr_start)/params.snr_step+1)
    codeword_len = int(params.real_test_len/rate_constrain)
    ber_channel = np.zeros((1, num_ber))
    ber_info = np.zeros((1, num_ber))
    
    # eval mode
    for idx in np.arange(0, num_ber):
        snr = params.snr_start+idx*params.snr_step
        
        info = np.random.randint(2, size = (1, params.real_test_len + dummy_len))
        codeword = NRZI_converter.forward_coding(RLL_modulator.forward_coding(info))
        
        rf_signal = disk_read_channel.RF_signal(codeword)
        equalizer_input = disk_read_channel.awgn(rf_signal, snr)
        pr_signal = target_pr_channel.target_channel(codeword)
        pr_signal_noise = target_pr_channel.awgn(pr_signal, snr)
        
        length = equalizer_input.shape[1]
        detectword = np.empty((1, 0))
        detectword_pr = np.empty((1, 0))
        for pos in range(0, length - params.overlap_length, params.eval_length):
            
            codeword_truncation = codeword[:, pos:pos+params.eval_length+params.overlap_length]
            rf_signal_truncation = rf_signal[:, pos:pos+params.eval_length+params.overlap_length]
            equalizer_input_truncation = equalizer_input[:, pos:pos+params.eval_length+params.overlap_length]
            pr_signal_truncation = pr_signal[:, pos:pos+params.eval_length+params.overlap_length]
            pr_signal_noise_truncation = pr_signal_noise[:, pos:pos+params.eval_length+params.overlap_length]
            
            pr_adaptive_equalizer.equalizer_input = equalizer_input_truncation
            detector_input = pr_adaptive_equalizer.equalized_signal()
            
            dec_tmp, metric_next = viterbi_detector.vit_dec(detector_input, ini_metric)
            ini_metric = metric_next
            detectword = np.append(detectword, dec_tmp, axis=1)
            
            dec_tmp_pr, metric_next_pr = viterbi_detector.vit_dec(pr_signal_noise_truncation, ini_metric_pr)
            ini_metric_pr = metric_next_pr
            detectword_pr = np.append(detectword_pr, dec_tmp_pr, axis=1)
            
            # # eval mode 
            # # Guarantees that the equalizer can track changes in channel characteristics
            # pr_adaptive_equalizer.equalizer_input  = equalizer_input_truncation
            # pr_adaptive_equalizer.reference_signal = pr_signal_truncation
            # detector_input_train, error_signal, error_signal_square, equalizer_coeffs = pr_adaptive_equalizer.lms()
            
            # Normalized_t = np.linspace(1, params.eval_length+params.overlap_length, params.eval_length+params.overlap_length)
            # Xs = [
            #     Normalized_t,
            #     Normalized_t,
            #     Normalized_t,
            #     Normalized_t,
            #     Normalized_t,
            #     Normalized_t
            # ]
            # Ys = [
            #     {'data': codeword_truncation.reshape(-1), 'label': 'binary Sequence'}, 
            #     {'data': rf_signal_truncation.reshape(-1), 'label': 'rf_signal_truncation', 'color': 'red'},
            #     {'data': equalizer_input_truncation.reshape(-1), 'label': f'equalizer_input_truncation_snr{snr}', 'color': 'red'},
            #     {'data': pr_signal_truncation.reshape(-1), 'label': 'pr_signal_truncation', 'color': 'red'},
            #     {'data': detector_input.reshape(-1), 'label': 'detector_input', 'color': 'red'},
            #     {'data': pr_signal_noise_truncation.reshape(-1), 'label': 'pr_signal_noise_truncation', 'color': 'red'},
            # ]
            # titles = [
            #     'codeword_truncation',
            #     'rf_signal_truncation',
            #     f'equalizer_input_truncation_snr{snr}',
            #     'pr_signal_truncation',
            #     'detector_input',
            #     'pr_signal_noise_truncation'
            # ]
            # xlabels = ["Time (t/T)"]
            # ylabels = [
            #     "Binary",
            #     "Amplitude",
            #     "Amplitude",
            #     "Amplitude",
            #     "Amplitude",
            #     "Amplitude",
            # ]
            # plot_separated(
            #     Xs=Xs, 
            #     Ys=Ys, 
            #     titles=titles,     
            #     xlabels=xlabels, 
            #     ylabels=ylabels
            # )
            
            # Xs = [
            #     Normalized_t,
            #     Normalized_t,
            #     Normalized_t,
            #     np.arange(equalizer_coeffs.shape[1])
            # ]
            # Ys = [
            #     {'data': detector_input_train.reshape(-1), 'label': 'detector_input_train', 'color': 'red'},
            #     {'data': error_signal.reshape(-1), 'label': 'error_signal', 'color': 'red'},
            #     {'data': error_signal_square.reshape(-1), 'label': 'error_signal_square', 'color': 'red'},
            #     {'data': equalizer_coeffs.reshape(-1), 'label': 'equalizer_coeffs', 'color': 'red'}
            # ]
            # titles = [
            #     'detector_input_train',
            #     'error_signal',
            #     'error_signal_square',
            #     'equalizer_coeffs'
            # ]
            # xlabels = [
            #     "Time (t/T)",
            #     "Time (t/T)",
            #     "Time (t/T)",
            #     "equalizer_coeffs idx"
            # ]
            # ylabels = ["Amplitude"]
            # plot_separated(
            #     Xs=Xs, 
            #     Ys=Ys, 
            #     titles=titles,     
            #     xlabels=xlabels, 
            #     ylabels=ylabels
            # )
        
        print("The SNR is:")
        print(snr)
        ber = (np.count_nonzero(np.abs(codeword[:, 0:codeword_len] - detectword[:, 0:codeword_len])) 
               / codeword_len)
        print("The bit error rate (BER) is:")
        print(ber)
        ber_pr = (np.count_nonzero(np.abs(codeword[:, 0:codeword_len] - detectword_pr[:, 0:codeword_len])) 
               / codeword_len)
        print("The bit error rate (BER) in Target PR channel is:")
        print(ber_pr)

## Detector: Viterbi detector
class Viterbi(object):
    def __init__(self, params:Params, channel_dict, ini_metric):
        self.params = params
        self.channel_dict = channel_dict
        self.ini_metric = ini_metric
        self.num_state = self.channel_dict['num_state']
    
    def vit_dec(self, r_truncation, ini_metric):
        
        r_len = r_truncation.shape[1]
        ini_metric_trun = ini_metric
        path_survivor = np.zeros((self.num_state, r_len))
        state_metric_trun = np.zeros((self.num_state, self.params.eval_length))
        
        for idx in range(r_len):
            state_path, state_metric = self.metric(r_truncation[:, idx], 
                                                   ini_metric_trun)
            
            ini_metric_trun = state_metric
            path_survivor[:, idx:idx+1] = state_path
            if idx == self.params.eval_length-1:
                state_metric_next = state_metric
        
        state_min = np.argmin(state_metric, axis=0)[0]
        path = self.path_convert(path_survivor)
        dec_word = self.path_to_word(path, state_min)
        
        return dec_word[:, :self.params.eval_length], state_metric_next
        
    def metric(self, r, metric_last):
        '''
        Input: branch metrics at one time step
        Output: branch metric and survivor metric for the next step 
        Mapping: choose the shorest path between adjacent states
        '''
        
        path_survivor, metric_survivor = (np.zeros((self.num_state, 1)), 
                                          np.zeros((self.num_state, 1)))
        
        for state in range(self.num_state):
            set_in = np.where(self.channel_dict['state_machine'][:, 1]==state)[0]
            metric_tmp = np.zeros((set_in.shape[0], 1))
            for i in range(set_in.shape[0]):
                metric_tmp[i, :] = (metric_last[self.channel_dict['state_machine'][set_in[i], 0], :][0] + 
                                    self.euclidean_distance(r, self.channel_dict['in_out'][set_in[i], 1]))
            metric_survivor[state, :] = metric_tmp.min()
            # if we find equal minimum branch metric, we choose the upper path
            path_survivor[state, :] = (
                self.channel_dict['state_machine'][set_in[np.where(metric_tmp==metric_tmp.min())[0][0]], 0])
        return path_survivor, metric_survivor
                
    
    def path_convert(self, path_survivor):
        '''
        Input: (num_state, length) array
        Output: (num_state, length) array
        Mapping: Viterbi detector for a truncation part
        '''
        
        path_truncation = np.zeros(path_survivor.shape)
        path_truncation[:, -1:] = path_survivor[:, -1:]
        for state in range(self.num_state):
            for i in range(path_survivor.shape[1]-2, -1, -1):
                path_truncation[state, i] = int(path_survivor[
                    int(path_truncation[state, i+1]), i])
        
        return path_truncation
    
    def path_to_word(self, path, state):
        '''
        Input: (1, length) array
        Output: (1, length) array
        Mapping: connection between two states determines one word
        '''
        
        length = path.shape[1]
        word = np.zeros((1, length))
        for i in range(length-1):
            idx = find_index(self.channel_dict['state_machine'], path[state, i : i+2])
            word[:, i] = self.channel_dict['in_out'][idx, 0]
        
        idx = find_index(self.channel_dict['state_machine'], np.array([path[state, -1], state]))
        word[:, -1] = self.channel_dict['in_out'][idx, 0]
        return word
    
    def euclidean_distance(self, x, y):
        return np.sum((x - y) ** 2)

if __name__ == '__main__':
    params = Params()
    realistic_sys(params)