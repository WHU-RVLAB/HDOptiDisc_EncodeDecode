
class Params:
    """
    @brief Parameter class
    """
    def __init__(self):
        """
        @brief initialization
        """
        # io dir or files
        self.model_dir = "../model/"
        self.result_file = 'result.txt'
        self.equalizer_coeffs_dir = "../data"
        self.equalizer_coeffs_jitter_sine_file = "../data/equalizer_coeffs_jitter_sine.txt"
        self.equalizer_coeffs_jitter_file = "../data/equalizer_coeffs_jitter.txt"
        self.equalizer_coeffs_sine_file = "../data/equalizer_coeffs_sine.txt"
        self.equalizer_coeffs_file = "../data/equalizer_coeffs.txt"
        
        # plot params
        self.num_plots = 5
        self.eye_diagram_truncation = 3
        
        # equalizer params
        self.equalizer_train_len = 50000
        self.snr_train = 30 # add noise while train equalizer
        self.equalizer_taps_num = 15
        self.equalizer_mu = 0.01
        
        # detector/decoder params
        self.eval_info_len = 1000000
        
        # rf channel params
        self.bd_scaling_para = 0.165 # equal to 0.5*E_b, as bd read channel, it equal to 0.165
        self.tap_bd_num = 6
        self.jitteron = True
        self.addsineon = True
        self.signal_norm = True
        
        # target channel params
        self.PR_coefs = [1, 2, 2, 2, 1]
        
        # awgn params
        self.truncation4energy = 5000
        
        # jitter params
        self.jcl_start = 0.04
        self.jcl_stop = 0.10
        self.upsample_factor = 100
        
        # modules test params
        self.module_test_len = 1000
        
        # dataset params
        self.block_length = 900
        self.snr_size = 40
        self.snr_start = 5
        self.snr_stop = 45
        self.snr_step = 1
        self.train_num_probs = 120
        self.test_num_probs = 120
        self.val_num_probs = 600
        
        # dataloader params
        self.batch_size_train = 600
        self.batch_size_test = 600
        self.batch_size_val = 600

        # general model arch params
        self.classifier_input_size = 6 # dimension of a feature should always equal to length of channel memory length
        self.classifier_output_size = 1 # model determine whether the current bit is 0 or 1
        self.nlp_input_size = 1
        self.nlp_output_size = 1
        # self.model_arch = "lr"
        # self.model_arch = "xgboost"
        # self.model_arch = "mlp"
        # self.model_arch = "cnn"
        # self.model_arch = "unet"
        self.model_arch = "rnn"
        # self.model_arch = "rnn_scratch"
        # self.model_arch = "transformer"
        
        # mlp model arch params
        self.mlp_d_model = 12
        self.mlp_hidden_size = 24
        self.mlp_dropout_ratio = 0.0
        
        # cnn model arch params
        self.cnn_d_model = 6
        self.cnn_hidden_size = 8
        self.cnn_kernel_size = 3
        self.cnn_dropout_ratio = 0.0
        
        # unet1D model arch params
        self.unet_d_model = 16
        self.unet_base_filters = 4
        self.unet_kernel_size = 3
        self.unet_dropout_ratio = 0.1
        
        # rnn model arch params
        self.rnn_d_model = 6
        self.rnn_hidden_size = 8
        self.rnn_layer = 1
        self.rnn_dropout_ratio = 0.0
        
        # transformer model arch params
        self.transformer_d_model = 6
        self.transformer_nhead   = 2
        self.transformer_hidden_size = 8
        self.transformer_encoder_layers = 1
        self.transformer_decoder_layers = 1
        self.transformer_dropout_ratio = 0.0
        
        # train params
        self.num_epoch = 40
        self.eval_freq = 5
        self.eval_start = 0
        self.print_freq_ep = 5
        
        # optimizer params
        self.learning_rate = 0.001 
        self.momentum = 0.9
        self.weight_decay = 0.0001
        
        # model infer params
        self.eval_length = 60
        self.overlap_length = 0
        
        # npml params
        self.noise_predictor_nums = 8
    
