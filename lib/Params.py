
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
        self.equalizer_coeffs_file = "../data/equalizer_coeffs.txt"
        
        # plot params
        self.num_plots = 5
        
        # equalizer params
        self.equalizer_train_len = 1000000
        self.snr_train = 20 # add noise while train equalizer
        
        # detector/decoder params
        self.eval_info_len = 5000
        
        # rf channel params
        self.tap_bd_num = 6
        
        # target channel params
        self.PR_coefs = [1, 2, 2, 2, 1]
        
        # awgn params
        self.truncation4energy = 5000
        
        # drop len in channel
        self.drop_len = 60
        
        # dataset params
        self.train_set_batches = 10
        self.test_set_batches = 10
        self.validate_set_batches = 10
        self.data_train_len = 5000
        self.data_test_len = 5000
        self.data_val_len = 5000
        self.snr_start = 5
        self.snr_stop = 45
        self.snr_step = 1
        
        # dataloader params
        self.batch_size_train = 600
        self.batch_size_test = 600
        self.batch_size_val = 600

        # general model arch params
        self.inputx_size = 5
        self.inputy_size = 1
        self.output_size = 1
        # self.model_arch = "rnn"
        self.model_arch = "transformer"
        
        # rnn model arch params
        self.rnn_d_model = 5
        self.rnn_hidden_size = 50
        self.rnn_layer = 4
        self.rnn_dropout_ratio = 0.1
        
        # transformer model arch params
        self.transformer_d_model = 16
        self.transformer_nhead   = 4
        self.transformer_hidden_size = 50
        self.transformer_encoder_layers = 4
        self.transformer_decoder_layers = 4
        self.transformer_dropout_ratio = 0.1
        
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
        self.overlap_length = 60
    
