
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
        self.model_file = 'model.pth.tar'
        self.equalizer_coeffs_dir = "../data"
        self.equalizer_coeffs_file = "../data/equalizer_coeffs.txt"
        
        # plot params
        self.num_plots = 5
        
        # equalizer params
        self.equalizer_train_len = 1000000
        
        # bd symbol response params
        self.tap_bd_num = 6
        
        # target channel params
        self.PR_coefs = [1, 2, 2, 2, 1]
        
        # awgn params
        self.truncation4energy = 5000
        
        # dataset params
        self.train_set_batches = 100
        self.test_set_batches = 50
        self.validate_set_batches = 10000
        self.batch_size_train = 300
        self.batch_size_test = 600
        self.batch_size_val = 600

        # model arch params
        self.input_size = 5
        self.rnn_input_size = 5
        self.rnn_hidden_size =50
        self.output_size = 1
        self.rnn_layer = 10
        self.rnn_dropout_ratio = 0
        
        # train params
        self.num_epoch = 400
        self.eval_freq = 5
        self.eval_start = 200
        self.print_freq_ep = 5
        
        # optimizer params
        self.learning_rate = 0.001 
        self.momentum = 0.9
        self.weight_decay = 0.0001
        
        # model infer params
        self.real_test_len = 1000000
        self.real_eval_len = 5000
        self.eval_length = 60
        self.overlap_length = 60
        self.snr_start = 5
        self.snr_stop = 45
        self.snr_step = 1
        self.snr_train = 20 # add noise while train equalizer
    
