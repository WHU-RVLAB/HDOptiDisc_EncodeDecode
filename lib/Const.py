import numpy as np

# Constant
def RLL_state_machine():
    # (1,7)-RLL constraint, 4 states, 4 error propagations
    # Encoder_Dict[a][b]: a stands for each state, b stands for (1 - input tags, 2 - output words, 3 - next state)
    encoder_dict = {
        1 : {
            'input' : np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            'output' : np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0]]),
            'next_state' : np.array([[1], [2], [3], [3]])
        },
        2 : {
            'input' : np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            'output' : np.array([[1, 0, 0], [1, 0, 0], [1, 0, 1], [1, 0, 1]]),
            'next_state' : np.array([[1], [2], [3], [4]])
        },
        3 : {
            'input' : np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            'output' : np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 1]]),
            'next_state' : np.array([[1], [2], [3], [4]])
        },
        4 : {
            'input' : np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            'output' : np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0]]),
            'next_state' : np.array([[1], [2], [3], [3]])
        }
    }
    
    encoder_definite = {'m' : 0, 'a' : 2}
    
    return encoder_dict, encoder_definite

# Constant
def Target_channel_state_machine():
    # channel state machine
    channel_dict = {
        'state_machine' : np.array([
            [0, 0], [0, 1], [1, 2], [2, 3], [2, 4], [3, 7], [4, 8], [4, 9], 
            [5, 0], [5, 1], [6, 2], [7, 5], [7, 6], [8, 7], [9, 8], [9, 9]
        ]),
        'in_out' : np.array([
            [0, 0], [1, 1], [1, 3], [0, 4], [1, 5], [0, 4], [0, 6], [1, 7], 
            [0, 1], [1, 2], [1, 4], [0, 3], [1, 4], [0, 5], [0, 7], [1, 8]
        ]),
        'state_label' : np.array([
            [0, 0, 0, 0, 0], [0, 0, 0, 1, 1], [0, 0, 1, 1, 2],
            [0, 1, 1, 0, 3], [0, 1, 1, 1, 4], [1, 0, 0, 0, 5],
            [1, 0, 0, 1, 6], [1, 1, 0, 0, 7], [1, 1, 1, 0, 8], [1, 1, 1, 1, 9]
        ]),
        'num_state' : 10,
        'ini_state' : 0
    }
    
    return channel_dict

# Constant
def Target_channel_dummy_bits():
    # List of transition paths from zero state to start state
    dummy_start_paths = {
        0 : np.array([[0, 0, 0, 0, 0]]), 
        1 : np.array([[0, 0, 0, 0, 1]]),
        2 : np.array([[0, 0, 0, 1, 2]]), 
        3 : np.array([[0, 0, 1, 2, 3]]),
        4 : np.array([[0, 0, 1, 2, 4]]), 
        5 : np.array([[1, 2, 3, 7, 5]]),
        6 : np.array([[1, 2, 3, 7, 6]]), 
        7 : np.array([[0, 1, 2, 3, 7]]),
        8 : np.array([[0, 1, 2, 4, 8]]), 
        9 : np.array([[0, 1, 2, 4, 9]])
    }

    # corresponds to input bits in dummy_start_paths
    dummy_start_input = {
        0 : np.array([[0, 0, 0, 0, 0]]), 
        1 : np.array([[0, 0, 0, 0, 1]]),
        2 : np.array([[0, 0, 0, 1, 1]]), 
        3 : np.array([[0, 0, 1, 1, 0]]),
        4 : np.array([[0, 0, 1, 1, 1]]), 
        5 : np.array([[1, 1, 0, 0, 0]]), 
        6 : np.array([[1, 1, 0, 0, 1]]), 
        7 : np.array([[0, 1, 1, 0, 0]]), 
        8 : np.array([[0, 1, 1, 1, 0]]), 
        9 : np.array([[0, 1, 1, 1, 1]]),
    }

    # corresponds to channel output in dummy_start_paths
    dummy_start_output = {
        0 : np.array([[0, 0, 0, 0, 0]]), 
        1 : np.array([[0, 0, 0, 0, 1]]),
        2 : np.array([[0, 0, 0, 1, 3]]), 
        3 : np.array([[0, 0, 1, 3, 4]]),
        4 : np.array([[0, 0, 1, 3, 5]]), 
        5 : np.array([[1, 3, 4, 4, 3]]),
        6 : np.array([[1, 3, 4, 4, 4]]), 
        7 : np.array([[0, 1, 3, 4, 4]]),
        8 : np.array([[0, 1, 3, 5, 6]]), 
        9 : np.array([[0, 1, 3, 3, 7]])
    }

    dummy_start_eval = np.array([[0, 0, 0, 0, 0]])

    # List of transition paths from end state to zero state 
    dummy_end_paths = {
        0 : np.array([[0, 0, 0, 0, 0]]), 
        1 : np.array([[2, 3, 7, 5, 0]]),
        2 : np.array([[3, 7, 5, 0, 0]]), 
        3 : np.array([[7, 5, 0, 0, 0]]),
        4 : np.array([[8, 7, 5, 0, 0]]), 
        5 : np.array([[0, 0, 0, 0, 0]]),
        6 : np.array([[2, 3, 7, 5, 0]]), 
        7 : np.array([[5, 0, 0, 0, 0]]),
        8 : np.array([[7, 5, 0, 0, 0]]), 
        9 : np.array([[8, 7, 5, 0, 0]])
    }

    # corresponds to input bits in dummy_end_paths
    dummy_end_input = {
        0 : np.array([[0, 0, 0, 0, 0]]), 
        1 : np.array([[1, 0, 0, 0, 0]]),
        2 : np.array([[0, 0, 0, 0, 0]]), 
        3 : np.array([[0, 0, 0, 0, 0]]),
        4 : np.array([[0, 0, 0, 0, 0]]), 
        5 : np.array([[0, 0, 0, 0, 0]]), 
        6 : np.array([[1, 0, 0, 0, 0]]), 
        7 : np.array([[0, 0, 0, 0, 0]]), 
        8 : np.array([[0, 0, 0, 0, 0]]), 
        9 : np.array([[0, 0, 0, 0, 0]]),
    }
    
    # corresponds to channel output in dummy_end_paths
    dummy_end_output = {
        0 : np.array([[0, 0, 0, 0, 0]]), 
        1 : np.array([[3, 4, 4, 3, 1]]),
        2 : np.array([[4, 4, 3, 1, 0]]), 
        3 : np.array([[4, 3, 1, 0, 0]]),
        4 : np.array([[6, 5, 3, 1, 0]]), 
        5 : np.array([[1, 0, 0, 0, 0]]),
        6 : np.array([[4, 4, 4, 3, 1]]), 
        7 : np.array([[3, 1, 0, 0, 0]]), 
        8 : np.array([[5, 3, 1, 0, 0]]), 
        9 : np.array([[7, 5, 3, 1, 0]]),
    }
    
    dummy_end_eval = np.array([[0, 0, 0, 0, 0]])

    
    return dummy_start_paths, dummy_start_input, dummy_start_output, dummy_start_eval, \
    dummy_end_paths, dummy_end_input, dummy_end_output, dummy_end_eval