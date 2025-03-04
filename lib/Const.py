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
        ], dtype=np.float32),
        'state_label' : np.array([
            [0, 0, 0, 0, 0], [0, 0, 0, 1, 1], [0, 0, 1, 1, 2],
            [0, 1, 1, 0, 3], [0, 1, 1, 1, 4], [1, 0, 0, 0, 5],
            [1, 0, 0, 1, 6], [1, 1, 0, 0, 7], [1, 1, 1, 0, 8], [1, 1, 1, 1, 9]
        ]),
        'num_state' : 10,
        'ini_state' : 0
    }
    
    return channel_dict