import torch
import numpy as np
import os

if __name__ == "__main__":
    model_path = 'ai_sys/train/model/nlp_rnn_quant.pth.tar'
    model_state_dict = torch.load(model_path, weights_only=False)

    save_dir = 'ai_sys/deploy/weights'
    os.makedirs(save_dir, exist_ok=True)
    for key in model_state_dict:
        print(key)
        for subkey in model_state_dict[key]:
            print(subkey)
            if type(subkey) ==  str:
                print(model_state_dict[key][subkey])
                data = model_state_dict[key][subkey].to(torch.device("cpu"))
                if 'bias' in subkey:
                    save_data = data.data.cpu().numpy()
                    # save_data_around = save_data
                    save_data_around = np.around(save_data, decimals=4)
                    print(save_data)
                    print(save_data_around)
                    
                    np.save(os.path.join(save_dir, f"{key}_{subkey}.npy"), save_data_around)
                    print(f"Saved: {key}_{subkey}.npy, shape: {save_data_around.shape}")
                else:
                    save_data1 = data.int_repr().cpu().numpy()
                    # save_data_around1 = save_data1
                    save_data_around1 = np.around(save_data1, decimals=4)
                    # print(save_data1)
                    # print(save_data_around1)
                    
                    np.save(os.path.join(save_dir, f"{key}_{subkey}.npy"), save_data_around1)
                    print(f"Saved: {key}_{subkey}.npy, shape: {save_data_around1.shape}")
                    
                    save_data2 = data.q_scale()
                    # save_data_around2 = save_data2
                    save_data_around2 = np.around(save_data2, decimals=4)
                    # print(save_data2)
                    # print(save_data_around2)
                    
                    np.save(os.path.join(save_dir, f"{key}_{subkey}_scale.npy"), save_data_around2)
                    print(f"Saved: {key}_{subkey}_scale.npy")
                    
                    print(save_data_around1 * save_data_around2)
            else:
                if 'bias' in key:
                    save_data = subkey.data.cpu().numpy()
                    # save_data_around = save_data
                    save_data_around = np.around(save_data, decimals=4)
                    print(save_data)
                    print(save_data_around)
                    
                    np.save(os.path.join(save_dir, f"{key}.npy"), save_data_around)
                    print(f"Saved: {key}.npy, shape: {save_data_around.shape}")
                else:
                    save_data1 = subkey.int_repr().cpu().numpy()
                    # save_data_around1 = save_data1
                    save_data_around1 = np.around(save_data1, decimals=4)
                    # print(save_data1)
                    # print(save_data_around1)
                    
                    np.save(os.path.join(save_dir, f"{key}.npy"), save_data_around1)
                    print(f"Saved: {key}.npy, shape: {save_data_around1.shape}")
                    
                    save_data2 = subkey.q_scale()
                    # save_data_around2 = save_data2
                    save_data_around2 = np.around(save_data2, decimals=4)
                    # print(save_data2)
                    # print(save_data_around2)
                    
                    np.save(os.path.join(save_dir, f"{key}_scale.npy"), save_data_around2)
                    print(f"Saved: {key}_scale.npy")
                    
                    print(save_data_around1 * save_data_around2)
        
