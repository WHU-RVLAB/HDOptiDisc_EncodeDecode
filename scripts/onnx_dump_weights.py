import onnx
import numpy as np
import os
import re

def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|\.]', '_', name)

if __name__ == "__main__":
    model_path = 'ai_sys/train/onnx/nlp_rnn.onnx'
    onnx_model = onnx.load(model_path)

    save_dir = 'ai_sys/deploy/weights'
    os.makedirs(save_dir, exist_ok=True)

    for tensor in onnx_model.graph.initializer:
        name = tensor.name
        safe_name = sanitize_filename(name)
        weight_array = onnx.numpy_helper.to_array(tensor)
        weight_array_around = np.around(weight_array, decimals=4)
        np.save(os.path.join(save_dir, f"{safe_name}.npy"), weight_array_around)
        print(f"Saved: {safe_name}, shape: {weight_array.shape}")
