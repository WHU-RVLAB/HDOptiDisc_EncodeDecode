import sys
import os
import numpy as np
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__))))
from lib.Params import Params
from lib.Utils import plot_altogether
sys.path.pop()   

def find_result_files(path):
    result_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('_result.txt'):
                result_files.append(os.path.join(root, file))
    return result_files
    
if __name__ == '__main__': 
    params = Params()
    X = range(params.snr_start, params.snr_stop + 1)
    Ys = []
    
    data_dir = params.algorithm_result_dir
    files = find_result_files(data_dir)
    colors = [
        '#1f77b4',  # 蓝色
        '#ff7f0e',  # 橙色
        '#2ca02c',  # 绿色
        '#d62728',  # 红色
        '#9467bd',  # 紫色
        '#8c564b',  # 棕色
        '#e377c2',  # 粉色
        '#7f7f7f',  # 灰色
        '#bcbd22',  # 黄色
        '#17becf'   # 青色
    ]
    
    for idx, file_path in enumerate(files):
        
        filename = os.path.basename(file_path)
        label = filename.split('_')[0:-1]
        
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.read().splitlines()
            data = [float(item) for item in data]
        
        data = np.array(data, dtype=np.float64)
        data = np.where(data <= 1e-5, 1e-5, data)
        data = np.log10(data)
        
        color = colors[idx % len(colors)]
        
        Ys.append({
            'data': data,
            'label': label,
            'color': color
        })
        
    plot_altogether(
        X=X, 
        Ys=Ys, 
        title='Bers of different algorithm Decoders for BDs',     
        xlabel="SNRs", 
        ylabel="Bers(log10)"
    )