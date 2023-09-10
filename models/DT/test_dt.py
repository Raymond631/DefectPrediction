# 单个arff数据集文件处理
import os

import arff
import numpy as np
import pandas as pd


def read_arff_file(file_path):
    data, meta = arff.loadarff(file_path)
    data_array = np.array(data.tolist())
    features = data_array[:, :-1]
    labels = data_array[:, -1]
    labels = np.where(labels == b'N', 0, 1)
    return features, labels

def folder_arff(folder_path):
    arff_files = [f for f in os.listdir(folder_path) if f.endswith('.arff')]
    combined_data = pd.DataFrame()
    for filename in arff_files:
        file_path = os.path.join(folder_path, filename)
        data, meta = arff.loadarff(file_path)
        df = pd.DataFrame(data)
        combined_data = pd.concat([combined_data, df], ignore_index=True)
    combined_data.iloc[:, -1] = combined_data.iloc[:, -1].apply(lambda x: 0 if x == b'N' else 1)
    X = combined_data.iloc[:, :-1]
    y = combined_data.iloc[:, -1].astype(int)
    return X,y