import os
import torch
import numpy as np
import pandas as pd
import pickle


def get_id_to_item(data_dir, dataset):
    with open(os.path.join(data_dir, f"id_to_item_{dataset}.pkl"), "rb") as pf:
        id_to_item = pickle.load(pf)
    return id_to_item


def get_inter_data(data_dir, dataset):
    train = pd.read_parquet(os.path.join(data_dir, f"train_{dataset}.parquet"))
    valid = pd.read_parquet(os.path.join(data_dir, f"valid_{dataset}.parquet"))
    test = pd.read_parquet(os.path.join(data_dir, f"test_{dataset}.parquet"))
    return (train, valid, test)


def get_feature(content_dir, dataset, feature_cols, dtype=torch.float):
    return [
        torch.tensor(
            np.load(os.path.join(content_dir, f"{col}_{dataset}.npy")),
            dtype=dtype,
        )
        for col in feature_cols
    ]
