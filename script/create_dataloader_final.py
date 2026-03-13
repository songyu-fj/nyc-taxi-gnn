"""
create_dataloader_final.py
数据加载器（最终版）
输入：X_train/val/test.npy 及三个邻接矩阵
输出：返回DataLoader，并保存归一化参数到 norm_params_final.npy
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

def safe_normalize_adj(adj_matrix):
    if isinstance(adj_matrix, np.ndarray):
        adj = torch.FloatTensor(adj_matrix)
    else:
        adj = adj_matrix.clone()
    adj.fill_diagonal_(0)
    adj = adj + torch.eye(adj.size(0))
    rowsum = adj.sum(dim=1).clamp(min=1e-6)
    d_inv_sqrt = torch.diag(torch.pow(rowsum, -0.5))
    norm_adj = d_inv_sqrt @ adj @ d_inv_sqrt
    return norm_adj

class IndustrialTaxiDataset(Dataset):
    def __init__(self, features, adj_spatial, adj_functional, adj_poi,
                 window_size=9, horizon=3, stride=1):
        self.n_nodes, self.n_timesteps = features.shape
        self.window_size = window_size
        self.horizon = horizon
        self.stride = stride
        self.features = torch.FloatTensor(features)
        self.adj_spatial = adj_spatial
        self.adj_functional = adj_functional
        self.adj_poi = adj_poi
        self.n_samples = max(0, (self.n_timesteps - self.window_size - self.horizon) // stride + 1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_size
        x = self.features[:, start:end].unsqueeze(-1)
        y = self.features[:, end:end + self.horizon]
        return x, y, self.adj_spatial, self.adj_functional, self.adj_poi

def industrial_collate_fn(batch):
    x_list, y_list, adj_s_list, adj_f_list, adj_p_list = zip(*batch)
    x_batch = torch.stack(x_list, dim=0)
    y_batch = torch.stack(y_list, dim=0)
    return x_batch, y_batch, adj_s_list[0], adj_f_list[0], adj_p_list[0]

def create_industrial_dataloaders(batch_size=64, window_size=9, horizon=3, stride=1):
    print("="*60)
    print("📦 创建数据加载器")
    print("="*60)

    data_dir = '../data/processed/split/'
    required = ['X_train.npy', 'X_val.npy', 'X_test.npy',
                'A_spatial.npy', 'A_functional.npy', 'A_poi.npy']
    for f in required:
        if not os.path.exists(os.path.join(data_dir, f)):
            raise FileNotFoundError(f"缺失文件: {f}")

    X_train_raw = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_val_raw = np.load(os.path.join(data_dir, 'X_val.npy'))
    X_test_raw = np.load(os.path.join(data_dir, 'X_test.npy'))

    A_spatial_raw = np.load(os.path.join(data_dir, 'A_spatial.npy'))
    A_functional_raw = np.load(os.path.join(data_dir, 'A_functional.npy'))
    A_poi_raw = np.load(os.path.join(data_dir, 'A_poi.npy'))

    print("归一化邻接矩阵...")
    A_spatial_norm = safe_normalize_adj(A_spatial_raw)
    A_functional_norm = safe_normalize_adj(A_functional_raw)
    A_poi_norm = safe_normalize_adj(A_poi_raw)

    mean = X_train_raw.mean()
    std = X_train_raw.std().clip(min=1e-6)
    X_train = (X_train_raw - mean) / std
    X_val = (X_val_raw - mean) / std
    X_test = (X_test_raw - mean) / std
    norm_params = {'mean': mean, 'std': std}

    np.save(os.path.join(data_dir, 'norm_params_final.npy'), norm_params)

    train_dataset = IndustrialTaxiDataset(X_train, A_spatial_norm, A_functional_norm, A_poi_norm,
                                          window_size, horizon, stride)
    val_dataset = IndustrialTaxiDataset(X_val, A_spatial_norm, A_functional_norm, A_poi_norm,
                                        window_size, horizon, stride)
    test_dataset = IndustrialTaxiDataset(X_test, A_spatial_norm, A_functional_norm, A_poi_norm,
                                         window_size, horizon, stride)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=industrial_collate_fn, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=industrial_collate_fn, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=industrial_collate_fn, num_workers=2)

    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(val_dataset)}")
    print(f"测试样本数: {len(test_dataset)}")
    print("="*60)

    return train_loader, val_loader, test_loader, norm_params

if __name__ == '__main__':
    train_loader, _, _, _ = create_industrial_dataloaders(batch_size=4)
    x, y, adj_s, adj_f, adj_p = next(iter(train_loader))
    print(f"x shape: {x.shape}, y shape: {y.shape}")
    print(f"adj_s shape: {adj_s.shape}")