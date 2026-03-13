"""
04_build_functional_adjacency.py
构建功能相似图（基于保留后的训练集）
输入：data/processed/split/X_train.npy
输出：data/processed/split/A_functional.npy
"""

import numpy as np
import os
from tqdm import tqdm

# ==================== 配置 ====================
CONFIG = {
    'input_path': '../data/processed/split/X_train.npy',
    'output_dir': '../data/processed/split/',
    'time_slices_per_day': 72,
    'top_k': 15,
    'similarity_threshold': 0.3,
    'min_degree': 5,
}

def extract_time_aware_features(X_train, T=72):
    n_nodes, n_timesteps = X_train.shape
    n_days = n_timesteps // T
    usable = n_days * T
    X = X_train[:, :usable].reshape(n_nodes, n_days, T)

    avg_pattern = X.mean(axis=1)
    std_pattern = X.std(axis=1)

    morning_idx = slice(24, 36)
    evening_idx = slice(51, 63)
    daytime_idx = slice(24, 72)
    night_idx = slice(0, 24)

    morning_avg = avg_pattern[:, morning_idx].mean(axis=1, keepdims=True)
    evening_avg = avg_pattern[:, evening_idx].mean(axis=1, keepdims=True)
    day_avg = avg_pattern[:, daytime_idx].mean(axis=1, keepdims=True)
    night_avg = avg_pattern[:, night_idx].mean(axis=1, keepdims=True)
    day_night_ratio = day_avg / (night_avg + 1e-5)

    peak_time = np.argmax(avg_pattern, axis=1, keepdims=True) / T
    peak_intensity = np.max(avg_pattern, axis=1, keepdims=True)
    avg_intensity = avg_pattern.mean(axis=1, keepdims=True)
    peak_ratio = peak_intensity / (avg_intensity + 1e-5)

    cv = std_pattern.mean(axis=1, keepdims=True) / (avg_intensity + 1e-5)

    scalar_features = np.hstack([morning_avg, evening_avg, day_night_ratio,
                                 peak_time, peak_ratio, cv])
    feature_matrix = np.hstack([avg_pattern, std_pattern, scalar_features])
    return feature_matrix

def compute_sparse_adj(features, top_k, threshold):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    features_norm = features / norms
    sim = features_norm @ features_norm.T
    np.fill_diagonal(sim, -1)

    n = features.shape[0]
    adj = np.zeros((n, n), dtype=np.float32)
    for i in tqdm(range(n), desc="选择Top-K邻居"):
        s = sim[i].copy()
        valid = np.where(s >= threshold)[0]
        if len(valid) >= top_k:
            top_idx = valid[np.argsort(s[valid])[-top_k:]]
        else:
            top_idx = np.argsort(s)[-top_k:]
        adj[i, top_idx] = 1.0
        adj[top_idx, i] = 1.0
    return adj, sim

def ensure_min_degree(adj, sim, min_degree):
    degrees = adj.sum(axis=1)
    low_nodes = np.where(degrees < min_degree)[0]
    if len(low_nodes) == 0:
        return adj
    print(f"发现 {len(low_nodes)} 个度数不足的节点，进行修复...")
    sim_filled = sim.copy()
    np.fill_diagonal(sim_filled, -1)
    for node in low_nodes:
        needed = min_degree - degrees[node]
        if needed <= 0:
            continue
        candidates = np.argsort(sim_filled[node])[-(needed+10):]
        candidates = [c for c in candidates if sim_filled[node, c] > 0 and adj[node, c] == 0]
        for c in candidates[:needed]:
            adj[node, c] = 1.0
            adj[c, node] = 1.0
    return adj

def safe_normalize_adj(adj):
    adj = adj + np.eye(adj.shape[0])
    deg = adj.sum(axis=1).clip(min=1e-6)
    deg_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    return deg_inv_sqrt @ adj @ deg_inv_sqrt

def main():
    print("="*60)
    print("🧠 构建功能相似图（基于保留网格）")
    print("="*60)

    if not os.path.exists(CONFIG['input_path']):
        print(f"❌ 训练数据不存在: {CONFIG['input_path']}")
        return

    X_train = np.load(CONFIG['input_path'])
    print(f"训练集形状: {X_train.shape}")

    print("提取时间感知特征...")
    features = extract_time_aware_features(X_train, CONFIG['time_slices_per_day'])
    print(f"特征矩阵形状: {features.shape}")

    print("计算相似度并稀疏化...")
    adj, sim = compute_sparse_adj(features, CONFIG['top_k'], CONFIG['similarity_threshold'])

    if CONFIG['min_degree'] > 0:
        adj = ensure_min_degree(adj, sim, CONFIG['min_degree'])

    print("归一化...")
    adj_norm = safe_normalize_adj(adj)

    out_path = os.path.join(CONFIG['output_dir'], 'A_functional.npy')
    np.save(out_path, adj_norm)
    print(f"✅ 保存至: {out_path}")
    print(f"   节点数: {adj.shape[0]}")
    print(f"   边数: {np.count_nonzero(adj) // 2}")

if __name__ == '__main__':
    main()
    '''
    ============================================================
🧠 构建功能相似图（基于保留网格）
============================================================
训练集形状: (412, 8712)
提取时间感知特征...
特征矩阵形状: (412, 150)
计算相似度并稀疏化...
选择Top-K邻居: 100%|██████████| 412/412 [00:00<00:00, 48057.55it/s]
归一化...
✅ 保存至: ../data/processed/split/A_functional.npy
   节点数: 412
   边数: 4249

进程已结束，退出代码为 0'''