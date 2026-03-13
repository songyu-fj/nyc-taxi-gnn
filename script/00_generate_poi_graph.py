"""
00_generate_poi_graph.py
基于POI类别构建语义相似图（剔除稀疏网格后）
输入：data/processed/cleaned/poi_with_h3.csv, data/processed/split/retained_grids.npy
输出：data/processed/split/A_poi.npy
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer
import os

# ==================== 配置 ====================
CONFIG = {
    'poi_path': '../data/processed/poi_with_h3.csv',
    'retained_grids_path': '../data/processed/split/retained_grids.npy',
    'output_path': '../data/processed/split/A_poi.npy',
    'use_tfidf': True,
    'top_k_neighbors': 15,
    'original_n_nodes': 1351,
}

def sparsify_by_topk(similarity_matrix, k):
    n = similarity_matrix.shape[0]
    sparse = np.zeros_like(similarity_matrix)
    k = min(k, n-1)
    for i in range(n):
        topk_idx = np.argsort(similarity_matrix[i])[-k-1:][::-1]
        topk_idx = topk_idx[topk_idx != i][:k]
        sparse[i, topk_idx] = similarity_matrix[i, topk_idx]
        sparse[topk_idx, i] = similarity_matrix[topk_idx, i]
    return sparse

def safe_normalize_adj(adj):
    adj = adj + np.eye(adj.shape[0])
    deg = adj.sum(axis=1).clip(min=1e-6)
    deg_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    return deg_inv_sqrt @ adj @ deg_inv_sqrt

def main():
    print("="*60)
    print("📌 构建POI语义相似图（剔除稀疏网格后）")
    print("="*60)

    # 1. 读取保留的网格索引
    if not os.path.exists(CONFIG['retained_grids_path']):
        print(f"❌ 保留网格索引文件不存在: {CONFIG['retained_grids_path']}")
        return
    retained_indices = np.load(CONFIG['retained_grids_path'])  # 原始grid_idx数组
    print(f"保留网格数: {len(retained_indices)}")

    # 2. 加载POI数据
    if not os.path.exists(CONFIG['poi_path']):
        print(f"❌ POI文件不存在: {CONFIG['poi_path']}")
        return
    poi_df = pd.read_csv(CONFIG['poi_path'])
    print(f"POI记录数: {len(poi_df)}")
    print("可用列:", poi_df.columns.tolist())

    required_cols = ['grid_idx', 'FACILITY TYPE']
    for col in required_cols:
        if col not in poi_df.columns:
            print(f"❌ 缺少必要列: {col}")
            return

    # 3. 构建原始网格的POI计数矩阵（所有1351个网格）
    print("构建原始特征矩阵...")
    poi_features_all = pd.crosstab(poi_df['grid_idx'], poi_df['FACILITY TYPE'])
    all_grids = range(CONFIG['original_n_nodes'])
    poi_features_all = poi_features_all.reindex(all_grids, fill_value=0)
    feature_matrix_all = poi_features_all.values.astype(np.float32)

    # 4. 根据保留的网格索引筛选
    feature_matrix = feature_matrix_all[retained_indices, :]
    print(f"筛选后特征矩阵形状: {feature_matrix.shape}")

    if CONFIG['use_tfidf']:
        print("应用TF-IDF...")
        transformer = TfidfTransformer(norm='l2', use_idf=True)
        feature_matrix = transformer.fit_transform(feature_matrix).toarray()

    print("计算相似度矩阵...")
    sim_matrix = cosine_similarity(feature_matrix)
    np.fill_diagonal(sim_matrix, 0)

    print(f"稀疏化（Top-{CONFIG['top_k_neighbors']}）...")
    adj_sparse = sparsify_by_topk(sim_matrix, CONFIG['top_k_neighbors'])
    adj_sparse = np.maximum(adj_sparse, adj_sparse.T)

    print("归一化...")
    adj_norm = safe_normalize_adj(adj_sparse)

    os.makedirs(os.path.dirname(CONFIG['output_path']), exist_ok=True)
    np.save(CONFIG['output_path'], adj_norm)
    print(f"✅ 保存至: {CONFIG['output_path']}")
    print(f"   边数: {np.count_nonzero(adj_sparse) // 2}")

if __name__ == '__main__':
    main()