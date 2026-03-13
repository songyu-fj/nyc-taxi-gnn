"""
02_build_spatial_adj.py
构建空间邻接图（基于保留的六边形网格）
输入：data/map/nyc_h3_res8_indexed.gpkg, data/processed/split/retained_grids.npy
输出：data/processed/split/A_spatial.npy
"""

import numpy as np
import geopandas as gpd
from scipy.spatial import KDTree
import os

# ==================== 配置 ====================
CONFIG = {
    'hex_grid_path': '../data/map/nyc_h3_res8_indexed.gpkg',
    'retained_grids_path': '../data/processed/split/retained_grids.npy',
    'output_path': '../data/processed/split/A_spatial.npy',
    'target_crs': 'EPSG:2263',
    'k_neighbors': 6,
}

def main():
    print("="*60)
    print("🗺️ 构建空间邻接图（基于保留网格）")
    print("="*60)

    # 1. 读取保留的网格索引
    if not os.path.exists(CONFIG['retained_grids_path']):
        print(f"❌ 保留网格索引文件不存在: {CONFIG['retained_grids_path']}")
        return
    retained_indices = np.load(CONFIG['retained_grids_path'])
    print(f"保留网格数: {len(retained_indices)}")

    # 2. 加载原始网格文件
    if not os.path.exists(CONFIG['hex_grid_path']):
        print(f"❌ 网格文件不存在: {CONFIG['hex_grid_path']}")
        return

    hex_gdf = gpd.read_file(CONFIG['hex_grid_path'])
    if hex_gdf.crs != CONFIG['target_crs']:
        print(f"转换坐标系: {hex_gdf.crs} -> {CONFIG['target_crs']}")
        hex_gdf = hex_gdf.to_crs(CONFIG['target_crs'])

    # 确保按 grid_idx 排序，并提取保留的行
    hex_gdf = hex_gdf.sort_values('grid_idx').reset_index(drop=True)
    hex_gdf_retained = hex_gdf.iloc[retained_indices].reset_index(drop=True)  # 新索引从0开始
    print(f"筛选后网格数量: {len(hex_gdf_retained)}")

    # 获取中心点坐标
    centroids = hex_gdf_retained.geometry.centroid
    coords = np.column_stack((centroids.x, centroids.y))

    # 构建KNN邻接矩阵
    n = len(coords)
    tree = KDTree(coords)
    dist, idx = tree.query(coords, k=CONFIG['k_neighbors']+1)

    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        neighbors = idx[i, 1:]  # 排除自身
        adj[i, neighbors] = 1.0
    adj = np.maximum(adj, adj.T)  # 对称化

    os.makedirs(os.path.dirname(CONFIG['output_path']), exist_ok=True)
    np.save(CONFIG['output_path'], adj)
    print(f"✅ 空间邻接矩阵已保存至: {CONFIG['output_path']}")
    print(f"   节点数: {n}")
    print(f"   边数: {np.count_nonzero(adj) // 2}")

if __name__ == '__main__':
    main()