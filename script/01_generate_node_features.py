"""
01_generate_node_features.py
聚合行程数据生成节点特征，剔除稀疏网格
输入：data/mapped_integer/ 下的黄车和绿车数据
输出：data/processed/split/X_train.npy, X_val.npy, X_test.npy, dataset_info.txt, retained_grids.npy
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import calendar

# ==================== 配置 ====================
CONFIG = {
    'yellow_dir': '../data/mapped_integer/',
    'green_dir': '../data/mapped_integer/',
    'yellow_pattern': '{:04d}-{:02d}.parquet',
    'green_pattern': 'g_{:04d}-{:02d}.parquet',
    'output_dir': '../data/processed/',
    'split_output_dir': '../data/processed/split/',
    'time_col': 'tpep_pickup_datetime',
    'grid_id_col': 'grid_idx',
    'time_slices_per_day': 72,
    'train_months': 4,
    'val_months': 1,
    'test_months': 1,
    'original_n_nodes': 1351,
    'sparsity_threshold': 5,  # 日均订单量阈值（低于此的网格剔除）
    'chunksize': 500000,
}

def process_month(year, month):
    month_days = calendar.monthrange(year, month)[1]
    total_slots = month_days * CONFIG['time_slices_per_day']
    month_data = np.zeros((CONFIG['original_n_nodes'], total_slots), dtype=np.uint32)

    yellow_file = os.path.join(CONFIG['yellow_dir'], CONFIG['yellow_pattern'].format(year, month))
    if os.path.exists(yellow_file):
        print(f"  读取黄车: {yellow_file}")
        month_data = aggregate_file(yellow_file, month_data, year, month, month_days)
    else:
        print(f"  黄车文件不存在: {yellow_file}")

    green_file = os.path.join(CONFIG['green_dir'], CONFIG['green_pattern'].format(year, month))
    if os.path.exists(green_file):
        print(f"  读取绿车: {green_file}")
        month_data = aggregate_file(green_file, month_data, year, month, month_days)
    else:
        print(f"  绿车文件不存在: {green_file}")

    return month_data

def aggregate_file(filepath, month_data, year, month, month_days):
    total_slots = month_days * CONFIG['time_slices_per_day']
    start_of_month = pd.Timestamp(year, month, 1)

    df = pd.read_parquet(filepath, columns=[CONFIG['time_col'], CONFIG['grid_id_col']])
    df = df.dropna(subset=[CONFIG['time_col'], CONFIG['grid_id_col']])
    df[CONFIG['time_col']] = pd.to_datetime(df[CONFIG['time_col']])

    delta = (df[CONFIG['time_col']] - start_of_month).dt.total_seconds() // 1200
    time_idx = delta.astype(int)

    mask = (time_idx >= 0) & (time_idx < total_slots)
    valid_grid = df.loc[mask, CONFIG['grid_id_col']].values.astype(int)
    valid_time = time_idx[mask].values

    np.add.at(month_data, (valid_grid, valid_time), 1)
    return month_data

def main():
    print("="*60)
    print("🚕 生成节点特征（剔除稀疏网格版）")
    print("="*60)

    os.makedirs(CONFIG['split_output_dir'], exist_ok=True)

    months = [(2016, m) for m in range(1, 7)]
    print(f"将处理 {len(months)} 个月份的数据")

    all_months_data = []
    month_lengths = []

    for year, month in tqdm(months, desc="处理月份"):
        print(f"\n📅 {year}-{month:02d}")
        month_matrix = process_month(year, month)
        if month_matrix is not None:
            all_months_data.append(month_matrix)
            month_days = calendar.monthrange(year, month)[1]
            month_lengths.append(month_days * CONFIG['time_slices_per_day'])
        else:
            print(f"  警告：{year}-{month:02d} 处理失败，跳过")

    if not all_months_data:
        print("❌ 没有成功处理任何数据")
        return

    print("\n合并全量数据...")
    full_matrix = np.hstack(all_months_data)
    print(f"全量矩阵形状: {full_matrix.shape}")
    print(f"总订单量: {full_matrix.sum():,}")

    # --- 剔除稀疏网格 ---
    print("\n计算每个网格的总订单量...")
    grid_total_orders = full_matrix.sum(axis=1)  # (1351,)
    total_days = sum(month_lengths) // CONFIG['time_slices_per_day']
    threshold = CONFIG['sparsity_threshold'] * total_days
    print(f"日均订单量阈值: {CONFIG['sparsity_threshold']} 单，对应总阈值: {threshold:.0f} 单")
    print(f"总天数: {total_days}")

    retained_mask = grid_total_orders >= threshold
    retained_indices = np.where(retained_mask)[0]  # 原始grid_idx
    print(f"保留网格数: {len(retained_indices)}，剔除网格数: {CONFIG['original_n_nodes'] - len(retained_indices)}")

    # 保存保留的原始索引
    np.save(os.path.join(CONFIG['split_output_dir'], 'retained_grids.npy'), retained_indices)

    # 剔除稀疏网格后的矩阵
    full_matrix_filtered = full_matrix[retained_mask, :]  # (新节点数, 总时间片)
    print(f"剔除后矩阵形状: {full_matrix_filtered.shape}")
    print(f"剔除后总订单量: {full_matrix_filtered.sum():,}")

    # --- 按月份切分 ---
    TRAIN = CONFIG['train_months']
    VAL = CONFIG['val_months']
    TEST = CONFIG['test_months']

    cum_lengths = np.cumsum([0] + month_lengths)
    train_end = cum_lengths[TRAIN]
    val_end = cum_lengths[TRAIN + VAL]
    total = cum_lengths[-1]

    print(f"\n切分点: 训练结束 {train_end}, 验证结束 {val_end}, 总 {total}")

    X_train = full_matrix_filtered[:, :train_end]
    X_val = full_matrix_filtered[:, train_end:val_end]
    X_test = full_matrix_filtered[:, val_end:]

    print(f"训练集: {X_train.shape}, 订单 {X_train.sum():,}")
    print(f"验证集: {X_val.shape}, 订单 {X_val.sum():,}")
    print(f"测试集: {X_test.shape}, 订单 {X_test.sum():,}")

    # 保存
    np.save(os.path.join(CONFIG['split_output_dir'], 'X_train.npy'), X_train)
    np.save(os.path.join(CONFIG['split_output_dir'], 'X_val.npy'), X_val)
    np.save(os.path.join(CONFIG['split_output_dir'], 'X_test.npy'), X_test)

    # 保存元数据
    with open(os.path.join(CONFIG['output_dir'], 'dataset_info.txt'), 'w') as f:
        f.write(f"原始节点数: {CONFIG['original_n_nodes']}\n")
        f.write(f"保留节点数: {len(retained_indices)}\n")
        f.write(f"剔除阈值: 日均 {CONFIG['sparsity_threshold']} 单\n")
        f.write(f"时间粒度: 20分钟\n")
        f.write(f"总月份: {len(months)}\n")
        f.write(f"训练月份数: {TRAIN}, 验证: {VAL}, 测试: {TEST}\n")
        f.write(f"总时间片: {total}\n")
        f.write(f"原始总订单: {full_matrix.sum():,}\n")
        f.write(f"剔除后总订单: {full_matrix_filtered.sum():,}\n")

    print("✅ 特征生成完成！")

if __name__ == '__main__':
    main()
'''
C:\AAAAPPS\anaconda3\envs\pytorch\python.exe C:\workspace\NYC_fhvhv\script_s\01_generate_node_features.py 
============================================================
🚕 生成节点特征（剔除稀疏网格版）
============================================================
将处理 6 个月份的数据

📅 2016-01
  读取黄车: ../data/mapped_integer/2016-01.parquet
处理月份:   0%|          | 0/6 [00:00<?, ?it/s]  读取绿车: ../data/mapped_integer/g_2016-01.parquet

📅 2016-02
  读取黄车: ../data/mapped_integer/2016-02.parquet
处理月份:  17%|█▋        | 1/6 [00:02<00:11,  2.28s/it]  读取绿车: ../data/mapped_integer/g_2016-02.parquet
处理月份:  33%|███▎      | 2/6 [00:06<00:12,  3.20s/it]
📅 2016-03
  读取黄车: ../data/mapped_integer/2016-03.parquet
  读取绿车: ../data/mapped_integer/g_2016-03.parquet
处理月份:  50%|█████     | 3/6 [00:09<00:09,  3.15s/it]
📅 2016-04
  读取黄车: ../data/mapped_integer/2016-04.parquet
  读取绿车: ../data/mapped_integer/g_2016-04.parquet

📅 2016-05
  读取黄车: ../data/mapped_integer/2016-05.parquet
处理月份:  67%|██████▋   | 4/6 [00:11<00:05,  2.96s/it]  读取绿车: ../data/mapped_integer/g_2016-05.parquet

📅 2016-06
  读取黄车: ../data/mapped_integer/2016-06.parquet
处理月份:  83%|████████▎ | 5/6 [00:14<00:02,  2.92s/it]  读取绿车: ../data/mapped_integer/g_2016-06.parquet

合并全量数据...
处理月份: 100%|██████████| 6/6 [00:17<00:00,  2.85s/it]
全量矩阵形状: (1351, 13104)
总订单量: 76,280,051

计算每个网格的总订单量...
日均订单量阈值: 5 单，对应总阈值: 910 单
总天数: 182
保留网格数: 412，剔除网格数: 939
剔除后矩阵形状: (412, 13104)
剔除后总订单量: 76,170,261

切分点: 训练结束 8712, 验证结束 10944, 总 13104
训练集: (412, 8712), 订单 50,970,298
验证集: (412, 2232), 订单 13,006,752
测试集: (412, 2160), 订单 12,193,211
✅ 特征生成完成！'''