import pandas as pd
import geopandas as gpd
import h3
import glob
import os
import time

# ================= ⚙️ 配置区域 =================
# 1. 原始订单数据路径
INPUT_DIR = r"C:\workspace\NYC_fhvhv\data\processed\cleaned"

# 2. 输出路径 (包含整数 grid_idx)
OUTPUT_DIR = r"C:\workspace\NYC_fhvhv\data\mapped_integer"

# 3. 网格底图路径
GRID_PATH = "../data/map/nyc_h3_res8_indexed.gpkg"

# 4. H3 分辨率 (必须与生成网格时一致)
RESOLUTION = 8

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_h3_mapping_from_gpkg():
    """
    直接从 GPKG 文件中读取映射关系，不进行几何计算。
    假设：H3 字符串存储在 index 中，或者名为 'h3_index', 'hex_id' 的列中。
    """
    print(f"1. 读取网格文件: {GRID_PATH}")
    gdf = gpd.read_file(GRID_PATH)

    # 确保 grid_idx 存在
    if 'grid_idx' not in gdf.columns:
        raise ValueError("❌ 网格文件中找不到 'grid_idx' 列！")

    # --- 自动寻找 H3 字符串列 ---
    h3_col = None

    # 情况 A: Index 就是 H3 字符串 (最常见)
    # 检查索引的第一项是否是字符串且长度类似 H3 (15位左右)
    first_idx = gdf.index[0]
    if isinstance(first_idx, str) and len(first_idx) > 10:
        print("   ✅ 检测到 Index 为 H3 字符串，将使用 Index 作为映射键。")
        # 创建映射字典: { index_value : grid_idx_value }
        return gdf['grid_idx'].to_dict()

    # 情况 B: 在列里面找
    potential_cols = ['index', 'hex_id', 'h3_id', 'h3_code']
    for col in potential_cols:
        if col in gdf.columns:
            h3_col = col
            break

    if h3_col:
        print(f"   ✅ 检测到 H3 字符串列: '{h3_col}'")
        # 创建映射字典: { h3_col_value : grid_idx_value }
        return gdf.set_index(h3_col)['grid_idx'].to_dict()

    # 情况 C: 既然用户说有，但没自动找到，打印列名提示
    print("❌ 未自动找到 H3 字符串列。当前列名有:", list(gdf.columns))
    print("   (如果 H3 字符串在某一列，请修改脚本中的 h3_col 变量)")
    # 强制尝试使用 'index' 列 (如果 reset_index 过)
    if 'index' in gdf.columns:
        return gdf.set_index('index')['grid_idx'].to_dict()

    raise ValueError("无法定位 H3 字符串列，请检查网格文件结构。")


def map_orders_fast():
    # 1. 获取映射字典
    try:
        h3_map = get_h3_mapping_from_gpkg()
    except Exception as e:
        print(e)
        return

    print(f"   映射表构建完成！共 {len(h3_map)} 个网格。")
    print(f"   示例: {list(h3_map.keys())[0]} -> {list(h3_map.values())[0]}")

    # 2. 批量处理订单
    print("\n2. 开始处理订单文件...")
    files = glob.glob(os.path.join(INPUT_DIR, "*.parquet"))

    start_total = time.time()
    total_rows = 0
    total_matched = 0

    for i, f in enumerate(files):
        fname = os.path.basename(f)
        print(f"[{i + 1}/{len(files)}] 处理: {fname} ... ", end="")

        try:
            df = pd.read_parquet(f)
            original_len = len(df)

            # --- 核心加速步骤 ---
            # 1. 利用经纬度算出 H3 字符串
            # 使用列表推导式 (比 apply 快很多)
            h3_indices = [
                h3.latlng_to_cell(lat, lon, RESOLUTION)
                for lat, lon in zip(df['pickup_latitude'], df['pickup_longitude'])
            ]

            # 2. 查字典映射为整数 ID
            # map 操作利用哈希表查找，速度极快
            df['grid_idx'] = pd.Series(h3_indices).map(h3_map)

            # 3. 丢弃匹配不上的行 (NaN)
            # 这一步会自动过滤掉：坐标在网格范围外的、坐标错误的
            df_clean = df.dropna(subset=['grid_idx'])

            # 转为整数
            df_clean['grid_idx'] = df_clean['grid_idx'].astype(int)

            # 调整列顺序，把 ID 放第一列
            cols = ['grid_idx'] + [c for c in df_clean.columns if c != 'grid_idx']
            df_clean = df_clean[cols]

            # 保存
            out_path = os.path.join(OUTPUT_DIR, fname)
            df_clean.to_parquet(out_path, index=False)

            match_rate = (len(df_clean) / original_len) * 100
            print(f"完成! 匹配率: {match_rate:.1f}% ({len(df_clean)}/{original_len})")

            total_rows += original_len
            total_matched += len(df_clean)

        except Exception as e:
            print(f"失败! {e}")

    print("\n" + "=" * 30)
    print(f"🎉 处理完毕!")
    print(f"   总耗时: {time.time() - start_total:.2f} 秒")
    print(f"   总输入行数: {total_rows}")
    print(f"   总输出行数: {total_matched}")
    print(f"   被丢弃行数: {total_rows - total_matched} (不在网格范围内)")
    print("=" * 30)


if __name__ == "__main__":
    map_orders_fast()