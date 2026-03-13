import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import os
import glob

def standardize_taxi_columns(df):
    """
    将出租车数据列名统一为黄车格式（小写）
    支持绿车和黄车，绿车特有列名将被映射为黄车列名。
    """
    # 1. 所有列名转为小写
    df.columns = [col.lower().strip() for col in df.columns]

    # 2. 绿车 -> 黄车 列名映射（仅需处理绿车特有的列）
    column_mapping = {
        # 时间列
        'lpep_pickup_datetime': 'tpep_pickup_datetime',
        'lpep_dropoff_datetime': 'tpep_dropoff_datetime',
        # 坐标列（绿车原始可能为大写，但小写后已在上面处理，这里仅作确保）
        'pickup_longitude': 'pickup_longitude',   # 已一致，保留
        'pickup_latitude': 'pickup_latitude',
        'dropoff_longitude': 'dropoff_longitude',
        'dropoff_latitude': 'dropoff_latitude',
        # 距离列
        'trip_distance': 'trip_distance',         # 黄车已存在，保留
        # 绿车特有的距离列名（若原始为'Trip_distance'，小写后为'trip_distance'，已匹配）
        # 无需额外映射，因为绿车的'Trip_distance'小写后即'trip_distance'，与黄车一致
    }
    # 实际上绿车的'Trip_distance'小写后是'trip_distance'，与黄车列名相同，所以无需重命名
    # 但绿车的时间列小写后为'lpep_pickup_datetime'，需要映射为'tpep_pickup_datetime'
    # 所以仅需映射时间列
    column_mapping = {
        'lpep_pickup_datetime': 'tpep_pickup_datetime',
        'lpep_dropoff_datetime': 'tpep_dropoff_datetime'
    }
    # 注意：绿车中也可能存在大小写混合的其他列，但清洗只用上述列，无需处理其余列

    df = df.rename(columns=column_mapping)
    return df

def clean_taxi_data_parquet(input_file, output_file, zones_shapefile="taxi_zones.shp"):
    print(f"\n📂 正在加载Parquet数据: {input_file}")
    df = pd.read_parquet(input_file)
    original_count = len(df)
    print(f"✅ 原始数据: {original_count:,} 条记录")

    # ========== 列名标准化（统一为黄车格式） ==========
    print("\n0. 标准化列名...")
    df = standardize_taxi_columns(df)
    print("   列名已统一为黄车格式（小写）")

    # 1. 时间清洗
    print("\n1. 清洗时间...")
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds()

    time_mask = df['trip_duration'].between(60, 12 * 3600)
    df = df[time_mask]
    print(f"   移除时长异常: {original_count - len(df):,} 条")

    # 2. 坐标清洗 - 使用taxi_zones.shp过滤
    print("\n2. 加载并处理地理围栏...")
    zones = gpd.read_file(zones_shapefile)
    zone_gdf = zones.to_crs('EPSG:4326')

    # 创建上下车点几何
    pickup_points = gpd.points_from_xy(df['pickup_longitude'], df['pickup_latitude'])
    dropoff_points = gpd.points_from_xy(df['dropoff_longitude'], df['dropoff_latitude'])
    pickup_gdf = gpd.GeoDataFrame(
        geometry=pickup_points,
        index=df.index,
        crs=zone_gdf.crs
    )
    dropoff_gdf = gpd.GeoDataFrame(
        geometry=dropoff_points,
        index=df.index,
        crs=zone_gdf.crs
    )

    pickup_in_zone = gpd.sjoin(pickup_gdf, zone_gdf, how="inner", predicate="within")
    dropoff_in_zone = gpd.sjoin(dropoff_gdf, zone_gdf, how="inner", predicate="within")

    valid_pickup_indices = pickup_in_zone.index.unique()
    valid_dropoff_indices = dropoff_in_zone.index.unique()
    coord_mask = df.index.isin(valid_pickup_indices) & df.index.isin(valid_dropoff_indices)
    df = df[coord_mask]
    print(f"   移除坐标异常: {time_mask.sum() - len(df):,} 条")

    # 3. 速度清洗
    print("\n3. 清洗速度...")
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        return R * 2 * np.arcsin(np.sqrt(a))

    df['avg_speed_mph'] = df['trip_distance'] / (df['trip_duration'] / 3600)
    speed_mask = df['avg_speed_mph'].between(0.5, 80)
    df = df[speed_mask]
    print(f"   移除速度异常: {coord_mask.sum() - len(df):,} 条")

    # 最终结果
    final_count = len(df)
    print(f"\n🎯 清洗完成！")
    print(f"   保留记录: {final_count:,} 条 ({final_count / original_count * 100:.1f}%)")
    print(f"   移除记录: {original_count - final_count:,} 条")

    df.to_parquet(output_file, index=False)
    print(f"\n💾 已保存清洗后数据至: {output_file} (Parquet格式)")
    return df


if __name__ == "__main__":
    # 路径配置
    RAW_DIR = "../data/raw/trip_data"
    PROCESSED_DIR = "../data/processed/cleaned/"
    ZONES_FILE = "../data/raw/taxi_zones/taxi_zones.shp"

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    parquet_files = glob.glob(os.path.join(RAW_DIR, "*.parquet"))
    print(f"🔍 找到 {len(parquet_files)} 个 Parquet 文件待处理")

    success_count = 0
    fail_count = 0

    for file_path in parquet_files:
        file_name = os.path.basename(file_path)
        output_path = os.path.join(PROCESSED_DIR, file_name)

        try:
            print(f"\n{'='*60}")
            print(f"开始处理: {file_name}")
            clean_taxi_data_parquet(file_path, output_path, ZONES_FILE)
            success_count += 1
        except Exception as e:
            print(f"❌ 处理文件 {file_name} 时出错: {e}")
            fail_count += 1

    print(f"\n{'='*60}")
    print(f"批量处理完成！成功: {success_count} 个，失败: {fail_count} 个")
    '''
    C:\AAAAPPS\anaconda3\envs\pytorch\python.exe C:\workspace\NYC_fhvhv\preprocessing\统一列名以及清洗数据.py 
🔍 找到 12 个 Parquet 文件待处理

============================================================
开始处理: green_tripdata_2016-01.parquet

📂 正在加载Parquet数据: ../data/raw/trip_data\green_tripdata_2016-01.parquet
✅ 原始数据: 1,445,285 条记录

0. 标准化列名...
   列名已统一为黄车格式（小写）

1. 清洗时间...
   移除时长异常: 35,541 条

2. 加载并处理地理围栏...
   移除坐标异常: 6,191 条

3. 清洗速度...
   移除速度异常: 7,460 条

🎯 清洗完成！
   保留记录: 1,396,093 条 (96.6%)
   移除记录: 49,192 条

💾 已保存清洗后数据至: ../data/processed/cleaned/green_tripdata_2016-01.parquet (Parquet格式)

============================================================
开始处理: green_tripdata_2016-02.parquet

📂 正在加载Parquet数据: ../data/raw/trip_data\green_tripdata_2016-02.parquet
✅ 原始数据: 1,510,722 条记录

0. 标准化列名...
   列名已统一为黄车格式（小写）

1. 清洗时间...
   移除时长异常: 34,615 条

2. 加载并处理地理围栏...
   移除坐标异常: 6,488 条

3. 清洗速度...
   移除速度异常: 7,206 条

🎯 清洗完成！
   保留记录: 1,462,413 条 (96.8%)
   移除记录: 48,309 条

💾 已保存清洗后数据至: ../data/processed/cleaned/green_tripdata_2016-02.parquet (Parquet格式)

============================================================
开始处理: green_tripdata_2016-03.parquet

📂 正在加载Parquet数据: ../data/raw/trip_data\green_tripdata_2016-03.parquet
✅ 原始数据: 1,576,393 条记录

0. 标准化列名...
   列名已统一为黄车格式（小写）

1. 清洗时间...
   移除时长异常: 36,764 条

2. 加载并处理地理围栏...
   移除坐标异常: 6,408 条

3. 清洗速度...
   移除速度异常: 7,339 条

🎯 清洗完成！
   保留记录: 1,525,882 条 (96.8%)
   移除记录: 50,511 条

💾 已保存清洗后数据至: ../data/processed/cleaned/green_tripdata_2016-03.parquet (Parquet格式)

============================================================
开始处理: green_tripdata_2016-04.parquet

📂 正在加载Parquet数据: ../data/raw/trip_data\green_tripdata_2016-04.parquet
✅ 原始数据: 1,543,925 条记录

0. 标准化列名...
   列名已统一为黄车格式（小写）

1. 清洗时间...
   移除时长异常: 35,890 条

2. 加载并处理地理围栏...
   移除坐标异常: 6,156 条

3. 清洗速度...
   移除速度异常: 7,594 条

🎯 清洗完成！
   保留记录: 1,494,285 条 (96.8%)
   移除记录: 49,640 条

💾 已保存清洗后数据至: ../data/processed/cleaned/green_tripdata_2016-04.parquet (Parquet格式)

============================================================
开始处理: green_tripdata_2016-05.parquet

📂 正在加载Parquet数据: ../data/raw/trip_data\green_tripdata_2016-05.parquet
✅ 原始数据: 1,536,979 条记录

0. 标准化列名...
   列名已统一为黄车格式（小写）

1. 清洗时间...
   移除时长异常: 35,842 条

2. 加载并处理地理围栏...
   移除坐标异常: 6,207 条

3. 清洗速度...
   移除速度异常: 8,060 条

🎯 清洗完成！
   保留记录: 1,486,870 条 (96.7%)
   移除记录: 50,109 条

💾 已保存清洗后数据至: ../data/processed/cleaned/green_tripdata_2016-05.parquet (Parquet格式)

============================================================
开始处理: green_tripdata_2016-06.parquet

📂 正在加载Parquet数据: ../data/raw/trip_data\green_tripdata_2016-06.parquet
✅ 原始数据: 1,404,726 条记录

0. 标准化列名...
   列名已统一为黄车格式（小写）

1. 清洗时间...
   移除时长异常: 32,843 条

2. 加载并处理地理围栏...
   移除坐标异常: 5,483 条

3. 清洗速度...
   移除速度异常: 8,176 条

🎯 清洗完成！
   保留记录: 1,358,224 条 (96.7%)
   移除记录: 46,502 条

💾 已保存清洗后数据至: ../data/processed/cleaned/green_tripdata_2016-06.parquet (Parquet格式)

============================================================
开始处理: yellow_tripdata_2016-01.parquet

📂 正在加载Parquet数据: ../data/raw/trip_data\yellow_tripdata_2016-01.parquet
✅ 原始数据: 10,906,858 条记录

0. 标准化列名...
   列名已统一为黄车格式（小写）

1. 清洗时间...
   移除时长异常: 103,292 条

2. 加载并处理地理围栏...
   移除坐标异常: 186,414 条

3. 清洗速度...
   移除速度异常: 12,550 条

🎯 清洗完成！
   保留记录: 10,604,602 条 (97.2%)
   移除记录: 302,256 条

💾 已保存清洗后数据至: ../data/processed/cleaned/yellow_tripdata_2016-01.parquet (Parquet格式)

============================================================
开始处理: yellow_tripdata_2016-02.parquet

📂 正在加载Parquet数据: ../data/raw/trip_data\yellow_tripdata_2016-02.parquet
✅ 原始数据: 11,382,049 条记录

0. 标准化列名...
   列名已统一为黄车格式（小写）

1. 清洗时间...
   移除时长异常: 106,460 条

2. 加载并处理地理围栏...
   移除坐标异常: 193,242 条

3. 清洗速度...
   移除速度异常: 14,247 条

🎯 清洗完成！
   保留记录: 11,068,100 条 (97.2%)
   移除记录: 313,949 条

💾 已保存清洗后数据至: ../data/processed/cleaned/yellow_tripdata_2016-02.parquet (Parquet格式)

============================================================
开始处理: yellow_tripdata_2016-03.parquet

📂 正在加载Parquet数据: ../data/raw/trip_data\yellow_tripdata_2016-03.parquet
✅ 原始数据: 12,210,952 条记录

0. 标准化列名...
   列名已统一为黄车格式（小写）

1. 清洗时间...
   移除时长异常: 116,360 条

2. 加载并处理地理围栏...
   移除坐标异常: 199,986 条

3. 清洗速度...
   移除速度异常: 13,999 条

🎯 清洗完成！
   保留记录: 11,880,607 条 (97.3%)
   移除记录: 330,345 条

💾 已保存清洗后数据至: ../data/processed/cleaned/yellow_tripdata_2016-03.parquet (Parquet格式)

============================================================
开始处理: yellow_tripdata_2016-04.parquet

📂 正在加载Parquet数据: ../data/raw/trip_data\yellow_tripdata_2016-04.parquet
✅ 原始数据: 11,934,338 条记录

0. 标准化列名...
   列名已统一为黄车格式（小写）

1. 清洗时间...
   移除时长异常: 120,552 条

2. 加载并处理地理围栏...
   移除坐标异常: 187,908 条

3. 清洗速度...
   移除速度异常: 14,516 条

🎯 清洗完成！
   保留记录: 11,611,362 条 (97.3%)
   移除记录: 322,976 条

💾 已保存清洗后数据至: ../data/processed/cleaned/yellow_tripdata_2016-04.parquet (Parquet格式)

============================================================
开始处理: yellow_tripdata_2016-05.parquet

📂 正在加载Parquet数据: ../data/raw/trip_data\yellow_tripdata_2016-05.parquet
✅ 原始数据: 11,836,853 条记录

0. 标准化列名...
   列名已统一为黄车格式（小写）

1. 清洗时间...
   移除时长异常: 114,599 条

2. 加载并处理地理围栏...
   移除坐标异常: 167,668 条

3. 清洗速度...
   移除速度异常: 15,818 条

🎯 清洗完成！
   保留记录: 11,538,768 条 (97.5%)
   移除记录: 298,085 条

💾 已保存清洗后数据至: ../data/processed/cleaned/yellow_tripdata_2016-05.parquet (Parquet格式)

============================================================
开始处理: yellow_tripdata_2016-06.parquet

📂 正在加载Parquet数据: ../data/raw/trip_data\yellow_tripdata_2016-06.parquet
✅ 原始数据: 11,135,470 条记录

0. 标准化列名...
   列名已统一为黄车格式（小写）

1. 清洗时间...
   移除时长异常: 109,478 条

2. 加载并处理地理围栏...
   移除坐标异常: 157,581 条

3. 清洗速度...
   移除速度异常: 15,557 条

🎯 清洗完成！
   保留记录: 10,852,854 条 (97.5%)
   移除记录: 282,616 条

💾 已保存清洗后数据至: ../data/processed/cleaned/yellow_tripdata_2016-06.parquet (Parquet格式)

============================================================
批量处理完成！成功: 12 个，失败: 0 个

进程已结束，退出代码为 0
'''