import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import glob
import os
import numpy as np

# ================= ⚙️ 全局配置 =================
# 数据路径（请根据你的实际情况修改）
INPUT_DIR_CLEANED = r"C:\workspace\NYC_fhvhv\data\processed\cleaned"
INPUT_DIR_MAPPED = r"C:\workspace\NYC_fhvhv\data\mapped_integer"
POI_FILE = r"../data/processed/POIcleaned.csv"
GRID_PATH = r"../data/map/nyc_h3_res8_indexed.gpkg"
OUTPUT_DIR = r"../output"

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 字体设置（支持中文）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid", {'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']})

# ================= 分析函数 1：星期/小时需求 =================
def analyze_weekday_hour():
    print("\n=== 1. 星期/小时需求分析 ===")
    files = glob.glob(os.path.join(INPUT_DIR_CLEANED, "*.parquet"))
    if not files:
        print("⚠️ 未找到 cleaned 数据，跳过")
        return

    daily_hourly_counts = []
    for f in files:
        try:
            df = pd.read_parquet(f, columns=['tpep_pickup_datetime'])
            df.columns = ['ts']
            df['ts'] = pd.to_datetime(df['ts'])
            df = df[(df['ts'] >= '2016-01-01') & (df['ts'] <= '2016-06-30')]
            if df.empty: continue
            df['date'] = df['ts'].dt.date
            df['hour'] = df['ts'].dt.hour
            counts = df.groupby(['date', 'hour']).size().reset_index(name='count')
            daily_hourly_counts.append(counts)
        except Exception as e:
            print(f"⚠️ 跳过文件 {os.path.basename(f)}: {e}")

    if not daily_hourly_counts:
        print("❌ 无有效数据")
        return

    full_df = pd.concat(daily_hourly_counts)
    full_df['date'] = pd.to_datetime(full_df['date'])
    full_df['weekday'] = full_df['date'].dt.dayofweek

    # 绘图1：24小时变化图
    hourly_trend = full_df.groupby(['weekday', 'hour'])['count'].mean().reset_index()
    plt.figure(figsize=(14, 8))
    weekday_map = {0: '周一', 1: '周二', 2: '周三', 3: '周四', 4: '周五', 5: '周六', 6: '周日'}
    palette = sns.color_palette("husl", 7)
    for wd in range(7):
        subset = hourly_trend[hourly_trend['weekday'] == wd]
        linestyle = '--' if wd >= 5 else '-'
        linewidth = 2.5 if wd >= 5 else 1.5
        plt.plot(subset['hour'], subset['count'],
                 label=weekday_map[wd],
                 color=palette[wd],
                 linestyle=linestyle,
                 linewidth=linewidth,
                 marker='o', markersize=4)
    plt.title('周一至周日 24小时网约车平均需求量变化', fontsize=16, fontweight='bold')
    plt.xlabel('时间 (小时)'); plt.ylabel('平均每小时订单量')
    plt.xticks(range(0, 24)); plt.grid(True, alpha=0.3); plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/hourly_demand_by_weekday.png", dpi=300, bbox_inches='tight')
    print("  ✅ 已保存 hourly_demand_by_weekday.png")

    # 绘图2：星期平均日需求柱状图
    daily_sum = full_df.groupby(['date', 'weekday'])['count'].sum().reset_index()
    weekly_avg = daily_sum.groupby('weekday')['count'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    bars = plt.bar(weekly_avg['weekday'], weekly_avg['count'],
                   color=sns.color_palette("Paired", 7), alpha=0.8, edgecolor='black')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.title('周一至周日 平均全天出行需求分布', fontsize=16, fontweight='bold')
    plt.xlabel('星期'); plt.ylabel('平均日订单量')
    plt.xticks(range(7), [weekday_map[i] for i in range(7)])
    plt.ylim(0, weekly_avg['count'].max() * 1.15)
    plt.savefig(f"{OUTPUT_DIR}/average_daily_demand_by_weekday.png", dpi=300, bbox_inches='tight')
    print("  ✅ 已保存 average_daily_demand_by_weekday.png")

# ================= 分析函数 2：空间出行需求 =================
def analyze_spatial_maps():
    print("\n=== 2. 空间出行需求分析 ===")
    # 读取网格底图
    if not os.path.exists(GRID_PATH):
        print("⚠️ 网格文件不存在，跳过")
        return
    gdf_grid = gpd.read_file(GRID_PATH)
    if 'grid_idx' in gdf_grid.columns:
        gdf_grid['grid_idx'] = gdf_grid['grid_idx'].astype(int)
        gdf_grid = gdf_grid.set_index('grid_idx')
    print(f"  网格加载完成，共 {len(gdf_grid)} 个格子")

    files = glob.glob(os.path.join(INPUT_DIR_MAPPED, "*.parquet"))
    if not files:
        print("⚠️ 未找到 mapped_integer 数据，跳过")
        return

    agg_list = []
    unique_dates = set()
    for f in files:
        try:
            df = pd.read_parquet(f, columns=['tpep_pickup_datetime', 'grid_idx'])
            df['ts'] = pd.to_datetime(df['tpep_pickup_datetime'])
            df = df[(df['ts'] >= '2016-01-01') & (df['ts'] <= '2016-06-30')]
            if df.empty: continue
            unique_dates.update(df['ts'].dt.date.astype(str).unique())
            df['day_type'] = np.where(df['ts'].dt.dayofweek >= 5, 'Weekend', 'Weekday')
            df['grid_idx'] = df['grid_idx'].astype(int)
            counts = df.groupby(['grid_idx', 'day_type']).size().reset_index(name='trip_count')
            agg_list.append(counts)
        except Exception as e:
            print(f"⚠️ 跳过 {os.path.basename(f)}: {e}")

    if not agg_list:
        print("❌ 无有效数据")
        return

    full_df = pd.concat(agg_list)
    total_counts = full_df.groupby(['grid_idx', 'day_type'])['trip_count'].sum().reset_index()

    # 计算天数
    dates_df = pd.DataFrame({'date': list(unique_dates)})
    dates_df['date'] = pd.to_datetime(dates_df['date'])
    dates_df['day_type'] = np.where(dates_df['date'].dt.dayofweek >= 5, 'Weekend', 'Weekday')
    days_count = dates_df['day_type'].value_counts()
    n_weekdays = days_count.get('Weekday', 1)
    n_weekends = days_count.get('Weekend', 1)

    def calculate_avg(row):
        days = n_weekdays if row['day_type'] == 'Weekday' else n_weekends
        return row['trip_count'] / days
    total_counts['avg_daily_trips'] = total_counts.apply(calculate_avg, axis=1)

    # 关联地理信息
    df_weekday = total_counts[total_counts['day_type'] == 'Weekday'].set_index('grid_idx')
    df_weekend = total_counts[total_counts['day_type'] == 'Weekend'].set_index('grid_idx')
    gdf_weekday = gdf_grid.join(df_weekday, how='inner')
    gdf_weekend = gdf_grid.join(df_weekend, how='inner')

    if len(gdf_weekday) == 0:
        print("❌ 关联后无数据")
        return

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    max_val = max(gdf_weekday['avg_daily_trips'].quantile(0.98), gdf_weekend['avg_daily_trips'].quantile(0.98))
    plot_kwds = {
        'column': 'avg_daily_trips',
        'cmap': 'inferno',
        'vmin': 0,
        'vmax': max_val,
        'legend': True,
        'legend_kwds': {'label': "平均日订单量 (Trips/Day)", 'shrink': 0.6}
    }
    gdf_weekday.plot(ax=axes[0], **plot_kwds)
    axes[0].set_title(f'工作日 (Weekday) 平均出行需求\n(基于 {n_weekdays} 天)', fontsize=16)
    axes[0].axis('off')
    gdf_weekend.plot(ax=axes[1], **plot_kwds)
    axes[1].set_title(f'周末 (Weekend) 平均出行需求\n(基于 {n_weekends} 天)', fontsize=16)
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/spatial_demand_distribution.png", dpi=300)
    print("  ✅ 已保存 spatial_demand_distribution.png")

# ================= 分析函数 3：POI 分析 =================
def analyze_poi():
    print("\n=== 3. POI 分析 ===")
    if not os.path.exists(POI_FILE):
        print("⚠️ POI 文件不存在，跳过")
        return

    # 映射字典（硬编码）
    type_map = {
        1: '居住 (Residential)', 2: '教育 (Education)', 3: '文化 (Cultural)',
        4: '休闲 (Recreational)', 5: '社福 (Social Services)', 6: '交通 (Transportation)',
        7: '商业 (Commercial)', 8: '政府 (Government)', 9: '宗教 (Religious)',
        10: '医疗 (Health)', 11: '公官 (Public Safety)', 12: '水域 (Water)',
        13: '杂项 (Misc)', 14: '景点 (Attraction)'
    }
    dom_map = {
        1: {1: '封闭社区', 2: '私人开发', 3: '公共住房', 4: '构成部分', 5: '其他'},
        2: {1: '公立小学', 2: '公立初中', 3: '公立高中', 4: '私立小学', 5: '私立初中', 6: '私立高中',
            7: '大学', 8: '其他', 9: '公立幼教', 10: '公立K-8', 11: '公立K-12', 12: '公立中学',
            13: '教学楼', 14: '附属楼', 15: '私立幼教', 16: '私立K-8', 17: '私立K-12', 18: '私立中学'},
        3: {1: '中心', 2: '图书馆', 3: '剧院', 4: '博物馆', 5: '其他'},
        4: {1: '公园', 2: '游乐园', 3: '高尔夫', 4: '海滩', 5: '花园', 6: '动物园', 7: '中心',
            8: '体育', 9: '游乐场', 10: '其他', 11: '泳池', 12: '花园'},
        5: {1: '育儿', 2: '日托', 3: '成人日托', 4: '养老院', 5: '收容所', 6: '其他'},
        6: {1: '巴士站', 2: '渡轮', 3: '车场', 4: '机场', 5: '直升机', 6: '码头', 7: '栈桥',
            8: '桥梁', 9: '隧道', 10: '出入口', 11: '航运', 12: '其他'},
        7: {1: '中心', 2: '商业', 3: '市场', 4: '酒店', 5: '餐厅', 6: '其他'},
        8: {1: '办公', 2: '法院', 3: '邮局', 4: '领馆', 5: '大使馆', 6: '军事', 7: '其他'},
        9: {1: '教堂', 2: '犹太', 3: '寺庙', 4: '修道院', 5: '清真寺', 6: '其他'},
        10: {1: '医院', 2: '住院', 3: '诊所', 4: '其他'},
        11: {1: '警局', 2: '检查站', 3: '消防云梯', 4: '消防营', 5: '监狱', 6: '消防泵',
             7: '特种', 8: '分部', 9: '小队', 10: '其他警务', 11: '其他', 12: '其他消防'},
        12: {1: '岛', 2: '河', 3: '湖', 4: '溪', 5: '其他', 6: '池塘'},
        13: {1: '地标', 2: 'POI', 3: '墓地', 4: '其他'},
        14: {1: '地标', 2: '小景点'}
    }

    df = pd.read_csv(POI_FILE)
    df['Category'] = df['FACILITY TYPE'].map(type_map).fillna('未知大类')
    df['SubCategory'] = df.apply(lambda row: dom_map.get(row['FACILITY TYPE'], {}).get(row['FACILITY DOMAINS'], '未知小类'), axis=1)

    # 一级分类统计
    stats_cat = df['Category'].value_counts().reset_index()
    stats_cat.columns = ['一级分类', '数量']
    stats_cat['占比%'] = (stats_cat['数量'] / len(df) * 100).round(2)
    stats_cat.to_csv(f"{OUTPUT_DIR}/poi_一级分类统计.csv", index=False, encoding='utf-8-sig')

    # 详细分类统计
    stats_detail = df.groupby(['Category', 'SubCategory']).size().reset_index(name='数量')
    stats_detail = stats_detail.sort_values(['Category', '数量'], ascending=[True, False])
    stats_detail.to_csv(f"{OUTPUT_DIR}/poi_详细分类统计.csv", index=False, encoding='utf-8-sig')

    # 绘图：一级分类饼图
    plt.figure(figsize=(10, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(stats_cat)))
    plt.pie(stats_cat['数量'], labels=stats_cat['一级分类'], autopct='%1.1f%%',
            startangle=140, colors=colors, pctdistance=0.85)
    plt.title('POI 一级分类占比')
    plt.savefig(f"{OUTPUT_DIR}/poi_一级分类饼图.png", bbox_inches='tight', dpi=300)
    plt.close()

    # 条形图
    plt.figure(figsize=(12, 6))
    sns.barplot(data=stats_cat, x='数量', y='一级分类', palette='viridis')
    for i, v in enumerate(stats_cat['数量']):
        plt.text(v, i, f" {v}", va='center')
    plt.title('POI 一级分类数量统计')
    plt.savefig(f"{OUTPUT_DIR}/poi_一级分类条形图.png", bbox_inches='tight', dpi=300)
    plt.close()

    # 二级分类细分图（Grid Pie Charts）
    cats = df['Category'].unique()
    rows = (len(cats) // 4) + 1
    fig, axes = plt.subplots(rows, 4, figsize=(20, 5 * rows))
    axes = axes.flatten()
    for i, cat in enumerate(cats):
        ax = axes[i]
        sub_data = df[df['Category'] == cat]['SubCategory'].value_counts()
        if len(sub_data) > 0:
            ax.pie(sub_data, labels=sub_data.index, autopct='%1.0f%%', startangle=90,
                   textprops={'fontsize': 8})
            ax.set_title(cat, fontsize=12, fontweight='bold')
        else:
            ax.axis('off')
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/poi_二级分类细分图.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("  ✅ POI分析完成，图表已保存")

# ================= 分析函数 4：时间序列趋势 =================
def analyze_temporal_trend():
    print("\n=== 4. 2016上半年日趋势分析 ===")
    files = glob.glob(os.path.join(INPUT_DIR_MAPPED, "*.parquet"))
    if not files:
        print("⚠️ 未找到 mapped_integer 数据，跳过")
        return

    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f, columns=['tpep_pickup_datetime'])
            df.columns = ['ts']
            dfs.append(df)
        except:
            continue
    if not dfs:
        print("❌ 无有效数据")
        return

    full_df = pd.concat(dfs)
    full_df['ts'] = pd.to_datetime(full_df['ts'])
    daily = full_df['ts'].dt.floor('D').value_counts().sort_index()
    daily = daily['2016-01-01':'2016-06-30']

    if len(daily) == 0:
        print("❌ 无2016上半年数据")
        return

    plt.figure(figsize=(14, 6))
    plt.plot(daily.index, daily.values, color='#1f77b4', alpha=0.6, linewidth=1.5, label='每日订单量')
    rolling_7d = daily.rolling(window=7).mean()
    plt.plot(rolling_7d.index, rolling_7d.values, color='#d62728', linewidth=2.5, label='7日趋势线 (移动平均)')
    plt.title('2016年上半年 纽约出租车日需求量趋势 (1月-6月)', fontsize=16, fontweight='bold')
    plt.ylabel('订单量'); plt.xlabel('日期')
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/demand_trend_2016_H1.png", dpi=300)
    daily.to_csv(f"{OUTPUT_DIR}/daily_counts_2016_H1.csv", header=['count'])
    print("  ✅ 已保存 demand_trend_2016_H1.png 和 daily_counts_2016_H1.csv")

# ================= 主程序 =================
if __name__ == "__main__":
    print("开始执行所有数据分析，输出目录:", OUTPUT_DIR)
    analyze_weekday_hour()
    analyze_spatial_maps()
    analyze_poi()
    analyze_temporal_trend()
    print("\n所有分析完成！")