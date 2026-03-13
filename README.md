[README.md](https://github.com/user-attachments/files/25958222/README.md)
# NYC Taxi GNN Demo

基于多图神经网络和纽约出租车数据（黄车+绿车）的区域流量预测系统。该项目利用多图神经网络融合空间邻接、POI语义和功能相似性，对纽约市各区域的未来打车需求进行预测。

## 数据说明

- **原始数据**：2016年1月-6月 NYC TLC 黄车和绿车行程数据，包含上车时间、地点等信息。
- **处理流程**：
  - 将行程映射到 H3 分辨率8的六边形网格（共1351个网格）。
  - 筛选日均订单 ≥5 的活跃网格（最终保留412个网格）。
  - 按20分钟时间片聚合订单量，得到节点特征矩阵 `X_*.npy`。
  - 构建三种图：
    - 空间邻接图（基于KNN，6个最近邻居）
    - POI语义图（基于网格内POI类别的TF-IDF+余弦相似度）
    - 功能相似图（基于时间序列的日模式、高峰特征等）

## 环境配置
1. 创建环境：`conda create -n nyc_gnn python=3.9`
2. 激活环境：`conda activate nyc_gnn`
3. 安装依赖：`pip install -r requirements.txt`

## 文件说明
- `muti_graph_gcn_final`:模型定义文件
- `run_all_analyze`:分析POI数据，出租车数据集
-  `cleandata_final`:清洗所有数据
- `00_generate_poi_graph`:生成POI语义图
- `01_generate_node_features`:生成节点特征矩阵
- `02_build_spatial_adj`:构建空间邻接图
- `04_build_funtional_adjacency`:构建功能相似图
- `create_dataloader_final`:数据加载器
- `train_final`:训练脚本

## 快速开始

### 1. 数据预处理（按顺序运行）

确保原始数据已放置在正确路径（可根据脚本内配置调整路径）：

bash

```
python cleandata_final.py
python run_all_analyze.py
python 00_generate_poi_graph.py
python 01_generate_node_features.py
python 02_build_spatial_adj.py
python 04_build_functional_adjacency.py
```



### 2. 训练模型

bash

```
python train_final.py
```



训练完成后，最佳模型保存在 `results/paper_experiment/best_model.pth`，测试集指标将打印在控制台。

### 3. 使用预训练模型预测

运行预测示例（需准备输入数据）：

bash

```
python predict.py --sample data/example_input.csv
```



## 依赖库

主要依赖见 `requirements.txt`，包括：

- torch
- numpy
- pandas
- geopandas
- scikit-learn
- scipy
- matplotlib
- tqdm
- pyarrow (用于读取parquet)
