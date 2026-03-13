"""
multi_graph_gcn_final.py
多图GCN模型（适配数据加载器已归一化的邻接矩阵）
输入: x (batch, nodes, window, 1), adj_s, adj_f, adj_p (已归一化的邻接矩阵)
输出: (batch, nodes, horizon)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GraphConvLayer(nn.Module):
    """图卷积层：直接使用传入的归一化邻接矩阵（不再重复归一化）"""

    def __init__(self, in_features: int, out_features: int, use_bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=use_bias)
        # 初始化
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
        if use_bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, T, C_in) 或 (B, N, C_in)
        adj: (N, N) 已归一化且含自环的邻接矩阵
        """
        if x.dim() == 4:
            B, N, T, C = x.shape
            # 重塑为 (B*T, N, C) 以进行批量矩阵乘法
            x_reshaped = x.permute(0, 2, 1, 3).reshape(-1, N, C)
            # 使用已归一化的 adj，无需再次归一化
            x_agg = torch.bmm(adj.expand(B * T, N, N), x_reshaped)
            x_out = self.linear(x_agg)
            # 恢复原始维度 (B, N, T, C_out)
            x_out = x_out.view(B, T, N, -1).permute(0, 2, 1, 3)
            return x_out
        else:
            # 3D输入： (B, N, C)
            return self.linear(torch.matmul(adj, x))


class SpatioTemporalBlock(nn.Module):
    """时空融合块：集成了多图卷积与时间卷积"""

    def __init__(self, in_channels: int, spatial_hidden: int, tcn_hidden: int,
                 temporal_kernel_size: int = 3, activation: nn.Module = nn.GELU):
        super().__init__()
        self.activation = activation()

        # 1. 多图卷积分支
        self.spatial_gc = GraphConvLayer(in_channels, spatial_hidden)
        self.functional_gc = GraphConvLayer(in_channels, spatial_hidden)
        self.poi_gc = GraphConvLayer(in_channels, spatial_hidden)

        # 2. 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(spatial_hidden * 3, tcn_hidden),
            nn.LayerNorm(tcn_hidden),
            self.activation
        )
        nn.init.kaiming_normal_(self.fusion[0].weight, nonlinearity='relu')
        nn.init.zeros_(self.fusion[0].bias)

        # 3. 时间卷积层（TCN）- 深度可分离卷积
        self.temporal_conv = nn.Conv1d(
            in_channels=tcn_hidden,
            out_channels=tcn_hidden,
            kernel_size=temporal_kernel_size,
            padding=temporal_kernel_size // 2,
            padding_mode='replicate',
            groups=tcn_hidden
        )
        self.temporal_norm = nn.LayerNorm(tcn_hidden)
        nn.init.kaiming_normal_(self.temporal_conv.weight, nonlinearity='relu')
        nn.init.zeros_(self.temporal_conv.bias)

        # 4. 残差投影
        self.residual_proj = (
            nn.Linear(in_channels, tcn_hidden)
            if in_channels != tcn_hidden
            else nn.Identity()
        )

    def _reshape_for_tcn(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """将 (B, N, T, C) 重塑为 (B*N, C, T) 供TCN使用"""
        original_shape = x.shape
        x_reshaped = x.permute(0, 1, 3, 2).flatten(0, 1)  # (B*N, C, T)
        return x_reshaped, original_shape

    def _reshape_from_tcn(self, x: torch.Tensor, original_shape: Tuple) -> torch.Tensor:
        """从 (B*N, C, T) 恢复为 (B, N, T, C)"""
        B, N, T, C = original_shape
        return x.view(B, N, C, T).permute(0, 1, 3, 2)

    def forward(self, x, adj_spatial, adj_functional, adj_poi):
        residual = x

        # --- 多图空间特征提取 ---
        s = self.activation(self.spatial_gc(x, adj_spatial))
        f = self.activation(self.functional_gc(x, adj_functional))
        p = self.activation(self.poi_gc(x, adj_poi))

        # --- 特征融合 ---
        combined = torch.cat([s, f, p], dim=-1)
        fused = self.fusion(combined)  # (B, N, T, tcn_hidden)

        # --- 时间卷积 ---
        tcn_input, orig_shape = self._reshape_for_tcn(fused)
        tcn_output = self.temporal_conv(tcn_input)
        tcn_output = self._reshape_from_tcn(tcn_output, orig_shape)

        tcn_output = self.temporal_norm(tcn_output)
        tcn_output = self.activation(tcn_output)

        # --- 残差连接 ---
        return tcn_output + self.residual_proj(residual)


class RobustMultiGraphGCN(nn.Module):
    """稳健版多图神经网络"""

    def __init__(self, window_size: int, horizon: int,
                 block_hidden: int = 64, num_blocks: int = 2,
                 use_simple_output: bool = True):
        super().__init__()
        self.window_size = window_size
        self.horizon = horizon

        # 输入投影
        self.input_proj = nn.Linear(1, block_hidden)
        nn.init.kaiming_normal_(self.input_proj.weight, nonlinearity='relu')
        nn.init.zeros_(self.input_proj.bias)

        # 时空块堆叠
        self.blocks = nn.ModuleList([
            SpatioTemporalBlock(
                in_channels=block_hidden,
                spatial_hidden=block_hidden,
                tcn_hidden=block_hidden
            ) for _ in range(num_blocks)
        ])

        # 输出头策略
        self.use_simple_output = use_simple_output
        if use_simple_output:
            # 使用最后一个时间步的特征
            self.output_layer = nn.Sequential(
                nn.Linear(block_hidden, block_hidden // 2),
                nn.GELU(),
                nn.Linear(block_hidden // 2, horizon)
            )
        else:
            # 使用所有时间步的特征
            self.output_layer = nn.Sequential(
                nn.Linear(block_hidden * window_size, block_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(block_hidden, horizon)
            )

    def forward(self, x, adj_s, adj_f, adj_p):
        # x: (B, N, T) 或 (B, N, T, 1)
        if x.dim() == 3:
            x = x.unsqueeze(-1)  # (B, N, T, 1)

        h = F.gelu(self.input_proj(x))  # (B, N, T, block_hidden)

        for block in self.blocks:
            h = block(h, adj_s, adj_f, adj_p)

        if self.use_simple_output:
            h_last = h[:, :, -1, :]  # (B, N, block_hidden)
            output = self.output_layer(h_last)
        else:
            B, N, T, C = h.shape
            h_flat = h.reshape(B, N, -1)  # (B, N, T*C)
            output = self.output_layer(h_flat)

        return output  # (B, N, horizon)