import os
import sys
import copy
import math
import numpy as np

import torch 
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch_geometric.nn import GCNConv,NNConv
from torch_geometric.data.batch import Batch
from torch_geometric.data.batch import Data as GraphData
import blocks
from einops import rearrange
# from mamba_ssm import Mamba
import numbers


class RobustCrossModalFusion(nn.Module):
    def __init__(self, dim=128, num_heads=4, expansion_ratio=4, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, "特征维度必须能被注意力头数整除"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # 交叉注意力投影
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        # 动态门控系统
        self.gate_gen = nn.Sequential(
            nn.Linear(2 * dim, dim // 2),
            nn.LeakyReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )

        # 特征增强模块
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, expansion_ratio * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion_ratio * dim, dim),
            nn.Dropout(dropout)
        )

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, f1, f2):
        """输入输出形状均为 (batch_size, feature_dim)"""
        B = f1.size(0)  # 动态获取batch_size

        # ---- 交叉注意力阶段 ----
        # 投影并拆分多头 [B, D] -> [B, H, hD]
        q = self.query(f1).view(B, self.num_heads, self.head_dim)
        k = self.key(f2).view(B, self.num_heads, self.head_dim)
        v = self.value(f2).view(B, self.num_heads, self.head_dim)

        # 计算注意力权重
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)

        # 注意力加权聚合
        attended = (attn @ v).view(B, self.dim)  # [B, D]

        # ---- 动态门控融合 ----
        gate = self.gate_gen(torch.cat([f1, f2], dim=1))  # [B, 1]
        fused = f1 + self.dropout(gate * attended)  # 残差连接
        # fused=f1+f2
        # ---- 特征增强 ----
        fused = self.norm(fused)
        return self.ffn(fused)
class SingleModalAttention(nn.Module):
    def __init__(self, dim=128, num_heads=4, expansion_ratio=4, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, "特征维度必须能被注意力头数整除"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # 自注意力投影层（qkv均来自同一特征）
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        # 动态门控系统（输入改为原始特征+注意力后特征）
        self.gate_gen = nn.Sequential(
            nn.Linear(2 * dim, dim // 2),
            nn.LeakyReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )

        # 特征增强模块（保持不变）
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, expansion_ratio * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion_ratio * dim, dim),
            nn.Dropout(dropout)
        )

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, f):
        """输入输出形状均为 (batch_size, feature_dim)"""
        B = f.size(0)

        # ---- 自注意力阶段 ----
        # 投影并拆分多头 [B, D] -> [B, H, hD]
        q = self.query(f).view(B, self.num_heads, self.head_dim)
        k = self.key(f).view(B, self.num_heads, self.head_dim)
        v = self.value(f).view(B, self.num_heads, self.head_dim)

        # 计算注意力权重
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)

        # 注意力加权聚合
        attended = (attn @ v).view(B, self.dim)  # [B, D]

        # ---- 动态门控融合 ----
        # 拼接原始特征和注意力后特征
        gate = self.gate_gen(torch.cat([f, attended], dim=1))  # [B, 1]
        fused = f + self.dropout(gate * attended)  # 残差连接

        # ---- 特征增强 ----
        fused = self.norm(fused)
        return self.ffn(fused)
class NodeProjection(nn.Module):
    """独立处理每个节点的维度不一致问题"""

    def __init__(self, input_dims, unified_dim=128):
        super().__init__()
        # 为每个节点创建独立的全连接层
        self.proj_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 256),
                nn.ReLU(),
                nn.Linear(256, unified_dim)
            ) for dim in input_dims
        ])

    def forward(self, features):
        # features: list of 6 tensors with shapes (32, dim)
        projected = []
        for f, layer in zip(features, self.proj_layers):
            projected.append(layer(f))
        return torch.stack(projected, dim=1)  # (32,6,64)


class AdaptiveGCNLayer(nn.Module):
    """自适应图卷积层"""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.node_transform = nn.Linear(in_dim, out_dim)
        self.relation_learner = nn.Parameter(torch.randn(6, 6))

    def forward(self, H):
        # H shape: (batch,6,in_dim)
        batch_size = H.size(0)

        # 动态生成带残差的邻接矩阵
        adj = F.softmax(self.relation_learner, dim=-1)
        identity_adj = torch.eye(6, device=H.device)
        combined_adj = adj + identity_adj  # 残差连接

        # 节点特征变换
        transformed = self.node_transform(H)  # (32,6,out_dim)

        # 信息传递
        aggregated = torch.einsum('ij,bjk->bik', combined_adj, transformed)
        return F.leaky_relu(aggregated + transformed)  # 残差连接


class GatedAttentionPool(nn.Module):
    """门控注意力池化"""

    def __init__(self, feat_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.Sigmoid()
        )
        self.attn = nn.Linear(feat_dim, 1)

    def forward(self, H):
        # H shape: (batch,6,feat_dim)
        gate_values = self.gate(H.mean(1, keepdim=True))  # (batch,1,feat_dim)
        gated_H = H * gate_values

        attn_scores = F.softmax(self.attn(gated_H), dim=1)
        return torch.sum(attn_scores * gated_H, dim=1)  # (batch,feat_dim)

###################################For tactile#######################################
class TactileGCN(nn.Module):
    def __init__(self,input_dims=[330,330,340,320,380,223]):
        super().__init__()
        self.projection = NodeProjection(input_dims)

        # 图卷积模块
        self.gcn1 = AdaptiveGCNLayer(128, 128)
        self.gcn2 = AdaptiveGCNLayer(128, 128)
        # self.pooling = nn.Sequential(
        #             nn.Linear(128, 128),
        #             nn.ReLU()
        #         )
        # 池化模块
        self.pooling = GatedAttentionPool(128)

        # self.pose_linear = nn.Linear(128, 7)

    def forward(self,x_list):

        x = self.projection(x_list)  # (32,6,64)

        # 图卷积处理
        x = self.gcn1(x)
        x = self.gcn2(x)
        # x = x.mean(dim=1)
        x = self.pooling(x)
        # x = self.pose_linear(x)
        # 池化输出
        return x
# class NodeProjection(nn.Module):
#     """统一不同节点的特征维度（保持原实现）"""
#
#     def __init__(self, input_dims, unified_dim=128):
#         super().__init__()
#         self.proj_layers = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(dim, 256),
#                 nn.ReLU(),
#                 nn.Linear(256, unified_dim)
#             ) for dim in input_dims
#         ])
#
#     def forward(self, features):
#         projected = []
#         for f, layer in zip(features, self.proj_layers):
#             projected.append(layer(f))
#         return torch.stack(projected, dim=1)  # (batch, 6, 128)
#
#
# class GCNLayer(nn.Module):
#     """普通GCN层（使用固定邻接矩阵）"""
#
#     def __init__(self, in_dim, out_dim, num_nodes=6):
#         super().__init__()
#         self.linear = nn.Linear(in_dim, out_dim)
#
#         # 创建归一化邻接矩阵（应在init中直接注册）
#         adj = torch.ones(num_nodes, num_nodes) + torch.eye(num_nodes)
#         rowsum = adj.sum(1)
#         d_inv_sqrt = torch.pow(rowsum, -0.5)
#         norm_adj = torch.diag(d_inv_sqrt) @ adj @ torch.diag(d_inv_sqrt)  # 临时变量
#
#         # 直接注册buffer（不需要赋值给self）
#         self.register_buffer('norm_adj', norm_adj)
#
#     def forward(self, H):
#         transformed = self.linear(H)
#         aggregated = torch.einsum('ij,bjk->bik', self.norm_adj, transformed)
#         return F.leaky_relu(aggregated)
#
#
# class TactileGCN(nn.Module):
#     def __init__(self, input_dims=[330, 330, 340, 320, 380, 223]):
#         super().__init__()
#         self.projection = NodeProjection(input_dims)
#
#         # 普通GCN层（使用固定邻接矩阵）
#         self.gcn1 = GCNLayer(128, 128)
#         self.gcn2 = GCNLayer(128, 128)
#
#         # 改用平均池化
#         self.pooling = nn.Sequential(
#             nn.Linear(128, 128),
#             nn.ReLU()
#         )
#
#     def forward(self, x_list):
#         x = self.projection(x_list)  # (batch, 6, 128)
#         x = self.gcn1(x)
#         x = self.gcn2(x)
#         x = x.mean(dim=1)  # 平均池化 (batch, 128)
#         x = self.pooling(x)  # 最终投影
#         return x

###################################For Vision#######################################
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks,input_channel=3,num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        
        self.linear1 = nn.Linear(8192, 1024)#For specify 128*128 input
        # self.linear1 = nn.Linear(32768, 1024)#For specify 256,256 input
        self.bn_final=nn.BatchNorm1d(1024)
        self.linear2 = nn.Linear(1024, num_classes)
        self.linear = nn.Linear(8192, num_classes)#For specify 128*128 input
        # self.wmb1=WMB(dim=64)
        # self.wmb2=WMB(dim=128)
        # self.wmb3=WMB(dim=256)
        # self.wmb4=WMB(dim=512)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out =  F.relu(self.bn_final(self.linear1(out)))#add batch normal
        out = self.linear2(out)
        return out

def ResNet18(output_feature):
    return ResNet(BasicBlock, [2, 2, 2, 2],input_channel=4,num_classes=output_feature)

class ImageCNN(nn.Module):
    """ 
    Encoder network for image input list.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
    """
    def __init__(self,Flag_Merge=False):
        super().__init__()
        self.Flag_Merge=Flag_Merge

        self.res18=ResNet18(128)
        # self.fusion=SingleModalAttention(dim=128, num_heads=4)
        self.pose_linear=nn.Linear(128,7)

    def forward(self,inputs):
        image_feature=self.res18(inputs)
        # image_feature=self.fusion(image_feature)
        output=self.pose_linear(image_feature)

        ###Use for norm output###
        xyz,ori=output[:,:3],output[:,3:]
        ori=ori/torch.unsqueeze(torch.sqrt(torch.sum(torch.square(ori),dim=1)),dim=0).T
        # return xyz,ori
        output=torch.cat([xyz,ori],dim=1)
        ###Use for norm output###

        if self.Flag_Merge:
            return output,image_feature

        return output
    
###################################For Merge#######################################
class MergeModel(nn.Module):
    def __init__(self,Flag_Merge=False):
        super().__init__()
        self.Flag_Merge=Flag_Merge
        self.tactileGCN=TactileGCN()
        self.imageCNN=ImageCNN(Flag_Merge=True)
        self.fusion=RobustCrossModalFusion(dim=128, num_heads=4)
        self.pose_linear=nn.Linear(128,7)

    def forward(self,rgbd_data,tactile_data):
        _,image_feature=self.imageCNN(rgbd_data)
        tactile_feature=self.tactileGCN(tactile_data)
        merge_feature = self.fusion(image_feature, tactile_feature)


        # merge_feature=torch.cat([image_feature,tactile_feature],dim=1)
        # merge_feature=image_feature
        predict_pose=self.pose_linear(merge_feature)

        if self.Flag_Merge:
            return predict_pose,merge_feature
        else:
            return predict_pose



#
# if __name__ == '__main__':
#     # test_tactile_output()
#     # test_vision_output()
#     # test_merge_output()
#     # test_selectLSTM()
