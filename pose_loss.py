import os
import copy
import random
import open3d as o3d
import numpy as np
import torch,roma

from torch.nn.modules.loss import _Loss
import torch.nn as nn
import torch.nn.functional as F

def matrix_from_quaternion(quaternion, pos=None):
    """
    Return homogeneous rotation matrix from quaternion.
    Input is qx qy qz qw
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    q = q[[3, 0, 1, 2]]   # to w x y z

    if pos is None:
        pos = np.zeros(3)

    n = np.dot(q, q)
    if n < 1e-20:
        return np.identity(4)
    q *= np.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], pos[0]],
        [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], pos[1]],
        [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], pos[2]],
        [0.0, 0.0, 0.0, 1.0]])


def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.

    LAYOUT : [X, Y, Z, W]

    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.
    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4,))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / np.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                      [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                      [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                      [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[:, np.argmax(w)]
        if q[0] < 0.0:
            np.negative(q, q)
    return q


def project_points(points, K):
    us = np.divide(points[:, 0]*K[0, 0], points[:, 2]) + K[0, 2]
    vs = np.divide(points[:, 1]*K[1, 1], points[:, 2]) + K[1, 2]
    us = np.round(us).astype(np.int32).reshape(-1, 1)
    vs = np.round(vs).astype(np.int32).reshape(-1, 1)
    return np.hstack((us, vs))


# class Loss6D(_Loss):
#
#     def __init__(self,o3d_point_cloud,num_points=1000):
#         super(Loss6D, self).__init__(True)
#         self.model_points=np.asarray(o3d_point_cloud.points).astype(np.float32)
#
#         #downsample the points
#         dellist = [j for j in range(0, len(self.model_points))]
#         dellist = random.sample(dellist, len(self.model_points) - num_points)
#         self.model_points = np.delete(self.model_points, dellist, axis=0)
#
#         #change to tensor type
#         self.model_points=torch.tensor(self.model_points).cuda()
#
#     def forward(self,predict_pose,target_pose,model_points=None,sym=False):
#         if model_points is None:
#             model_points=self.model_points
#         bs=predict_pose.shape[0]
#
#         #1: change quaternion to matrix
#         #Norm the q
#         predict_q=predict_pose[:,3:]/torch.norm(predict_pose[:,3:],dim=1,keepdim=True)
#         predict_t=predict_pose[:,:3].view(bs,-1,3)
#         target_q=target_pose[:,3:]/torch.norm(target_pose[:,3:],dim=1,keepdim=True)
#         target_t=target_pose[:,:3].view(bs,-1,3)
#
#         predict_rotation_matrix=roma.unitquat_to_rotmat(predict_q)
#         target_rotation_matrix=roma.unitquat_to_rotmat(target_q)
#
#         #Change points cloud to same shape of batch size
#         model_points=model_points.repeat(bs,1).view(bs,-1,3)
#
#         #2: change model_points
#         predict_points=torch.matmul(model_points,predict_rotation_matrix)+predict_t
#         target_points=torch.matmul(model_points,target_rotation_matrix)+target_t
#
#
#         #3: calculate dist
#         dis=torch.sum(torch.mean(torch.norm(predict_points-target_points,dim=2),dim=1))
#
#         return dis
def compute_rotation_loss(pred_6d, gt_6d):
    """
    pred_6d/gt_6d: [B,6] 前两列向量
    """
    # 构造旋转矩阵的前两列
    a1, a2 = pred_6d[:, :3], pred_6d[:, 3:]
    b1, b2 = gt_6d[:, :3], gt_6d[:, 3:]

    # Gram-Schmidt正交化
    a1 = F.normalize(a1, p=2, dim=1)
    a2 = F.normalize(a2 - torch.sum(a2 * a1, dim=1, keepdim=True) * a1, p=2, dim=1)
    a3 = torch.cross(a1, a2)

    b1 = F.normalize(b1, p=2, dim=1)
    b2 = F.normalize(b2 - torch.sum(b2 * b1, dim=1, keepdim=True) * b1, p=2, dim=1)
    b3 = torch.cross(b1, b2)

    # 计算旋转矩阵差异 (L2距离)
    R_loss = F.mse_loss(torch.stack([a1, a2, a3], dim=2),
                        torch.stack([b1, b2, b3], dim=2))
    return R_loss


def compute_translation_loss(pred_t, gt_t, scale_factor=1):
    """
    pred_t/gt_t: [B,3] 平移分量
    """
    eps = 1e-8
    # 用Log-Cosh函数抑制离群点影响
    t_diff = scale_factor * (pred_t - gt_t + eps)
    # t_loss = torch.mean(torch.log(torch.cosh(t_diff) + eps))
    weights = 1.0 / (t_diff.detach().abs() + eps)  # 误差越小权重越大
    return torch.mean(weights * t_diff ** 2)
    # return t_loss


class Pose6DLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha  # 平移/旋转损失的平衡系数

    def forward(self, pred_pose, gt_pose):
        """
        pred_pose: [B,7] 预测姿态 (tx, ty, tz, qx, qy, qz, qw)
        gt_pose:   [B,7] 真实姿态
        """
        eps = 1e-8
        # 平移分量损失
        # t_loss = F.mse_loss(pred_pose[:, :3], gt_pose[:, :3]) * 10000.

        t_loss = compute_translation_loss(pred_pose[:, :3], gt_pose[:, :3]) * 1000.

        # 旋转分量损失（四元数）
        pred_q = pred_pose[:, 3:]
        gt_q = gt_pose[:, 3:]

        # 关键步骤1：四元数归一化
        pred_q = pred_q / (torch.norm(pred_q, p=2, dim=1, keepdim=True) + eps)
        gt_q = gt_q / (torch.norm(gt_q, p=2, dim=1, keepdim=True) + eps)

        # 关键步骤2：计算旋转差异
        dot_product = torch.sum(pred_q * gt_q, dim=1)
        dot_product = torch.clamp(dot_product, -1.0 + eps, 1.0 - eps)  # 避免精确±1
        angle_diff = 2 * torch.acos(dot_product)
        # 旋转损失计算（两种可选方法）
        q_loss = torch.mean(angle_diff) * 100.   # 直接角度差异



        # q_loss = torch.mean(1 - dot_product**2) # 余弦相似度变体

        # 综合损失
        total_loss = t_loss + self.alpha * q_loss
        return total_loss
