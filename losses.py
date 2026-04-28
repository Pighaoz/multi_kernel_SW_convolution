"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

import torch.nn.functional as F



class SupConGapLoss(nn.Module):
    """
    组合损失：L = SupConLoss(proj) - gap_weight * Gap(enc)
    """
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07,
                 gap_weight=0.1, normalize=True):
        super().__init__()
        self.supcon = SupConLoss(temperature=temperature,
                                 contrast_mode=contrast_mode,
                                 base_temperature=base_temperature)
        self.gap_weight = gap_weight
        self.normalize = normalize
        self.last_gap = 0.0
        self.last_pos_sim = 0.0
        self.last_neg_sim = 0.0

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: 这里的 features 是一个元组 (proj_features, enc_features)
                      每个 tensor 的形状应该是 [bsz, n_views, dim]
        """
        # === 1. 解包特征 ===
        if isinstance(features, (tuple, list)):
            proj_features, enc_features = features
        else:
            # 兼容旧代码：如果只传了一个 tensor，就都用它
            proj_features = features
            enc_features = features

        # === 2. 计算 SupCon Loss (使用投影头特征) ===
        # SupConLoss 内部会根据 labels/mask 自动判断是有监督还是无监督
        sup_loss = self.supcon(proj_features, labels, mask)

        # === 3. 计算 Gap (使用编码器特征) ===
        
        # [修正]：使用 unbind + cat 来展平，确保顺序是 [View1_All, View2_All]
        if enc_features.dim() == 3:
            enc_features_flat = torch.cat(torch.unbind(enc_features, dim=1), dim=0)
        else:
            enc_features_flat = enc_features

        # 强制 L2 归一化
        z = F.normalize(enc_features_flat, p=2, dim=1)

        # 计算相似度矩阵
        sim_mat = torch.matmul(z, z.T)

        # === [关键修正]：完全复刻 SupConLoss 的 Mask 生成逻辑 ===
        batch_size = enc_features.shape[0]
        contrast_count = enc_features.shape[1]
        device = enc_features.device

        # 1. 生成基础 Mask (Batch x Batch)
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            # 无监督模式 (SimCLR): Mask 为单位矩阵
            mask_gap = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            # 有监督模式 (SupCon): Mask 由标签相等性决定
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask_gap = torch.eq(labels, labels.T).float().to(device)
        else:
            # 自定义模式
            mask_gap = mask.float().to(device)

        # 2. 扩展 Mask (Tile)
        # Mask 重复方式：[M, M; M, M]，对应 [View1, View2] 的数据结构
        mask_gap = mask_gap.repeat(contrast_count, contrast_count)
        
        # 3. 排除对角线 (Self-contrast)
        # 无论是有监督还是无监督，样本自己和自己都不算“正样本对”用于计算 Loss/Gap
        logits_mask = torch.scatter(
            torch.ones_like(mask_gap), 1,
            torch.arange(batch_size * contrast_count).view(-1, 1).to(device), 0
        )
        
        # 4. 定义正负样本掩码
        # mask_gap=1 且不是对角线 -> 正样本
        pos_mask = mask_gap.bool() & logits_mask.bool()
        # mask_gap=0 且不是对角线 -> 负样本
        neg_mask = (~mask_gap.bool()) & logits_mask.bool()

        # === 4. 计算 Gap ===
        pos_cnt = pos_mask.sum(dim=1).clamp(min=1)
        neg_cnt = neg_mask.sum(dim=1).clamp(min=1)
        
        pos_sim = (sim_mat * pos_mask.float()).sum(dim=1) / pos_cnt
        neg_sim = (sim_mat * neg_mask.float()).sum(dim=1) / neg_cnt
        
        gap = (pos_sim - neg_sim).mean()

        # === 5. 组合 Loss ===
        # 改为 Margin 形式: SupCon + weight * (1 - Gap)
        # 这样 Loss 更容易保持为正数，且梯度方向不变（依然是最大化 Gap）
        total_loss = sup_loss + self.gap_weight * (1.0 - gap)
        
        self.last_gap = gap.detach().item()
        self.last_pos_sim = pos_sim.mean().detach().item()
        self.last_neg_sim = neg_sim.mean().detach().item()
        return total_loss
