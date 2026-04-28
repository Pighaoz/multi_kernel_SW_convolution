# contmix_timeseries_dual.py
"""
ContMixTimeSeries Dual-Stream (time-domain ContMix with multi-kernel local mixing)
Only model definitions — no data loading / training loops.

Key classes:
- MultiKernelLocalMix1D: depthwise conv branches with different kernel sizes + per-branch per-channel learnable weights -> fused -> pointwise conv
- GlobalMix1D: temporal multi-head attention (same as previously)
- ContMix1DBlock: uses MultiKernelLocalMix1D + token MLP + GlobalMix1D + token MLP
- ContMixDualStream: full model that produces time-domain features or logits

Notes:
- Input shape for time branch: (B, C_in, L)
- The multi-kernel local mix uses per-branch per-channel learned scaling factors to let the network learn importance of each receptive field per channel.
"""
from typing import Optional, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.Swconv_1d import ShifthWiseConv1dImplicit

# -------------------------
# Utilities
# -------------------------
class MLP1D(nn.Module):
    def __init__(self, dim, hidden_dim=None, act=nn.GELU):
        super().__init__()
        hidden_dim = hidden_dim or (dim * 4)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, dim)
        )
    def forward(self, x):
        # x: (B, L, C)
        return self.net(x)

class DepthwiseConv1d(nn.Module):
    def __init__(self, channels, kernel_size=7, padding=None, bias=False):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        # 标准 1D 卷积：groups=1（不再是 depthwise）
        self.dw = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=1,
            bias=bias,
        )
    def forward(self, x):
        return self.dw(x)

# -------------------------
# Multi-kernel local mixing
# -------------------------
class MultiKernelLocalMix1D(nn.Module):
    def __init__(self,
                 channels: int,
                 kernel_sizes: List[int] = (7, 15, 23),
                 use_bias: bool = False,
                 pw_hidden: Optional[int] = None,
                 dropout: float = 0.0,
                 use_sw_conv: bool = True,
                 sw_ghost_ratio: float = 0.23,
                 sw_N_path: int = 2,
                 sw_N_rep: int = 4):
        """
        Args:
            channels: 输入通道数
            kernel_sizes: 核大小列表
            use_bias: 是否使用偏置
            pw_hidden: Pointwise卷积的隐藏层维度
            dropout: Dropout率
            use_sw_conv: 是否使用SW卷积
            sw_ghost_ratio: SW卷积的ghost比例
            sw_N_path: SW卷积的路径数
            sw_N_rep: SW卷积的重复次数
        """
        super().__init__()
        self.channels = channels
        self.kernel_sizes = list(kernel_sizes)
        self.use_sw_conv = use_sw_conv

        # 创建多核卷积分支
        self.branches = nn.ModuleList()
        for k in self.kernel_sizes:
            if use_sw_conv:
                # 使用SW卷积替换原有的深度卷积
                branch = ShifthWiseConv1dImplicit(
                    in_channels=channels,
                    out_channels=channels,
                    big_kernel=k,  # 使用当前核大小作为big_kernel
                    small_kernel=3,  # 固定小核为3
                    stride=1,
                    group=1,
                    bn=True,
                    use_small_conv=True,
                    ghost_ratio=sw_ghost_ratio,
                    N_path=sw_N_path,
                    N_rep=sw_N_rep,
                    bias=use_bias
                )
            else:
                # 回退到原有的深度卷积
                branch = DepthwiseConv1d(
                    channels=channels,
                    kernel_size=k,
                    padding=k // 2,  # 保持输出长度不变
                    bias=use_bias
                )
            self.branches.append(branch)

        # 为每个分支创建可学习的scale参数
        self.register_parameter('branch_scale',
                                nn.Parameter(0.01 * torch.randn(len(self.branches), channels, 1)))

        # Pointwise卷积部分
        hidden = pw_hidden or channels
        self.pw = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(hidden, channels, kernel_size=1, bias=True)
        )

        # 归一化和最终的dropout
        self.norm = nn.BatchNorm1d(channels)
        self.final_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量，形状为 [B, C, L]

        Returns:
            output: 输出张量，形状为 [B, C, L]
        """
        # 存储各分支的输出
        branch_outputs = []

        # 处理每个卷积分支
        for i, branch in enumerate(self.branches):
            # 分支卷积
            branch_out = branch(x)  # [B, C, L]

            # 应用可学习的scale
            scale = self.branch_scale[i].to(branch_out.device)  # [1, C, 1]
            scaled_out = branch_out * scale  # [B, C, L]

            branch_outputs.append(scaled_out)

        # 融合所有分支输出
        if len(branch_outputs) > 1:
            # 堆叠并求和: [num_branches, B, C, L] -> [B, C, L]
            fused = torch.stack(branch_outputs, dim=0).sum(dim=0)
        else:
            fused = branch_outputs[0]

        # Pointwise卷积
        pw_out = self.pw(fused)  # [B, C, L]

        # 残差连接 + 归一化
        output = x + pw_out  # 残差连接
        output = self.norm(output)  # 归一化
        output = self.final_dropout(output)  # Dropout

        return output

# -------------------------
# GlobalMix1D: temporal attention (batch_first)
# -------------------------
class GlobalMix1D(nn.Module):
    def __init__(self, dim, num_heads=4, attn_dropout=0.0, proj_dropout=0.0, causal=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.causal = causal
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_dropout)
        )
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        # x: (B, C, L) -> convert to (B, L, C)
        b, c, l = x.shape
        x_t = x.permute(0,2,1).contiguous()  # (B, L, C)
        attn_mask = None
        if self.causal:
            mask = torch.triu(torch.ones((l,l), device=x.device, dtype=torch.bool), diagonal=1)
            attn_mask = mask
        attn_out, _ = self.attn(x_t, x_t, x_t, attn_mask=attn_mask)
        out = self.proj(attn_out)
        out = self.norm(x_t + out)
        return out.permute(0,2,1).contiguous()

# -------------------------
# ContMix1DBlock with Multi-kernel LocalMix
# -------------------------
class ContMix1DBlock(nn.Module):
    def __init__(self,
                 dim,
                 kernel_sizes=(3,5,7,11),
                 num_heads=4,
                 mlp_ratio=2.0,
                 drop_path_rate=0.0,
                 layer_scale_init_value: Optional[float] = 1e-5,
                 causal: bool = False):
        super().__init__()
        self.dim = dim
        self.local = MultiKernelLocalMix1D(dim, kernel_sizes=kernel_sizes)
        self.ln1 = nn.LayerNorm(dim)
        self.mlp1 = MLP1D(dim, hidden_dim=int(dim*mlp_ratio))
        self.globalmix = GlobalMix1D(dim, num_heads=num_heads, causal=causal)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp2 = MLP1D(dim, hidden_dim=int(dim*mlp_ratio))

        if layer_scale_init_value is not None:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_3 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.layer_scale_1 = None
            self.layer_scale_2 = None
            self.layer_scale_3 = None

        self.drop_path_rate = drop_path_rate

    def _drop_path(self, x):
        if self.drop_path_rate == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_path_rate
        shape = (x.shape[0],) + (1,) * (x.dim()-1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        return x.div(keep_prob) * binary_tensor

    def forward(self, x):
        # x: (B, C, L)
        y = self.local(x)
        if self.layer_scale_1 is not None:
            ls = self.layer_scale_1.view(1,-1,1)
            y = ls * y
        y = self._drop_path(y)
        x = x + y

        x_t = x.permute(0,2,1).contiguous()
        z = self.ln1(x_t)
        z = self.mlp1(z)
        if self.layer_scale_2 is not None:
            ls2 = self.layer_scale_2.view(1,1,-1)
            z = ls2 * z
        z = self._drop_path(z)
        x_t = x_t + z
        x = x_t.permute(0,2,1).contiguous()

        g = self.globalmix(x)
        if self.layer_scale_3 is not None:
            ls3 = self.layer_scale_3.view(1,-1,1)
            g = ls3 * g
        g = self._drop_path(g)
        x = x + g

        x_t = x.permute(0,2,1).contiguous()
        z = self.ln2(x_t)
        z = self.mlp2(z)
        z = self._drop_path(z)
        x_t = x_t + z
        x = x_t.permute(0,2,1).contiguous()
        return x

# -------------------------
# Full model (single stream - time domain only)
# -------------------------
class ContMixDualStream(nn.Module):
    """
    ContMix for time-series (time-domain only):
    - Time branch: patch conv -> stack of ContMix1DBlocks -> pooled features
    - Head: optional classifier head
    """
    def __init__(self,
                 in_channels: int = 3,
                 embed_dim: int = 128,
                 patch_size: int = 8,
                 depth: int = 6,
                 kernel_sizes: List[int] = (3,5,7,11),
                 num_heads: int = 4,
                 mlp_ratio: float = 2.0,
                 drop_path_rate: float = 0.0,
                 layer_scale_init_value: Optional[float] = 1e-5,
                 pooling: str = 'avg',
                 num_classes: Optional[int] = None):
        super().__init__()
        assert pooling in ('avg','cls')
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        # time patch embedding
        self.patch_embed = nn.Conv1d(in_channels, embed_dim, 
                                     kernel_size=patch_size, 
                                     stride=patch_size ,
                                     bias=False)
        self.patch_norm = nn.LayerNorm(embed_dim)
        # blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        blocks = []
        for i in range(depth):
            blocks.append(ContMix1DBlock(dim=embed_dim,
                                         kernel_sizes=kernel_sizes,
                                         num_heads=num_heads,
                                         mlp_ratio=mlp_ratio,
                                         drop_path_rate=dpr[i],
                                         layer_scale_init_value=layer_scale_init_value))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(embed_dim)
        self.pooling = pooling

        # classifier head optional
        self.head = None
        if num_classes is not None:
            self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.patch_embed.weight, mode='fan_out', nonlinearity='relu')
        if self.head is not None:
            nn.init.normal_(self.head.weight, std=0.02)
            if self.head.bias is not None:
                nn.init.zeros_(self.head.bias)

    def forward(self, x):
        """
        x: (B, C_in, L)
        returns: if head exists -> logits (B, num_classes) else feature vector (B, embed_dim)
        """
        B, C, L = x.shape
        # time branch
        x_p = self.patch_embed(x)  # (B, E, L')
        x_t = x_p.permute(0,2,1).contiguous()  # (B, L', E)
        x_t = self.patch_norm(x_t)
        seq_len = x_t.shape[1]

        # back to (B, E, L')
        x_b = x_t.permute(0,2,1).contiguous()
        for blk in self.blocks:
            x_b = blk(x_b)
        x_t = x_b.permute(0,2,1).contiguous()
        x_t = self.norm(x_t)
        if self.pooling == 'avg':
            time_feat = x_t.mean(dim=1)  # (B, E)
        else:
            time_feat = x_t[:,0,:]
        time_feat = F.layer_norm(time_feat, (time_feat.shape[-1],))
        # optionally classification head
        if self.head is not None:
            logits = self.head(time_feat)
            return logits
        return time_feat


model_dict = {
    'ContMixDualStream': [ContMixDualStream, 128],
}

class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='ContMixDualStream', num_classes=9):
        super(LinearClassifier, self).__init__()
        if name not in model_dict:
            raise ValueError(f"Unsupported model: {name}")
        model_func, feat_dim = model_dict[name]
        if not isinstance(feat_dim, int) or feat_dim <= 0:
            raise ValueError(f"Invalid feature dimension: {feat_dim}")
        
        # Modified for scarce labels: Added BatchNorm, Dropout, and reduced hidden dim
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.BatchNorm1d(feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(feat_dim // 2, num_classes)
        )

    def forward(self, features):
        return self.fc(features)

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=128, out_dim=64):
        super().__init__()
        hidden_dim = in_dim 
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
            #nn.BatchNorm1d(out_dim),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        x = self.net(x)
        return F.normalize(x,dim=-1)

class ContMixContrastive(nn.Module):
    def __init__(self, encoder: nn.Module = None, proj_dim=128):
        super().__init__()
        # If encoder not provided, create a default one
        if encoder is None:
            self.encoder = ContMixDualStream(
                in_channels=3,
                embed_dim=128,
                patch_size=8,
                depth=10,
                num_classes=None
            )
        else:
            self.encoder = encoder

        # use encoder's embed_dim for projection head input
        self.proj_head = ProjectionHead(self.encoder.embed_dim, proj_dim)

    def forward(self, x):
        encoded_features = self.encoder(x)  # (B, embed_dim)
        projected_features = self.proj_head(encoded_features)  # (B, proj_dim)
        return projected_features,encoded_features

# -------------------------
# quick sanity check (run when module executed directly)
# -------------------------
if __name__ == "__main__":
    # quick test
    print("Sanity check for contmix_timeseries_dual_v2.py")
    # test model
    enc = ContMixDualStream(in_channels=3, embed_dim=64, depth=4, num_classes=None)
    x = torch.randn(2,3,8192)  # batch 2, 3 channels, length 8192
    f = enc(x)
    print("Feature shape:", f.shape)  # (2, embed_dim)

    # test contrastive wrapper
    wrapper = ContMixContrastive(encoder=enc, proj_dim=128)
    z = wrapper(x)
    print("Projected shape:", z.shape)  # (2, 128)

    # test LinearClassifier on random features
    clf = LinearClassifier(name='ContMixDualStream', num_classes=5)
    dummy_feat = torch.randn(2, 128)
    out = clf(dummy_feat)
    print("Classifier output shape:", out.shape)  # (2,5)