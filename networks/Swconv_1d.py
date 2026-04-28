import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from networks.AddShift_mp_module_1d import AddShift_mp_module_1d


use_sync_bn = False
use_se = False


def get_bn(channels):
    # For 1D we use BatchNorm1d
    if use_sync_bn and hasattr(nn, 'SyncBatchNorm'):
        return nn.SyncBatchNorm(channels)
    else:
        return nn.BatchNorm1d(channels)


class ShifthWiseConv1dImplicit(nn.Module):
    """
    Native 1D variant of the original shift-wise module.

    Accepts input shaped [B, C, L] and returns [B, C, L].
    """

    def __init__(self,
                 in_channels,
                 out_channels=None,
                 big_kernel=15,
                 small_kernel=3,
                 stride=1,
                 group=1,
                 bn=True,
                 use_small_conv=True,
                 ghost_ratio=0.23,
                 N_path=2,
                 N_rep=4,
                 bias=False):
        super().__init__()
        self.kernels = (small_kernel, big_kernel)
        self.stride = stride

        padding, real_pad = self.shift(self.kernels)
        self.pad = padding, real_pad
        self.nk = math.ceil(big_kernel / small_kernel)

        ghostN = int(in_channels * ghost_ratio)
        repN = in_channels - ghostN
        np.random.seed(123)
        ghost = np.random.choice(in_channels, ghostN, replace=False).tolist() if ghostN > 0 else []
        ghost.sort()
        rep = list(set(range(in_channels)) - set(ghost))
        rep.sort()
        assert len(rep) == repN, f'len(rep):{len(rep)}==repN:{repN}'
        self.ghost = torch.LongTensor(ghost)
        self.rep = torch.LongTensor(rep)

        out_n = repN * self.nk
        self.LoRA = None

        # depthwise Conv1d branches (per-path small conv)
        self.LoRAs = nn.ModuleList([
            nn.Conv1d(repN, out_n,
                      kernel_size=small_kernel, stride=stride,
                      padding=padding, groups=repN, bias=False)
            for _ in range(N_path)
        ])

        self.device = None
        print(f'[SwConv1d] ghost_ratio={ghost_ratio}, repN={repN}, N_path={N_path}')
        self.use_bn = bn

        # 1D shift-add module
        self.loras = AddShift_mp_module_1d(big_kernel, small_kernel, repN, out_n, N_rep)

        if bn:
            self.bn_lora1 = get_bn(repN)
            self.bn_lora2 = get_bn(repN)
            self.bn_small = get_bn(repN)
        else:
            self.bn_lora1 = None
            self.bn_lora2 = None
            self.bn_small = None

        self._printed_shape = False

    def ghost_mask(self):
        weight = 0
        for name, tensor in self.named_parameters():
            if len(tensor.size()) == 3:
                weight += torch.sum(torch.abs(tensor.detach()), (1, 2))
        weight = torch.sum((weight > 0).reshape(-1, self.nk), 1)
        ghost = (weight == 0).reshape(1, -1, 1).float()
        return ghost

    def forward(self, inputs):
        # inputs: [B, C, L]
        if not self._printed_shape:
            try:
                print(f'[SwConv1d] input shape: {tuple(inputs.shape)}; kernels: {self.kernels}')
            except Exception:
                pass
            self._printed_shape = True

        ori_b, ori_c, ori_l = inputs.shape
        if self.device is None:
            self.device = inputs.get_device()
            if self.device == -1:
                self.device = None
            else:
                self.ghost = self.ghost.to(self.device)
                self.rep = self.rep.to(self.device)

        ghost_inputs = torch.index_select(inputs, 1, self.ghost) if self.ghost.numel() > 0 else torch.zeros((ori_b, 0, ori_l), device=inputs.device)
        rep_inputs = torch.index_select(inputs, 1, self.rep)

        out = 0
        if self.LoRA is None:
            for split_conv in self.LoRAs:
                xx = split_conv(rep_inputs)
                out = out + xx
        else:
            out = self.LoRA(rep_inputs)

        # loras expects [B, out_n, L] and returns three [B, repN, L]
        x1, x2, x3 = self.loras(out, ori_b, 1, ori_l)

        if self.use_bn:
            x1 = self.bn_lora1(x1)
            x2 = self.bn_lora2(x2)
            x3 = self.bn_small(x3)

        x = x1 + x2 + x3 + rep_inputs

        x = torch.cat([x, ghost_inputs], dim=1)
        return x

    def shift(self, kernels):
        mink, maxk = min(kernels), max(kernels)
        nk = math.ceil(maxk / mink)
        # Use symmetric padding that preserves length for Conv1d with stride=1
        # padding = floor(mink/2) is the common choice to keep output length == input length
        padding = mink // 2
        mid = maxk // 2
        real_pad = []
        for i in range(nk):
            extra_pad = mid - i * mink - padding
            real_pad.append(extra_pad)
        return padding, real_pad

    def merge_branches(self):
        if self.LoRA is None:
            bias = True if self.LoRAs[0].bias else False
            LoRA = nn.Conv1d(in_channels=self.LoRAs[0].in_channels,
                             out_channels=self.LoRAs[0].out_channels,
                             kernel_size=self.LoRAs[0].kernel_size,
                             stride=self.LoRAs[0].stride,
                             padding=self.LoRAs[0].padding,
                             dilation=self.LoRAs[0].dilation,
                             groups=self.LoRAs[0].groups,
                             bias=bias)
            weight, biasdata = 0, 0
            for merged_conv in self.LoRAs:
                weight = weight + merged_conv.weight.data
                if bias:
                    biasdata = biasdata + merged_conv.bias.data
            LoRA.weight.data = weight
            if bias:
                LoRA.bias.data = biasdata
            self.LoRA = LoRA
            self.__delattr__('LoRAs')
