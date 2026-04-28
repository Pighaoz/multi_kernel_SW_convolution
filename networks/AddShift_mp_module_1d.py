import torch
import torch.nn as nn
import math

"""
A PyTorch-native true 1D variant of the AddShift_mp_module.
This replaces the placeholder 1x1 convolutions with actual spatial shifted-add operations.
It mimics the logic of the 2D version (shifting and accumulating features) using torch.roll
and zero-masking to emulate zero-padding.
"""

class AddShift_mp_module_1d(nn.Module):
    def __init__(self, big_kernel, small_kernel, repN, out_n, N_rep=4):
        """
        big_kernel: the target large receptive field size
        small_kernel: the actual underlying small conv size
        repN: the number of output channels (c_out)
        out_n: the number of input channels (c_in)
        N_rep: group factor (group_in)
        """
        super().__init__()
        self.device = None
        self.big_kernel = big_kernel
        self.small_kernel = small_kernel
        self.c_out = repN
        self.c_in = out_n
        self.group_in = N_rep
        
        self.nk = math.ceil(big_kernel / small_kernel)
        self.kernels = (small_kernel, big_kernel)
        
        # Calculate real padding shifts similar to the 2D logic
        self.padding, self.real_pad = self.get_shift_pads(self.kernels)
        
        # In the 2D algorithm, c_in is generally split to accumulate into c_out
        self.expected_c_in = self.c_out * self.group_in * self.nk
        if self.expected_c_in != self.c_in:
            # Fallback block if input projection dimensions don't securely match
            self.channel_align = nn.Conv1d(self.c_in, self.expected_c_in, kernel_size=1, bias=False)
        else:
            self.channel_align = nn.Identity()

        # Generate random shift indices for the 3 branches
        torch.manual_seed(123)
        self.register_buffer('shift_idx_1', self._generate_shuffle_idx())
        self.register_buffer('shift_idx_2', self._generate_shuffle_idx())
        self.register_buffer('shift_idx_3', self._generate_identity_idx())

    def get_shift_pads(self, kernels):
        mink, maxk = min(kernels), max(kernels)
        nk = math.ceil(maxk / mink) 
        
        # padding for the small convs to keep input/output length consistent
        padding = mink // 2  
        
        real_pad = []
        # Calculate perfectly symmetric offsets around 0
        center_i = (nk - 1) / 2.0
        for i in range(nk): 
            extra_pad = int((center_i - i) * mink)
            real_pad.append(extra_pad)
        return padding, real_pad

    def _generate_shuffle_idx(self):
        # Generates shift indices offsets for each channel
        indices = []
        for _ in range(self.group_in):
            group_idx = torch.cat([torch.randperm(self.nk) + i * self.nk for i in range(self.c_out)])
            indices.append(group_idx.int())
        return torch.cat(indices) # Shape: (group_in * c_out * nk)

    def _generate_identity_idx(self):
        # Generates identity/center shift indices equivalent to 2D
        indices = []
        for _ in range(self.group_in):
            group_idx = torch.randint(0, self.nk, [self.c_out]) + torch.arange(self.c_out) * self.nk
            group_idx_full = torch.repeat_interleave(group_idx, self.nk)
            indices.append(group_idx_full.int())
        return torch.cat(indices)

    def apply_shift_and_accumulate(self, x, shift_indices):
        """
        x: [B, expected_c_in, L]
        """
        B, C, L = x.shape
        # Create output tensor directly on the same device
        out = torch.zeros(B, self.c_out, L, device=x.device, dtype=x.dtype)
        
        for ch in range(C):
            # C is effectively N_rep * c_out * nk
            group_size = self.c_out * self.nk
            cout_idx = (ch % group_size) // self.nk

            # Compute actual rolling shift val
            pad_val_idx = shift_indices[ch].item() % self.nk
            shift_val = self.real_pad[pad_val_idx]
            
            chan_feature = x[:, ch, :] # [B, L]
            
            # Application of manual roll and pad-masking (simulation of zero-padding)
            if shift_val != 0:
                chan_shifted = torch.roll(chan_feature, shifts=shift_val, dims=-1)
                # Mask out the rolled-over parts
                if shift_val > 0:
                    chan_shifted[:, :shift_val] = 0
                else:
                    chan_shifted[:, shift_val:] = 0
            else:
                chan_shifted = chan_feature
                
            out[:, cout_idx, :] += chan_shifted
            
        return out

    def forward(self, x, ori_b=None, ori_h_or_1=None, ori_w_or_l=None):
        if x.dim() == 4 and x.size(2) == 1:
            x = x.squeeze(2)

        x = self.channel_align(x)

        # Apply specific shifts to simulate the 3 multi-path branches in 1D
        x1 = self.apply_shift_and_accumulate(x, self.shift_idx_1)
        x2 = self.apply_shift_and_accumulate(x, self.shift_idx_2)
        x3 = self.apply_shift_and_accumulate(x, self.shift_idx_3)

        return x1, x2, x3

__all__ = ['AddShift_mp_module_1d']