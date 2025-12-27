# kan/kan_block.py
"""
KANBlock: Kernel Attention Network block for decoder integration.

Design goals
- Lightweight multi-scale attention to improve boundary and cross-scale reasoning.
- Configurable channels, depth, and gating for ablations.

Usage
from kan.kan_block import KANBlock
kan = KANBlock(in_ch=256, out_ch=256, reduction=16, use_gate=True)
y = kan(x, skip=None)  # skip optional for cross-scale fusion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEConv(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class DepthwiseConv(nn.Module):
    def __init__(self, ch, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size, padding=kernel_size//2, groups=ch, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class KANBlock(nn.Module):
    def __init__(self, in_ch, out_ch=None, reduction=16, use_gate=True, mid_ch=None):
        super().__init__()
        out_ch = out_ch or in_ch
        mid_ch = mid_ch or max(in_ch // 2, 8)
        self.pre = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True)
        )
        # Multi-kernel depthwise branch
        self.dw3 = DepthwiseConv(mid_ch, kernel_size=3)
        self.dw5 = DepthwiseConv(mid_ch, kernel_size=5)
        self.dw7 = DepthwiseConv(mid_ch, kernel_size=7)
        self.fuse = nn.Sequential(
            nn.Conv2d(mid_ch * 3, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.se = SEConv(out_ch, reduction=reduction)
        self.use_gate = use_gate
        if use_gate:
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_ch, out_ch // 4, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch // 4, out_ch, 1),
                nn.Sigmoid()
            )

    def forward(self, x, skip=None):
        """
        x: tensor of shape [B,C,H,W]
        skip: optional skip connection from encoder or earlier decoder level
        returns tensor [B,out_ch,H,W]
        """
        # --- BEGIN dtype-handling patch ---
        # Remember original input dtype so we can cast output back
        input_dtype = x.dtype
        # Determine module (parameter) dtype; default to float32 if module has no params
        try:
            param_dtype = next(self.parameters()).dtype
        except StopIteration:
            param_dtype = torch.float32
        # If input dtype differs from module param dtype, cast input to param dtype
        if input_dtype != param_dtype:
            x = x.to(param_dtype)
            if skip is not None:
                skip = skip.to(param_dtype)
        # --- END dtype-handling patch ---
        
        z = self.pre(x)
        b1 = self.dw3(z)
        b2 = self.dw5(z)
        b3 = self.dw7(z)
        cat = torch.cat([b1, b2, b3], dim=1)
        out = self.fuse(cat)
        out = self.se(out)
        if skip is not None:
            # simple addition fusion, keep spatial alignment requirement
            if skip.shape[2:] != out.shape[2:]:
                skip = F.interpolate(skip, size=out.shape[2:], mode='bilinear', align_corners=False)
            out = out + skip
        if self.use_gate:
            g = self.gate(out)
            out = out * g


        # Cast output back to the original input dtype so downstream ops keep same dtype
        if out.dtype != input_dtype:
            out = out.to(input_dtype)
            
        return out
