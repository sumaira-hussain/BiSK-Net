# kan/integrate_kan.py
"""
Integration helpers to insert KANBlocks into an existing decoder.

Functions:
- make_kan_modules: create a ModuleDict of KAN blocks for specified decoder stages
- DecoderWithKAN: lightweight wrapper that applies KAN to selected multi-scale features
  before forwarding them to the original decoder.

Expected decoder interface: accepts (f4, f3, f2, f1, original_size=tuple).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
from kan.kan_block import KANBlock

def make_kan_modules(decoder_channels, kan_cfg):
    """
    decoder_channels: list[int] channels at each decoder stage (f4, f3, f2, f1 order)
    kan_cfg: dict { 'reduction': int, 'use_gate': bool, 'apply_at': list[int] }
    returns nn.ModuleDict of KAN modules keyed by stage index string
    """
    modules = nn.ModuleDict()
    for idx, ch in enumerate(decoder_channels):
        if idx in kan_cfg.get('apply_at', []):
            modules[f'kan_{idx}'] = KANBlock(
                in_ch=ch,
                out_ch=ch,
                reduction=kan_cfg.get('reduction', 16),
                use_gate=kan_cfg.get('use_gate', True)
            )
    return modules

class DecoderWithKAN(nn.Module):
    """
    Wrapper decoder that applies KAN modules to specified feature tensors before
    passing them into the base decoder.

    Expected base_decoder signature:
        base_decoder(f4, f3, f2, f1, original_size=tuple) -> output

    This wrapper applies KAN modules before calling the base decoder.
    """
    def __init__(self, base_decoder: nn.Module, kan_modules: nn.ModuleDict, apply_before: bool = True):
        super().__init__()
        self.base = base_decoder
        self.kan_modules = kan_modules or nn.ModuleDict()
        self.apply_before = apply_before
        if not self.apply_before:
            raise NotImplementedError("apply_before=False is not implemented in DecoderWithKAN")

    def forward(self, f4, f3, f2, f1, original_size, **kwargs):
        """
        f4, f3, f2, f1: tensors (or None) corresponding to decoder multi-scale features
        original_size: tuple(height, width) passed to base decoder for final upsampling
        """
        features = [f4, f3, f2, f1]
        processed_features = []

        for i, feat in enumerate(features):
            # handle missing/None features gracefully
            if feat is None:
                processed_features.append(None)
                continue

            key = f'kan_{i}'
            if key in self.kan_modules:
                kan_module = self.kan_modules[key]

                # ensure module is on same device as feature
                feat_device = feat.device
                kan_module = kan_module.to(feat_device)

                # remember original dtype so we can cast the KAN output back
                orig_dtype = feat.dtype

                # Determine parameter dtype; default to float32 if module has no params
                try:
                    param_dtype = next(kan_module.parameters()).dtype
                except StopIteration:
                    param_dtype = torch.float32

                # Explicitly convert the feature to the module's device AND dtype.
                # copy=False avoids an extra copy when not necessary.
                feat_for_kan = feat.to(device=feat_device, dtype=param_dtype, copy=False)

                # Run KAN with autocast disabled to avoid AMP dtype mismatch.
                with amp.autocast(enabled=False):
                    processed = kan_module(feat_for_kan)

                # Cast processed back to original dtype for downstream decoder
                if processed.dtype != orig_dtype:
                    processed_feat = processed.to(device=feat_device, dtype=orig_dtype)
                else:
                    processed_feat = processed

                processed_features.append(processed_feat)
            else:
                # no KAN for this stage
                processed_features.append(feat)

        # Unpack and forward to base decoder
        return self.base(
            processed_features[0],
            processed_features[1],
            processed_features[2],
            processed_features[3],
            original_size=original_size,
            **kwargs
        )


if __name__ == "__main__":
    # Quick test for DecoderWithKAN standalone (mock KANBlock for safe testing)
    from lib.skanet_baseline import BaselineDecoder

    class MockKANBlock(nn.Module):
        def __init__(self, in_ch, out_ch, **kwargs):
            super().__init__()
            self.conv = nn.Conv2d(in_ch, out_ch, 1)
        def forward(self, x, skip=None):
            return self.conv(x)

    def mock_make_kan_modules(decoder_channels, kan_cfg):
        modules = nn.ModuleDict()
        for idx, ch in enumerate(decoder_channels):
            if idx in kan_cfg.get('apply_at', []):
                modules[f'kan_{idx}'] = MockKANBlock(in_ch=ch, out_ch=ch)
        return modules

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_decoder = BaselineDecoder(num_classes=1).to(device)
    kan_cfg = {'apply_at': [1, 2], 'reduction': 16, 'use_gate': True}
    kan_modules = mock_make_kan_modules([512, 320, 128, 64], kan_cfg)
    model = DecoderWithKAN(base_decoder, kan_modules, apply_before=True).to(device)

    B, C, H, W = 2, 3, 352, 352
    f1 = torch.randn(B, 64, H//4, W//4, device=device)
    f2 = torch.randn(B, 128, H//8, W//8, device=device)
    f3 = torch.randn(B, 320, H//16, W//16, device=device)
    f4 = torch.randn(B, 512, H//32, W//32, device=device)

    with torch.no_grad():
        y = model(f4, f3, f2, f1, original_size=(H, W))
    print("DecoderWithKAN forward OK, output shape:", y.shape)
