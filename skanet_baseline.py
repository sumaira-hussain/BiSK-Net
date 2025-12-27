import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pvtv2 import pvt_v2_b2
import logging

# Assuming ConvBlock and other necessary components are defined elsewhere or implicitly available
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class BaselineDecoder(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        # Stage 4 (f4 -> d4)
        self.conv_f4 = ConvBlock(512, 256) # f4 is 512ch from PVTv2
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_d4 = ConvBlock(256 + 320, 256) # d4 + f3 (320ch from PVTv2)

        # Stage 3 (d4 -> d3)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_d3 = ConvBlock(256 + 128, 128) # d3 + f2 (128ch from PVTv2)

        # Stage 2 (d3 -> d2)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_d2 = ConvBlock(128 + 64, 64) # d2 + f1 (64ch from PVTv2)

        # Final output
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, f4, f3, f2, f1, original_size):
        d4 = self.conv_f4(f4)
        d4 = self.up4(d4)
        d4 = torch.cat([d4, f3], dim=1)
        d3 = self.conv_d4(d4)

        d3 = self.up3(d3)
        d3 = torch.cat([d3, f2], dim=1)
        d2 = self.conv_d3(d3)

        d2 = self.up2(d2)
        d2 = torch.cat([d2, f1], dim=1)
        d1 = self.conv_d2(d2)

        out = self.out_conv(d1)
        return F.interpolate(out, size=original_size, mode='bilinear', align_corners=False)


class SKANetPVT_Baseline(nn.Module):
    def __init__(self, num_classes=1, backbone_pretrained=True):
        super().__init__()
        self.backbone = pvt_v2_b2()
        if backbone_pretrained:
            try:
                state_dict = torch.load('pretrained/pvt_v2_b2.pth', map_location='cpu')
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
                logging.info(f"Baseline model - Pretrained loaded with: missing={missing}, unexpected={unexpected}")
            except Exception as e:
                logging.warning(f"Baseline model - Warning: Pretrained not loaded: {e}")

        self.decoder = BaselineDecoder(num_classes=num_classes)
        # Explicitly define decoder channels for KAN integration
        # Corresponds to f4, f3, f2, f1 in the order they are passed to the decoder.
        self.decoder_channels = [512, 320, 128, 64]

    def forward(self, x):
        f1, f2, f3, f4 = self.backbone.forward_features(x)
        output = self.decoder(f4, f3, f2, f1, original_size=x.shape[2:])
        return output

# Placeholder for main test block to avoid conflicts if this file is run directly
# The actual main test for skanet_baseline.py is typically run via `python -m lib.skanet_baseline`
# If this block causes issues, it can be removed.
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("--- Placeholder for SKANetPVT_Baseline self-test ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SKANetPVT_Baseline(backbone_pretrained=False).to(device)
    B, C, H, W = 4, 3, 352, 352
    x = torch.randn(B, C, H, W, device=device)
    with torch.no_grad():
        y = model(x)
    logging.info(f"Input shape: {x.shape}")
    logging.info(f"Output shape: {y.shape}")
    assert y.shape == (B, 1, H, W)
    logging.info("SKANetPVT_Baseline forward shape OK")
