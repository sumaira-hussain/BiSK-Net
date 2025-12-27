# lib/skanet_sdi.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pvtv2 import pvt_v2_b2
from lib.modules.sdi_v3 import SDIModule
from lib.skanet_baseline_v3 import BaselineDecoder  # reuse existing decoder

'''
class BoundaryHead(nn.Module):
    """Boundary prediction head for boundary-aware segmentation"""

    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.head(x)


class SKANetPVT_SDI(nn.Module):
    """
    PVTv2-B2 backbone + SDI fusion + progressive decoder (wrapper that reuses BaselineDecoder)
    Uses small 1x1 adapters to convert SDI outputs to the channel sizes expected by BaselineDecoder.
    """

    def __init__(self, num_classes=1, backbone_pretrained=True):
        super().__init__()
        self.backbone = pvt_v2_b2()
        if backbone_pretrained:
            try:
                state_dict = torch.load('pretrained/pvt_v2_b2.pth', map_location='cpu')
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
                print(f"SKANetPVT_SDI - Pretrained loaded with: missing={missing}, unexpected={unexpected}")
            except Exception as e:
                print(f"SKANetPVT_SDI - Warning: Pretrained not loaded: {e}")

        # SDI modules
        self.sdi2 = SDIModule(low_channels=512, high_channels=320, out_channels=512)  # F4->F3 -> R2 (512,22,22)
        self.sdi1 = SDIModule(low_channels=320, high_channels=128, out_channels=320)  # F3->F2 -> R1 (320,44,44)

        # small 1x1 adapters to match BaselineDecoder expected channels:
        self.r2_to_f3 = nn.Conv2d(512, 320, kernel_size=1, bias=False)
        self.r1_to_f2 = nn.Conv2d(320, 128, kernel_size=1, bias=False)

        # reuse baseline decoder
        self.decoder = BaselineDecoder(num_classes=num_classes)

        # Boundary head - uses d2 features (64 channels) from decoder
        self.boundary_head = BoundaryHead(in_channels=64, out_channels=1)

        # for KAN: decoder stage channels (f4,f3,f2,f1)
        self.decoder_channels = [512, 320, 128, 64]

        # init adapters (small kaiming init)
        nn.init.kaiming_normal_(self.r2_to_f3.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.r1_to_f2.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        f1, f2, f3, f4 = self.backbone.forward_features(x)

        # SDI fusion
        R2 = self.sdi2(f4, f3)  # (B,512,22,22)
        R1 = self.sdi1(f3, f2)  # (B,320,44,44)

        # adapt SDI outputs to decoder-expected encoder channel sizes
        f3_like = self.r2_to_f3(R2)  # (B,320,22,22) -> matches original f3
        f2_like = self.r1_to_f2(R1)  # (B,128,44,44) -> matches original f2

        # call baseline decoder: expects (f4,f3,f2,f1)
        seg_output = self.decoder(f4, f3_like, f2_like, f1, original_size=x.shape[2:])

        # Get d2 features for boundary prediction
        if hasattr(self.decoder, 'd2_features'):
            d2_features = self.decoder.d2_features
            boundary_output = self.boundary_head(d2_features)
            # REMOVED: boundary_output = torch.sigmoid(boundary_output)
        else:
            # Fallback: use final features for boundary (keep as logits)
            boundary_output = seg_output  # Keep as logits, not sigmoid

        # REMOVED: seg_output = torch.sigmoid(seg_output)

        return seg_output, boundary_output

'''

class SKANetPVT_SDI(nn.Module):
    # Added use_boundary flag (defaulting to True for your main experiments)
    def __init__(self, num_classes=1, backbone_pretrained=True, use_boundary=True):
        super().__init__()
        self.backbone = pvt_v2_b2()
        if backbone_pretrained:
            try:
                state_dict = torch.load('pretrained/pvt_v2_b2.pth', map_location='cpu')
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
                print(f"SKANetPVT_SDI - Pretrained loaded")
            except Exception as e:
                print(f"SKANetPVT_SDI - Warning: Pretrained not loaded: {e}")

        # SDI modules (Now using the updated SE-SDI automatically via import)
        self.sdi2 = SDIModule(low_channels=512, high_channels=320, out_channels=512)
        self.sdi1 = SDIModule(low_channels=320, high_channels=128, out_channels=320)

        # Adapters
        self.r2_to_f3 = nn.Conv2d(512, 320, kernel_size=1, bias=False)
        self.r1_to_f2 = nn.Conv2d(320, 128, kernel_size=1, bias=False)

        # --- UPDATE: Initialize Decoder with Boundary Head ---
        # Initialize Decoder based on the flag
        self.decoder = BaselineDecoder(num_classes=num_classes, return_boundary=use_boundary)

        # IMPORTANT: Expose boundary_head so training script knows to use Boundary Loss
        # Only expose boundary_head if we are actually using it
        if use_boundary:
            self.boundary_head = self.decoder.boundary_head

        self.decoder_channels = [512, 320, 128, 64]

        # Init adapters
        nn.init.kaiming_normal_(self.r2_to_f3.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.r1_to_f2.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        f1, f2, f3, f4 = self.backbone.forward_features(x)

        R2 = self.sdi2(f4, f3)
        R1 = self.sdi1(f3, f2)

        f3_like = self.r2_to_f3(R2)
        f2_like = self.r1_to_f2(R1)

        # Decoder now returns (seg, bnd) tuple
        out = self.decoder(f4, f3_like, f2_like, f1, original_size=x.shape[2:])
        return out

if __name__ == "__main__":
    # quick self-test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("--- SKANetPVT_SDI forward test ---")
    model = SKANetPVT_SDI(backbone_pretrained=False).to(device)
    B, C, H, W = 2, 3, 352, 352
    x = torch.randn(B, C, H, W, device=device)
    with torch.no_grad():
        seg_out, boundary_out = model(x)
    print("Input shape:", x.shape)
    print("Segmentation output shape:", seg_out.shape)
    print("Boundary output shape:", boundary_out.shape)
    assert seg_out.shape == (B, 1, H, W)
    assert boundary_out.shape == (B, 1, H, W)
    print("SKANetPVT_SDI forward shape OK")