# Updated SDI module implementing Hadamard attention fusion for cross-scale feature refinement
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel-wise feature recalibration."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SDIModule(nn.Module):
    def __init__(self, low_channels, high_channels, out_channels):
        super().__init__()
        # Low-level projection
        self.conv_low = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # High-level processing
        self.conv_high = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        # SE Gating Block (The Upgrade)
        self.se = SEBlock(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, f_low, f_high):
        # 1. Align resolution
        f_low_upsampled = F.interpolate(f_low, size=f_high.shape[2:], mode='bilinear', align_corners=False)
        tilde_f_low = self.conv_low(f_low_upsampled)
        tilde_f_high = self.conv_high(f_high)

        # 2. Hadamard Attention
        A = tilde_f_low * tilde_f_high

        # 3. SE Gating + Residual
        # Apply SE to the attention map before adding residual
        refined_A = self.se(A)

        R = self.relu(refined_A + tilde_f_high)
        return R

'''
# Simulation Checkpoint B: SDI Module Integration Test
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("--- Simulation Checkpoint B: SDI Module Integration Test ---")

    # Define feature map constants from PVTv2-B2
    B = 4  # batch size

    # Instantiate SDI modules
    sdi2 = SDIModule(low_channels=512, high_channels=320, out_channels=512).to(device)
    sdi1 = SDIModule(low_channels=320, high_channels=128, out_channels=320).to(device)

    # Generate dummy feature maps matching PVTv2-B2 outputs
    F4 = torch.randn(B, 512, 11, 11, device=device)
    F3 = torch.randn(B, 320, 22, 22, device=device)
    F2 = torch.randn(B, 128, 44, 44, device=device)

    print("Testing SDI_2 (generates R2 for decoder stage D1)...")
    R2 = sdi2(F4, F3)
    expected_R2_shape = (B, 512, 22, 22)
    assert R2.shape == expected_R2_shape, f"R2 shape mismatch. Expected {expected_R2_shape}, got {R2.shape}"
    print(f"Test 1 passed: R2 shape correct - {R2.shape}")

    print("Testing SDI_1 (generates R1 for decoder stage D2)...")
    R1 = sdi1(F3, F2)
    expected_R1_shape = (B, 320, 44, 44)
    assert R1.shape == expected_R1_shape, f"R1 shape mismatch. Expected {expected_R1_shape}, got {R1.shape}"
    print(f"Test 2 passed: R1 shape correct - {R1.shape}")

    # Parameter audit
    total_sdi_params = (sum(p.numel() for p in sdi1.parameters() if p.requires_grad) +
                        sum(p.numel() for p in sdi2.parameters() if p.requires_grad))
    print(f"Total trainable parameters in SDI modules: {total_sdi_params:,}")

    # Test gradient flow
    F4.requires_grad = True
    F3.requires_grad = True
    R2 = sdi2(F4, F3)
    loss = R2.mean()
    loss.backward()
    assert F4.grad is not None and F3.grad is not None, "Gradient flow check failed"
    print("Test 3 passed: Gradient flow working correctly")

    print("Simulation Checkpoint B: ALL TESTS PASSED - SDI Modules ready for integration")
'''