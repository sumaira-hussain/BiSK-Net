# losses_v2.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.losses import CompositeLoss


class CompositeBoundaryLoss(nn.Module):
    def __init__(self, lambda_seg=1.0, lambda_bnd=0.5):
        super().__init__()
        self.lambda_seg = lambda_seg
        self.lambda_bnd = lambda_bnd
        self.seg_loss = CompositeLoss(lambda_dice=1.0, lambda_bce=0.5)
        self.bnd_loss_fn = nn.MSELoss()

    def generate_boundary_label(self, mask):
        # Generate Sobel edges from GT mask on-the-fly
        kernel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).to(mask.device).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).to(mask.device).view(1, 1, 3, 3)

        b_x = F.conv2d(mask, kernel_x, padding=1)
        b_y = F.conv2d(mask, kernel_y, padding=1)
        boundary = torch.sqrt(b_x ** 2 + b_y ** 2 + 1e-8)

        # Normalize to 0-1
        boundary = (boundary > 0.1).float()
        return boundary

    def forward(self, seg_pred, bnd_pred, target):
        # Segmentation Loss
        L_seg = self.seg_loss(seg_pred, target)

        # Boundary Loss
        bnd_target = self.generate_boundary_label(target)
        L_bnd = self.bnd_loss_fn(torch.sigmoid(bnd_pred), bnd_target)

        total_loss = self.lambda_seg * L_seg + self.lambda_bnd * L_bnd
        return total_loss, L_seg, L_bnd

'''
class CompositeBoundaryLoss(nn.Module):
    """
    Composite Loss with Boundary Awareness: L = L_seg + λ * L_boundary
    Uses MSE loss for boundary prediction supervision
    """

    def __init__(self, lambda_seg=1.0, lambda_bnd=0.5):
        super().__init__()
        from lib.losses import CompositeLoss
        self.seg_criterion = CompositeLoss(lambda_dice=2.0, lambda_bce=0.5) #lambda_dice=1.0
        self.boundary_criterion = nn.MSELoss()
        self.lambda_bnd = lambda_bnd

    def sobel_filter(self, mask):
        """
        Generate boundary ground truth using Sobel filter
        mask: [B, 1, H, W] binary segmentation mask
        returns: boundary map [B, 1, H, W] with values 0-1
        """
        # Simple 3x3 Sobel kernel approximation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               device=mask.device, dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               device=mask.device, dtype=torch.float32).view(1, 1, 3, 3)

        grad_x = F.conv2d(mask, sobel_x, padding=1)
        grad_y = F.conv2d(mask, sobel_y, padding=1)

        # L1 norm of gradients
        boundary_map = torch.abs(grad_x) + torch.abs(grad_y)

        # --- FIX START ---
        # Normalize to 0-1 range based on max value instead of using Sigmoid
        if boundary_map.max() > 0:
            boundary_map = boundary_map / boundary_map.max()

        # Optional: Threshold to make it binary and clean
        boundary_map = (boundary_map > 0.1).float()

        return boundary_map

    def forward(self, seg_preds, boundary_preds, targets):
        """
        seg_preds: [B, 1, H, W] - segmentation predictions (sigmoid already applied)
        boundary_preds: [B, 1, H, W] - boundary predictions (sigmoid already applied)
        targets: [B, 1, H, W] - ground truth masks
        """
        # 1. Segmentation Loss - use raw logits for BCEWithLogitsLoss
        L_seg = self.seg_criterion(seg_preds, targets)

        # 2. Boundary Loss
        gt_boundary = self.sobel_filter(targets)
        L_boundary = self.boundary_criterion(boundary_preds, gt_boundary)

        # 3. Total Loss
        total_loss = L_seg + (self.lambda_bnd * L_boundary)

        return total_loss, L_seg, L_boundary
'''