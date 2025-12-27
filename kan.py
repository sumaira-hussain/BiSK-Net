# An enhanced KAN Bottleneck with custom spline activation, Depthwise Separable KAN Convolution and KANBottleneck module  with 3 stack layers.
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SplineActivation(nn.Module):
    """
    GPU-optimized spline-based activation function using pure PyTorch.
    Implements: φ(z) = ∑_{i=1}^N c_i * b_i(z) where N=10 B-splines
    """

    def __init__(self, num_splines=10, spline_order=3, grid_min=-2, grid_max=2):
        super().__init__()
        self.num_splines = num_splines
        self.spline_order = spline_order
        self.grid_min = grid_min
        self.grid_max = grid_max

        # Number of knots = num_splines + spline_order + 1
        num_knots = num_splines + spline_order + 1

        # Create knot vector - will be moved to device during forward pass
        self.register_buffer('knots', torch.linspace(grid_min, grid_max, num_knots))

        # Learnable coefficients c_i - initialized with small uniform distribution
        self.coeffs = nn.Parameter(torch.empty(num_splines).uniform_(-0.1, 0.1))

    def _compute_b_spline_basis_pytorch(self, x, i, k):
        """
        Pure PyTorch implementation of B-spline basis function evaluation
        using Cox-de Boor recursion formula
        """
        knots = self.knots

        # Base case: order 0
        if k == 0:
            return ((x >= knots[i]) & (x < knots[i + 1])).float()

        # Recursive cases
        term1_numerator = x - knots[i]
        term1_denominator = knots[i + k] - knots[i]
        term1 = torch.zeros_like(x)
        if term1_denominator != 0:
            term1 = term1_numerator / term1_denominator * self._compute_b_spline_basis_pytorch(x, i, k - 1)

        term2_numerator = knots[i + k + 1] - x
        term2_denominator = knots[i + k + 1] - knots[i + 1]
        term2 = torch.zeros_like(x)
        if term2_denominator != 0:
            term2 = term2_numerator / term2_denominator * self._compute_b_spline_basis_pytorch(x, i + 1, k - 1)

        return term1 + term2

    def _vectorized_b_spline_basis(self, x):
        """
        Vectorized computation of all B-spline basis functions
        x: input tensor of any shape
        returns: basis tensor of shape (..., num_splines)
        """
        original_shape = x.shape
        x_flat = x.reshape(-1)

        # Initialize basis matrix
        basis = torch.zeros(x_flat.shape[0], self.num_splines, device=x.device, dtype=x.dtype)

        # Compute each basis function
        for i in range(self.num_splines):
            basis[:, i] = self._compute_b_spline_basis_pytorch(x_flat, i, self.spline_order)

        # Handle edge case at maximum knot (include the last point)
        basis[:, -1] += (x_flat == self.knots[-1]).float()

        return basis.reshape(*original_shape, self.num_splines)

    def forward(self, x):
        """
        Forward pass of spline activation - entirely on GPU
        x: input tensor of any shape
        returns: activated tensor of same shape as x
        """
        # Ensure knots are on same device as x
        if self.knots.device != x.device:
            self.knots = self.knots.to(x.device)

        # Clamp input to knot range for stability
        x_clamped = torch.clamp(x, self.grid_min, self.grid_max)

        # Compute B-spline basis functions
        basis = self._vectorized_b_spline_basis(x_clamped)  # (..., num_splines)

        # Apply learnable coefficients: ∑ c_i * b_i(z)
        activated = torch.sum(basis * self.coeffs, dim=-1)

        return activated


class ConvKANLayer(nn.Module):
    """
    Implements the core KAN block for SKANet: Z_{k+1} = LayerNorm(Z_k + DwConv(phi(Z_k)))
    This block is designed for 512 channels (C=512) and uses Depthwise Convolution (DwConv).
    """

    def __init__(self, channels=512):
        super().__init__()

        # 1. Spline-based activation (phi) - Using the GPU-optimized implementation
        self.spline_activation = SplineActivation(num_splines=10)  # 10 B-splines [1]

        # 2. Depthwise Separable Convolution (DwConv)
        # Uses groups=channels to implement depthwise operation, minimizing parameters [1]
        self.depthwise_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            groups=channels,  # CRITICAL: Implements DwConv for efficiency
            bias=False
        )

        # 3. Layer Normalization
        # Applied across the channel dimension (C=512) after the residual block
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # 1. Apply spline activation (phi(Z_k))
        activated_x = self.spline_activation(x)

        # 2. Apply depthwise convolution (DwConv(phi(Z_k)))
        dw_conv_out = self.depthwise_conv(activated_x)

        # 3. Residual connection (Z_k + DwConv(...))
        out = x + dw_conv_out

        # 4. Layer Normalization (LayerNorm(...))
        # LayerNorm typically expects C to be the last dimension, requiring permute
        out = out.permute(0, 2, 3, 1)  # (B, H, W, C)
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)  # (B, C, H, W)

        return out


class KANBottleneck(nn.Module):
    """
    The KAN Bottleneck module processing the F4 feature map (512 channels, 11x11).
    Stacks 3 ConvKANLayer blocks.
    """

    def __init__(self, in_channels=512, depth=3):
        super().__init__()

        # Depth is set to 3 layers, adhering to the optimized architecture [1]
        self.layers = nn.ModuleList([
            ConvKANLayer(in_channels) for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Simulation Checkpoint A: KAN Bottleneck Unit Test on {device} ---")

    # Test the complete KAN Bottleneck
    kan_bottleneck = KANBottleneck(in_channels=512, depth=3).to(device)

    # Generate random input tensor F4 (B, 512, 11, 11)
    B, C, H, W = 4, 512, 11, 11
    test_input = torch.randn(B, C, H, W, device=device)

    # Perform forward pass
    with torch.no_grad():
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            output = kan_bottleneck(test_input)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms = start_event.elapsed_time(end_event)
        else:
            import time

            start_time = time.time()
            output = kan_bottleneck(test_input)
            elapsed_time_ms = (time.time() - start_time) * 1000

    # Validation Checks
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")

    # Assertion 1: Output Shape (B, 512, 11, 11)
    expected_shape = (B, C, H, W)
    assert output.shape == expected_shape, f"Shape mismatch. Expected {expected_shape}, got {output.shape}"
    print("✅ Assertion 1: Output shape is correct.")

    # Assertion 2: Parameter Audit
    total_params = sum(p.numel() for p in kan_bottleneck.parameters() if p.requires_grad)
    print(f"Total trainable parameters in KAN Bottleneck: {total_params:,}")

    expected_max_params = 20_000  # Conservative upper bound
    assert total_params <= expected_max_params, f"Parameter count {total_params} exceeds efficiency target"
    print("✅ Assertion 2: Parameter count confirms Depthwise Convolution efficiency.")

    print(f"Time elapsed for KAN Bottleneck (batch {B}): {elapsed_time_ms:.3f} ms")

    # Test gradient flow
    test_input.requires_grad = True
    output = kan_bottleneck(test_input)
    loss = output.mean()
    loss.backward()
    assert test_input.grad is not None, "Gradient flow check failed"
    print("✅ Assertion 3: Gradient flow is working correctly.")

    print("🎯 Simulation Checkpoint A: ALL TESTS PASSED - KAN Bottleneck is ready for integration!")