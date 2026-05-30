# HIBI-Net: An Efficient Hybrid Framework with Multi‑Scale Context Aggregation and Boundary Regularization for Polyp Segmentation

This repository contains the official PyTorch implementation of **HIBI‑Net** (Hybrid Interaction and Boundary‑Integrated Network), a high‑performance yet efficient hybrid architecture for automatic polyp segmentation in colonoscopy images. HIBI‑Net balances global semantic reasoning with fine boundary detail, achieving state‑of‑the‑art structural fidelity and real‑time inference on standard GPUs.

## Overview

Automated polyp segmentation is challenged by:
- High variability in polyp size, shape, and texture.
- Weak or ambiguous boundaries caused by specular highlights, motion blur, or mucosal folds.
- The semantic–spatial misalignment that arises when fusing CNN local features with Transformer global context.

HIBI‑Net overcomes these limitations through three novel components:

1. **Split‑Channel Multi‑scale Kernel Interaction (MKI) Bottleneck**  
   Partitions the feature space into four channel groups, applies depthwise convolutions with kernels 3, 5, and 7 in parallel, and combines them with an identity shortcut. Followed by channel‑wise (SE) and spatial gating, it captures multi‑scale context with only **0.69M** parameters.

2. **Gated Spatial‑Domain Interaction (gSDI) Module**  
   Uses upsampled semantic confidence maps as a selective gate to suppress background noise and specular artifacts in shallow encoder features. An SE block and a residual connection further refine the alignment, preserving only task‑relevant spatial details.

3. **Boundary‑Aware Dual Supervision**  
   A parameter‑free Sobel operator generates edge targets from ground‑truth masks. A composite loss (\( \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{region}} + 0.2\,\mathcal{L}_{\text{boundary}} \)) enforces contour consistency without learnable edge heads or curriculum‑based training.



