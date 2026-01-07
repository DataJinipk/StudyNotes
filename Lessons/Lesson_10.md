# Lesson 10: Computer Vision

**Topic:** Computer Vision: From Image Processing to Deep Learning Visual Understanding
**Prerequisites:** Lesson 8 (Neural Network Architectures), Lesson 5 (Deep Learning)
**Estimated Study Time:** 3-4 hours
**Difficulty:** Advanced

---

## Learning Objectives

Upon completion of this lesson, learners will be able to:

1. **Analyze** image representation, convolution operations, and feature extraction in CNNs
2. **Compare** landmark CNN architectures (LeNet, AlexNet, VGG, ResNet, EfficientNet) and their design principles
3. **Evaluate** object detection frameworks (R-CNN family, YOLO, SSD) for different application requirements
4. **Design** segmentation architectures (FCN, U-Net, DeepLab, Mask R-CNN) for pixel-level understanding
5. **Apply** modern computer vision techniques including Vision Transformers and self-supervised learning

---

## Introduction

Computer Vision enables machines to interpret and understand visual information from the world—images, videos, and 3D data. As the primary sensory modality for humans, vision presents unique challenges: the enormous variability in appearance due to viewpoint, lighting, occlusion, and the sheer complexity of the visual world.

The deep learning revolution transformed computer vision from labor-intensive feature engineering (SIFT, HOG, Haar cascades) to end-to-end learned representations. Convolutional Neural Networks (CNNs) automatically learn hierarchical features—from edges and textures to object parts and semantic concepts—directly from data. This paradigm shift enabled breakthrough performance across tasks: image classification, object detection, semantic segmentation, pose estimation, and generation.

Understanding CNN architectures, detection frameworks, and segmentation methods is essential for building visual AI systems that power autonomous vehicles, medical imaging, surveillance, augmented reality, and countless other applications.

---

## Core Concepts

### Concept 1: Image Representation and Preprocessing

Digital images are multi-dimensional arrays of pixel values that require careful preprocessing before neural network processing.

**Image Data Structure:**

```
Grayscale Image: H × W × 1
  - Each pixel: intensity value [0, 255] or [0.0, 1.0]

Color Image (RGB): H × W × 3
  - Three channels: Red, Green, Blue
  - Each pixel: (R, G, B) tuple

Example: 224 × 224 × 3 image
  - 224 pixels height
  - 224 pixels width
  - 3 color channels
  - Total: 150,528 values
```

**Preprocessing Pipeline:**

| Step | Operation | Purpose |
|------|-----------|---------|
| Resize | Scale to fixed size (224×224, 416×416) | Consistent input dimensions |
| Normalization | (pixel - mean) / std | Zero-centered, unit variance |
| Channel ordering | RGB ↔ BGR conversion | Library compatibility |
| Data type | uint8 → float32 | Numerical precision |

**Data Augmentation:**

```
Training augmentations:
├── Geometric
│   ├── Random crop (224×224 from 256×256)
│   ├── Horizontal flip (p=0.5)
│   ├── Random rotation (±15°)
│   └── Random scale (0.8-1.2x)
│
├── Photometric
│   ├── Color jitter (brightness, contrast, saturation)
│   ├── Random grayscale (p=0.1)
│   └── Gaussian blur
│
└── Advanced
    ├── Mixup (blend two images)
    ├── CutMix (paste patch from another image)
    └── RandAugment (automated augmentation)
```

**Normalization Standards:**

| Dataset | Mean (RGB) | Std (RGB) |
|---------|------------|-----------|
| ImageNet | [0.485, 0.456, 0.406] | [0.229, 0.224, 0.225] |
| COCO | [0.471, 0.448, 0.408] | [0.234, 0.239, 0.242] |
| Custom | Compute from training set | Compute from training set |

---

### Concept 2: Convolution Operation

The convolution operation is the fundamental building block of CNNs, enabling local feature detection with parameter sharing.

**Mathematical Definition:**

```
Output[i, j] = Σₘ Σₙ Input[i+m, j+n] × Kernel[m, n] + bias

Where:
- Input: H_in × W_in × C_in
- Kernel: K × K × C_in (per filter)
- Output: H_out × W_out × C_out

Output dimensions:
H_out = (H_in - K + 2P) / S + 1
W_out = (W_in - K + 2P) / S + 1

Where P = padding, S = stride
```

**Convolution Example (3×3 kernel, stride=1, no padding):**

```
Input (5×5):              Kernel (3×3):
┌─────────────────┐       ┌─────────┐
│ 1  2  3  4  5   │       │ 1  0 -1 │
│ 6  7  8  9  10  │       │ 2  0 -2 │
│ 11 12 13 14 15  │   *   │ 1  0 -1 │
│ 16 17 18 19 20  │       └─────────┘
│ 21 22 23 24 25  │
└─────────────────┘

Output (3×3):
┌───────────┐
│ -8 -8 -8  │
│ -8 -8 -8  │
│ -8 -8 -8  │
└───────────┘

Calculation for Output[0,0]:
= 1×1 + 2×0 + 3×(-1) + 6×2 + 7×0 + 8×(-2) + 11×1 + 12×0 + 13×(-1)
= 1 - 3 + 12 - 16 + 11 - 13 = -8
```

**Key Parameters:**

| Parameter | Description | Common Values |
|-----------|-------------|---------------|
| Kernel size | Filter dimensions | 1×1, 3×3, 5×5, 7×7 |
| Stride | Step size | 1 (preserve size), 2 (downsample) |
| Padding | Border pixels added | 'same' (preserve), 'valid' (no pad) |
| Dilation | Spacing in kernel | 1 (standard), 2+ (dilated/atrous) |

**Parameter Efficiency:**

```
Fully connected: H × W × C_in × H × W × C_out parameters
                 (every input connected to every output)

Convolution: K × K × C_in × C_out parameters
             (same filter applied everywhere)

Example: 224×224×64 → 224×224×128
  FC: 224 × 224 × 64 × 224 × 224 × 128 ≈ 409 trillion parameters!
  Conv (3×3): 3 × 3 × 64 × 128 = 73,728 parameters
```

---

### Concept 3: Pooling and Spatial Hierarchy

Pooling operations reduce spatial dimensions while retaining important information, building the spatial hierarchy fundamental to CNNs.

**Pooling Types:**

```
Max Pooling (2×2, stride=2):
┌─────────┐        ┌─────┐
│ 1  3 │ 2  4 │    │ 3 │ 4 │
│ 5  2 │ 8  1 │ →  │ 7 │ 8 │
├─────────┤        └─────┘
│ 4  7 │ 6  3 │
│ 2  1 │ 5  8 │
└─────────┘

Average Pooling (2×2, stride=2):
Same input → │ 2.75 │ 3.75 │
             │ 3.50 │ 5.50 │

Global Average Pooling:
H × W × C → 1 × 1 × C (average over all spatial positions)
```

**Pooling vs Strided Convolution:**

| Aspect | Pooling | Strided Conv |
|--------|---------|--------------|
| Parameters | None | Learned |
| Operation | Fixed (max/avg) | Learned |
| Modern preference | Less common | More common |
| Use case | Translation invariance | Learned downsampling |

**Receptive Field Growth:**

```
Layer 0 (input):     RF = 1×1
Layer 1 (3×3 conv):  RF = 3×3
Layer 2 (3×3 conv):  RF = 5×5
Layer 3 (2×2 pool):  RF = 6×6
Layer 4 (3×3 conv):  RF = 10×10
...

Receptive field grows with depth, enabling detection of larger patterns
```

**Spatial Hierarchy:**

```
Input Image (224×224)
    ↓ [Conv + Pool]
Feature Maps (112×112) - edges, colors
    ↓ [Conv + Pool]
Feature Maps (56×56) - textures, gradients
    ↓ [Conv + Pool]
Feature Maps (28×28) - parts, patterns
    ↓ [Conv + Pool]
Feature Maps (14×14) - object parts
    ↓ [Conv + Pool]
Feature Maps (7×7) - whole objects
    ↓ [Global Pool]
Feature Vector (1×1×C) - semantic representation
```

---

### Concept 4: CNN Architecture Evolution

The evolution of CNN architectures reveals key design principles that enable increasingly powerful visual recognition.

**LeNet-5 (1998) - The Pioneer:**

```
Input (32×32×1)
  → Conv(5×5, 6) → Pool(2×2) → 14×14×6
  → Conv(5×5, 16) → Pool(2×2) → 5×5×16
  → FC(120) → FC(84) → Output(10)

Parameters: ~60K
Innovation: First successful CNN for digit recognition
```

**AlexNet (2012) - Deep Learning Era:**

```
Input (224×224×3)
  → Conv(11×11, 96, stride=4) → Pool → 27×27×96
  → Conv(5×5, 256) → Pool → 13×13×256
  → Conv(3×3, 384) → Conv(3×3, 384) → Conv(3×3, 256) → Pool
  → FC(4096) → FC(4096) → Output(1000)

Parameters: ~60M
Innovations: ReLU, Dropout, GPU training, Data augmentation
Impact: Won ImageNet 2012, started deep learning revolution
```

**VGGNet (2014) - Depth Matters:**

```
Design principle: Stack 3×3 convolutions instead of larger kernels

VGG-16 structure:
  [Conv3-64] × 2 → Pool
  [Conv3-128] × 2 → Pool
  [Conv3-256] × 3 → Pool
  [Conv3-512] × 3 → Pool
  [Conv3-512] × 3 → Pool
  → FC(4096) → FC(4096) → Output(1000)

Parameters: ~138M
Insight: Two 3×3 convs = one 5×5 (same RF, fewer params, more nonlinearity)
```

**ResNet (2015) - Skip Connections:**

```
Residual Block:
          ┌──────────────────┐
    x ────┼──→ Conv → BN → ReLU → Conv → BN ──→ (+) ──→ ReLU → out
          └────────────────────────────────────↗
                    (identity shortcut)

ResNet-50 structure:
  Conv(7×7, 64, stride=2) → Pool
  → [Bottleneck(64, 256)] × 3      # Stage 1
  → [Bottleneck(128, 512)] × 4     # Stage 2
  → [Bottleneck(256, 1024)] × 6    # Stage 3
  → [Bottleneck(512, 2048)] × 3    # Stage 4
  → Global Avg Pool → FC(1000)

Parameters: ~25M (less than VGG despite being deeper!)
Innovation: Enabled 100+ layer networks
```

**Architecture Comparison:**

| Model | Year | Layers | Params | Top-5 Error | Key Innovation |
|-------|------|--------|--------|-------------|----------------|
| LeNet | 1998 | 5 | 60K | - | CNN for digits |
| AlexNet | 2012 | 8 | 60M | 15.3% | Deep learning era |
| VGG-16 | 2014 | 16 | 138M | 7.3% | Uniform 3×3 |
| ResNet-50 | 2015 | 50 | 25M | 3.6% | Skip connections |
| EfficientNet-B7 | 2019 | 66 | 66M | 2.9% | Compound scaling |

---

### Concept 5: Residual Learning and Deep Networks

Residual connections fundamentally changed what depth of networks could be trained effectively.

**The Degradation Problem:**

```
Observation (before ResNet):
  20-layer network: 7.5% error
  56-layer network: 8.4% error  ← Deeper is WORSE!

This is not overfitting (training error also higher)
Problem: Optimization difficulty, not representational capacity
```

**Residual Learning Hypothesis:**

```
Standard block: Learn H(x) directly
Residual block: Learn F(x) = H(x) - x, then output F(x) + x

Why easier?
  If optimal H(x) ≈ x (identity), standard block must learn complex identity
  Residual block: F(x) ≈ 0 is trivial (just set weights near zero)

In deep networks, many layers may need to be near-identity
Residual learning makes this easy
```

**Gradient Flow Analysis:**

```
Without residual:
∂L/∂x = ∂L/∂y × ∂H/∂x
Through N layers: ∂L/∂x₀ = ∏ᵢ (∂Hᵢ/∂x)  ← product of many terms

With residual:
∂L/∂x = ∂L/∂y × (∂F/∂x + 1)
Through N layers: ∂L/∂x₀ = ∏ᵢ (∂Fᵢ/∂x + 1)

The "+1" ensures gradient never completely vanishes!
Even if ∂F/∂x = 0, gradient is still 1 (identity path)
```

**Bottleneck Design:**

```
Standard block (for shallow ResNets):
  x → Conv(3×3) → BN → ReLU → Conv(3×3) → BN → (+x) → ReLU

Bottleneck block (for deep ResNets):
  x → Conv(1×1, reduce) → BN → ReLU
    → Conv(3×3) → BN → ReLU
    → Conv(1×1, expand) → BN → (+x) → ReLU

Example: 256 channels
  Standard: 256 → 256 → 256 (3×3×256×256 × 2 = 1.2M params)
  Bottleneck: 256 → 64 → 64 → 256 (64K + 37K + 16K = 117K params)
```

---

### Concept 6: Object Detection Fundamentals

Object detection extends classification to simultaneously identify what objects exist and where they are located.

**Task Definition:**

```
Input: Image (H × W × 3)
Output: List of detections
  - Bounding box: (x, y, width, height) or (x1, y1, x2, y2)
  - Class label: category of object
  - Confidence: detection probability
```

**Intersection over Union (IoU):**

```
IoU = Area of Intersection / Area of Union

  ┌─────────┐
  │    A    │
  │   ┌─────┼───┐
  │   │ A∩B │   │
  └───┼─────┘   │
      │    B    │
      └─────────┘

IoU = |A ∩ B| / |A ∪ B|

IoU thresholds:
  < 0.5: Poor localization
  0.5-0.75: Acceptable
  > 0.75: Good localization
  > 0.9: Excellent
```

**Non-Maximum Suppression (NMS):**

```
Problem: Multiple overlapping detections for same object

NMS Algorithm:
1. Sort detections by confidence
2. Select highest confidence detection
3. Remove all detections with IoU > threshold (0.5) with selected
4. Repeat until no detections remain

Example:
  Detection A: confidence=0.95, box=[10,10,50,50]
  Detection B: confidence=0.85, box=[12,12,48,48]  ← IoU=0.8 with A
  Detection C: confidence=0.75, box=[100,100,150,150]

  After NMS: Keep A and C, remove B (overlaps A)
```

**Anchor Boxes:**

```
Predefined box shapes at each spatial location:
  Scales: [32×32, 64×64, 128×128, 256×256, 512×512]
  Aspect ratios: [1:1, 1:2, 2:1]

Predictions are OFFSETS from anchors:
  pred_x = anchor_x + Δx × anchor_w
  pred_y = anchor_y + Δy × anchor_h
  pred_w = anchor_w × exp(Δw)
  pred_h = anchor_h × exp(Δh)

Why anchors?
  - Easier to predict small offsets than absolute coordinates
  - Different anchors specialize for different object shapes
```

---

### Concept 7: Two-Stage Detectors (R-CNN Family)

Two-stage detectors first propose regions of interest, then classify and refine each proposal.

**R-CNN (2014):**

```
Pipeline:
  1. Selective Search → ~2000 region proposals
  2. For each proposal:
     - Warp to 224×224
     - Extract CNN features (AlexNet)
     - Classify with SVM
     - Regress bounding box

Limitations:
  - 47 seconds per image (2000 CNN forward passes)
  - No end-to-end training
  - Storage for features
```

**Fast R-CNN (2015):**

```
Improvement: Share CNN computation

Pipeline:
  1. Run CNN on entire image → Feature map
  2. Selective Search → Region proposals
  3. RoI Pooling: Extract fixed-size features from variable proposals
  4. Classify + regress each RoI

RoI Pooling:
  Input: Variable-size region on feature map
  Output: Fixed 7×7 feature grid
  Method: Divide region into 7×7 bins, max-pool each bin

Speed: 0.5 seconds per image (vs 47s)
Training: End-to-end for detection (proposals still external)
```

**Faster R-CNN (2016):**

```
Key innovation: Region Proposal Network (RPN)

Architecture:
  Image → Backbone CNN → Feature Map
                            ↓
                    ┌───────┴───────┐
                    ↓               ↓
                   RPN          RoI Pooling
                    ↓               ↓
              Proposals ──────→ Classification
                                   + Box Regression

RPN details:
  - 3×3 conv sliding window on feature map
  - At each position: k anchor boxes (e.g., 9 = 3 scales × 3 ratios)
  - Output per anchor: objectness score (2) + box offsets (4)
  - Total: 6k outputs per position

Speed: ~5 FPS (real-time for many applications)
Training: Fully end-to-end
```

**Feature Pyramid Network (FPN):**

```
Problem: Objects at different scales need features at different resolutions

FPN Architecture:
  Bottom-up pathway: Standard CNN (progressively downsample)
  Top-down pathway: Upsample and merge with skip connections

    P5 (smallest, deepest) ─────────────────────────→ Large objects
     ↑
    (+) ← lateral connection from C5
     ↑
    P4 ────────────────────────────────────────────→ Medium-large
     ↑
    (+) ← lateral connection from C4
     ↑
    P3 ────────────────────────────────────────────→ Medium
     ↑
    P2 (largest, shallowest) ──────────────────────→ Small objects

Detections made at multiple pyramid levels → better scale handling
```

---

### Concept 8: One-Stage Detectors (YOLO, SSD)

One-stage detectors predict bounding boxes and classes directly in a single network pass.

**YOLO (You Only Look Once):**

```
YOLOv1 Design:
  1. Divide image into S×S grid (e.g., 7×7)
  2. Each cell predicts B bounding boxes + confidence
  3. Each cell predicts C class probabilities

  Output tensor: S × S × (B × 5 + C)
    - For S=7, B=2, C=20: 7×7×30

Per-box prediction (5 values):
  - x, y: center offset from cell (0-1)
  - w, h: size relative to image (0-1)
  - confidence: P(object) × IoU

Speed: 45 FPS (real-time!)
Limitation: Each cell predicts only one class → struggles with small/clustered objects
```

**YOLOv3-v8 Improvements:**

```
YOLOv3:
  - Multi-scale predictions (like FPN)
  - Better backbone (Darknet-53)
  - Anchor boxes per scale

YOLOv5/v8:
  - CSP backbone (better gradient flow)
  - PANet neck (bidirectional FPN)
  - Mosaic augmentation
  - Auto-anchor learning

Modern YOLO performance:
  YOLOv8-x: 53.9 mAP @ 640×640, 280 FPS on V100
```

**SSD (Single Shot Detector):**

```
Key insight: Detect at multiple feature map scales

Architecture:
  VGG-16 base → Additional conv layers

  Conv4_3 (38×38) ─→ Detect small objects
  Conv7 (19×19) ───→ Detect medium objects
  Conv8_2 (10×10) ─→ Detect medium-large
  Conv9_2 (5×5) ───→ Detect large
  Conv10_2 (3×3) ──→ Detect larger
  Conv11_2 (1×1) ──→ Detect largest

At each scale: predict boxes + classes for each anchor

Advantages:
  - Multi-scale without FPN complexity
  - 59 FPS on Titan X
```

**Detector Comparison:**

| Model | mAP (COCO) | Speed (FPS) | Best For |
|-------|------------|-------------|----------|
| Faster R-CNN | 42.0 | 5-15 | Accuracy-critical |
| SSD300 | 25.1 | 46 | Balanced |
| YOLOv3 | 33.0 | 30 | Real-time |
| YOLOv8-x | 53.9 | 280 | State-of-the-art |

---

### Concept 9: Semantic and Instance Segmentation

Segmentation provides pixel-level understanding, labeling every pixel with its class or instance.

**Segmentation Types:**

```
Semantic Segmentation:
  - Classify each pixel into categories
  - All "car" pixels get same label
  - Cannot distinguish car-1 from car-2

Instance Segmentation:
  - Classify pixels + separate instances
  - Car-1 pixels separate from car-2 pixels
  - Combines detection with segmentation

Panoptic Segmentation:
  - Unified: "stuff" (sky, road) + "things" (car, person)
  - Complete scene understanding
```

**Fully Convolutional Network (FCN):**

```
Key insight: Replace FC layers with convolutions for dense prediction

FCN Architecture:
  VGG backbone (all conv)
    → 1/32 resolution feature map
    → 1×1 conv for classification
    → Upsample 32× to original resolution

Skip connections improve detail:
  FCN-32s: Upsample 32× directly
  FCN-16s: Combine pool4 + 2× upsample of conv7
  FCN-8s: Combine pool3 + pool4 + conv7

Result: Pixel-level predictions at original resolution
```

**U-Net Architecture:**

```
Encoder-Decoder with Skip Connections:

Encoder (Contracting Path):
  64 → 128 → 256 → 512 → 1024
  Each level: [Conv → Conv → MaxPool]

Decoder (Expanding Path):
  1024 → 512 → 256 → 128 → 64
  Each level: [UpConv → Concat(skip) → Conv → Conv]

     Input (572×572)
         ↓
    ┌────┴────┐ Skip 1
    ↓         │
   Conv      ...
    ↓         │
   Pool       │
    ↓         │
   ...        │
    ↓         │
  Bottleneck  │
    ↓         │
   UpConv ────┘ Concat
    ↓
   Conv
    ↓
   Output (388×388) - per pixel class

Why effective:
  - Encoder captures context (what)
  - Decoder recovers location (where)
  - Skip connections preserve fine details for boundaries
```

**DeepLab with Atrous Convolution:**

```
Atrous (Dilated) Convolution:
  Insert gaps in convolution kernel to increase receptive field
  without increasing parameters or reducing resolution

  Standard 3×3 (dilation=1):  Dilated 3×3 (dilation=2):
    ○○○                         ○ ○ ○
    ○○○
    ○○○                         ○ ○ ○

                                ○ ○ ○

  Effective receptive field: (k-1) × dilation + 1
  Dilation=1: 3×3 RF, Dilation=2: 5×5 RF, Dilation=4: 9×9 RF

ASPP (Atrous Spatial Pyramid Pooling):
  Parallel atrous convs at multiple dilation rates
  → Captures multi-scale context
  → Concatenate for final prediction
```

**Mask R-CNN:**

```
Extension of Faster R-CNN for instance segmentation:

Faster R-CNN output: class + box per RoI
Mask R-CNN output: class + box + binary mask per RoI

Architecture:
  Faster R-CNN pipeline
    + Mask head: Small FCN predicting K binary masks (one per class)
    + RoI Align (instead of RoI Pool) for pixel-accurate masks

RoI Align improvement:
  RoI Pool: Quantizes to integer coordinates → misalignment
  RoI Align: Bilinear interpolation → sub-pixel precision

Result: Per-instance pixel-level masks
```

---

### Concept 10: Modern Computer Vision Advances

Contemporary computer vision incorporates transformers, self-supervised learning, and efficient architectures.

**Vision Transformer (ViT):**

```
Concept: Apply Transformer architecture to images

Approach:
  1. Split image into patches (16×16 pixels)
  2. Flatten patches to vectors (16×16×3 = 768)
  3. Add positional embeddings
  4. Process with Transformer encoder
  5. [CLS] token for classification

224×224 image → 14×14 = 196 patches (+ 1 CLS token)

Architecture:
  Patch Embedding → [Transformer Encoder] × L → MLP Head → Class

Performance:
  - Requires large data (ImageNet-21K pre-training)
  - Matches/exceeds CNNs at scale
  - Better scalability than CNNs
```

**CNN vs ViT:**

| Aspect | CNN | ViT |
|--------|-----|-----|
| Inductive bias | Local, translation equivariant | None (learns from data) |
| Data efficiency | Better with small data | Needs large pre-training |
| Scalability | Saturates at extreme scale | Continues improving |
| Compute | Efficient | O(n²) attention on patches |
| Best regime | < 100M training images | > 100M training images |

**Self-Supervised Learning:**

```
Learn representations without labels:

Contrastive Learning (SimCLR, MoCo):
  - Create augmented views of same image
  - Train: Same image views → similar embeddings
  - Different images → dissimilar embeddings

Masked Image Modeling (MAE):
  - Mask 75% of image patches
  - Reconstruct masked patches from visible ones
  - Learn visual representations through reconstruction

DINO (Self-distillation):
  - Student network learns from teacher network
  - Teacher = exponential moving average of student
  - No labels needed; learns semantic features
```

**Efficient Architectures:**

```
Mobile/Edge Deployment Considerations:
  - Parameters: Memory footprint
  - FLOPs: Computation cost
  - Latency: Actual inference time

MobileNet (Depthwise Separable Convolution):
  Standard conv: k × k × C_in × C_out
  Depthwise: k × k × 1 × C_in (spatial)
  Pointwise: 1 × 1 × C_in × C_out (channel mixing)

  Savings: k² factor fewer parameters

EfficientNet (Compound Scaling):
  Scale depth, width, AND resolution together:
  depth = α^φ, width = β^φ, resolution = γ^φ

  Constraint: α × β² × γ² ≈ 2 (double compute)

  Found optimal: α=1.2, β=1.1, γ=1.15

Model Comparison (ImageNet):
  | Model | Params | Top-1 Acc |
  |-------|--------|-----------|
  | MobileNetV3-S | 2.5M | 67.4% |
  | EfficientNet-B0 | 5.3M | 77.1% |
  | EfficientNet-B7 | 66M | 84.3% |
  | ViT-L/16 | 307M | 85.2% |
```

---

## Summary

Computer Vision enables machines to understand visual data through learned hierarchical representations. **Image preprocessing** (Concept 1) transforms raw pixels into normalized tensors with augmentation for robustness. **Convolution** (Concept 2) provides the fundamental operation for local feature detection with parameter sharing, while **pooling** (Concept 3) builds spatial hierarchy through progressive downsampling.

**CNN architecture evolution** (Concept 4) from LeNet through AlexNet, VGG, and ResNet established key design principles: depth matters, 3×3 convolutions are efficient, and **residual connections** (Concept 5) enable training very deep networks by providing gradient highways.

**Object detection** (Concept 6) extends classification to localization using anchor boxes, IoU metrics, and NMS post-processing. **Two-stage detectors** (Concept 7) like Faster R-CNN achieve high accuracy through region proposal networks, while **one-stage detectors** (Concept 8) like YOLO prioritize speed through direct prediction.

**Segmentation** (Concept 9) provides pixel-level understanding: semantic segmentation classifies every pixel, while instance segmentation additionally separates individual objects using architectures like U-Net, DeepLab, and Mask R-CNN.

**Modern advances** (Concept 10) include Vision Transformers challenging CNN dominance, self-supervised learning reducing label requirements, and efficient architectures enabling edge deployment. Understanding these foundations—from convolution operations through detection frameworks to modern innovations—is essential for building visual AI systems.

---

## References

- LeCun, Y., et al. (1998). "Gradient-Based Learning Applied to Document Recognition" (LeNet)
- Krizhevsky, A., et al. (2012). "ImageNet Classification with Deep Convolutional Neural Networks" (AlexNet)
- Simonyan, K. & Zisserman, A. (2015). "Very Deep Convolutional Networks for Large-Scale Image Recognition" (VGG)
- He, K., et al. (2016). "Deep Residual Learning for Image Recognition" (ResNet)
- Ren, S., et al. (2017). "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"
- Redmon, J., et al. (2016). "You Only Look Once: Unified, Real-Time Object Detection" (YOLO)
- Ronneberger, O., et al. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- Dosovitskiy, A., et al. (2021). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ViT)

---

*Generated from Computer Vision Study Notes | Lesson Skill*
