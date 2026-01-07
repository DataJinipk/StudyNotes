# Flashcard Set: Computer Vision

**Source:** notes/computer-vision/computer-vision-study-notes.md
**Concept Map Reference:** notes/computer-vision/concept-maps/computer-vision-concept-map.md
**Date Generated:** 2026-01-06
**Total Cards:** 5
**Distribution:** 2 Easy | 2 Medium | 1 Hard

---

## Critical Knowledge Summary

Concepts appearing in multiple cards (prioritize for review):
- **Convolution/Feature Maps**: Appears in Cards 1, 2, 3, 4, 5 (foundational operation)
- **Skip Connections**: Appears in Cards 2, 4, 5 (key architectural innovation)
- **Pooling**: Appears in Cards 1, 3 (dimension reduction)
- **Anchor Boxes**: Appears in Cards 3, 5 (detection mechanism)
- **Transfer Learning**: Appears in Cards 2, 5 (practical deployment)

---

## Flashcards

---
### Card 1 | Easy
**Cognitive Level:** Remember/Understand
**Concept:** Convolution, Feature Maps, and Pooling
**Source Section:** Core Concepts 1, 2, 3
**Concept Map Centrality:** Convolution (10), Feature Maps (6), Pooling (5)

**FRONT (Question):**
Explain the convolution operation in CNNs: what inputs does it take, what output does it produce, and how do the key parameters (kernel size, stride, padding) affect the output dimensions?

**BACK (Answer):**
**Convolution Operation:**
```
Input: Feature map (H_in × W_in × C_in)
       + Learnable filter/kernel (K × K × C_in)

Output: Feature map (H_out × W_out × 1 per filter)
```

**How It Works:**
1. Filter slides across input at each spatial position
2. Element-wise multiplication + sum = one output pixel
3. Filter detects specific pattern (edge, texture, shape)
4. Multiple filters → multiple output channels (feature maps)

**Key Parameters:**

| Parameter | Effect on Output | Example |
|-----------|-----------------|---------|
| **Kernel Size (K)** | Receptive field of filter | 3×3 (common), 5×5, 7×7 |
| **Stride (S)** | Step size; S>1 downsamples | S=2 halves dimensions |
| **Padding (P)** | Border handling | "same" preserves size |

**Output Dimension Formula:**
```
H_out = (H_in - K + 2P) / S + 1
W_out = (W_in - K + 2P) / S + 1
```

**Example:**
```
Input: 32×32×3, Filter: 3×3, Stride: 1, Padding: 1
Output: (32 - 3 + 2×1)/1 + 1 = 32×32 (same size)

Input: 32×32×3, Filter: 3×3, Stride: 2, Padding: 0
Output: (32 - 3 + 0)/2 + 1 = 15×15 (downsampled)
```

**Pooling (Dimension Reduction):**
```
Max Pooling 2×2, Stride 2:
┌─────┬─────┐
│ 1 3 │ 2 1 │     ┌───┬───┐
│ 4 2 │ 6 3 │ ──► │ 4 │ 6 │  (takes max in each 2×2)
├─────┼─────┤     └───┴───┘
│ 5 1 │ 2 3 │
│ 3 4 │ 1 2 │
└─────┴─────┘
```

**Critical Knowledge Flag:** Yes - Foundation for all CNN operations

---

---
### Card 2 | Easy
**Cognitive Level:** Understand
**Concept:** ResNet and Skip Connections
**Source Section:** Core Concepts 5, 6
**Concept Map Centrality:** Skip Connections (7), ResNet (6)

**FRONT (Question):**
What problem do skip connections (residual connections) solve, and how does learning F(x) = H(x) - x instead of H(x) directly enable training of very deep networks?

**BACK (Answer):**
**The Problem: Degradation in Deep Networks**
```
Observation (pre-ResNet):
- 20-layer network: good training, good test performance
- 56-layer network: WORSE training, WORSE test performance

This is NOT overfitting (training is also worse).
Deeper networks should be at least as good as shallower ones
(could just copy shallow layers + identity for extra layers).

Something prevents learning identity mappings.
```

**The Solution: Residual Learning**
```
Standard Block:          Residual Block:
      ┌───┐                   ┌───┐
  x ──┤ F ├── H(x)        x ──┤ F ├───┬── F(x) + x
      └───┘               │   └───┘   │
                          └───────────┘ (skip connection)

Learn: H(x)               Learn: F(x) = H(x) - x
                          Output: F(x) + x = H(x)
```

**Why This Works:**

| Aspect | Without Skip | With Skip |
|--------|-------------|-----------|
| **Learning target** | Complete H(x) | Residual F(x) = H(x) - x |
| **Identity mapping** | Must learn W ≈ I (hard) | Just learn F(x) ≈ 0 (easy) |
| **Gradient flow** | Through all weights (can vanish) | Direct path + weight path |
| **Deep networks** | Degrade after ~20 layers | 100+ layers work well |

**Gradient Flow:**
```
During backpropagation:

Standard: ∂L/∂x = ∂L/∂H × ∂H/∂x   (gradients multiply through layers)

Residual: ∂L/∂x = ∂L/∂H × (1 + ∂F/∂x)
                          │
                    Direct path (identity)
                    ensures gradient ≥ original gradient
```

**Key Insight:**
- If additional layers should be identity (no transformation needed)
- Easier to push F(x) → 0 than to learn H(x) = x
- Skip connection provides "shortcut" for gradients
- Network can be arbitrarily deep without degradation

**Architecture Impact:**
```
ResNet-50:  ~25M parameters, 50 layers
ResNet-101: ~44M parameters, 101 layers
ResNet-152: ~60M parameters, 152 layers

All train successfully; deeper = more accurate (up to a point)
```

**Critical Knowledge Flag:** Yes - Enabled the deep learning revolution in vision

---

---
### Card 3 | Medium
**Cognitive Level:** Apply/Analyze
**Concept:** Object Detection: Two-Stage vs. One-Stage
**Source Section:** Core Concepts 7, 8, 9
**Concept Map Centrality:** R-CNN (6), YOLO (5)

**FRONT (Question):**
Compare two-stage detectors (Faster R-CNN) and one-stage detectors (YOLO) architecturally. For a self-driving car application requiring both high accuracy and real-time performance, which would you choose and why?

**BACK (Answer):**
**Two-Stage Architecture (Faster R-CNN):**
```
Stage 1: Region Proposal Network (RPN)
┌─────────────────────────────────────────────────────┐
│ Image → Backbone CNN → Feature Map                  │
│                            ↓                        │
│                    RPN (3×3 conv + classifiers)    │
│                            ↓                        │
│              ~300 Region Proposals (objectness)     │
└─────────────────────────────────────────────────────┘

Stage 2: Classification + Box Refinement
┌─────────────────────────────────────────────────────┐
│ For each proposal:                                  │
│   RoI Pooling → FC layers → Class + BBox           │
│                                                     │
│ Output: Class labels + refined bounding boxes      │
└─────────────────────────────────────────────────────┘
```

**One-Stage Architecture (YOLO):**
```
Single Stage: Direct Prediction
┌─────────────────────────────────────────────────────┐
│ Image → Backbone CNN → Feature Map (e.g., 13×13)   │
│                            ↓                        │
│ Each grid cell predicts:                            │
│   - B bounding boxes (x, y, w, h, confidence)       │
│   - C class probabilities                           │
│                            ↓                        │
│ Output: 13×13×(B×5 + C) tensor                     │
│ (e.g., 13×13×(5×5 + 80) for COCO)                  │
└─────────────────────────────────────────────────────┘
```

**Comparison Table:**

| Aspect | Faster R-CNN | YOLO |
|--------|-------------|------|
| **Speed** | 5-15 FPS | 30-150+ FPS |
| **Accuracy (mAP)** | Higher (~40-45% on COCO) | Slightly lower (~35-43%) |
| **Small objects** | Better (multi-scale proposals) | Historically weaker |
| **Training** | More complex (two stages) | End-to-end, simpler |
| **Memory** | Higher (proposal storage) | Lower |

**Self-Driving Car Recommendation:**

```
Requirements:
- Real-time: 30+ FPS minimum for safety
- High accuracy: Miss = potential accident
- Multiple object types: Vehicles, pedestrians, signs
- Varying sizes: Close cars (large) to distant pedestrians (small)

Recommendation: Modern YOLO (v5/v7/v8) or hybrid approach

Reasoning:
1. Speed requirement eliminates classic Faster R-CNN
2. Modern YOLO versions closed accuracy gap significantly
3. Multi-scale detection (FPN-like) handles size variation
4. Can ensemble multiple YOLO models at different scales

Alternative: Use both
- YOLO for real-time primary detection
- Faster R-CNN for periodic high-accuracy verification
- Critical objects (pedestrians) get extra processing
```

**Key Tradeoffs for Choice:**

| Use Faster R-CNN When | Use YOLO When |
|----------------------|---------------|
| Accuracy is paramount | Real-time is required |
| Offline processing OK | Embedded/edge deployment |
| Small objects critical | Speed > last % accuracy |
| Research/benchmarking | Production systems |

**Critical Knowledge Flag:** Yes - Core decision in detection system design

---

---
### Card 4 | Medium
**Cognitive Level:** Apply/Analyze
**Concept:** Segmentation Architectures
**Source Section:** Core Concepts 10
**Concept Map Centrality:** U-Net (5), Segmentation (5)

**FRONT (Question):**
Design a U-Net architecture for medical image segmentation (512×512 CT scans, binary tumor segmentation). Explain the encoder-decoder structure, the role of skip connections, and why U-Net is particularly effective for medical imaging.

**BACK (Answer):**
**U-Net Architecture:**

```
                          Encoder                    Decoder
                        (Contracting)               (Expanding)

Input: 512×512×1                                    Output: 512×512×1
       ↓                                                   ↑
┌──────────────┐                               ┌──────────────┐
│ Conv 3×3×64  │◄────── Skip Connection ──────►│ Conv 3×3×64  │
│ Conv 3×3×64  │                               │ Conv 3×3×64  │
└──────┬───────┘                               └──────▲───────┘
       │ MaxPool 2×2                           UpConv 2×2 │
       ▼                                                   │
┌──────────────┐                               ┌──────────────┐
│ Conv 3×3×128 │◄────── Skip Connection ──────►│ Conv 3×3×128 │
│ Conv 3×3×128 │                               │ Conv 3×3×128 │
└──────┬───────┘                               └──────▲───────┘
       │ MaxPool 2×2                           UpConv 2×2 │
       ▼                                                   │
┌──────────────┐                               ┌──────────────┐
│ Conv 3×3×256 │◄────── Skip Connection ──────►│ Conv 3×3×256 │
│ Conv 3×3×256 │                               │ Conv 3×3×256 │
└──────┬───────┘                               └──────▲───────┘
       │ MaxPool 2×2                           UpConv 2×2 │
       ▼                                                   │
┌──────────────┐                               ┌──────────────┐
│ Conv 3×3×512 │◄────── Skip Connection ──────►│ Conv 3×3×512 │
│ Conv 3×3×512 │                               │ Conv 3×3×512 │
└──────┬───────┘                               └──────▲───────┘
       │ MaxPool 2×2                           UpConv 2×2 │
       ▼                                                   │
    ┌──────────────────────────────────────────────────────┘
    │                   Bottleneck
    │              ┌──────────────┐
    └─────────────►│ Conv 3×3×1024│
                   │ Conv 3×3×1024│
                   └──────────────┘
```

**Key Components:**

| Component | Purpose | Implementation |
|-----------|---------|----------------|
| **Encoder** | Extract hierarchical features | Conv + Pool (downsample) |
| **Bottleneck** | Deepest features, largest receptive field | 1024 channels at 32×32 |
| **Decoder** | Recover spatial resolution | UpConv (upsample) + Conv |
| **Skip Connections** | Preserve fine-grained details | Concatenate encoder → decoder |
| **Output** | Pixel-wise prediction | 1×1 Conv + Sigmoid |

**Why Skip Connections Are Critical:**

```
Without skip connections:
- Encoder compresses to low resolution (32×32)
- Fine spatial details lost in compression
- Decoder must "imagine" precise boundaries
- Result: Blurry, imprecise segmentation

With skip connections:
- High-resolution features passed directly to decoder
- Decoder combines: semantic info (deep) + spatial info (shallow)
- Result: Precise boundary delineation
```

**Why U-Net Excels in Medical Imaging:**

| Reason | Explanation |
|--------|-------------|
| **Data efficiency** | Works with limited labeled data (expensive to annotate) |
| **Precise boundaries** | Skip connections preserve tumor edges |
| **Full resolution** | Output same size as input; no information loss |
| **Proven architecture** | Years of medical imaging validation |
| **Data augmentation** | Elastic deformations effective for medical data |

**Implementation Details for 512×512 Binary Segmentation:**

```python
# Pseudo-architecture
class UNet(nn.Module):
    def __init__(self):
        # Encoder
        self.enc1 = DoubleConv(1, 64)      # 512→512
        self.enc2 = DoubleConv(64, 128)    # 256→256
        self.enc3 = DoubleConv(128, 256)   # 128→128
        self.enc4 = DoubleConv(256, 512)   # 64→64

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)  # 32→32

        # Decoder (with skip concatenation)
        self.dec4 = DoubleConv(1024+512, 512)
        self.dec3 = DoubleConv(512+256, 256)
        self.dec2 = DoubleConv(256+128, 128)
        self.dec1 = DoubleConv(128+64, 64)

        # Output
        self.out = nn.Conv2d(64, 1, 1)  # 1×1 conv
        # + Sigmoid for binary segmentation

# Loss: Binary Cross-Entropy + Dice Loss (handles class imbalance)
```

**Critical Knowledge Flag:** Yes - Dominant architecture for medical segmentation

---

---
### Card 5 | Hard
**Cognitive Level:** Evaluate/Synthesize
**Concept:** Complete Computer Vision Pipeline
**Source Section:** All Core Concepts
**Concept Map Centrality:** Integrates all high-centrality nodes

**FRONT (Question):**
Design a complete computer vision system for an autonomous drone that must: (1) detect and classify ground vehicles in real-time (cars, trucks, motorcycles), (2) estimate distance to each vehicle, and (3) operate on embedded GPU with 8GB memory and 30 FPS requirement. Address architecture selection, training strategy, and deployment optimization.

**BACK (Answer):**
**System Architecture Overview:**

```
┌─────────────────────────────────────────────────────────────┐
│                    Drone Vision Pipeline                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌────────┐ │
│  │ Camera  │───►│Preprocess│───►│ Backbone │───►│  Head  │ │
│  │ 720p    │    │ Resize   │    │ Efficient│    │ Detect │ │
│  │ 30 FPS  │    │ Normalize│    │   Net    │    │  +     │ │
│  └─────────┘    └──────────┘    └──────────┘    │ Depth  │ │
│                                                  └────┬───┘ │
│                                                       │     │
│                                               ┌───────▼───┐ │
│                                               │   Output  │ │
│                                               │ BBox+Class│ │
│                                               │ +Distance │ │
│                                               └───────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**1. Architecture Selection:**

```
Backbone: EfficientNet-B0 or MobileNetV3
- Designed for edge devices
- Good accuracy/efficiency tradeoff
- ~5M parameters (fits in 8GB with room for other components)

Detection Head: YOLOv5-small or YOLOv8-nano
- One-stage for real-time
- Multi-scale detection for varying vehicle sizes
- ~7M additional parameters

Depth Estimation: Shared backbone + lightweight depth head
- Leverage same features as detection
- MLP or small decoder for depth regression
```

**Multi-Task Architecture:**

```
                    Input: 640×480 RGB
                           ↓
              ┌────────────────────────┐
              │    EfficientNet-B0     │
              │    (Shared Backbone)   │
              └───────────┬────────────┘
                          │
           ┌──────────────┼──────────────┐
           ↓              ↓              ↓
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ Detection│   │   Depth  │   │ (Future) │
    │   Head   │   │   Head   │   │ Tracking │
    └────┬─────┘   └────┬─────┘   └──────────┘
         │              │
         ▼              ▼
    BBox + Class    Depth Map
    (per vehicle)   (or per-box depth)
```

**2. Training Strategy:**

```
Phase 1: Transfer Learning
- Start with ImageNet-pretrained EfficientNet
- Freeze backbone, train heads on drone vehicle dataset
- Learning rate: 1e-3 for heads

Phase 2: Fine-tuning
- Unfreeze backbone, train end-to-end
- Lower learning rate: 1e-4 for backbone, 1e-3 for heads
- Data augmentation: perspective transforms (simulate altitude changes)

Phase 3: Joint Training
- Multi-task loss: L = L_detect + λ × L_depth
- Balance detection and depth with λ (tune on validation)

Data Requirements:
- Detection: ~5000 labeled aerial images with vehicle boxes
- Depth: Synthetic data (flight simulators) + sparse real depth (LiDAR)
- Augmentation: Rotation, scale, haze, lighting variation
```

**Loss Functions:**

```python
# Detection Loss (YOLO-style)
L_detect = L_box + L_objectness + L_class
         = CIoU_loss + BCE_loss + BCE_loss

# Depth Loss (per detected vehicle)
L_depth = SmoothL1(predicted_depth, ground_truth_depth)

# Total Multi-Task Loss
L_total = L_detect + 0.5 × L_depth
```

**3. Deployment Optimization:**

```
Memory Budget (8GB):
- Model weights (FP16): ~25MB
- Input buffer: 640×480×3 × 4 bytes = ~3.7MB
- Feature maps: ~200MB (peak during inference)
- Depth output: ~1MB
- Detection output: ~1MB
- Margin for runtime: ~7.5GB available

Speed Optimizations:

| Technique | Speedup | Trade-off |
|-----------|---------|-----------|
| TensorRT optimization | 2-3x | NVIDIA-specific |
| FP16 inference | 1.5-2x | Minimal accuracy loss |
| INT8 quantization | 2-3x | Requires calibration |
| Input resolution (640→480) | 1.4x | May miss small vehicles |
| Batch size = 1 | Optimal for real-time | No batching benefit |

Target: 640×480 input, ~33ms inference = 30 FPS ✓
```

**4. Distance Estimation Approaches:**

```
Option A: Monocular Depth Network
- Train depth head on synthetic + real data
- Output: Relative depth map
- Convert to absolute using camera calibration

Option B: Geometry-Based (if vehicle size known)
- Apparent size in pixels → distance
- Known truck width (2.5m) + detected box width → depth
- Simple, interpretable, no training needed

Option C: Hybrid
- Use detection box for coarse depth (geometry)
- Refine with learned depth features
- Most robust approach

Chosen: Hybrid approach
- Fast geometric estimate from detection
- Learned refinement when accuracy critical
```

**5. Production Considerations:**

```
Safety & Reliability:
- Confidence thresholding: Only report detections > 0.7
- Temporal smoothing: Track detections across frames
- Fallback: If detection fails, use previous frame + motion model
- Monitoring: Log inference time, detection count per frame

Edge Cases:
- Occlusion: Multiple overlapping vehicles
- Scale: Vehicles at varying altitudes (need multi-scale)
- Weather: Train with augmented fog, rain, sun glare
- Night: May need thermal camera input (separate model)

Continuous Improvement:
- Log edge cases for retraining
- A/B test model updates in simulation
- Gradual rollout with monitoring
```

**Summary Architecture:**

```
Input: 640×480 RGB @ 30 FPS
Model: EfficientNet-B0 + YOLO-style head + Depth MLP
Output: Up to 100 detections/frame (class, box, depth)
Latency: ~30ms on Jetson Xavier NX (with TensorRT)
Memory: ~1GB model + runtime
Accuracy: ~85% mAP for vehicles, ~10% depth error
```

**Critical Knowledge Flag:** Yes - Integrates detection, depth, deployment optimization

---

---

## Export Formats

### Anki-Compatible (Tab-Separated)
```
Explain convolution operation with kernel, stride, padding	Filter slides over input, element-wise multiply + sum. Output: (H-K+2P)/S+1. Kernel=pattern detector, Stride=step size (>1 downsamples), Padding=border handling (same preserves size).	easy::convolution::cv
What problem do skip connections solve in deep networks?	Degradation: deeper networks train worse. Skip connections learn residual F(x)=H(x)-x, output F(x)+x. Identity mapping easy (push F→0). Direct gradient path prevents vanishing.	easy::resnet::cv
Compare Faster R-CNN vs YOLO for self-driving	Faster R-CNN: two-stage, 5-15 FPS, higher accuracy. YOLO: one-stage, 30-150 FPS, real-time. For self-driving: modern YOLO (v5+) provides both speed and accuracy needed.	medium::detection::cv
Design U-Net for medical image segmentation	Encoder (downsample) → Bottleneck → Decoder (upsample). Skip connections concatenate encoder→decoder features. Preserves precise boundaries. Works with limited labeled data.	medium::segmentation::cv
Design drone vision system with detection + depth	EfficientNet backbone + YOLO head + depth MLP. Multi-task training. TensorRT + FP16 for 30 FPS on 8GB GPU. Hybrid depth: geometry + learned refinement.	hard::system::cv
```

### CSV Format
```csv
"Front","Back","Difficulty","Concept","Centrality"
"Convolution operation mechanics?","Filter slides, multiply-sum. Output=(H-K+2P)/S+1. Stride>1 downsamples.","Easy","Foundations","Critical"
"Why skip connections?","Solve degradation. Learn residual F(x), output F(x)+x. Easy identity, direct gradients.","Easy","Architecture","Critical"
"Faster R-CNN vs YOLO?","Two-stage vs one-stage. Accuracy vs speed. Modern YOLO closes gap.","Medium","Detection","High"
"U-Net for medical imaging?","Encoder-decoder with skip connections. Preserves boundaries. Data efficient.","Medium","Segmentation","High"
"Drone vision system?","EfficientNet+YOLO+depth. Multi-task. TensorRT optimization.","Hard","System Design","Integration"
```

---

## Source Mapping

| Card | Source Sections | Concept Map Nodes | Key Terms |
|------|-----------------|-------------------|-----------|
| 1 | Concepts 1, 2, 3 | Convolution, Feature Maps, Pooling | Kernel, stride, padding, max pooling |
| 2 | Concepts 5, 6 | ResNet, Skip Connections | Residual, identity, gradient flow |
| 3 | Concepts 7, 8, 9 | R-CNN, YOLO, Detection | Anchor, IoU, NMS, real-time |
| 4 | Concept 10 | U-Net, Segmentation | Encoder, decoder, semantic |
| 5 | All concepts | Full integration | Multi-task, deployment, optimization |
