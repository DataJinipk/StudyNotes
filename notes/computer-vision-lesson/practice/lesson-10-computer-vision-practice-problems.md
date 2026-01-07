# Practice Problems: Lesson 10 - Computer Vision

**Source:** Lessons/Lesson_10.md
**Subject Area:** AI Learning - Computer Vision: From Image Processing to Deep Learning Visual Understanding
**Date Generated:** 2026-01-08
**Total Problems:** 5

---

## Problem Distribution

| Difficulty | Count | Type | Focus Area |
|------------|-------|------|------------|
| Warm-Up | 1 | Direct concept application | Convolution computation |
| Skill-Builder | 2 | Multi-step procedural | IoU/NMS, receptive field |
| Challenge | 1 | Complex synthesis | Detection system design |
| Debug/Fix | 1 | Identify and correct errors | CNN architecture debugging |

---

## Problem 1: Convolution Output Computation (Warm-Up)

**Difficulty:** Warm-Up
**Estimated Time:** 15 minutes
**Concepts:** Convolution operation, output dimensions, feature maps

### Problem Statement

Given the following input and convolution parameters, compute the output feature map.

**Input (6×6 grayscale image):**
```
┌─────────────────────────┐
│  1   2   3   0   1   2  │
│  0   1   2   3   0   1  │
│  1   0   1   2   3   0  │
│  2   1   0   1   2   3  │
│  3   2   1   0   1   2  │
│  0   3   2   1   0   1  │
└─────────────────────────┘
```

**Convolution Parameters:**
- Kernel size: 3×3
- Stride: 2
- Padding: 0 (valid)
- Number of filters: 1

**Kernel (edge detector):**
```
┌───────────┐
│ -1  0  1  │
│ -2  0  2  │
│ -1  0  1  │
└───────────┘
```

**Tasks:**
a) Calculate the output dimensions
b) Compute the complete output feature map
c) What visual feature does this kernel detect?
d) How many parameters does this convolution layer have?

---

### Hints

<details>
<summary>Hint 1 (Dimensions)</summary>
Output size formula: ((Input - Kernel + 2×Padding) / Stride) + 1
For this problem: ((6 - 3 + 0) / 2) + 1 = 2
Output is 2×2.
</details>

<details>
<summary>Hint 2 (Computation)</summary>
For each output position, multiply the kernel element-wise with the input region, then sum all products.
First output position uses input[0:3, 0:3].
</details>

<details>
<summary>Hint 3 (Pattern Recognition)</summary>
This is a Sobel filter variant. Look at the kernel: negative on left, positive on right. It responds to horizontal changes (vertical edges).
</details>

---

### Solution

**a) Output Dimensions:**

```
Formula: O = ((I - K + 2P) / S) + 1

Where:
  I = Input size = 6
  K = Kernel size = 3
  P = Padding = 0
  S = Stride = 2

O = ((6 - 3 + 0) / 2) + 1 = (3 / 2) + 1 = 1.5 + 1 = 2 (floor)

Output dimensions: 2 × 2 × 1 (one filter)
```

**b) Complete Output Feature Map:**

**Position (0,0): Input region [0:3, 0:3]**
```
Input:          Kernel:
┌─────────┐     ┌───────────┐
│ 1  2  3 │     │ -1  0  1  │
│ 0  1  2 │  *  │ -2  0  2  │
│ 1  0  1 │     │ -1  0  1  │
└─────────┘     └───────────┘

= (1×-1) + (2×0) + (3×1) + (0×-2) + (1×0) + (2×2) + (1×-1) + (0×0) + (1×1)
= -1 + 0 + 3 + 0 + 0 + 4 + -1 + 0 + 1
= 6
```

**Position (0,1): Input region [0:3, 2:5]** (stride=2, so start at column 2)
```
Input:          Kernel:
┌─────────┐     ┌───────────┐
│ 3  0  1 │     │ -1  0  1  │
│ 2  3  0 │  *  │ -2  0  2  │
│ 1  2  3 │     │ -1  0  1  │
└─────────┘     └───────────┘

= (3×-1) + (0×0) + (1×1) + (2×-2) + (3×0) + (0×2) + (1×-1) + (2×0) + (3×1)
= -3 + 0 + 1 + -4 + 0 + 0 + -1 + 0 + 3
= -4
```

**Position (1,0): Input region [2:5, 0:3]** (stride=2, so start at row 2)
```
Input:          Kernel:
┌─────────┐     ┌───────────┐
│ 1  0  1 │     │ -1  0  1  │
│ 2  1  0 │  *  │ -2  0  2  │
│ 3  2  1 │     │ -1  0  1  │
└─────────┘     └───────────┘

= (1×-1) + (0×0) + (1×1) + (2×-2) + (1×0) + (0×2) + (3×-1) + (2×0) + (1×1)
= -1 + 0 + 1 + -4 + 0 + 0 + -3 + 0 + 1
= -6
```

**Position (1,1): Input region [2:5, 2:5]**
```
Input:          Kernel:
┌─────────┐     ┌───────────┐
│ 1  2  3 │     │ -1  0  1  │
│ 0  1  2 │  *  │ -2  0  2  │
│ 1  0  1 │     │ -1  0  1  │
└─────────┘     └───────────┘

= (1×-1) + (2×0) + (3×1) + (0×-2) + (1×0) + (2×2) + (1×-1) + (0×0) + (1×1)
= -1 + 0 + 3 + 0 + 0 + 4 + -1 + 0 + 1
= 6
```

**Final Output Feature Map:**
```
┌─────────┐
│  6  -4  │
│ -6   6  │
└─────────┘
```

**c) Visual Feature Detected:**

This is a **Sobel-X filter** (vertical edge detector):
- Kernel has negative weights on left, positive on right
- Computes horizontal gradient (change in x-direction)
- **Detects vertical edges** (transitions from dark-to-light or light-to-dark horizontally)

```
Interpretation:
- Positive output (6): Light on right, dark on left (→ gradient)
- Negative output (-6): Dark on right, light on left (← gradient)
- Near-zero output: No horizontal change
```

**d) Parameter Count:**

```
Kernel parameters: 3 × 3 = 9 weights
Bias: 1 (one per filter)
Total: 9 + 1 = 10 parameters

For C_in input channels and C_out filters:
Parameters = K × K × C_in × C_out + C_out (biases)
This example: 3 × 3 × 1 × 1 + 1 = 10
```

---

## Problem 2: IoU and NMS Computation (Skill-Builder)

**Difficulty:** Skill-Builder
**Estimated Time:** 25 minutes
**Concepts:** IoU calculation, Non-Maximum Suppression, detection post-processing

### Problem Statement

An object detector produces the following detections for a single image. Apply NMS to produce the final output.

**Detections (format: [x1, y1, x2, y2], confidence, class):**

| Detection | Box Coordinates | Confidence | Class |
|-----------|----------------|------------|-------|
| A | [10, 10, 50, 50] | 0.95 | car |
| B | [12, 8, 48, 52] | 0.88 | car |
| C | [100, 100, 150, 140] | 0.80 | car |
| D | [15, 15, 45, 48] | 0.75 | car |
| E | [105, 95, 155, 145] | 0.70 | car |
| F | [200, 200, 250, 250] | 0.65 | person |

**NMS Parameters:**
- IoU threshold: 0.5
- Apply NMS per class

**Tasks:**
a) Calculate IoU between detections A and B
b) Calculate IoU between detections A and D
c) Calculate IoU between detections C and E
d) Apply NMS and list the final detections kept
e) Explain why detection F is not affected by NMS with other detections

---

### Hints

<details>
<summary>Hint 1 (IoU Formula)</summary>
IoU = Intersection Area / Union Area
Union = Area_A + Area_B - Intersection
Intersection = max(0, min(x2_A, x2_B) - max(x1_A, x1_B)) × max(0, min(y2_A, y2_B) - max(y1_A, y1_B))
</details>

<details>
<summary>Hint 2 (NMS Process)</summary>
1. Sort by confidence (descending)
2. Pick highest confidence, add to output
3. Remove all boxes with IoU > threshold with picked box
4. Repeat until no boxes remain
</details>

<details>
<summary>Hint 3 (Per-Class NMS)</summary>
NMS is applied separately per class. A "car" detection never suppresses a "person" detection regardless of overlap.
</details>

---

### Solution

**a) IoU between A and B:**

```
Box A: [10, 10, 50, 50]  → Area_A = (50-10) × (50-10) = 40 × 40 = 1600
Box B: [12, 8, 48, 52]   → Area_B = (48-12) × (52-8) = 36 × 44 = 1584

Intersection:
  x_left = max(10, 12) = 12
  x_right = min(50, 48) = 48
  y_top = max(10, 8) = 10
  y_bottom = min(50, 52) = 50

  width = 48 - 12 = 36
  height = 50 - 10 = 40
  Intersection = 36 × 40 = 1440

Union = 1600 + 1584 - 1440 = 1744

IoU(A,B) = 1440 / 1744 = 0.826 ≈ 0.83
```

**b) IoU between A and D:**

```
Box A: [10, 10, 50, 50]  → Area_A = 1600
Box D: [15, 15, 45, 48]  → Area_D = (45-15) × (48-15) = 30 × 33 = 990

Intersection:
  x_left = max(10, 15) = 15
  x_right = min(50, 45) = 45
  y_top = max(10, 15) = 15
  y_bottom = min(50, 48) = 48

  width = 45 - 15 = 30
  height = 48 - 15 = 33
  Intersection = 30 × 33 = 990

Union = 1600 + 990 - 990 = 1600

IoU(A,D) = 990 / 1600 = 0.619 ≈ 0.62
```

**c) IoU between C and E:**

```
Box C: [100, 100, 150, 140] → Area_C = 50 × 40 = 2000
Box E: [105, 95, 155, 145]  → Area_E = 50 × 50 = 2500

Intersection:
  x_left = max(100, 105) = 105
  x_right = min(150, 155) = 150
  y_top = max(100, 95) = 100
  y_bottom = min(140, 145) = 140

  width = 150 - 105 = 45
  height = 140 - 100 = 40
  Intersection = 45 × 40 = 1800

Union = 2000 + 2500 - 1800 = 2700

IoU(C,E) = 1800 / 2700 = 0.667 ≈ 0.67
```

**d) NMS Application:**

```
Step 1: Separate by class
  Car: A(0.95), B(0.88), C(0.80), D(0.75), E(0.70)
  Person: F(0.65)

Step 2: NMS for "car" class
  Sort by confidence: A(0.95) > B(0.88) > C(0.80) > D(0.75) > E(0.70)

  Iteration 1:
    - Select A (highest confidence) → Output: [A]
    - Check IoU(A, B) = 0.83 > 0.5 → Remove B
    - Check IoU(A, C) ≈ 0 (no overlap) → Keep C
    - Check IoU(A, D) = 0.62 > 0.5 → Remove D
    - Check IoU(A, E) ≈ 0 (no overlap) → Keep E
    Remaining: C(0.80), E(0.70)

  Iteration 2:
    - Select C (highest remaining) → Output: [A, C]
    - Check IoU(C, E) = 0.67 > 0.5 → Remove E
    Remaining: (none)

  Car detections: [A, C]

Step 3: NMS for "person" class
  Only F remains → Output: [F]
  (No suppression needed)

Final Output: [A, C, F]
```

**Summary Table:**

| Detection | Confidence | Kept/Removed | Reason |
|-----------|------------|--------------|--------|
| A | 0.95 | **Kept** | Highest car confidence |
| B | 0.88 | Removed | IoU(A,B)=0.83 > 0.5 |
| C | 0.80 | **Kept** | No overlap with A |
| D | 0.75 | Removed | IoU(A,D)=0.62 > 0.5 |
| E | 0.70 | Removed | IoU(C,E)=0.67 > 0.5 |
| F | 0.65 | **Kept** | Only person detection |

**e) Why F is Not Affected:**

```
Detection F is class "person", while A-E are class "car".

NMS is applied PER CLASS:
- Car detections only compete with other car detections
- Person detections only compete with other person detections

Even if F overlapped spatially with car detections, it would not be suppressed
because they are different classes. This allows the detector to output
overlapping objects of different types (e.g., person in car, dog on chair).
```

---

## Problem 3: Receptive Field Calculation (Skill-Builder)

**Difficulty:** Skill-Builder
**Estimated Time:** 25 minutes
**Concepts:** Receptive field, CNN depth, spatial hierarchy

### Problem Statement

Calculate the receptive field for a CNN with the following architecture, and analyze its implications for object detection.

**Architecture:**
```
Layer 1: Conv 3×3, stride 1, padding 1
Layer 2: Conv 3×3, stride 1, padding 1
Layer 3: MaxPool 2×2, stride 2
Layer 4: Conv 3×3, stride 1, padding 1
Layer 5: Conv 3×3, stride 1, padding 1
Layer 6: MaxPool 2×2, stride 2
Layer 7: Conv 3×3, stride 1, padding 1
```

**Tasks:**
a) Calculate the receptive field after each layer
b) What is the final receptive field in pixels?
c) If the input image is 224×224, what is the spatial size at each layer?
d) This network is used for detecting objects. What is the minimum object size it can reliably detect? What objects might it struggle with?

---

### Hints

<details>
<summary>Hint 1 (RF Formula)</summary>
For each layer: RF_new = RF_prev + (kernel_size - 1) × stride_product
Where stride_product is the cumulative stride of all previous layers.
</details>

<details>
<summary>Hint 2 (Stride Accumulation)</summary>
Each pooling layer doubles the effective stride. After two 2×2 pools, the cumulative stride is 4.
</details>

<details>
<summary>Hint 3 (Object Detection)</summary>
Objects should be larger than the receptive field for reliable detection. Very small objects (smaller than RF) may be missed or poorly localized.
</details>

---

### Solution

**a) Receptive Field After Each Layer:**

```
RF Formula: RF_l = RF_{l-1} + (k - 1) × j_{l-1}

Where:
  RF_l = receptive field at layer l
  k = kernel size at layer l
  j_{l-1} = cumulative stride before layer l (called "jump")

Initial: RF_0 = 1, j_0 = 1

Layer 1 (Conv 3×3, s=1):
  j_1 = j_0 × 1 = 1
  RF_1 = RF_0 + (3-1) × j_0 = 1 + 2 × 1 = 3

Layer 2 (Conv 3×3, s=1):
  j_2 = j_1 × 1 = 1
  RF_2 = RF_1 + (3-1) × j_1 = 3 + 2 × 1 = 5

Layer 3 (MaxPool 2×2, s=2):
  j_3 = j_2 × 2 = 2
  RF_3 = RF_2 + (2-1) × j_2 = 5 + 1 × 1 = 6

Layer 4 (Conv 3×3, s=1):
  j_4 = j_3 × 1 = 2
  RF_4 = RF_3 + (3-1) × j_3 = 6 + 2 × 2 = 10

Layer 5 (Conv 3×3, s=1):
  j_5 = j_4 × 1 = 2
  RF_5 = RF_4 + (3-1) × j_4 = 10 + 2 × 2 = 14

Layer 6 (MaxPool 2×2, s=2):
  j_6 = j_5 × 2 = 4
  RF_6 = RF_5 + (2-1) × j_5 = 14 + 1 × 2 = 16

Layer 7 (Conv 3×3, s=1):
  j_7 = j_6 × 1 = 4
  RF_7 = RF_6 + (3-1) × j_6 = 16 + 2 × 4 = 24
```

**Summary Table:**

| Layer | Operation | Kernel | Stride | Cum. Stride (j) | Receptive Field |
|-------|-----------|--------|--------|-----------------|-----------------|
| Input | - | - | - | 1 | 1×1 |
| 1 | Conv | 3×3 | 1 | 1 | 3×3 |
| 2 | Conv | 3×3 | 1 | 1 | 5×5 |
| 3 | MaxPool | 2×2 | 2 | 2 | 6×6 |
| 4 | Conv | 3×3 | 1 | 2 | 10×10 |
| 5 | Conv | 3×3 | 1 | 2 | 14×14 |
| 6 | MaxPool | 2×2 | 2 | 4 | 16×16 |
| 7 | Conv | 3×3 | 1 | 4 | **24×24** |

**b) Final Receptive Field: 24×24 pixels**

**c) Spatial Size at Each Layer (224×224 input):**

```
Input: 224 × 224
Layer 1 (Conv 3×3, s=1, p=1): (224 - 3 + 2)/1 + 1 = 224 × 224
Layer 2 (Conv 3×3, s=1, p=1): 224 × 224
Layer 3 (MaxPool 2×2, s=2): 224 / 2 = 112 × 112
Layer 4 (Conv 3×3, s=1, p=1): 112 × 112
Layer 5 (Conv 3×3, s=1, p=1): 112 × 112
Layer 6 (MaxPool 2×2, s=2): 112 / 2 = 56 × 56
Layer 7 (Conv 3×3, s=1, p=1): 56 × 56

Final feature map: 56 × 56 × C (where C = number of filters)
```

**d) Object Detection Implications:**

```
Minimum Reliable Object Size: ~24×24 pixels (receptive field size)

Objects larger than 24×24:
  ✓ Full object fits in receptive field
  ✓ Can detect shape, texture, parts
  ✓ Reliable detection and localization

Objects smaller than 24×24:
  ⚠ Object smaller than RF means each feature position
    sees context beyond the object
  ⚠ May detect but poor localization
  ⚠ Features contaminated by background

Objects much smaller (< 12×12):
  ✗ Object is small fraction of RF
  ✗ Likely to miss or false positive
  ✗ Need higher resolution or FPN

Practical Guideline:
  - Objects should be 1-3× the receptive field for optimal detection
  - For this network: 24-72 pixel objects detected best
  - For 224×224 input: objects covering 10-30% of image

Struggle cases:
  - Small objects (< 24 pixels): Distant people, small vehicles
  - Very large objects (> 150 pixels): Only parts visible in RF
  - Solution: Multi-scale features (FPN) or higher resolution
```

---

## Problem 4: Object Detection System Design (Challenge)

**Difficulty:** Challenge
**Estimated Time:** 40 minutes
**Concepts:** Complete detection pipeline, architecture selection, deployment

### Problem Statement

Design an object detection system for a **drone-based wildlife monitoring** application with the following requirements:

**Requirements:**
1. Detect and classify 20 animal species from aerial footage
2. Process 4K video (3840×2160) at 10+ FPS
3. Handle objects ranging from 50 to 2000 pixels
4. Detect partially occluded animals (e.g., in vegetation)
5. Run on NVIDIA Jetson AGX Orin (275 TOPS INT8)
6. Minimize false positives (avoid disturbing animals unnecessarily)

**Design a complete system specifying:**
a) Input preprocessing and resolution strategy
b) Detector architecture and backbone selection
c) Multi-scale detection approach
d) Post-processing and tracking strategy
e) Optimization for edge deployment
f) Training data requirements and augmentation strategy

---

### Hints

<details>
<summary>Hint 1 (Resolution Tradeoff)</summary>
4K is too large for direct processing. Consider tiled processing, intelligent cropping, or resolution reduction with multi-scale detection.
</details>

<details>
<summary>Hint 2 (Multi-Scale)</summary>
50-2000 pixel range is huge (40× difference). Need strong multi-scale features—FPN is essential. Consider detecting at multiple input resolutions.
</details>

<details>
<summary>Hint 3 (Edge Optimization)</summary>
Jetson AGX Orin has excellent INT8 performance. TensorRT optimization with INT8 quantization can provide 3-5× speedup over FP32.
</details>

---

### Solution

**Wildlife Drone Detection System Design:**

**a) Input Preprocessing and Resolution Strategy:**

```yaml
resolution_strategy:
  problem: 4K (3840×2160) is too large for real-time detection
  solution: Adaptive tiled processing

  approach:
    step_1_coarse_scan:
      - Downsample 4K → 960×540 (4× reduction)
      - Run lightweight detector for region proposals
      - Identify regions of interest (potential animals)

    step_2_fine_detection:
      - Crop ROIs at full 4K resolution
      - Max crop size: 1280×1280 per ROI
      - Run main detector on crops

    fallback:
      - If no ROIs found, tile full frame (3×2 grid)
      - Process tiles at 1280×720 each

  preprocessing:
    normalization: ImageNet mean/std
    color_space: RGB (aerial images benefit from color)
    augmentation_at_inference: None (deterministic)
```

**b) Detector Architecture Selection:**

```yaml
detector: YOLOv8-medium with FPN

rationale:
  - YOLOv8: Best accuracy/speed tradeoff for real-time
  - Medium variant: Balance between nano (too weak) and large (too slow)
  - FPN: Essential for 50-2000 pixel range

backbone: CSPDarknet53
  - Strong feature extraction
  - Efficient gradient flow
  - Good for INT8 quantization

neck: PANet (Path Aggregation Network)
  - Bidirectional FPN
  - Better small object features

head: Decoupled head
  - Separate classification and localization
  - Better for diverse object sizes

detection_scales:
  P3 (80×80): Small objects (50-150 pixels)
  P4 (40×40): Medium objects (150-500 pixels)
  P5 (20×20): Large objects (500-2000 pixels)
```

**c) Multi-Scale Detection Approach:**

```yaml
multi_scale_strategy:
  challenge: 40× size variation (50 to 2000 pixels)

  approach_1_fpn_scales:
    - P3 (stride 8): Detects objects 32-128 pixels
    - P4 (stride 16): Detects objects 64-256 pixels
    - P5 (stride 32): Detects objects 128-512 pixels
    - P6 (stride 64): Detects objects 256-1024 pixels (added)

  approach_2_input_scaling:
    - Process each ROI at multiple scales: [0.5×, 1.0×, 2.0×]
    - Merge detections with scale-aware NMS

  approach_3_attention_for_small:
    - Add CBAM (Convolutional Block Attention) to P3
    - Helps focus on small animal features

  anchor_design:
    - K-means clustering on training data
    - Per-scale anchors optimized for wildlife shapes
    - Ratios: [0.5, 1.0, 2.0] (animals vary in shape)
```

**d) Post-Processing and Tracking:**

```yaml
post_processing:
  nms:
    iou_threshold: 0.5
    confidence_threshold: 0.4 (adjustable)
    soft_nms: True (better for occlusion)

  false_positive_reduction:
    - Minimum detection size: 40 pixels (filter noise)
    - Aspect ratio constraints: 0.2 < h/w < 5.0
    - Temporal consistency: Require 2+ frames for alert
    - Confidence boosting: Higher threshold in known empty areas

tracking:
  algorithm: ByteTrack
    - Good for varying speeds (stationary to running)
    - Handles re-appearance after occlusion
    - Low computational overhead

  track_management:
    - New track: 3 consecutive detections required
    - Lost track: Keep 30 frames before deletion
    - Re-ID: Appearance features for long-term tracking

  output:
    - Track ID, species, bounding box sequence
    - Movement vector (for behavior analysis)
    - Confidence history
```

**e) Edge Deployment Optimization:**

```yaml
jetson_optimization:
  target_device: Jetson AGX Orin (275 TOPS INT8)

  tensorrt_optimization:
    precision: INT8 (with FP16 accumulation)
    calibration: 1000 representative images
    expected_speedup: 3-4× over FP32

  model_optimization:
    - Export to ONNX, then TensorRT engine
    - Fuse BatchNorm into Conv layers
    - Use NHWC format (better for Orin)

  memory_management:
    - Input buffer pool (avoid allocation)
    - Pinned memory for CPU-GPU transfer
    - Stream processing (overlap compute/transfer)

  performance_budget:
    coarse_scan (960×540): ~2ms
    fine_detection (1280×1280): ~25ms per ROI
    tracking: ~1ms
    total_budget: <100ms (10+ FPS)

  power_profile:
    mode: MAXN (maximum performance)
    expected_power: ~60W
    thermal: Active cooling required
```

**f) Training Data Requirements:**

```yaml
data_requirements:
  minimum_images: 50,000+ annotated
  per_class: 2,500+ instances (20 classes)

  data_sources:
    - Existing wildlife datasets (iNaturalist, LILA)
    - Drone footage collection campaigns
    - Synthetic augmentation with cut-paste

  annotation:
    format: COCO (bounding boxes)
    quality: Double-annotated, expert review
    attributes: Species, pose (standing/lying), occlusion level

augmentation_strategy:
  geometric:
    - Random rotation (drone viewing angles vary)
    - Random scale (simulates altitude changes)
    - Random crop (simulates partial visibility)

  photometric:
    - Brightness/contrast (lighting variation)
    - Hue shift (seasonal vegetation changes)
    - Blur (motion blur from drone movement)

  domain-specific:
    - Copy-paste augmentation (place animals on backgrounds)
    - Mosaic augmentation (4 images combined)
    - Fog/haze simulation (weather conditions)

  hard_negative_mining:
    - Collect false positive examples (rocks, shadows)
    - Add to training with "background" label
    - Iterate to reduce false positives

training_schedule:
  epochs: 300
  optimizer: SGD with momentum
  lr_schedule: Cosine annealing with warmup
  batch_size: 16 (limited by GPU memory during training)
```

---

## Problem 5: CNN Architecture Debugging (Debug/Fix)

**Difficulty:** Debug/Fix
**Estimated Time:** 20 minutes
**Concepts:** CNN architecture, training issues, dimension mismatches

### Problem Statement

A colleague designed a CNN for medical image classification but it's not training properly. Review the architecture and identify the issues.

**Model Code:**
```python
import torch
import torch.nn as nn

class MedicalCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        # Feature extraction
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        # Missing batch norm here

        # Classifier
        self.fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.8)  # Very high dropout

    def forward(self, x):
        # Input: (B, 3, 224, 224)

        x = self.pool1(self.bn1(self.conv1(x)))  # Missing ReLU
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.relu(self.conv4(x))  # No pooling, no batch norm

        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)

        return x

# Training code
model = MedicalCNN(num_classes=5)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # Very high LR

# Training loop (abbreviated)
for epoch in range(100):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

**Training Logs:**
```
Epoch 1: Loss = 1.61, Accuracy = 20.0% (random)
Epoch 10: Loss = 1.60, Accuracy = 20.5%
Epoch 50: Loss = 1.58, Accuracy = 22.0%
Epoch 100: Loss = 1.55, Accuracy = 24.0%
```

**Tasks:**
a) Identify at least 6 issues in the model architecture and training setup
b) Explain why each issue causes poor performance
c) Provide corrected code
d) What accuracy improvement would you expect after fixes?

---

### Hints

<details>
<summary>Hint 1 (Activation)</summary>
Check where ReLU is applied. The first block is missing activation after batch norm—this loses non-linearity.
</details>

<details>
<summary>Hint 2 (Regularization)</summary>
Dropout of 0.8 means 80% of neurons are zeroed. This is extremely high and prevents learning. Standard is 0.2-0.5.
</details>

<details>
<summary>Hint 3 (Learning Rate)</summary>
0.1 is very high for pre-trained CNN fine-tuning and even for training from scratch with batch norm. Standard is 0.01 or lower.
</details>

---

### Solution

**a) Issues Identified:**

| Issue | Location | Severity |
|-------|----------|----------|
| 1. Missing ReLU after conv1 | forward() line 1 | High |
| 2. Missing BatchNorm for conv4 | __init__ | Medium |
| 3. Missing pooling after conv4 | forward() | High |
| 4. Dropout 0.8 is too high | __init__ | High |
| 5. Learning rate 0.1 is too high | optimizer | High |
| 6. No learning rate scheduler | training loop | Medium |
| 7. Dimension mismatch potential | fc1 input size | High |
| 8. No data augmentation | training loop | Medium |

**b) Explanation of Each Issue:**

**Issue 1: Missing ReLU after conv1**
```python
# Current (wrong):
x = self.pool1(self.bn1(self.conv1(x)))  # Linear output!

# Problem:
# Conv → BN → Pool produces LINEAR features
# Without ReLU, the non-linearity needed for learning is missing
# Network's first layer cannot learn useful patterns
```

**Issue 2: Missing BatchNorm for conv4**
```python
# Current:
self.conv4 = nn.Conv2d(256, 512, ...)
# No bn4 defined

# Problem:
# Inconsistent normalization
# conv4 outputs have different scale than earlier layers
# Makes optimization harder
```

**Issue 3: Missing pooling after conv4**
```python
# Current:
x = self.relu(self.conv4(x))  # No pooling

# Problem:
# After pool3: 14×14 (224 → 112 → 56 → 28 → 14)
# After conv4 (no pool): Still 14×14
# But fc1 expects 7×7 based on input size (512 * 7 * 7)
# This causes dimension MISMATCH!
```

**Issue 4: Dropout 0.8**
```python
self.dropout = nn.Dropout(0.8)

# Problem:
# 80% of neurons zeroed → only 20% active
# Network can barely learn with so few active units
# Underfitting, not regularization

# Standard: 0.2-0.5 for CNNs
```

**Issue 5: Learning Rate 0.1**
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Problem:
# 0.1 is extremely high for CNN training
# Causes oscillation, may overshoot good solutions
# With BatchNorm, even 0.01 can be aggressive

# Standard: 0.01 for from-scratch, 0.001 for fine-tuning
```

**Issue 6: No learning rate scheduler**
```python
# Missing: scheduler = ...

# Problem:
# Constant LR doesn't allow refinement
# Early: Need moderate LR to explore
# Late: Need low LR to converge

# Fix: Add StepLR, CosineAnnealing, etc.
```

**Issue 7: Dimension Mismatch**
```python
self.fc1 = nn.Linear(512 * 7 * 7, 1024)

# Calculation:
# Input: 224×224
# After conv1+pool1 (stride 2, pool stride 2): 56×56
# After pool2 (stride 2): 28×28
# After pool3 (stride 2): 14×14
# After conv4 (no pool): 14×14  ← Still 14×14!

# fc1 expects: 512 × 7 × 7 = 25,088
# Actual input: 512 × 14 × 14 = 100,352
# This would cause a RUNTIME ERROR!
```

**c) Corrected Code:**

```python
import torch
import torch.nn as nn

class MedicalCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        # Feature extraction
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)  # FIX: Added BatchNorm
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # FIX: Added pooling

        # Classifier - FIX: Use Global Average Pooling instead of fixed size
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 256)  # FIX: Adjusted dimensions
        self.fc2 = nn.Linear(256, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # FIX: Reduced from 0.8

    def forward(self, x):
        # Input: (B, 3, 224, 224)

        x = self.relu(self.bn1(self.conv1(x)))  # FIX: Added ReLU
        x = self.pool1(x)  # 56×56

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)  # 28×28

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)  # 14×14

        x = self.relu(self.bn4(self.conv4(x)))  # FIX: Added bn4
        x = self.pool4(x)  # 7×7

        x = self.global_pool(x)  # FIX: Global pooling → 1×1
        x = x.view(x.size(0), -1)  # Flatten to (B, 512)

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


# Training code - FIXED
model = MedicalCNN(num_classes=5)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # FIX: Lower LR, AdamW
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)  # FIX: Added scheduler

# Training loop (abbreviated)
for epoch in range(100):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    scheduler.step()  # FIX: Update LR
```

**d) Expected Improvement:**

| Metric | Original | After Fixes |
|--------|----------|-------------|
| Epoch 1 Accuracy | 20% (random) | 40-50% |
| Epoch 10 Accuracy | 20.5% | 70-75% |
| Epoch 50 Accuracy | 22% | 85-88% |
| Epoch 100 Accuracy | 24% | 88-92% |

**Key Improvements:**
- ReLU addition: Enables feature learning (+30-40%)
- Dimension fix: Prevents crash, proper learning
- Dropout reduction: Allows network to learn (+10-15%)
- LR reduction: Stable training (+5-10%)
- AdamW + scheduler: Better convergence (+3-5%)

---

## Common Mistakes Summary

| Mistake | Why It's Wrong | Prevention |
|---------|---------------|------------|
| Missing activation | Network becomes linear | Always: Conv → BN → ReLU |
| Dimension mismatch | Runtime error or wrong learning | Calculate sizes through network |
| High dropout (> 0.5) | Prevents learning | Use 0.2-0.4 for CNNs |
| High learning rate | Unstable training | Start with 0.001, tune down |
| Inconsistent normalization | Unbalanced gradients | Add BN after every Conv |
| No LR scheduler | Poor convergence | Use cosine or step decay |

---

## Self-Assessment Guide

| Score | Mastery Level | Recommended Action |
|-------|---------------|-------------------|
| 5/5 problems | **Mastery** | Ready for CV research and production |
| 4/5 problems | **Proficient** | Review specific gaps |
| 3/5 problems | **Developing** | Re-study core concepts |
| 1-2/5 problems | **Foundational** | Complete Lesson 10 review |

---

*Generated from Lesson 10: Computer Vision | Practice Problems Skill*
