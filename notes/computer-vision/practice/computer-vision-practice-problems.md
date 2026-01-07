# Practice Problems: Computer Vision

**Source:** notes/computer-vision/computer-vision-study-notes.md
**Concept Map Reference:** notes/computer-vision/concept-maps/computer-vision-concept-map.md
**Date Generated:** 2026-01-06
**Total Problems:** 5
**Distribution:** 1 Warm-Up | 2 Skill-Builder | 1 Challenge | 1 Debug/Fix

---

## Problem Distribution Strategy

| Problem | Type | Concepts Tested | Difficulty | Time Est. |
|---------|------|-----------------|------------|-----------|
| P1 | Warm-Up | Convolution Dimensions | Low | 10-15 min |
| P2 | Skill-Builder | ResNet Block Design | Medium | 20-25 min |
| P3 | Skill-Builder | IoU and NMS Calculation | Medium | 20-25 min |
| P4 | Challenge | Detection System Design | High | 35-45 min |
| P5 | Debug/Fix | Training Failures | Medium | 25-30 min |

---

## Problems

---

### Problem 1 | Warm-Up
**Concept:** Convolution Output Dimensions
**Source Section:** Core Concepts 2, 3
**Concept Map Node:** Convolution (10 connections)
**Related Flashcard:** Card 1
**Estimated Time:** 10-15 minutes

#### Problem Statement

You are designing a CNN for processing 256×256 RGB images. Calculate the output dimensions after each layer in the following sequence:

**Network Architecture:**
```
Input: 256×256×3

Layer 1: Conv2D(filters=32, kernel=3×3, stride=1, padding='same')
Layer 2: MaxPool2D(pool_size=2×2, stride=2)
Layer 3: Conv2D(filters=64, kernel=3×3, stride=2, padding='valid')
Layer 4: Conv2D(filters=128, kernel=5×5, stride=1, padding='same')
Layer 5: GlobalAveragePooling2D()
Layer 6: Dense(10)
```

**Tasks:**
1. Calculate the output shape (H × W × C) after each layer
2. Calculate the total number of parameters in the Conv2D layers (ignore biases for simplicity)
3. What would happen if Layer 3 used padding='same' instead of 'valid'?

---

#### Hints

<details>
<summary>Hint 1 (Conceptual)</summary>

Output dimension formula:
- For 'valid' padding: H_out = floor((H_in - K) / S) + 1
- For 'same' padding: H_out = ceil(H_in / S)

Parameters per Conv layer = K × K × C_in × C_out
</details>

<details>
<summary>Hint 2 (Approach)</summary>

Work through each layer sequentially:
- Layer 1: 'same' padding preserves spatial dimensions
- Layer 2: MaxPool with stride 2 halves dimensions
- Layer 3: 'valid' + stride 2 reduces further
</details>

<details>
<summary>Hint 3 (Solution Direction)</summary>

After Layer 2: 128×128×32
Layer 3 with 'valid': (128 - 3)/2 + 1 = 63
</details>

---

#### Solution

**Task 1: Output Shapes**

```
Input: 256×256×3

Layer 1: Conv2D(32, 3×3, stride=1, padding='same')
├── H_out = 256 (same padding preserves)
├── W_out = 256
├── C_out = 32 (number of filters)
└── Output: 256×256×32

Layer 2: MaxPool2D(2×2, stride=2)
├── H_out = 256 / 2 = 128
├── W_out = 256 / 2 = 128
├── C_out = 32 (unchanged)
└── Output: 128×128×32

Layer 3: Conv2D(64, 3×3, stride=2, padding='valid')
├── H_out = floor((128 - 3) / 2) + 1 = floor(125/2) + 1 = 62 + 1 = 63
├── W_out = 63
├── C_out = 64
└── Output: 63×63×64

Layer 4: Conv2D(128, 5×5, stride=1, padding='same')
├── H_out = 63 (same padding preserves)
├── W_out = 63
├── C_out = 128
└── Output: 63×63×128

Layer 5: GlobalAveragePooling2D()
├── Reduces each channel to single value
├── H_out = 1, W_out = 1
└── Output: 128 (or 1×1×128, then flattened)

Layer 6: Dense(10)
└── Output: 10 (class logits)
```

**Summary Table:**
```
| Layer | Operation | Output Shape |
|-------|-----------|--------------|
| Input | - | 256×256×3 |
| 1 | Conv 3×3, s=1, same | 256×256×32 |
| 2 | MaxPool 2×2, s=2 | 128×128×32 |
| 3 | Conv 3×3, s=2, valid | 63×63×64 |
| 4 | Conv 5×5, s=1, same | 63×63×128 |
| 5 | GlobalAvgPool | 128 |
| 6 | Dense | 10 |
```

**Task 2: Parameter Count**

```
Layer 1: Conv2D(32, 3×3)
├── Input channels: 3
├── Output channels: 32
├── Parameters = 3 × 3 × 3 × 32 = 864

Layer 3: Conv2D(64, 3×3)
├── Input channels: 32
├── Output channels: 64
├── Parameters = 3 × 3 × 32 × 64 = 18,432

Layer 4: Conv2D(128, 5×5)
├── Input channels: 64
├── Output channels: 128
├── Parameters = 5 × 5 × 64 × 128 = 204,800

Total Conv Parameters: 864 + 18,432 + 204,800 = 224,096
```

**Task 3: Effect of 'same' padding on Layer 3**

```
With padding='valid' (current):
├── Output: 63×63×64
└── Information at borders is reduced

With padding='same':
├── H_out = ceil(128 / 2) = 64
├── Output: 64×64×64
└── Spatial dimensions exactly halved (cleaner)

Implications:
- Layer 4 would be 64×64×128 instead of 63×63×128
- Slightly more computation (~2% more)
- Easier to calculate dimensions mentally
- Common practice: use 'same' padding for cleaner architecture
```

---

#### Common Mistakes

| Mistake | Why It's Wrong | Correct Approach |
|---------|----------------|------------------|
| Forgetting stride in formula | Stride affects output size | Include stride: (H-K+2P)/S + 1 |
| Confusing 'same' and 'valid' | Different padding behaviors | 'same' preserves size; 'valid' reduces |
| Channels unchanged in conv | Conv changes channel count | C_out = number of filters |
| Wrong parameter formula | Missing input channels | K × K × C_in × C_out |

---

#### Extension Challenge

Add batch normalization after each Conv layer. How many additional trainable parameters does this add? (Hint: BN has γ and β parameters per channel)

---

---

### Problem 2 | Skill-Builder
**Concept:** ResNet Block Design
**Source Section:** Core Concepts 5, 6
**Concept Map Node:** Skip Connections (7), ResNet (6)
**Related Flashcard:** Card 2
**Estimated Time:** 20-25 minutes

#### Problem Statement

You are implementing a ResNet-style architecture and need to design two types of residual blocks:

**Type A: Identity Block** (when input and output dimensions match)
- Input: 56×56×64
- Output: 56×56×64 (same dimensions)

**Type B: Projection Block** (when dimensions change)
- Input: 56×56×64
- Output: 28×28×128 (spatial halved, channels doubled)

**Tasks:**
1. Design the Identity Block with 3×3 convolutions and show the skip connection
2. Design the Projection Block, explaining how to match dimensions for the skip connection
3. Write the forward pass in pseudocode for both blocks
4. Explain why the projection block needs a 1×1 convolution in the skip path

---

#### Hints

<details>
<summary>Hint 1 (Conceptual)</summary>

Identity block: Input can be added directly to output since dimensions match.

Projection block: Need to transform input to match output dimensions using 1×1 conv with stride 2.
</details>

<details>
<summary>Hint 2 (Approach)</summary>

Standard ResNet bottleneck structure:
1. 1×1 conv (reduce channels)
2. 3×3 conv (process)
3. 1×1 conv (expand channels)
Plus the skip connection.
</details>

<details>
<summary>Hint 3 (Solution Direction)</summary>

For projection: use 1×1 conv with stride=2 in skip path to simultaneously halve spatial dimensions and change channel count.
</details>

---

#### Solution

**Task 1: Identity Block Design**

```
Identity Block (56×56×64 → 56×56×64)
═══════════════════════════════════════

Input: 56×56×64
    │
    ├─────────────────────────┐ (Skip Connection - Identity)
    │                         │
    ▼                         │
┌─────────────────────┐       │
│ Conv 1×1, 64→16     │       │
│ (Reduce channels)   │       │
│ BatchNorm + ReLU    │       │
└──────────┬──────────┘       │
           ▼                   │
┌─────────────────────┐       │
│ Conv 3×3, 16→16     │       │
│ (Main processing)   │       │
│ Stride=1, Pad=same  │       │
│ BatchNorm + ReLU    │       │
└──────────┬──────────┘       │
           ▼                   │
┌─────────────────────┐       │
│ Conv 1×1, 16→64     │       │
│ (Expand channels)   │       │
│ BatchNorm (no ReLU) │       │
└──────────┬──────────┘       │
           │                   │
           ▼                   │
        [  +  ] ◄─────────────┘ (Element-wise addition)
           │
           ▼
        ReLU
           │
           ▼
Output: 56×56×64
```

**Task 2: Projection Block Design**

```
Projection Block (56×56×64 → 28×28×128)
════════════════════════════════════════

Input: 56×56×64
    │
    ├─────────────────────────┐
    │                         │
    │                         ▼
    │               ┌─────────────────────┐
    │               │ Conv 1×1, 64→128    │ (Projection)
    │               │ Stride=2            │
    │               │ BatchNorm           │
    │               └──────────┬──────────┘
    │                          │
    ▼                          │ (Now 28×28×128)
┌─────────────────────┐        │
│ Conv 1×1, 64→32     │        │
│ Stride=1            │        │
│ BatchNorm + ReLU    │        │
└──────────┬──────────┘        │
           ▼                    │
┌─────────────────────┐        │
│ Conv 3×3, 32→32     │        │
│ Stride=2 (↓spatial) │        │
│ Pad=same            │        │
│ BatchNorm + ReLU    │        │
└──────────┬──────────┘        │
           │ (Now 28×28×32)    │
           ▼                    │
┌─────────────────────┐        │
│ Conv 1×1, 32→128    │        │
│ (Expand channels)   │        │
│ BatchNorm (no ReLU) │        │
└──────────┬──────────┘        │
           │ (28×28×128)       │
           ▼                    │
        [  +  ] ◄──────────────┘ (Both 28×28×128 now)
           │
           ▼
        ReLU
           │
           ▼
Output: 28×28×128
```

**Task 3: Forward Pass Pseudocode**

```python
class IdentityBlock(nn.Module):
    def __init__(self, in_channels, bottleneck_channels):
        super().__init__()
        # Main path
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)

        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)

        self.conv3 = nn.Conv2d(bottleneck_channels, in_channels, 1)
        self.bn3 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x  # Save input for skip connection

        # Main path
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))  # No ReLU before addition

        # Skip connection + final activation
        out = out + identity  # Element-wise addition
        out = F.relu(out)

        return out


class ProjectionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck_channels, stride=2):
        super().__init__()
        # Main path
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)

        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3,
                               stride=stride, padding=1)  # Stride here!
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)

        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Projection shortcut (1×1 conv to match dimensions)
        self.projection = nn.Conv2d(in_channels, out_channels, 1, stride=stride)
        self.bn_proj = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Project input to match output dimensions
        identity = self.bn_proj(self.projection(x))

        # Main path
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        # Skip connection + final activation
        out = out + identity
        out = F.relu(out)

        return out
```

**Task 4: Why 1×1 Convolution for Projection?**

```
Problem: Skip connection requires element-wise addition
         Both tensors must have IDENTICAL dimensions

Input:  56×56×64  (H=56, W=56, C=64)
Output: 28×28×128 (H=28, W=28, C=128)

Dimension Mismatches:
1. Spatial: 56→28 (need to halve)
2. Channels: 64→128 (need to double)

Solution: 1×1 Convolution with stride=2

┌────────────────────────────────────────────────┐
│ Conv2d(in_channels=64, out_channels=128,       │
│        kernel_size=1, stride=2)                │
│                                                │
│ Effect:                                        │
│ - kernel_size=1: No spatial mixing, just       │
│   linear combination of channels               │
│ - stride=2: Subsamples spatially (56→28)       │
│ - out_channels=128: Changes channel dimension  │
│                                                │
│ Input:  56×56×64                               │
│ Output: 28×28×128 ✓ (matches main path)        │
└────────────────────────────────────────────────┘

Why 1×1 specifically?
- Minimal computation (just channel mixing)
- No spatial receptive field overlap
- Efficient projection
- Alternative (3×3) would work but wastes computation
```

---

#### Common Mistakes

| Mistake | Why It's Wrong | Correct Approach |
|---------|----------------|------------------|
| ReLU before addition | Limits gradient flow through skip | ReLU after addition only |
| Forgetting projection | Can't add mismatched tensors | Always project when dimensions differ |
| Wrong stride placement | Must halve in both paths | Put stride=2 in conv2 (main) AND projection |
| No BatchNorm on projection | Inconsistent normalization | BN on projection path too |

---

#### Extension Challenge

Implement a "Pre-activation" ResNet block where BatchNorm and ReLU come before convolution. How does this change gradient flow?

---

---

### Problem 3 | Skill-Builder
**Concept:** IoU and Non-Maximum Suppression
**Source Section:** Core Concepts 7
**Concept Map Node:** IoU (4), NMS (3)
**Related Flashcard:** Card 3
**Estimated Time:** 20-25 minutes

#### Problem Statement

A YOLO detector has produced the following raw predictions for detecting "cars" in an image:

**Detections (format: [x1, y1, x2, y2, confidence]):**
```
Box A: [100, 100, 200, 200, 0.95]
Box B: [110, 105, 205, 210, 0.88]
Box C: [300, 300, 400, 400, 0.75]
Box D: [105, 95,  195, 195, 0.70]
Box E: [310, 290, 420, 410, 0.65]
```

**Tasks:**
1. Calculate IoU between Box A and Box B
2. Calculate IoU between Box A and Box D
3. Apply Non-Maximum Suppression with IoU threshold = 0.5 and return the final detections
4. Explain why NMS is necessary and what would happen without it

---

#### Hints

<details>
<summary>Hint 1 (Conceptual)</summary>

IoU = Intersection Area / Union Area

Union = Area_A + Area_B - Intersection

For intersection: find overlapping rectangle coordinates.
</details>

<details>
<summary>Hint 2 (Approach)</summary>

Intersection coordinates:
- x1_inter = max(x1_A, x1_B)
- y1_inter = max(y1_A, y1_B)
- x2_inter = min(x2_A, x2_B)
- y2_inter = min(y2_A, y2_B)

If x2_inter < x1_inter or y2_inter < y1_inter, no intersection.
</details>

<details>
<summary>Hint 3 (Solution Direction)</summary>

NMS algorithm:
1. Sort by confidence (descending)
2. Select highest confidence box
3. Remove all boxes with IoU > threshold
4. Repeat until no boxes remain
</details>

---

#### Solution

**Task 1: IoU between Box A and Box B**

```
Box A: [100, 100, 200, 200]
Box B: [110, 105, 205, 210]

Step 1: Calculate intersection coordinates
┌────────────────────────────────────────┐
│ x1_inter = max(100, 110) = 110         │
│ y1_inter = max(100, 105) = 105         │
│ x2_inter = min(200, 205) = 200         │
│ y2_inter = min(200, 210) = 200         │
└────────────────────────────────────────┘

Step 2: Calculate intersection area
┌────────────────────────────────────────┐
│ width_inter  = 200 - 110 = 90          │
│ height_inter = 200 - 105 = 95          │
│ area_inter   = 90 × 95 = 8,550         │
└────────────────────────────────────────┘

Step 3: Calculate individual areas
┌────────────────────────────────────────┐
│ Area_A = (200-100) × (200-100)         │
│        = 100 × 100 = 10,000            │
│                                        │
│ Area_B = (205-110) × (210-105)         │
│        = 95 × 105 = 9,975              │
└────────────────────────────────────────┘

Step 4: Calculate union and IoU
┌────────────────────────────────────────┐
│ Union = Area_A + Area_B - Intersection │
│       = 10,000 + 9,975 - 8,550         │
│       = 11,425                         │
│                                        │
│ IoU = 8,550 / 11,425 = 0.748           │
└────────────────────────────────────────┘

IoU(A, B) = 0.748 (74.8% overlap)
```

**Task 2: IoU between Box A and Box D**

```
Box A: [100, 100, 200, 200]
Box D: [105, 95,  195, 195]

Step 1: Intersection coordinates
┌────────────────────────────────────────┐
│ x1_inter = max(100, 105) = 105         │
│ y1_inter = max(100, 95)  = 100         │
│ x2_inter = min(200, 195) = 195         │
│ y2_inter = min(200, 195) = 195         │
└────────────────────────────────────────┘

Step 2: Intersection area
┌────────────────────────────────────────┐
│ width_inter  = 195 - 105 = 90          │
│ height_inter = 195 - 100 = 95          │
│ area_inter   = 90 × 95 = 8,550         │
└────────────────────────────────────────┘

Step 3: Individual areas
┌────────────────────────────────────────┐
│ Area_A = 10,000 (from before)          │
│ Area_D = (195-105) × (195-95)          │
│        = 90 × 100 = 9,000              │
└────────────────────────────────────────┘

Step 4: Union and IoU
┌────────────────────────────────────────┐
│ Union = 10,000 + 9,000 - 8,550         │
│       = 10,450                         │
│                                        │
│ IoU = 8,550 / 10,450 = 0.818           │
└────────────────────────────────────────┘

IoU(A, D) = 0.818 (81.8% overlap)
```

**Task 3: Apply NMS with threshold = 0.5**

```
Initial detections sorted by confidence:
┌─────┬────────────────────────────┬────────────┐
│ Box │ Coordinates                │ Confidence │
├─────┼────────────────────────────┼────────────┤
│  A  │ [100, 100, 200, 200]       │ 0.95       │
│  B  │ [110, 105, 205, 210]       │ 0.88       │
│  C  │ [300, 300, 400, 400]       │ 0.75       │
│  D  │ [105, 95,  195, 195]       │ 0.70       │
│  E  │ [310, 290, 420, 410]       │ 0.65       │
└─────┴────────────────────────────┴────────────┘

NMS Algorithm Execution:

Round 1:
├── Select Box A (highest confidence: 0.95)
├── Calculate IoU with remaining boxes:
│   ├── IoU(A, B) = 0.748 > 0.5 → SUPPRESS B
│   ├── IoU(A, C) = 0 (no overlap) → KEEP C
│   ├── IoU(A, D) = 0.818 > 0.5 → SUPPRESS D
│   └── IoU(A, E) = 0 (no overlap) → KEEP E
└── Remaining: [A, C, E]

Round 2:
├── A already selected
├── Select Box C (next highest: 0.75)
├── Calculate IoU with remaining:
│   └── IoU(C, E):
│       - x1_inter = max(300, 310) = 310
│       - y1_inter = max(300, 290) = 300
│       - x2_inter = min(400, 420) = 400
│       - y2_inter = min(400, 410) = 400
│       - Intersection = 90 × 100 = 9,000
│       - Area_C = 100 × 100 = 10,000
│       - Area_E = 110 × 120 = 13,200
│       - Union = 10,000 + 13,200 - 9,000 = 14,200
│       - IoU = 9,000 / 14,200 = 0.634 > 0.5 → SUPPRESS E
└── Remaining: [A, C]

Round 3:
└── No more boxes to process

FINAL DETECTIONS:
┌─────┬────────────────────────────┬────────────┐
│ Box │ Coordinates                │ Confidence │
├─────┼────────────────────────────┼────────────┤
│  A  │ [100, 100, 200, 200]       │ 0.95       │
│  C  │ [300, 300, 400, 400]       │ 0.75       │
└─────┴────────────────────────────┴────────────┘
```

**Task 4: Why NMS is Necessary**

```
Without NMS:
┌─────────────────────────────────────────────────────────────┐
│ Problem: Multiple detections for SAME object                │
│                                                             │
│ Dense prediction (YOLO, SSD) outputs predictions at many    │
│ locations. A single car might trigger:                      │
│ - High confidence at object center                          │
│ - Medium confidence slightly offset                         │
│ - Lower confidence at nearby grid cells                     │
│                                                             │
│ Result: 5 boxes for 1 car = confusing and incorrect!        │
│                                                             │
│    ┌──────────────────────┐                                │
│    │ ┌────────────────┐   │                                │
│    │ │ ┌────────────┐ │   │  Three overlapping             │
│    │ │ │   CAR      │ │   │  detections for                │
│    │ │ └────────────┘ │   │  the same car                  │
│    │ └────────────────┘   │                                │
│    └──────────────────────┘                                │
└─────────────────────────────────────────────────────────────┘

With NMS:
┌─────────────────────────────────────────────────────────────┐
│ Solution: Keep only best detection per object               │
│                                                             │
│ 1. Trust highest confidence detection                       │
│ 2. Remove overlapping duplicates (IoU > threshold)          │
│ 3. Remaining boxes = unique objects                         │
│                                                             │
│    ┌────────────┐                                          │
│    │   CAR      │  Single clean detection                  │
│    └────────────┘                                          │
│                                                             │
│ IoU threshold tuning:                                       │
│ - Too low (0.3): Might suppress distinct nearby objects     │
│ - Too high (0.7): Might keep duplicate detections           │
│ - Typical: 0.5 works well for most cases                    │
└─────────────────────────────────────────────────────────────┘
```

---

#### Common Mistakes

| Mistake | Why It's Wrong | Correct Approach |
|---------|----------------|------------------|
| Forgetting to subtract intersection from union | Double-counts overlap | Union = A + B - Intersection |
| Not sorting by confidence first | Wrong boxes get selected | Always sort descending by confidence |
| Calculating IoU for non-overlapping boxes | No intersection exists | Check if intersection is valid first |
| Applying NMS across different classes | Different objects suppressed | NMS should be per-class |

---

#### Extension Challenge

Implement Soft-NMS where instead of completely removing overlapping boxes, you decay their confidence scores based on IoU. Compare results with standard NMS.

---

---

### Problem 4 | Challenge
**Concept:** End-to-End Detection System Design
**Source Section:** Core Concepts 7, 8, 9
**Concept Map Node:** Detection cluster
**Related Flashcard:** Card 3, Card 5
**Estimated Time:** 35-45 minutes

#### Problem Statement

A retail company wants to build a system to count customers and detect shopping cart usage from ceiling-mounted cameras. The requirements are:

**Functional Requirements:**
- Detect people and shopping carts in real-time video (1080p, 30 FPS)
- Track individuals across frames to count unique customers
- Distinguish empty carts from carts with items
- Generate hourly reports of customer count and cart usage

**Technical Constraints:**
- Processing must happen on-premises (privacy)
- Hardware: NVIDIA RTX 3080 (10GB VRAM)
- Latency: <100ms per frame
- Multiple camera feeds (up to 8 cameras)

**Tasks:**
1. Design the detection architecture (backbone, detection head, input resolution)
2. Define the training data requirements and labeling strategy
3. Design the multi-camera processing pipeline
4. Propose the tracking approach for counting unique customers
5. Identify potential failure cases and mitigation strategies

---

#### Hints

<details>
<summary>Hint 1 (Conceptual)</summary>

For real-time multi-camera: Need efficient model that can process 8×30 = 240 FPS total. Consider batching, model optimization, or temporal subsampling.
</details>

<details>
<summary>Hint 2 (Approach)</summary>

Detection + Tracking: Use detection model for object localization, then associate detections across frames with tracking algorithm (DeepSORT, ByteTrack).
</details>

<details>
<summary>Hint 3 (Solution Direction)</summary>

- YOLOv8-medium at 640×640 can achieve ~50 FPS on RTX 3080
- Process cameras in batches of 4 (batched inference)
- Track each camera independently, aggregate counts
</details>

---

#### Solution

**Task 1: Detection Architecture**

```
Architecture Selection: YOLOv8-medium

Justification:
├── Speed: ~50 FPS at 640×640 on RTX 3080 (single stream)
├── Accuracy: 50.2% mAP on COCO (sufficient for retail)
├── Model size: ~26M parameters (~52MB FP16)
└── Well-suited for person/object detection

Input Configuration:
├── Resolution: 640×640 (downscale from 1080p)
├── Preserve aspect ratio with padding
└── FP16 inference for 2× speedup

Model Architecture:
┌─────────────────────────────────────────────────────┐
│ Input: 640×640×3                                    │
│        ↓                                            │
│ Backbone: CSPDarknet53 (modified)                   │
│        ↓                                            │
│ Neck: PANet (Path Aggregation Network)              │
│   - Multi-scale feature fusion                      │
│   - Outputs at 80×80, 40×40, 20×20                  │
│        ↓                                            │
│ Head: Decoupled detection heads                     │
│   - Classification: Person, Cart-empty, Cart-full   │
│   - Regression: Bounding box coordinates            │
│        ↓                                            │
│ Post-processing: NMS (IoU=0.5, conf=0.25)           │
└─────────────────────────────────────────────────────┘

Classes (3):
├── Person
├── Cart-empty
└── Cart-full

Memory Budget:
├── Model (FP16): ~52MB
├── Per-frame processing: ~200MB
├── Batch of 4 frames: ~800MB
├── Tracking state: ~100MB
└── Total: ~1.2GB (well under 10GB limit)
```

**Task 2: Training Data Requirements**

```
Dataset Requirements:
┌─────────────────────────────────────────────────────────────┐
│ Quantity:                                                   │
│ ├── Minimum: 5,000 labeled frames                          │
│ ├── Recommended: 10,000-15,000 frames                      │
│ └── Balance: ~40% person, 30% cart-empty, 30% cart-full    │
│                                                             │
│ Diversity:                                                  │
│ ├── Multiple store locations (lighting varies)             │
│ ├── Different times of day                                 │
│ ├── Various crowd densities (sparse to crowded)            │
│ ├── Occlusion scenarios (people behind carts)              │
│ └── Edge cases (children, wheelchairs, strollers)          │
└─────────────────────────────────────────────────────────────┘

Labeling Strategy:
┌─────────────────────────────────────────────────────────────┐
│ 1. Bounding Boxes:                                          │
│    - Tight boxes around each person/cart                    │
│    - Include partially visible objects (>30% visible)       │
│                                                             │
│ 2. Classification Guidelines:                               │
│    - Cart-empty: Visible basket interior empty              │
│    - Cart-full: Any visible items in basket                 │
│    - Ambiguous: Label as cart-full (conservative)           │
│                                                             │
│ 3. Quality Control:                                         │
│    - Double-labeling for 10% of data                        │
│    - Review disagreements                                   │
│    - Inter-annotator agreement > 90%                        │
│                                                             │
│ 4. Active Learning:                                         │
│    - Deploy initial model                                   │
│    - Collect high-uncertainty predictions                   │
│    - Prioritize labeling difficult cases                    │
└─────────────────────────────────────────────────────────────┘

Data Augmentation:
├── Geometric: Random crop, flip, scale (0.5-1.5)
├── Photometric: Brightness, contrast, hue jitter
├── Mosaic: Combine 4 images (YOLO-style)
└── Mixup: Blend images for regularization
```

**Task 3: Multi-Camera Processing Pipeline**

```
Pipeline Architecture:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Camera 1 ─┐                                               │
│   Camera 2 ─┤     ┌─────────────┐     ┌──────────────┐     │
│   Camera 3 ─┼────►│ Frame       │────►│ Batched      │     │
│   Camera 4 ─┘     │ Aggregator  │     │ Detection    │     │
│                   │ (async)     │     │ (batch=4)    │     │
│   Camera 5 ─┐     └─────────────┘     └──────┬───────┘     │
│   Camera 6 ─┤                                │             │
│   Camera 7 ─┼────► [Same pipeline] ──────────┤             │
│   Camera 8 ─┘                                │             │
│                                              ▼             │
│                                    ┌──────────────┐        │
│                                    │ Per-Camera   │        │
│                                    │ Tracking     │        │
│                                    └──────┬───────┘        │
│                                           │                │
│                                           ▼                │
│                                    ┌──────────────┐        │
│                                    │ Aggregation  │        │
│                                    │ & Reporting  │        │
│                                    └──────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Processing Strategy:
├── Group cameras into 2 batches of 4
├── Alternate batches to balance load
├── Each batch: 4 frames × 640×640 × 3 = ~5MB input
├── Inference time per batch: ~80ms
├── Total throughput: 2 batches × 4 frames / 80ms = 100 FPS
├── Per camera: 100/8 = 12.5 FPS (subsample from 30)

Latency Analysis:
├── Frame acquisition: ~10ms
├── Preprocessing (resize, normalize): ~5ms
├── Detection (batch=4): ~80ms
├── Tracking: ~5ms per camera × 4 = ~20ms
├── Total: ~115ms (slightly over 100ms target)

Optimization to hit 100ms:
├── Use TensorRT optimization: 80ms → 50ms
├── Async preprocessing while previous batch processes
└── Final: ~70-80ms per batch ✓
```

**Task 4: Tracking for Unique Customer Counting**

```
Tracking Algorithm: ByteTrack (or DeepSORT)

Why ByteTrack:
├── Handles occlusions well (important in retail)
├── Associates low-confidence detections
├── Simple, fast, effective
└── No re-identification features needed (within single camera)

Tracking Pipeline:
┌─────────────────────────────────────────────────────────────┐
│ For each camera independently:                              │
│                                                             │
│ Frame t detections: [(box, class, conf), ...]              │
│         ↓                                                   │
│ 1. Predict existing track positions (Kalman filter)        │
│         ↓                                                   │
│ 2. Associate detections to tracks (Hungarian algorithm)    │
│    - IoU-based matching                                    │
│    - High-confidence first, then low-confidence            │
│         ↓                                                   │
│ 3. Update matched tracks                                   │
│         ↓                                                   │
│ 4. Initialize new tracks for unmatched detections          │
│         ↓                                                   │
│ 5. Remove stale tracks (no match for N frames)             │
│         ↓                                                   │
│ Output: Active tracks with unique IDs                      │
└─────────────────────────────────────────────────────────────┘

Counting Logic:
┌─────────────────────────────────────────────────────────────┐
│ Entry/Exit Detection:                                       │
│ ├── Define entry zone (e.g., door region)                  │
│ ├── Define exit zone (e.g., checkout area)                 │
│ └── Count when track crosses zone boundary                 │
│                                                             │
│ Unique Customer Counting:                                   │
│ ├── Each track ID = one counting event                     │
│ ├── Track must persist > 30 frames (1 second) to count     │
│ └── Prevents false counts from noise/artifacts             │
│                                                             │
│ Cart Association:                                           │
│ ├── Match cart to nearest person (spatial proximity)       │
│ ├── Person-cart pair if distance < threshold               │
│ └── Track cart fullness state changes                      │
└─────────────────────────────────────────────────────────────┘
```

**Task 5: Failure Cases and Mitigations**

```
┌─────────────────────────────────────────────────────────────┐
│ Failure Case 1: Severe Occlusion                           │
│ ├── Problem: Person hidden behind cart/shelf               │
│ ├── Impact: Missed detection, track fragmentation          │
│ ├── Mitigation:                                            │
│ │   - ByteTrack recovers low-confidence detections         │
│ │   - Kalman prediction maintains track during occlusion   │
│ │   - Re-ID features for track recovery (optional)         │
│ └── Fallback: Accept some counting error (~5%)             │
├─────────────────────────────────────────────────────────────┤
│ Failure Case 2: Crowded Scenes                             │
│ ├── Problem: Many overlapping people                       │
│ ├── Impact: ID switches, merged detections                 │
│ ├── Mitigation:                                            │
│ │   - Train with crowded scene data                        │
│ │   - Lower NMS IoU threshold in crowds                    │
│ │   - Multiple camera angles for disambiguation            │
│ └── Fallback: Aggregate estimates with confidence          │
├─────────────────────────────────────────────────────────────┤
│ Failure Case 3: Lighting Changes                           │
│ ├── Problem: Night mode, emergency lights                  │
│ ├── Impact: Detection confidence drops                     │
│ ├── Mitigation:                                            │
│ │   - Include varied lighting in training data             │
│ │   - Automatic gain/exposure adjustment                   │
│ │   - Confidence threshold adaptation                      │
│ └── Fallback: Flag low-confidence periods in reports       │
├─────────────────────────────────────────────────────────────┤
│ Failure Case 4: Cart Classification Errors                 │
│ ├── Problem: Empty vs. full cart ambiguous from above      │
│ ├── Impact: Incorrect cart usage statistics                │
│ ├── Mitigation:                                            │
│ │   - Collect diverse cart images for training             │
│ │   - Track cart state changes (empty→full)                │
│ │   - Temporal smoothing (don't flip-flop)                 │
│ └── Fallback: Report with confidence intervals             │
├─────────────────────────────────────────────────────────────┤
│ Failure Case 5: Camera Failure / Obstruction               │
│ ├── Problem: Feed lost or blocked                          │
│ ├── Impact: Missing data for that region                   │
│ ├── Mitigation:                                            │
│ │   - Camera health monitoring (frame rate, brightness)    │
│ │   - Alert on anomalies                                   │
│ │   - Overlapping camera coverage where possible           │
│ └── Fallback: Mark data as incomplete in reports           │
└─────────────────────────────────────────────────────────────┘
```

---

#### Common Mistakes

| Mistake | Why It's Wrong | Correct Approach |
|---------|----------------|------------------|
| Processing cameras sequentially | Wastes GPU parallelism | Batch multiple frames together |
| One model per camera | Memory explosion | Share model, batch inference |
| Global tracking across cameras | People look different from each angle | Track per-camera, aggregate counts |
| No temporal filtering | Noisy detections cause count errors | Require track persistence to count |

---

#### Extension Challenge

Extend the system to estimate store heatmaps showing which areas customers spend the most time. What additional tracking information would you need to collect?

---

---

### Problem 5 | Debug/Fix
**Concept:** Training Failures
**Source Section:** All Core Concepts
**Concept Map Node:** CNN training issues
**Related Flashcard:** Card 2, Card 5
**Estimated Time:** 25-30 minutes

#### Problem Statement

A colleague is training a ResNet-50 for medical image classification (X-rays into 4 disease categories). They report the following issues:

**Training Configuration:**
```
- Dataset: 8,000 training images, 2,000 validation
- Input: 512×512 grayscale X-rays
- Model: ResNet-50 (ImageNet pretrained)
- Optimizer: SGD, lr=0.01, momentum=0.9
- Batch size: 32
- Epochs: 50
```

**Observed Issues:**

```
Issue 1: Loss explodes after epoch 3
- Epoch 1: loss=2.3, val_loss=2.1
- Epoch 2: loss=1.8, val_loss=1.7
- Epoch 3: loss=1.2, val_loss=1.3
- Epoch 4: loss=NaN, val_loss=NaN

Issue 2: After fixing Issue 1, severe overfitting
- Epoch 20: train_acc=95%, val_acc=62%
- Epoch 30: train_acc=99%, val_acc=58%
- Gap keeps increasing

Issue 3: Class imbalance
- Class distribution: A=5000, B=2000, C=700, D=300
- Model predicts class A for 80% of samples

Issue 4: Poor performance on rotated images
- Normal X-rays: 85% accuracy
- Rotated X-rays (in production): 45% accuracy
```

**Tasks:**
1. Diagnose the root cause of each issue
2. Propose specific fixes for each issue
3. Provide a corrected training configuration
4. Suggest validation strategies to catch these issues earlier

---

#### Hints

<details>
<summary>Hint 1 (Conceptual)</summary>

Issue 1: Learning rate too high causes gradient explosion
Issue 2: Model capacity too large for dataset size
Issue 3: Need weighted loss or sampling strategy
Issue 4: Data augmentation missing
</details>

<details>
<summary>Hint 2 (Approach)</summary>

For pretrained models on new domains:
- Lower learning rate for pretrained layers
- Higher learning rate for new classifier head
- Gradual unfreezing can help
</details>

<details>
<summary>Hint 3 (Solution Direction)</summary>

Typical fixes:
1. lr=0.001 instead of 0.01, with warmup
2. Dropout, weight decay, early stopping
3. Class weights inversely proportional to frequency
4. Random rotation augmentation (±180° for X-rays)
</details>

---

#### Solution

**Task 1: Root Cause Diagnosis**

```
Issue 1: Loss Explodes (NaN)
┌─────────────────────────────────────────────────────────────┐
│ Root Cause: Learning rate too high for fine-tuning          │
│                                                             │
│ Evidence:                                                   │
│ - Loss decreasing initially (learning happening)            │
│ - Sudden explosion (gradients grew too large)               │
│ - NaN indicates numerical overflow                          │
│                                                             │
│ Why: ImageNet-pretrained weights are already good.          │
│ High LR makes large updates → overshoots → diverges         │
│                                                             │
│ Contributing factors:                                       │
│ - SGD without gradient clipping                             │
│ - No learning rate warmup                                   │
│ - All layers same LR (pretrained should be lower)           │
└─────────────────────────────────────────────────────────────┘

Issue 2: Severe Overfitting
┌─────────────────────────────────────────────────────────────┐
│ Root Cause: Model too complex for dataset size              │
│                                                             │
│ Evidence:                                                   │
│ - Train accuracy 99%, val accuracy 58%                      │
│ - Gap increases with more training                          │
│                                                             │
│ Why: ResNet-50 has ~25M parameters                          │
│ 8,000 images insufficient to constrain all parameters       │
│ Model memorizes training set                                │
│                                                             │
│ Contributing factors:                                       │
│ - No regularization (dropout, weight decay)                 │
│ - No data augmentation                                      │
│ - No early stopping                                         │
│ - Training too many epochs                                  │
└─────────────────────────────────────────────────────────────┘

Issue 3: Class Imbalance
┌─────────────────────────────────────────────────────────────┐
│ Root Cause: Unbalanced class distribution                   │
│                                                             │
│ Evidence:                                                   │
│ - Class A: 5000 (62.5%), B: 2000 (25%), C: 700 (8.75%)     │
│ - D: 300 (3.75%)                                           │
│ - Model predicts A 80% of time (follows prior)             │
│                                                             │
│ Why: Standard cross-entropy treats all samples equally      │
│ Model learns: "predict majority class = low loss"           │
│ Rare classes contribute little to total loss                │
└─────────────────────────────────────────────────────────────┘

Issue 4: Rotation Sensitivity
┌─────────────────────────────────────────────────────────────┐
│ Root Cause: No rotation augmentation during training        │
│                                                             │
│ Evidence:                                                   │
│ - 85% on normal, 45% on rotated (almost random for 4-class)│
│                                                             │
│ Why: CNN features are NOT rotation invariant                │
│ Model learned features specific to upright X-rays           │
│ Rotated images activate different (untrained) features      │
│                                                             │
│ Note: Medical X-rays often rotated in practice              │
│ (patient positioning, scanner orientation)                  │
└─────────────────────────────────────────────────────────────┘
```

**Task 2: Specific Fixes**

```
Fix for Issue 1: Learning Rate and Stability
┌─────────────────────────────────────────────────────────────┐
│ 1. Reduce learning rate:                                    │
│    - Classifier head: lr=0.001                              │
│    - Pretrained layers: lr=0.0001 (10× lower)               │
│                                                             │
│ 2. Add learning rate warmup:                                │
│    - Start at lr/10, increase linearly for 5 epochs         │
│                                                             │
│ 3. Use learning rate scheduler:                             │
│    - ReduceLROnPlateau or CosineAnnealing                   │
│                                                             │
│ 4. Add gradient clipping:                                   │
│    - max_norm=1.0                                           │
│                                                             │
│ 5. Switch optimizer:                                        │
│    - AdamW often more stable than SGD for fine-tuning       │
└─────────────────────────────────────────────────────────────┘

Fix for Issue 2: Regularization
┌─────────────────────────────────────────────────────────────┐
│ 1. Add dropout before classifier:                           │
│    - Dropout(p=0.5) before final FC layer                   │
│                                                             │
│ 2. Add weight decay:                                        │
│    - weight_decay=1e-4 in optimizer                         │
│                                                             │
│ 3. Implement early stopping:                                │
│    - Monitor val_loss, patience=5 epochs                    │
│    - Restore best weights                                   │
│                                                             │
│ 4. Data augmentation (see Issue 4 fix)                      │
│                                                             │
│ 5. Consider smaller model:                                  │
│    - ResNet-18 or EfficientNet-B0 may be sufficient         │
│                                                             │
│ 6. Freeze early layers:                                     │
│    - Train only last 2-3 blocks initially                   │
│    - Gradually unfreeze                                     │
└─────────────────────────────────────────────────────────────┘

Fix for Issue 3: Class Imbalance
┌─────────────────────────────────────────────────────────────┐
│ Option A: Weighted Cross-Entropy Loss                       │
│                                                             │
│   weights = [1/5000, 1/2000, 1/700, 1/300]                 │
│   weights = weights / sum(weights)  # normalize             │
│   # Result: [0.026, 0.066, 0.188, 0.439]                    │
│   loss = CrossEntropyLoss(weight=weights)                   │
│                                                             │
│ Option B: Oversampling minority classes                     │
│                                                             │
│   sampler = WeightedRandomSampler(sample_weights)          │
│   # Each class sampled equally often                        │
│                                                             │
│ Option C: Focal Loss                                        │
│                                                             │
│   FL = -α(1-p)^γ log(p)                                    │
│   # Down-weights easy (majority) examples                   │
│   # γ=2 is common                                          │
│                                                             │
│ Recommended: Weighted loss + stratified validation          │
└─────────────────────────────────────────────────────────────┘

Fix for Issue 4: Rotation Augmentation
┌─────────────────────────────────────────────────────────────┐
│ Add comprehensive data augmentation:                        │
│                                                             │
│ transforms.Compose([                                        │
│     transforms.RandomRotation(180),      # Full rotation   │
│     transforms.RandomHorizontalFlip(),                      │
│     transforms.RandomVerticalFlip(),     # X-rays can flip │
│     transforms.RandomAffine(                                │
│         degrees=0,                                          │
│         translate=(0.1, 0.1),           # Shift            │
│         scale=(0.9, 1.1)                # Scale            │
│     ),                                                      │
│     transforms.ColorJitter(                                 │
│         brightness=0.2,                                     │
│         contrast=0.2                                        │
│     ),                                                      │
│     transforms.GaussianBlur(kernel_size=3),  # Noise       │
│     transforms.Normalize(mean, std)                         │
│ ])                                                          │
│                                                             │
│ Note: X-rays specifically benefit from:                     │
│ - Full 360° rotation (no "up" direction)                   │
│ - Flips (left/right often equivalent)                       │
│ - Slight elastic deformations                               │
└─────────────────────────────────────────────────────────────┘
```

**Task 3: Corrected Training Configuration**

```python
# Corrected Configuration

# Model setup
model = models.resnet50(pretrained=True)
model.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3)  # Grayscale input
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(2048, 4)
)

# Differential learning rates
optimizer = torch.optim.AdamW([
    {'params': model.conv1.parameters(), 'lr': 1e-4},
    {'params': model.layer1.parameters(), 'lr': 1e-4},
    {'params': model.layer2.parameters(), 'lr': 1e-4},
    {'params': model.layer3.parameters(), 'lr': 5e-4},
    {'params': model.layer4.parameters(), 'lr': 5e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
], weight_decay=1e-4)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2
)

# Weighted loss for class imbalance
class_weights = torch.tensor([0.026, 0.066, 0.188, 0.439])
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Data augmentation
train_transforms = transforms.Compose([
    transforms.RandomRotation(180),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

# Training parameters
config = {
    'batch_size': 32,
    'epochs': 50,
    'early_stopping_patience': 7,
    'gradient_clip_norm': 1.0,
    'warmup_epochs': 3
}

# Early stopping
early_stopper = EarlyStopping(patience=7, restore_best_weights=True)
```

**Task 4: Validation Strategies**

```
Catch Issues Earlier:
┌─────────────────────────────────────────────────────────────┐
│ 1. Monitor gradient norms                                   │
│    - Log ||∇W|| each batch                                  │
│    - Alert if > 10× initial value                          │
│    - Catches exploding gradients before NaN                 │
│                                                             │
│ 2. Track per-class metrics                                  │
│    - Not just overall accuracy                              │
│    - Confusion matrix each epoch                            │
│    - F1 score per class                                     │
│    - Catches class imbalance issues                         │
│                                                             │
│ 3. Validation set augmentation test                         │
│    - Run inference on augmented val set                     │
│    - Compare to non-augmented                               │
│    - Large gap = augmentation needed in training            │
│                                                             │
│ 4. Learning curves from epoch 1                             │
│    - Plot train vs val loss/accuracy                        │
│    - Overfitting visible early (diverging curves)           │
│                                                             │
│ 5. Holdout test set                                         │
│    - Never used during development                          │
│    - Final evaluation only                                  │
│    - Catches overfitting to validation set                  │
│                                                             │
│ 6. Cross-validation for small datasets                      │
│    - 5-fold CV more reliable than single split              │
│    - Especially important with 8K images                    │
└─────────────────────────────────────────────────────────────┘
```

---

#### Common Mistakes

| Mistake | Why It's Wrong | Correct Approach |
|---------|----------------|------------------|
| Same LR for all layers | Pretrained layers need smaller updates | Differential learning rates |
| Ignoring class distribution | Biased predictions | Weighted loss or sampling |
| No augmentation | Model doesn't generalize | Augmentation essential |
| Training until loss=0 | Severe overfitting | Early stopping on val metric |

---

#### Extension Challenge

Design a learning rate finder experiment that would identify the optimal learning rate before training begins. Implement the approach where you gradually increase LR and plot loss vs. LR to find the steepest descent region.

---

---

## Skills Integration Summary

This practice problem set integrates with the full skill chain:

```
Study Notes (10 Concepts)
        ↓
Concept Map (28 concepts, 45 relationships)
        ↓
Flashcards (5 cards: 2E/2M/1H)
        ↓
Practice Problems ← YOU ARE HERE
        ↓
Quiz (5 questions: 2MC/2SA/1E)
```

| Problem | Concepts Practiced | Prepares For |
|---------|-------------------|--------------|
| P1 | Convolution dimensions | Quiz Q1 |
| P2 | ResNet, skip connections | Quiz Q2 |
| P3 | IoU, NMS, detection | Quiz Q3 |
| P4 | Full detection system | Quiz Q5 |
| P5 | Training debugging | Quiz Q4 |
