# Flashcards: Lesson 10 - Computer Vision

**Source:** Lessons/Lesson_10.md
**Subject Area:** AI Learning - Computer Vision: From Image Processing to Deep Learning Visual Understanding
**Date Generated:** 2026-01-08
**Total Cards:** 5 (2 Easy, 2 Medium, 1 Hard)

---

## Card Distribution

| Difficulty | Count | Bloom's Level | Focus Area |
|------------|-------|---------------|------------|
| Easy | 2 | Remember/Understand | Core concepts, architecture comparison |
| Medium | 2 | Apply/Analyze | Detection analysis, architecture selection |
| Hard | 1 | Evaluate/Synthesize | Complete CV system design |

---

## Easy Cards

### Card 1: CNN Architecture Evolution

**[FRONT]**
Compare the four landmark CNN architectures (LeNet, AlexNet, VGG, ResNet) in terms of their depth, key innovations, and impact on the field.

**[BACK]**
**CNN Architecture Evolution:**

| Architecture | Year | Depth | Parameters | Key Innovation | Impact |
|--------------|------|-------|------------|----------------|--------|
| **LeNet-5** | 1998 | 5 | 60K | First successful CNN | Proved CNNs work for vision |
| **AlexNet** | 2012 | 8 | 60M | ReLU, Dropout, GPU training | Started deep learning revolution |
| **VGG-16** | 2014 | 16 | 138M | Uniform 3×3 convolutions | Depth matters for accuracy |
| **ResNet-50** | 2015 | 50 | 25M | Skip connections | Enabled 100+ layer networks |

**Key Design Principles Established:**

| Principle | Architecture | Explanation |
|-----------|--------------|-------------|
| Depth improves accuracy | VGG | 16 layers >> 8 layers |
| 3×3 is optimal | VGG | Two 3×3 = one 5×5 (fewer params) |
| Skip connections | ResNet | Solve gradient vanishing |
| ReLU > Sigmoid | AlexNet | Faster training, no saturation |

**Performance Progression (ImageNet Top-5 Error):**
```
LeNet (1998): N/A (MNIST only)
AlexNet (2012): 15.3%
VGG-16 (2014): 7.3%
ResNet-152 (2015): 3.6%
EfficientNet-B7 (2019): 2.9%
```

**Difficulty:** Easy | **Bloom's Level:** Remember

---

### Card 2: Object Detection Metrics

**[FRONT]**
Explain the key metrics used in object detection: IoU (Intersection over Union), mAP (mean Average Precision), and the role of NMS (Non-Maximum Suppression). How do these metrics evaluate detection quality?

**[BACK]**
**Object Detection Metrics:**

**1. IoU (Intersection over Union):**
```
IoU = Area of Intersection / Area of Union

  ┌─────────┐
  │Predicted│
  │   ┌─────┼───┐
  │   │ IoU │   │ Ground Truth
  └───┼─────┘   │
      └─────────┘

IoU = 0.0: No overlap
IoU = 0.5: 50% overlap (typical threshold)
IoU = 1.0: Perfect overlap
```

**IoU Interpretation:**

| IoU Range | Quality | Use Case |
|-----------|---------|----------|
| < 0.5 | Poor | Usually false positive |
| 0.5-0.75 | Acceptable | COCO standard threshold |
| 0.75-0.9 | Good | Strict evaluation |
| > 0.9 | Excellent | High-precision required |

**2. mAP (mean Average Precision):**
```
For each class:
  1. Sort detections by confidence
  2. Calculate precision/recall at each threshold
  3. Compute AP = area under precision-recall curve

mAP = mean of AP across all classes

Common variants:
  mAP@0.5: IoU threshold = 0.5 (VOC standard)
  mAP@0.75: IoU threshold = 0.75 (stricter)
  mAP@[0.5:0.95]: Average over IoU thresholds (COCO standard)
```

**3. NMS (Non-Maximum Suppression):**
```
Purpose: Remove duplicate detections for same object

Algorithm:
1. Sort detections by confidence (high to low)
2. Select top detection, add to final list
3. Remove all detections with IoU > threshold (0.5) with selected
4. Repeat until no detections remain

Before NMS: 10 boxes on one car
After NMS: 1 box (highest confidence)
```

**Difficulty:** Easy | **Bloom's Level:** Understand

---

## Medium Cards

### Card 3: Detector Architecture Selection

**[FRONT]**
A robotics company needs object detection for three applications. For each, recommend either Faster R-CNN or YOLOv8 and justify your choice considering accuracy, speed, and deployment constraints.

**Applications:**
a) Warehouse robot identifying packages on shelves (100+ categories, must catch small items)
b) Real-time pedestrian detection for autonomous vehicle (30+ FPS required)
c) Medical device detecting surgical instruments in OR video (99.9% precision required)

**[BACK]**
**Detector Selection Analysis:**

**a) Warehouse Robot (100+ categories, small items)**

**Recommendation: Faster R-CNN with FPN**

| Factor | Faster R-CNN | YOLOv8 |
|--------|--------------|--------|
| Small object detection | Excellent (FPN multi-scale) | Good but worse |
| Many categories | Better precision per class | May confuse similar items |
| Speed requirement | 5-10 FPS acceptable | Overkill |

**Justification:**
- FPN provides dedicated feature maps for small packages
- Two-stage approach better for fine-grained category distinctions
- Robot doesn't need real-time (packages don't move quickly)
- High precision more important than speed in inventory management

---

**b) Autonomous Vehicle Pedestrian Detection (30+ FPS)**

**Recommendation: YOLOv8**

| Factor | Faster R-CNN | YOLOv8 |
|--------|--------------|--------|
| Speed | ~10 FPS | 100+ FPS |
| Latency | Higher | Lower |
| Safety-critical | - | Real-time response |

**Justification:**
- 30+ FPS mandatory for safety—Faster R-CNN cannot achieve this
- YOLOv8 reaches 100+ FPS with good accuracy
- Pedestrian detection is relatively "easy" (large objects, distinct shape)
- Lower latency = faster reaction to pedestrians

**Additional consideration:** Use TensorRT optimization for even faster inference.

---

**c) Medical Surgical Instruments (99.9% precision)**

**Recommendation: Faster R-CNN with ensemble + threshold tuning**

| Factor | Faster R-CNN | YOLOv8 |
|--------|--------------|--------|
| Precision at high recall | Better | Good but lower |
| False positive rate | Lower | Higher |
| Fine-grained localization | Better (RoI Align) | Acceptable |

**Justification:**
- Medical applications demand extremely low false positive rate
- Faster R-CNN's two-stage refinement improves precision
- Can set high confidence threshold (0.95+) and still maintain detection
- Speed not critical (video analysis can be near real-time)

**Safety strategy:**
```
Primary: Faster R-CNN (confidence > 0.9)
Verification: Second model for disagreement cases
Human review: Below 0.95 confidence
```

**Difficulty:** Medium | **Bloom's Level:** Apply

---

### Card 4: Segmentation Architecture Analysis

**[FRONT]**
Analyze the design choices in U-Net architecture: Why does it use skip connections? Why encoder-decoder instead of just encoder? What problem does each component solve? Compare to FCN and DeepLab approaches.

**[BACK]**
**U-Net Architecture Analysis:**

**Architecture Diagram:**
```
                    Skip 1                Skip 2
Input ─→ Enc1 ─────────────────────────────────────→ Dec4 ─→ Output
           │                                           ↑
           ↓         Skip 3                Skip 4      │
         Enc2 ─────────────────────────────→ Dec3 ────┘
           │                                   ↑
           ↓                                   │
         Enc3 ──────────────────────→ Dec2 ───┘
           │                           ↑
           ↓                           │
         Enc4 ────────→ Bottleneck → Dec1
```

**Component Analysis:**

**1. Encoder (Contracting Path):**
```
Purpose: Capture WHAT is in the image
- Progressive downsampling (2× each level)
- Increasing channels (64→128→256→512)
- Larger receptive field = global context

Without encoder: Cannot understand semantic content
```

**2. Decoder (Expanding Path):**
```
Purpose: Recover WHERE things are
- Progressive upsampling (2× each level)
- Decreasing channels (512→256→128→64)
- Recover spatial resolution

Without decoder: Output would be 32× smaller than input
```

**3. Skip Connections:**
```
Purpose: Preserve fine-grained details for boundaries

Problem they solve:
  Encoder captures "there's a cell here" (semantics)
  But loses "exactly where is the boundary?" (details)

Skip connections bring back:
  - High-resolution feature maps
  - Edge information
  - Precise localization

Without skip connections: Blurry boundaries
```

**Comparison Table:**

| Architecture | Encoder | Decoder | Skip Connections | Multi-scale |
|--------------|---------|---------|------------------|-------------|
| **FCN-32s** | VGG | Simple upsample | None | No |
| **FCN-8s** | VGG | Multi-scale | Pool3, Pool4 | Limited |
| **U-Net** | Custom | Symmetric | All levels | Yes |
| **DeepLab** | ResNet | ASPP | Optional | Atrous pyramid |

**FCN vs U-Net:**
```
FCN-32s:
  - 32× upsampling in one step
  - Coarse boundaries
  - Fast but imprecise

U-Net:
  - Gradual upsampling with skip at each level
  - Sharp boundaries
  - More parameters but better quality
```

**DeepLab vs U-Net:**
```
DeepLab (Atrous Convolution):
  - Maintains resolution longer (dilated convs)
  - ASPP captures multi-scale context
  - Better for diverse object sizes

U-Net:
  - Better for precise boundaries (medical)
  - Simpler architecture
  - Works well with limited data
```

**When to use each:**

| Use Case | Best Architecture | Reason |
|----------|------------------|--------|
| Medical imaging | U-Net | Precise boundaries, limited data |
| Street scenes | DeepLab | Multi-scale objects, diverse classes |
| Simple segmentation | FCN-8s | Fast, acceptable quality |
| Instance segmentation | Mask R-CNN | Need object instances |

**Difficulty:** Medium | **Bloom's Level:** Analyze

---

## Hard Cards

### Card 5: Complete Computer Vision System Design

**[FRONT]**
Design a complete computer vision system for a retail store that must:
1. Count customers entering/exiting (accuracy matters for conversion rate)
2. Track customer paths through store (heatmap analytics)
3. Detect shoplifting behaviors (real-time alerting)
4. Analyze shelf inventory (low-stock detection)
5. Run on edge devices (Jetson Xavier, 15W power budget)

Specify: camera setup, model architectures, deployment strategy, and system integration.

**[BACK]**
**Retail Computer Vision System Architecture:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    RETAIL CV SYSTEM ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐    │
│  │ Entry Cameras    │   │ Ceiling Cameras  │   │ Shelf Cameras    │    │
│  │ (2× 1080p)       │   │ (8× 720p)        │   │ (4× 1080p)       │    │
│  └────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘    │
│           │                      │                      │               │
│           ▼                      ▼                      ▼               │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │              EDGE PROCESSING (Jetson Xavier × 3)                  │  │
│  ├──────────────────────────────────────────────────────────────────┤  │
│  │                                                                   │  │
│  │  Xavier 1: Entry/Exit      Xavier 2: Tracking     Xavier 3: Shelf│  │
│  │  - People detection        - Multi-camera track   - Object detect│  │
│  │  - Direction counting      - Path aggregation     - Stock level  │  │
│  │  - ReID for accuracy       - Behavior analysis    - Planogram    │  │
│  │                                                                   │  │
│  └───────────────────────────────┬──────────────────────────────────┘  │
│                                  │                                      │
│                                  ▼                                      │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                      CENTRAL SERVER (Cloud/On-prem)               │  │
│  │  - Analytics aggregation   - Alert management   - Dashboard       │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

**1. Customer Counting (Entry/Exit)**

```yaml
camera_setup:
  location: Store entrance, ceiling-mounted angled view
  resolution: 1920×1080 @ 30fps
  count: 2 (covering both directions)

model_architecture:
  detection: YOLOv8-nano (optimized for Jetson)
    - Input: 640×640
    - Speed: 45 FPS on Xavier
    - mAP: 37.3 (sufficient for people)

  counting_logic:
    - Define entry/exit lines in frame
    - Track person centroids across frames
    - Count line crossings with direction

  re-identification:
    - OSNet-small for appearance features
    - Prevents double-counting (person exits, re-enters)
    - Match threshold: 0.7 cosine similarity

accuracy_measures:
  - Bi-directional verification (in ≈ out over day)
  - Periodic manual audit calibration
  - Target: 98% counting accuracy
```

---

**2. Customer Path Tracking (Heatmaps)**

```yaml
camera_setup:
  location: Ceiling grid covering store floor
  resolution: 1280×720 @ 15fps (lower is sufficient)
  count: 8 cameras with overlapping FOV

model_architecture:
  detection: YOLOv8-nano (shared with counting)

  tracking: ByteTrack
    - Motion-based association
    - Handles occlusions well
    - Runs at 30+ FPS

  multi_camera_fusion:
    - Homography mapping to floor plan
    - Track handoff at camera boundaries
    - Global trajectory reconstruction

output:
  - Per-customer path (anonymized)
  - Aggregate heatmap (dwell time per zone)
  - Zone transition matrix

privacy_compliance:
  - No face storage
  - Aggregate only (no individual tracking)
  - GDPR: Anonymous count data only
```

---

**3. Shoplifting Detection (Real-time)**

```yaml
model_architecture:
  approach: Anomaly detection + rule-based triggers

  stage_1_detection:
    - Person detection (shared YOLOv8)
    - Hand detection for concealment
    - Object detection (products)

  stage_2_behavior:
    action_recognition: SlowFast-R50 (temporal modeling)
    suspicious_actions:
      - Extended shelf interaction without basket add
      - Concealment motion (hand to pocket/bag)
      - Product-person spatial anomaly

  stage_3_alerting:
    - Confidence threshold: 0.85 (reduce false alarms)
    - Alert to security staff (not automated action)
    - Video clip attachment for review

training_data:
  - Normal shopping behavior: 10K clips
  - Simulated suspicious: 2K clips (actors)
  - Real incidents: As collected (privacy-compliant)

accuracy_target:
  - Recall: 80% (catch most incidents)
  - Precision: 60% (some false alarms acceptable)
  - False alarm rate: < 5 per day
```

---

**4. Shelf Inventory Analysis**

```yaml
camera_setup:
  location: Shelf-facing, fixed mount
  resolution: 1920×1080 (high res for product detail)
  trigger: Periodic capture (every 5 minutes) + motion-triggered

model_architecture:
  detection: YOLOv8-medium
    - Trained on store's product catalog
    - 500+ SKU categories
    - mAP target: 75%

  analysis:
    planogram_compliance:
      - Compare detected layout to expected planogram
      - Flag misplaced products

    stock_level:
      - Estimate fill level per shelf section
      - Threshold alerts: < 20% stock

    pricing_verification:
      - OCR on price tags
      - Cross-reference with POS system

deployment:
  - Batch processing (not real-time)
  - 4× per hour full scan
  - Alert dashboard for restocking
```

---

**5. Edge Deployment Strategy (15W Budget)**

```yaml
hardware_allocation:
  device: NVIDIA Jetson Xavier NX (3 units)
  power: 15W mode per device

  xavier_1_entry:
    cameras: 2 (entry)
    models: YOLOv8-nano + OSNet-small
    load: ~80% GPU utilization

  xavier_2_tracking:
    cameras: 8 (ceiling)
    models: YOLOv8-nano + ByteTrack + SlowFast (temporal)
    load: ~90% GPU utilization

  xavier_3_shelf:
    cameras: 4 (shelves)
    models: YOLOv8-medium + OCR
    load: ~70% GPU utilization

optimization_techniques:
  - TensorRT: 2-3× speedup
  - FP16 inference: 2× speedup, minimal accuracy loss
  - Batched inference: Group frames where possible
  - Resolution scaling: 720p for tracking, 1080p for detection

model_compression:
  - Quantization: INT8 where accuracy allows
  - Pruning: Remove 30% of channels
  - Knowledge distillation: Large → small model
```

---

**6. System Integration**

```yaml
data_flow:
  edge_to_cloud:
    - Aggregated counts (1 min intervals)
    - Anonymized path data
    - Alert events with video clips
    - Inventory snapshots

  latency_requirements:
    - Counting: < 1 second (real-time display)
    - Tracking: < 2 seconds (analytics)
    - Shoplifting: < 5 seconds (security response)
    - Inventory: 15 minutes (batch acceptable)

api_integration:
  - POS system: Conversion rate = sales / footfall
  - Inventory management: Auto-restock triggers
  - Security: Alert dispatch system
  - Analytics dashboard: Grafana/custom

reliability:
  - Local buffering: 24h of data on edge
  - Automatic failover: Continue counting if cloud disconnected
  - Health monitoring: Alert on camera/device failure
```

**Difficulty:** Hard | **Bloom's Level:** Synthesize

---

## Critical Knowledge Flags

The following concepts appear across multiple cards and represent essential knowledge:

| Concept | Cards | Significance |
|---------|-------|--------------|
| CNN architectures | 1, 3, 4 | Foundation of CV |
| Object detection metrics | 2, 3, 5 | Evaluation fundamentals |
| Skip connections | 1, 4, 5 | Key design pattern |
| Real-time vs accuracy tradeoff | 3, 5 | Practical deployment |

---

## Study Recommendations

### Before These Cards
- Review Lesson 8 (Neural Network Architectures) for building blocks
- Review Lesson 5 (Deep Learning) for training concepts

### After Mastering These Cards
- Implement a simple object detector using PyTorch
- Train U-Net on a segmentation dataset
- Deploy a model on edge device (Jetson Nano)

### Spaced Repetition Schedule
| Session | Focus |
|---------|-------|
| Day 1 | Cards 1-2 (foundations) |
| Day 3 | Cards 3-4 (analysis) |
| Day 7 | Card 5 (synthesis), review 1-4 |
| Day 14 | Full review |

---

*Generated from Lesson 10: Computer Vision | Flashcard Skill*
