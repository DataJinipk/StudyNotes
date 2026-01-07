# Assessment Quiz: Lesson 10 - Computer Vision

**Source:** Lessons/Lesson_10.md
**Subject Area:** AI Learning - Computer Vision: From Image Processing to Deep Learning Visual Understanding
**Date Generated:** 2026-01-08
**Total Questions:** 5
**Estimated Time:** 35-45 minutes

---

## Instructions

This assessment evaluates your understanding of Computer Vision concepts, from convolution operations through CNN architectures to object detection and segmentation. Answer all questions completely, showing your reasoning where applicable.

**Question Distribution:**
- Multiple Choice (2): Conceptual understanding (Remember/Understand)
- Short Answer (2): Application and analysis (Apply/Analyze)
- Essay (1): Synthesis and evaluation (Evaluate/Synthesize)

---

## Part A: Multiple Choice (10 points each)

### Question 1: Residual Connections

**Which statement correctly explains why ResNet's skip connections enable training of very deep networks?**

A) Skip connections reduce the number of parameters, making the network smaller and easier to train

B) Skip connections create gradient highways that prevent vanishing gradients by ensuring the gradient always has a direct path with factor (+1) regardless of layer weights

C) Skip connections allow the network to skip difficult layers entirely, so only easy-to-train layers are used during backpropagation

D) Skip connections replace batch normalization, which was the main cause of training difficulty in deep networks

---

### Question 2: Object Detection Approaches

**A team is choosing between Faster R-CNN and YOLOv8 for a traffic monitoring system that must detect vehicles, pedestrians, and cyclists. The system must process 1080p video at 25 FPS with high accuracy for safety-critical decisions. Which statement correctly describes the tradeoff?**

A) Faster R-CNN is always more accurate than YOLO, so it should be used despite being slower; the team should simply use more powerful hardware

B) YOLOv8 can achieve 25+ FPS on appropriate hardware while maintaining competitive accuracy with Faster R-CNN, making it suitable for this real-time application

C) Both detectors have identical accuracy and speed characteristics, so the choice is arbitrary

D) YOLO cannot detect multiple object classes in a single image, so Faster R-CNN is required for this multi-class scenario

---

## Part B: Short Answer (15 points each)

### Question 3: Segmentation Architecture Selection

**Context:** You are designing segmentation systems for three different applications. For each, recommend the most appropriate architecture and justify your choice.

**Applications:**

a) **Medical tumor segmentation** from MRI scans where precise boundaries are critical and training data is limited (500 labeled scans) (5 points)

b) **Autonomous driving scene parsing** requiring real-time segmentation of roads, vehicles, pedestrians, and signs at multiple scales (5 points)

c) **Warehouse robotics** needing to identify and separate individual product instances for pick-and-place operations (5 points)

---

### Question 4: Convolution and Feature Hierarchy Analysis

**Context:** Consider a simple CNN processing 224×224 RGB images:

```
Layer 1: Conv 3×3 (64 filters), stride 1, padding 1, ReLU
Layer 2: MaxPool 2×2, stride 2
Layer 3: Conv 3×3 (128 filters), stride 1, padding 1, ReLU
Layer 4: MaxPool 2×2, stride 2
Layer 5: Conv 3×3 (256 filters), stride 1, padding 1, ReLU
Layer 6: Global Average Pooling
Layer 7: Fully Connected (256 → 10 classes)
```

**Tasks:**

a) Calculate the spatial dimensions and receptive field at each layer (5 points)

b) Explain what type of features each layer typically learns, from edges to semantic concepts (5 points)

c) If we need to detect 16×16 pixel objects reliably, is this architecture sufficient? Justify with receptive field analysis (5 points)

---

## Part C: Essay (30 points)

### Question 5: Modern Computer Vision System Comparison

**Prompt:** A medical imaging startup is building a system to analyze chest X-rays that must:

1. **Classify** images into normal/abnormal with 5 specific pathologies
2. **Localize** abnormalities with bounding boxes
3. **Segment** affected lung regions pixel-by-pixel
4. **Handle** varying image quality from different hospital equipment
5. **Provide explainability** for clinical decisions

The startup is debating three architectural approaches:

1. **CNN-based pipeline:** Separate ResNet classifiers, Faster R-CNN detector, and U-Net segmenter
2. **Unified Transformer:** Single ViT-based model with task-specific heads
3. **Hybrid approach:** CNN backbone with Transformer attention for multi-task learning

**Your essay must address:**

1. **Architecture Comparison** (8 points)
   - Strengths and weaknesses of each approach for medical imaging
   - Data efficiency considerations (medical datasets are typically small)
   - Computational requirements for training and inference

2. **Multi-Task Learning Strategy** (7 points)
   - How classification, detection, and segmentation can share features
   - Task conflicts and how to resolve them
   - Joint vs. sequential training approaches

3. **Medical Domain Considerations** (7 points)
   - Handling limited labeled data (transfer learning strategies)
   - Explainability requirements for clinical acceptance
   - Regulatory considerations (FDA approval implications)

4. **Recommendation** (8 points)
   - Your recommended approach with justification
   - Implementation roadmap
   - Risk mitigation strategies

**Evaluation Criteria:**
- Technical accuracy of architectural descriptions
- Thoughtful analysis of medical imaging requirements
- Practical considerations for clinical deployment
- Well-reasoned recommendation

**Word Limit:** 600-800 words

---

## Answer Key

### Question 1: Residual Connections

**Correct Answer: B**

**Explanation:**

| Statement | Assessment |
|-----------|------------|
| **A** | Incorrect. Skip connections ADD parameters (projection shortcuts) and don't reduce network size. |
| **B** | Correct. The gradient through a residual block is: ∂L/∂x = ∂L/∂y × (∂F/∂x + 1). The "+1" from the identity path ensures gradients never completely vanish. |
| **C** | Incorrect. Skip connections don't skip layers; all layers are computed. The skip provides an ADDITIVE identity path, not a bypass. |
| **D** | Incorrect. Skip connections complement batch normalization, not replace it. ResNet uses both BN and skip connections. |

**Mathematical Proof:**

```
Without skip connection:
  y = F(x)
  ∂y/∂x = ∂F/∂x
  Through N layers: gradient = ∏ᵢ ∂Fᵢ/∂x (products of small terms → vanishes)

With skip connection:
  y = F(x) + x
  ∂y/∂x = ∂F/∂x + 1  ← Always at least 1!
  Through N layers: gradient = ∏ᵢ (∂Fᵢ/∂x + 1) (never vanishes completely)
```

**Understanding Gap:** If you selected A, review that skip connections add, not reduce, paths. If you selected C, understand that both paths (F(x) and x) are computed and added.

---

### Question 2: Object Detection Approaches

**Correct Answer: B**

**Explanation:**

| Statement | Assessment |
|-----------|------------|
| **A** | Incorrect. "Always more accurate" is false—modern YOLO versions match Faster R-CNN accuracy. Hardware scaling doesn't solve fundamental speed limits. |
| **B** | Correct. YOLOv8 achieves 50+ mAP on COCO while running at 100+ FPS on modern GPUs. It's well-suited for 25 FPS real-time requirements. |
| **C** | Incorrect. Detectors have different speed-accuracy profiles. Faster R-CNN prioritizes accuracy; YOLO prioritizes speed. |
| **D** | Incorrect. YOLO absolutely detects multiple classes—it predicts class probabilities for all categories at each grid cell/anchor. |

**Performance Comparison:**

| Detector | mAP (COCO) | Speed (FPS) | Suitable for 25 FPS? |
|----------|------------|-------------|---------------------|
| Faster R-CNN | ~42 | 5-15 | No |
| YOLOv5-m | ~45 | 60+ | Yes |
| YOLOv8-m | ~50 | 100+ | Yes |

**Traffic Monitoring Justification:**
- 25 FPS requirement eliminates two-stage detectors
- YOLOv8 accuracy is sufficient for traffic safety
- Multi-class detection is native to YOLO architecture

**Understanding Gap:** If you selected A, review modern YOLO performance metrics. If you selected D, review YOLO's multi-class prediction mechanism.

---

### Question 3: Segmentation Architecture Selection

**Model Answer:**

**a) Medical Tumor Segmentation (U-Net)**

**Recommendation:** U-Net with pre-trained encoder

**Justification:**

| Factor | U-Net Advantage | Alternative Weakness |
|--------|-----------------|---------------------|
| Limited data (500 scans) | Skip connections preserve details with less data | DeepLab needs more data for atrous convolutions |
| Precise boundaries | Symmetric encoder-decoder + skip connections | FCN loses boundary detail |
| Transfer learning | Can use ImageNet pre-trained encoder | ViT needs large pre-training |

**Architecture specifics:**
```
Encoder: ResNet-34 (pre-trained on ImageNet)
  - Transfer learned features reduce data needs
Decoder: U-Net style with skip connections
  - Recovers spatial precision for boundaries
Loss: Dice loss + BCE (handles class imbalance)
```

**Data efficiency strategy:**
- Pre-trained encoder: Only train decoder from scratch
- Augmentation: Rotation, elastic deformation (anatomically plausible)
- 500 scans sufficient with aggressive augmentation

---

**b) Autonomous Driving Scene Parsing (DeepLabV3+)**

**Recommendation:** DeepLabV3+ or real-time variant (BiSeNet)

**Justification:**

| Factor | DeepLabV3+ Advantage | U-Net Limitation |
|--------|---------------------|------------------|
| Multi-scale objects | ASPP captures multiple receptive fields | Fixed receptive field per level |
| Real-time requirement | Efficient backbone options | Symmetric decoder is slower |
| Many classes (20+) | Proven on Cityscapes (19 classes) | Designed for binary/few classes |

**Architecture specifics:**
```
Backbone: MobileNetV3 (real-time) or ResNet-101 (accuracy)
Neck: ASPP (Atrous Spatial Pyramid Pooling)
  - Parallel dilated convolutions: rates [6, 12, 18]
  - Captures cars (large) and pedestrians (small)
Decoder: Simple upsampling with low-level feature fusion
```

**Real-time strategy:**
- BiSeNet variant: Spatial path + Context path
- Target: 30+ FPS on automotive GPU (Jetson Orin)

---

**c) Warehouse Robotics Instance Segmentation (Mask R-CNN)**

**Recommendation:** Mask R-CNN or YOLACT

**Justification:**

| Factor | Mask R-CNN Advantage | Semantic Seg Limitation |
|--------|---------------------|------------------------|
| Individual instances | Detects + segments each product separately | Cannot distinguish two identical products |
| Pick-and-place | Provides bounding box + mask for grasping | Only pixel labels, no instances |
| Varying products | Trained on diverse products, generalizes | Struggles with unseen products |

**Architecture specifics:**
```
Backbone: ResNet-50-FPN
  - FPN handles products of varying sizes
Detection head: Faster R-CNN style
  - RPN proposes product locations
Mask head: Small FCN per detected instance
  - 28×28 binary mask per product
```

**Why not semantic segmentation:**
```
Scene: Two identical boxes side-by-side

Semantic: Both boxes = "box" class (same label)
  → Robot cannot distinguish which to pick

Instance: Box-1 and Box-2 (separate instances)
  → Robot knows exactly which object to grasp
```

---

### Question 4: Convolution and Feature Hierarchy Analysis

**Model Answer:**

**a) Spatial Dimensions and Receptive Field:**

| Layer | Operation | Output Size | Receptive Field |
|-------|-----------|-------------|-----------------|
| Input | - | 224×224×3 | 1×1 |
| Layer 1 | Conv 3×3, s=1, p=1 | 224×224×64 | 3×3 |
| Layer 2 | MaxPool 2×2, s=2 | 112×112×64 | 4×4 |
| Layer 3 | Conv 3×3, s=1, p=1 | 112×112×128 | 8×8 |
| Layer 4 | MaxPool 2×2, s=2 | 56×56×128 | 10×10 |
| Layer 5 | Conv 3×3, s=1, p=1 | 56×56×256 | 14×14 |
| Layer 6 | Global Avg Pool | 1×1×256 | 224×224 |
| Layer 7 | FC | 10 | - |

**Receptive Field Calculation:**
```
RF formula: RF_new = RF_prev + (kernel - 1) × cumulative_stride

Layer 1: RF = 1 + (3-1) × 1 = 3
Layer 2: RF = 3 + (2-1) × 1 = 4 (stride becomes 2)
Layer 3: RF = 4 + (3-1) × 2 = 8
Layer 4: RF = 8 + (2-1) × 2 = 10 (stride becomes 4)
Layer 5: RF = 10 + (3-1) × 4 = 18  [Note: I initially calculated 14, let me recalculate]

Actually:
After pool2, cumulative stride = 2×2 = 4
Layer 5: RF = 10 + (3-1) × 4 = 18

Correction to table: Layer 5 RF = 18×18
```

**b) Features Learned at Each Layer:**

| Layer | Feature Type | Examples |
|-------|-------------|----------|
| **Layer 1** | Low-level edges | Horizontal/vertical edges, color gradients, blobs |
| **Layer 2** | (pooling) | Aggregates edges into texture primitives |
| **Layer 3** | Mid-level textures | Corner patterns, texture patches, simple shapes |
| **Layer 4** | (pooling) | Combines textures into parts |
| **Layer 5** | High-level parts | Object parts (wheels, eyes), complex patterns |
| **Layer 6** | (GAP) | Global object representation |
| **Layer 7** | Semantic | Class-specific decision boundaries |

**Visual Analogy:**
```
Layer 1: "I see edges at 45°, blue-to-red gradient"
Layer 3: "I see a corner pattern with texture"
Layer 5: "I see something that looks like a wheel"
Layer 7: "This is probably a car"
```

**c) Detecting 16×16 Pixel Objects:**

**Analysis:**
```
Final convolutional layer RF: 18×18 pixels

For 16×16 object detection:
- Object size: 16×16
- Receptive field: 18×18 (covers the object)
- RF > Object size: YES ✓

However, consideration needed:
- Object should be 1-2× RF for reliable detection
- 16×16 is ~0.9× of 18×18 RF → borderline
```

**Verdict: Borderline sufficient, but not optimal**

**Justification:**
```
Sufficient because:
- RF (18) > object size (16)
- Each output position "sees" the entire object
- Classification possible

Not optimal because:
- Object fills almost entire RF → no context
- Small position shift changes features significantly
- Better: RF = 2-3× object size for robust detection

Improvement options:
1. Add more layers to increase RF
2. Use FPN for multi-scale detection
3. Use higher resolution input
```

---

### Question 5: Modern Computer Vision System Comparison

**Rubric (30 points total):**

| Component | Excellent (Full) | Adequate (Half) | Insufficient (Minimal) |
|-----------|------------------|-----------------|------------------------|
| Architecture Comparison (8) | Detailed analysis of all three with medical-specific considerations | General comparison without medical context | Missing major architectural details |
| Multi-Task Strategy (7) | Concrete feature sharing plan with conflict resolution | General discussion without specifics | Vague or missing |
| Medical Considerations (7) | Addresses data, explainability, and regulatory comprehensively | Covers some aspects | Ignores domain requirements |
| Recommendation (8) | Well-justified choice with implementation roadmap | Reasonable but incomplete justification | Weak or unsupported |

**Model Answer:**

**1. Architecture Comparison**

**CNN Pipeline (ResNet + Faster R-CNN + U-Net):**
The traditional approach uses specialized architectures for each task. ResNet-50 provides strong feature extraction for classification, achieving high accuracy on ImageNet and transferring well to medical imaging. Faster R-CNN offers precise localization through its two-stage approach, important for identifying specific abnormalities. U-Net excels at segmentation with limited data due to its skip connections preserving spatial detail.

*Strengths:* Each component is optimized for its task; well-understood behavior; extensive medical imaging literature.
*Weaknesses:* No feature sharing between tasks; redundant computation; complex pipeline maintenance.
*Data efficiency:* Moderate—can use ImageNet pre-training but each task trained separately.
*Compute:* High—three separate forward passes per image.

**Unified Transformer (ViT-based):**
A single Vision Transformer with task-specific heads processes images through patch embeddings and self-attention layers. The [CLS] token feeds classification, while spatial tokens enable detection and segmentation.

*Strengths:* Unified feature extraction; attention provides natural explainability; scales well with data.
*Weaknesses:* Requires large pre-training data (ViT needs 300M+ images for optimal performance); medical datasets are typically 10-100K images.
*Data efficiency:* Poor without ImageNet-21K pre-training; even with pre-training, fine-tuning requires substantial data.
*Compute:* Moderate training, but efficient inference once trained.

**Hybrid CNN-Transformer:**
Combines CNN backbone (efficient local features) with Transformer attention (global reasoning). Example: ResNet backbone feeding Transformer encoder for multi-task heads.

*Strengths:* CNN inductive bias helps with limited data; Transformer attention captures long-range pathology relationships; balanced compute.
*Weaknesses:* Architectural complexity; hyperparameter tuning for hybrid components.
*Data efficiency:* Good—CNN backbone transfers from ImageNet; Transformer learns task relationships with moderate data.

**2. Multi-Task Learning Strategy**

**Feature Sharing Architecture:**
```
Shared Backbone (ResNet-50 or CNN-Transformer hybrid)
    ↓
Task-Specific Heads:
  ├── Classification head (Global pooling → FC)
  ├── Detection head (FPN → RPN → RoI heads)
  └── Segmentation head (Decoder with skip connections)
```

**Task Conflicts and Resolution:**
Classification wants global features summarizing the entire image, while segmentation needs pixel-level detail—these can conflict. Detection falls between, needing both localization and recognition.

*Resolution strategies:*
- **Gradient normalization:** Scale task gradients to similar magnitudes
- **Uncertainty weighting:** Learn task weights based on homoscedastic uncertainty
- **Feature pyramid sharing:** Classification uses high-level features (P5), segmentation uses multi-scale (P2-P5)

**Training Approach:**
Sequential pre-training followed by joint fine-tuning works best for medical imaging:
1. Pre-train backbone on ImageNet
2. Fine-tune classification first (most labeled data typically)
3. Add detection/segmentation heads, joint training with classification frozen
4. Final joint fine-tuning with all tasks

**3. Medical Domain Considerations**

**Limited Labeled Data:**
Medical datasets are typically 1-50K images with expensive expert annotations. Transfer learning is essential:
- ImageNet pre-training provides general visual features
- Domain-specific pre-training on unlabeled X-rays (self-supervised) further improves
- Few-shot learning techniques for rare pathologies

**Explainability Requirements:**
Clinicians must understand model decisions for trust and liability. Approaches:
- **Attention visualization:** Show which image regions influenced classification
- **GradCAM:** Generate saliency maps for CNN decisions
- **Detection confidence:** Clearly communicate uncertainty in localizations
- **Segmentation overlay:** Visual confirmation of affected regions

For FDA approval, explainability documentation is increasingly required. The hybrid approach naturally provides attention maps from the Transformer component.

**Regulatory Considerations:**
FDA Class II medical device clearance requires:
- Clinical validation studies with diverse patient populations
- Clear documentation of model limitations
- Monitoring protocols for deployed systems
- Version control and change management

Single-model approaches (unified Transformer) simplify regulatory documentation compared to multi-model pipelines.

**4. Recommendation**

**Recommended: Hybrid CNN-Transformer approach**

**Justification:**
1. *Data efficiency:* CNN backbone leverages ImageNet pre-training effectively with limited medical data
2. *Multi-task capability:* Shared features reduce annotation requirements; Transformer attention enables task interaction
3. *Explainability:* Attention weights provide interpretable visualization for clinician review
4. *Regulatory simplicity:* Single model easier to validate than three-model pipeline

**Implementation Roadmap:**

*Phase 1 (Months 1-3):* Baseline development
- Implement ResNet-50 backbone with Transformer encoder
- Classification head only; validate on public chest X-ray datasets
- Target: 90%+ AUC on normal/abnormal classification

*Phase 2 (Months 4-6):* Multi-task extension
- Add detection head for abnormality localization
- Add segmentation head for affected region delineation
- Joint training with task weighting

*Phase 3 (Months 7-9):* Clinical validation
- Partner with hospitals for prospective evaluation
- Explainability interface development
- FDA pre-submission meeting

**Risk Mitigation:**
- *Data scarcity:* Augment with synthetic data; self-supervised pre-training on unlabeled X-rays
- *Task conflict:* Monitor per-task metrics; fall back to separate models if joint training degrades
- *Regulatory delays:* Parallel path—deploy research-use-only version while pursuing clearance

The hybrid approach balances the data efficiency of CNNs with the flexibility and explainability of Transformers, making it well-suited for medical imaging where labeled data is precious and clinical interpretability is mandatory.

---

## Performance Interpretation Guide

| Score | Mastery Level | Recommended Action |
|-------|---------------|-------------------|
| 90-100% | **Mastery** | Ready for CV research and production systems |
| 75-89% | **Proficient** | Review specific gaps, implement CV projects |
| 60-74% | **Developing** | Re-study core architecture concepts |
| Below 60% | **Foundational** | Complete re-review of Lesson 10 |

---

## Review Recommendations by Question

| If You Struggled With | Review These Sections |
|----------------------|----------------------|
| Question 1 | Lesson 10: Residual connections, gradient flow |
| Question 2 | Lesson 10: Object detection frameworks, YOLO vs R-CNN |
| Question 3 | Lesson 10: Segmentation architectures, U-Net, DeepLab, Mask R-CNN |
| Question 4 | Lesson 10: Convolution, receptive field, feature hierarchy |
| Question 5 | Lesson 10: Modern advances, ViT, multi-task learning |

---

*Generated from Lesson 10: Computer Vision | Quiz Skill*
