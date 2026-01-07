# Assessment Quiz: Computer Vision

**Source Material:** notes/computer-vision/flashcards/computer-vision-flashcards.md
**Practice Problems:** notes/computer-vision/practice/computer-vision-practice-problems.md
**Concept Map:** notes/computer-vision/concept-maps/computer-vision-concept-map.md
**Original Study Notes:** notes/computer-vision/computer-vision-study-notes.md
**Date Generated:** 2026-01-06
**Total Questions:** 5
**Estimated Completion Time:** 30-40 minutes
**Distribution:** 2 Multiple Choice | 2 Short Answer | 1 Essay

---

## Instructions

- **Multiple Choice:** Select the single best answer
- **Short Answer:** Respond in 2-4 sentences
- **Essay:** Provide a comprehensive response (1-2 paragraphs)

---

## Questions

### Question 1 | Multiple Choice
**Cognitive Level:** Remember/Understand
**Concept Tested:** Convolution Operation
**Source Section:** Core Concepts 2
**Concept Map Node:** Convolution (10 connections)
**Related Flashcard:** Card 1
**Related Practice Problem:** P1

An input feature map of size 64×64×32 is processed by a Conv2D layer with 64 filters, kernel size 3×3, stride 2, and padding 'same'. What is the output shape, and how many parameters does this layer have (including biases)?

A) Output: 32×32×64; Parameters: 18,496

B) Output: 31×31×64; Parameters: 18,432

C) Output: 32×32×64; Parameters: 18,432

D) Output: 64×64×64; Parameters: 18,496

---

### Question 2 | Multiple Choice
**Cognitive Level:** Remember/Understand
**Concept Tested:** Skip Connections
**Source Section:** Core Concepts 6
**Concept Map Node:** Skip Connections (7 connections)
**Related Flashcard:** Card 2
**Related Practice Problem:** P2

In a ResNet block, the skip connection adds the input x directly to the output of the convolutional layers F(x). Which statement best explains why this enables training of much deeper networks?

A) Skip connections reduce the total number of parameters, making optimization faster and preventing overfitting

B) Skip connections allow gradient to flow directly through the identity path, preventing vanishing gradients, and make learning identity mappings trivial (just push F(x) toward 0)

C) Skip connections provide additional non-linearity that helps the network learn more complex features

D) Skip connections enable parallel processing of different input regions, speeding up computation

---

### Question 3 | Short Answer
**Cognitive Level:** Apply/Analyze
**Concept Tested:** Object Detection Metrics
**Source Section:** Core Concepts 7
**Concept Map Node:** IoU (4), NMS (3)
**Related Flashcard:** Card 3
**Related Practice Problem:** P3
**Expected Response Length:** 3-4 sentences

A detection model outputs 5 bounding boxes for cars in an image with confidences [0.9, 0.85, 0.7, 0.6, 0.4]. After applying NMS with IoU threshold 0.5, only 2 boxes remain. Explain what NMS accomplished, what the IoU threshold controls, and describe a scenario where lowering the threshold to 0.3 would be problematic.

---

### Question 4 | Short Answer
**Cognitive Level:** Apply/Analyze
**Concept Tested:** Segmentation Architecture
**Source Section:** Core Concepts 10
**Concept Map Node:** U-Net (5), Skip Connections in Segmentation
**Related Flashcard:** Card 4
**Related Practice Problem:** P4
**Expected Response Length:** 3-4 sentences

U-Net uses skip connections between the encoder and decoder, while a simple encoder-decoder without skip connections exists. Explain what information is lost during encoding that the skip connections preserve, and why this is particularly important for medical image segmentation where precise boundary delineation is required.

---

### Question 5 | Essay
**Cognitive Level:** Evaluate/Synthesize
**Concepts Tested:** Complete Vision Pipeline
**Source Sections:** All Core Concepts
**Concept Map:** Full pathway traversal
**Related Flashcard:** Cards 3, 4, 5
**Related Practice Problem:** P4, P5
**Expected Response Length:** 1-2 paragraphs

You are designing a computer vision system for a warehouse that must: (1) detect packages on conveyor belts, (2) read barcodes/QR codes on packages, (3) identify damaged packages, and (4) operate at 60 FPS on edge devices with 4GB memory. The system will process 1280×720 video from 4 cameras.

Design a complete solution addressing: (a) how you would handle the three different tasks (detection, OCR, damage classification)—single multi-task model or separate models?; (b) architecture selection for each component with memory/speed justification; (c) how you would handle the 4-camera requirement within the compute budget; (d) training data strategy for the damage detection task where damaged packages are rare; and (e) what failure modes you would monitor in production.

**Evaluation Criteria:**
- [ ] Justifies multi-task vs. separate model decision
- [ ] Selects appropriate architectures with resource reasoning
- [ ] Addresses multi-camera processing within constraints
- [ ] Proposes solution for rare damage class
- [ ] Identifies meaningful failure modes

---

## Answer Key

### Question 1 | Multiple Choice
**Correct Answer:** A

**Explanation:**
**Output shape calculation:**
- With 'same' padding and stride=2: H_out = ceil(H_in / stride) = ceil(64/2) = 32
- Similarly, W_out = 32
- Number of output channels = number of filters = 64
- Output shape: **32×32×64**

**Parameter calculation:**
- Each filter has size: 3×3×32 (kernel × kernel × input_channels)
- Parameters per filter: 3×3×32 = 288
- Total filters: 64
- Weights: 288×64 = 18,432
- Biases: 64 (one per filter)
- Total parameters: 18,432 + 64 = **18,496**

**Why Other Options Are Incorrect:**
- B) Correct parameters but wrong output size ('valid' would give 31×31, but 'same' gives 32×32)
- C) Correct output but missing biases in parameter count
- D) Wrong output size (stride=2 halves spatial dimensions) though parameters are correct

**Understanding Gap Indicator:**
If answered incorrectly, review the convolution dimension formula and parameter counting in Card 1 and Practice Problem 1.

---

### Question 2 | Multiple Choice
**Correct Answer:** B

**Explanation:**
Skip connections solve two problems that prevented training very deep networks:

**1. Vanishing Gradients:**
During backpropagation, gradients multiply through layers. With many layers, gradients can shrink exponentially, preventing early layers from learning. The skip connection provides a direct path:
```
∂L/∂x = ∂L/∂(F(x)+x) × (∂F/∂x + 1)
                              │
                    Identity term ensures gradient ≥ original
```

**2. Degradation Problem:**
Without skip connections, deeper networks performed *worse* than shallower ones on training data (not just overfitting). This occurs because learning an identity mapping H(x)=x through convolutions is difficult. With skip connections:
- To learn identity: just push F(x) → 0
- Network learns F(x) = H(x) - x (the residual)
- Output is F(x) + x = H(x)

**Why Other Options Are Incorrect:**
- A) Skip connections don't reduce parameters; they add them if projection is needed
- C) Skip connections add linearity (identity), not non-linearity
- D) Skip connections don't enable parallelism; computation is still sequential

**Understanding Gap Indicator:**
If answered incorrectly, review the residual learning explanation in Card 2 and Practice Problem 2.

---

### Question 3 | Short Answer
**Model Answer:**

NMS (Non-Maximum Suppression) eliminated 3 redundant boxes that likely detected the same car multiple times, keeping only the 2 best non-overlapping detections. The IoU threshold (0.5) determines how much overlap is tolerated—boxes with IoU > 0.5 with a higher-confidence box are suppressed as duplicates. Lowering the threshold to 0.3 would be problematic in crowded scenes with closely parked cars: two genuinely different vehicles that happen to partially overlap (e.g., one car behind another with 35% IoU) would be incorrectly suppressed, causing the detector to miss one car entirely. This creates a tradeoff between removing duplicates and preserving distinct but nearby objects.

**Key Components Required:**
- [ ] Explains NMS removes duplicate detections
- [ ] Explains IoU threshold controls overlap tolerance
- [ ] Describes problematic scenario with low threshold (suppressing distinct objects)

**Partial Credit Guidance:**
- Full credit: All three components with clear explanation
- Partial credit: Explains NMS purpose but vague on threshold effects
- No credit: Misunderstands NMS or IoU

**Understanding Gap Indicator:**
If answered poorly, review IoU calculation and NMS algorithm in Practice Problem 3.

---

### Question 4 | Short Answer
**Model Answer:**

During encoding, the progressive pooling/striding reduces spatial resolution (e.g., 512×512 → 32×32), losing fine-grained spatial information about exact edge locations, small structures, and texture details. While the encoder captures *what* features exist (semantic information), it loses *where* precisely they are located. Skip connections preserve this high-resolution spatial information by concatenating encoder feature maps directly to the decoder, allowing the network to combine deep semantic understanding with precise localization.

For medical segmentation, this is critical because tumor boundaries, vessel edges, or lesion margins must be delineated with pixel-level precision. A 1-2 pixel error in boundary detection could mean the difference between accurately measuring tumor size or missing a small metastasis. Without skip connections, the decoder must "hallucinate" fine details, resulting in blurry, imprecise boundaries—unacceptable for clinical decisions like surgical planning or treatment monitoring.

**Key Components Required:**
- [ ] Identifies spatial/localization information lost during encoding
- [ ] Explains skip connections preserve high-resolution details
- [ ] Connects to medical imaging need for precise boundaries

**Partial Credit Guidance:**
- Full credit: Clear explanation of information loss, skip connection role, and medical relevance
- Partial credit: Understands skip connections help but vague on what information is preserved
- No credit: Misunderstands encoder-decoder architecture

**Understanding Gap Indicator:**
If answered poorly, review U-Net architecture in Card 4 and the encoder-decoder explanation.

---

### Question 5 | Essay
**Model Answer:**

**Task Architecture Decision:**
I would use a **hybrid approach**: a shared backbone with task-specific heads for detection and damage classification, plus a separate lightweight OCR model. The detection and damage tasks both require understanding package visual features (shape, edges, anomalies), making feature sharing beneficial and memory-efficient. OCR is fundamentally different—it processes text regions with different input requirements—so a separate specialized model is appropriate. This architecture avoids the complexity of a single monolithic model while enabling efficient shared computation.

**Component Architectures:**

For **detection + damage**, I would use YOLOv8-nano (3.2M parameters, ~6MB) with two output heads: one for package bounding boxes (single class) and one for damage classification (binary: damaged/undamaged) applied to detected regions. This fits easily in 4GB with room for multiple inference streams. The nano variant achieves ~100+ FPS on edge GPUs like Jetson Nano at 640×480 resolution.

For **OCR**, I would use a lightweight model like PaddleOCR-mobile or TrOCR-small, processing only the detected package regions (cropped and rectified). Running OCR only on detected regions (~100×100 crops) rather than full frames dramatically reduces computation. Total OCR overhead: ~10ms per package.

**Multi-Camera Processing:**
With 4 cameras at 60 FPS = 240 frames/second total throughput required. On 4GB edge device:
- Downsample input to 640×480 (sufficient for package detection)
- Batch process 4 frames simultaneously (one per camera)
- Target: 240 FPS / 4 = 60 FPS per batch, or ~17ms per batch
- YOLOv8-nano at 640×480 achieves ~8ms per frame with TensorRT
- 4-frame batch: ~12-15ms achievable with proper batching

Alternatively, process at 30 FPS per camera (still sufficient for conveyor speeds) for comfortable headroom.

**Damage Detection Training Strategy:**
Damaged packages are rare (~1-2%), creating severe class imbalance. Solutions:
1. **Oversampling**: Replicate damaged examples 5-10×
2. **Synthetic damage**: Generate training data by artificially adding dents, tears, water stains to normal packages using image editing or GANs
3. **Transfer learning**: Pre-train damage classifier on general anomaly detection datasets
4. **Focal loss**: Down-weight easy (undamaged) examples, focus learning on hard (damaged) cases
5. **Active learning**: Deploy initial model, collect predictions near decision boundary for human labeling

Combine synthetic generation (to bootstrap) with active learning (to improve on real distribution).

**Production Failure Modes:**
Monitor:
1. **Detection miss rate**: Packages passing without detection (verify with downstream sensors)
2. **OCR failure rate**: Barcodes detected but not decoded (flag for manual entry)
3. **Damage false positives**: High rate causes unnecessary diverts (costly)
4. **Damage false negatives**: Damaged packages shipped (customer complaints)
5. **Latency spikes**: Frames dropped = packages missed
6. **Camera degradation**: Blur, occlusion, lighting changes affecting accuracy
7. **Distribution shift**: New package types, label designs not in training data

Set alerts for each metric; establish automated retraining pipeline when thresholds exceeded.

**Evaluation Rubric:**

| Criterion | Excellent (4) | Proficient (3) | Developing (2) | Beginning (1) |
|-----------|---------------|----------------|----------------|---------------|
| Architecture Decision | Clear multi-task vs separate reasoning | Reasonable choice with some justification | Makes a choice but weak reasoning | No justification |
| Model Selection | Specific models with memory/speed numbers | Appropriate models, less specific | Generic models named | No model selection |
| Multi-Camera | Concrete throughput calculation | Addresses problem with solution | Mentions challenge | Ignores constraint |
| Class Imbalance | Multiple concrete techniques | 1-2 techniques | Mentions problem | Ignores imbalance |
| Failure Modes | 5+ specific, monitored metrics | 3-4 failure modes | 1-2 generic issues | No failure consideration |

**Understanding Gap Indicator:**
If response lacks depth, this may indicate:
- Limited experience with edge deployment constraints
- Weak understanding of multi-task learning tradeoffs
- Insufficient attention to production monitoring

---

## Diagnostic Feedback & Review Recommendations

### Performance-Based Review Guide

| If Incorrect On... | Gap Identified | Review Section | Priority |
|--------------------|----------------|----------------|----------|
| Question 1 | Convolution mechanics | Core Concepts 2 + Practice P1 | High |
| Question 2 | Skip connection purpose | Core Concepts 6 + Flashcard 2 | High |
| Question 3 | Detection post-processing | Core Concepts 7 + Practice P3 | Medium |
| Question 4 | Segmentation architecture | Core Concepts 10 + Flashcard 4 | Medium |
| Question 5 | Full system integration | All sections | Low |

### Targeted Review Recommendations

#### For Multiple Choice Errors (Questions 1-2)
**Indicates:** Foundational concept gaps
**Action:** Review:
- Study Notes: Core Concepts 2 (Convolution) and 6 (Skip Connections)
- Flashcards: Cards 1 and 2
- Practice Problems: P1 (dimension calculation) and P2 (ResNet block)
**Focus On:** Understanding the mechanics, not just memorizing formulas

#### For Short Answer Errors (Questions 3-4)
**Indicates:** Application difficulties
**Action:** Practice:
- Practice Problems: P3 (IoU/NMS calculation) and P4 (detection system)
- Concept Map: Detection and Segmentation clusters
**Focus On:** Connecting algorithms to their practical purposes

#### For Essay Weakness (Question 5)
**Indicates:** Integration challenges
**Action:** Review interconnections:
- Concept Map: Full pathway traversal
- Practice Problem P4 and P5 for system thinking
- Study Notes: Practical Applications section
**Focus On:** Building complete systems from components

### Mastery Indicators

- **5/5 Correct:** Strong mastery; ready for computer vision projects
- **4/5 Correct:** Good understanding; review indicated gap
- **3/5 Correct:** Moderate understanding; systematic review needed
- **2/5 or below:** Foundational gaps; restart from Concept Map Critical Path

---

## Skill Chain Traceability

```
Study Notes (Source) ──────────────────────────────────────────────────────┐
    │                                                                      │
    │  10 Core Concepts, 13 Key Terms, 4 Applications                      │
    │                                                                      │
    ├────────────┬────────────┬────────────┬────────────┐                  │
    │            │            │            │            │                  │
    ▼            ▼            ▼            ▼            ▼                  │
Concept Map  Flashcards   Practice    Quiz                                 │
    │            │        Problems      │                                  │
    │ 28 concepts│ 5 cards   │ 5 problems│ 5 questions                     │
    │ 45 rels    │ 2E/2M/1H  │ 1W/2S/1C/1D│ 2MC/2SA/1E                     │
    │ 4 pathways │           │           │                                 │
    │            │           │           │                                 │
    └─────┬──────┴─────┬─────┴─────┬─────┘                                 │
          │            │           │                                       │
          │ Centrality │ Practice  │                                       │
          │ → Card     │ → Quiz    │                                       │
          │ difficulty │ distractors│                                      │
          │            │           │                                       │
          └────────────┴───────────┴───────────────────────────────────────┘
                                   │
                          Quiz integrates ALL
                          upstream materials
```

---

## Complete 5-Skill Chain Summary

| Skill | Output | Key Contribution to Chain |
|-------|--------|---------------------------|
| study-notes-creator | 10 concepts, theory, applications | Foundation content |
| concept-map | 28 nodes, 45 relationships, 4 pathways | Structure + learning paths |
| flashcards | 5 cards (2E/2M/1H) | Memorization + critical flags |
| practice-problems | 5 problems (1W/2S/1C/1D) | Procedural fluency + common mistakes |
| quiz | 5 questions (2MC/2SA/1E) | Assessment + diagnostic feedback |
