# Assessment Quiz: Deep Learning

**Source Material:** notes/deep-learning/flashcards/deep-learning-flashcards.md
**Practice Problems:** notes/deep-learning/practice/deep-learning-practice-problems.md
**Concept Map:** notes/deep-learning/concept-maps/deep-learning-concept-map.md
**Original Study Notes:** notes/deep-learning/deep-learning-study-notes.md
**Date Generated:** 2026-01-07
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
**Concept Tested:** Optimizers
**Source Section:** Core Concepts 4
**Concept Map Node:** Adam (11 connections)
**Related Flashcard:** Card 1
**Related Practice Problem:** P1

What is the primary advantage of Adam over SGD with momentum, and what is the primary advantage of SGD with momentum over Adam?

A) Adam uses less memory; SGD converges faster

B) Adam has adaptive per-parameter learning rates enabling faster convergence; SGD often achieves better final generalization with proper tuning

C) Adam handles sparse gradients better; SGD is more stable with large batch sizes

D) Adam requires no hyperparameter tuning; SGD works better with non-convex loss landscapes

---

### Question 2 | Multiple Choice
**Cognitive Level:** Remember/Understand
**Concept Tested:** Batch Normalization
**Source Section:** Core Concepts 7
**Concept Map Node:** Batch Normalization (7 connections)
**Related Flashcard:** Card 3
**Related Practice Problem:** P3

Which statement correctly describes the behavior difference between Batch Normalization during training versus inference?

A) During training, BatchNorm uses batch statistics; during inference, it uses running statistics accumulated during training, because test batches may be size 1 or have different distributions

B) During training, BatchNorm normalizes to mean=0, std=1; during inference, it skips normalization entirely to preserve the learned representation

C) During training, BatchNorm uses running statistics; during inference, it computes fresh statistics from the test batch for accuracy

D) There is no difference; BatchNorm always computes statistics from the current batch regardless of training or inference mode

---

### Question 3 | Short Answer
**Cognitive Level:** Apply/Analyze
**Concept Tested:** Skip Connections
**Source Section:** Core Concepts 8
**Concept Map Node:** Skip Connections (9 connections)
**Related Flashcard:** Card 2
**Related Practice Problem:** P2
**Expected Response Length:** 3-4 sentences

A student asks: "If skip connections just add the input to the output, doesn't that mean the network learns nothing useful?" Explain why this reasoning is incorrect and what skip connections actually enable the network to learn.

---

### Question 4 | Short Answer
**Cognitive Level:** Apply/Analyze
**Concept Tested:** Regularization Selection
**Source Section:** Core Concepts 6
**Concept Map Node:** Dropout (6), Weight Decay (3)
**Related Flashcard:** Card 3
**Related Practice Problem:** P3
**Expected Response Length:** 3-4 sentences

You observe that your model achieves 98% training accuracy but only 72% validation accuracy after 50 epochs. Training loss is 0.05 while validation loss is 1.2. Describe two different regularization approaches you would apply and explain the mechanism by which each addresses this specific problem.

---

### Question 5 | Essay
**Cognitive Level:** Evaluate/Synthesize
**Concepts Tested:** Complete Training Pipeline
**Source Sections:** All Core Concepts
**Concept Map:** Full pathway traversal
**Related Flashcard:** Card 5
**Related Practice Problem:** P4
**Expected Response Length:** 1-2 paragraphs

You are training a deep neural network for autonomous vehicle perception that must classify road objects (vehicles, pedestrians, cyclists, signs) from camera images. The system must: (1) achieve >95% accuracy on safety-critical classes (pedestrians, cyclists), (2) run inference at 30 FPS on an embedded GPU with 8GB memory, (3) handle varying lighting conditions (day, night, rain, fog), and (4) be robust to rare edge cases like unusual vehicle types or partially occluded pedestrians.

Design a complete training strategy addressing: (a) architecture selection balancing accuracy and inference speed; (b) loss function design to prioritize safety-critical classes; (c) data augmentation strategy for lighting and weather robustness; (d) regularization approach to improve generalization to edge cases; and (e) what metrics beyond accuracy you would monitor to ensure the model is production-ready.

**Evaluation Criteria:**
- [ ] Selects efficient architecture with speed justification
- [ ] Designs appropriate loss weighting for safety
- [ ] Proposes domain-relevant augmentation
- [ ] Addresses edge case generalization
- [ ] Identifies safety-relevant monitoring metrics

---

## Answer Key

### Question 1 | Multiple Choice
**Correct Answer:** B

**Explanation:**
Adam and SGD with momentum represent different tradeoffs:

**Adam's Advantage - Faster Convergence:**
- Maintains per-parameter adaptive learning rates
- First moment (momentum) + second moment (RMSprop-like scaling)
- Parameters with large gradients get smaller updates (and vice versa)
- Less sensitive to initial learning rate choice
- Typically converges in fewer epochs

**SGD's Advantage - Better Generalization:**
- Empirical observation: SGD often reaches flatter minima
- Flatter minima correlate with better generalization
- The "noise" in SGD updates provides implicit regularization
- With proper tuning (LR schedule), often achieves 0.5-1% higher final accuracy

**Research insight:** The adaptive learning rates in Adam may converge to sharper minima that generalize slightly worse.

**Why Other Options Are Incorrect:**
- A) Adam actually uses *more* memory (stores first and second moments); SGD typically converges *slower*
- C) Both handle sparse gradients; large batch behavior isn't the key difference
- D) Both require hyperparameter tuning; both work on non-convex landscapes

---

### Question 2 | Multiple Choice
**Correct Answer:** A

**Explanation:**
Batch Normalization behaves differently in training vs. inference:

**Training Mode:**
```python
# Compute batch statistics
μ_B = mean(x, dim=batch)
σ_B = std(x, dim=batch)

# Normalize using batch statistics
x_norm = (x - μ_B) / σ_B

# Update running statistics (exponential moving average)
running_mean = 0.9 * running_mean + 0.1 * μ_B
running_var = 0.9 * running_var + 0.1 * σ_B²
```

**Inference Mode:**
```python
# Use accumulated running statistics (NOT batch statistics)
x_norm = (x - running_mean) / sqrt(running_var)
```

**Why this matters:**
- Test batch may be size 1 → batch statistics meaningless
- Test distribution may differ from training → running stats more stable
- Ensures deterministic inference (same input → same output)

**Critical reminder:** Always call `model.eval()` before inference!

**Why Other Options Are Incorrect:**
- B) BatchNorm still normalizes during inference; it doesn't skip anything
- C) Reversed: training uses batch stats, inference uses running stats
- D) There IS a crucial difference; this is a common bug source

---

### Question 3 | Short Answer
**Model Answer:**

The student's reasoning confuses "adding" with "doing nothing." Skip connections add the input x to F(x), where F(x) is the transformation learned by the convolutional layers. The network doesn't learn "nothing"—it learns the residual F(x) = H(x) - x, which represents the *difference* between the desired output and the input. This is actually easier to learn than the full transformation H(x) directly.

The key insight is that if the optimal transformation is close to identity (common in deep networks), F(x) just needs to be pushed toward zero—a much simpler learning target than constructing an identity mapping through convolutions. Additionally, skip connections provide a direct gradient pathway during backpropagation, ensuring that even early layers receive meaningful gradient signals. Without skip connections, gradients must flow through every transformation, potentially vanishing or exploding over 100+ layers.

**Key Components Required:**
- [ ] Clarifies that network learns F(x), not "nothing"
- [ ] Explains residual learning concept (learning the difference)
- [ ] Notes that learning F(x)≈0 is easier than learning H(x)=x
- [ ] Mentions gradient flow benefit

**Partial Credit Guidance:**
- Full credit: All components with clear explanation
- Partial credit: Understands residual concept but vague on why it helps
- No credit: Cannot explain what the network actually learns

---

### Question 4 | Short Answer
**Model Answer:**

The metrics clearly indicate overfitting: the 26% accuracy gap and 24× loss ratio show the model has memorized training data rather than learning generalizable patterns. Two effective interventions:

**1. Dropout (e.g., p=0.5 before final layers):** Randomly zeroes activations during training, forcing the network to learn redundant representations that don't rely on any single neuron. This prevents co-adaptation where neurons become overly specialized to training examples. At test time, all neurons are active but scaled, providing an ensemble effect.

**2. Strong Data Augmentation (random crops, flips, color jitter, mixup):** Artificially expands the effective training set by presenting varied versions of each image. This prevents memorization of exact pixel patterns and forces the network to learn invariant features. For a gap this large, aggressive augmentation like mixup (blending images and labels) or CutMix would be particularly effective.

**Key Components Required:**
- [ ] Correctly diagnoses overfitting from metrics
- [ ] Proposes two distinct regularization approaches
- [ ] Explains mechanism of each (how it prevents overfitting)
- [ ] Mechanisms are specific and accurate

**Partial Credit Guidance:**
- Full credit: Both techniques with accurate mechanisms
- Partial credit: Correct techniques but vague explanations
- No credit: Wrong diagnosis or mechanisms

---

### Question 5 | Essay
**Model Answer:**

**(a) Architecture Selection:**

I would use **EfficientNet-B0 or MobileNetV3-Large** as the base architecture. EfficientNet-B0 achieves ~77% ImageNet top-1 accuracy with only 5.3M parameters and ~0.39B FLOPs, enabling real-time inference on embedded GPUs. MobileNetV3 is specifically optimized for mobile/embedded deployment with hardware-aware neural architecture search. Either architecture fits comfortably in 8GB memory and can achieve 30+ FPS on modern embedded GPUs (Jetson Xavier: ~50 FPS for EfficientNet-B0). I would avoid larger models like ResNet-50 (25M params) which would sacrifice inference speed without proportional accuracy gains for this task.

**(b) Loss Function for Safety-Critical Classes:**

```python
# Class weights: higher for pedestrians/cyclists
class_weights = {
    'vehicle': 1.0,
    'pedestrian': 3.0,    # Safety-critical
    'cyclist': 3.0,       # Safety-critical
    'sign': 1.5,
    'background': 0.5
}

# Focal loss to handle imbalance + weight critical classes
loss = FocalLoss(gamma=2.0, weights=class_weights)

# Additionally, monitor per-class recall for pedestrian/cyclist
# Set threshold: must achieve >98% recall on safety classes before deployment
```

The weighted focal loss down-weights easy examples (confident correct predictions) while up-weighting hard examples and safety-critical classes. This ensures the model doesn't achieve high average accuracy by excelling at common classes while failing on critical ones.

**(c) Data Augmentation for Lighting/Weather:**

```python
train_transform = transforms.Compose([
    # Geometric
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomPerspective(distortion_scale=0.2),

    # Lighting/Weather simulation
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.1),  # Simulate IR/night vision
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),  # Rain/fog blur

    # Domain-specific
    RandomRain(p=0.2),      # Synthetic rain drops
    RandomFog(p=0.2),       # Atmospheric fog
    RandomSunGlare(p=0.1),  # Sun reflections
    RandomNightMode(p=0.2), # Low-light simulation

    # Robust augmentation
    transforms.RandAugment(num_ops=2, magnitude=9),
])
```

For autonomous vehicles, I would supplement synthetic augmentation with real adverse-weather data collection and potentially use domain adaptation techniques (e.g., training on synthetic fog then fine-tuning on real fog data).

**(d) Regularization for Edge Cases:**

Edge cases (unusual vehicles, partial occlusions) suffer from limited training examples. Solutions:

1. **Mixup/CutMix augmentation:** Blends images creating novel combinations; exposes model to "in-between" examples it hasn't seen
2. **Label smoothing (0.1):** Prevents overconfident predictions; model maintains uncertainty on unusual inputs
3. **Test-time augmentation (TTA):** During inference on low-confidence predictions, run multiple augmented versions and average
4. **Uncertainty estimation:** Use MC Dropout or ensemble to flag predictions where the model is uncertain for human review

**(e) Production Monitoring Metrics:**

| Metric | Threshold | Purpose |
|--------|-----------|---------|
| Pedestrian recall | >98% | Safety-critical detection rate |
| Cyclist recall | >98% | Safety-critical detection rate |
| False positive rate | <2% | Avoid unnecessary braking |
| Latency p99 | <33ms | Real-time requirement (30 FPS) |
| Confidence calibration | ECE <0.05 | Reliable uncertainty estimates |
| Night-mode accuracy | Within 5% of day | Weather robustness |
| OOD detection rate | Flag >90% of unknown objects | Novel object handling |

**Additional monitoring:**
- Per-class confusion matrices updated daily
- Attention map visualization on failure cases
- Confidence histogram drift detection
- Shadow model comparison (run updated model alongside production)

**Evaluation Rubric:**

| Criterion | Excellent (4) | Proficient (3) | Developing (2) | Beginning (1) |
|-----------|---------------|----------------|----------------|---------------|
| Architecture | Specific model with FPS numbers | Appropriate choice, less detail | Generic "efficient" model | No consideration of constraints |
| Loss Design | Weighted focal loss with safety rationale | Some class weighting | Mentions imbalance | Standard cross-entropy |
| Augmentation | Domain-specific weather simulation | Standard + some domain | Basic augmentation | None mentioned |
| Edge Cases | Multiple techniques with mechanisms | 1-2 techniques | Mentions problem | Ignores edge cases |
| Monitoring | 5+ metrics with thresholds | 3-4 metrics | 1-2 generic metrics | Only accuracy |

---

## Diagnostic Feedback & Review Recommendations

### Performance-Based Review Guide

| If Incorrect On... | Gap Identified | Review Section | Priority |
|--------------------|----------------|----------------|----------|
| Question 1 | Optimizer understanding | Core Concepts 4 + Practice P1 | High |
| Question 2 | BatchNorm mechanics | Core Concepts 7 + Flashcard 3 | High |
| Question 3 | Skip connection purpose | Core Concepts 8 + Practice P2 | Medium |
| Question 4 | Regularization application | Core Concepts 6 + Practice P3 | Medium |
| Question 5 | System integration | All sections | Low |

### Targeted Review Recommendations

#### For Multiple Choice Errors (Questions 1-2)
**Indicates:** Foundational concept gaps
**Action:** Review:
- Study Notes: Core Concepts 4 (Optimizers) and 7 (Normalization)
- Flashcards: Cards 1 and 3
- Practice Problems: P1 (optimizer selection) and P5 (BatchNorm in debug)
**Focus On:** Understanding mechanisms, not just memorizing when to use each

#### For Short Answer Errors (Questions 3-4)
**Indicates:** Application difficulties
**Action:** Practice:
- Practice Problems: P2 (ResNet design) and P3 (overfitting diagnosis)
- Concept Map: Architecture and Regularization clusters
**Focus On:** Connecting technique mechanisms to their effects

#### For Essay Weakness (Question 5)
**Indicates:** Integration challenges
**Action:** Review interconnections:
- Concept Map: Full pathway traversal
- Practice Problem P4 (complete pipeline design)
- Study Notes: All practical applications
**Focus On:** Building complete systems with justified tradeoffs

### Mastery Indicators

- **5/5 Correct:** Strong mastery; ready for deep learning projects
- **4/5 Correct:** Good understanding; review indicated gap
- **3/5 Correct:** Moderate understanding; systematic review needed
- **2/5 or below:** Foundational gaps; restart from Concept Map Critical Path

---

## Skill Chain Traceability

```
Study Notes (Source) ──────────────────────────────────────────────────────┐
    │                                                                      │
    │  10 Core Concepts, 12 Key Terms, 4 Applications                      │
    │                                                                      │
    ├────────────┬────────────┬────────────┬────────────┐                  │
    │            │            │            │            │                  │
    ▼            ▼            ▼            ▼            ▼                  │
Concept Map  Flashcards   Practice    Quiz                                 │
    │            │        Problems      │                                  │
    │ 35 concepts│ 5 cards   │ 5 problems│ 5 questions                     │
    │ 52 rels    │ 2E/2M/1H  │ 1W/2S/1C/1D│ 2MC/2SA/1E                     │
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
| concept-map | 35 nodes, 52 relationships, 4 pathways | Structure + learning paths |
| flashcards | 5 cards (2E/2M/1H) | Memorization + critical flags |
| practice-problems | 5 problems (1W/2S/1C/1D) | Procedural fluency + debugging |
| quiz | 5 questions (2MC/2SA/1E) | Assessment + diagnostic feedback |
