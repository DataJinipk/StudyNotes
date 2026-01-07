# Practice Problems: Deep Learning

**Source:** notes/deep-learning/deep-learning-study-notes.md
**Concept Map:** notes/deep-learning/concept-maps/deep-learning-concept-map.md
**Flashcards:** notes/deep-learning/flashcards/deep-learning-flashcards.md
**Date Generated:** 2026-01-07
**Total Problems:** 5
**Distribution:** 1 Warm-Up | 2 Skill-Builder | 1 Challenge | 1 Debug/Fix

---

## Problem Distribution

| Type | Count | Purpose | Time Estimate |
|------|-------|---------|---------------|
| Warm-Up | 1 | Activate prior knowledge; build confidence | 10-15 min |
| Skill-Builder | 2 | Develop core procedural fluency | 20-30 min each |
| Challenge | 1 | Extend to complex scenarios | 40-50 min |
| Debug/Fix | 1 | Identify and correct common errors | 20-25 min |

---

## Problem 1 | Warm-Up
**Concept:** Optimizer Selection and Learning Rate
**Difficulty:** ⭐☆☆☆☆
**Estimated Time:** 15 minutes
**Prerequisites:** Basic calculus, gradient descent intuition

### Problem Statement

You're starting a new image classification project and need to configure the optimizer. Given the following training scenarios, select the appropriate optimizer and initial learning rate, then justify your choice:

**Scenario A:** Quick prototype to validate data pipeline; need results in 1 hour
**Scenario B:** Final training run for production model; have 1 week and want best accuracy
**Scenario C:** Fine-tuning a pre-trained BERT model on a small dataset (5K samples)

For each scenario, specify:
1. Optimizer choice (SGD, SGD+Momentum, Adam, AdamW)
2. Initial learning rate
3. Learning rate schedule (if any)
4. Brief justification (1-2 sentences)

### Hints

<details>
<summary>Hint 1 (General Principle)</summary>
Adam converges faster but SGD often achieves better final results with proper tuning.
</details>

<details>
<summary>Hint 2 (Fine-tuning)</summary>
Pre-trained models need lower learning rates to preserve learned features.
</details>

<details>
<summary>Hint 3 (Transformers)</summary>
AdamW is the standard for Transformer models due to proper weight decay handling.
</details>

### Solution

**Scenario A: Quick Prototype (1 hour)**
```python
optimizer = Adam(model.parameters(), lr=1e-3)
scheduler = None  # No time for fancy scheduling
```

| Choice | Value | Justification |
|--------|-------|---------------|
| Optimizer | **Adam** | Fast convergence, minimal tuning needed |
| Learning Rate | **1e-3** | Adam default; works for most architectures |
| Schedule | **None** | Prototyping; keep it simple |

**Reasoning:** Adam's adaptive learning rates per parameter enable quick convergence without extensive hyperparameter search. Default settings usually work.

---

**Scenario B: Final Production Training (1 week)**
```python
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
# Or: MultiStepLR at epochs [60, 120, 160] with gamma=0.1
```

| Choice | Value | Justification |
|--------|-------|---------------|
| Optimizer | **SGD + Momentum** | Often achieves best final accuracy |
| Learning Rate | **0.1** | Standard starting point for CNNs |
| Schedule | **Cosine Annealing** or **Step Decay** | Gradually refine to sharp minimum |

**Reasoning:** For best accuracy when time permits, SGD with momentum typically outperforms Adam. Higher initial LR (0.1) with decay allows exploration then fine-tuning.

---

**Scenario C: BERT Fine-tuning (5K samples)**
```python
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,
    num_training_steps=total_steps
)
```

| Choice | Value | Justification |
|--------|-------|---------------|
| Optimizer | **AdamW** | Standard for Transformers; proper weight decay |
| Learning Rate | **2e-5** | Low LR preserves pre-trained knowledge |
| Schedule | **Linear warmup + decay** | Stabilizes early training, then refines |

**Reasoning:** Pre-trained models need low learning rates to avoid destroying learned representations. AdamW handles weight decay correctly for Transformers. Warmup prevents early instability.

### Key Takeaways

1. **Prototyping:** Adam with defaults for speed
2. **Best accuracy:** SGD + momentum with scheduling
3. **Fine-tuning:** AdamW with low LR and warmup
4. **Rule of thumb:** LR is the most important hyperparameter—start with established defaults

---

## Problem 2 | Skill-Builder
**Concept:** ResNet Architecture Design
**Difficulty:** ⭐⭐⭐☆☆
**Estimated Time:** 25 minutes
**Prerequisites:** CNN basics, skip connections concept

### Problem Statement

Design a ResNet-style architecture for CIFAR-10 classification (32×32 images, 10 classes). Your network should have approximately 1 million parameters.

**Requirements:**
1. Draw the architecture showing residual blocks
2. Calculate the number of parameters for each stage
3. Explain where you need projection shortcuts vs. identity shortcuts
4. Specify normalization and activation placement

**Constraints:**
- Input: 32×32×3
- Output: 10 classes
- Target parameters: ~1M (±100K)
- Use residual blocks with 2 convolutions each

### Hints

<details>
<summary>Hint 1 (CIFAR Structure)</summary>
CIFAR-10 images are small (32×32), so use stride=1 in the first conv (unlike ImageNet ResNets that start with stride=2).
</details>

<details>
<summary>Hint 2 (Parameter Counting)</summary>
Conv params = kernel_h × kernel_w × in_channels × out_channels + out_channels (bias)
For 3×3 conv: 9 × C_in × C_out + C_out
</details>

<details>
<summary>Hint 3 (Projection)</summary>
When spatial dimensions or channel counts change between input and output of a block, you need a projection shortcut (1×1 conv).
</details>

### Solution

**Architecture Overview:**

```
Input: 32×32×3
    │
    ▼
┌─────────────────────────────────┐
│ Conv 3×3, 16 filters, stride=1  │  32×32×16
│ BatchNorm → ReLU                │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ Stage 1: 3 Residual Blocks      │  32×32×16
│ [16 → 16 channels]              │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ Stage 2: 3 Residual Blocks      │  16×16×32
│ [16 → 32 channels, first stride=2]│
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ Stage 3: 3 Residual Blocks      │  8×8×64
│ [32 → 64 channels, first stride=2]│
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ Global Average Pooling          │  1×1×64
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ Fully Connected: 64 → 10        │  10 classes
└─────────────────────────────────┘
```

**Residual Block Detail:**

```
Input x
    │
    ├──────────────────────────────────┐
    ▼                                  │
┌──────────────────┐                   │
│ Conv 3×3         │                   │ Shortcut
│ BatchNorm        │                   │ (identity or
│ ReLU             │                   │  1×1 projection)
└────────┬─────────┘                   │
         ▼                             │
┌──────────────────┐                   │
│ Conv 3×3         │                   │
│ BatchNorm        │                   │
└────────┬─────────┘                   │
         │                             │
         ▼                             │
       (+)  ◄──────────────────────────┘
         │
         ▼
       ReLU
         │
       Output
```

**Parameter Calculation:**

**Initial Conv:**
- 3×3 conv: 3×3×3×16 = 432
- BatchNorm: 16×2 = 32 (γ, β)
- **Total: 464**

**Stage 1 (3 blocks, 16→16):**
Each block:
- Conv1: 3×3×16×16 = 2,304
- BN1: 32
- Conv2: 3×3×16×16 = 2,304
- BN2: 32
- Block total: 4,672
- Stage 1 total: 4,672 × 3 = **14,016**

**Stage 2 (3 blocks, 16→32):**
First block (with projection):
- Conv1: 3×3×16×32 = 4,608
- BN1: 64
- Conv2: 3×3×32×32 = 9,216
- BN2: 64
- Projection: 1×1×16×32 = 512 + BN: 64
- First block: 14,528

Remaining 2 blocks (32→32):
- Conv1: 3×3×32×32 = 9,216
- BN1: 64
- Conv2: 3×3×32×32 = 9,216
- BN2: 64
- Block total: 18,560
- Remaining: 18,560 × 2 = 37,120

- Stage 2 total: 14,528 + 37,120 = **51,648**

**Stage 3 (3 blocks, 32→64):**
First block (with projection):
- Conv1: 3×3×32×64 = 18,432
- BN1: 128
- Conv2: 3×3×64×64 = 36,864
- BN2: 128
- Projection: 1×1×32×64 = 2,048 + BN: 128
- First block: 57,728

Remaining 2 blocks (64→64):
- Conv1: 3×3×64×64 = 36,864
- BN1: 128
- Conv2: 3×3×64×64 = 36,864
- BN2: 128
- Block total: 73,984
- Remaining: 73,984 × 2 = 147,968

- Stage 3 total: 57,728 + 147,968 = **205,696**

**Final FC:**
- 64 × 10 + 10 = **650**

**Total Parameters:**
```
Initial:   464
Stage 1:   14,016
Stage 2:   51,648
Stage 3:   205,696
FC:        650
─────────────────
Total:     272,474 ≈ 270K
```

**Scaling to ~1M parameters:**
To reach 1M, double initial channels: 16 → 32 → 64 → 128

| Stage | Channels | Blocks | Params |
|-------|----------|--------|--------|
| Initial | 32 | 1 | 896 |
| Stage 1 | 32 | 3 | 56,064 |
| Stage 2 | 64 | 3 | 206,592 |
| Stage 3 | 128 | 3 | 822,784 |
| FC | 128→10 | 1 | 1,290 |
| **Total** | | | **~1.09M** |

**Shortcut Types:**

| Transition | Dimensions | Shortcut Type |
|------------|------------|---------------|
| Stage 1 blocks | 32×32×32 → 32×32×32 | Identity (x) |
| Stage 1→2 first | 32×32×32 → 16×16×64 | Projection 1×1 conv, stride=2 |
| Stage 2 other | 16×16×64 → 16×16×64 | Identity (x) |
| Stage 2→3 first | 16×16×64 → 8×8×128 | Projection 1×1 conv, stride=2 |
| Stage 3 other | 8×8×128 → 8×8×128 | Identity (x) |

### Key Takeaways

1. **CIFAR-specific:** No initial pooling/striding (images already small)
2. **Projection shortcuts:** Only when dimensions change
3. **Channel progression:** Double channels when halving spatial resolution
4. **Parameter distribution:** Later stages dominate (more channels)

---

## Problem 3 | Skill-Builder
**Concept:** Diagnosing and Fixing Overfitting
**Difficulty:** ⭐⭐⭐☆☆
**Estimated Time:** 25 minutes
**Prerequisites:** Regularization techniques, training dynamics

### Problem Statement

You're training a CNN for medical image classification. Here are your training metrics:

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|------------|-----------|----------|---------|
| 1 | 2.15 | 32% | 2.10 | 34% |
| 5 | 0.85 | 71% | 0.92 | 68% |
| 10 | 0.31 | 89% | 0.78 | 73% |
| 20 | 0.08 | 97% | 1.24 | 71% |
| 30 | 0.02 | 99% | 1.89 | 68% |

**Current Configuration:**
```python
model = CNN(num_classes=5)  # ~10M parameters
optimizer = Adam(lr=1e-3)
criterion = CrossEntropyLoss()
batch_size = 32
epochs = 30
# No regularization applied
# Dataset: 3,000 training images, 500 validation
```

**Tasks:**
1. Diagnose the problem with specific evidence from the metrics
2. Propose 5 interventions ranked by expected impact
3. Implement the top 3 interventions in code
4. Predict the expected metric improvements

### Hints

<details>
<summary>Hint 1 (Diagnosis)</summary>
Compare training vs validation trends. When do they diverge?
</details>

<details>
<summary>Hint 2 (Root Cause)</summary>
Consider the ratio: 10M parameters vs 3,000 training samples.
</details>

<details>
<summary>Hint 3 (Medical Imaging)</summary>
Medical datasets are small. What techniques specifically help with limited data?
</details>

### Solution

**Step 1: Diagnosis**

**Problem: Severe Overfitting**

Evidence:
- Training loss: 2.15 → 0.02 (99% reduction)
- Validation loss: 2.10 → 0.78 (best) → 1.89 (final)
- Gap at epoch 30: Train 99% vs Val 68% (31% gap!)
- Val performance peaks at epoch 5-10, then degrades

**Root Causes:**
1. **Extreme overparameterization:** 10M params ÷ 3K samples = 3,333 params per sample
2. **No regularization:** Model memorizes training data
3. **Long training:** 30 epochs is excessive given overfitting starts at ~5

```
Metric Visualization:
Accuracy
100%├────────────────────●●●●●●── Train
    │                 ●●
 80%├───────────●●●●●
    │         ●    ╲
 70%├───────●────────●●●●────── Val (degrading)
    │      ●
 60%├────●
    │
    └──────────────────────────── Epoch
        5    10   15   20   25   30
```

**Step 2: Interventions Ranked by Impact**

| Rank | Intervention | Expected Impact | Difficulty |
|------|--------------|-----------------|------------|
| 1 | **Data Augmentation** | High | Low |
| 2 | **Transfer Learning** | High | Medium |
| 3 | **Early Stopping** | High | Low |
| 4 | **Dropout** | Medium-High | Low |
| 5 | **Weight Decay** | Medium | Low |
| 6 | **Reduce Model Size** | Medium | Medium |
| 7 | **Label Smoothing** | Low-Medium | Low |

**Step 3: Implementation**

**Intervention 1: Data Augmentation**
```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),  # Often valid for medical images
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2),  # Cutout-style augmentation
])

# For medical: consider domain-specific augmentations
# - Elastic deformation
# - Intensity variations
# - Mixup / CutMix
```

**Intervention 2: Transfer Learning + Fine-tuning**
```python
import torchvision.models as models

# Use pre-trained model
model = models.resnet18(pretrained=True)  # Much smaller than 10M, pre-trained

# Freeze early layers (optional, helps with very small datasets)
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 5)
)

# Unfreeze later layers for fine-tuning
for param in model.layer4.parameters():
    param.requires_grad = True

# Lower learning rate for pre-trained layers
optimizer = Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
])
```

**Intervention 3: Early Stopping + Regularization**
```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_model = None

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

# Training loop with early stopping
early_stopping = EarlyStopping(patience=5)
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

for epoch in range(100):  # High max, rely on early stopping
    train_loss = train_epoch(model, train_loader)
    val_loss = validate(model, val_loader)

    if early_stopping(val_loss, model):
        print(f"Early stopping at epoch {epoch}")
        model.load_state_dict(early_stopping.best_model)
        break
```

**Complete Training Configuration:**
```python
# Combined solution
model = models.resnet18(pretrained=True)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(512, 5)
)

# Optimizer with weight decay
optimizer = AdamW([
    {'params': model.layer3.parameters(), 'lr': 1e-5},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
], weight_decay=0.01)

# Label smoothing
criterion = CrossEntropyLoss(label_smoothing=0.1)

# Learning rate scheduling
scheduler = CosineAnnealingLR(optimizer, T_max=50)

# Early stopping
early_stopping = EarlyStopping(patience=7)
```

**Step 4: Expected Improvements**

| Metric | Before | After (Expected) | Reasoning |
|--------|--------|------------------|-----------|
| Best Val Acc | 73% | **82-88%** | Transfer learning + augmentation |
| Train-Val Gap | 31% | **5-10%** | Regularization suite |
| Best Epoch | 10 | **15-25** | Early stopping prevents degradation |
| Val Loss (final) | 1.89 | **0.4-0.6** | Better generalization |

### Key Takeaways

1. **Data augmentation is king** for small datasets—effectively multiplies data
2. **Transfer learning** provides strong priors, reducing needed data by 10-100×
3. **Early stopping** is free regularization—always use it
4. **Combine techniques:** No single fix solves severe overfitting

---

## Problem 4 | Challenge
**Concept:** End-to-End Training Pipeline Design
**Difficulty:** ⭐⭐⭐⭐☆
**Estimated Time:** 45 minutes
**Prerequisites:** All deep learning concepts

### Problem Statement

Design a complete training pipeline for a 100-layer image classification network from scratch. The network will be trained on a custom dataset of 100,000 images across 1,000 classes (similar to ImageNet scale).

**Requirements:**
1. Architecture design with justification
2. Complete training configuration
3. Memory and compute budget analysis
4. Training stability considerations
5. Monitoring and debugging strategy

**Constraints:**
- Hardware: Single NVIDIA A100 (40GB)
- Training budget: 48 hours maximum
- Target accuracy: >75% top-1

### Solution

**1. Architecture Design**

**Choice: ResNet-101 variant with modern improvements**

```
Architecture: ResNet-101-D
- "D" variant: Replace 7×7 stem with 3 stacked 3×3 convs
- Improves accuracy with similar params

┌─────────────────────────────────────────────────────────────┐
│ Stem (replaces 7×7 conv):                                   │
│   3×3 conv, 32, stride=2 → BN → ReLU                       │
│   3×3 conv, 32, stride=1 → BN → ReLU                       │
│   3×3 conv, 64, stride=1 → BN → ReLU                       │
│   MaxPool 3×3, stride=2                                     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: 3 × Bottleneck blocks  [64, 64, 256]  56×56×256   │
│ Stage 2: 4 × Bottleneck blocks  [128, 128, 512] 28×28×512  │
│ Stage 3: 23 × Bottleneck blocks [256, 256, 1024] 14×14×1024│
│ Stage 4: 3 × Bottleneck blocks  [512, 512, 2048] 7×7×2048  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Global Average Pool → FC 2048→1000 → Softmax               │
└─────────────────────────────────────────────────────────────┘

Total layers: 101 (counting conv layers)
Parameters: ~44.5M
```

**Bottleneck Block:**
```
Input (C channels)
    │
    ├─────────────────────────────────┐
    ▼                                 │
1×1 conv, C/4 → BN → ReLU            │
    ▼                                 │ Shortcut
3×3 conv, C/4 → BN → ReLU            │ (identity or
    ▼                                 │  1×1 projection)
1×1 conv, C → BN                     │
    ▼                                 │
   (+) ◄──────────────────────────────┘
    ▼
   ReLU
```

**2. Complete Training Configuration**

```python
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast

# ═══════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════
model = ResNet101D(num_classes=1000)

# Initialize weights (important for deep networks!)
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)

model.apply(init_weights)

# Zero-initialize last BN in each residual block
# (makes initial residual = 0, easing optimization)
for m in model.modules():
    if isinstance(m, Bottleneck):
        nn.init.constant_(m.bn3.weight, 0)

# ═══════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.TrivialAugmentWide(),  # Modern auto-augment
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

train_loader = DataLoader(
    train_dataset,
    batch_size=256,  # Determined by memory analysis
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    drop_last=True,
)

# ═══════════════════════════════════════════════════════════════
# OPTIMIZER & SCHEDULER
# ═══════════════════════════════════════════════════════════════
base_lr = 0.1
epochs = 90

optimizer = SGD(
    model.parameters(),
    lr=base_lr,
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True
)

# Warmup + Cosine decay
warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=5)
main_scheduler = CosineAnnealingLR(optimizer, T_max=epochs-5)
scheduler = SequentialLR(optimizer, [warmup_scheduler, main_scheduler], milestones=[5])

# ═══════════════════════════════════════════════════════════════
# LOSS & REGULARIZATION
# ═══════════════════════════════════════════════════════════════
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# ═══════════════════════════════════════════════════════════════
# MIXED PRECISION
# ═══════════════════════════════════════════════════════════════
scaler = GradScaler()

# ═══════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════
for epoch in range(epochs):
    model.train()

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()

        # Gradient clipping (optional, for stability)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

    scheduler.step()

    # Validation
    if epoch % 5 == 0:
        val_acc = validate(model, val_loader)
        print(f"Epoch {epoch}: Val Acc = {val_acc:.2%}")
```

**3. Memory Budget Analysis**

```
Model Parameters: 44.5M × 4 bytes = 178 MB (FP32)
                  44.5M × 2 bytes = 89 MB (FP16)

Optimizer State (SGD + momentum):
    Momentum buffer: 44.5M × 4 = 178 MB
    Total optimizer: ~180 MB

Activations (batch_size=256, largest layer):
    Stem output: 256 × 64 × 56 × 56 × 4 = 205 MB
    Stage 3 (largest): 256 × 1024 × 14 × 14 × 4 = 205 MB
    Total activations: ~2-3 GB (varies by layer)

Gradients: ~180 MB (FP32)

Mixed Precision Savings:
    Activations in FP16: ~1-1.5 GB (50% reduction)
    Master weights in FP32: 178 MB (kept for accuracy)

Total Memory Estimate:
    Model (FP16):      89 MB
    Master weights:    178 MB
    Optimizer:         180 MB
    Activations:       1.5 GB (with AMP)
    Gradients:         90 MB (FP16)
    Workspace:         1 GB (cuDNN, etc.)
    ─────────────────────────────
    Total:             ~3.5 GB

Available (A100 40GB): 40 GB
Headroom: 36.5 GB → Can increase batch size to 512-1024!
```

**Revised batch size:** 512 (better GPU utilization)

**4. Training Stability Considerations**

| Issue | Solution | Implementation |
|-------|----------|----------------|
| Vanishing gradients | Skip connections | ResNet architecture |
| Exploding gradients | Gradient clipping | `clip_grad_norm_(1.0)` |
| Loss spikes | Learning rate warmup | 5-epoch linear warmup |
| NaN loss | Loss scaling | GradScaler for mixed precision |
| Poor initialization | Kaiming init + zero-init last BN | Custom init function |
| Batch stat instability | Larger batch size | 512 with memory headroom |

**5. Monitoring & Debugging**

```python
# Logging with wandb or tensorboard
import wandb

wandb.init(project="resnet101-training")

def log_metrics(epoch, train_loss, val_loss, val_acc, lr):
    wandb.log({
        "epoch": epoch,
        "train/loss": train_loss,
        "val/loss": val_loss,
        "val/accuracy": val_acc,
        "lr": lr,
    })

def log_gradients(model):
    """Monitor gradient health"""
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    wandb.log({"gradients/total_norm": total_norm})

def log_weight_stats(model):
    """Monitor weight distributions"""
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            wandb.log({
                f"weights/{name}_mean": param.data.mean(),
                f"weights/{name}_std": param.data.std(),
            })
```

**Key Metrics to Monitor:**

| Metric | Healthy Range | Warning Sign |
|--------|---------------|--------------|
| Gradient norm | 0.1 - 10 | >100 or ~0 |
| Loss | Decreasing | Spikes, NaN |
| LR | Following schedule | Stuck |
| GPU memory | <90% | OOM errors |
| Throughput | >1000 img/s | <500 img/s |

**Training Timeline (48 hours):**
```
Epochs: 90
Batch size: 512
Steps per epoch: 100,000 / 512 ≈ 195
Total steps: 17,550
Time per step: ~0.5s (with A100)
Total time: ~2.5 hours per epoch × 90 = ~225 hours ❌

Adjustment: Reduce to 30 epochs with stronger augmentation
Time: 30 × 2.5h = 75 hours ❌

Final adjustment: Batch 1024, 50 epochs
Steps per epoch: ~98
Time per step: ~0.4s (better GPU utilization)
Total: 50 × 98 × 0.4s ≈ 32 hours ✓
```

### Key Takeaways

1. **Memory analysis first:** Determine max batch size before other decisions
2. **Mixed precision is essential:** 2× memory savings, 1.5-2× speedup
3. **Initialization matters:** Zero-init last BN stabilizes deep networks
4. **Monitor everything:** Gradient norms catch problems early
5. **Time budget:** Always calculate total training time upfront

---

## Problem 5 | Debug/Fix
**Concept:** Training Failures and Solutions
**Difficulty:** ⭐⭐⭐☆☆
**Estimated Time:** 25 minutes
**Prerequisites:** Optimization, regularization, architecture

### Problem Statement

Review the following training code and logs. Identify all issues and provide fixes.

**Code:**
```python
import torch
import torch.nn as nn
from torch.optim import Adam

class DeepNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 1024),
            nn.Sigmoid(),        # Issue 1
            nn.Linear(1024, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, 512),
            nn.Sigmoid(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.layers(x.view(-1, 784))

model = DeepNetwork()
optimizer = Adam(model.parameters(), lr=0.1)  # Issue 2
criterion = nn.CrossEntropyLoss()

# No learning rate scheduler  # Issue 3

for epoch in range(100):
    for batch_idx, (data, target) in enumerate(train_loader):
        output = model(data)
        loss = criterion(output, target)

        loss.backward()        # Issue 4
        optimizer.step()

    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```

**Training Log:**
```
Epoch 0: Loss = 2.3024
Epoch 1: Loss = 2.3019
Epoch 2: Loss = 2.3021
Epoch 3: Loss = 2.3018
...
Epoch 50: Loss = 2.3016
Epoch 51: Loss = 2.3022
...
Epoch 99: Loss = 2.3015
```

**Tasks:**
1. Identify all 6+ issues in the code
2. Explain why each causes problems
3. Provide the corrected code
4. Explain why the loss is stuck at ~2.30 (10-class random guessing)

### Solution

**Issue Analysis:**

**Issue 1: Sigmoid Activations**
```python
# PROBLEM
nn.Sigmoid()  # Used 5 times in deep network

# WHY IT'S BAD
# - Output range [0,1] with derivative max 0.25
# - Through 5 layers: gradient multiplies by ≤0.25^5 = 0.001
# - Vanishing gradients → early layers don't learn

# FIX
nn.ReLU()  # Gradient is 1 for positive values
# Or: nn.LeakyReLU(0.1), nn.GELU()
```

**Issue 2: Learning Rate Too High**
```python
# PROBLEM
optimizer = Adam(model.parameters(), lr=0.1)

# WHY IT'S BAD
# - Adam default is 0.001; 0.1 is 100× higher
# - Updates are too large → overshoots minima
# - Can cause divergence or oscillation

# FIX
optimizer = Adam(model.parameters(), lr=0.001)  # Or 1e-4 to 3e-4
```

**Issue 3: No Learning Rate Scheduler**
```python
# PROBLEM
# No scheduler defined

# WHY IT'S BAD
# - Fixed LR can't adapt to training stages
# - Early: need higher LR for exploration
# - Late: need lower LR for fine-tuning

# FIX
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
# Or: CosineAnnealingLR, StepLR
```

**Issue 4: Missing optimizer.zero_grad()**
```python
# PROBLEM
loss.backward()
optimizer.step()
# Gradients accumulate across batches!

# WHY IT'S BAD
# - Gradients from previous batches add up
# - Effective batch size grows unboundedly
# - Eventually causes gradient explosion or NaN

# FIX
optimizer.zero_grad()  # ADD THIS BEFORE backward()
loss.backward()
optimizer.step()
```

**Issue 5: No Batch Normalization**
```python
# PROBLEM
# Deep network without normalization

# WHY IT'S BAD
# - Activations can shift/scale dramatically between layers
# - Training becomes unstable
# - Requires very careful initialization

# FIX: Add BatchNorm after each linear layer (before activation)
nn.Linear(784, 1024),
nn.BatchNorm1d(1024),  # ADD
nn.ReLU(),
```

**Issue 6: No Weight Initialization**
```python
# PROBLEM
# Default PyTorch initialization may not be optimal

# FIX
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)

model.apply(init_weights)
```

**Issue 7: Loss printed is only last batch**
```python
# PROBLEM
print(f"Epoch {epoch}: Loss = {loss.item():.4f}")  # Only last batch!

# FIX: Track average loss
total_loss = 0
for batch_idx, (data, target) in enumerate(train_loader):
    ...
    total_loss += loss.item()
avg_loss = total_loss / len(train_loader)
print(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}")
```

**Why Loss is Stuck at 2.30:**

```python
# Random guessing on 10 classes:
# CrossEntropy = -log(1/10) = log(10) ≈ 2.303

# The network is NOT LEARNING because:
# 1. Vanishing gradients (sigmoid) → early layers frozen
# 2. Missing zero_grad → gradient accumulation corrupts updates
# 3. High LR with Adam → unstable updates
# Combined effect: network outputs uniform probabilities
```

**Corrected Code:**

```python
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

class DeepNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),              # FIX 1: ReLU instead of Sigmoid

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),   # FIX 5: Add BatchNorm
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Dropout(0.5),        # Added regularization

            nn.Linear(512, 10)
        )

        # FIX 6: Proper initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.layers(x.view(-1, 784))

model = DeepNetwork()
optimizer = Adam(model.parameters(), lr=0.001)  # FIX 2: Lower LR
scheduler = CosineAnnealingLR(optimizer, T_max=100)  # FIX 3: Add scheduler
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    model.train()
    total_loss = 0  # FIX 7: Track average

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # FIX 4: Zero gradients!

        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()  # Update learning rate

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.6f}")
```

**Expected Results After Fixes:**

| Epoch | Before (Broken) | After (Fixed) |
|-------|-----------------|---------------|
| 0 | 2.302 | 2.15 |
| 10 | 2.301 | 0.45 |
| 50 | 2.302 | 0.08 |
| 99 | 2.301 | 0.03 |

### Key Takeaways

1. **zero_grad() is mandatory:** Forgetting it is a common silent bug
2. **Sigmoid kills deep networks:** Use ReLU variants
3. **Adam LR ≠ SGD LR:** Adam typically uses 10-100× lower LR
4. **Loss = log(num_classes)** means random guessing—network isn't learning
5. **BatchNorm stabilizes training:** Essential for deep networks without skip connections

---

## Problem Summary

| Problem | Type | Concepts | Difficulty | Key Learning |
|---------|------|----------|------------|--------------|
| P1 | Warm-Up | Optimizer Selection | ⭐ | Context-appropriate choices |
| P2 | Skill-Builder | ResNet Architecture | ⭐⭐⭐ | Parameter counting, shortcuts |
| P3 | Skill-Builder | Overfitting Diagnosis | ⭐⭐⭐ | Regularization strategies |
| P4 | Challenge | Training Pipeline | ⭐⭐⭐⭐ | End-to-end design |
| P5 | Debug/Fix | Training Failures | ⭐⭐⭐ | Common bugs and fixes |

---

## Cross-References

| Problem | Study Notes Section | Concept Map Node | Flashcard |
|---------|---------------------|------------------|-----------|
| P1 | Concept 4-5: Optimizers, LR Scheduling | Adam (11) | Card 1 |
| P2 | Concept 8: Skip Connections | Skip Connections (9) | Card 2 |
| P3 | Concept 6-7: Regularization, Normalization | BatchNorm (7), Dropout (6) | Card 3 |
| P4 | All Concepts | Full pipeline | Card 5 |
| P5 | Concepts 2, 4, 6, 7 | Activation, Optimizer, Regularization | Cards 1-3 |
