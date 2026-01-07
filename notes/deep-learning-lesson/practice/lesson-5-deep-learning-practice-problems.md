# Practice Problems: Lesson 5 - Deep Learning

**Source:** Lessons/Lesson_5.md
**Subject Area:** AI Learning - Deep Learning: Neural Networks, Optimization, and Training Techniques
**Date Generated:** 2026-01-08
**Total Problems:** 5
**Distribution:** 1 Warm-Up | 2 Skill-Builder | 1 Challenge | 1 Debug/Fix

---

## Problem Overview

| # | Type | Concept Focus | Difficulty | Estimated Time |
|---|------|---------------|------------|----------------|
| 1 | Warm-Up | Gradient Flow Computation | Low | 15-20 min |
| 2 | Skill-Builder | Optimizer Selection | Medium | 20-25 min |
| 3 | Skill-Builder | Regularization Design | Medium | 25-30 min |
| 4 | Challenge | Complete Training Pipeline | High | 45-60 min |
| 5 | Debug/Fix | Training Failure Diagnosis | Medium | 25-30 min |

---

## Problem 1: Warm-Up
### Gradient Flow Computation

**Concept:** Backpropagation and Gradient Flow (Core Concept 4)
**Cognitive Level:** Apply
**Prerequisites:** Chain rule, basic calculus

---

**Problem Statement:**

Consider a simple 3-layer network (no bias terms for simplicity):
```
Input: x
Layer 1: h1 = ReLU(W1 × x)
Layer 2: h2 = ReLU(W2 × h1)
Output: y = W3 × h2
Loss: L = 0.5 × (y - target)²
```

Given:
- x = 2
- W1 = 0.5, W2 = -0.3, W3 = 0.8
- target = 1

**Tasks:**

1. Compute the forward pass (h1, h2, y, L)
2. Compute ∂L/∂W3 using the chain rule
3. Compute ∂L/∂W2 (be careful with ReLU gradient)
4. Compute ∂L/∂W1
5. Explain what happens to ∂L/∂W1 if W2 were positive instead of negative

---

**Progressive Hints:**

<details>
<summary>Hint 1 (Conceptual)</summary>

Forward pass:
- h1 = ReLU(W1 × x) = ReLU(0.5 × 2) = ReLU(1) = 1
- h2 = ReLU(W2 × h1) = ReLU(-0.3 × 1) = ReLU(-0.3) = 0 ← ReLU kills negative!
</details>

<details>
<summary>Hint 2 (Procedural)</summary>

ReLU gradient:
- d(ReLU(z))/dz = 1 if z > 0, else 0

Since h2 = 0 (from ReLU(-0.3)), the gradient through W2 is 0!
This is the "dying ReLU" phenomenon in action.
</details>

<details>
<summary>Hint 3 (Solution Direction)</summary>

Chain rule for ∂L/∂W2:
∂L/∂W2 = ∂L/∂y × ∂y/∂h2 × ∂h2/∂(W2×h1) × ∂(W2×h1)/∂W2

But ∂h2/∂(W2×h1) = 0 because ReLU input was negative!
</details>

---

**Solution:**

**1. Forward Pass:**
```
h1 = ReLU(W1 × x) = ReLU(0.5 × 2) = ReLU(1.0) = 1.0
h2 = ReLU(W2 × h1) = ReLU(-0.3 × 1.0) = ReLU(-0.3) = 0.0  ← Killed by ReLU!
y = W3 × h2 = 0.8 × 0.0 = 0.0
L = 0.5 × (y - target)² = 0.5 × (0 - 1)² = 0.5
```

**2. Compute ∂L/∂W3:**
```
∂L/∂y = y - target = 0 - 1 = -1
∂y/∂W3 = h2 = 0

∂L/∂W3 = ∂L/∂y × ∂y/∂W3 = (-1) × 0 = 0
```

**3. Compute ∂L/∂W2:**
```
Chain rule:
∂L/∂W2 = ∂L/∂y × ∂y/∂h2 × ∂h2/∂(pre_relu2) × ∂(pre_relu2)/∂W2

Where pre_relu2 = W2 × h1 = -0.3

∂L/∂y = -1
∂y/∂h2 = W3 = 0.8
∂h2/∂(pre_relu2) = 0  ← Because pre_relu2 = -0.3 < 0, ReLU gradient is 0!
∂(pre_relu2)/∂W2 = h1 = 1

∂L/∂W2 = (-1) × 0.8 × 0 × 1 = 0
```

**4. Compute ∂L/∂W1:**
```
Chain rule continues:
∂L/∂W1 = ∂L/∂y × ∂y/∂h2 × ∂h2/∂h1 × ∂h1/∂(pre_relu1) × ∂(pre_relu1)/∂W1

∂h2/∂h1 = ∂h2/∂(pre_relu2) × ∂(pre_relu2)/∂h1 = 0 × W2 = 0

∂L/∂W1 = ... × 0 × ... = 0
```

**All gradients are 0! The network cannot learn.**

**5. If W2 were positive (e.g., W2 = 0.3):**
```
Forward:
h1 = ReLU(0.5 × 2) = 1.0
h2 = ReLU(0.3 × 1.0) = 0.3  ← Now positive, ReLU passes it!
y = 0.8 × 0.3 = 0.24
L = 0.5 × (0.24 - 1)² = 0.29

Backward:
∂L/∂y = -0.76
∂y/∂h2 = 0.8
∂h2/∂(pre_relu2) = 1  ← Now gradient flows!
∂(pre_relu2)/∂W2 = h1 = 1

∂L/∂W2 = (-0.76) × 0.8 × 1 × 1 = -0.608 ≠ 0

Gradients would flow, and learning could occur!
```

**Key Insight:** This demonstrates the "dying ReLU" problem. When ReLU receives consistently negative inputs, gradients become 0, and those weights never update. Solutions include:
- LeakyReLU: Small gradient for negative inputs
- Careful initialization: Keep activations in positive regime initially
- Residual connections: Provide alternative gradient paths

---

**Common Mistakes:**
| Mistake | Why It's Wrong | Prevention |
|---------|---------------|------------|
| Forgetting ReLU clips gradient | ReLU(x) has gradient 0 for x < 0 | Always check sign of pre-ReLU value |
| Ignoring chain rule order | Must multiply all terms in chain | Write out full chain before computing |
| Computing gradient of ReLU output | Gradient is w.r.t. input to ReLU | ∂ReLU(z)/∂z, not ∂ReLU(z)/∂ReLU(z) |

---

## Problem 2: Skill-Builder
### Optimizer Selection and Configuration

**Concept:** Optimization Algorithms (Core Concept 5)
**Cognitive Level:** Apply/Analyze
**Prerequisites:** Understanding of SGD, Adam, AdamW

---

**Problem Statement:**

You're consulting for three teams, each with different training scenarios. Recommend the optimal optimizer configuration for each.

**Scenario A: Fine-tuning BERT for sentiment analysis**
- Pre-trained BERT-base (110M parameters)
- Training data: 50,000 labeled reviews
- Hardware: Single V100 GPU
- Goal: Quick convergence, good accuracy

**Scenario B: Training ResNet-50 from scratch on ImageNet**
- Random initialization
- 1.2M training images
- Hardware: 8× A100 GPUs
- Goal: Achieve state-of-the-art accuracy

**Scenario C: Training a small MLP for tabular data**
- 3-layer MLP (10K parameters)
- Training data: 5,000 samples
- Hardware: CPU
- Goal: Avoid overfitting, good generalization

**For each scenario, specify:**
1. Optimizer choice (SGD/Adam/AdamW) with justification
2. Learning rate and schedule
3. Weight decay setting
4. Any additional configuration (momentum, betas, etc.)

---

**Progressive Hints:**

<details>
<summary>Hint 1 (Conceptual)</summary>

General guidelines:
- Fine-tuning pre-trained models: AdamW with low LR
- Training CNNs from scratch: SGD + momentum for best final accuracy
- Small models with limited data: Careful regularization matters most
</details>

<details>
<summary>Hint 2 (Procedural)</summary>

For BERT fine-tuning:
- AdamW is standard (matches pre-training)
- LR: 2e-5 to 5e-5 (much lower than pre-training)
- Warmup: ~10% of training steps
</details>

<details>
<summary>Hint 3 (Solution Direction)</summary>

For ImageNet ResNet:
- SGD + momentum (0.9) often beats Adam for final accuracy
- LR: 0.1 with step decay at epochs 30, 60, 90
- Weight decay: 1e-4
- Large batch: Scale LR linearly
</details>

---

**Solution:**

**Scenario A: Fine-tuning BERT**

| Setting | Value | Justification |
|---------|-------|---------------|
| **Optimizer** | AdamW | Standard for Transformers; decoupled weight decay |
| **Learning Rate** | 2e-5 | Low LR for fine-tuning; preserves pre-trained features |
| **Schedule** | Linear warmup (10%) + linear decay | Warmup stabilizes Adam; decay prevents overfitting |
| **Weight Decay** | 0.01 | Standard for BERT fine-tuning |
| **Betas** | (0.9, 0.999) | Default Adam values work well |
| **Epochs** | 3-4 | BERT fine-tuning typically converges quickly |

**Configuration:**
```python
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)
```

**Why not SGD:** Adam's adaptive LR helps different layers (pre-trained backbone vs. new head) learn at appropriate rates.

---

**Scenario B: Training ResNet-50 on ImageNet**

| Setting | Value | Justification |
|---------|-------|---------------|
| **Optimizer** | SGD + Momentum | Best final accuracy for CNNs; well-studied |
| **Learning Rate** | 0.1 × (batch_size / 256) | Linear scaling rule for large batch |
| **Schedule** | Step decay: ×0.1 at epochs 30, 60, 90 | Classic ImageNet schedule |
| **Weight Decay** | 1e-4 | Standard regularization for ImageNet |
| **Momentum** | 0.9 | Accelerates convergence |
| **Epochs** | 90-100 | Standard ImageNet training |

**Configuration:**
```python
# With 8 GPUs, batch_size = 256 per GPU = 2048 total
effective_lr = 0.1 * (2048 / 256)  # = 0.8

optimizer = SGD(
    model.parameters(),
    lr=effective_lr,
    momentum=0.9,
    weight_decay=1e-4
)
scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
```

**Why not Adam:** SGD + momentum consistently achieves 0.5-1% better top-1 accuracy on ImageNet with proper tuning.

---

**Scenario C: Small MLP on Tabular Data**

| Setting | Value | Justification |
|---------|-------|---------------|
| **Optimizer** | Adam | Fast convergence; less tuning needed |
| **Learning Rate** | 1e-3 | Standard Adam default |
| **Schedule** | ReduceLROnPlateau | Adapt based on validation loss |
| **Weight Decay** | 1e-2 to 1e-1 | Higher WD for small data regime |
| **Dropout** | 0.3-0.5 | Additional regularization critical |
| **Early Stopping** | patience=10 | Stop when validation degrades |

**Configuration:**
```python
optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# During training:
if val_loss < best_val_loss:
    best_val_loss = val_loss
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= 10:
        break  # Early stopping
```

**Key insight:** With only 5,000 samples and 10K parameters, overfitting is the main concern. Use multiple regularization techniques (weight decay, dropout, early stopping) rather than focusing on optimizer choice.

---

**Summary Table:**

| Scenario | Optimizer | LR | Weight Decay | Key Factor |
|----------|-----------|-----|--------------|------------|
| A (BERT fine-tune) | AdamW | 2e-5 | 0.01 | Preserve pre-trained features |
| B (ResNet scratch) | SGD+momentum | 0.1-0.8 | 1e-4 | Best final accuracy |
| C (Small MLP) | Adam | 1e-3 | 0.01-0.1 | Prevent overfitting |

---

**Common Mistakes:**
| Mistake | Why It's Wrong | Prevention |
|---------|---------------|------------|
| Same LR for fine-tuning and scratch | Fine-tuning needs much lower LR | Reduce LR 10-100× for fine-tuning |
| Adam for final ImageNet accuracy | SGD typically achieves better final results | Use SGD+momentum for production CNN training |
| Ignoring regularization for small data | Small data = high overfitting risk | Prioritize regularization over optimizer |

---

## Problem 3: Skill-Builder
### Regularization Strategy Design

**Concept:** Regularization Techniques (Core Concept 7)
**Cognitive Level:** Apply/Analyze
**Prerequisites:** Understanding of dropout, weight decay, data augmentation

---

**Problem Statement:**

You're training a vision model for medical image classification (chest X-rays). The setup:
- Model: ResNet-18 (11M parameters)
- Training data: 10,000 labeled X-rays
- Classes: 5 disease categories + normal
- Challenge: Limited data, must generalize to new hospitals

You observe the following during initial training:
```
Epoch 10: Train Loss = 0.12, Val Loss = 0.45, Train Acc = 96%, Val Acc = 78%
Epoch 20: Train Loss = 0.03, Val Loss = 0.62, Train Acc = 99%, Val Acc = 75%
Epoch 30: Train Loss = 0.01, Val Loss = 0.81, Train Acc = 100%, Val Acc = 72%
```

**Tasks:**

1. Diagnose the problem based on the training curves
2. Design a comprehensive regularization strategy (at least 4 techniques)
3. For each technique, specify configuration and expected effect
4. Predict new training curves after applying your strategy
5. Discuss how you'd validate that the model generalizes to new hospitals

---

**Progressive Hints:**

<details>
<summary>Hint 1 (Conceptual)</summary>

The curves show classic overfitting:
- Train loss → 0, train acc → 100% (memorization)
- Val loss increasing, val acc decreasing (worse generalization)

Gap between train and val performance indicates high variance.
</details>

<details>
<summary>Hint 2 (Procedural)</summary>

Regularization toolkit for medical imaging:
- Data augmentation (rotation, flip, scale, intensity)
- Dropout (0.3-0.5 in classifier head)
- Weight decay (1e-4 to 1e-3)
- Early stopping (stop at epoch ~10 based on val loss)
- Label smoothing (0.1)
- Transfer learning from ImageNet
</details>

<details>
<summary>Hint 3 (Solution Direction)</summary>

For hospital generalization:
- Heavy augmentation simulates variation across hospitals
- Mixup/CutMix creates virtual training examples
- Consider test-time augmentation at inference
- Validate on held-out hospital (not just held-out images)
</details>

---

**Solution:**

**1. Diagnosis:**

| Symptom | Interpretation |
|---------|----------------|
| Train loss → 0.01 | Model memorizes training data |
| Val loss increasing | Model gets worse on unseen data |
| Train acc 100%, Val acc 72% | 28% gap indicates severe overfitting |
| Val acc decreasing over time | Training past optimal point |

**Diagnosis:** Severe overfitting due to limited data (10K images) relative to model capacity (11M parameters). The model is memorizing X-ray specific artifacts rather than learning generalizable disease features.

---

**2. Comprehensive Regularization Strategy:**

| Technique | Configuration | Expected Effect |
|-----------|---------------|-----------------|
| **Transfer Learning** | Pre-train on ImageNet, fine-tune | Provides general visual features; reduces parameters to learn |
| **Data Augmentation** | See below | Expands effective training set; improves invariance |
| **Dropout** | 0.5 in classifier head | Forces redundant representations |
| **Weight Decay** | 1e-3 (higher than default) | Penalizes large weights; smoother decision boundary |
| **Early Stopping** | patience=5 on val loss | Stops before severe overfitting |
| **Label Smoothing** | ε=0.1 | Prevents overconfident predictions |

**Data Augmentation Pipeline (Medical-specific):**
```python
train_transforms = A.Compose([
    # Geometric (common across hospitals)
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),

    # Intensity (accounts for different X-ray machines)
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    A.GaussNoise(var_limit=(10, 50), p=0.3),

    # Advanced
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),  # Simulates artifacts
    A.Normalize(mean=0.5, std=0.25),
])
```

---

**3. Configuration Details:**

**Transfer Learning Setup:**
```python
# Load ImageNet pre-trained ResNet-18
model = torchvision.models.resnet18(pretrained=True)

# Replace classifier for 6 classes
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(512, 6)
)

# Different LR for backbone vs head
optimizer = AdamW([
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
], weight_decay=1e-3)
```

**Label Smoothing:**
```python
# Instead of hard targets [0, 0, 1, 0, 0, 0]
# Use soft targets [0.02, 0.02, 0.9, 0.02, 0.02, 0.02]
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

---

**4. Predicted Training Curves After Regularization:**

```
Before (overfitting):
Epoch 10: Train Loss = 0.12, Val Loss = 0.45, Train Acc = 96%, Val Acc = 78%
Epoch 20: Train Loss = 0.03, Val Loss = 0.62, Train Acc = 99%, Val Acc = 75%
Epoch 30: Train Loss = 0.01, Val Loss = 0.81, Train Acc = 100%, Val Acc = 72%

After (regularized):
Epoch 10: Train Loss = 0.35, Val Loss = 0.40, Train Acc = 88%, Val Acc = 84%
Epoch 20: Train Loss = 0.28, Val Loss = 0.35, Train Acc = 90%, Val Acc = 87%
Epoch 30: Train Loss = 0.25, Val Loss = 0.33, Train Acc = 91%, Val Acc = 88%
Epoch 40: Train Loss = 0.24, Val Loss = 0.32, Train Acc = 91%, Val Acc = 88%
(Converges, no more improvement)
```

**Key changes:**
- Train loss higher (regularization prevents memorization)
- Val loss lower (better generalization)
- Gap between train/val much smaller
- Val accuracy improves with training (no degradation)

---

**5. Validating Hospital Generalization:**

| Validation Strategy | Implementation |
|---------------------|----------------|
| **Held-out hospital** | Reserve 1-2 hospitals entirely for testing (not just images) |
| **Cross-hospital CV** | Leave-one-hospital-out cross-validation |
| **Distribution shift metrics** | Compare activation distributions across hospitals |
| **Subgroup analysis** | Report accuracy per hospital, per X-ray machine |

**Testing Protocol:**
```
Training: Hospitals A, B, C (8,000 images)
Validation: Hospital D (1,000 images) - tune hyperparameters
Test: Hospital E (1,000 images) - final evaluation (never seen during development)

Report:
- Overall accuracy on Hospital E
- Per-class accuracy on Hospital E
- Calibration metrics (ECE) on Hospital E
- Comparison of performance across hospitals
```

---

**Common Mistakes:**
| Mistake | Why It's Wrong | Prevention |
|---------|---------------|------------|
| Augmentation too weak | Medical images vary significantly across machines | Use strong, diverse augmentation |
| Validating on same hospital | Doesn't test generalization | Hold out entire hospitals |
| No transfer learning | 10K images insufficient for ResNet from scratch | Always start from pre-trained |

---

## Problem 4: Challenge
### Complete Training Pipeline Design

**Concept:** All Core Concepts Integration
**Cognitive Level:** Synthesize
**Prerequisites:** Full understanding of optimization, regularization, normalization, architecture

---

**Problem Statement:**

Design a complete training pipeline for a vision-language model that generates image captions.

**Model Architecture:**
- Vision encoder: ViT-B/16 (86M parameters)
- Language decoder: GPT-2 Small (117M parameters)
- Cross-attention layers connecting them
- Total: ~250M parameters

**Training Setup:**
- Dataset: 3M image-caption pairs
- Hardware: 4× A100 80GB GPUs
- Training budget: 1 week
- Goal: Generate accurate, fluent captions

**Your pipeline must address:**

1. **Initialization Strategy:** How to initialize each component
2. **Optimizer Configuration:** Optimizer choice, LR for different components
3. **Learning Rate Schedule:** Warmup, decay, total steps
4. **Normalization:** Where and what type
5. **Regularization:** Prevent overfitting strategies
6. **Distributed Training:** How to use 4 GPUs efficiently
7. **Mixed Precision:** Configuration for memory and speed
8. **Monitoring:** What metrics to track and warning signs
9. **Checkpointing:** When and how to save

---

**Progressive Hints:**

<details>
<summary>Hint 1 (Conceptual)</summary>

Key insight: Different components need different treatment:
- ViT encoder: Pre-trained, needs low LR
- GPT-2 decoder: Pre-trained, needs low LR
- Cross-attention: Random init, needs higher LR
</details>

<details>
<summary>Hint 2 (Procedural)</summary>

Common approach:
1. Initialize from pre-trained ViT and GPT-2
2. Initialize cross-attention randomly (Xavier/He)
3. Use layer-wise LR decay (lower LR for earlier layers)
4. Train ~100K steps with batch size 256
</details>

<details>
<summary>Hint 3 (Solution Direction)</summary>

Distributed training:
- Data parallelism: Replicate model on 4 GPUs
- Effective batch: 64 per GPU × 4 = 256
- Gradient accumulation if needed for larger effective batch
- Sync gradients with DistributedDataParallel
</details>

---

**Solution:**

**1. Initialization Strategy:**

| Component | Initialization | Rationale |
|-----------|----------------|-----------|
| **ViT Encoder** | Pre-trained on ImageNet-21K | Visual features already learned; 86M params too many to train from scratch |
| **GPT-2 Decoder** | Pre-trained on WebText | Language generation already learned |
| **Cross-Attention** | Xavier uniform | Random; will be trained from scratch |
| **Projection Layers** | Xavier uniform, small std | Connect encoder output dim to decoder input dim |

```python
# Load pre-trained components
vit_encoder = ViT.from_pretrained('google/vit-base-patch16-224')
gpt2_decoder = GPT2LMHeadModel.from_pretrained('gpt2')

# Initialize cross-attention (new layers)
for module in cross_attention_layers:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)
```

---

**2. Optimizer Configuration:**

| Parameter Group | Learning Rate | Weight Decay | Rationale |
|-----------------|---------------|--------------|-----------|
| **ViT encoder (layers 0-6)** | 1e-6 | 0.01 | Lower layers, more general features |
| **ViT encoder (layers 7-12)** | 5e-6 | 0.01 | Higher layers, more task-specific |
| **Cross-attention** | 5e-5 | 0.01 | New layers, need faster learning |
| **GPT-2 decoder (early)** | 1e-6 | 0.01 | Preserve language ability |
| **GPT-2 decoder (late)** | 5e-6 | 0.01 | Adapt to captioning task |
| **LM head** | 1e-5 | 0.01 | Output layer, moderate adaptation |

```python
optimizer = AdamW([
    {'params': vit_encoder.layers[:6].parameters(), 'lr': 1e-6},
    {'params': vit_encoder.layers[6:].parameters(), 'lr': 5e-6},
    {'params': cross_attention.parameters(), 'lr': 5e-5},
    {'params': gpt2_decoder.transformer.h[:6].parameters(), 'lr': 1e-6},
    {'params': gpt2_decoder.transformer.h[6:].parameters(), 'lr': 5e-6},
    {'params': gpt2_decoder.lm_head.parameters(), 'lr': 1e-5},
], weight_decay=0.01, betas=(0.9, 0.98))
```

---

**3. Learning Rate Schedule:**

```
Total training:
- 3M images / 256 batch = ~12K steps per epoch
- 1 week budget ≈ 8-10 epochs ≈ 100K steps

Schedule:
- Warmup: 5K steps (5% of training)
- Peak LR: As specified per group
- Decay: Cosine to 10% of peak
- Min LR: 1e-7
```

```python
total_steps = 100000
warmup_steps = 5000

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
```

---

**4. Normalization Configuration:**

| Component | Normalization | Placement | Rationale |
|-----------|---------------|-----------|-----------|
| **ViT encoder** | LayerNorm | Pre-LN (already in ViT) | Transformer standard |
| **Cross-attention** | LayerNorm | Pre-LN | Match Transformer pattern |
| **GPT-2 decoder** | LayerNorm | Pre-LN (GPT-2 default) | Already configured |

```python
# Cross-attention block with Pre-LN
class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model):
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=12)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model)

    def forward(self, x, encoder_output):
        # Pre-LN for cross-attention
        x = x + self.cross_attn(self.norm1(x), encoder_output, encoder_output)
        # Pre-LN for FFN
        x = x + self.ffn(self.norm2(x))
        return x
```

---

**5. Regularization Strategy:**

| Technique | Configuration | Component |
|-----------|---------------|-----------|
| **Dropout** | 0.1 in attention, 0.1 in FFN | All Transformer layers |
| **Weight Decay** | 0.01 via AdamW | All parameters |
| **Label Smoothing** | 0.1 | Caption generation loss |
| **Image Augmentation** | RandAugment, RandomCrop, HorizontalFlip | Training images |
| **Gradient Clipping** | max_norm=1.0 | Prevent gradient explosion |

```python
# Loss with label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Image augmentation
image_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    RandAugment(num_ops=2, magnitude=9),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
```

---

**6. Distributed Training Configuration:**

```python
# PyTorch DistributedDataParallel setup
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])

# Create model on specific GPU
model = ImageCaptionModel().to(local_rank)
model = DDP(model, device_ids=[local_rank])

# Distributed sampler for data
train_sampler = DistributedSampler(train_dataset)
train_loader = DataLoader(
    train_dataset,
    batch_size=64,  # Per GPU
    sampler=train_sampler,
    num_workers=8,
    pin_memory=True
)

# Effective batch size: 64 × 4 = 256
```

---

**7. Mixed Precision Configuration:**

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    optimizer.zero_grad()

    # Forward pass in mixed precision
    with autocast(dtype=torch.bfloat16):
        images, captions = batch
        outputs = model(images, captions[:, :-1])
        loss = criterion(outputs, captions[:, 1:])

    # Backward pass with gradient scaling
    scaler.scale(loss).backward()

    # Unscale and clip gradients
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
```

**Memory Estimate:**
```
Model (BF16): 250M × 2 bytes = 500MB
Optimizer states (FP32): 250M × 8 bytes = 2GB
Gradients (BF16): 250M × 2 bytes = 500MB
Activations (batch 64): ~15GB
Total per GPU: ~18GB (fits in 80GB with room for larger batch)
```

---

**8. Monitoring Metrics:**

| Metric | Track For | Warning Sign |
|--------|-----------|--------------|
| **Training loss** | Learning progress | Not decreasing after warmup |
| **Validation loss** | Generalization | Increasing while train decreases |
| **BLEU/CIDEr score** | Caption quality | Plateaus or decreases |
| **Gradient norm** | Training stability | Spikes or consistently high |
| **Learning rate** | Schedule correctness | Not following expected curve |
| **GPU memory** | OOM prevention | Approaching limit |
| **Throughput** | Efficiency | Sudden drops (data loading issue) |

```python
# Logging setup
for step, batch in enumerate(train_loader):
    # ... training step ...

    if step % 100 == 0:
        wandb.log({
            'train_loss': loss.item(),
            'learning_rate': scheduler.get_last_lr()[0],
            'grad_norm': compute_grad_norm(model),
            'gpu_memory_gb': torch.cuda.max_memory_allocated() / 1e9,
            'throughput': images_per_second,
        })

    if step % 1000 == 0:
        val_metrics = evaluate(model, val_loader)
        wandb.log({
            'val_loss': val_metrics['loss'],
            'val_bleu': val_metrics['bleu'],
            'val_cider': val_metrics['cider'],
        })
```

---

**9. Checkpointing Strategy:**

| Checkpoint Type | Frequency | What to Save |
|-----------------|-----------|--------------|
| **Regular** | Every 5K steps | Model, optimizer, scheduler, step, RNG state |
| **Best** | When val BLEU improves | Same as regular |
| **Final** | End of training | Same + full training config |

```python
def save_checkpoint(model, optimizer, scheduler, step, path):
    torch.save({
        'step': step,
        'model_state_dict': model.module.state_dict(),  # Unwrap DDP
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state(),
    }, path)

# Save regular checkpoints
if step % 5000 == 0:
    save_checkpoint(model, optimizer, scheduler, step,
                   f'checkpoints/step_{step}.pt')

# Save best model
if val_bleu > best_val_bleu:
    best_val_bleu = val_bleu
    save_checkpoint(model, optimizer, scheduler, step,
                   'checkpoints/best_model.pt')
```

---

**Common Mistakes:**
| Mistake | Why It's Wrong | Prevention |
|---------|---------------|------------|
| Same LR for all components | Pre-trained parts should have lower LR | Use parameter groups |
| No warmup | Cross-attention is random; Adam moments wrong | Always warmup |
| Forgetting DDP sync | Gradients not averaged across GPUs | Use DDP properly |

---

## Problem 5: Debug/Fix
### Training Failure Diagnosis

**Concept:** Training Dynamics (All Concepts)
**Cognitive Level:** Analyze
**Prerequisites:** Understanding of optimization, gradient flow, normalization

---

**Problem Statement:**

A team is training a 12-layer Transformer encoder for text classification and encountering issues. Analyze each scenario and provide the diagnosis and fix.

**Scenario A:**
```
Training log:
Step 0: Loss = 2.3
Step 100: Loss = 2.3
Step 1000: Loss = 2.3
Step 5000: Loss = 2.3
(Loss is constant, never decreases)

Model: 12-layer Transformer encoder
Optimizer: Adam, lr=1e-4
Normalization: LayerNorm (Post-LN)
Initialization: Default PyTorch
```

**Scenario B:**
```
Training log:
Step 0: Loss = 2.3
Step 100: Loss = 0.8
Step 200: Loss = 0.4
Step 300: Loss = NaN
(Loss decreases then explodes)

Model: Same as above
Optimizer: Adam, lr=1e-2
Initialization: Default PyTorch
Warmup: None
```

**Scenario C:**
```
Training log:
Step 0: Loss = 2.3, Val Loss = 2.3
Step 5000: Loss = 0.1, Val Loss = 0.9
Step 10000: Loss = 0.05, Val Loss = 1.2
Step 15000: Loss = 0.02, Val Loss = 1.5
(Train loss very low, val loss increasing)

Model: Same as above
Data: 5,000 training samples, 1,000 validation
Regularization: None
Dropout: 0.0
```

**For each scenario:**
1. Diagnose the root cause
2. Explain the mechanism causing the problem
3. Provide the specific fix
4. Explain how to prevent this in the future

---

**Solution:**

**Scenario A: Loss Never Decreases**

**Diagnosis:** Gradient flow is blocked or severely diminished

**Investigation Steps:**
```python
# Check gradient magnitudes
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm = {param.grad.norm():.6f}")

# Typical finding:
# layer_0.attention.weight: grad_norm = 0.000001
# layer_11.attention.weight: grad_norm = 0.001
# → Early layers have ~1000x smaller gradients!
```

**Root Cause:** Post-LN in deep Transformer + default initialization causes vanishing gradients. With Post-LN, gradients must flow through LayerNorm after residual connection, which can diminish gradients in deep networks.

**Fix:**
```python
# Option 1: Switch to Pre-LN (recommended)
class PreLNTransformerLayer(nn.Module):
    def forward(self, x):
        x = x + self.attention(self.norm1(x))  # Pre-LN
        x = x + self.ffn(self.norm2(x))        # Pre-LN
        return x

# Option 2: Better initialization
for module in model.modules():
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1/math.sqrt(2 * num_layers))

# Option 3: Increase learning rate (may help but not root fix)
optimizer = Adam(model.parameters(), lr=1e-3)
```

**Prevention:**
- Always use Pre-LN for deep Transformers (>6 layers)
- Monitor per-layer gradient norms early in training
- Use proven architectures (BERT, GPT-2 configs)

---

**Scenario B: Loss Explodes to NaN**

**Diagnosis:** Learning rate too high + no warmup → gradient explosion

**Root Cause:**
1. LR=1e-2 is very high for Adam (typical is 1e-4 to 1e-3)
2. No warmup means Adam's moment estimates (m, v) are inaccurate early
3. Combined effect: huge initial updates → activations explode → NaN

**Mechanism:**
```
Step 0: Adam moments initialized to 0
Step 1-10: Moment estimates very noisy
Step 1-10: Update = gradient / (sqrt(v) + eps) where v ≈ 0
         → Huge updates because dividing by tiny v
         → Weights become very large
         → Activations overflow → NaN
```

**Fix:**
```python
# Fix 1: Lower learning rate
optimizer = Adam(model.parameters(), lr=1e-4)

# Fix 2: Add warmup
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,  # 10% of training or ~1000 steps
    num_training_steps=10000
)

# Fix 3: Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Combined (all three):
optimizer = Adam(model.parameters(), lr=3e-4)  # Moderate LR
scheduler = get_linear_schedule_with_warmup(optimizer, 1000, 10000)

for batch in dataloader:
    loss = model(batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clip
    optimizer.step()
    scheduler.step()  # Warmup
    optimizer.zero_grad()
```

**Prevention:**
- Always use warmup for Transformers
- Start with conservative LR (1e-4), increase if stable
- Always use gradient clipping (1.0 is safe default)

---

**Scenario C: Severe Overfitting**

**Diagnosis:** Massive overfitting—model memorizing training data

**Root Cause:**
- 12-layer Transformer ≈ 100M+ parameters
- Only 5,000 training samples
- No regularization at all
- Model has capacity to perfectly memorize training data

**Evidence:**
```
Ratio: Parameters / Samples = 100M / 5000 = 20,000:1 (way overparameterized!)
Train loss → 0.02 (memorized)
Val loss increasing (not generalizing)
```

**Fix:**
```python
# Fix 1: Use pre-trained model instead of training from scratch
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Fix 2: Add regularization (multiple techniques)
model = TransformerEncoder(dropout=0.3)  # High dropout
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.1)  # High WD

# Fix 3: Data augmentation (for text)
def augment_text(text):
    # Back-translation, synonym replacement, etc.
    return augmented_text

# Fix 4: Early stopping
best_val_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(100):
    train_loss = train_epoch(model, train_loader)
    val_loss = evaluate(model, val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        save_best_model(model)
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# Fix 5: Reduce model capacity (if not using pre-trained)
model = TransformerEncoder(
    num_layers=4,    # Fewer layers
    hidden_dim=256,  # Smaller hidden
    num_heads=4      # Fewer heads
)
```

**Expected Results After Fix:**
```
With pre-trained BERT + fine-tuning + early stopping:
Epoch 1: Train Loss = 0.5, Val Loss = 0.4
Epoch 3: Train Loss = 0.3, Val Loss = 0.3
Epoch 5: Train Loss = 0.2, Val Loss = 0.28 (early stop here)
```

**Prevention:**
- Always start with pre-trained models for limited data
- Include dropout (0.1-0.5) and weight decay (0.01-0.1)
- Use early stopping based on validation metrics
- Consider data augmentation for text

---

**Summary Table:**

| Scenario | Problem | Root Cause | Fix |
|----------|---------|------------|-----|
| A | Loss stuck | Vanishing gradients (Post-LN) | Switch to Pre-LN |
| B | Loss → NaN | High LR + no warmup | Lower LR + warmup + clipping |
| C | Overfitting | Too few samples, no regularization | Pre-trained model + dropout + early stopping |

---

**Common Debugging Checklist:**

```
□ Loss not decreasing?
  → Check gradient norms per layer
  → Verify learning rate not too low
  → Check for vanishing gradients (Pre-LN vs Post-LN)

□ Loss becomes NaN?
  → Reduce learning rate
  → Add warmup
  → Add gradient clipping
  → Check for division by zero in code

□ Train/val diverging?
  → Add regularization (dropout, weight decay)
  → Use pre-trained models
  → Add early stopping
  → Get more data or augment
```

---

## Self-Assessment Guide

### Mastery Checklist

| Problem | Mastery Indicator | Check |
|---------|-------------------|-------|
| **1 (Warm-Up)** | Can compute gradients through ReLU | ☐ |
| **2 (Skill-Builder)** | Can select optimizer for scenario | ☐ |
| **3 (Skill-Builder)** | Can design regularization strategy | ☐ |
| **4 (Challenge)** | Can design complete training pipeline | ☐ |
| **5 (Debug/Fix)** | Can diagnose training failures | ☐ |

### Progression Path

```
If struggled with Problem 1:
  → Review: Backpropagation, ReLU gradient
  → Flashcard: Card 1 (Easy)

If struggled with Problem 2:
  → Review: Optimizers section
  → Flashcard: Card 2 (Easy)

If struggled with Problem 3:
  → Review: Regularization section
  → Flashcard: Cards 3-4 (Medium)

If struggled with Problem 4:
  → Review: All Core Concepts
  → Flashcard: Card 5 (Hard)

If struggled with Problem 5:
  → Review: Gradient flow, normalization
  → All Flashcards for comprehensive review
```

---

*Generated from Lesson 5: Deep Learning | Practice Problems Skill*
