# Flashcards: Deep Learning

**Source:** notes/deep-learning/deep-learning-study-notes.md
**Concept Map:** notes/deep-learning/concept-maps/deep-learning-concept-map.md
**Date Generated:** 2026-01-07
**Total Cards:** 5
**Distribution:** 2 Easy | 2 Medium | 1 Hard

---

## Card Distribution Strategy

| Difficulty | Count | Concepts Covered | Selection Rationale |
|------------|-------|------------------|---------------------|
| Easy | 2 | Optimization, Skip Connections | Critical centrality (9-11 connections); foundational practice |
| Medium | 2 | Regularization, Generative Models | High centrality; practical application |
| Hard | 1 | Training at Scale | Integration across concepts; production relevance |

---

## Easy Cards

### Card 1 of 5 | Easy
**Concept:** Optimizers and Learning Rate
**Centrality:** Critical (11 connections)
**Related Concepts:** SGD, Momentum, Adam, AdamW, Learning Rate Scheduling

#### Front
Compare SGD with momentum and Adam. When would you choose each optimizer, and what is the most critical hyperparameter for both?

#### Back
**SGD with Momentum:**
- Updates: v = βv + ∇L; θ = θ - ηv
- Accumulates gradient history (momentum β ≈ 0.9)
- Often achieves **better final performance** with careful tuning
- Preferred for: Final training runs, when you have time to tune

**Adam (Adaptive Moment Estimation):**
- Maintains per-parameter adaptive learning rates
- Combines momentum (first moment) + RMSprop (second moment)
- **Fast convergence**, less sensitive to initial LR
- Preferred for: Prototyping, Transformers, when you need quick results

**AdamW** (recommended over Adam):
- Fixes weight decay implementation (decoupled from gradient)
- Default for modern Transformers

| Aspect | SGD + Momentum | Adam/AdamW |
|--------|---------------|------------|
| Convergence | Slower | Faster |
| Final performance | Often better | Slightly worse |
| Tuning needed | More | Less |
| Best for | CNNs, final runs | Transformers, prototyping |

**Most Critical Hyperparameter: Learning Rate**
- Too high → Divergence (loss explodes or oscillates)
- Too low → Slow convergence, stuck in local minima
- Use LR scheduling: warmup → peak → decay

#### Mnemonic
**"Adam for Acceleration, SGD for Superiority"** — Adam converges fast; SGD often achieves better final results.

#### Common Misconceptions
- ❌ Adam always outperforms SGD (SGD often better with proper tuning)
- ❌ Default Adam LR (0.001) works everywhere (often too high; try 1e-4 to 3e-4)
- ❌ Higher learning rate = faster training (too high causes instability)

---

### Card 2 of 5 | Easy
**Concept:** Skip Connections and ResNets
**Centrality:** Critical (9 connections)
**Related Concepts:** Residual Learning, Vanishing Gradients, DenseNet

#### Front
What problem do skip connections solve, and how does the residual learning formulation make training deep networks easier?

#### Back
**Problem Solved: Degradation Problem**

Without skip connections, very deep networks perform *worse* than shallower ones—even on training data. This isn't overfitting; it's optimization difficulty.

**Two issues addressed:**

1. **Vanishing Gradients:**
   - Gradients multiply through layers during backprop
   - Many multiplications → gradients shrink exponentially
   - Skip connections provide direct gradient paths

2. **Identity Mapping Difficulty:**
   - Learning H(x) = x through convolutions is hard
   - With skip connections: output = F(x) + x
   - To learn identity, just push F(x) → 0 (easier!)

**Residual Learning:**
```
Traditional: Learn H(x) directly
Residual:    Learn F(x) = H(x) - x, then output F(x) + x

If optimal is near identity:
- Traditional must learn complex identity mapping
- Residual just learns F(x) ≈ 0
```

**Gradient Flow:**
```
∂L/∂x = ∂L/∂(F(x)+x) × (∂F/∂x + 1)
                              ↑
                    Identity term ensures gradient ≥ 1
```

**Impact:** Enabled networks with 100+ layers (ResNet-152, ResNet-1001)

#### Mnemonic
**"Skip to Stay Alive"** — Skip connections keep gradients alive in deep networks.

#### Common Misconceptions
- ❌ Skip connections add many parameters (identity shortcuts add zero)
- ❌ The problem was overfitting (it was optimization/degradation)
- ❌ Skip connections are only for CNNs (Transformers use them too!)

---

## Medium Cards

### Card 3 of 5 | Medium
**Concept:** Regularization and Normalization
**Centrality:** High (7 connections each)
**Related Concepts:** Dropout, Weight Decay, BatchNorm, LayerNorm, Data Augmentation

#### Front
Explain the mechanisms of Dropout, Batch Normalization, and Weight Decay. How does each prevent overfitting, and when would you use BatchNorm vs. LayerNorm?

#### Back
**Dropout:**
- **Mechanism:** Randomly zero p% of activations during training
- **Why it works:**
  - Prevents co-adaptation of neurons
  - Implicit ensemble of 2^n sub-networks
  - Forces redundant representations
- **Usage:** After dense layers; p=0.5 typical; scale outputs at test time

**Weight Decay (L2 Regularization):**
- **Mechanism:** Add λΣw² to loss; equivalently, decay weights by (1-ηλ) each step
- **Why it works:**
  - Penalizes large weights → smoother functions
  - Encourages using all features rather than relying on few
- **Usage:** λ = 1e-4 to 1e-2; use AdamW for proper implementation

**Batch Normalization:**
- **Mechanism:** Normalize activations to mean=0, var=1 across batch, then scale/shift
- **Why it works:**
  - Reduces internal covariate shift
  - Enables higher learning rates
  - Provides mild regularization (batch noise)
- **Formula:** BN(x) = γ × (x - μ_B) / σ_B + β

**BatchNorm vs LayerNorm:**

| Aspect | BatchNorm | LayerNorm |
|--------|-----------|-----------|
| Normalizes across | Batch dimension | Feature dimension |
| Batch dependency | Yes (needs batch stats) | No (single sample OK) |
| Best for | CNNs, large batches | Transformers, RNNs, small batches |
| Inference | Uses running statistics | Same as training |

**Decision Rule:**
- CNNs with batch ≥ 32: **BatchNorm**
- Transformers, any batch size: **LayerNorm**
- Small batches: **LayerNorm** or GroupNorm

#### Mnemonic
**"DROP weights, BATCH across samples, LAYER across features"**

#### Common Misconceptions
- ❌ BatchNorm is just for faster training (also regularizes)
- ❌ Dropout during inference (only during training!)
- ❌ More dropout = more regularization always helps (too much hurts learning)

#### Critical Flag
⚠️ BatchNorm behaves differently in training vs. eval mode. Always call `model.eval()` before inference!

---

### Card 4 of 5 | Medium
**Concept:** Generative Models
**Centrality:** High (6 connections)
**Related Concepts:** GANs, VAEs, Diffusion Models, Latent Space

#### Front
Compare GANs, VAEs, and Diffusion Models. What are their training objectives, strengths, and weaknesses?

#### Back
**Generative Adversarial Networks (GANs):**

**Architecture:** Generator G vs. Discriminator D
```
Objective: min_G max_D [E[log D(x)] + E[log(1 - D(G(z)))]]
```

| Strengths | Weaknesses |
|-----------|------------|
| Sharp, realistic images | Training instability |
| Fast sampling | Mode collapse (limited diversity) |
| No explicit density | No encoder (can't get latent of real image) |

**Variational Autoencoders (VAEs):**

**Architecture:** Encoder → Latent z ~ N(μ, σ) → Decoder
```
Objective: Maximize ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))
           (Reconstruction)    (Regularization)
```

| Strengths | Weaknesses |
|-----------|------------|
| Stable training | Blurry outputs |
| Principled probabilistic model | Limited sample quality |
| Encoder gives latent codes | KL term limits expressivity |

**Diffusion Models (Current SOTA):**

**Process:** Gradually add noise (forward), learn to denoise (reverse)
```
Forward: x_t = √(α_t) x_0 + √(1-α_t) ε
Reverse: Learn to predict ε from x_t (denoising)
```

| Strengths | Weaknesses |
|-----------|------------|
| Best image quality | Very slow sampling (many steps) |
| Stable training | High compute for training |
| Mode coverage (no collapse) | Memory intensive |

**Summary Comparison:**

| Model | Quality | Speed | Stability | Diversity |
|-------|---------|-------|-----------|-----------|
| GAN | High | Fast | Unstable | Mode collapse risk |
| VAE | Medium | Fast | Stable | Good |
| Diffusion | Highest | Slow | Stable | Excellent |

#### Mnemonic
**"GANs fight, VAEs compress, Diffusion denoises"**

#### Common Misconceptions
- ❌ GANs are still SOTA for images (Diffusion models surpassed them)
- ❌ VAE blurriness is unfixable (VQ-VAE and hierarchical VAEs help)
- ❌ Diffusion is just for images (also works for audio, video, molecules)

---

## Hard Cards

### Card 5 of 5 | Hard
**Concept:** Training at Scale
**Centrality:** Integration (spans optimization, architecture, efficiency)
**Related Concepts:** Data Parallelism, Model Parallelism, Mixed Precision, Gradient Checkpointing

#### Front
You need to train a 10B parameter language model on a cluster with 64 A100 GPUs (80GB each). Design the training strategy addressing: parallelism approach, memory optimization, numerical precision, and learning rate considerations for large batch training.

#### Back
**1. Parallelism Strategy:**

**Data Parallelism (Primary):**
- Replicate model on each GPU
- Each GPU processes different batch
- Synchronize gradients via AllReduce
- Scale: 64 GPUs × batch_per_GPU

**But 10B params ≈ 40GB in FP32** — fits in 80GB but tight!

**Add Model Parallelism:**
- **Tensor Parallelism:** Split layers across GPUs (e.g., 8-way TP)
- **Pipeline Parallelism:** Different layers on different GPUs
- **Typical config:** 8-way tensor parallel × 8-way data parallel = 64 GPUs

```
GPU Layout (example):
┌─────────────────────────────────────────────┐
│  TP Group 1 (8 GPUs) │ TP Group 2 (8 GPUs)  │  ... (8 groups)
│  [Layer shards]      │ [Layer shards]       │
│  Same data batch     │ Different batch      │  Data Parallel
└─────────────────────────────────────────────┘
```

**2. Memory Optimization:**

| Technique | Memory Saved | Trade-off |
|-----------|--------------|-----------|
| Mixed Precision (FP16/BF16) | ~50% | Minimal accuracy loss |
| Gradient Checkpointing | ~60-70% | 20-30% slower |
| ZeRO (optimizer sharding) | ~8× | Communication overhead |
| Activation Checkpointing | Variable | Recompute activations |

**Recommended:** Mixed precision + ZeRO Stage 2 + selective checkpointing

**3. Mixed Precision Training:**

```python
# PyTorch AMP example
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**BF16 vs FP16:**
- BF16: Same exponent range as FP32; more stable; preferred for LLMs
- FP16: Needs loss scaling to prevent underflow; higher precision mantissa

**4. Large Batch Learning Rate:**

**Linear Scaling Rule:** LR × k when batch size × k
```
Base: batch=256, LR=1e-4
64 GPUs × 32 per GPU = 2048 batch
Scale: LR = 1e-4 × (2048/256) = 8e-4
```

**BUT: Requires warmup!**
```
Warmup steps: 5-10% of training
LR schedule: 0 → peak_LR (linear warmup) → decay (cosine)
```

**5. Complete Configuration:**

```yaml
# Example config for 10B model on 64×A100
model:
  params: 10B
  precision: bf16

parallelism:
  tensor_parallel: 8
  data_parallel: 8  # 8×8=64 GPUs
  pipeline_parallel: 1

optimization:
  optimizer: AdamW
  base_lr: 1e-4
  scaled_lr: 8e-4  # for 2048 batch
  weight_decay: 0.1
  warmup_steps: 2000
  schedule: cosine

memory:
  mixed_precision: bf16
  gradient_checkpointing: true
  zero_stage: 2

training:
  global_batch_size: 2048
  micro_batch_size: 32  # per GPU
  gradient_accumulation: 1
```

**Key Monitoring:**
- Loss scale (for FP16)
- Gradient norms (detect instability)
- Memory utilization per GPU
- Communication overhead

#### Common Misconceptions
- ❌ Just add more GPUs for larger models (memory per GPU matters)
- ❌ FP16 always works (BF16 more stable for LLMs; FP16 needs careful scaling)
- ❌ Linear LR scaling works without warmup (large batches need warmup!)
- ❌ More parallelism is always better (communication overhead limits scaling)

---

## Anki Export Format

```
# Card 1 - Easy - Optimizers
Compare SGD with momentum and Adam. When would you choose each optimizer, and what is the most critical hyperparameter?	SGD+Momentum: accumulates gradients, often better final performance, needs tuning. Adam: adaptive per-parameter LR, fast convergence, less tuning. AdamW preferred for Transformers. Most critical hyperparameter: Learning Rate. Use scheduling (warmup → decay).	deep-learning optimization

# Card 2 - Easy - Skip Connections
What problem do skip connections solve, and how does residual learning make training easier?	Solves degradation problem (deep nets train worse than shallow). Skip connections: output = F(x) + x. To learn identity, just push F(x)→0 (easier than learning H(x)=x). Provides gradient highway: ∂L/∂x includes +1 term ensuring gradient flow.	deep-learning architecture

# Card 3 - Medium - Regularization
Explain Dropout, BatchNorm, and Weight Decay mechanisms. When use BatchNorm vs LayerNorm?	Dropout: randomly zero activations; prevents co-adaptation. Weight Decay: penalize large weights; smoother functions. BatchNorm: normalize across batch; enables higher LR. Use BatchNorm for CNNs (large batches), LayerNorm for Transformers (batch-independent).	deep-learning regularization

# Card 4 - Medium - Generative Models
Compare GANs, VAEs, and Diffusion Models.	GANs: adversarial training, sharp images, unstable, mode collapse. VAEs: encoder-decoder, stable, blurry, principled. Diffusion: iterative denoising, best quality, slow sampling, stable. Current SOTA: Diffusion models.	deep-learning generative

# Card 5 - Hard - Training at Scale
Design training for 10B param model on 64 GPUs.	Use tensor parallelism (8-way) × data parallelism (8-way). Memory: BF16 mixed precision + ZeRO Stage 2 + gradient checkpointing. LR: linear scaling with warmup (base_LR × batch_scale). Monitor: gradient norms, loss scale, memory.	deep-learning scale
```

---

## Review Schedule

| Card | First Review | Second Review | Third Review | Mastery Review |
|------|--------------|---------------|--------------|----------------|
| Card 1 (Easy) | Day 1 | Day 3 | Day 7 | Day 14 |
| Card 2 (Easy) | Day 1 | Day 3 | Day 7 | Day 14 |
| Card 3 (Medium) | Day 1 | Day 4 | Day 10 | Day 21 |
| Card 4 (Medium) | Day 2 | Day 5 | Day 12 | Day 25 |
| Card 5 (Hard) | Day 3 | Day 7 | Day 14 | Day 30 |

---

## Cross-References

| Card | Study Notes Section | Concept Map Node | Practice Problem |
|------|---------------------|------------------|------------------|
| Card 1 | Concept 4: Optimizers | Adam (11 connections) | Problem 1 |
| Card 2 | Concept 8: Skip Connections | Skip Connections (9) | Problem 2 |
| Card 3 | Concepts 6-7: Regularization, Normalization | BatchNorm (7), Dropout (6) | Problem 3 |
| Card 4 | Concept 9: Generative Models | GANs (6), VAEs (4) | Problem 4 |
| Card 5 | Concept 10: Training at Scale | Data Parallelism (4) | Problem 5 |
