# Flashcard Set: Lesson 5 - Deep Learning

**Source:** Lessons/Lesson_5.md
**Subject Area:** AI Learning - Deep Learning: Neural Networks, Optimization, and Training Techniques
**Date Generated:** 2026-01-08
**Total Cards:** 5
**Distribution:** 2 Easy | 2 Medium | 1 Hard

---

## Critical Knowledge Summary

Concepts appearing in multiple cards (prioritize for review):
- **Gradient Flow**: Appears in Cards 1, 2, 4, 5 (training foundation)
- **Optimization**: Appears in Cards 2, 3, 5 (parameter updates)
- **Regularization**: Appears in Cards 3, 4, 5 (generalization)
- **Normalization**: Appears in Cards 4, 5 (training stability)

---

## Flashcards

---
### Card 1 | Easy
**Cognitive Level:** Remember
**Concept:** Backpropagation and Gradient Flow
**Source Section:** Core Concepts - Concept 4

**FRONT (Question):**
What is the vanishing gradient problem, what causes it, and what are three solutions?

**BACK (Answer):**
**Definition:**
Vanishing gradients occur when gradients shrink exponentially as they propagate backward through a deep network, causing early layers to receive near-zero gradients and fail to learn.

**Cause:**
Gradients are products of many terms (chain rule):
```
∂Loss/∂W_1 = ∂Loss/∂h_n × ∂h_n/∂h_{n-1} × ... × ∂h_2/∂h_1 × ∂h_1/∂W_1
```
If each term < 1 consistently, the product → 0 exponentially.

**Contributing Factors:**
| Factor | Why It Causes Vanishing |
|--------|------------------------|
| Sigmoid/tanh activation | Gradient approaches 0 in saturated regions |
| Poor initialization | Activations start in saturated regions |
| Deep networks | More multiplicative terms in chain |

**Three Solutions:**

| Solution | How It Helps |
|----------|--------------|
| **ReLU activation** | Gradient = 1 for positive inputs (no shrinkage) |
| **Residual connections** | y = F(x) + x → ∂y/∂x = ∂F/∂x + 1 (always ≥1) |
| **Proper initialization** | Xavier/He init keeps activations in non-saturated region |

**Additional solutions:** BatchNorm/LayerNorm, gradient clipping (for exploding), LSTM/GRU (for RNNs)

**Critical Knowledge Flag:** Yes - Understanding gradient flow is essential for deep network training

---

---
### Card 2 | Easy
**Cognitive Level:** Understand
**Concept:** Optimizers (SGD, Adam, AdamW)
**Source Section:** Core Concepts - Concept 5

**FRONT (Question):**
Compare SGD with momentum, Adam, and AdamW. When would you choose each optimizer?

**BACK (Answer):**
**Optimizer Comparison:**

| Optimizer | Update Rule | Key Property |
|-----------|-------------|--------------|
| **SGD** | θ = θ - η∇L | Simple, requires careful LR tuning |
| **SGD + Momentum** | v = βv + ∇L; θ = θ - ηv | Accumulates velocity; escapes saddle points |
| **Adam** | Uses first (m) and second (v) moment estimates; adaptive per-parameter LR | Fast convergence; good default |
| **AdamW** | Adam + decoupled weight decay | Proper L2 regularization; preferred for Transformers |

**When to Choose:**

| Scenario | Best Choice | Rationale |
|----------|-------------|-----------|
| **CNNs, well-tuned pipeline** | SGD + momentum | Often achieves best final accuracy with proper tuning |
| **Quick experiments, new tasks** | Adam | Fast convergence, less hyperparameter sensitivity |
| **Transformers, LLMs** | AdamW | Decoupled weight decay interacts properly with adaptive LR |
| **Very large models** | AdamW + LAMB | Layer-wise scaling for large batch training |

**Key Hyperparameters:**
```
SGD: lr=0.1, momentum=0.9
Adam: lr=1e-3, β1=0.9, β2=0.999, ε=1e-8
AdamW: lr=1e-4 to 1e-3, weight_decay=0.01-0.1
```

**Critical Insight:** AdamW fixes the issue where Adam's L2 regularization scales with gradient magnitude, making regularization inconsistent across parameters.

**Critical Knowledge Flag:** Yes - Optimizer choice significantly impacts training dynamics

---

---
### Card 3 | Medium
**Cognitive Level:** Apply
**Concept:** Learning Rate Scheduling
**Source Section:** Core Concepts - Concept 6

**FRONT (Question):**
You're training a Transformer model for 100,000 steps. Design a learning rate schedule including:
1. Warmup configuration
2. Decay strategy
3. Justification for each choice

**BACK (Answer):**
**Recommended Schedule: Linear Warmup + Cosine Decay**

```
Peak LR: 1e-4
Warmup steps: 2,000 (2% of training)
Min LR: 1e-5 (10% of peak)

Schedule:
Steps 0-2,000: Linear warmup (0 → 1e-4)
Steps 2,000-100,000: Cosine decay (1e-4 → 1e-5)
```

**Visual:**
```
LR
1e-4 |      ___________
     |     /           \
     |    /             \
     |   /               \___
1e-5 |  /                    \
     |_/________________________
       0    2K              100K  Steps
```

**Component Justification:**

| Component | Configuration | Justification |
|-----------|---------------|---------------|
| **Warmup** | 2,000 steps, linear | Adam's moment estimates are inaccurate early; prevents divergence from large initial updates |
| **Peak LR** | 1e-4 | Standard for AdamW with Transformers; higher may diverge |
| **Cosine decay** | To 10% of peak | Smooth decay spends more time at low LR for fine-tuning; no sharp transitions |
| **Min LR** | 1e-5 (not 0) | Prevents complete stagnation; allows continued small updates |

**Alternative Options:**

| Alternative | When to Use |
|-------------|-------------|
| **Longer warmup (5%)** | Very large batches, unstable training |
| **Linear decay** | Simpler, nearly as effective |
| **Cosine with restarts** | Want periodic exploration; multi-stage training |
| **Constant after warmup** | Short fine-tuning runs |

**Implementation (PyTorch):**
```python
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=2000,
    num_training_steps=100000
)
```

**Critical Knowledge Flag:** Yes - LR schedule is critical for Transformer training stability

---

---
### Card 4 | Medium
**Cognitive Level:** Analyze
**Concept:** Normalization (BatchNorm vs LayerNorm)
**Source Section:** Core Concepts - Concept 8

**FRONT (Question):**
Analyze why Transformers use LayerNorm instead of BatchNorm. Compare the two across: normalization dimension, batch dependency, inference behavior, and use cases.

**BACK (Answer):**
**Comparison Table:**

| Aspect | BatchNorm | LayerNorm |
|--------|-----------|-----------|
| **Normalizes over** | Batch dimension | Feature dimension |
| **Formula** | x̂ = (x - μ_B)/σ_B | x̂ = (x - μ_L)/σ_L |
| **Statistics** | Mean/std across batch | Mean/std across features |
| **Batch dependency** | Yes (needs batch stats) | No (per-sample) |
| **Training vs Inference** | Different (running stats at inference) | Same |
| **Best for** | CNNs, large fixed batches | Transformers, RNNs, variable batches |

**Why Transformers Use LayerNorm:**

| Reason | Explanation |
|--------|-------------|
| **Variable sequence length** | Batches have different sequence lengths; batch stats unreliable |
| **Small effective batch** | With long sequences, effective batch per position is small |
| **Consistent train/inference** | No running statistics to maintain; same behavior always |
| **Autoregressive generation** | At inference, batch size = 1; BatchNorm would fail |

**Visual Difference:**
```
Input tensor: [Batch, Sequence, Features]

BatchNorm: Normalize over Batch dimension for each (Seq, Feature)
           → Same feature normalized across different samples

LayerNorm: Normalize over Features dimension for each (Batch, Seq)
           → Each position normalized independently
```

**Placement in Transformers:**

| Variant | Formula | Stability |
|---------|---------|-----------|
| **Post-LN** | LN(x + sublayer(x)) | Original; needs careful warmup |
| **Pre-LN** | x + sublayer(LN(x)) | Modern default; more stable |

**When to Use Each:**

| Use Case | Choice |
|----------|--------|
| Image classification (CNN) | BatchNorm |
| Transformer encoder/decoder | LayerNorm (Pre-LN) |
| RNN/LSTM | LayerNorm |
| Batch size = 1 at inference | LayerNorm |
| Style transfer | Instance Norm |

**Critical Knowledge Flag:** Yes - Normalization choice fundamentally affects architecture design

---

---
### Card 5 | Hard
**Cognitive Level:** Synthesize
**Concept:** Complete Training Pipeline Design
**Source Section:** All Core Concepts

**FRONT (Question):**
Design a complete training pipeline for a 24-layer Transformer language model (1B parameters) trained from scratch on 100B tokens. Your design must include:

1. Architecture decisions (normalization, activation, residuals)
2. Optimization setup (optimizer, LR schedule, gradient handling)
3. Regularization strategy
4. Efficiency techniques for large-scale training
5. Monitoring and debugging strategy

Justify each choice based on deep learning principles.

**BACK (Answer):**
**1. Architecture Decisions:**

| Component | Choice | Justification |
|-----------|--------|---------------|
| **Normalization** | Pre-LN (RMSNorm) | More stable gradient flow; RMSNorm is faster than full LayerNorm |
| **Activation** | SwiGLU (Swish-Gated Linear Unit) | Outperforms GELU; gating provides additional expressiveness |
| **Residuals** | Standard x + sublayer(x) | Essential for 24-layer gradient flow |
| **Position** | RoPE | Relative position; good length generalization |
| **Attention** | Multi-head with Flash Attention | O(1) memory; exact computation |

```
Layer structure:
x → RMSNorm → Attention → + → RMSNorm → FFN(SwiGLU) → +
  ↑___________________________↑_________________________↑
       (residual connections)
```

**2. Optimization Setup:**

| Component | Configuration | Justification |
|-----------|---------------|---------------|
| **Optimizer** | AdamW | Proper weight decay for Transformers |
| **β1, β2** | 0.9, 0.95 | β2=0.95 (not 0.999) for stability with large models |
| **Weight decay** | 0.1 | Standard for LLM training |
| **Peak LR** | 3e-4 | Scaled for 1B model |
| **Warmup** | 2000 steps (~0.1% of training) | Stabilize Adam moments |
| **Schedule** | Cosine decay to 3e-5 | Smooth transition to fine-tuning regime |
| **Gradient clipping** | max_norm=1.0 | Prevent exploding gradients |

**LR Schedule:**
```
Steps 0-2K:      Linear warmup (0 → 3e-4)
Steps 2K-end:   Cosine decay (3e-4 → 3e-5)
```

**3. Regularization Strategy:**

| Technique | Configuration | Why |
|-----------|---------------|-----|
| **Dropout** | 0.0 (none) | Large data (100B tokens) provides sufficient regularization |
| **Weight decay** | 0.1 (via AdamW) | Implicit regularization; prevents weight explosion |
| **Label smoothing** | 0.0 | Not typically used for LM pre-training |
| **Data** | No augmentation | Text; rely on data scale |

**Rationale:** With 100B tokens, the primary concern is underfitting, not overfitting. Minimal explicit regularization; rely on data scale and implicit regularization from AdamW.

**4. Efficiency Techniques:**

| Technique | Implementation | Benefit |
|-----------|----------------|---------|
| **Mixed precision** | BF16 forward/backward, FP32 master weights | 2x memory savings, faster compute |
| **Flash Attention** | Fused kernel implementation | O(1) memory, 2-4x speedup |
| **Gradient accumulation** | Accumulate 8 micro-batches | Effective batch = 4M tokens |
| **Data parallelism** | Distributed across 64 GPUs | Linear speedup |
| **Activation checkpointing** | Checkpoint every 4 layers | Trade 25% speed for 60% memory |

**Memory Budget (per GPU, 80GB):**
```
Model params (BF16):    ~2GB
Optimizer states (FP32): ~8GB
Activations (with ckpt): ~30GB
KV cache + misc:         ~10GB
Available for batch:     ~30GB
```

**5. Monitoring and Debugging:**

| Metric | Monitor For | Action If Abnormal |
|--------|-------------|-------------------|
| **Training loss** | Steady decrease | Check LR, data loading |
| **Gradient norm** | Stable, no spikes | Clip gradients, reduce LR |
| **Per-layer grad norms** | Similar across layers | Check normalization, residuals |
| **Activation stats** | No NaN/Inf | Check initialization, LR |
| **Learning rate** | Following schedule | Verify scheduler |
| **Memory usage** | Below GPU limit | Reduce batch, add checkpointing |

**Debugging Checklist:**
```
□ Loss not decreasing → Check LR (too high/low), data pipeline
□ Loss spikes/NaN → Gradient clipping, reduce LR, check data
□ Early layers not learning → Verify Pre-LN, check residuals
□ Memory OOM → Add activation checkpointing, reduce batch
□ Slow training → Verify mixed precision, Flash Attention enabled
```

**Training Script Pseudocode:**
```python
model = Transformer(
    layers=24, d_model=2048, heads=16,
    norm='rmsnorm', activation='swiglu', pos='rope'
)

optimizer = AdamW(
    model.parameters(),
    lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1
)

scheduler = CosineWithWarmup(
    optimizer, warmup_steps=2000, total_steps=total_steps
)

scaler = GradScaler()  # For mixed precision

for batch in dataloader:
    with autocast(dtype=bfloat16):
        loss = model(batch).mean()

    scaler.scale(loss / accumulation_steps).backward()

    if step % accumulation_steps == 0:
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()
```

**Critical Knowledge Flag:** Yes - Integrates all deep learning concepts for production LLM training

---

---

## Export Formats

### Anki-Compatible (Tab-Separated)
```
What causes vanishing gradients and 3 solutions?	Gradients shrink exponentially (product of <1 terms). Solutions: ReLU (grad=1), residuals (∂y/∂x has +1), proper init (Xavier/He).	easy::gradients::deeplearning
Compare SGD, Adam, AdamW - when to use each?	SGD: best final accuracy with tuning. Adam: fast convergence, default. AdamW: Transformers (decoupled weight decay).	easy::optimizers::deeplearning
Design LR schedule for 100K step Transformer training	Linear warmup (2K steps) + cosine decay. Warmup: Adam moments inaccurate early. Cosine: smooth, more time at low LR.	medium::scheduling::deeplearning
Why do Transformers use LayerNorm not BatchNorm?	LayerNorm: per-sample (no batch dependency), consistent train/inference, works with variable sequences, batch=1 at inference.	medium::normalization::deeplearning
Design complete 1B Transformer training pipeline	Pre-LN RMSNorm, SwiGLU, RoPE, AdamW (β2=0.95), cosine decay, no dropout (large data), BF16, Flash Attention, gradient checkpointing.	hard::pipeline::deeplearning
```

### CSV Format
```csv
"Front","Back","Difficulty","Concept","Cognitive_Level"
"What causes vanishing gradients and solutions?","Product of <1 terms. Solutions: ReLU, residuals, proper initialization","Easy","Gradient Flow","Remember"
"Compare SGD, Adam, AdamW","SGD: tuned accuracy. Adam: fast default. AdamW: Transformers.","Easy","Optimizers","Understand"
"Design Transformer LR schedule","Warmup (2%) + cosine decay. Warmup stabilizes Adam; cosine smoothly reduces.","Medium","Scheduling","Apply"
"Why LayerNorm for Transformers?","Per-sample, batch-independent, consistent behavior, works at inference.","Medium","Normalization","Analyze"
"Design 1B Transformer pipeline","Pre-LN, SwiGLU, RoPE, AdamW, cosine, BF16, Flash Attention, checkpointing.","Hard","Pipeline","Synthesize"
```

---

## Source Mapping

| Card | Source Section | Key Terminology | Bloom's Level |
|------|----------------|-----------------|---------------|
| 1 | Core Concepts - Concept 4 | Vanishing gradient, ReLU, residual, initialization | Remember |
| 2 | Core Concepts - Concept 5 | SGD, momentum, Adam, AdamW | Understand |
| 3 | Core Concepts - Concept 6 | Warmup, cosine decay, learning rate | Apply |
| 4 | Core Concepts - Concept 8 | BatchNorm, LayerNorm, Pre-LN | Analyze |
| 5 | All Core Concepts | Complete training pipeline | Synthesize |

---

## Spaced Repetition Schedule

| Card | Initial Interval | Difficulty Multiplier | Recommended Review |
|------|------------------|----------------------|-------------------|
| 1 (Easy) | 1 day | 2.5x | Foundation - review first |
| 2 (Easy) | 1 day | 2.5x | Review with Card 1 |
| 3 (Medium) | 3 days | 2.0x | After mastering Cards 1-2 |
| 4 (Medium) | 3 days | 2.0x | Architecture design focus |
| 5 (Hard) | 7 days | 1.5x | Review after all others mastered |

---

## Connection to Other Lessons

| Deep Learning Concept | Transformers (L4) | LLMs (L3) | Prompt Engineering (L2) |
|-----------------------|-------------------|-----------|-------------------------|
| Gradient Flow | Residuals in attention blocks | Training stability | N/A |
| Normalization | Pre-LN vs Post-LN | Training at scale | N/A |
| Optimizers | AdamW default | RLHF optimization | N/A |
| Regularization | Dropout patterns | Fine-tuning strategies | N/A |
| Activation Functions | GELU/SwiGLU in FFN | Model architecture | N/A |

---

*Generated from Lesson 5: Deep Learning | Flashcards Skill*
