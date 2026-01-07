# Assessment Quiz: Lesson 5 - Deep Learning

**Source:** Lessons/Lesson_5.md
**Subject Area:** AI Learning - Deep Learning: Neural Networks, Optimization, and Training Techniques
**Date Generated:** 2026-01-08
**Total Questions:** 5
**Estimated Time:** 35-45 minutes

---

## Instructions

This assessment evaluates your understanding of deep learning fundamentals, including neural network architecture, gradient flow, optimization, normalization, and training at scale. Answer all questions completely, showing your reasoning where applicable.

**Question Distribution:**
- Multiple Choice (2): Conceptual understanding (Remember/Understand)
- Short Answer (2): Application and analysis (Apply/Analyze)
- Essay (1): Synthesis and evaluation (Evaluate/Synthesize)

---

## Part A: Multiple Choice (10 points each)

### Question 1: Gradient Flow Problems

**Which of the following correctly identifies BOTH a cause of vanishing gradients AND a technique that directly addresses it?**

A) Cause: Using sigmoid activations in deep networks; Solution: Applying gradient clipping

B) Cause: Products of gradients less than 1 compounding through layers; Solution: Using ReLU activation functions

C) Cause: Learning rate set too high; Solution: Implementing batch normalization

D) Cause: Using cross-entropy loss instead of MSE; Solution: Adding dropout regularization

---

### Question 2: Optimizer Selection

**A research team is training a 7-billion parameter language model with the following requirements: decoupled weight regularization, adaptive learning rates per parameter, and compatibility with mixed-precision training. Which optimizer configuration best satisfies these requirements?**

A) SGD with momentum (β = 0.9) and L2 regularization

B) Adam with β₁ = 0.9, β₂ = 0.999, and weight decay in the loss function

C) AdamW with β₁ = 0.9, β₂ = 0.95, and decoupled weight decay λ = 0.1

D) RMSprop with momentum and gradient clipping at 1.0

---

## Part B: Short Answer (15 points each)

### Question 3: Normalization Architecture

**Context:** You are designing a Transformer-based model and must choose between BatchNorm and LayerNorm, and between Pre-LN and Post-LN placement.

**Tasks:**
a) Explain why LayerNorm is preferred over BatchNorm for Transformer architectures. (5 points)

b) Compare the gradient flow properties of Pre-LN versus Post-LN placement, and explain why Pre-LN enables training deeper models without warmup. (5 points)

c) Given the equation for a Post-LN residual block: `y = LayerNorm(x + Sublayer(x))`, write the equivalent Pre-LN formulation and explain the key difference in how gradients flow. (5 points)

---

### Question 4: Learning Rate Scheduling

**Context:** You are configuring training for a BERT-style model with 110M parameters, training for 1 million steps with a peak learning rate of 1e-4.

**Tasks:**
a) Design a complete learning rate schedule specifying: warmup strategy, peak phase (if any), and decay strategy. Justify each choice. (7 points)

b) Explain what would likely happen if you removed the warmup phase entirely and started training at the peak learning rate. Connect your answer to Adam's moment estimate initialization. (4 points)

c) If training shows good progress until step 800K but then validation loss starts increasing, what schedule modification would you consider, and why? (4 points)

---

## Part C: Essay (30 points)

### Question 5: Training Pipeline Synthesis

**Prompt:** You have been tasked with training a 1.3 billion parameter Transformer language model from scratch on a cluster of 8 GPUs, each with 40GB memory. The training dataset contains 300 billion tokens. Design a complete training configuration addressing all major components.

**Your essay must address:**

1. **Optimization Configuration** (7 points)
   - Optimizer selection with hyperparameters and justification
   - Learning rate schedule (warmup, peak, decay)
   - Gradient handling strategy

2. **Regularization Strategy** (7 points)
   - Weight decay configuration
   - Dropout placement and rates (or justification for not using dropout)
   - Any additional regularization techniques

3. **Memory Management** (8 points)
   - Batch size determination given memory constraints
   - Parallelism strategy across 8 GPUs
   - Mixed precision configuration
   - Gradient checkpointing decision with tradeoff analysis

4. **Training Stability Measures** (8 points)
   - Normalization strategy (type and placement)
   - Gradient clipping configuration
   - Monitoring metrics for detecting training instabilities
   - Recovery strategies if instabilities are detected

**Evaluation Criteria:**
- Technical accuracy of recommendations
- Justification connecting choices to underlying principles
- Coherent integration of components into unified training system
- Awareness of tradeoffs and alternative approaches

**Word Limit:** 600-800 words

---

## Answer Key

### Question 1: Gradient Flow Problems

**Correct Answer: B**

**Explanation:**
- **Option B is correct** because:
  - The cause is accurate: when gradients are multiplied through many layers and each gradient factor is less than 1, the product shrinks exponentially (e.g., 0.5^50 approaches zero)
  - ReLU directly addresses this by providing gradient of exactly 1 for positive inputs, preventing the multiplication of values less than 1

- **Option A is incorrect** because gradient clipping addresses exploding gradients (values > 1 compounding), not vanishing gradients

- **Option C is incorrect** because high learning rate causes instability and overshooting, not vanishing gradients; batch normalization helps but isn't a direct solution to the gradient multiplication problem

- **Option D is incorrect** because loss function choice doesn't cause vanishing gradients in hidden layers, and dropout is for regularization, not gradient flow

**Understanding Gap:** If you selected A, review the distinction between vanishing (gradients → 0) and exploding (gradients → ∞) gradient problems and their respective solutions.

---

### Question 2: Optimizer Selection

**Correct Answer: C**

**Explanation:**
- **Option C (AdamW)** satisfies all requirements:
  - **Decoupled weight decay:** AdamW applies weight decay directly to parameters (θ = θ - λθ), separate from gradient updates
  - **Adaptive learning rates:** Inherits Adam's per-parameter learning rate adaptation via moment estimates
  - **Mixed precision compatibility:** Standard choice for large-scale training with established mixed-precision implementations

- **Option A (SGD with L2)** lacks adaptive learning rates and couples regularization with gradients

- **Option B (Adam with loss-based weight decay)** couples weight decay with gradients, leading to incorrect regularization strength for parameters with large gradient magnitudes

- **Option D (RMSprop)** lacks the first moment estimate (momentum) of Adam and isn't the standard choice for large language models

**Understanding Gap:** If you selected B, review the critical distinction between L2 regularization (added to loss, coupled with gradients) and decoupled weight decay (applied directly to parameters).

---

### Question 3: Normalization Architecture

**Model Answer:**

**a) LayerNorm vs BatchNorm for Transformers (5 points)**

LayerNorm is preferred for Transformers for three key reasons:

1. **Batch independence:** LayerNorm normalizes across the feature dimension within each sample, so it doesn't require batch statistics. This is critical for:
   - Variable sequence lengths in the same batch
   - Autoregressive generation where batch size is often 1

2. **Sequence position consistency:** BatchNorm would normalize across batch and potentially across sequence positions, treating positions differently based on batch composition

3. **Training-inference consistency:** LayerNorm behaves identically during training and inference, avoiding the running statistics issues of BatchNorm

**b) Pre-LN vs Post-LN Gradient Flow (5 points)**

**Post-LN gradient flow:**
- Gradients must flow through LayerNorm after each sublayer
- LayerNorm includes division by standard deviation, which can attenuate gradients
- Deep networks accumulate these attenuations, requiring careful initialization and warmup

**Pre-LN gradient flow:**
- Residual connection provides direct additive path: `y = x + Sublayer(LayerNorm(x))`
- Gradient highway: ∂y/∂x = 1 + ∂Sublayer/∂x, ensuring gradient ≥ 1
- The "+1" term ensures gradients never vanish regardless of sublayer gradient

**Why Pre-LN removes warmup requirement:** The direct gradient path means early training doesn't suffer from vanishing gradients even with randomly initialized, poorly conditioned weights.

**c) Pre-LN Formulation (5 points)**

Post-LN: `y = LayerNorm(x + Sublayer(x))`
Pre-LN: `y = x + Sublayer(LayerNorm(x))`

**Key difference:** In Pre-LN, the identity path (x → y) is completely unobstructed. The gradient ∂y/∂x always contains a "+1" term from the skip connection, creating a gradient highway that allows signal to flow directly from output to input regardless of how the sublayer gradients behave.

---

### Question 4: Learning Rate Scheduling

**Model Answer:**

**a) Complete LR Schedule Design (7 points)**

```
Schedule Design for 110M BERT, 1M steps, peak LR = 1e-4:

1. Warmup Phase: Steps 0 → 10,000 (1% of training)
   - Strategy: Linear warmup from 0 to 1e-4
   - Justification: Allows Adam's moment estimates to stabilize
     before applying full learning rate

2. Peak Phase: None (immediate transition to decay)
   - Justification: For pre-training, continuous decay is preferred
     as the model continuously improves and benefits from
     gradually reduced step sizes

3. Decay Phase: Steps 10,000 → 1,000,000
   - Strategy: Cosine decay from 1e-4 to 1e-5 (10x reduction)
   - Justification: Smooth decay avoids discontinuities;
     final LR of 1e-5 allows fine convergence
```

**b) Consequences of Removing Warmup (4 points)**

Without warmup, starting at full learning rate would likely cause:
- **Divergence or instability** in early training
- **Root cause:** Adam's second moment (v) is initialized to 0. The effective learning rate is η/√(v + ε). With v ≈ 0 initially, the effective rate becomes η/√ε, which is enormous
- After sufficient steps, v accumulates gradient statistics and provides appropriate scaling
- Warmup allows v to populate while keeping actual parameter updates small

**c) Addressing Late-Stage Validation Increase (4 points)**

If validation loss increases at step 800K while training loss continues decreasing:
- **Diagnosis:** Overfitting beginning
- **Schedule modification:** Implement early stopping, reverting to checkpoint around step 750-800K
- **Alternative:** If continuing training is necessary, more aggressive LR decay could help—switch to faster cosine schedule targeting a lower final LR
- **Root cause consideration:** May also need to increase weight decay or add dropout, but schedule modification is the direct response

---

### Question 5: Training Pipeline Synthesis

**Rubric (30 points total):**

| Component | Excellent (Full Points) | Adequate (Half Points) | Insufficient (Minimal Points) |
|-----------|------------------------|------------------------|-------------------------------|
| Optimization (7 pts) | AdamW with justified hyperparameters, complete LR schedule with warmup/decay, gradient clipping rationale | Correct optimizer choice with incomplete justification, reasonable schedule | Wrong optimizer or missing schedule components |
| Regularization (7 pts) | Weight decay with value and justification, informed dropout decision, additional techniques | Mentions weight decay and dropout without full integration | Missing major regularization considerations |
| Memory Management (8 pts) | Batch size calculation, correct parallelism for 8 GPUs, mixed precision details, checkpointing tradeoff | Addresses most components but lacks depth | Missing critical memory strategies |
| Training Stability (8 pts) | Pre-LN justification, gradient clipping value, specific monitoring metrics, recovery procedures | Mentions stability techniques without integration | Incomplete stability considerations |

**Model Answer:**

**1. Optimization Configuration**

For a 1.3B parameter Transformer, I recommend **AdamW** with:
- β₁ = 0.9, β₂ = 0.95 (slightly lower β₂ for large models per recent findings)
- ε = 1e-8
- Decoupled weight decay λ = 0.1

**Learning rate schedule:**
- Peak LR: 6e-4 (following approximate scaling: larger models can use larger LR)
- Warmup: Linear warmup over 2,000 steps to allow moment estimate stabilization
- Decay: Cosine decay to 6e-5 over remaining training

**Gradient handling:** Global gradient clipping at norm 1.0 to prevent exploding gradients from occasional bad batches without overly constraining normal updates.

**2. Regularization Strategy**

**Weight decay (λ = 0.1):** Applied to all weights except biases, LayerNorm parameters, and embedding tables. This prevents unbounded weight growth while allowing these critical parameters to optimize freely.

**Dropout decision:** For pre-training on 300B tokens, I recommend **minimal or no dropout** (0.0-0.1). With this data scale, the model is unlikely to overfit, and dropout would slow convergence. The dataset size provides implicit regularization.

**Additional techniques:**
- Data shuffling across epochs with different random seeds
- Token-level regularization through the language modeling objective itself

**3. Memory Management**

**Batch size calculation:**
- 1.3B parameters ≈ 5.2GB in FP32, 2.6GB in FP16
- With activations for 2048 token sequences: ~8-10GB per GPU in mixed precision
- Target micro-batch: 4-8 sequences per GPU
- Global batch size: 32-64 sequences × 8 GPUs = 256-512 sequences
- Use gradient accumulation to reach larger effective batches (e.g., 2048 sequences)

**Parallelism strategy:** Data parallelism across 8 GPUs is sufficient for 1.3B. Each GPU holds the full model and processes different data. Use DistributedDataParallel with NCCL backend for efficient gradient synchronization.

**Mixed precision:** Enable BF16 compute with FP32 master weights. BF16 preferred over FP16 for its larger dynamic range, avoiding the need for loss scaling. Maintains FP32 for optimizer states and master weight copy.

**Gradient checkpointing:** Enable for memory savings. With 24+ layers, checkpoint every 2nd layer. Tradeoff: ~30% compute overhead but enables larger batch sizes or sequences, often a net positive for throughput.

**4. Training Stability Measures**

**Normalization:** Pre-LN (LayerNorm before each sublayer) is essential. This creates gradient highways through residual connections, enabling stable training without extensive warmup and supporting the deep architecture.

**Gradient clipping:** Global norm clipping at 1.0 prevents outlier updates while allowing normal gradient magnitudes to pass unaffected.

**Monitoring metrics:**
- Loss curves (train and validation) with smoothing
- Gradient norm statistics (mean, max, % clipped)
- Learning rate confirmation (verify schedule is applied)
- Parameter update ratios (updates / weights ≈ 1e-3 is healthy)
- Activation statistics per layer (detect dead neurons or explosion)

**Recovery strategies:**
- If gradient norms spike: Reduce LR by 50%, continue from checkpoint
- If loss diverges: Revert to previous checkpoint, reduce LR
- If loss plateaus: Verify data pipeline, check for learning rate decay errors
- Maintain checkpoints every 1000 steps with last 5 retained

This integrated pipeline balances throughput, stability, and generalization for efficient large-scale training.

---

## Performance Interpretation Guide

| Score | Mastery Level | Recommended Action |
|-------|---------------|-------------------|
| 90-100% | **Mastery** | Ready for advanced topics (distributed training, architecture research) |
| 75-89% | **Proficient** | Review specific gaps indicated by missed questions |
| 60-74% | **Developing** | Re-study optimization and normalization sections |
| Below 60% | **Foundational** | Complete re-review of Lesson 5, focus on gradient flow concepts |

---

## Review Recommendations by Question

| If You Struggled With | Review These Sections |
|----------------------|----------------------|
| Question 1 | Lesson 5: Gradient Flow, Vanishing/Exploding Gradients |
| Question 2 | Lesson 5: Optimization, AdamW, Weight Decay |
| Question 3 | Lesson 5: Normalization, Pre-LN vs Post-LN |
| Question 4 | Lesson 5: Learning Rate Scheduling, Warmup |
| Question 5 | Entire Lesson 5, particularly Training at Scale |

---

*Generated from Lesson 5: Deep Learning | Quiz Skill*
