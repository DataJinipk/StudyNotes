# Practice Problems: Neural Networks

**Source:** notes/neural-networks/neural-networks-study-notes.md
**Concept Map Reference:** notes/neural-networks/concept-maps/neural-networks-concept-map.md
**Date Generated:** 2026-01-06
**Total Problems:** 5
**Estimated Total Time:** 75-90 minutes
**Distribution:** 1 Warm-Up | 2 Skill-Builder | 1 Challenge | 1 Debug/Fix

---

## Overview

### Concepts Practiced
| Concept | Problems | Mastery Indicator |
|---------|----------|-------------------|
| Forward Propagation | P1, P2 | Can compute outputs manually |
| Backpropagation | P2 | Can derive gradients using chain rule |
| Activation Functions | P3 | Can select appropriate activation for task |
| CNN Architecture | P4 | Can design network for image task |
| Training Diagnosis | P5 | Can identify and fix training issues |

### Recommended Approach
1. Attempt each problem before looking at hints
2. Use hints progressively—don't skip to solution
3. After solving, read solution to compare approaches
4. Review Common Mistakes even if you solved correctly
5. Attempt Extension Challenges for deeper mastery

### Self-Assessment Guide
| Problems Solved (no hints) | Mastery Level | Recommendation |
|---------------------------|---------------|----------------|
| 5/5 | Expert | Ready for implementation projects |
| 4/5 | Proficient | Review one gap area |
| 3/5 | Developing | More practice recommended |
| 2/5 or below | Foundational | Re-review study notes first |

---

## Problems

---

## Problem 1: Manual Forward Propagation

**Type:** Warm-Up
**Concepts Practiced:** Forward Propagation, Neuron Computation
**Estimated Time:** 10 minutes
**Prerequisites:** Understanding of neuron formula

### Problem Statement

Given a simple neural network with:
- **Input:** `x = [2, 3]` (2 features)
- **Hidden layer:** 2 neurons
- **Output layer:** 1 neuron (regression)
- **Activation:** ReLU for hidden, Linear for output

Weights and biases:
```
Hidden Layer:
  Neuron 1: w = [0.5, -0.3], b = 0.1
  Neuron 2: w = [-0.2, 0.4], b = -0.1

Output Layer:
  Neuron 1: w = [0.6, 0.8], b = 0.2
```

### Requirements

- [ ] Compute the hidden layer outputs (after ReLU activation)
- [ ] Compute the final network output
- [ ] Show all intermediate calculations

### Hints (Progressive)

<details>
<summary>Hint 1: Getting Started</summary>

For each neuron, compute: `z = Σ(w_i × x_i) + b`, then apply activation.

Start with Hidden Neuron 1: `z = 0.5×2 + (-0.3)×3 + 0.1`

</details>

<details>
<summary>Hint 2: Key Insight</summary>

ReLU activation: `output = max(0, z)`

If z is negative, output becomes 0. If z is positive, output equals z.

</details>

<details>
<summary>Hint 3: Nearly There</summary>

After computing hidden outputs h1 and h2, the output layer computes:
`output = 0.6×h1 + 0.8×h2 + 0.2`

(No activation for linear output in regression)

</details>

### Solution

<details>
<summary>Click to reveal full solution</summary>

**Approach:**
Compute layer-by-layer, applying weights, biases, and activations in sequence.

**Step-by-Step Solution:**

**Hidden Layer:**

*Neuron 1:*
```
z1 = (0.5 × 2) + (-0.3 × 3) + 0.1
z1 = 1.0 - 0.9 + 0.1 = 0.2
h1 = ReLU(0.2) = max(0, 0.2) = 0.2
```

*Neuron 2:*
```
z2 = (-0.2 × 2) + (0.4 × 3) + (-0.1)
z2 = -0.4 + 1.2 - 0.1 = 0.7
h2 = ReLU(0.7) = max(0, 0.7) = 0.7
```

**Hidden layer output:** `h = [0.2, 0.7]`

**Output Layer:**
```
output = (0.6 × 0.2) + (0.8 × 0.7) + 0.2
output = 0.12 + 0.56 + 0.2 = 0.88
```

**Final Answer:** `output = 0.88`

**Why This Works:**
Forward propagation is simply applying the neuron formula layer-by-layer. The hidden layer transforms the 2D input into a different 2D representation, and the output layer combines this into a single prediction.

</details>

### Common Mistakes

- ❌ **Mistake:** Forgetting to apply ReLU, just using raw z values
  - **Why it happens:** Activation step is easy to skip mentally
  - **How to avoid:** Always write "apply activation" as explicit step

- ❌ **Mistake:** Applying ReLU to output layer in regression
  - **Why it happens:** Applying same activation everywhere
  - **How to avoid:** Regression outputs are unbounded; use linear (no activation)

### Extension Challenge

What if Hidden Neuron 1 had weights `w = [-0.5, -0.3]` (both negative)? How would ReLU affect the network's ability to use this neuron's information?

---

---

## Problem 2: Backpropagation Gradient Computation

**Type:** Skill-Builder
**Concepts Practiced:** Backpropagation, Chain Rule, Gradient Descent
**Estimated Time:** 20 minutes
**Prerequisites:** Calculus (chain rule), Forward propagation

### Problem Statement

Using the same network from Problem 1, assume:
- **True label:** `y = 1.0`
- **Network prediction:** `ŷ = 0.88` (from Problem 1)
- **Loss function:** MSE = `(y - ŷ)²`
- **Learning rate:** `η = 0.1`

Compute the gradient of the loss with respect to the output layer weights (`w_out = [0.6, 0.8]`) and update them using gradient descent.

### Requirements

- [ ] Compute the loss value
- [ ] Compute ∂Loss/∂w_out using chain rule
- [ ] Update w_out using gradient descent
- [ ] Show chain rule decomposition

### Hints (Progressive)

<details>
<summary>Hint 1: Getting Started</summary>

First compute the loss: `L = (1.0 - 0.88)² = ?`

Then use chain rule: `∂L/∂w = ∂L/∂ŷ × ∂ŷ/∂w`

</details>

<details>
<summary>Hint 2: Key Insight</summary>

For MSE loss `L = (y - ŷ)²`:
- `∂L/∂ŷ = -2(y - ŷ) = -2(1.0 - 0.88) = -0.24`

For output `ŷ = w1×h1 + w2×h2 + b`:
- `∂ŷ/∂w1 = h1 = 0.2`
- `∂ŷ/∂w2 = h2 = 0.7`

</details>

<details>
<summary>Hint 3: Nearly There</summary>

Combine using chain rule:
- `∂L/∂w1 = ∂L/∂ŷ × ∂ŷ/∂w1 = -0.24 × 0.2`

Update rule: `w_new = w_old - η × gradient`

</details>

### Solution

<details>
<summary>Click to reveal full solution</summary>

**Approach:**
Apply chain rule to decompose gradient computation, then use gradient descent update rule.

**Step-by-Step Solution:**

**1. Compute Loss:**
```
L = (y - ŷ)² = (1.0 - 0.88)² = (0.12)² = 0.0144
```

**2. Chain Rule Decomposition:**

```
∂L/∂w_out = ∂L/∂ŷ × ∂ŷ/∂w_out
```

**3. Compute ∂L/∂ŷ:**
```
L = (y - ŷ)²
∂L/∂ŷ = 2(y - ŷ) × (-1) = -2(y - ŷ) = -2(1.0 - 0.88) = -2(0.12) = -0.24
```

**4. Compute ∂ŷ/∂w:**

Since `ŷ = w1×h1 + w2×h2 + b` where h1=0.2, h2=0.7:
```
∂ŷ/∂w1 = h1 = 0.2
∂ŷ/∂w2 = h2 = 0.7
```

**5. Complete Gradients:**
```
∂L/∂w1 = -0.24 × 0.2 = -0.048
∂L/∂w2 = -0.24 × 0.7 = -0.168
```

**6. Gradient Descent Update:**
```
w1_new = 0.6 - 0.1 × (-0.048) = 0.6 + 0.0048 = 0.6048
w2_new = 0.8 - 0.1 × (-0.168) = 0.8 + 0.0168 = 0.8168
```

**Final Answer:** `w_out = [0.6048, 0.8168]`

**Why This Works:**
The gradient is negative because our prediction (0.88) is less than the target (1.0). Subtracting a negative gradient increases the weights, which will increase the output toward the target. This is gradient descent moving toward lower loss.

</details>

### Common Mistakes

- ❌ **Mistake:** Wrong sign on gradient (∂L/∂ŷ = +2(y-ŷ) instead of -2(y-ŷ))
  - **Why it happens:** Forgetting the chain rule includes ∂ŷ/∂ŷ = -1 in (y-ŷ)²
  - **How to avoid:** Carefully differentiate: d/dŷ[(y-ŷ)²] = 2(y-ŷ)×(-1)

- ❌ **Mistake:** Using input x instead of hidden output h for ∂ŷ/∂w
  - **Why it happens:** Confusing which values feed into output layer
  - **How to avoid:** Draw network; trace what each weight multiplies

### Extension Challenge

Compute the gradients for the hidden layer weights. How does ReLU's derivative affect backpropagation? (Hint: ReLU'(z) = 1 if z > 0, else 0)

---

---

## Problem 3: Activation Function Selection

**Type:** Skill-Builder
**Concepts Practiced:** Activation Functions, Output Layer Design
**Estimated Time:** 15 minutes
**Prerequisites:** Understanding of different activation functions

### Problem Statement

For each of the following tasks, select the appropriate output layer activation function and loss function. Justify each choice.

| Task | Output Layer Activation | Loss Function | Justification |
|------|------------------------|---------------|---------------|
| A. Predict house prices ($10K-$1M) | ? | ? | ? |
| B. Classify emails as spam/not-spam | ? | ? | ? |
| C. Classify images into 10 categories | ? | ? | ? |
| D. Predict probability of multiple tags (multi-label) | ? | ? | ? |

### Requirements

- [ ] Select activation for each task
- [ ] Select loss function for each task
- [ ] Provide clear justification connecting task requirements to function properties

### Hints (Progressive)

<details>
<summary>Hint 1: Getting Started</summary>

Ask: What range should the output have?
- Unbounded continuous → Linear (no activation)
- Single probability (0-1) → Sigmoid
- Probability distribution (sum to 1) → Softmax

</details>

<details>
<summary>Hint 2: Key Insight</summary>

Loss functions should match output type:
- Continuous targets → MSE or MAE
- Binary classification → Binary Cross-Entropy
- Multi-class (one label) → Categorical Cross-Entropy

</details>

<details>
<summary>Hint 3: Nearly There</summary>

Multi-label (D) is tricky: each tag is independent binary classification.
Think of it as multiple sigmoid outputs, not softmax.

</details>

### Solution

<details>
<summary>Click to reveal full solution</summary>

**Approach:**
Match output requirements to activation properties, then select compatible loss.

| Task | Activation | Loss | Justification |
|------|------------|------|---------------|
| **A. House prices** | **Linear (none)** | **MSE** | Prices are continuous, unbounded positive values. Linear allows any output; MSE penalizes large errors for regression. |
| **B. Spam detection** | **Sigmoid** | **Binary Cross-Entropy** | Binary classification needs probability output (0-1). Sigmoid squashes to this range; BCE measures probability divergence. |
| **C. 10-class image** | **Softmax** | **Categorical Cross-Entropy** | Mutually exclusive classes need probability distribution summing to 1. Softmax ensures this; CCE handles one-hot targets. |
| **D. Multi-label tags** | **Sigmoid (per output)** | **Binary Cross-Entropy (per output)** | Tags are independent; image can have multiple. Each output is independent binary decision, not competing distribution. |

**Key Insight for D:**
- **Softmax** = "Pick ONE from these options" (probabilities sum to 1)
- **Multiple Sigmoids** = "Decide YES/NO for EACH option independently"

**Conceptual Connection:**
Activation functions constrain output range to match what the task requires. Loss functions then measure error in a way appropriate for that output type.

</details>

### Common Mistakes

- ❌ **Mistake:** Using Softmax for multi-label classification
  - **Why it happens:** Both involve multiple outputs; easy to confuse
  - **How to avoid:** Ask: "Can multiple labels be true simultaneously?" If yes → independent sigmoids

- ❌ **Mistake:** Using Sigmoid for multi-class single-label
  - **Why it happens:** Sigmoid gives probabilities, seems sufficient
  - **How to avoid:** Sigmoid outputs don't sum to 1; Softmax ensures proper distribution

### Extension Challenge

What if you needed to predict both house price AND probability of sale within 30 days from the same network? Design a multi-output architecture with appropriate activations.

---

---

## Problem 4: CNN Architecture Design

**Type:** Challenge
**Concepts Practiced:** CNN Architecture, Convolution, Pooling, Network Design
**Estimated Time:** 20 minutes
**Prerequisites:** Understanding of CNN components

### Problem Statement

Design a CNN architecture for classifying medical X-ray images into 4 diagnostic categories:
- **Input:** 512×512 grayscale images
- **Output:** 4 classes (Normal, Pneumonia, Tuberculosis, COVID-19)
- **Constraints:**
  - Model must have < 5 million parameters
  - Must achieve reasonable spatial reduction before dense layers

Design the architecture specifying:
1. Layer sequence (Conv, Pool, Dense)
2. Number of filters per conv layer
3. Kernel sizes
4. Activation functions
5. Output layer design

Calculate approximate parameter count to verify constraint.

### Requirements

- [ ] Complete layer-by-layer architecture
- [ ] Appropriate spatial dimension reduction
- [ ] Parameter count estimation
- [ ] Justification for design choices

### Hints (Progressive)

<details>
<summary>Hint 1: Getting Started</summary>

Standard CNN pattern:
```
[Conv → ReLU → Pool] × N → Flatten → Dense → Output
```

Start with few filters (32), increase with depth (64, 128...).
512×512 is large; need aggressive pooling.

</details>

<details>
<summary>Hint 2: Key Insight</summary>

Parameter counting:
- Conv layer: `(kernel_h × kernel_w × input_channels + 1) × num_filters`
- Dense layer: `(input_size + 1) × output_size`

To reduce 512×512 to manageable size, use multiple pooling layers or strided convolutions.

</details>

<details>
<summary>Hint 3: Nearly There</summary>

After several pool layers reducing 512→256→128→64→32→16, flattening 16×16×128 = 32,768.
Dense layer 32,768 → 256 alone = 8M parameters (too many!).

Add another pool or use Global Average Pooling to reduce further.

</details>

### Solution

<details>
<summary>Click to reveal full solution</summary>

**Approach:**
Design progressively deeper feature extraction with aggressive spatial reduction, ending with Global Average Pooling to minimize dense layer parameters.

**Architecture:**

```
Layer                          Output Shape      Parameters
──────────────────────────────────────────────────────────────
Input                          512×512×1         0
Conv2D(32, 3×3, ReLU)          510×510×32        320
MaxPool2D(2×2)                 255×255×32        0
Conv2D(64, 3×3, ReLU)          253×253×64        18,496
MaxPool2D(2×2)                 126×126×64        0
Conv2D(128, 3×3, ReLU)         124×124×128       73,856
MaxPool2D(2×2)                 62×62×128         0
Conv2D(128, 3×3, ReLU)         60×60×128         147,584
MaxPool2D(2×2)                 30×30×128         0
Conv2D(256, 3×3, ReLU)         28×28×256         295,168
MaxPool2D(2×2)                 14×14×256         0
Conv2D(256, 3×3, ReLU)         12×12×256         590,080
GlobalAveragePool2D            256               0
Dense(128, ReLU)               128               32,896
Dropout(0.5)                   128               0
Dense(4, Softmax)              4                 516
──────────────────────────────────────────────────────────────
Total Parameters: ~1.16 million ✓ (under 5M constraint)
```

**Design Justifications:**

1. **Filter progression (32→64→128→256):** Standard doubling; more filters capture more complex features at deeper layers.

2. **3×3 kernels:** Standard choice; efficient and effective. Two 3×3 convs have same receptive field as one 5×5 with fewer parameters.

3. **Multiple MaxPool layers:** Reduce 512→255→126→62→30→14→12 before global pooling. Necessary for large input.

4. **Global Average Pooling:** Converts 12×12×256 to 256 (not 12×12×256=36,864). Massive parameter reduction vs. Flatten.

5. **Small dense layer (128):** After GAP, only 256 inputs; small dense layer prevents overfitting on medical images (often limited data).

6. **Dropout (0.5):** Regularization critical for medical imaging where datasets are small.

7. **Softmax output (4):** Mutually exclusive diagnosis categories.

</details>

### Common Mistakes

- ❌ **Mistake:** Flattening too early with large spatial dimensions
  - **Why it happens:** Following simple tutorials that use small inputs
  - **How to avoid:** Calculate flatten size; use Global Average Pooling for large inputs

- ❌ **Mistake:** Too many filters in early layers
  - **Why it happens:** Thinking more = better
  - **How to avoid:** Early layers detect simple features; don't need many filters

### Extension Challenge

Modify the architecture to use residual connections (ResNet-style skip connections). Where would you add them, and how do they help with training deeper networks?

---

---

## Problem 5: Training Diagnosis and Debugging

**Type:** Debug/Fix
**Concepts Practiced:** Overfitting, Training Diagnosis, Hyperparameter Tuning
**Estimated Time:** 15 minutes
**Prerequisites:** Understanding of training dynamics, loss curves

### Problem Statement

You're training a neural network and observe the following training curves:

```
Epoch  Train Loss  Train Acc  Val Loss   Val Acc
─────────────────────────────────────────────────
1      2.45        0.15       2.48       0.14
2      1.82        0.35       1.90       0.33
3      1.24        0.55       1.45       0.48
4      0.78        0.72       1.32       0.52
5      0.45        0.85       1.41       0.51
6      0.22        0.94       1.68       0.49
7      0.09        0.98       2.01       0.47
8      0.03        0.99       2.45       0.45
```

**Tasks:**
1. Diagnose what is happening to the model
2. Identify at which epoch the problem becomes apparent
3. Propose three specific techniques to address the issue
4. Explain how each technique works mechanically

### Requirements

- [ ] Correct diagnosis with evidence from data
- [ ] Identify critical epoch
- [ ] Three specific mitigation techniques
- [ ] Mechanistic explanation for each technique

### Hints (Progressive)

<details>
<summary>Hint 1: Getting Started</summary>

Look at the divergence between train and validation metrics. When do they start moving in opposite directions?

</details>

<details>
<summary>Hint 2: Key Insight</summary>

The pattern: Train loss ↓↓ while Val loss ↑↑ is the classic signature of one specific problem.

The model is learning the training data "too well."

</details>

<details>
<summary>Hint 3: Nearly There</summary>

The problem is overfitting. Mitigation techniques include:
- Regularization (L1/L2, Dropout)
- Early stopping
- Data augmentation
- Reducing model capacity

Explain HOW each prevents memorization.

</details>

### Solution

<details>
<summary>Click to reveal full solution</summary>

**Approach:**
Analyze train vs. validation metrics divergence, identify overfitting, propose mechanistic solutions.

**1. Diagnosis: OVERFITTING**

**Evidence:**
- Training loss continuously decreases (2.45 → 0.03)
- Training accuracy reaches 99%
- Validation loss INCREASES after epoch 3 (1.45 → 2.45)
- Validation accuracy DECREASES after epoch 4 (52% → 45%)

The model is memorizing training examples instead of learning generalizable patterns.

**2. Critical Epoch: Epoch 4**

At epoch 4, validation loss stops improving (1.32) while train loss continues to drop. By epoch 5, validation loss increases (1.41) — clear overfitting signal. **Optimal stopping point would be epoch 3-4.**

**3. Three Mitigation Techniques:**

| Technique | Mechanism | Implementation |
|-----------|-----------|----------------|
| **Early Stopping** | Stop training when validation loss stops improving | Monitor val_loss; stop after N epochs without improvement; restore best weights |
| **Dropout** | Randomly zero neurons during training; prevents co-adaptation | Add Dropout(0.3-0.5) layers; neurons can't rely on specific other neurons |
| **L2 Regularization** | Add penalty term λΣw² to loss; discourages large weights | Add kernel_regularizer=l2(0.01) to layers; simpler functions preferred |

**Mechanistic Explanations:**

**Early Stopping:**
- Training continues only while generalization improves
- Once model starts memorizing (val loss ↑), training stops
- Best model checkpoint is restored
- Like stopping studying when you've learned the concepts but before you start memorizing specific example wordings

**Dropout:**
- Each forward pass randomly drops 30-50% of neurons
- Network can't rely on any single neuron; must distribute knowledge
- Creates implicit ensemble of thinned networks
- Prevents complex co-adaptations that fit noise

**L2 Regularization (Weight Decay):**
- Loss becomes: `L_total = L_data + λΣw²`
- Large weights are penalized; optimizer prefers smaller weights
- Smaller weights → simpler, smoother functions
- Smoother functions are less likely to fit noise

**Additional Technique (Bonus): Data Augmentation**
- Create variations of training images (rotation, flip, zoom)
- More diverse training data prevents memorization
- Model must learn invariant features, not specific pixel patterns

</details>

### Common Mistakes

- ❌ **Mistake:** Diagnosing as underfitting because val accuracy is low
  - **Why it happens:** Focusing only on final validation metric
  - **How to avoid:** Look at TRENDS; underfitting = both metrics poor and improving together

- ❌ **Mistake:** Applying solutions without understanding mechanism
  - **Why it happens:** Treating techniques as "magic"
  - **How to avoid:** Always ask "HOW does this prevent memorization?"

### Extension Challenge

The learning curves also show train accuracy reaching 99% while best validation is only 52%. Besides overfitting, what might this large gap indicate about the dataset itself? (Hint: Consider dataset shift or labeling issues)

---

---

## Summary

### Key Takeaways
1. **Forward propagation** is mechanical: apply weights, add bias, activate—layer by layer
2. **Backpropagation** uses chain rule to attribute error to each weight
3. **Activation selection** must match output requirements (range, distribution)
4. **CNN design** requires balancing depth, parameters, and spatial reduction
5. **Training diagnosis** requires monitoring train/val divergence, not just final metrics

### Concepts by Problem
| Problem | Primary Concepts | Secondary Concepts |
|---------|-----------------|-------------------|
| P1 (Warm-Up) | Forward Propagation | ReLU Activation |
| P2 (Skill-Builder) | Backpropagation, Chain Rule | Gradient Descent |
| P3 (Skill-Builder) | Activation Functions | Loss Functions |
| P4 (Challenge) | CNN Architecture | Parameter Efficiency |
| P5 (Debug/Fix) | Overfitting | Regularization |

### Next Steps
- If struggled with P1-P2: Review mathematical foundations in study notes
- If struggled with P3: Review activation function properties
- If struggled with P4: Practice with architecture visualization tools
- If struggled with P5: Train actual models and observe learning curves
- Ready for assessment: Proceed to quiz skill
