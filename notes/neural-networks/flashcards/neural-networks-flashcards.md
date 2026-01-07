# Flashcard Set: Neural Networks

**Source:** notes/neural-networks/neural-networks-study-notes.md
**Concept Map Reference:** notes/neural-networks/concept-maps/neural-networks-concept-map.md
**Date Generated:** 2026-01-06
**Total Cards:** 5
**Distribution:** 2 Easy | 2 Medium | 1 Hard

---

## Critical Knowledge Summary

Concepts appearing in multiple cards (prioritize for review):
- **Backpropagation**: Appears in Cards 2, 5 (central training algorithm)
- **Activation Functions**: Appears in Cards 1, 2, 5 (enables learning)
- **Gradient Descent**: Appears in Cards 2, 5 (optimization method)
- **Architecture Selection**: Appears in Cards 3, 4, 5 (application matching)

---

## Flashcards

---
### Card 1 | Easy
**Cognitive Level:** Remember
**Concept:** Neuron Components and Activation Functions
**Source Section:** Core Concepts 1, 3
**Concept Map Centrality:** Neuron (4), Activation (5)

**FRONT (Question):**
What are the three core components of an artificial neuron, and why are activation functions essential for neural network learning?

**BACK (Answer):**
**Three Core Components:**
1. **Weights:** Learnable parameters determining input importance
2. **Bias:** Learnable threshold shift for activation
3. **Activation Function:** Non-linear transformation of weighted sum

**Neuron Computation:** `output = activation(Σ(weight_i × input_i) + bias)`

**Why Activation Functions Are Essential:**
Without activation functions, any deep network collapses to a single linear transformation:
- `Layer1: W1 × input`
- `Layer2: W2 × (W1 × input) = (W2 × W1) × input = W_combined × input`

No matter how many layers, the result is just one linear function. Activation functions introduce **non-linearity**, enabling networks to learn complex, non-linear mappings.

**Common Activations:**
- **ReLU:** `max(0, x)` — Default for hidden layers
- **Sigmoid:** `1/(1+e^(-x))` — Binary classification output
- **Softmax:** Multi-class probability distribution

**Critical Knowledge Flag:** Yes - Foundation for understanding why depth matters

---

---
### Card 2 | Easy
**Cognitive Level:** Understand
**Concept:** Training Loop: Forward Prop → Loss → Backprop → Gradient Descent
**Source Section:** Core Concepts 4, 5, 6, 7
**Concept Map Centrality:** Backpropagation (8 - Central), Gradient Descent (6)

**FRONT (Question):**
Describe the four-step neural network training loop. What does each step accomplish, and how do they connect?

**BACK (Answer):**
**The Training Loop:**

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│    ┌──────────────┐      ┌──────────────┐              │
│    │   Forward    │ ───► │    Loss      │              │
│    │ Propagation  │      │  Computation │              │
│    └──────────────┘      └──────┬───────┘              │
│           ▲                      │                      │
│           │                      ▼                      │
│    ┌──────┴───────┐      ┌──────────────┐              │
│    │   Weight     │ ◄─── │   Backprop   │              │
│    │   Update     │      │  (Gradients) │              │
│    └──────────────┘      └──────────────┘              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

| Step | What It Does | Output |
|------|--------------|--------|
| **1. Forward Propagation** | Pass input through layers, compute prediction | Predicted output |
| **2. Loss Computation** | Compare prediction to actual label | Scalar error value |
| **3. Backpropagation** | Compute gradient of loss w.r.t. each weight using chain rule | Gradients for all weights |
| **4. Gradient Descent** | Update weights: `w = w - learning_rate × gradient` | New weights |

**Connection:** Each step feeds the next. Repeat for many batches/epochs until loss converges.

**Critical Knowledge Flag:** Yes - Core training algorithm understanding

---

---
### Card 3 | Medium
**Cognitive Level:** Apply
**Concept:** CNN Architecture and Components
**Source Section:** Core Concepts 8
**Concept Map Centrality:** CNN (5 - High)

**FRONT (Question):**
You need to build an image classifier for 224×224 RGB images with 10 classes. Design a CNN architecture specifying: (1) the purpose of convolution layers, (2) the purpose of pooling layers, (3) why CNNs are preferred over dense networks for images, and (4) a sample layer sequence.

**BACK (Answer):**
**1. Convolution Layers:**
- Slide learnable filters across input
- Detect local features (edges → textures → shapes → objects)
- Parameter sharing: same filter weights used across all positions
- Output: feature maps highlighting detected patterns

**2. Pooling Layers:**
- Downsample spatial dimensions (e.g., 2×2 max pooling halves dimensions)
- Reduces computation and parameters
- Provides translation invariance (slight shifts don't change output)
- Common: Max pooling (takes maximum in window)

**3. Why CNN > Dense for Images:**
| Aspect | Dense Network | CNN |
|--------|---------------|-----|
| Parameters | 224×224×3 = 150K inputs × neurons = millions | Filters shared across positions = thousands |
| Spatial awareness | None; flattened pixels | Preserves spatial relationships |
| Translation invariance | None; must relearn shifted patterns | Built-in via convolution |

**4. Sample Architecture:**
```
Input: 224×224×3
Conv2D(32 filters, 3×3) + ReLU → 222×222×32
MaxPool(2×2) → 111×111×32
Conv2D(64 filters, 3×3) + ReLU → 109×109×64
MaxPool(2×2) → 54×54×64
Conv2D(128 filters, 3×3) + ReLU → 52×52×128
MaxPool(2×2) → 26×26×128
Flatten → 86,528
Dense(256) + ReLU
Dense(10) + Softmax → 10 class probabilities
```

**Critical Knowledge Flag:** Yes - Primary architecture for image tasks

---

---
### Card 4 | Medium
**Cognitive Level:** Analyze
**Concept:** Architecture Selection: RNN vs Transformer
**Source Section:** Core Concepts 9, 10
**Concept Map Centrality:** RNN (4), Transformer (4)

**FRONT (Question):**
Compare RNNs and Transformers for sequence processing. Analyze: (1) how each handles sequential dependencies, (2) the vanishing gradient problem, (3) parallelization capability, and (4) when to choose each architecture.

**BACK (Answer):**
**1. Sequential Dependency Handling:**

| Mechanism | RNN | Transformer |
|-----------|-----|-------------|
| **Approach** | Sequential: hidden state passed step-by-step | Parallel: attention connects all positions |
| **Memory** | Hidden state accumulates history | Attention scores computed for all pairs |
| **Long-range** | Information decays over distance | Direct connections regardless of distance |

**2. Vanishing Gradient Problem:**
- **RNN:** Gradients multiply through time steps; long sequences → gradients vanish/explode
- **LSTM/GRU:** Gating mechanisms partially solve; still struggles with very long sequences
- **Transformer:** No sequential gradient path; attention gradients flow directly → no vanishing

**3. Parallelization:**
- **RNN:** Sequential by design; step t requires step t-1; slow training
- **Transformer:** All positions processed simultaneously; massive parallelism; fast on GPUs

**4. When to Choose:**

| Choose RNN/LSTM When | Choose Transformer When |
|----------------------|-------------------------|
| Limited compute/memory | Sufficient GPU resources |
| Very long sequences + streaming | Moderate sequence length (< 4096 typically) |
| Real-time processing required | Training speed is priority |
| Simpler implementation needed | State-of-the-art performance required |

**Modern Default:** Transformers dominate NLP and increasingly vision. RNNs still used for specific real-time/streaming applications.

**Critical Knowledge Flag:** Yes - Critical architecture decision for sequential data

---

---
### Card 5 | Hard
**Cognitive Level:** Synthesize
**Concept:** Complete Neural Network Training Pipeline
**Source Section:** All Core Concepts
**Concept Map Centrality:** Integrates all high-centrality nodes

**FRONT (Question):**
Synthesize a complete neural network training pipeline for a sentiment classification task (positive/negative movie reviews). Address: (1) input preprocessing, (2) architecture selection with justification, (3) loss function and activation choices, (4) training process including backpropagation, (5) hyperparameter considerations, and (6) evaluation strategy to detect overfitting.

**BACK (Answer):**
**1. Input Preprocessing:**
- **Tokenization:** Split text into tokens (words or subwords)
- **Vocabulary:** Map tokens to integer IDs
- **Padding/Truncation:** Fixed sequence length (e.g., 256 tokens)
- **Embedding:** Convert IDs to dense vectors (learned or pre-trained like GloVe)

**2. Architecture Selection:**
**Choice: Transformer-based (e.g., fine-tune BERT) or LSTM**

| Option | Justification |
|--------|---------------|
| **Transformer (BERT)** | State-of-the-art for NLP; pre-trained knowledge; handles long-range context via attention |
| **LSTM** | Simpler; less compute; still effective for sentiment where local patterns matter |

*Recommended:* Start with pre-trained Transformer for best results; LSTM if resources limited.

**3. Loss Function and Activations:**
- **Output Activation:** Sigmoid (binary classification: P(positive))
- **Loss Function:** Binary Cross-Entropy: `-[y·log(p) + (1-y)·log(1-p)]`
- **Hidden Activations:** ReLU (dense layers), GELU (Transformer standard)

**4. Training Process:**
```
For each epoch:
    For each batch:
        1. Forward prop: input → embeddings → model → sigmoid → prediction
        2. Compute loss: BCE(prediction, label)
        3. Backpropagation: ∂Loss/∂weights via chain rule
        4. Gradient descent: weights -= learning_rate × gradients
    Evaluate on validation set
    Save if best validation performance
```

**5. Hyperparameters:**
| Hyperparameter | Typical Value | Consideration |
|----------------|---------------|---------------|
| Learning rate | 2e-5 (BERT), 1e-3 (LSTM) | Too high → divergence; too low → slow |
| Batch size | 16-32 | Limited by GPU memory |
| Epochs | 3-5 (BERT), 10-20 (LSTM) | Early stopping on validation |
| Dropout | 0.1-0.3 | Regularization against overfitting |

**6. Overfitting Detection & Mitigation:**

| Signal | Diagnosis | Mitigation |
|--------|-----------|------------|
| Train acc ↑, Val acc flat | Overfitting starting | Increase dropout, early stop |
| Train loss ↓↓, Val loss ↑ | Clear overfitting | Reduce model size, add regularization |
| Train ≈ Val, both poor | Underfitting | Increase capacity, more epochs |

**Monitoring:**
- Plot train/val loss curves
- Use early stopping (stop when val loss stops improving)
- Keep test set completely separate for final evaluation

**Critical Knowledge Flag:** Yes - Integrates all core neural network concepts

---

---

## Export Formats

### Anki-Compatible (Tab-Separated)
```
What are the 3 neuron components and why are activations essential?	Weights (input importance), Bias (threshold shift), Activation (non-linearity). Without activations, deep networks collapse to single linear function.	easy::neuron::nn
Describe the 4-step training loop	1. Forward prop (predict), 2. Loss (measure error), 3. Backprop (compute gradients via chain rule), 4. Gradient descent (update weights). Repeat.	easy::training::nn
Design CNN for image classification	Convolution (detect features), Pooling (downsample), parameter sharing. Conv→Pool→Conv→Pool→Flatten→Dense→Softmax.	medium::cnn::nn
Compare RNN vs Transformer	RNN: sequential, vanishing gradients, slow. Transformer: parallel, attention connects all, fast. Modern default: Transformer.	medium::architecture::nn
Synthesize sentiment classification pipeline	Preprocess→Tokenize→Embed→Model(BERT/LSTM)→Sigmoid→BCE Loss→Backprop→Adam. Monitor train/val curves for overfitting.	hard::pipeline::nn
```

### CSV Format
```csv
"Front","Back","Difficulty","Concept","Centrality"
"Neuron components and activation necessity?","Weights, Bias, Activation. Activations enable non-linear learning.","Easy","Neuron/Activation","High"
"4-step training loop?","Forward prop → Loss → Backprop → Gradient descent","Easy","Training","Critical"
"CNN design for images?","Conv (features) + Pool (downsample) + Dense. Parameter sharing.","Medium","CNN","High"
"RNN vs Transformer?","RNN: sequential, gradients vanish. Transformer: parallel, attention.","Medium","Architectures","Medium"
"Sentiment classification pipeline?","Tokenize→Embed→Model→Sigmoid→BCE→Backprop→Adam. Early stopping.","Hard","Full Pipeline","Integration"
```

---

## Source Mapping

| Card | Source Sections | Concept Map Nodes | Key Terms |
|------|-----------------|-------------------|-----------|
| 1 | Concepts 1, 3 | Neuron, Activation | Weights, bias, ReLU, sigmoid |
| 2 | Concepts 4, 5, 6, 7 | Forward Prop, Loss, Backprop, GD | Chain rule, gradient, learning rate |
| 3 | Concept 8 | CNN, Convolution, Pooling | Feature maps, parameter sharing |
| 4 | Concepts 9, 10 | RNN, Transformer, Attention | Hidden state, vanishing gradient |
| 5 | All concepts | All high-centrality | Full training pipeline |
