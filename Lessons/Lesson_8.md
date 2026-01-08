
# Lesson 8: Neural Network Architectures

**Topic:** Neural Network Architectures: Design Patterns, Components, and Modern Innovations
**Prerequisites:** Lesson 5 (Deep Learning), Lesson 4 (Transformers)
**Estimated Study Time:** 3-4 hours
**Difficulty:** Advanced

---

## Learning Objectives

Upon completion of this lesson, learners will be able to:

1. **Analyze** the fundamental building blocks of neural networks (layers, activations, normalization, connections)
2. **Compare** major architecture families (MLP, CNN, RNN, Transformer, GNN) and their structural properties
3. **Evaluate** architectural design patterns (encoder-decoder, U-Net, residual, attention) for specific tasks
4. **Design** appropriate network architectures given problem constraints and data characteristics
5. **Apply** modern architectural innovations including mixture-of-experts, state space models, and efficient attention variants

---

## Introduction

Neural network architecture—the structural design of how layers, connections, and operations are organized—is one of the most impactful decisions in deep learning. The same learning algorithm applied to different architectures produces vastly different capabilities: CNNs excel at images, Transformers dominate language, and specialized architectures enable graph reasoning, 3D understanding, and multimodal processing.

This lesson provides a comprehensive survey of neural network architectures, from foundational components to cutting-edge innovations. Understanding these patterns enables practitioners to select appropriate architectures for new problems and contribute to architectural innovation.

---

## Core Concepts

### Concept 1: Foundational Building Blocks

Every neural network architecture is composed of fundamental building blocks that can be combined in various ways.

**Linear Layers (Dense/Fully Connected):**

```
y = Wx + b

Where:
- W: Weight matrix (output_dim × input_dim)
- x: Input vector
- b: Bias vector
- y: Output vector
```

| Property | Value |
|----------|-------|
| Parameters | input_dim × output_dim + output_dim |
| Computation | O(input × output) |
| Inductive bias | None (learns any linear transformation) |

**Activation Functions:**

| Function | Formula | Properties | Use Case |
|----------|---------|------------|----------|
| ReLU | max(0, x) | Sparse, fast, dying neuron risk | Default hidden layers |
| GELU | x·Φ(x) | Smooth, Transformer standard | Transformers, modern nets |
| SiLU/Swish | x·σ(x) | Self-gated, smooth | Modern CNNs, MLPs |
| Softmax | exp(xᵢ)/Σexp(xⱼ) | Probability distribution | Classification output |
| Sigmoid | 1/(1+e⁻ˣ) | Bounded [0,1] | Binary output, gates |
| Tanh | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | Bounded [-1,1], zero-centered | RNN hidden states |

**Normalization Layers:**

| Type | Normalizes Over | Best For |
|------|-----------------|----------|
| BatchNorm | Batch dimension | CNNs with large batches |
| LayerNorm | Feature dimension | Transformers, RNNs |
| GroupNorm | Channel groups | CNNs with small batches |
| RMSNorm | Features (no centering) | Efficient Transformers |
| InstanceNorm | Single instance channels | Style transfer |

**Dropout:**

```
Training:  x_out = x * mask / (1 - p), where mask ~ Bernoulli(1-p)
Inference: x_out = x (no dropout)
```

Regularization technique that randomly zeros activations, preventing co-adaptation.

---

### Concept 2: Feedforward Networks (MLPs)

The Multi-Layer Perceptron (MLP) is the foundational architecture: stacked linear layers with non-linear activations.

**Standard MLP Block:**

```
Input → Linear → Activation → Linear → Activation → ... → Output
         ↓           ↓
      (hidden)    (non-linear)
```

**Modern MLP Block (as in Transformers):**

```
x → Linear(d, 4d) → GELU → Linear(4d, d) → output

Expansion ratio of 4× is standard
```

**MLP-Mixer Architecture:**

Demonstrated that pure MLPs can compete with CNNs/Transformers:

```
Input patches → Token-mixing MLP → Channel-mixing MLP → ... → Output
                (mix across patches)  (mix within patches)
```

**When to Use MLPs:**
- Tabular data (structured features)
- As components within larger architectures
- When inductive biases of CNNs/Transformers aren't needed

---

### Concept 3: Convolutional Neural Networks (CNNs)

CNNs exploit spatial structure through local connectivity and weight sharing.

**Convolution Operation:**

```
Output[i,j] = Σₘ Σₙ Input[i+m, j+n] × Kernel[m, n]

For each spatial position, apply the same kernel
```

**Key CNN Components:**

| Component | Function | Parameters |
|-----------|----------|------------|
| Conv2d | Spatial feature extraction | kernel_size, stride, padding |
| MaxPool | Downsample, translation invariance | pool_size, stride |
| AvgPool | Smooth downsample | pool_size |
| Global AvgPool | Spatial dims → single value | None |

**CNN Design Evolution:**

```
LeNet (1998)        → Simple stack: Conv-Pool-Conv-Pool-FC
AlexNet (2012)      → Deeper, ReLU, dropout, GPU training
VGG (2014)          → Very deep (16-19 layers), 3×3 kernels only
ResNet (2015)       → Skip connections enable 100+ layers
EfficientNet (2019) → Compound scaling of depth/width/resolution
ConvNeXt (2022)     → CNN design modernized with Transformer tricks
```

**Receptive Field:**

The receptive field is the input region that affects a given output position:

```
Receptive Field = 1 + Σ (kernel_size - 1) × stride_product

Deeper networks → larger receptive fields → global context
```

**Modern CNN Block (ConvNeXt style):**

```
x → DepthwiseConv 7×7 → LayerNorm → Linear 4× → GELU → Linear 1× → + x
```

---

### Concept 4: Recurrent Neural Networks (RNNs)

RNNs process sequences by maintaining hidden state across time steps.

**Vanilla RNN:**

```
hₜ = tanh(Wₕₕ · hₜ₋₁ + Wₓₕ · xₜ + b)
yₜ = Wₕᵧ · hₜ
```

**Problem:** Vanishing/exploding gradients over long sequences.

**LSTM (Long Short-Term Memory):**

```
Forget gate:  fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)
Input gate:   iₜ = σ(Wi · [hₜ₋₁, xₜ] + bi)
Cell update:  C̃ₜ = tanh(Wc · [hₜ₋₁, xₜ] + bc)
Cell state:   Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ
Output gate:  oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)
Hidden state: hₜ = oₜ ⊙ tanh(Cₜ)
```

Gates control information flow, enabling long-term memory.

**GRU (Gated Recurrent Unit):**

Simplified LSTM with two gates (reset, update) instead of three:

```
Update gate: zₜ = σ(Wz · [hₜ₋₁, xₜ])
Reset gate:  rₜ = σ(Wr · [hₜ₋₁, xₜ])
Candidate:   h̃ₜ = tanh(W · [rₜ ⊙ hₜ₋₁, xₜ])
Hidden:      hₜ = (1 - zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ
```

**Bidirectional RNNs:**

```
Forward:  h→ₜ processes left-to-right
Backward: h←ₜ processes right-to-left
Combined: hₜ = [h→ₜ; h←ₜ] (concatenate)
```

Captures both past and future context.

**RNN Limitations:**
- Sequential computation (slow)
- Gradient flow issues despite gates
- Largely replaced by Transformers for most tasks

---

### Concept 5: Transformer Architecture

Transformers use attention mechanisms for parallel sequence processing.

**Self-Attention Mechanism:**

```
Q = XWQ,  K = XWK,  V = XWV

Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V
```

Every position attends to every other position, weighted by relevance.

**Multi-Head Attention:**

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) Wᴼ

where headᵢ = Attention(QWᵢQ, KWᵢK, VWᵢV)
```

Multiple heads capture different relationship types.

**Transformer Block:**

```
┌─────────────────────────────────────────────────────┐
│  x ──┬─→ LayerNorm → MultiHeadAttn → + ──┬─→ output │
│      └──────────────────────────────────┘           │
│                                                     │
│  → ──┬─→ LayerNorm → FFN(MLP) ──────→ + ──→        │
│      └──────────────────────────────────┘           │
└─────────────────────────────────────────────────────┘
```

Pre-LN variant (shown) is more stable than Post-LN.

**Transformer Variants:**

| Variant | Structure | Use Case |
|---------|-----------|----------|
| Encoder-only | Bidirectional attention | BERT, classification |
| Decoder-only | Causal (left-to-right) attention | GPT, generation |
| Encoder-Decoder | Cross-attention between enc/dec | T5, translation |

**Positional Encoding:**

Since attention is position-agnostic, position information must be added:

| Method | Description |
|--------|-------------|
| Sinusoidal | Fixed sin/cos functions at different frequencies |
| Learned | Trainable embedding per position |
| RoPE | Rotary position embedding; relative positions |
| ALiBi | Linear attention bias based on distance |

---

### Concept 6: Encoder-Decoder Architectures

The encoder-decoder pattern separates input understanding from output generation.

**General Structure:**

```
Input → [Encoder] → Latent Representation → [Decoder] → Output
         Compress      Information bottleneck    Reconstruct/Generate
```

**Applications:**

| Domain | Encoder | Decoder | Example |
|--------|---------|---------|---------|
| Translation | Text encoder | Text decoder | T5, mBART |
| Image segmentation | Image encoder | Pixel decoder | SegNet |
| Image generation | Text/Image encoder | Image decoder | Stable Diffusion |
| Speech | Audio encoder | Text decoder | Whisper |

**Sequence-to-Sequence with Attention:**

```
Encoder: h₁, h₂, ..., hₙ = Encoder(x₁, ..., xₙ)

Decoder step t:
  context = Attention(sₜ, h₁..ₙ)  # Attend to encoder outputs
  sₜ₊₁, yₜ = Decoder(sₜ, context, yₜ₋₁)
```

Cross-attention allows decoder to focus on relevant encoder positions.

---

### Concept 7: U-Net and Skip Connection Architectures

U-Net combines encoder-decoder with skip connections for dense prediction tasks.

**U-Net Structure:**

```
Input ───────────────────────────────────────────────► Skip
   ↓                                                    │
Encoder Block 1 ────────────────────────────────────► Skip │
   ↓                                                    │ │
Encoder Block 2 ────────────────────────────────────► Skip │ │
   ↓                                                    │ │ │
   ... (bottleneck) ...                                 │ │ │
   ↓                                                    │ │ │
Decoder Block 2 ◄────────── Concatenate ◄──────────────┘ │ │
   ↓                                                      │ │
Decoder Block 1 ◄────────── Concatenate ◄────────────────┘ │
   ↓                                                        │
Output ◄────────────────── Concatenate ◄────────────────────┘
```

**Key Properties:**
- Skip connections preserve high-resolution features
- Enables precise localization for segmentation
- Foundation for diffusion model U-Nets

**Applications:**
- Medical image segmentation
- Diffusion model denoising
- Image-to-image translation
- Depth estimation

**Residual Connections (ResNet):**

```
y = F(x) + x

Where F(x) is the residual function (conv layers)
```

Enables training very deep networks (100+ layers) by providing gradient highways.

---

### Concept 8: Attention Variants and Efficient Transformers

Standard attention has O(n²) complexity. Many variants improve efficiency.

**Attention Complexity:**

```
Standard: O(n² · d) time, O(n²) memory
Where n = sequence length, d = dimension
```

**Efficient Attention Methods:**

| Method | Complexity | Approach |
|--------|------------|----------|
| Sparse Attention | O(n√n) | Attend to subset of positions |
| Linear Attention | O(n) | Kernel trick approximation |
| Flash Attention | O(n²) time, O(n) memory | Memory-efficient CUDA kernels |
| Sliding Window | O(n·w) | Local attention window |
| Longformer | O(n) | Combine local + global tokens |

**Flash Attention:**

```
Standard: Materialize full n×n attention matrix
Flash: Tile computation, never materialize full matrix

Result: Same output, 2-4× faster, O(n) memory
```

Flash Attention is now standard for Transformer training.

**Multi-Query Attention (MQA):**

```
Standard MHA: Separate K, V for each head
MQA: Single K, V shared across all heads

Benefit: Faster inference (smaller KV cache)
```

**Grouped-Query Attention (GQA):**

```
MHA: h heads × separate KV = h KV heads
MQA: h heads × shared KV = 1 KV head
GQA: h heads × grouped KV = g KV heads (1 < g < h)

Balance between quality and efficiency
```

---

### Concept 9: Mixture of Experts (MoE)

MoE architectures scale model capacity while maintaining computational efficiency.

**MoE Layer:**

```
Input x → Router → Top-k Experts → Weighted Sum → Output

Router: g(x) = softmax(Wᵣ · x)
Output: y = Σᵢ gᵢ(x) · Expertᵢ(x)  for top-k experts
```

**Key Concepts:**

| Concept | Description |
|---------|-------------|
| Router | Network that selects which experts to use |
| Expert | Independent sub-network (typically FFN) |
| Top-k routing | Only activate k experts per token (k=1 or 2) |
| Load balancing | Auxiliary loss to distribute load across experts |

**MoE Benefits:**

```
Standard FFN:  All parameters active for all inputs
MoE with k=2:  Only 2/N experts active per token

Example: Mixtral 8×7B
- 8 experts × 7B each = 56B total parameters
- Only 2 experts active = ~13B compute per token
```

**MoE Challenges:**
- Load imbalance (some experts underused)
- Training instability
- Communication overhead in distributed training

---

### Concept 10: Graph Neural Networks (GNNs)

GNNs process graph-structured data by aggregating information from node neighborhoods.

**Message Passing Framework:**

```
For each layer:
1. Message: mᵢⱼ = Message(hᵢ, hⱼ, eᵢⱼ)  # From neighbor j to node i
2. Aggregate: mᵢ = Aggregate({mᵢⱼ : j ∈ N(i)})  # Combine messages
3. Update: hᵢ' = Update(hᵢ, mᵢ)  # Update node representation
```

**Common GNN Architectures:**

| Architecture | Aggregation | Message |
|--------------|-------------|---------|
| GCN | Mean | Linear(hⱼ) |
| GraphSAGE | Mean/Max/LSTM | Linear(hⱼ) |
| GAT | Attention-weighted | Attention(hᵢ, hⱼ) |
| GIN | Sum + MLP | MLP(hⱼ) |

**Graph Attention (GAT):**

```
αᵢⱼ = softmax(LeakyReLU(aᵀ[Whᵢ || Whⱼ]))  # Attention coefficient
hᵢ' = σ(Σⱼ αᵢⱼ · Whⱼ)  # Weighted aggregation
```

**Applications:**
- Molecular property prediction
- Social network analysis
- Recommendation systems
- Knowledge graph reasoning
- Traffic prediction

---

## Modern Architectural Innovations

### State Space Models (SSMs) / Mamba

Linear-time alternative to Transformers for long sequences:

```
h'(t) = Ah(t) + Bx(t)  # Continuous state update
y(t) = Ch(t) + Dx(t)   # Output

Discretized for sequences with selective state updates
```

**Mamba Advantages:**
- O(n) complexity vs O(n²) for attention
- Hardware-efficient implementation
- Strong performance on long sequences

### Vision Transformers (ViT)

Apply Transformers to images by treating patches as tokens:

```
Image (224×224) → 16×16 patches → 196 tokens → Transformer → Classification
```

**Variants:**
- DeiT: Data-efficient training with distillation
- Swin: Hierarchical with shifted windows
- MAE: Masked autoencoder pre-training

### Diffusion Model Architectures

Combine U-Net with Transformer components:

```
Noisy Image → U-Net Encoder → (Cross-Attention with text) → U-Net Decoder → Denoised
                    ↑
              Time embedding
```

**DiT (Diffusion Transformer):** Replace U-Net with pure Transformer.

---

## Architecture Selection Guide

| Task | Primary Architecture | Alternative |
|------|---------------------|-------------|
| Image classification | CNN (EfficientNet, ConvNeXt) | ViT |
| Object detection | CNN backbone + detection head | DETR (Transformer) |
| Image segmentation | U-Net, Mask R-CNN | SegFormer |
| Text classification | Encoder Transformer (BERT) | BiLSTM |
| Text generation | Decoder Transformer (GPT) | — |
| Translation | Encoder-Decoder Transformer | — |
| Time series | Transformer, LSTM | TCN |
| Graph data | GNN (GAT, GCN) | Graph Transformer |
| Audio | CNN + Transformer (Whisper) | RNN |
| Multimodal | Vision-Language Transformer | — |

---

## Architectural Design Principles

### Principle 1: Match Architecture to Data Structure

| Data Structure | Architectural Feature |
|----------------|----------------------|
| Sequential | Causal attention or recurrence |
| Spatial (2D) | Convolutions or local attention |
| Graph | Message passing |
| Set (unordered) | Permutation-invariant aggregation |

### Principle 2: Depth vs. Width

```
Depth: More layers → more abstraction levels
Width: More neurons per layer → more capacity per level

ResNet: Enables extreme depth (1000+ layers)
Wide networks: Often easier to train than very deep
```

### Principle 3: Compute vs. Memory Tradeoffs

| Technique | Effect |
|-----------|--------|
| Gradient checkpointing | Less memory, more compute |
| Mixed precision | Less memory, faster compute |
| MoE | More parameters, same compute |
| Attention alternatives | Less compute for long sequences |

---

## Summary

Neural network architectures provide the structural foundation for deep learning capabilities. MLPs offer flexible function approximation; CNNs exploit spatial structure through convolutions; RNNs maintain sequential memory through hidden states; Transformers enable parallel attention-based processing. Modern innovations include efficient attention variants, mixture of experts for scaling, state space models for long sequences, and specialized architectures for graphs and multimodal data.

Architecture selection should match data structure (spatial, sequential, graph), task requirements (classification, generation, dense prediction), and computational constraints. Understanding architectural building blocks—layers, activations, normalization, connections—enables both effective application of existing architectures and contribution to architectural innovation.

---

## Quick Reference

### Architecture Comparison

| Architecture | Inductive Bias | Complexity | Best For |
|--------------|----------------|------------|----------|
| MLP | None | O(n·d²) | Tabular, components |
| CNN | Local, translation equivariant | O(n·k²·c²) | Images, spatial |
| RNN/LSTM | Sequential | O(n·d²) | Short sequences |
| Transformer | Global (attention) | O(n²·d) | Text, modern tasks |
| GNN | Graph structure | O(E·d²) | Graphs, molecules |

### Key Equations

| Component | Equation |
|-----------|----------|
| Linear | y = Wx + b |
| Convolution | y[i] = Σⱼ x[i+j] · k[j] |
| Self-Attention | Attn = softmax(QKᵀ/√d)V |
| LSTM Cell | Cₜ = fₜ⊙Cₜ₋₁ + iₜ⊙C̃ₜ |
| Residual | y = F(x) + x |
| LayerNorm | y = (x-μ)/σ · γ + β |

### Modern Best Practices

- Use Pre-LN for Transformers (more stable)
- Flash Attention for memory efficiency
- RMSNorm for efficient normalization
- GQA for inference efficiency
- LoRA for efficient fine-tuning
- Mixed precision (BF16) for training

---

*Next Lesson: Lesson 9 - Natural Language Processing*
