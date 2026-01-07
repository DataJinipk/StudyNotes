# Flashcards: Lesson 8 - Neural Network Architectures

**Source:** Lessons/Lesson_8.md
**Subject Area:** AI Learning - Neural Network Architectures: Design Patterns and Modern Innovations
**Date Generated:** 2026-01-08
**Total Cards:** 5 (2 Easy, 2 Medium, 1 Hard)

---

## Card Distribution

| Difficulty | Count | Bloom's Level | Focus Area |
|------------|-------|---------------|------------|
| Easy | 2 | Remember/Understand | Core concepts, architecture comparison |
| Medium | 2 | Apply/Analyze | Design decisions, complexity analysis |
| Hard | 1 | Evaluate/Synthesize | Architecture design, tradeoff analysis |

---

## Easy Cards

### Card 1: Architecture Family Comparison

**[FRONT]**
Compare the four major neural network architecture families (MLP, CNN, RNN, Transformer) in terms of their inductive biases, computational complexity, and primary use cases.

**[BACK]**
**Architecture Family Comparison:**

| Architecture | Inductive Bias | Complexity | Primary Use Case |
|--------------|----------------|------------|------------------|
| **MLP** | None (fully connected) | O(n·d²) | Tabular data, components |
| **CNN** | Local, translation equivariant | O(n·k²·c²) | Images, spatial data |
| **RNN/LSTM** | Sequential, recurrent | O(n·d²) sequential | Short sequences (legacy) |
| **Transformer** | Global attention | O(n²·d) | Text, modern tasks |

**Inductive Bias Explanations:**
- **MLP:** No structural assumptions; learns any relationship equally
- **CNN:** Assumes local patterns matter, features are translation-invariant
- **RNN:** Assumes sequential dependencies, earlier context influences later
- **Transformer:** Assumes any position can relate to any other (global)

**Key Insight:** Match architecture to data structure—CNNs for spatial, Transformers for sequences needing global context, GNNs for graphs.

**Difficulty:** Easy | **Bloom's Level:** Remember

---

### Card 2: Attention Mechanism Fundamentals

**[FRONT]**
Explain the self-attention mechanism in Transformers. What are Q, K, V? What does the softmax over QKᵀ compute, and why is scaling by √dₖ necessary?

**[BACK]**
**Self-Attention Mechanism:**

```
Q = XWQ  (Queries: what am I looking for?)
K = XWK  (Keys: what do I contain?)
V = XWV  (Values: what information do I provide?)

Attention(Q, K, V) = softmax(QKᵀ / √dₖ) × V
```

**Component Functions:**

| Component | Dimension | Purpose |
|-----------|-----------|---------|
| Q (Query) | n × dₖ | "What I'm searching for" |
| K (Key) | n × dₖ | "What I can be matched with" |
| V (Value) | n × dᵥ | "What I contribute if matched" |
| QKᵀ | n × n | Similarity scores between all pairs |
| softmax | n × n | Convert to probability weights (rows sum to 1) |

**Why √dₖ Scaling?**

```
Without scaling: QKᵀ values grow with dimension dₖ
Large values → softmax approaches one-hot
One-hot attention → gradient vanishes

With scaling: QKᵀ/√dₖ keeps values in reasonable range
Softmax produces smooth distribution
Gradients flow to multiple positions
```

**Example:** If dₖ = 64, scale by √64 = 8 to normalize variance.

**Difficulty:** Easy | **Bloom's Level:** Understand

---

## Medium Cards

### Card 3: Efficient Attention Selection

**[FRONT]**
You are designing a system to process documents of 100,000 tokens. Standard Transformer attention is infeasible. Compare three efficient attention alternatives (Flash Attention, Sparse Attention, Linear Attention) and recommend which to use for this task.

**[BACK]**
**Efficient Attention Comparison for Long Documents:**

| Method | Time | Memory | Quality | Implementation |
|--------|------|--------|---------|----------------|
| Standard | O(n²) | O(n²) | Best | Simple |
| Flash Attention | O(n²) | O(n) | Same as standard | CUDA kernels |
| Sparse Attention | O(n√n) | O(n√n) | Good | Pattern design |
| Linear Attention | O(n) | O(n) | Lower | Kernel approximation |

**For 100,000 tokens:**

| Method | Memory (approx) | Feasibility |
|--------|-----------------|-------------|
| Standard | 100K² × 4B = 40TB | Impossible |
| Flash | O(n) ≈ 400MB | Feasible |
| Sparse | O(n√n) ≈ 1.3GB | Feasible |
| Linear | O(n) ≈ 400MB | Feasible |

**Recommendation: Flash Attention + Sparse/Sliding Window**

```
Strategy:
1. Use Flash Attention for memory efficiency (same quality)
2. Add sliding window attention (local context)
3. Include global tokens for document-level understanding

Implementation: Longformer-style architecture
- Local window: 512 tokens (handles most context)
- Global tokens: First token, section headers
- Flash kernels: Memory efficient at each scale
```

**Rationale:**
- Flash Attention provides full attention quality with O(n) memory
- For 100K tokens, even Flash needs hierarchical approach
- Sliding window + global tokens capture both local and document-level patterns

**Difficulty:** Medium | **Bloom's Level:** Apply

---

### Card 4: Residual Connection Analysis

**[FRONT]**
Explain why residual connections (skip connections) enable training of very deep networks. Include the gradient flow analysis and explain the "gradient highway" concept.

**[BACK]**
**Residual Connections Enable Deep Networks:**

**Standard Layer:**
```
y = F(x)  where F is conv/linear + activation
```

**Residual Layer:**
```
y = F(x) + x  (identity shortcut)
```

**Gradient Flow Analysis:**

Without residual:
```
∂L/∂x = ∂L/∂y × ∂y/∂x = ∂L/∂y × ∂F/∂x

Through N layers: ∂L/∂x₀ = ∏ᵢ ∂Fᵢ/∂xᵢ

If each |∂F/∂x| < 1: gradient vanishes exponentially
If each |∂F/∂x| > 1: gradient explodes exponentially
```

With residual:
```
∂L/∂x = ∂L/∂y × ∂(F(x)+x)/∂x = ∂L/∂y × (∂F/∂x + 1)

The "+1" term creates a gradient highway!
```

**Gradient Highway Concept:**

```
Layer N:    ∂L/∂xₙ = ∂L/∂y × (∂F/∂x + 1)
                              └── Always ≥ 1

Through N residual layers:
∂L/∂x₀ = ∂L/∂y × (∂Fₙ/∂x + 1) × ... × (∂F₁/∂x + 1)

Each factor contains "+1" ensuring:
- Gradient never vanishes completely
- Direct path from output to any layer
```

**Practical Impact:**

| Network | Without Residual | With Residual |
|---------|------------------|---------------|
| 20 layers | Trainable | Trainable |
| 50 layers | Difficult | Trainable |
| 100 layers | Fails | Trainable |
| 1000 layers | Impossible | Trainable |

**Key Insight:** Residual connections don't just help—they fundamentally change what's learnable. The network learns the residual F(x) = H(x) - x, which is often easier than learning H(x) directly.

**Difficulty:** Medium | **Bloom's Level:** Analyze

---

## Hard Cards

### Card 5: Complete Architecture Design

**[FRONT]**
Design a neural network architecture for a multimodal document understanding system that must:
1. Process documents up to 50 pages with text, tables, and images
2. Answer questions requiring cross-page reasoning
3. Handle documents in multiple languages
4. Run inference on a single A100 GPU (80GB)

Specify: encoder architectures, attention mechanisms, positional encoding, and memory optimization strategies.

**[BACK]**
**Multimodal Document Understanding Architecture:**

```
┌─────────────────────────────────────────────────────────────────┐
│                     DOCUMENT ENCODER                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        │
│  │ Text Encoder │   │ Table Encoder│   │ Image Encoder │        │
│  │ (mBERT/XLM-R)│   │ (TableFormer)│   │ (ViT-L/14)   │        │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘        │
│         │                  │                   │                 │
│         └─────────────────┼───────────────────┘                 │
│                           ▼                                      │
│              ┌────────────────────────┐                         │
│              │  Modality Fusion Layer │                         │
│              │  (Cross-attention)     │                         │
│              └───────────┬────────────┘                         │
│                          ▼                                       │
│              ┌────────────────────────┐                         │
│              │  Hierarchical Encoder  │                         │
│              │  Page → Section → Doc  │                         │
│              └───────────┬────────────┘                         │
└──────────────────────────┼──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     REASONING MODULE                             │
├─────────────────────────────────────────────────────────────────┤
│  Question ───►  Cross-Attention to Document  ───►  Answer       │
│                 (Longformer-style attention)                    │
└─────────────────────────────────────────────────────────────────┘
```

**Component Specifications:**

**1. Text Encoder:**
```yaml
model: XLM-RoBERTa-large
reason: Multilingual support (100+ languages)
max_tokens_per_page: 512
strategy: Process pages independently, then aggregate
```

**2. Table Encoder:**
```yaml
model: TableFormer or TAPAS
approach: Linearize tables with special tokens
  [ROW_1] cell1 [SEP] cell2 [SEP] ...
preserve: Row/column position embeddings
```

**3. Image Encoder:**
```yaml
model: ViT-L/14 (frozen CLIP vision encoder)
resolution: 224×224 per image region
tokens_per_image: 196 (14×14 patches)
strategy: Treat page screenshots as additional modality
```

**4. Hierarchical Document Encoding:**
```yaml
level_1_page:
  method: Perceiver-style cross-attention
  latent_tokens: 64 per page
  reduces: ~4000 tokens/page → 64 latents

level_2_section:
  method: Transformer over page latents
  attention: Sliding window (8 pages) + global

level_3_document:
  method: Global attention over section summaries
  output: Document-level representation
```

**5. Attention Mechanism:**
```yaml
type: Longformer-style hierarchical
local_window: 512 tokens
global_tokens:
  - Question tokens (always attend globally)
  - Page start tokens
  - Table headers

implementation: Flash Attention 2 for memory efficiency
```

**6. Positional Encoding:**
```yaml
within_page: RoPE (rotary position embedding)
cross_page: Learned absolute position + page number embedding
tables: 2D position encoding (row, column)
images: ViT patch positions
```

**Memory Optimization for 80GB A100:**

```
Component Memory Budget:
├── Text encoder (frozen): ~2GB
├── Image encoder (frozen): ~1GB
├── Fusion layers: ~4GB
├── Hierarchical encoder: ~8GB
├── Reasoning module: ~4GB
├── KV cache: ~20GB (for 50 pages)
├── Activations: ~30GB
└── Buffer: ~11GB
Total: ~80GB

Optimization Strategies:
1. Flash Attention 2: Reduce attention memory 4×
2. Gradient checkpointing: Trade compute for memory
3. BF16 mixed precision: Halve weight memory
4. Perceiver latent compression: 4000 → 64 tokens/page
5. Frozen encoders: No optimizer states
```

**Inference Pipeline:**
```python
def process_document(pages, question):
    # 1. Encode each page independently (parallelizable)
    page_encodings = []
    for page in pages:
        text_enc = text_encoder(page.text)
        table_enc = [table_encoder(t) for t in page.tables]
        image_enc = image_encoder(page.screenshot)
        page_enc = modality_fusion(text_enc, table_enc, image_enc)
        page_latents = perceiver_compress(page_enc)  # 64 tokens
        page_encodings.append(page_latents)

    # 2. Hierarchical encoding
    doc_encoding = hierarchical_encoder(page_encodings)

    # 3. Question-conditioned reasoning
    question_enc = text_encoder(question)
    answer = reasoning_module(question_enc, doc_encoding)

    return answer
```

**Difficulty:** Hard | **Bloom's Level:** Synthesize

---

## Critical Knowledge Flags

The following concepts appear across multiple cards and represent essential knowledge:

| Concept | Cards | Significance |
|---------|-------|--------------|
| Attention mechanism | 2, 3, 5 | Foundation of modern architectures |
| Residual connections | 4, 5 | Enables deep network training |
| Complexity analysis | 1, 3, 5 | Critical for practical design |
| Architecture selection | 1, 3, 5 | Matching structure to task |

---

## Study Recommendations

### Before These Cards
- Review Lesson 4 (Transformers) for attention details
- Review Lesson 5 (Deep Learning) for gradient flow concepts

### After Mastering These Cards
- Implement a simple Transformer from scratch
- Experiment with different attention patterns
- Read architecture papers (ResNet, ViT, Longformer)

### Spaced Repetition Schedule
| Session | Focus |
|---------|-------|
| Day 1 | Cards 1-2 (foundations) |
| Day 3 | Cards 3-4 (analysis) |
| Day 7 | Card 5 (synthesis), review 1-4 |
| Day 14 | Full review |

---

*Generated from Lesson 8: Neural Network Architectures | Flashcard Skill*
