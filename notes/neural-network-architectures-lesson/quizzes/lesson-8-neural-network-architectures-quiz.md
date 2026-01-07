# Assessment Quiz: Lesson 8 - Neural Network Architectures

**Source:** Lessons/Lesson_8.md
**Subject Area:** AI Learning - Neural Network Architectures: Design Patterns and Modern Innovations
**Date Generated:** 2026-01-08
**Total Questions:** 5
**Estimated Time:** 35-45 minutes

---

## Instructions

This assessment evaluates your understanding of neural network architectures, including fundamental building blocks, major architecture families, design patterns, and modern innovations. Answer all questions completely, showing your reasoning where applicable.

**Question Distribution:**
- Multiple Choice (2): Conceptual understanding (Remember/Understand)
- Short Answer (2): Application and analysis (Apply/Analyze)
- Essay (1): Synthesis and evaluation (Evaluate/Synthesize)

---

## Part A: Multiple Choice (10 points each)

### Question 1: Inductive Biases

**Which statement correctly describes the relationship between architecture choice and inductive biases for learning?**

A) CNNs assume global relationships between all pixels equally, making them ideal for images with long-range dependencies

B) Transformers assume local spatial patterns are most important, which is why they excel at image classification without modification

C) RNNs assume sequential dependencies where earlier elements influence later ones, but struggle with very long sequences due to gradient flow issues

D) MLPs have the strongest inductive biases, which is why they require less data than CNNs or Transformers to achieve good performance

---

### Question 2: Efficient Attention Mechanisms

**A research team is training a language model on documents with 32,000 tokens. Standard attention requires storing a 32K × 32K matrix, which exceeds their GPU memory. They're considering Flash Attention, Sparse Attention, and Linear Attention. Which statement is correct?**

A) Flash Attention reduces the time complexity from O(n²) to O(n), making it faster but potentially lower quality

B) Flash Attention maintains O(n²) time complexity but achieves O(n) memory by computing attention in tiles without materializing the full matrix

C) Sparse Attention and Linear Attention produce identical outputs to standard attention but with lower complexity

D) Linear Attention is always preferred over Flash Attention because it has lower computational complexity

---

## Part B: Short Answer (15 points each)

### Question 3: Architecture Selection

**Context:** You are designing neural network architectures for three different tasks at a company. For each task, recommend a primary architecture family and justify your choice.

**Tasks:**

a) **Fraud Detection:** Analyze transaction records with 50 features (amount, time, merchant category, etc.) to classify as fraud/legitimate. Dataset has 1M transactions. (5 points)

b) **Satellite Image Segmentation:** Label each pixel in 1024×1024 satellite images as one of 20 land-use categories. Objects range from 16×16 to 512×512 pixels. (5 points)

c) **Molecular Property Prediction:** Predict solubility of drug molecules represented as graphs with atoms as nodes and bonds as edges. Molecules have 10-100 atoms. (5 points)

---

### Question 4: Residual Connections Analysis

**Context:** A colleague claims that residual connections are "just a trick" and that a regular deep network should be able to learn the same functions.

**Tasks:**

a) Explain mathematically why residual connections enable training of much deeper networks. Show the gradient flow comparison. (5 points)

b) Beyond gradient flow, explain the "residual learning" hypothesis—why learning F(x) = H(x) - x might be easier than learning H(x) directly. (5 points)

c) ResNet-152 outperforms ResNet-18 on ImageNet despite both having residual connections. If residual connections solve the depth problem, why does additional depth still help? What are the limits? (5 points)

---

## Part C: Essay (30 points)

### Question 5: Modern Architecture Comparison

**Prompt:** You are advising a startup building a document AI system that needs to process long documents (up to 100 pages, ~50,000 tokens) with text, tables, and images. The CEO asks you to compare three architectural approaches:

1. **Traditional Approach:** Encoder-only Transformer (like BERT) with chunking
2. **Efficient Transformers:** Longformer or BigBird with sparse attention patterns
3. **State Space Models:** Mamba or similar S4-based architectures

**Your essay must address:**

1. **Architectural Analysis** (8 points)
   - Key mechanisms of each approach
   - Theoretical complexity comparison (time and memory)
   - How each handles the 50K token context

2. **Quality vs. Efficiency Tradeoffs** (7 points)
   - Where each architecture excels
   - Known limitations or failure modes
   - Impact on downstream task performance

3. **Multimodal Considerations** (7 points)
   - How to incorporate images and tables in each architecture
   - Cross-modal attention requirements
   - Additional architectural components needed

4. **Recommendation** (8 points)
   - Your recommended approach with justification
   - Hybrid strategies if applicable
   - Implementation considerations

**Evaluation Criteria:**
- Technical accuracy of architectural descriptions
- Thoughtful analysis of tradeoffs
- Practical considerations for real deployment
- Well-reasoned recommendation

**Word Limit:** 600-800 words

---

## Answer Key

### Question 1: Inductive Biases

**Correct Answer: C**

**Explanation:**

| Statement | Assessment |
|-----------|------------|
| **A (CNNs)** | Incorrect. CNNs assume *local* spatial patterns via convolution kernels. They do NOT assume global relationships—that's Transformers. |
| **B (Transformers)** | Incorrect. Transformers assume *any* position can relate to *any* other (global). They don't assume local patterns, which is why ViT needs large datasets. |
| **C (RNNs)** | Correct. RNNs process sequentially, assuming temporal dependencies. Vanishing gradients make long sequences difficult, which motivated LSTM/Transformer development. |
| **D (MLPs)** | Incorrect. MLPs have *no* inductive bias (fully connected), requiring MORE data than architectures with appropriate biases for structured data. |

**Understanding Gap:** If you selected A or B, review the definition of inductive bias and how each architecture encodes assumptions about data structure.

---

### Question 2: Efficient Attention Mechanisms

**Correct Answer: B**

**Explanation:**

| Statement | Assessment |
|-----------|------------|
| **A** | Incorrect. Flash Attention maintains O(n²) *time* complexity—it computes the same attention, just more memory-efficiently. |
| **B** | Correct. Flash Attention uses tiled computation to avoid materializing the full n×n matrix, achieving O(n) memory while keeping exact attention computation. |
| **C** | Incorrect. Sparse and Linear Attention are *approximations* that produce different (lower quality) outputs than standard attention. |
| **D** | Incorrect. Linear Attention often has lower quality; Flash Attention is preferred when exact attention is needed and time is acceptable. |

**Flash Attention Key Insight:**
```
Standard: Compute full QKᵀ matrix → Softmax → Multiply by V
Flash: Tile computation, fuse operations, never store full matrix
Result: Same output, O(n²) time, O(n) memory
```

**Understanding Gap:** If you selected A, review the distinction between time and memory complexity. Flash Attention optimizes memory, not time.

---

### Question 3: Architecture Selection

**Model Answer:**

**a) Fraud Detection - MLP (Feedforward Network)**

**Recommendation:** MLP with 3-5 hidden layers, BatchNorm, and Dropout

**Justification:**
- **Data structure:** Tabular data with 50 independent features—no spatial or sequential structure
- **Inductive bias:** MLPs have no structural assumptions, appropriate for heterogeneous tabular features
- **Alternatives considered:**
  - CNNs: Inappropriate—no spatial relationship between features
  - Transformers: Overkill for 50 features; attention overhead unnecessary
  - Tree ensembles (XGBoost): Actually a strong alternative for tabular data

**Architecture:**
```
Input(50) → Dense(256) → BN → ReLU → Dropout(0.3)
         → Dense(128) → BN → ReLU → Dropout(0.3)
         → Dense(64)  → BN → ReLU
         → Dense(1)   → Sigmoid
```

---

**b) Satellite Image Segmentation - U-Net (CNN with Skip Connections)**

**Recommendation:** U-Net or similar encoder-decoder with skip connections

**Justification:**
- **Data structure:** 2D spatial data requiring dense per-pixel prediction
- **Multi-scale requirement:** Objects from 16×16 to 512×512 require features at multiple scales
- **Skip connections:** Preserve high-resolution details for precise boundaries

**Why U-Net:**
- Encoder captures multi-scale features (16×16 objects at deep layers, 512×512 at early layers)
- Decoder recovers spatial resolution
- Skip connections combine low-level boundaries with high-level semantics

**Alternative:** SegFormer (Transformer-based) if data is abundant and compute available.

---

**c) Molecular Property Prediction - Graph Neural Network (GNN)**

**Recommendation:** Message Passing Neural Network (MPNN) or Graph Attention Network (GAT)

**Justification:**
- **Data structure:** Molecules are naturally graphs—atoms as nodes, bonds as edges
- **Permutation invariance:** Atom ordering shouldn't matter; GNNs are permutation-equivariant
- **Variable size:** GNNs handle graphs with 10-100 nodes naturally

**Why GNN:**
- Message passing captures local chemical environment
- Aggregation learns molecular fingerprint
- Established success in drug discovery (SchNet, DimeNet, ChemProp)

**Architecture:**
```
Atom features → GNN layers (message passing) → Global pooling → MLP → Solubility
```

---

### Question 4: Residual Connections Analysis

**Model Answer:**

**a) Mathematical Gradient Flow Analysis (5 points)**

**Without Residual Connections:**
```
Layer N: y = F(x)
Gradient: ∂L/∂x = ∂L/∂y × ∂F/∂x

Through N layers:
∂L/∂x₀ = ∏ᵢ₌₁ⁿ ∂Fᵢ/∂xᵢ
```

If each |∂F/∂x| < 1 (common with sigmoid/tanh): Product → 0 exponentially (vanishing)
If each |∂F/∂x| > 1: Product → ∞ exponentially (exploding)

**With Residual Connections:**
```
Layer N: y = F(x) + x
Gradient: ∂L/∂x = ∂L/∂y × (∂F/∂x + 1)
                            └── Always ≥ 1!

Through N layers:
∂L/∂x₀ = ∏ᵢ₌₁ⁿ (∂Fᵢ/∂xᵢ + 1)
```

Each factor contains "+1", ensuring gradient never vanishes completely. Even if ∂F/∂x = 0, gradient is still 1.

**b) Residual Learning Hypothesis (5 points)**

**Intuition:** In many cases, the optimal transformation is close to identity (H(x) ≈ x).

**Without residual:** Network must learn H(x) directly
- If optimal H(x) ≈ x, weights must learn complex identity approximation
- Small deviations from identity require precise weight configuration

**With residual:** Network learns F(x) = H(x) - x
- If optimal H(x) ≈ x, then optimal F(x) ≈ 0
- Learning F(x) = 0 is trivial (just set weights near zero)
- Learning small refinements is easier than learning full transformation

**Evidence:** Deep residual networks with hundreds of layers often learn F(x) with small magnitude, confirming that layers learn refinements rather than complete transformations.

**c) Why More Depth Still Helps (5 points)**

**Residual connections solve optimization, not representation:**

| Aspect | What Residuals Provide | What Depth Provides |
|--------|----------------------|---------------------|
| Optimization | Gradient highways | — |
| Capacity | — | More parameters, more abstraction levels |
| Feature hierarchy | — | Progressive feature refinement |
| Effective RF | — | Larger receptive field in CNNs |

**Why ResNet-152 > ResNet-18:**
- More layers = more representational capacity
- Deeper hierarchy of features (edges → textures → parts → objects)
- More parameters to fit complex functions
- Residuals ensure these layers can actually be trained

**Limits of depth:**
- Diminishing returns: ResNet-1000 barely outperforms ResNet-152
- Compute cost: Linear increase in FLOPs
- Overfitting: More parameters require more data
- Practical: ResNet-50/101 are often sweet spots for ImageNet

---

### Question 5: Modern Architecture Comparison

**Rubric (30 points total):**

| Component | Excellent (Full) | Adequate (Half) | Insufficient (Minimal) |
|-----------|------------------|-----------------|------------------------|
| Architectural Analysis (8) | Accurate mechanisms and complexity for all three | Mostly correct with minor errors | Major misunderstandings |
| Tradeoffs (7) | Nuanced analysis with specific examples | General comparison without depth | Missing key tradeoffs |
| Multimodal (7) | Concrete strategies for each architecture | General discussion | Vague or missing |
| Recommendation (8) | Well-justified with practical considerations | Reasonable but incomplete justification | Weak or unsupported |

**Model Answer:**

**1. Architectural Analysis**

**BERT with Chunking:**
The traditional approach splits 50K tokens into chunks (e.g., 512 tokens each), processes independently, then aggregates. Each chunk uses full O(n²) attention within its window.

*Complexity:* O(C × 512²) time and memory per chunk, where C = 100 chunks. No cross-chunk attention—information cannot flow between chunks during encoding.

*50K Context Handling:* Processes chunks independently, relies on downstream aggregation. Misses cross-page dependencies entirely.

**Longformer/BigBird:**
These architectures combine local sliding-window attention (O(n×w) where w is window size) with sparse global attention tokens. Longformer uses global tokens at [CLS] and task-specific positions; BigBird adds random attention patterns.

*Complexity:* O(n×w + n×g) where g is global tokens, effectively O(n) for fixed w, g. Memory scales linearly with sequence length.

*50K Context Handling:* Full sequence fits in memory. Local attention captures nearby context; global tokens enable document-level reasoning. Some information loss compared to full attention.

**Mamba/SSM:**
State space models treat sequences as continuous dynamical systems, discretized for neural networks. They maintain a fixed-size hidden state updated recurrently but enable parallel training via convolution.

*Complexity:* O(n×d) time and memory—truly linear in sequence length. No attention matrix computed.

*50K Context Handling:* Handles arbitrary length with constant memory per position. Information propagates through state dynamics, though long-range dependencies are compressed into fixed-size state.

**2. Quality vs. Efficiency Tradeoffs**

**BERT Chunking excels** at: Tasks where local context suffices (sentence classification, NER). Well-understood, many pretrained checkpoints available.
*Limitations:* Cannot reason across chunk boundaries. Multi-hop questions spanning pages fail.

**Longformer/BigBird excels** at: Document-level tasks requiring some global reasoning. Question answering over long documents. Good balance of quality and efficiency.
*Limitations:* Sparse patterns may miss relevant connections. Global tokens become bottleneck for information. Still slower than SSMs.

**Mamba excels** at: Very long sequences (100K+) where attention is prohibitive. Streaming applications. Language modeling with long context.
*Limitations:* Less mature than Transformers. May struggle with tasks requiring precise retrieval (attention's strength). Fixed state size limits information capacity.

**Task Performance Impact:** For retrieval-heavy tasks (find specific clause in document), attention-based models likely outperform SSMs. For summarization and general understanding, SSMs may match or exceed with better efficiency.

**3. Multimodal Considerations**

**BERT + Chunking:** Each modality processed separately, then combined. Images encoded via ViT, linearized into tokens. Challenge: ensuring related image and text chunks align.

**Longformer:** Can interleave image and text tokens. Place image tokens as "global" tokens to enable cross-modal attention. Tables linearized with special tokens. Works well but increases sequence length.

**Mamba:** Multimodal extension less explored. Could interleave modalities in sequence, but state dynamics designed for homogeneous sequences. May need separate encoders per modality with fusion layer.

**Recommendation for multimodal:** Longformer-style architecture with image tokens as global attention points provides best balance of cross-modal reasoning and efficiency.

**4. Recommendation**

For this document AI system, I recommend **Efficient Transformer (Longformer-style) with hierarchical processing**:

**Primary Architecture:**
- Longformer backbone with 4K local window + global tokens
- Process each page as unit, then cross-page attention layer
- Image patches encoded via frozen ViT, injected as global tokens
- Tables linearized with row/column position embeddings

**Justification:**
1. *50K tokens manageable:* O(n) memory makes full document processing feasible
2. *Cross-modal reasoning:* Global attention tokens enable image-text interaction
3. *Maturity:* Well-tested, pretrained checkpoints available
4. *Accuracy:* Attention-based retrieval for finding specific information

**Hybrid Strategy:** Use Mamba for initial document encoding (efficient), then Transformer for final reasoning over compressed representations. This combines SSM efficiency with attention's retrieval strength.

**Implementation Considerations:**
- Start with pretrained Longformer, fine-tune on document QA
- Use Flash Attention for memory efficiency
- Hierarchical approach: page-level then document-level for tractability
- Monitor quality on cross-page reasoning tasks during development

The document AI domain benefits from attention's ability to precisely locate information across modalities, making efficient Transformers the most appropriate choice despite SSMs' superior efficiency.

---

## Performance Interpretation Guide

| Score | Mastery Level | Recommended Action |
|-------|---------------|-------------------|
| 90-100% | **Mastery** | Ready for architecture research and design |
| 75-89% | **Proficient** | Review specific gaps, implement architectures |
| 60-74% | **Developing** | Re-study core architecture families |
| Below 60% | **Foundational** | Complete re-review of Lesson 8 |

---

## Review Recommendations by Question

| If You Struggled With | Review These Sections |
|----------------------|----------------------|
| Question 1 | Lesson 8: Inductive biases, architecture families |
| Question 2 | Lesson 8: Efficient attention, Flash Attention |
| Question 3 | Lesson 8: Architecture selection guide |
| Question 4 | Lesson 8: Residual connections, gradient flow |
| Question 5 | Lesson 8: Transformers, SSMs, multimodal |

---

*Generated from Lesson 8: Neural Network Architectures | Quiz Skill*
