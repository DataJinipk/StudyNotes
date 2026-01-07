# Practice Problems: Lesson 4 - Transformers

**Source:** Lessons/Lesson_4.md
**Subject Area:** AI Learning - Transformer Architecture: Attention Mechanisms, Model Variants, and Implementation
**Date Generated:** 2026-01-08
**Total Problems:** 5
**Distribution:** 1 Warm-Up | 2 Skill-Builder | 1 Challenge | 1 Debug/Fix

---

## Problem Overview

| # | Type | Concept Focus | Difficulty | Estimated Time |
|---|------|---------------|------------|----------------|
| 1 | Warm-Up | Multi-Head Attention Dimensions | Low | 10-15 min |
| 2 | Skill-Builder | Architecture Selection | Medium | 20-25 min |
| 3 | Skill-Builder | Positional Encoding Implementation | Medium | 25-30 min |
| 4 | Challenge | Efficient Attention Design | High | 45-60 min |
| 5 | Debug/Fix | Transformer Training Issues | Medium | 20-25 min |

---

## Problem 1: Warm-Up
### Multi-Head Attention Dimensions

**Concept:** Multi-Head Attention (Core Concept 2)
**Cognitive Level:** Apply
**Prerequisites:** Understanding of attention projections

---

**Problem Statement:**

You are implementing a transformer encoder with the following specifications:
- Model dimension (d_model): 512
- Number of attention heads (h): 8
- Feed-forward inner dimension: 2048
- Sequence length: 100 tokens

**Tasks:**

1. Calculate the dimension of Q, K, V for each attention head (d_k and d_v)
2. Determine the shapes of W_Q, W_K, W_V weight matrices (both per-head and combined)
3. Calculate the shape of the output projection matrix W_O
4. Compute the total number of parameters in one multi-head attention layer
5. Calculate the attention score matrix shape and total FLOPs for one attention computation

---

**Progressive Hints:**

<details>
<summary>Hint 1 (Conceptual)</summary>

For multi-head attention:
- Each head operates on d_k = d_model / h dimensions
- All h heads combined should output d_model dimensions
- Weight matrices project from d_model to h × d_k total
</details>

<details>
<summary>Hint 2 (Procedural)</summary>

Shape calculations:
- d_k = d_v = 512 / 8 = 64 per head
- W_Q shape: [d_model, h × d_k] = [512, 512]
- Attention scores shape: [h, seq_len, seq_len] = [8, 100, 100]
</details>

<details>
<summary>Hint 3 (Solution Direction)</summary>

Parameter count:
- W_Q, W_K, W_V: 3 × (512 × 512) = 786,432
- W_O: 512 × 512 = 262,144
- Total: 1,048,576 (ignoring biases)
</details>

---

**Solution:**

**1. Per-Head Dimensions:**
```
d_k = d_v = d_model / h = 512 / 8 = 64

Each head:
- Query dimension: 64
- Key dimension: 64
- Value dimension: 64
```

**2. Weight Matrix Shapes:**

| Matrix | Per-Head Shape | Combined Shape | Notes |
|--------|---------------|----------------|-------|
| W_Q | [512, 64] (one head) | [512, 512] | Projects d_model → all h heads |
| W_K | [512, 64] (one head) | [512, 512] | Same as W_Q |
| W_V | [512, 64] (one head) | [512, 512] | Same as W_Q |

**3. Output Projection:**
```
W_O shape: [h × d_v, d_model] = [512, 512]

After concatenating h heads: [seq_len, h × d_v] = [100, 512]
After W_O projection: [seq_len, d_model] = [100, 512]
```

**4. Parameter Count:**
```
W_Q: 512 × 512 = 262,144
W_K: 512 × 512 = 262,144
W_V: 512 × 512 = 262,144
W_O: 512 × 512 = 262,144
─────────────────────────────
Total (no bias): 1,048,576 parameters
With biases (+4 × 512): 1,050,624 parameters
```

**5. Attention Computation:**
```
Attention scores per head: [seq_len, seq_len] = [100, 100]
All heads: [h, seq_len, seq_len] = [8, 100, 100] = 80,000 scores

FLOPs for attention (per head):
- Q @ K^T: 100 × 64 × 100 = 640,000
- Softmax: ~3 × 100 × 100 = 30,000
- Weights @ V: 100 × 100 × 64 = 640,000
- Per head total: ~1.3M FLOPs

All 8 heads: ~10.4M FLOPs for attention only
(Excludes projection FLOPs: 4 × 100 × 512 × 512 ≈ 100M)
```

**Summary Table:**

| Metric | Value |
|--------|-------|
| d_k = d_v | 64 |
| W_Q, W_K, W_V shape | [512, 512] each |
| W_O shape | [512, 512] |
| Total parameters | ~1.05M |
| Attention scores | 80,000 (8 × 100 × 100) |

---

**Common Mistakes:**
| Mistake | Why It's Wrong | Prevention |
|---------|---------------|------------|
| d_k = d_model | Should be d_model/h | Remember: heads split the dimension |
| Forgetting W_O | Output projection is part of multi-head | Count all 4 weight matrices |
| Wrong FLOPs for matmul | [m,k] × [k,n] = m×k×n FLOPs | Track dimensions carefully |

---

## Problem 2: Skill-Builder
### Architecture Selection

**Concept:** Model Variants and Architecture Selection (Core Concept 6)
**Cognitive Level:** Analyze
**Prerequisites:** Understanding encoder, decoder, encoder-decoder variants

---

**Problem Statement:**

A machine learning team is building several NLP applications. For each application below, analyze the requirements and recommend the optimal transformer architecture (encoder-only, decoder-only, or encoder-decoder). Provide technical justification.

**Applications:**

| App | Description | Input | Output |
|-----|-------------|-------|--------|
| A | Email spam classifier | Full email text | Binary: spam/not spam |
| B | Customer support chatbot | User message history | Response message |
| C | Meeting transcript summarizer | 1-hour meeting transcript | 3-paragraph summary |
| D | SQL query generator | Natural language question + schema | SQL query |
| E | Semantic search engine | Search query | Ranked document IDs |

**For each application, specify:**
1. Recommended architecture
2. Pre-training objective best suited
3. Key technical reason for your choice
4. Any architectural modifications needed

---

**Progressive Hints:**

<details>
<summary>Hint 1 (Conceptual)</summary>

Decision framework:
- Encoder-only: Full bidirectional context needed; output is classification/embedding
- Decoder-only: Output is generated text; autoregressive generation natural
- Encoder-decoder: Input and output are both sequences; need to transform one to another
</details>

<details>
<summary>Hint 2 (Procedural)</summary>

For each app, ask:
1. Is generation required? (Yes → decoder involved)
2. Is full input context needed? (Yes → encoder helps)
3. Are input and output structurally different? (Yes → encoder-decoder)
4. Is the output a label/embedding? (Yes → encoder-only)
</details>

<details>
<summary>Hint 3 (Solution Direction)</summary>

Quick mapping:
- A (Classification) → Encoder
- B (Chatbot) → Decoder
- C (Summarization) → Either (enc-dec traditional, decoder modern)
- D (SQL generation) → Decoder or Enc-Dec
- E (Search) → Encoder (for embeddings)
</details>

---

**Solution:**

**Application A: Email Spam Classifier**

| Aspect | Decision |
|--------|----------|
| **Architecture** | Encoder-only (BERT/RoBERTa) |
| **Pre-training** | Masked Language Modeling (MLM) |
| **Key Reason** | Classification requires understanding full email; no generation needed |
| **Modifications** | Add classification head on [CLS] token; fine-tune on spam dataset |

**Technical Justification:**
- Bidirectional attention sees entire email context
- Spammers hide intent throughout message—beginning, middle, end all matter
- Output is binary label, not generated text
- [CLS] token aggregates document-level representation

---

**Application B: Customer Support Chatbot**

| Aspect | Decision |
|--------|----------|
| **Architecture** | Decoder-only (GPT/Llama) |
| **Pre-training** | Causal Language Modeling (CLM) |
| **Key Reason** | Response generation is autoregressive; chat history fits prompt paradigm |
| **Modifications** | Fine-tune on support conversations; add retrieval for knowledge base |

**Technical Justification:**
- Chat response is generated token-by-token (autoregressive)
- History + current message format naturally fits decoder prompting
- Decoder-only scales better and has richer pre-trained options
- Modern decoder models excel at instruction-following after SFT

---

**Application C: Meeting Transcript Summarizer**

| Aspect | Decision |
|--------|----------|
| **Architecture** | Encoder-decoder (T5/BART) OR Decoder-only (Llama) |
| **Pre-training** | Span corruption (T5) OR CLM (decoder) |
| **Key Reason** | Transform long input to condensed output; enc-dec natural but decoder viable |
| **Modifications** | Handle long context (chunking or long-context model); abstractive generation |

**Technical Justification:**

| Architecture | Pros | Cons |
|--------------|------|------|
| Encoder-decoder | Natural for seq2seq; encoder captures full doc | Two models; potentially complex |
| Decoder-only | Single model; strong generation | Must handle long context in prompt |

**Recommendation:** Encoder-decoder (BART) if doc fits context; Decoder-only (Llama-long) if need 100K+ context with modern long-context models.

---

**Application D: SQL Query Generator**

| Aspect | Decision |
|--------|----------|
| **Architecture** | Decoder-only (CodeLlama/GPT-4) |
| **Pre-training** | Causal LM on code + SQL |
| **Key Reason** | SQL is code; code models excel; generation is natural fit |
| **Modifications** | Include schema in prompt; use constrained decoding for valid SQL |

**Technical Justification:**
- SQL query is generated (autoregressive output)
- Code-pretrained decoders understand SQL syntax
- Schema + question → query fits prompt paradigm
- Can use grammar-constrained decoding for validity

Alternative: Encoder-decoder if you want explicit separation of understanding (encoder on question+schema) and generation (decoder for SQL). But modern decoder-only code models are typically stronger.

---

**Application E: Semantic Search Engine**

| Aspect | Decision |
|--------|----------|
| **Architecture** | Encoder-only (BERT/E5/BGE) |
| **Pre-training** | MLM + Contrastive learning (for retrieval) |
| **Key Reason** | Need dense embeddings for similarity search; no generation |
| **Modifications** | Use retrieval-tuned encoder (E5, BGE); mean pooling for doc embedding |

**Technical Justification:**
- Output is embedding vector, not generated text
- Documents encoded offline; query encoded at runtime
- Similarity = dot product of embeddings
- Encoder bidirectionally understands full query meaning
- Specialized retrieval models outperform general BERT

---

**Summary Decision Matrix:**

| App | Architecture | Pre-training | Key Factor |
|-----|--------------|--------------|------------|
| A | Encoder | MLM | Classification, no generation |
| B | Decoder | CLM | Autoregressive response generation |
| C | Enc-Dec or Decoder | Span/CLM | Seq2seq transformation |
| D | Decoder | Code CLM | SQL generation, code domain |
| E | Encoder | MLM + Contrastive | Embedding for similarity |

---

**Common Mistakes:**
| Mistake | Why It's Wrong | Prevention |
|---------|---------------|------------|
| Decoder for classification | Decoder's autoregressive nature is overhead for classification | Use encoder when output is label/embedding |
| Encoder for generation | Encoder cannot generate; produces representations | Use decoder when output is generated text |
| Always enc-dec for seq2seq | Modern decoder-only often matches or beats for many seq2seq tasks | Consider decoder-only for simpler pipeline |

---

## Problem 3: Skill-Builder
### Positional Encoding Implementation

**Concept:** Positional Encoding (Core Concept 3)
**Cognitive Level:** Apply
**Prerequisites:** Understanding why position encoding is needed

---

**Problem Statement:**

Implement sinusoidal positional encoding from scratch. Then analyze its properties and compare to alternatives.

**Part A: Implementation**

Write Python code to generate sinusoidal positional encodings for:
- max_length: 1000
- d_model: 512

Your implementation should:
1. Generate the PE matrix with the standard formula
2. Plot the encodings for positions 0-50 and dimensions 0-8
3. Verify that PE[pos+k] can be represented as a linear function of PE[pos]

**Part B: Analysis**

Answer the following:
1. Why do we use sin for even dimensions and cos for odd dimensions?
2. Why does the frequency decrease as dimension index increases?
3. What is the maximum sequence position this encoding can theoretically distinguish?
4. When would you choose learned embeddings over sinusoidal?

---

**Progressive Hints:**

<details>
<summary>Hint 1 (Conceptual)</summary>

The sinusoidal formula:
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

The 10000^(2i/d_model) term creates wavelengths from 2π to 10000×2π.
</details>

<details>
<summary>Hint 2 (Procedural)</summary>

Implementation outline:
```python
position = np.arange(max_length)[:, np.newaxis]  # [max_length, 1]
dim = np.arange(d_model)[np.newaxis, :]          # [1, d_model]
angle = position / (10000 ** (2 * (dim // 2) / d_model))
PE[:, 0::2] = np.sin(angle[:, 0::2])
PE[:, 1::2] = np.cos(angle[:, 1::2])
```
</details>

<details>
<summary>Hint 3 (Solution Direction)</summary>

Linear relationship: For any k, PE[pos+k] = PE[pos] @ M_k for some matrix M_k

This is because:
sin(a+b) = sin(a)cos(b) + cos(a)sin(b)
cos(a+b) = cos(a)cos(b) - sin(a)sin(b)

So relative positions can be computed from absolute positions via linear transformation.
</details>

---

**Solution:**

**Part A: Implementation**

```python
import numpy as np
import matplotlib.pyplot as plt

def sinusoidal_positional_encoding(max_length, d_model):
    """
    Generate sinusoidal positional encodings.

    Formula:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    # Create position and dimension indices
    position = np.arange(max_length)[:, np.newaxis]  # Shape: [max_length, 1]
    dim_index = np.arange(d_model)[np.newaxis, :]     # Shape: [1, d_model]

    # Calculate the angle for each position and dimension
    # Use (dim_index // 2) to pair even/odd dimensions
    angle_rates = 1 / (10000 ** (2 * (dim_index // 2) / d_model))
    angles = position * angle_rates  # Shape: [max_length, d_model]

    # Apply sin to even indices, cos to odd indices
    pe = np.zeros((max_length, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])  # Even dimensions
    pe[:, 1::2] = np.cos(angles[:, 1::2])  # Odd dimensions

    return pe

# Generate encodings
max_length = 1000
d_model = 512
PE = sinusoidal_positional_encoding(max_length, d_model)

print(f"PE shape: {PE.shape}")  # [1000, 512]
print(f"PE[0, :4]: {PE[0, :4]}")  # [0, 1, 0, 1] (sin(0), cos(0), ...)

# Plot first 50 positions, first 8 dimensions
plt.figure(figsize=(12, 6))
plt.imshow(PE[:50, :8], aspect='auto', cmap='RdBu')
plt.xlabel('Dimension')
plt.ylabel('Position')
plt.colorbar(label='Encoding Value')
plt.title('Sinusoidal Positional Encodings')
plt.show()
```

**Verifying Linear Relationship:**

```python
def verify_linear_relationship(PE, pos, k):
    """
    Verify that PE[pos+k] can be expressed as linear transformation of PE[pos].

    For sin/cos pairs at dimension 2i:
    sin(pos+k) = sin(pos)cos(k) + cos(pos)sin(k)
    cos(pos+k) = cos(pos)cos(k) - sin(pos)sin(k)

    This is a linear transformation!
    """
    # For each pair of dimensions (sin, cos), compute transformation matrix
    d_model = PE.shape[1]

    reconstructed = np.zeros(d_model)
    for i in range(0, d_model, 2):
        # Get frequency for this dimension pair
        freq = 1 / (10000 ** (i / d_model))

        # Rotation matrix for relative position k
        cos_k = np.cos(k * freq)
        sin_k = np.sin(k * freq)

        # Apply rotation to (sin(pos*f), cos(pos*f))
        sin_pos = PE[pos, i]
        cos_pos = PE[pos, i + 1]

        reconstructed[i] = sin_pos * cos_k + cos_pos * sin_k      # sin(pos+k)
        reconstructed[i + 1] = cos_pos * cos_k - sin_pos * sin_k  # cos(pos+k)

    # Compare with actual PE[pos+k]
    actual = PE[pos + k]
    error = np.max(np.abs(reconstructed - actual))

    return error < 1e-10  # Should be numerically identical

# Test
print(f"Linear relationship holds for k=1: {verify_linear_relationship(PE, 10, 1)}")
print(f"Linear relationship holds for k=50: {verify_linear_relationship(PE, 10, 50)}")
print(f"Linear relationship holds for k=100: {verify_linear_relationship(PE, 10, 100)}")
# All should print True
```

**Part B: Analysis**

**1. Why sin for even, cos for odd?**

| Reason | Explanation |
|--------|-------------|
| **Orthogonal basis** | sin and cos are orthogonal; provide complementary information |
| **Linear relative position** | sin(a+b) and cos(a+b) can be computed from sin(a), cos(a) via linear transformation |
| **Unique encoding** | Each position has unique (sin, cos) pair at each frequency |

Without both, we couldn't compute relative positions as linear functions.

**2. Why decreasing frequency with dimension?**

| Dimension Range | Frequency | Wavelength | What It Captures |
|-----------------|-----------|------------|------------------|
| Low (0-50) | High | Short (2π to ~100) | Fine-grained position differences |
| Middle (50-400) | Medium | Medium | Moderate-range patterns |
| High (400-512) | Low | Long (~10000×2π) | Long-range position patterns |

Different frequency bands capture position information at different scales, similar to how Fourier transforms decompose signals.

**3. Maximum distinguishable position:**

```
Maximum wavelength: 10000 × 2π ≈ 62,832

Theoretical max position: ~62,832 before lowest frequency completes a cycle

In practice: Performance degrades before this limit due to:
- Limited precision in attention
- Training only on shorter sequences
- High-frequency dimensions aliasing
```

**4. When to choose learned over sinusoidal:**

| Choose Learned When | Reason |
|---------------------|--------|
| Fixed maximum length | No extrapolation needed |
| Rich positional patterns | Learned can capture task-specific patterns |
| Fine-tuning expected | Positions can adapt to domain |

| Choose Sinusoidal When | Reason |
|------------------------|--------|
| Variable/long sequences | Generalizes to unseen lengths |
| No task-specific position needs | Generic position is sufficient |
| Want no additional parameters | Sinusoidal is parameter-free |

**Modern Recommendation:** Use RoPE or ALiBi instead of either—they combine benefits.

---

**Common Mistakes:**
| Mistake | Why It's Wrong | Prevention |
|---------|---------------|------------|
| Using 2i instead of 2i//2 | Creates wrong frequency pattern | dim_index // 2 pairs dimensions |
| Forgetting 10000 base | Frequencies will be wrong | Double-check the formula |
| Adding PE after other layers | PE should be added to input embeddings | Add PE immediately to token embeddings |

---

## Problem 4: Challenge
### Efficient Attention Design

**Concept:** Efficiency and Scaling (Core Concept 7)
**Cognitive Level:** Synthesize
**Prerequisites:** Full understanding of attention mechanism and complexity

---

**Problem Statement:**

A healthcare company needs to process medical records averaging 75,000 tokens each. Standard attention (O(n²)) is infeasible:
- 75,000² = 5.6 billion attention scores per layer
- Memory: ~22GB per layer at FP32 just for attention matrix

**Design an efficient attention mechanism that:**
1. Achieves sub-quadratic complexity (target: O(n·k) where k << n)
2. Maintains quality for both local patterns (nearby symptoms) and global patterns (diagnosis-treatment relationships across document)
3. Works with existing pre-trained transformer weights (no full retraining)
4. Can be incrementally applied as documents grow (for streaming scenarios)

**Your design must include:**

1. **Attention Pattern Design:** Specify which positions attend to which
2. **Complexity Analysis:** Prove your complexity claim
3. **Quality Preservation:** How do you handle long-range dependencies?
4. **Implementation Strategy:** How to adapt pre-trained models?
5. **Streaming Extension:** How to handle documents that grow incrementally?

---

**Progressive Hints:**

<details>
<summary>Hint 1 (Conceptual)</summary>

Successful efficient attention patterns combine:
- **Local attention:** Each position attends to ±w neighbors (O(n·w))
- **Global tokens:** Selected positions attend to/from all positions
- **Stride/dilated patterns:** Periodic long-range connections

Look at: Longformer, BigBird, or sliding window approaches.
</details>

<details>
<summary>Hint 2 (Procedural)</summary>

Design approach:
1. Local window (w=512): Captures nearby context
2. Global tokens: [CLS], section headers, diagnosis codes
3. Stride pattern: Every 1000th token attends globally

Complexity: n×w + n×g + n×(n/s) where g=global tokens, s=stride
For w=512, g=100, s=1000: O(n×512 + n×100 + n×75) ≈ O(n×700) = O(n·k)
</details>

<details>
<summary>Hint 3 (Solution Direction)</summary>

For pre-trained compatibility:
- Initialize sparse attention with pre-trained weights
- Add position interpolation for longer contexts
- Fine-tune on medical data with sparse pattern

For streaming:
- KV cache for seen tokens
- Only compute attention for new token against cached KVs
- Periodic full recomputation for global tokens
</details>

---

**Solution:**

**1. Attention Pattern Design: Hierarchical Sparse Attention**

```
┌─────────────────────────────────────────────────────────────────┐
│                    ATTENTION PATTERN DESIGN                     │
│                                                                 │
│  Pattern: LOCAL + GLOBAL + HIERARCHICAL                         │
└─────────────────────────────────────────────────────────────────┘

COMPONENT 1: Local Sliding Window
─────────────────────────────────
- Each position attends to positions [i-w, i+w]
- Window size w = 512 tokens
- Captures: Nearby symptoms, local context, sentence structure

Visualization (simplified, w=3):
Position:  0  1  2  3  4  5  6  7  8  9
Token 4:   .  [  X  X  X  X  X  ]  .  .
Token 7:   .  .  .  .  [  X  X  X  X  X  ]

COMPONENT 2: Global Tokens
──────────────────────────
- Selected tokens attend to ALL positions and are attended BY all
- Global tokens include:
  * [CLS] token (position 0)
  * Section headers (detected by pattern matching)
  * Medical entity mentions (diagnoses, medications)
  * Every 1000th position (periodic global)

Typical global token count: ~100-200 for 75K document

COMPONENT 3: Hierarchical Long-Range
────────────────────────────────────
- Beyond local window: Sparse stride pattern
- Attend to positions at fixed strides: {i+1024, i+2048, i+4096, ...}
- Captures long-range dependencies with logarithmic connections

Combined Pattern for position i:
- Local: [max(0, i-512), min(n, i+512)]
- Global: All global tokens (bidirectional)
- Hierarchical: {i+1024, i+2048, i+4096, i-1024, i-2048, i-4096}
```

**Attention Mask Visualization:**
```
          Position
          0    1K   2K   3K   4K   ...  75K
     0   [G]   G    G    G    G         G      ← Global token (attends all)
Position 1K   G   [L]   L    H              ← Local window + hierarchical
     2K   G    L   [L]   L    H
     3K   G    H    L   [L]   L    H
     ...
    75K   G                   H   L [L]

Legend:
G = Global connection
L = Local window connection
H = Hierarchical stride connection
[ ] = Current position
```

**2. Complexity Analysis**

```
For sequence length n = 75,000:

Local Attention:
- Each position: 2×w = 1,024 connections
- Total: n × 2w = 75,000 × 1,024 ≈ 77M connections

Global Tokens:
- g = 150 global tokens (estimated)
- Each global attends to n: g × n = 150 × 75,000 = 11.25M
- Each position attends to g globals: n × g = 75,000 × 150 = 11.25M
- Total: 22.5M connections

Hierarchical:
- log₂(n) stride levels ≈ 16 levels
- Each position: 2 × 16 = 32 connections
- Total: n × 32 = 2.4M connections

TOTAL: 77M + 22.5M + 2.4M ≈ 102M connections

COMPARISON:
- Full attention: n² = 5.6 billion connections
- Our approach: ~102M connections
- Reduction: 55× less computation

COMPLEXITY: O(n × (w + g + log(n))) = O(n × k)
where k = w + g + log(n) ≈ 1,024 + 150 + 16 ≈ 1,200
```

**3. Quality Preservation for Long-Range Dependencies**

| Challenge | Solution | Why It Works |
|-----------|----------|--------------|
| **Cross-document references** | Hierarchical stride ensures O(log n) path between any positions | Information can flow in log(n) layers |
| **Important entities** | Global tokens for medical entities | Diagnoses, meds always accessible |
| **Section relationships** | Section headers as global | Enables "diagnosis in section A relates to treatment in section B" |
| **Fine-grained local** | 512-token window | Symptom descriptions stay connected |

**Information Flow Analysis:**
```
For positions p1=1000, p2=60000 (59K apart):

With full attention: Direct connection (1 hop)

With our pattern:
- p1 → Global token (1 hop)
- Global → p2 (1 hop)
- OR: p1 → hierarchical stride → ... → p2 (log(59K) ≈ 16 hops)

Effective: Any position reaches any other in O(log n) hops
```

**4. Implementation Strategy for Pre-trained Models**

```python
class EfficientMedicalAttention(nn.Module):
    """
    Adapter that converts pre-trained attention to efficient sparse pattern.
    Preserves pre-trained weights while enabling longer context.
    """

    def __init__(self, pretrained_attention, config):
        super().__init__()
        # Reuse pre-trained Q, K, V, O projections
        self.W_Q = pretrained_attention.W_Q  # Keep weights
        self.W_K = pretrained_attention.W_K
        self.W_V = pretrained_attention.W_V
        self.W_O = pretrained_attention.W_O

        # New: Global token indicators (learned during fine-tuning)
        self.global_token_predictor = nn.Linear(config.d_model, 1)

        # Configuration
        self.window_size = 512
        self.hierarchical_strides = [1024, 2048, 4096, 8192]
        self.max_global_tokens = 200

    def compute_attention_mask(self, seq_len, global_positions):
        """Generate sparse attention mask."""
        # Start with local window mask
        mask = self._local_window_mask(seq_len, self.window_size)

        # Add global token connections
        mask = self._add_global_connections(mask, global_positions)

        # Add hierarchical stride connections
        mask = self._add_hierarchical_connections(mask, self.hierarchical_strides)

        return mask  # Shape: [seq_len, seq_len], sparse

    def forward(self, x, attention_mask=None):
        # Identify global tokens (section headers, entities, periodic)
        global_scores = self.global_token_predictor(x).squeeze(-1)
        global_positions = self._select_global_tokens(global_scores)

        # Generate efficient attention mask
        efficient_mask = self.compute_attention_mask(x.size(1), global_positions)

        # Standard Q, K, V computation (uses pre-trained weights)
        Q = x @ self.W_Q
        K = x @ self.W_K
        V = x @ self.W_V

        # Sparse attention computation (only compute non-masked positions)
        output = self._sparse_attention(Q, K, V, efficient_mask)

        return output @ self.W_O
```

**Fine-tuning Strategy:**
```
Phase 1: Position Interpolation
- Extend RoPE/ALiBi to 75K positions
- Interpolate between trained positions

Phase 2: Sparse Pattern Adaptation
- Train global token predictor on medical data
- Keep Q/K/V/O weights frozen initially

Phase 3: Full Fine-tuning
- Unfreeze all weights
- Train on medical record tasks
- Use efficient attention throughout
```

**5. Streaming Extension**

```
STREAMING SCENARIO: Document arrives token-by-token or in chunks

┌─────────────────────────────────────────────────────────────────┐
│                    STREAMING ATTENTION                          │
└─────────────────────────────────────────────────────────────────┘

KV CACHE STRUCTURE:
─────────────────────
1. Full cache for recent window: [current_pos - w : current_pos]
2. Compressed cache for global tokens: Selected KVs always stored
3. Hierarchical cache: KVs at stride positions only

Cache management:
```python
class StreamingEfficientAttention:
    def __init__(self, window_size=512):
        self.window_size = window_size
        self.kv_cache = {
            'local': [],      # Recent window (circular buffer)
            'global': [],     # Global token KVs (persistent)
            'hierarchical': {}  # Stride positions (dict by stride)
        }
        self.current_position = 0

    def process_new_tokens(self, new_tokens):
        """Process incoming tokens incrementally."""
        for token in new_tokens:
            # Compute Q, K, V for new token
            q, k, v = self.compute_qkv(token)

            # Determine what this token attends to
            attend_positions = self._get_attend_positions(self.current_position)

            # Gather relevant KVs from cache
            relevant_kvs = self._gather_cached_kvs(attend_positions)

            # Compute attention output
            output = self._attention(q, relevant_kvs['k'], relevant_kvs['v'])

            # Update caches
            self._update_local_cache(k, v)
            if self._is_global_token(token, self.current_position):
                self._update_global_cache(k, v, self.current_position)
            if self.current_position % 1024 == 0:  # Hierarchical stride
                self._update_hierarchical_cache(k, v, self.current_position)

            self.current_position += 1

            yield output

    def _get_attend_positions(self, pos):
        """Get all positions current position should attend to."""
        positions = set()

        # Local window
        for i in range(max(0, pos - self.window_size), pos):
            positions.add(i)

        # Global tokens (always attend)
        positions.update(self.global_token_positions)

        # Hierarchical (logarithmic strides back)
        for stride in [1024, 2048, 4096, 8192]:
            if pos - stride >= 0:
                positions.add((pos - stride) // stride * stride)

        return positions
```

**Memory Analysis for Streaming:**
```
Window cache: w × d × 2 = 512 × 512 × 2 = 0.5MB per layer
Global cache: g × d × 2 = 200 × 512 × 2 = 0.2MB per layer
Hierarchical cache: log(n) × d × 2 ≈ 16 × 512 × 2 = 0.016MB per layer

Total per layer: ~0.7MB
For 32 layers: ~22MB (vs 22GB for full attention)

Reduction: 1000× less memory for streaming
```

---

**Common Mistakes:**
| Mistake | Why It's Wrong | Prevention |
|---------|---------------|------------|
| Only local attention | Loses all long-range dependencies | Add global tokens and hierarchical |
| Too many global tokens | Complexity becomes O(n²) if g = O(n) | Cap global tokens (e.g., 200) |
| Ignoring cache management | Memory grows unbounded in streaming | Use circular buffers, eviction |

---

## Problem 5: Debug/Fix
### Transformer Training Issues

**Concept:** Architecture Components (Concepts 4-6)
**Cognitive Level:** Analyze
**Prerequisites:** Understanding of transformer training dynamics

---

**Problem Statement:**

A team is training a transformer model and encountering several issues. Analyze each issue and provide the fix.

**Issue 1: Exploding Loss**
```
Training log:
Epoch 1, Step 100: Loss = 4.2
Epoch 1, Step 200: Loss = 8.7
Epoch 1, Step 300: Loss = 156.3
Epoch 1, Step 400: Loss = NaN

Model config:
- d_model: 1024
- d_k: 1024 (full dimension, single head)
- Learning rate: 1e-3
- No gradient clipping
```

**Issue 2: Poor Generation Quality**
```
Symptom: Model generates repetitive text
"The cat sat on the mat. The cat sat on the mat. The cat sat on the mat..."

Model config:
- Architecture: Decoder-only
- Positional encoding: Learned (max_length=512)
- Generation: Greedy decoding (temperature=0)
- Training data: Properly shuffled
```

**Issue 3: Slow Convergence**
```
Training log:
Epoch 1: Loss = 5.2
Epoch 10: Loss = 5.1
Epoch 50: Loss = 4.9
Epoch 100: Loss = 4.8
(Baseline model converges to 2.1 by epoch 100)

Model config:
- Layer normalization: Post-LN (original architecture)
- Initialization: Xavier uniform for all weights
- Residual connections: Present
- Layers: 24
```

**For each issue:**
1. Diagnose the root cause
2. Explain why this causes the observed behavior
3. Provide the specific fix
4. Explain how to prevent this in future

---

**Progressive Hints:**

<details>
<summary>Hint 1 (Conceptual)</summary>

Issue 1: Think about what happens to attention scores when d_k is large and unscaled.

Issue 2: Greedy decoding + certain training patterns = repetition loops.

Issue 3: Post-LN in deep networks has known gradient flow issues.
</details>

<details>
<summary>Hint 2 (Procedural)</summary>

Issue 1:
- Check the attention formula: softmax(QK^T / √d_k)
- With d_k=1024, √d_k ≈ 32
- If not scaling, dot products grow with d_k

Issue 2:
- Greedy always picks highest probability
- If "The cat" is common, it self-reinforces

Issue 3:
- Post-LN: output = LN(x + sublayer(x))
- Gradient must flow through LN after residual
- Deep networks: gradient diminishes
</details>

<details>
<summary>Hint 3 (Solution Direction)</summary>

Issue 1: Add √d_k scaling to attention

Issue 2: Use sampling with temperature or nucleus sampling (top-p)

Issue 3: Switch to Pre-LN: output = x + sublayer(LN(x))
</details>

---

**Solution:**

**Issue 1: Exploding Loss**

**Diagnosis:** Missing √d_k scaling in attention computation

**Root Cause Analysis:**
```
With d_k = 1024 (no scaling):

Q and K vectors have variance ~1 (from initialization)
Dot product Q·K has variance ~d_k = 1024
For d_k = 1024, dot products are ~32x larger than expected

Effect on softmax:
- Large dot products → softmax approaches one-hot
- Gradients of softmax vanish at extremes
- Some positions get near-zero gradient
- Other positions get exploding gradients
- Loss becomes unstable → NaN
```

**Fix:**
```python
# BEFORE (broken):
attention_scores = torch.matmul(Q, K.transpose(-2, -1))
attention_weights = F.softmax(attention_scores, dim=-1)

# AFTER (fixed):
d_k = Q.size(-1)
attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
attention_weights = F.softmax(attention_scores, dim=-1)
```

**Additional Safeguards:**
1. Use gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
2. Use lower learning rate for attention: Start with 1e-4 or use warmup
3. Consider multi-head attention (d_k = d_model/h is smaller)

**Prevention:**
- Always verify attention implementation includes scaling
- Use established attention implementations (nn.MultiheadAttention)
- Monitor gradient norms during training

---

**Issue 2: Poor Generation Quality (Repetition)**

**Diagnosis:** Greedy decoding creates repetition loops when high-probability sequences self-reinforce

**Root Cause Analysis:**
```
Greedy decoding: Always select argmax(P(next_token | context))

Problem scenario:
1. "The cat sat" → highest prob next: "on" (0.4)
2. "The cat sat on" → highest prob: "the" (0.3)
3. "The cat sat on the" → highest prob: "mat" (0.35)
4. "...on the mat." → highest prob: "The" (0.25)  ← Loop starts
5. "mat. The" → highest prob: "cat" (0.4)
6. ...repeats indefinitely

Why this happens:
- Training data has common phrases
- Greedy decoding has no diversity mechanism
- Once in a loop, highest probability stays in loop
- Model is not "wrong"—it's doing exactly what greedy asks
```

**Fix:**
```python
# BEFORE (greedy, causes repetition):
def generate(model, prompt, max_length):
    for _ in range(max_length):
        logits = model(prompt)
        next_token = logits[-1].argmax()  # Always picks highest
        prompt = torch.cat([prompt, next_token.unsqueeze(0)])
    return prompt

# AFTER (nucleus sampling with temperature):
def generate(model, prompt, max_length, temperature=0.8, top_p=0.9):
    for _ in range(max_length):
        logits = model(prompt)
        logits = logits[-1] / temperature  # Add diversity

        # Nucleus (top-p) sampling
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        prompt = torch.cat([prompt, next_token])
    return prompt
```

**Additional Repetition Prevention:**
```python
# Repetition penalty
def apply_repetition_penalty(logits, generated_tokens, penalty=1.2):
    for token_id in set(generated_tokens):
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty
    return logits
```

**Prevention:**
- Never use greedy (temperature=0) for long-form generation
- Standard settings: temperature=0.7-1.0, top_p=0.9
- Add repetition penalty for production systems
- Consider beam search with n-gram blocking for critical applications

---

**Issue 3: Slow Convergence**

**Diagnosis:** Post-LN architecture with deep network (24 layers) causes gradient flow problems

**Root Cause Analysis:**
```
Post-LN architecture (original Transformer):
output = LayerNorm(x + sublayer(x))

Gradient flow for N=24 layers:
∂Loss/∂layer_1 = ∂Loss/∂output × ∂output/∂layer_24 × ... × ∂layer_2/∂layer_1

Problem:
- Each LayerNorm transformation affects gradient magnitude
- With Post-LN, gradient must pass through LN AFTER residual
- Residual gradient (identity) gets normalized/scaled
- After 24 layers, early layer gradients are diminished

Result:
- Early layers get small gradients
- Updates are tiny
- Convergence is slow
- Deep networks may not train at all
```

**Fix: Switch to Pre-LN**
```python
# BEFORE (Post-LN, slow convergence):
class PostLNTransformerLayer(nn.Module):
    def forward(self, x):
        # Attention sublayer
        attention_out = self.attention(x)
        x = self.norm1(x + attention_out)  # LN after residual

        # FFN sublayer
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)  # LN after residual
        return x

# AFTER (Pre-LN, better gradient flow):
class PreLNTransformerLayer(nn.Module):
    def forward(self, x):
        # Attention sublayer
        attention_out = self.attention(self.norm1(x))  # LN before sublayer
        x = x + attention_out  # Clean residual

        # FFN sublayer
        ffn_out = self.ffn(self.norm2(x))  # LN before sublayer
        x = x + ffn_out  # Clean residual
        return x

# Note: Final layer norm needed after Pre-LN stack
class PreLNTransformer(nn.Module):
    def __init__(self, layers, d_model):
        self.layers = nn.ModuleList(layers)
        self.final_norm = nn.LayerNorm(d_model)  # Important!

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)  # Normalize final output
```

**Why Pre-LN Works Better:**
```
Pre-LN gradient flow:
∂Loss/∂layer_k = ∂Loss/∂output × (1 + ∂sublayer/∂input)

- The "1" from residual provides direct gradient path
- Gradient flows through identity without transformation
- LN normalization happens inside sublayer, not on gradient path
- Early layers receive stronger gradients
- Deep networks (24+ layers) train stably
```

**Additional Improvements:**
```python
# Better initialization for deep networks
def init_weights(module, n_layers):
    if isinstance(module, nn.Linear):
        # Scale down residual path initialization
        std = 0.02 / math.sqrt(2 * n_layers)
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
```

**Prevention:**
- Use Pre-LN for any transformer with >6 layers
- Pre-LN is now the default in modern architectures (GPT-2+, Llama, etc.)
- If using Post-LN, reduce depth or use careful initialization
- Monitor gradient norms per layer during training

---

**Summary of Issues and Fixes:**

| Issue | Root Cause | Fix | Prevention |
|-------|------------|-----|------------|
| Exploding Loss | Missing √d_k scaling | Add scaling to attention | Use standard attention implementation |
| Repetitive Generation | Greedy decoding loops | Use sampling (temp, top-p) | Never greedy for long generation |
| Slow Convergence | Post-LN gradient issues | Switch to Pre-LN | Use Pre-LN for deep networks |

---

## Self-Assessment Guide

### Mastery Checklist

| Problem | Mastery Indicator | Check |
|---------|-------------------|-------|
| **1 (Warm-Up)** | Can calculate MHA dimensions without reference | ☐ |
| **2 (Skill-Builder)** | Can justify architecture choice for any task | ☐ |
| **3 (Skill-Builder)** | Can implement positional encoding from scratch | ☐ |
| **4 (Challenge)** | Can design efficient attention for long sequences | ☐ |
| **5 (Debug/Fix)** | Can diagnose and fix transformer training issues | ☐ |

### Progression Path

```
If struggled with Problem 1:
  → Review: Multi-Head Attention section, dimension calculations
  → Flashcard: Card 1-2 (Easy)

If struggled with Problem 2:
  → Review: Model Variants section
  → Flashcard: Card 4 (Medium)

If struggled with Problem 3:
  → Review: Positional Encoding section
  → Flashcard: Card 3 (Medium)

If struggled with Problem 4:
  → Review: Efficiency and Scaling section
  → Flashcard: Card 5 (Hard)

If struggled with Problem 5:
  → Review: Encoder/Decoder Architecture sections
  → All Flashcards for comprehensive review
```

---

## Extension Challenges

### For Problem 1:
Calculate the memory requirements for storing KV cache during generation for a 13B parameter model with 40 layers, 32 heads, 4096 context length, at FP16 precision.

### For Problem 2:
Design a hybrid architecture that uses encoder for document processing and decoder for query-conditioned generation. What are the interface points?

### For Problem 3:
Implement RoPE (Rotary Position Embedding) from scratch. Show mathematically why it encodes relative positions.

### For Problem 4:
Extend your efficient attention design to support cross-document attention for retrieval-augmented generation where you need to attend across 10 retrieved documents.

### For Problem 5:
Design a diagnostic dashboard that would detect all three issues (exploding loss, repetition, slow convergence) automatically during training. What metrics would you monitor?

---

*Generated from Lesson 4: Transformers | Practice Problems Skill*
