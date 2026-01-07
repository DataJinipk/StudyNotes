# Flashcards: Transformers

**Source:** notes/transformers/transformers-study-notes.md
**Concept Map:** notes/transformers/concept-maps/transformers-concept-map.md
**Date Generated:** 2026-01-07
**Total Cards:** 5
**Distribution:** 2 Easy | 2 Medium | 1 Hard

---

## Easy Cards

### Card 1 of 5 | Easy
**Concept:** Self-Attention and Multi-Head Attention
**Centrality:** Critical (11 connections)

#### Front
What is self-attention in transformers, and why is scaling by sqrt(dk) necessary? How does multi-head attention extend this mechanism?

#### Back
**Self-Attention** computes a representation of each position by attending to all positions in the sequence:

1. **Compute Q, K, V:** Linear projections of input
2. **Attention scores:** score(i,j) = qi . kj / sqrt(dk)
3. **Softmax:** Normalize to get attention weights
4. **Output:** Weighted sum of values

**Why sqrt(dk) scaling:**
- Dot products grow with dimension dk
- Large values push softmax into saturation (very small gradients)
- Scaling keeps variance stable regardless of dimension

**Multi-Head Attention:**
- Run h parallel attention operations with different projections
- Each head: dk = d/h dimensions
- Concatenate and project: enables learning diverse attention patterns
- Same total computation as single large head

#### Mnemonic
**"QKV + Scale + Softmax = Attention"**

---

### Card 2 of 5 | Easy
**Concept:** Encoder vs Decoder Architecture
**Centrality:** High (7 connections each)

#### Front
Compare transformer encoder and decoder architectures. What is the key difference in attention masking, and when would you use each?

#### Back
| Aspect | Encoder | Decoder |
|--------|---------|---------|
| **Attention** | Bidirectional (all-to-all) | Causal (past only) |
| **Masking** | No mask | Upper triangular mask |
| **Context** | Sees full input | Only sees previous tokens |
| **Use case** | Understanding tasks | Generation tasks |

**Encoder (BERT-style):**
- Each position attends to all positions
- Ideal when full context is available
- Classification, NER, QA, embeddings

**Decoder (GPT-style):**
- Position i can only attend to positions <= i
- Required for autoregressive generation
- Text generation, language modeling

**Encoder-Decoder (T5-style):**
- Encoder processes input bidirectionally
- Decoder generates output with cross-attention to encoder
- Translation, summarization

#### Mnemonic
**"Encoder = Understanding, Decoder = Generating"**

---

## Medium Cards

### Card 3 of 5 | Medium
**Concept:** BERT vs GPT
**Centrality:** High (6 connections each)

#### Front
Compare BERT and GPT in terms of architecture, pre-training objective, and suitable applications. When would you choose one over the other?

#### Back
| Aspect | BERT | GPT |
|--------|------|-----|
| **Architecture** | Encoder-only | Decoder-only |
| **Attention** | Bidirectional | Causal (unidirectional) |
| **Pre-training** | Masked Language Modeling (15% tokens masked) | Causal Language Modeling (predict next token) |
| **Context** | Sees left AND right | Only sees left |
| **Output** | Contextual embeddings | Generated tokens |

**Choose BERT when:**
- Task has complete input available (classification, NER, QA)
- Need high-quality embeddings/representations
- Not generating new text

**Choose GPT when:**
- Need to generate text
- In-context learning / few-shot prompting
- Open-ended tasks, chatbots, code generation

**Key Insight:**
- BERT: Better at understanding (bidirectional = richer representations)
- GPT: Better at generation (autoregressive = coherent output)

#### Common Misconceptions
- GPT CAN do classification (via prompting), just not optimally designed for it
- BERT CANNOT generate text naturally (no causal structure)

---

### Card 4 of 5 | Medium
**Concept:** Positional Encoding
**Centrality:** High (5 connections)

#### Front
Why do transformers need positional encoding? Compare sinusoidal, learned, and rotary (RoPE) positional encodings.

#### Back
**Why Needed:**
Self-attention is permutation equivariant—shuffling inputs shuffles outputs identically. Without position info, "cat sat on mat" = "mat on sat cat" to the model.

**Sinusoidal (Original Transformer):**
```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```
- Fixed, no parameters
- Can extrapolate to longer sequences (in theory)
- Different frequencies capture different scales

**Learned Positions:**
- Trainable embedding per position
- More flexible, but limited to training length
- Cannot extrapolate beyond max trained length

**RoPE (Rotary Position Embedding):**
- Encodes relative positions in attention computation
- Rotates query and key vectors based on position
- Better length generalization
- Used in Llama, Mistral, modern LLMs

**ALiBi (Attention with Linear Biases):**
- Adds position-based bias to attention scores
- No position embeddings added to tokens
- Excellent extrapolation to longer sequences

#### Key Insight
Modern models (Llama, Mistral) use **RoPE** for better length generalization than absolute positions.

---

## Hard Cards

### Card 5 of 5 | Hard
**Concept:** Complete Transformer System Design
**Centrality:** Integration (spans all concepts)

#### Front
Design a transformer-based system for a document question-answering task where documents can be 100,000 tokens long. Address: (1) architecture choice, (2) handling long context, (3) training strategy, and (4) inference optimization.

#### Back
**1. Architecture Choice:**

For QA, we need both understanding (encode document) and generation (produce answer):
- **Option A:** Encoder-decoder (T5-style) — encode document, generate answer
- **Option B:** Decoder-only with retrieval (RAG) — chunk document, retrieve relevant parts
- **Recommendation:** RAG + Decoder-only for 100K tokens (pure attention impractical)

**2. Handling Long Context:**

| Approach | Method | Trade-off |
|----------|--------|-----------|
| **Chunking + Retrieval** | Split into 512-token chunks, embed with retriever, fetch top-k | Loses global context |
| **Sparse Attention** | Longformer/BigBird patterns | Reduced expressiveness |
| **Flash Attention** | Memory-efficient exact attention | Still O(n^2) compute |
| **Hierarchical** | Summarize chunks, attend to summaries | Information loss |

**Recommended Architecture:**
```
Document (100K) → Chunk (512 each) → Embed (retriever)
                                           ↓
Question → Retrieve top-10 chunks → Concat with question
                                           ↓
                              Decoder-only LLM → Answer
```

**3. Training Strategy:**

- **Retriever:** Contrastive learning on question-passage pairs
- **Reader:** Fine-tune decoder on QA pairs with retrieved context
- **End-to-end:** Joint training with retriever in the loop (more complex)

**4. Inference Optimization:**

| Technique | Benefit |
|-----------|---------|
| **KV Cache** | Reuse key/value computations for autoregressive generation |
| **Quantization** | INT8/INT4 reduces memory and speeds up |
| **Speculative Decoding** | Small model drafts, large model verifies |
| **Batching** | Process multiple questions on same document |

**Production Architecture:**
```
┌─────────────────────────────────────────────────────────┐
│                   QA System                              │
├─────────────────────────────────────────────────────────┤
│  Document → Chunker → Embedder → Vector DB              │
│                                      ↓                   │
│  Question → Embedder ────────────→ Retriever            │
│                                      ↓                   │
│                              Top-K Chunks                │
│                                      ↓                   │
│  [Question + Chunks] ───────────→ LLM (Llama 70B)       │
│                                      ↓                   │
│                                   Answer                 │
└─────────────────────────────────────────────────────────┘
```

#### Key Trade-offs
- Pure 100K attention: Possible with Flash Attention but expensive
- RAG approach: More practical, loses some global coherence
- Hybrid: Hierarchical summarization + retrieval for best of both

---

## Review Schedule

| Card | First Review | Second Review | Third Review |
|------|--------------|---------------|--------------|
| Card 1 (Easy) | Day 1 | Day 3 | Day 7 |
| Card 2 (Easy) | Day 1 | Day 3 | Day 7 |
| Card 3 (Medium) | Day 1 | Day 4 | Day 10 |
| Card 4 (Medium) | Day 2 | Day 5 | Day 12 |
| Card 5 (Hard) | Day 3 | Day 7 | Day 14 |

---

## Cross-References

| Card | Study Notes Section | Concept Map Node | Practice Problem |
|------|---------------------|------------------|------------------|
| Card 1 | Concepts 1, 2 | Self-Attention (11) | Problem 1 |
| Card 2 | Concepts 4, 5 | Encoder (7), Decoder (7) | Problem 2 |
| Card 3 | Concepts 7, 8 | BERT (6), GPT (6) | Problem 3 |
| Card 4 | Concept 3 | Positional Encoding (5) | Problem 1 |
| Card 5 | All Concepts | Full integration | Problem 5 |
