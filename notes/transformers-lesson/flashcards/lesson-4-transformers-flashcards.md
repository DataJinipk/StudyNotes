# Flashcard Set: Lesson 4 - Transformers

**Source:** Lessons/Lesson_4.md
**Subject Area:** AI Learning - Transformer Architecture: Attention Mechanisms, Model Variants, and Implementation
**Date Generated:** 2026-01-08
**Total Cards:** 5
**Distribution:** 2 Easy | 2 Medium | 1 Hard

---

## Critical Knowledge Summary

Concepts appearing in multiple cards (prioritize for review):
- **Self-Attention**: Appears in Cards 1, 2, 5 (fundamental mechanism)
- **Multi-Head Attention**: Appears in Cards 2, 4, 5 (architectural pattern)
- **Positional Encoding**: Appears in Cards 1, 3, 5 (sequence order)
- **Architecture Variants**: Appears in Cards 4, 5 (encoder/decoder selection)

---

## Flashcards

---
### Card 1 | Easy
**Cognitive Level:** Remember
**Concept:** Scaled Dot-Product Attention
**Source Section:** Core Concepts - Concept 1

**FRONT (Question):**
Write the complete formula for scaled dot-product attention and explain the purpose of each component.

**BACK (Answer):**
**Formula:**
```
Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V
```

**Component Breakdown:**

| Component | Formula Part | Purpose |
|-----------|--------------|---------|
| **Q (Query)** | Q = X @ W_Q | "What am I looking for?" - represents the search |
| **K (Key)** | K = X @ W_K | "What do I contain?" - represents content for matching |
| **V (Value)** | V = X @ W_V | "What do I provide?" - actual information to retrieve |
| **Q @ K^T** | Dot product | Computes compatibility scores between all position pairs |
| **√d_k** | Scaling factor | Prevents large dot products from pushing softmax to extremes (vanishing gradients) |
| **softmax** | Normalization | Converts scores to probability distribution (weights sum to 1) |
| **@ V** | Weighted sum | Aggregates values according to attention weights |

**Output Shape:** [sequence_length, d_v] - same positions, new representations

**Complexity:** O(n²d) where n = sequence length, d = dimension

**Key Insight:** Each output position is a weighted combination of ALL input positions, with weights determined by learned query-key compatibility.

**Critical Knowledge Flag:** Yes - Foundation of all transformer models

---

---
### Card 2 | Easy
**Cognitive Level:** Understand
**Concept:** Multi-Head Attention
**Source Section:** Core Concepts - Concept 2

**FRONT (Question):**
Why does multi-head attention use h heads with dimension d_k = d_model/h instead of a single head with full dimension? What do different heads learn to capture?

**BACK (Answer):**
**Why Multiple Heads:**

| Reason | Explanation |
|--------|-------------|
| **Diverse attention patterns** | Single head = one attention pattern; multiple heads = multiple simultaneous patterns |
| **Representation subspaces** | Each head attends in a different learned subspace, capturing different relationships |
| **No additional compute** | h heads × d_k dimensions = 1 head × d_model dimensions (same total FLOPs) |

**What Different Heads Learn:**

| Head Type | Pattern | Example |
|-----------|---------|---------|
| **Positional heads** | Attend to nearby positions | Captures local n-gram patterns |
| **Syntactic heads** | Attend to grammatically related words | Subject → verb agreement |
| **Semantic heads** | Attend to similar meaning | "happy" ↔ "joyful" |
| **Copy heads** | Attend to identical/repeated tokens | Tracking entity mentions |

**Computation:**
```
For each head i:
    head_i = Attention(X @ W_Q^i, X @ W_K^i, X @ W_V^i)

MultiHead = Concat(head_1, ..., head_h) @ W_O
```

**Key Insight:** Heads specialize automatically during training without explicit supervision—the model discovers useful attention patterns from data.

**Typical Values:** h = 8 to 32 heads; d_k = 64 to 128 per head

**Critical Knowledge Flag:** Yes - Enables transformer expressiveness

---

---
### Card 3 | Medium
**Cognitive Level:** Apply
**Concept:** Positional Encoding Selection
**Source Section:** Core Concepts - Concept 3

**FRONT (Question):**
You're designing a transformer for processing legal documents that average 50,000 tokens but occasionally reach 200,000 tokens. Your training data only contains documents up to 32,000 tokens. Compare three positional encoding approaches (Learned, Sinusoidal, RoPE/ALiBi) and recommend the best choice with justification.

**BACK (Answer):**
**Requirement Analysis:**
- Average: 50K tokens (exceeds training max)
- Maximum: 200K tokens (6x training length)
- Training: Only up to 32K tokens
- **Critical need:** Length generalization beyond training

**Approach Comparison:**

| Approach | Max Length | Generalization | Quality at 200K |
|----------|------------|----------------|-----------------|
| **Learned** | Fixed (32K) | None - crashes beyond training | Not applicable |
| **Sinusoidal** | Unlimited | Limited - degrades smoothly | Poor - patterns break down |
| **RoPE** | Unlimited | Good - relative positions | Moderate - some degradation |
| **ALiBi** | Unlimited | Excellent - linear bias | Good - graceful degradation |

**Detailed Analysis:**

| Approach | Mechanism | Why It Matters for This Task |
|----------|-----------|------------------------------|
| **Learned** | Lookup table: position → embedding | Cannot process position 32,001. System fails. |
| **Sinusoidal** | sin/cos at different frequencies | Positions extrapolate but attention patterns trained on shorter sequences don't transfer well |
| **RoPE** | Rotate Q/K vectors by position | Encodes relative position (i-j); attention score depends on distance, not absolute position |
| **ALiBi** | Subtract m×|i-j| from attention scores | Directly penalizes distant attention; no position in embedding; scales to arbitrary length |

**Recommendation: ALiBi**

| Justification | Explanation |
|---------------|-------------|
| **Length generalization** | Best empirical performance beyond training length |
| **No parameters** | No learned position embeddings to overfit |
| **Interpretable** | Clear bias toward local attention |
| **Efficiency** | No additional computation for embeddings |

**Alternative:** RoPE if you need relative position information in a way that preserves attention pattern shapes. ALiBi's linear penalty may over-penalize important long-range dependencies.

**Implementation Note:**
```
# ALiBi: Add bias to attention scores before softmax
attention_bias[i,j] = -m * |i - j|  # m is head-specific slope
scores = Q @ K^T + attention_bias
```

**Critical Knowledge Flag:** Yes - Position encoding determines max context capability

---

---
### Card 4 | Medium
**Cognitive Level:** Analyze
**Concept:** Architecture Variant Selection
**Source Section:** Core Concepts - Concept 6

**FRONT (Question):**
Analyze the following three tasks and determine which transformer architecture (encoder-only, decoder-only, encoder-decoder) is optimal for each. Justify your selections based on task requirements.

**Tasks:**
1. Classifying customer support tickets by urgency
2. Generating Python code from natural language descriptions
3. Translating legal contracts from English to German

**BACK (Answer):**
**Task 1: Ticket Classification → Encoder-Only (BERT)**

| Analysis | Reasoning |
|----------|-----------|
| **Full context needed** | Must read entire ticket to assess urgency |
| **No generation** | Output is class label, not generated text |
| **Bidirectional** | "URGENT: call me back" - both parts needed |
| **Why not decoder** | Autoregressive is unnecessary overhead for classification |

**Implementation:**
```
Input: [CLS] ticket_text [SEP]
Output: [CLS] representation → Linear → urgency_class
```

---

**Task 2: Code Generation → Decoder-Only (GPT/CodeLlama)**

| Analysis | Reasoning |
|----------|-----------|
| **Generation task** | Output is generated token-by-token |
| **Autoregressive natural** | Code is written left-to-right |
| **Context utilization** | Description in prompt, generate completion |
| **Why not enc-dec** | Single sequence (description → code) fits decoder paradigm |

**Implementation:**
```
Input: "Write Python function to calculate fibonacci: def fib(n):"
Output: Generate token by token: "    if n <= 1:..."
```

---

**Task 3: Legal Translation → Encoder-Decoder (T5/mT5)**

| Analysis | Reasoning |
|----------|-----------|
| **Full source needed** | Must understand entire contract before translating |
| **Different languages** | Input/output are structurally different |
| **Cross-attention** | Decoder queries encoder for source context |
| **Faithful translation** | Encoder captures full meaning; decoder generates faithfully |

**Implementation:**
```
Encoder input: "This agreement shall terminate upon..."
Cross-attention: Decoder attends to encoder representation
Decoder output: "Diese Vereinbarung endet bei..."
```

**Summary Decision Matrix:**

| Task | Architecture | Key Deciding Factor |
|------|--------------|---------------------|
| Ticket classification | Encoder-only | Understanding, no generation |
| Code generation | Decoder-only | Autoregressive generation |
| Legal translation | Encoder-decoder | Transform sequence to sequence |

**Critical Knowledge Flag:** Yes - Architecture selection is fundamental design decision

---

---
### Card 5 | Hard
**Cognitive Level:** Synthesize
**Concept:** Complete Transformer System Design
**Source Section:** All Core Concepts, Practical Applications, Case Study

**FRONT (Question):**
Design a complete transformer-based system for a medical literature search engine that:
1. Takes natural language queries ("What are side effects of metformin in elderly patients?")
2. Searches a database of 10 million medical abstracts
3. Returns ranked relevant abstracts with generated explanations of why each is relevant

Your design must specify:
- Architecture choices for each component
- Attention mechanisms and positional encodings
- How you handle the scale (10M documents)
- Efficiency considerations
- Quality vs. latency trade-offs

**BACK (Answer):**
**System Overview:**

```
┌─────────────────────────────────────────────────────────────────┐
│                     QUERY PROCESSING                            │
│ ─────────────────────────────────────────                       │
│ Input: "What are side effects of metformin in elderly?"         │
│                                                                 │
│ Component: Encoder Model (BERT-style)                           │
│ Rationale: Full query understanding, no generation needed       │
│ Output: Dense query embedding (768-1024 dimensions)             │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     RETRIEVAL STAGE                             │
│ ─────────────────────────────────────────                       │
│ Method: Approximate Nearest Neighbor (ANN) search               │
│                                                                 │
│ Pre-computed: All 10M abstracts encoded with same encoder       │
│ At query time: Query embedding → ANN index → Top 100 candidates │
│                                                                 │
│ Efficiency: O(log n) retrieval via HNSW/IVF indexing           │
│ Latency: ~10ms for 10M documents                               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     RE-RANKING STAGE                            │
│ ─────────────────────────────────────────                       │
│ Component: Cross-Encoder (BERT-style)                           │
│                                                                 │
│ Input: [CLS] query [SEP] abstract [SEP]                        │
│ Attention: Full cross-attention between query and abstract      │
│ Output: Relevance score                                         │
│                                                                 │
│ Applied to: Top 100 from retrieval → Re-rank → Top 10          │
│ Latency: ~100ms (100 abstracts × ~1ms each with batching)      │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   EXPLANATION GENERATION                        │
│ ─────────────────────────────────────────                       │
│ Component: Decoder-Only (GPT-style, small/medium)               │
│                                                                 │
│ Input: "Query: [query]\nAbstract: [abstract]\nExplain relevance:" │
│ Attention: Causal self-attention                                │
│ Output: "This abstract addresses metformin side effects..."     │
│                                                                 │
│ Applied to: Top 10 results                                      │
│ Latency: ~500ms (parallel generation for 10 abstracts)          │
└─────────────────────────────────────────────────────────────────┘
```

**Architecture Decisions:**

| Component | Architecture | Specific Model | Rationale |
|-----------|--------------|----------------|-----------|
| **Query Encoder** | Encoder-only | PubMedBERT (domain-adapted) | Bidirectional; medical pre-training |
| **Document Encoder** | Encoder-only | Same as query | Shared embedding space |
| **Re-ranker** | Encoder-only (cross) | MedBERT fine-tuned | Full attention between query-doc |
| **Explainer** | Decoder-only | Llama-2-7B medical LoRA | Generation needed; smaller for latency |

**Attention Mechanism Details:**

| Stage | Attention Type | Positional Encoding | Why |
|-------|---------------|---------------------|-----|
| **Query/Doc Encoding** | Bidirectional self-attention | Learned (512 max) | Abstracts fit; full understanding |
| **Re-ranking** | Bidirectional cross | Learned + segment embeddings | Query-doc interaction |
| **Explanation** | Causal self-attention | RoPE | Generation; potential long context |

**Scaling Strategy for 10M Documents:**

| Challenge | Solution | Complexity |
|-----------|----------|------------|
| Store 10M embeddings | FAISS IVF index | O(√n) search |
| Compute 10M embeddings | Batch offline; update incrementally | One-time + O(new docs) |
| Index size | Product quantization (768d → 64 bytes) | ~640MB for 10M |
| Retrieval speed | GPU-accelerated ANN | 10ms at scale |

**Efficiency Optimizations:**

| Optimization | Applied To | Impact |
|--------------|------------|--------|
| **Flash Attention** | All components | 2x speed, 10x memory |
| **INT8 Quantization** | Re-ranker, Explainer | 2x inference speed |
| **KV Caching** | Explainer generation | Essential for latency |
| **Batched Inference** | Re-ranking (100 docs) | 10x throughput |
| **Speculative Decoding** | Explainer | 2x generation speed |

**Quality vs. Latency Trade-offs:**

| Decision | Quality Impact | Latency Impact | Chosen Setting |
|----------|----------------|----------------|----------------|
| Top-K retrieval | More candidates = better recall | More re-ranking time | K=100 (good balance) |
| Re-ranker size | Larger = better relevance | Slower inference | BERT-base (not large) |
| Explanation length | Longer = more informative | More generation time | Max 100 tokens |
| Explanation model | Larger = better explanations | Slower generation | 7B (not 70B) |

**Total Latency Budget:**
```
Query encoding:    ~10ms
ANN retrieval:     ~10ms
Re-ranking (100):  ~100ms
Explanation (10):  ~500ms (parallelized)
─────────────────────────
Total:             ~620ms target (sub-second)
```

**Quality Assurance:**

| Concern | Mitigation |
|---------|------------|
| Retrieval misses | Re-ranker can recover if doc retrieved |
| Explanation hallucination | Constrain to cite abstract content |
| Medical accuracy | Domain-adapted models; human review flagging |

**Critical Knowledge Flag:** Yes - Integrates all transformer concepts in production system

---

---

## Export Formats

### Anki-Compatible (Tab-Separated)
```
Write scaled dot-product attention formula	Attention(Q,K,V) = softmax(QK^T/√d_k)V. Q=query, K=key, V=value, √d_k prevents vanishing gradients, softmax normalizes weights.	easy::attention::transformer
Why multi-head vs single-head attention?	Multiple heads = diverse patterns (positional, syntactic, semantic). Same compute: h×d_k = d_model. Heads specialize automatically.	easy::architecture::transformer
Select positional encoding for 200K doc processing	ALiBi: best length generalization, no parameters, linear distance penalty. RoPE alternative for relative position. Learned fails beyond training.	medium::position::transformer
Match tasks to transformer architectures	Classification→Encoder (understanding). Generation→Decoder (autoregressive). Translation→Enc-Dec (seq2seq cross-attention).	medium::selection::transformer
Design medical literature search system	Query encoder (BERT)→ANN retrieval (FAISS)→Cross-encoder re-rank→Decoder explanation. Flash attention, quantization, batching for efficiency.	hard::system::transformer
```

### CSV Format
```csv
"Front","Back","Difficulty","Concept","Cognitive_Level"
"Write scaled dot-product attention formula","Attention(Q,K,V) = softmax(QK^T/√d_k)V with Q/K/V projections, scaling, softmax, aggregation","Easy","Attention","Remember"
"Why multi-head vs single-head attention?","Multiple patterns, same compute, automatic specialization (positional, syntactic, semantic, copy heads)","Easy","Multi-Head","Understand"
"Select positional encoding for long documents","ALiBi for best generalization; RoPE for relative; Learned fails beyond training length","Medium","Position","Apply"
"Match tasks to transformer architectures","Encoder=understanding, Decoder=generation, Enc-Dec=seq2seq. Match architecture to task structure.","Medium","Variants","Analyze"
"Design medical search transformer system","Multi-stage: encode→retrieve→rerank→explain with appropriate architecture per stage","Hard","System Design","Synthesize"
```

---

## Source Mapping

| Card | Source Section | Key Terminology | Bloom's Level |
|------|----------------|-----------------|---------------|
| 1 | Core Concepts - Concept 1 | Q/K/V, softmax, √d_k scaling | Remember |
| 2 | Core Concepts - Concept 2 | Multi-head, subspaces, head specialization | Understand |
| 3 | Core Concepts - Concept 3 | RoPE, ALiBi, learned, sinusoidal | Apply |
| 4 | Core Concepts - Concept 6 | Encoder, decoder, enc-dec, causal mask | Analyze |
| 5 | All Concepts + Case Study | Full system: encoding, retrieval, generation | Synthesize |

---

## Spaced Repetition Schedule

| Card | Initial Interval | Difficulty Multiplier | Recommended Review |
|------|------------------|----------------------|-------------------|
| 1 (Easy) | 1 day | 2.5x | Foundation - review first |
| 2 (Easy) | 1 day | 2.5x | Review with Card 1 |
| 3 (Medium) | 3 days | 2.0x | After mastering Cards 1-2 |
| 4 (Medium) | 3 days | 2.0x | Architecture decision skill |
| 5 (Hard) | 7 days | 1.5x | Review after all others mastered |

---

## Connection to Other Lessons

| Transformer Concept | LLMs (Lesson 3) | Prompt Engineering (Lesson 2) | Agent Skills (Lesson 1) |
|---------------------|-----------------|-------------------------------|-------------------------|
| Self-Attention | LLMs are built on transformers | Attention explains prompt structure effects | Skills must fit attention patterns |
| Positional Encoding | Context window limits | Position affects information retrieval | Skills chain within position constraints |
| Encoder vs Decoder | GPT=decoder, BERT=encoder | Decoder models need careful prompting | Different architectures need different skills |
| Multi-Head | Explains diverse capabilities | Different heads capture different prompt aspects | Skills leverage head specialization |
| Efficiency | KV cache, quantization | Long prompts hit efficiency limits | Skill chains must consider latency |

---

*Generated from Lesson 4: Transformers | Flashcards Skill*
