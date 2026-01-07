# Assessment Quiz: Transformers

**Source Material:** notes/transformers/flashcards/transformers-flashcards.md
**Practice Problems:** notes/transformers/practice/transformers-practice-problems.md
**Concept Map:** notes/transformers/concept-maps/transformers-concept-map.md
**Original Study Notes:** notes/transformers/transformers-study-notes.md
**Date Generated:** 2026-01-07
**Total Questions:** 5
**Estimated Completion Time:** 30-40 minutes
**Distribution:** 2 Multiple Choice | 2 Short Answer | 1 Essay

---

## Instructions

- **Multiple Choice:** Select the single best answer
- **Short Answer:** Respond in 2-4 sentences
- **Essay:** Provide a comprehensive response (1-2 paragraphs)

---

## Questions

### Question 1 | Multiple Choice
**Cognitive Level:** Remember/Understand
**Concept Tested:** Self-Attention Mechanism
**Source Section:** Core Concepts 1, 2
**Concept Map Node:** Self-Attention (11 connections)
**Related Flashcard:** Card 1
**Related Practice Problem:** P1

In scaled dot-product attention, why is the dot product between queries and keys divided by sqrt(dk)?

A) To reduce computational complexity from O(n^2) to O(n log n)

B) To prevent the dot products from growing too large with dimension, which would push softmax into regions with very small gradients

C) To normalize the attention weights so they sum to 1 across all positions

D) To enable the attention mechanism to learn relative positional information

---

### Question 2 | Multiple Choice
**Cognitive Level:** Remember/Understand
**Concept Tested:** Encoder vs Decoder Architecture
**Source Section:** Core Concepts 4, 5, 7, 8
**Concept Map Node:** Encoder (7), Decoder (7), BERT (6), GPT (6)
**Related Flashcard:** Card 2, Card 3
**Related Practice Problem:** P2

Which statement correctly describes the key architectural difference between BERT and GPT, and its implications?

A) BERT uses cross-attention while GPT uses self-attention, making BERT better for translation tasks

B) BERT uses bidirectional attention while GPT uses causal masking, making BERT better for understanding tasks and GPT better for generation tasks

C) BERT is trained on more data than GPT, making it more capable at all tasks

D) BERT uses positional encodings while GPT does not, giving BERT better position awareness

---

### Question 3 | Short Answer
**Cognitive Level:** Apply/Analyze
**Concept Tested:** Positional Encoding
**Source Section:** Core Concepts 3
**Concept Map Node:** Positional Encoding (5)
**Related Flashcard:** Card 4
**Related Practice Problem:** P1
**Expected Response Length:** 3-4 sentences

A colleague proposes removing positional encodings from a transformer to reduce parameters. Explain why this would be problematic and what information would be lost. How do modern approaches like RoPE differ from the original sinusoidal encodings?

---

### Question 4 | Short Answer
**Cognitive Level:** Apply/Analyze
**Concept Tested:** Architecture Selection
**Source Section:** Core Concepts 7, 8, 9
**Concept Map Node:** BERT (6), GPT (6), T5 (3)
**Related Flashcard:** Card 3
**Related Practice Problem:** P2, P3
**Expected Response Length:** 3-4 sentences

You need to build a system that takes a product review and extracts: (1) the sentiment (positive/negative), (2) specific product features mentioned, and (3) generates a one-sentence summary. For each sub-task, recommend whether to use BERT-style or GPT-style architecture and briefly justify your choice.

---

### Question 5 | Essay
**Cognitive Level:** Evaluate/Synthesize
**Concepts Tested:** Complete Transformer System
**Source Sections:** All Core Concepts
**Concept Map:** Full pathway traversal
**Related Flashcard:** Card 5
**Related Practice Problem:** P4, P5
**Expected Response Length:** 1-2 paragraphs

A startup wants to build a code review assistant that can: (1) understand code context across an entire repository (potentially 500K+ tokens), (2) identify potential bugs and suggest fixes, (3) generate explanations of complex code sections, and (4) run on a single A100 GPU with <2 second latency per query.

Design the system architecture addressing: (a) how to handle the massive context (500K tokens); (b) which transformer variant(s) to use for each sub-task; (c) efficiency techniques to meet latency requirements; (d) training approach if fine-tuning is needed; and (e) key trade-offs in your design.

**Evaluation Criteria:**
- [ ] Proposes viable context handling strategy (RAG, chunking, hierarchical)
- [ ] Selects appropriate architectures for different sub-tasks
- [ ] Identifies relevant efficiency techniques
- [ ] Addresses training/fine-tuning considerations
- [ ] Discusses key trade-offs

---

## Answer Key

### Question 1 | Multiple Choice
**Correct Answer:** B

**Explanation:**
The sqrt(dk) scaling is essential for stable training:

**The Problem:**
- Dot product of two d-dimensional vectors with unit variance has variance d
- For large dk (e.g., 64), dot products can be very large (~±8 for random vectors)
- Large values push softmax into saturation: softmax([10, 0, 0]) ≈ [1, 0, 0]
- In saturation, gradients are nearly zero → learning stalls

**The Solution:**
- Divide by sqrt(dk) → variance becomes 1 regardless of dimension
- Softmax operates in a reasonable range with meaningful gradients

**Mathematical View:**
```
If qi, kj ~ N(0, 1) independently:
E[qi · kj] = 0
Var[qi · kj] = dk

After scaling by sqrt(dk):
Var[qi · kj / sqrt(dk)] = 1
```

**Why Other Options Are Wrong:**
- A) Scaling doesn't change complexity; it's still O(n^2)
- C) Softmax normalizes to sum to 1, not the scaling factor
- D) Positional information comes from positional encodings, not scaling

---

### Question 2 | Multiple Choice
**Correct Answer:** B

**Explanation:**
| Aspect | BERT | GPT |
|--------|------|-----|
| **Attention** | Bidirectional (sees all tokens) | Causal (sees only past) |
| **Masking** | No mask in self-attention | Upper triangular mask |
| **Pre-training** | Masked Language Modeling | Causal Language Modeling |
| **Best for** | Understanding (classification, NER) | Generation (text, code) |

**Why Bidirectional Helps Understanding:**
- For "The bank by the river" vs "The bank gave me a loan"
- BERT sees context on BOTH sides to disambiguate "bank"
- GPT only sees left context at each position

**Why Causal Helps Generation:**
- Generation is inherently left-to-right
- Can't see future tokens when generating
- Causal training matches inference exactly

**Why Other Options Are Wrong:**
- A) Both use self-attention; cross-attention is for encoder-decoder
- C) Data quantity isn't the architectural difference
- D) Both use positional encodings

---

### Question 3 | Short Answer
**Model Answer:**

Removing positional encodings would be catastrophic because self-attention is permutation equivariant—without position information, the model treats "dog bites man" identically to "man bites dog." The order of tokens carries crucial semantic meaning that would be completely lost. The model would only learn bag-of-words-like representations, unable to capture syntax, grammar, or sequential dependencies.

Modern approaches like RoPE (Rotary Position Embedding) differ from sinusoidal encodings in that they encode *relative* positions rather than absolute positions. RoPE applies position-dependent rotations to query and key vectors, so the attention score between two tokens depends on their relative distance, not their absolute positions. This enables better generalization to sequence lengths longer than seen during training, which is why models like Llama and Mistral use RoPE.

**Key Components Required:**
- [ ] Explains permutation equivariance problem
- [ ] Notes loss of sequential/syntactic information
- [ ] Distinguishes relative vs absolute position encoding
- [ ] Mentions length generalization benefit of RoPE

---

### Question 4 | Short Answer
**Model Answer:**

**Sentiment Analysis → BERT-style:** This is a classification task where the full review is available. Bidirectional attention allows the model to see context on both sides, crucial for handling negations ("not bad") and sentiment modifiers. Use encoder with classification head on [CLS] token.

**Feature Extraction → BERT-style:** This is token-level classification (NER-like). Need to tag each word as feature or not-feature. Bidirectional context helps identify feature boundaries and disambiguate terms. Use encoder with token classification head.

**Summary Generation → GPT-style:** This requires generating new text (the summary). Autoregressive generation with decoder ensures coherent, fluent output. Can prompt with review text followed by "Summary:" and let the model generate. Alternatively, could use encoder-decoder (T5-style) for more structured seq2seq approach.

**Key Components Required:**
- [ ] Correctly identifies BERT for sentiment (classification)
- [ ] Correctly identifies BERT for features (token classification)
- [ ] Correctly identifies GPT for summary (generation)
- [ ] Provides justification based on bidirectional vs autoregressive properties

---

### Question 5 | Essay
**Model Answer:**

**(a) Handling 500K Token Context:**

Direct attention over 500K tokens is computationally infeasible (O(n^2) = 250 billion operations). I recommend a **RAG-based hierarchical approach**:

```
Repository (500K tokens)
        ↓
Chunking (512-2048 token chunks)
        ↓
Embedding (code embedding model like CodeBERT)
        ↓
Vector Database (store chunk embeddings)

At query time:
Query → Embed → Retrieve top-k relevant chunks → Feed to LLM
```

For code review, also include: file structure context, function signatures from imported modules, and recent git diff. Use a sliding window approach for the specific file under review (4K-8K tokens of local context).

**(b) Architecture Selection:**

| Task | Architecture | Reasoning |
|------|--------------|-----------|
| Code understanding | Encoder (CodeBERT) | Bidirectional context for semantic understanding |
| Bug detection | Encoder + classification | Pattern matching against known bug patterns |
| Fix suggestion | Decoder (CodeLlama) | Need to generate code |
| Explanation | Decoder (GPT-style) | Generate natural language |

**Primary model:** Use CodeLlama 34B or similar code-specialized decoder-only model for generation tasks, with retrieved context injected into prompt.

**(c) Efficiency Techniques:**

| Technique | Implementation | Benefit |
|-----------|----------------|---------|
| Flash Attention 2 | `attn_implementation="flash_attention_2"` | 2-4x speedup |
| KV Cache | Enable for autoregressive generation | Avoid recomputation |
| 8-bit Quantization | `load_in_8bit=True` | 2x memory reduction |
| vLLM / TGI | Optimized serving framework | Batching, PagedAttention |
| Speculative Decoding | Small draft model + verification | 2x generation speed |

**Latency Budget (2 seconds):**
- Retrieval: ~100ms
- Encoding query: ~50ms
- LLM inference (500 tokens output): ~1.5s with optimizations
- Post-processing: ~100ms

**(d) Training Approach:**

1. **Retriever:** Fine-tune CodeBERT on code search pairs (query → relevant chunk)
2. **Bug detector:** Train classifier on labeled bug datasets (e.g., Defects4J)
3. **Generator:** Start with CodeLlama, fine-tune with LoRA on:
   - Code review datasets (human reviewer comments)
   - Bug fix pairs (buggy code → fixed code)
   - Code explanation pairs (code → documentation)

Use QLoRA to fit 34B model fine-tuning on single A100.

**(e) Key Trade-offs:**

| Trade-off | Choice Made | What We Lose |
|-----------|-------------|--------------|
| Full repo attention vs RAG | RAG | Global context awareness |
| Model size (7B vs 34B) | 34B with quantization | Some precision |
| Latency vs quality | Optimize for 2s | Could be better with more compute |
| Fine-tuning vs prompting | LoRA fine-tune | Generalization to new patterns |

**Critical Design Decision:** The retrieval quality is the bottleneck—if we retrieve wrong context, the LLM can't help. Invest heavily in retriever quality and include multiple retrieval signals (semantic similarity, file structure, import graph).

**Production Architecture:**
```
┌────────────────────────────────────────────────────────────┐
│                    Code Review Assistant                    │
├────────────────────────────────────────────────────────────┤
│  Repository → AST Parser → Chunker → CodeBERT → Vector DB  │
│                                                             │
│  Query: "Review this function for bugs"                     │
│           ↓                                                 │
│  Retriever: Top-10 related chunks + file context            │
│           ↓                                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  CodeLlama 34B (8-bit, Flash Attn, vLLM)            │   │
│  │  Prompt: [Context] + [Code] + [Task: review/explain] │   │
│  └─────────────────────────────────────────────────────┘   │
│           ↓                                                 │
│  Response: Bug analysis / Suggestions / Explanation         │
└────────────────────────────────────────────────────────────┘
```

---

## Diagnostic Feedback & Review Recommendations

### Performance-Based Review Guide

| If Incorrect On... | Gap Identified | Review Section | Priority |
|--------------------|----------------|----------------|----------|
| Question 1 | Attention mechanics | Core Concepts 1 + Flashcard 1 | High |
| Question 2 | Architecture differences | Core Concepts 4, 5, 7, 8 + Flashcard 2, 3 | High |
| Question 3 | Positional encoding | Core Concepts 3 + Flashcard 4 | Medium |
| Question 4 | Architecture selection | Concepts 7-9 + Practice P2 | Medium |
| Question 5 | System integration | All sections | Low |

### Mastery Indicators

- **5/5 Correct:** Strong mastery; ready for transformer implementation
- **4/5 Correct:** Good understanding; review indicated gap
- **3/5 Correct:** Moderate understanding; systematic review needed
- **2/5 or below:** Foundational gaps; restart from Concept Map Critical Path

---

## Skill Chain Traceability

```
Study Notes (Source) ──────────────────────────────────────────────────────┐
    │                                                                      │
    │  10 Core Concepts, 12 Key Terms, 4 Applications                      │
    │                                                                      │
    ├────────────┬────────────┬────────────┬────────────┐                  │
    │            │            │            │            │                  │
    ▼            ▼            ▼            ▼            ▼                  │
Concept Map  Flashcards   Practice    Quiz                                 │
    │            │        Problems      │                                  │
    │ 32 concepts│ 5 cards   │ 5 problems│ 5 questions                     │
    │ 48 rels    │ 2E/2M/1H  │ 1W/2S/1C/1D│ 2MC/2SA/1E                     │
    │ 4 pathways │           │           │                                 │
    │            │           │           │                                 │
    └─────┬──────┴─────┬─────┴─────┬─────┘                                 │
          │            │           │                                       │
          │ Centrality │ Practice  │                                       │
          │ → Card     │ → Quiz    │                                       │
          │ difficulty │ distractors│                                      │
          │            │           │                                       │
          └────────────┴───────────┴───────────────────────────────────────┘
                                   │
                          Quiz integrates ALL
                          upstream materials
```

---

## Complete 5-Skill Chain Summary

| Skill | Output | Key Contribution to Chain |
|-------|--------|---------------------------|
| study-notes-creator | 10 concepts, theory, applications | Foundation content |
| concept-map | 32 nodes, 48 relationships, 4 pathways | Structure + learning paths |
| flashcards | 5 cards (2E/2M/1H) | Memorization + critical concepts |
| practice-problems | 5 problems (1W/2S/1C/1D) | Procedural fluency + debugging |
| quiz | 5 questions (2MC/2SA/1E) | Assessment + diagnostic feedback |
