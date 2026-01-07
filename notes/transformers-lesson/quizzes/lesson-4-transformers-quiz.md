# Assessment Quiz: Lesson 4 - Transformers

**Source:** Lessons/Lesson_4.md
**Subject Area:** AI Learning - Transformer Architecture: Attention Mechanisms, Model Variants, and Implementation
**Date Generated:** 2026-01-08
**Total Questions:** 5
**Distribution:** 2 Multiple Choice | 2 Short Answer | 1 Essay

---

## Quiz Overview

| # | Type | Concept | Cognitive Level | Points |
|---|------|---------|-----------------|--------|
| 1 | Multiple Choice | Scaled Dot-Product Attention | Remember/Understand | 10 |
| 2 | Multiple Choice | Architecture Variants | Understand | 10 |
| 3 | Short Answer | Multi-Head Attention | Apply/Analyze | 20 |
| 4 | Short Answer | Positional Encoding | Analyze | 20 |
| 5 | Essay | Complete Transformer Design | Synthesize/Evaluate | 40 |

**Total Points:** 100
**Recommended Time:** 45-60 minutes

---

## Questions

---

### Question 1 | Multiple Choice
**Concept:** Scaled Dot-Product Attention
**Cognitive Level:** Remember/Understand
**Points:** 10

In the attention formula `Attention(Q, K, V) = softmax(QK^T / √d_k) V`, what happens if the √d_k scaling factor is removed when d_k is large (e.g., 1024)?

**A.** The attention weights become more uniform across all positions

**B.** The dot products become very large, pushing softmax toward one-hot outputs and causing vanishing gradients

**C.** The model learns faster because gradients flow more directly

**D.** The Value vectors are weighted incorrectly, producing outputs with wrong magnitudes

---

### Question 2 | Multiple Choice
**Concept:** Transformer Architecture Variants
**Cognitive Level:** Understand
**Points:** 10

A company needs to build a system that takes a customer complaint email and generates a formal response letter. Which transformer architecture is most appropriate?

**A.** Encoder-only (BERT-style): Use the [CLS] token representation to generate the response

**B.** Decoder-only (GPT-style): Concatenate the complaint and generate the response autoregressively

**C.** Encoder-decoder (T5-style): Encode the complaint with encoder, generate response with decoder using cross-attention

**D.** Either B or C would work well; decoder-only is simpler while encoder-decoder is more natural for this task

---

### Question 3 | Short Answer
**Concept:** Multi-Head Attention
**Cognitive Level:** Apply/Analyze
**Points:** 20

A transformer model has the following configuration:
- d_model = 768
- Number of heads (h) = 12
- Sequence length = 256

**Questions:**

**(a)** Calculate d_k (the dimension per head) and explain why transformers use multiple smaller heads instead of one large head with full d_model dimension. (8 points)

**(b)** Research has shown that different attention heads learn specialized patterns (positional, syntactic, semantic). For a sentiment analysis task on movie reviews, describe THREE specific attention patterns that different heads might learn, and explain why each would be useful. (12 points)

---

### Question 4 | Short Answer
**Concept:** Positional Encoding
**Cognitive Level:** Analyze
**Points:** 20

A research team is building a transformer for processing DNA sequences that can be millions of base pairs long. Their training data contains sequences up to 100,000 tokens, but at inference time they need to handle sequences up to 1,000,000 tokens.

**Questions:**

**(a)** Explain why learned positional embeddings would fail for this use case. What specific limitation makes them unsuitable? (6 points)

**(b)** Compare RoPE (Rotary Position Embedding) and ALiBi (Attention with Linear Biases) for this DNA sequence task. Analyze which would be better and justify your choice considering: length generalization, relative vs. absolute position, and computational overhead. (14 points)

---

### Question 5 | Essay
**Concept:** Complete Transformer System Design
**Cognitive Level:** Synthesize/Evaluate
**Points:** 40

**Prompt:**

Design a transformer-based system for a multilingual document translation service that must:

**Requirements:**
- Translate between 20 languages (any direction: English→French, Chinese→Spanish, etc.)
- Handle documents up to 10,000 tokens
- Achieve translation latency under 5 seconds for typical 1,000-token documents
- Support streaming output (show partial translations as they're generated)
- Maintain terminology consistency for technical documents

**Your essay must address:**

1. **Architecture Selection (10 points)**
   - Which transformer variant (encoder-only, decoder-only, encoder-decoder)?
   - How do you handle the multilingual aspect (separate models vs. single model)?
   - Justify your choices with technical reasoning

2. **Attention and Position Design (10 points)**
   - What attention pattern for 10,000-token documents?
   - Which positional encoding and why?
   - How do you ensure cross-lingual attention works effectively?

3. **Efficiency and Latency (10 points)**
   - How do you meet the 5-second latency requirement?
   - What optimizations for streaming output?
   - Memory considerations for long documents

4. **Quality Assurance (10 points)**
   - How do you maintain terminology consistency?
   - What mechanisms ensure translation faithfulness?
   - How do you handle domain-specific vocabulary?

**Evaluation Criteria:**
- Technical accuracy of transformer concepts
- Practical feasibility of proposed solutions
- Appropriate tradeoff analysis
- Integration of concepts across all lesson sections

---

## Answer Key

---

### Question 1 | Answer

**Correct Answer: B**

**Explanation:**
Without √d_k scaling, the dot products Q·K grow proportionally to d_k:
- Q and K vectors have components with variance ~1
- Dot product of two d_k-dimensional vectors has variance ~d_k
- For d_k=1024, dot products are ~32× larger than expected (√1024 = 32)

Large dot products cause:
1. **Softmax saturation:** softmax(large values) → one-hot-like distribution
2. **Gradient vanishing:** ∂softmax/∂input approaches 0 at extreme values
3. **Training instability:** Some parameters get no gradient, others explode

**Why other options are wrong:**
- **A:** Without scaling, attention becomes MORE concentrated (one-hot), not uniform
- **C:** The opposite happens—gradients vanish, training slows or fails
- **D:** Value magnitudes aren't directly affected by scaling; it's the weight distribution that breaks

**Source Reference:** Core Concepts - Concept 1: Scaled Dot-Product Attention

**Understanding Gap Indicator:** If missed, review the mathematical derivation of why variance of dot products scales with dimension.

---

### Question 2 | Answer

**Correct Answer: D**

**Explanation:**
Both decoder-only and encoder-decoder architectures can effectively handle this email-to-response generation task:

**Decoder-only approach:**
```
Input: "Complaint: [email text] Response:"
Generation: Autoregressive token-by-token output
```
- Simpler single-model architecture
- Modern decoder models (GPT-4, Llama) excel at this format
- Full context (complaint + response-so-far) available during generation

**Encoder-decoder approach:**
```
Encoder input: [email text]
Decoder: Generates response with cross-attention to encoded complaint
```
- More natural separation of understanding vs. generation
- Cross-attention explicitly connects response to complaint
- Traditional choice for this task (T5, BART)

**Why both work:**
- Task requires understanding input (complaint) and generating output (response)
- Both architectures can handle this seq2seq-like task
- Modern decoder-only models have largely closed the gap

**Why other options are wrong:**
- **A:** Encoder-only cannot generate text; [CLS] gives embedding, not response
- **B & C individually:** Both work, but claiming only one works is incomplete

**Source Reference:** Core Concepts - Concept 6: Model Variants and Architecture Selection

---

### Question 3 | Answer

**(a) Multi-Head Dimensions and Rationale (8 points)**

**Calculation:**
```
d_k = d_model / h = 768 / 12 = 64 per head
```

**Why Multiple Smaller Heads:**

| Single Head (d_k=768) | Multiple Heads (h=12, d_k=64) |
|-----------------------|------------------------------|
| One attention pattern | 12 different attention patterns |
| All capacity in one view | Diverse relationship capture |
| Limited expressiveness | Higher expressiveness |
| Same compute cost | Same compute cost |

**Technical Reasons:**
1. **Representation subspaces:** Each head operates in a different 64-dimensional subspace, allowing the model to attend to different types of relationships simultaneously

2. **Diverse patterns:** One head might learn positional attention while another learns semantic similarity—impossible with single head

3. **Ensemble effect:** Multiple heads provide robust attention through implicit ensemble, reducing sensitivity to any single pattern

4. **No compute penalty:** h heads × d_k = 1 head × d_model, so total FLOPs are identical

**Full Credit:** d_k=64 calculation (2 points) + 3 reasons why multiple heads (6 points)

---

**(b) Specialized Attention Patterns for Sentiment Analysis (12 points)**

**Pattern 1: Negation-Scope Attention**
| Aspect | Description |
|--------|-------------|
| **Pattern** | Negation words ("not", "never", "don't") attend strongly to the words they modify |
| **Example** | "The movie was not boring" → "not" attends to "boring" |
| **Why Useful** | Sentiment reversal requires connecting negation to the target; "not boring" = positive |

**Pattern 2: Sentiment Anchor Attention**
| Aspect | Description |
|--------|-------------|
| **Pattern** | Final [CLS] or aggregate position attends to sentiment-laden words |
| **Example** | [CLS] strongly attends to "amazing", "terrible", "disappointing" |
| **Why Useful** | Classification requires gathering sentiment signals scattered throughout review |

**Pattern 3: Comparative/Contrastive Attention**
| Aspect | Description |
|--------|-------------|
| **Pattern** | "but", "however", "although" attend to both clauses they connect |
| **Example** | "Acting was good but plot was weak" → "but" attends to "good" and "weak" |
| **Why Useful** | Contrastive constructions shift sentiment weight; need to understand both sides |

**Additional patterns (for full credit mention):**
- **Aspect-opinion linking:** Product features attend to their modifiers
- **Punctuation-emphasis:** Exclamation marks attend to emphatic words
- **Entity coreference:** Pronouns attend to their antecedents

**Scoring:**
- 4 points per pattern (description + example + utility)
- Partial credit for incomplete explanations

**Source Reference:** Core Concepts - Concept 2: Multi-Head Attention

---

### Question 4 | Answer

**(a) Why Learned Positional Embeddings Fail (6 points)**

**Full Credit Answer:**

Learned positional embeddings fail for this DNA task because:

1. **Fixed maximum length:** Learned embeddings are a lookup table of shape [max_length, d_model]. Positions beyond max_length have no embedding—the model cannot process them.

```
Training: positions 0-99,999 have learned embeddings
Inference: position 500,000 → index out of bounds → failure
```

2. **No extrapolation mechanism:** Unlike sinusoidal or RoPE which have mathematical structure extending to any position, learned embeddings are just stored vectors with no relationship between positions.

3. **Memory constraint:** Even if we trained on 1M positions:
   - Embedding table: 1,000,000 × 768 × 4 bytes = 3GB just for position embeddings
   - Training would require seeing positions 999,990-999,999 sufficiently—unlikely

**The specific limitation:** Learned embeddings cannot represent positions outside the training range because there's no generalization mechanism—each position is independently learned.

**Partial Credit:** Mentioning max length limit (3 points) without extrapolation explanation (3 points)

---

**(b) RoPE vs. ALiBi for DNA Sequences (14 points)**

**Comparison Analysis:**

| Criterion | RoPE | ALiBi |
|-----------|------|-------|
| **Length Generalization** | Good—tested to 4-8× training length | Excellent—tested to 10× and beyond |
| **Position Type** | Relative (i-j encoded in rotation) | Relative (bias based on |i-j|) |
| **Compute Overhead** | Moderate—rotates Q/K at each layer | Minimal—just adds bias to scores |
| **Memory Overhead** | None (rotation is applied, not stored) | None (bias computed on-the-fly) |
| **Information Content** | Rich—preserves exact relative position | Simple—only distance matters |

**For DNA Sequences Specifically:**

| Factor | Analysis |
|--------|----------|
| **10× length extrapolation needed** | ALiBi better—designed for extreme extrapolation |
| **DNA position semantics** | RoPE better—exact relative position matters in biology |
| **Computational efficiency** | ALiBi better—simpler at extreme lengths |
| **Pattern complexity** | RoPE better—DNA has complex positional patterns |

**Recommendation: RoPE with careful training**

**Justification:**

1. **Biological patterns require exact relative positions:** DNA has codons (3 base pairs), exon/intron boundaries at specific positions, regulatory sequences at fixed offsets. RoPE preserves "position 50 before this marker" information that ALiBi's simple distance decay loses.

2. **Mitigate extrapolation weakness:** Train on progressively longer sequences (curriculum), use NTK-aware interpolation for RoPE to improve extrapolation to 10× length.

3. **DNA is different from text:** ALiBi's assumption that nearby = more relevant works for text but DNA has long-range regulatory relationships (enhancers affecting genes 100K bases away).

**Alternative argument for ALiBi (also acceptable):**
If the task primarily uses local patterns (e.g., variant calling in local regions), ALiBi's simpler, more robust extrapolation may be preferred despite losing exact position information.

**Scoring:**
- Comparison table (6 points)
- Analysis specific to DNA (4 points)
- Justified recommendation (4 points)

**Source Reference:** Core Concepts - Concept 3: Positional Encoding

---

### Question 5 | Essay Rubric

**Total: 40 points**

---

**Section 1: Architecture Selection (10 points)**

| Points | Criteria |
|--------|----------|
| 9-10 | Clear architecture choice with strong justification; addresses multilingual approach comprehensively |
| 7-8 | Good architecture selection with solid reasoning; minor gaps |
| 5-6 | Basic architecture choice with limited justification |
| 3-4 | Architecture mentioned but reasoning weak |
| 0-2 | No clear architecture decision |

**Key Elements for Full Credit:**

**Architecture Choice:** Encoder-decoder (T5/mT5/NLLB) OR Decoder-only (GPT-style multilingual)

| Choice | Justification |
|--------|---------------|
| **Encoder-decoder** | Natural fit for translation; encoder captures source, decoder generates target; cross-attention provides explicit alignment |
| **Decoder-only** | Simpler; modern multilingual decoders competitive; single model to maintain |

**Multilingual Approach:**
- **Single multilingual model** (recommended): NLLB, mT5 style; shared representations across languages; zero-shot to new pairs
- **Language-specific adapters:** LoRA per language pair; efficient fine-tuning
- **Language tags:** "[FR→EN]" prefix to indicate translation direction

---

**Section 2: Attention and Position Design (10 points)**

| Points | Criteria |
|--------|----------|
| 9-10 | Addresses 10K token handling with specific attention pattern; appropriate position encoding choice with cross-lingual considerations |
| 7-8 | Good attention/position design with minor gaps |
| 5-6 | Basic consideration of long documents |
| 3-4 | Mentions attention but lacks detail |
| 0-2 | No attention design |

**Key Elements for Full Credit:**

**10K Token Handling:**
```
Option A: Efficient Attention
- Flash Attention for memory efficiency
- Sliding window (4K) + global tokens for cross-attention

Option B: Chunking with Context
- Split source into overlapping chunks
- Encode chunks; decoder attends to all
```

**Positional Encoding:**
- **RoPE or ALiBi** for length generalization
- Cross-lingual: Position in source vs. target handled separately; cross-attention connects them

**Cross-Lingual Attention:**
- Encoder positions = source language
- Decoder positions = target language
- Cross-attention handles alignment (no position conflict)

---

**Section 3: Efficiency and Latency (10 points)**

| Points | Criteria |
|--------|----------|
| 9-10 | Specific latency analysis; streaming implementation; memory optimization |
| 7-8 | Good efficiency considerations with minor gaps |
| 5-6 | Basic latency awareness |
| 3-4 | Mentions speed but no concrete approach |
| 0-2 | No efficiency discussion |

**Key Elements for Full Credit:**

**5-Second Latency Analysis:**
```
1,000 tokens × 2 (source + target avg) = 2,000 tokens

Time budget:
- Encoding: ~200ms (single forward pass)
- Decoding: 1,000 tokens × ~3ms/token = 3,000ms
- Overhead: ~300ms
- Total: ~3.5 seconds ✓ (under 5s)

For 10K tokens:
- Encoding: ~500ms
- Decoding: ~10K tokens × 3ms = 30s ❌
- Solution: Chunk processing, KV cache optimization
```

**Streaming Output:**
```python
def translate_streaming(source):
    encoder_output = model.encode(source)  # Batch encode

    for token in model.generate_streaming(encoder_output):
        yield token  # Send immediately
        # KV cache accumulates; no recomputation
```

**Memory Optimization:**
- KV cache for decoder (no recomputation)
- INT8/INT4 quantization for longer documents
- Gradient checkpointing not needed (inference only)

---

**Section 4: Quality Assurance (10 points)**

| Points | Criteria |
|--------|----------|
| 9-10 | Terminology consistency mechanism; faithfulness assurance; domain handling |
| 7-8 | Good quality measures with minor gaps |
| 5-6 | Basic quality awareness |
| 3-4 | Mentions quality but no concrete approach |
| 0-2 | No quality discussion |

**Key Elements for Full Credit:**

**Terminology Consistency:**
```
Approach: Constrained Decoding + Glossary

1. Pre-populate glossary: {source_term: target_term}
2. During decoding:
   - If source contains glossary term
   - Boost logits for glossary translation
   - Or: Forced decoding for exact matches
```

**Translation Faithfulness:**
- Attention-based coverage: Ensure all source tokens attended
- Backtranslation check: Translate back; compare to original
- Length ratio monitoring: Flag if output/input ratio anomalous

**Domain Handling:**
- Fine-tune on domain data (legal, medical, technical)
- Domain-specific glossaries
- Style tags: "[FORMAL]" or "[TECHNICAL]" prefix

---

**Sample High-Scoring Essay Excerpt:**

> "For this multilingual translation service, I recommend an encoder-decoder architecture based on NLLB-200 (No Language Left Behind), which was specifically designed for many-to-many multilingual translation. The encoder-decoder structure naturally separates source understanding from target generation, with cross-attention providing explicit alignment—critical for translation quality.
>
> For the multilingual aspect, a single massively multilingual model is preferred over 20×19=380 separate bilingual models for several reasons: shared representations enable transfer learning between related languages (Spanish-Portuguese share features), zero-shot translation is possible for unseen pairs by routing through a well-resourced language, and a single model dramatically simplifies deployment and maintenance.
>
> For 10,000-token documents, I employ Flash Attention for exact attention with memory efficiency, avoiding approximation quality loss. The encoder processes the full source document; for extremely long documents, I use hierarchical encoding with segment-level and document-level representations. Positional encoding uses ALiBi for its superior length generalization—translation must handle variable document lengths without quality degradation at the boundaries.
>
> Meeting the 5-second latency requirement for 1,000-token documents is feasible: encoding takes ~200ms, and with KV caching and speculative decoding (using a smaller draft model for the same language pair), generation achieves ~2ms/token, yielding ~2.2 seconds total. For streaming, tokens are yielded immediately upon generation; the KV cache accumulates so there's no recomputation, maintaining consistent per-token latency.
>
> Terminology consistency uses a constrained decoding approach: technical documents come with glossaries extracted from translation memory systems. During generation, when the decoder encounters a position corresponding to a glossary source term (detected via encoder-decoder attention patterns), logits for the glossary target term receive a significant boost. This isn't hard forcing—the model can override if context strongly disagrees—but biases toward consistency. For domain handling, we fine-tune domain-specific LoRA adapters (legal, medical, technical) selected based on document metadata..."

---

## Performance Interpretation

### Score Ranges

| Score | Level | Interpretation |
|-------|-------|----------------|
| 90-100 | Mastery | Ready to implement and optimize transformer systems |
| 80-89 | Proficient | Strong understanding; minor gaps in advanced concepts |
| 70-79 | Competent | Solid foundations; needs practice with complex design |
| 60-69 | Developing | Understands core concepts but struggles with integration |
| Below 60 | Foundational | Review core concepts before attempting system design |

### Recommended Review by Question

| If you struggled with... | Review these sections... |
|--------------------------|--------------------------|
| Q1 (Scaling) | Core Concept 1, √d_k derivation |
| Q2 (Architecture) | Core Concept 6, variant comparison |
| Q3 (Multi-Head) | Core Concept 2, head specialization |
| Q4 (Position) | Core Concept 3, encoding comparison |
| Q5 (System Design) | All Core Concepts, Case Study |

---

## Cross-Lesson Connections

This quiz assesses readiness to apply transformer knowledge in:

| Downstream Application | Required Understanding |
|------------------------|------------------------|
| **LLMs (Lesson 3)** | LLMs are decoder transformers; attention patterns affect prompting |
| **Prompt Engineering (Lesson 2)** | Attention mechanisms determine how prompts influence output |
| **Agent Skills (Lesson 1)** | Skills must work within transformer constraints |
| **Production Systems** | Efficiency and architecture selection for real deployments |

---

*Generated from Lesson 4: Transformers | Quiz Skill*
