# Assessment Quiz: Lesson 9 - Natural Language Processing

**Source:** Lessons/Lesson_9.md
**Subject Area:** AI Learning - Natural Language Processing: From Text Representation to Neural Language Understanding
**Date Generated:** 2026-01-08
**Total Questions:** 5
**Estimated Time:** 35-45 minutes

---

## Instructions

This assessment evaluates your understanding of Natural Language Processing concepts, from text representation methods through pre-trained language models and modern NLP applications. Answer all questions completely, showing your reasoning where applicable.

**Question Distribution:**
- Multiple Choice (2): Conceptual understanding (Remember/Understand)
- Short Answer (2): Application and analysis (Apply/Analyze)
- Essay (1): Synthesis and evaluation (Evaluate/Synthesize)

---

## Part A: Multiple Choice (10 points each)

### Question 1: Pre-trained Language Models

**Which statement correctly describes the key difference between BERT and GPT architectures and their implications for NLP tasks?**

A) BERT uses causal attention while GPT uses bidirectional attention, making BERT better for generation tasks and GPT better for understanding tasks

B) BERT uses bidirectional context through masked language modeling, making it excel at understanding tasks, while GPT uses causal (left-to-right) language modeling, making it excel at generation tasks

C) BERT and GPT use identical attention mechanisms but differ only in model size, with GPT being larger and therefore better at all tasks

D) BERT processes text word-by-word sequentially like an RNN, while GPT processes all tokens in parallel like a CNN

---

### Question 2: Attention Mechanism Properties

**A researcher is analyzing attention weights in a Transformer model processing the sentence "The cat that chased the mouse ran away." Which statement about attention behavior is most accurate?**

A) In self-attention, "ran" should attend primarily to "away" because they are adjacent words in the sentence

B) In self-attention, "ran" should attend strongly to "cat" because "cat" is the subject performing the action "ran," despite the intervening relative clause

C) Self-attention weights are determined solely by positional encoding, so "ran" attends most to nearby words regardless of grammatical relationships

D) The attention mechanism cannot capture relationships between "ran" and "cat" because they are separated by more than 3 tokens

---

## Part B: Short Answer (15 points each)

### Question 3: Tokenization Strategy Selection

**Context:** You are deploying NLP models for three different applications. For each application, recommend a tokenization strategy and justify your choice.

**Applications:**

a) **Social media sentiment analysis** processing tweets with hashtags, emojis, @mentions, and frequent misspellings (5 points)

b) **Medical report entity extraction** from clinical notes with drug names, dosages (e.g., "500mg q4h"), and medical abbreviations (5 points)

c) **Multilingual customer support** handling queries in 50+ languages including low-resource languages like Swahili and Thai (5 points)

---

### Question 4: LSTM vs Transformer Comparison

**Context:** A colleague argues that "LSTMs are obsolete—Transformers are better in every way, so we should always use Transformers."

**Tasks:**

a) Explain one fundamental advantage Transformers have over LSTMs and why it matters for training on large datasets (5 points)

b) Describe two scenarios where LSTMs might still be preferable to Transformers (5 points)

c) Explain the vanishing gradient problem in LSTMs and how the cell state architecture addresses it. Include the mathematical intuition. (5 points)

---

## Part C: Essay (30 points)

### Question 5: Modern NLP System Design

**Prompt:** A legal technology startup is building an AI-powered contract analysis system with the following requirements:

1. **Extract key clauses** (termination, liability, payment terms) from contracts (PDF/Word, 10-100 pages)
2. **Compare clauses** against company standard templates, flagging deviations
3. **Summarize** contract terms for non-legal stakeholders
4. **Answer questions** like "What is the notice period for termination?"
5. **Support multiple jurisdictions** (US, UK, EU) with different legal terminology

**Your essay must address:**

1. **Architecture Design** (8 points)
   - Document processing pipeline (handling long documents)
   - Model selection for each subtask
   - How components interact

2. **Handling Long Documents** (7 points)
   - Strategy for documents exceeding model context limits
   - Trade-offs between chunking approaches
   - Maintaining cross-reference understanding

3. **Domain Adaptation** (7 points)
   - Pre-training vs fine-tuning strategy
   - Data requirements and annotation approach
   - Handling legal jargon and jurisdiction-specific terminology

4. **Evaluation and Deployment** (8 points)
   - Metrics for each subtask
   - Quality assurance for legal accuracy
   - Production considerations

**Evaluation Criteria:**
- Technical accuracy of architectural choices
- Thoughtful analysis of trade-offs
- Practical considerations for legal domain
- Well-reasoned design decisions

**Word Limit:** 600-800 words

---

## Answer Key

### Question 1: Pre-trained Language Models

**Correct Answer: B**

**Explanation:**

| Statement | Assessment |
|-----------|------------|
| **A** | Incorrect. It reverses the architectures. BERT uses bidirectional attention; GPT uses causal. |
| **B** | Correct. BERT's masked LM sees both left and right context (bidirectional), ideal for classification/NER. GPT's causal LM predicts left-to-right, ideal for text generation. |
| **C** | Incorrect. The attention mechanisms are fundamentally different (bidirectional vs causal masking), not just size differences. |
| **D** | Incorrect. Both BERT and GPT are Transformers that process tokens in parallel using attention, not sequentially like RNNs. |

**Key Distinctions:**

```
BERT (Bidirectional):
  Input: "The [MASK] sat on the mat"
  Can see: "The" and "sat on the mat" when predicting mask
  Best for: Understanding tasks (classification, NER, QA)

GPT (Causal):
  Input: "The cat sat"
  Can see: Only "The cat" when predicting "sat"
  Best for: Generation tasks (text completion, dialogue)
```

**Understanding Gap:** If you selected A, review the difference between bidirectional and causal attention. If you selected C or D, review Transformer architecture fundamentals.

---

### Question 2: Attention Mechanism Properties

**Correct Answer: B**

**Explanation:**

| Statement | Assessment |
|-----------|------------|
| **A** | Incorrect. Attention is not primarily based on adjacency. Self-attention learns semantic/syntactic relevance regardless of distance. |
| **B** | Correct. Self-attention learns grammatical relationships. "ran" (verb) should attend to "cat" (subject) because attention captures subject-verb agreement despite intervening clause. |
| **C** | Incorrect. Positional encoding provides position information but doesn't determine attention weights. The Q/K dot products based on content determine attention. |
| **D** | Incorrect. Self-attention can capture arbitrary-distance relationships—this is its key advantage over RNNs. |

**Attention Pattern for "The cat that chased the mouse ran away":**

```
When computing attention for "ran":
  "The"    → 0.05 (determiner, low relevance)
  "cat"    → 0.45 (SUBJECT of "ran" - high attention)
  "that"   → 0.05 (relative pronoun)
  "chased" → 0.10 (verb in relative clause)
  "the"    → 0.03
  "mouse"  → 0.07 (object of relative clause)
  "ran"    → 0.15 (self-attention)
  "away"   → 0.10 (complements "ran")

Key insight: "cat" receives high attention despite being 5 positions away
```

**Understanding Gap:** If you selected A or C, review how attention weights are computed from Q/K similarity, not position. If you selected D, review self-attention's ability to capture long-range dependencies.

---

### Question 3: Tokenization Strategy Selection

**Model Answer:**

**a) Social Media Sentiment Analysis**

**Recommendation:** BPE with social media vocabulary + preprocessing

**Justification:**
- **Subword tokenization (BPE):** Handles misspellings ("gooood" → "go", "oo", "od") and novel words
- **Special preprocessing:**
  - Normalize hashtags: "#MachineLearning" → "machine learning"
  - Preserve emojis: Critical sentiment signals, keep as tokens
  - Handle @mentions: Replace with [USER] token or preserve
  - Character normalization: "loooove" → "love" before tokenization

**Alternative consideration:** Character-level model for extreme robustness to misspellings, but loses semantic grouping.

```
Example transformation:
Original:  "OMG @friend this movie is soooo goood!!! 😍 #BestMovieEver"
Processed: "omg [USER] this movie is so good ! 😍 best movie ever"
Tokenized: ["om", "g", "[USER]", "this", "movie", "is", "so", "good", "!", "😍", ...]
```

---

**b) Medical Report Entity Extraction**

**Recommendation:** Domain-specific BioWordPiece + medical preprocessing

**Justification:**
- **BioWordPiece (PubMedBERT tokenizer):** Pre-trained on medical literature, knows medical vocabulary
- **Critical preprocessing:**
  - Abbreviation expansion: "pt" → "patient", "hx" → "history"
  - Dosage normalization: "500mg q4h" → "500 mg every 4 hours"
  - Preserve drug names: Don't split "acetaminophen" into subwords
- **Custom vocabulary additions:** Add frequent drugs, procedures to prevent excessive subword splitting

```
Example:
Original:  "Pt presents with SOB, started on amoxicillin 500mg q8h"
Processed: "patient presents with shortness of breath, started on amoxicillin 500 mg every 8 hours"
Tokenized: ["patient", "presents", "with", "shortness", "of", "breath", ",", "started", "on", "amoxicillin", ...]
```

**Rationale:** Medical entity recognition requires preserving meaningful units (drug names, dosages). Generic tokenizers may split "amoxicillin" poorly.

---

**c) Multilingual Customer Support (50+ languages)**

**Recommendation:** SentencePiece Unigram (from mBERT or XLM-R)

**Justification:**
- **Language-agnostic:** SentencePiece treats text as raw bytes, handles any script (Thai, Arabic, Chinese)
- **Shared vocabulary:** Cross-lingual representations enable transfer from high to low-resource languages
- **No language-specific preprocessing:** Works consistently across 50+ languages
- **Handles code-mixing:** "My bestellung is not arrived" (German+English) tokenized sensibly

**Model recommendation:** XLM-RoBERTa tokenizer (250K vocabulary, 100 languages)

```
Cross-lingual example:
English: "Where is my order?" → ["▁Where", "▁is", "▁my", "▁order", "?"]
German:  "Wo ist meine Bestellung?" → ["▁Wo", "▁ist", "▁meine", "▁Best", "ellung", "?"]
Thai:    "คำสั่งซื้อของฉันอยู่ที่ไหน" → [Thai subword tokens]

All share the same embedding space for downstream tasks
```

---

### Question 4: LSTM vs Transformer Comparison

**Model Answer:**

**a) Transformer Advantage: Parallelization (5 points)**

**Key advantage:** Transformers process all sequence positions in parallel; LSTMs must process sequentially.

**Why it matters for large-scale training:**

```
LSTM training (sequence length n):
  Time step 1: h₁ = LSTM(x₁, h₀)
  Time step 2: h₂ = LSTM(x₂, h₁)  ← Must wait for h₁
  ...
  Time step n: hₙ = LSTM(xₙ, hₙ₋₁) ← Must wait for all previous

  Training time: O(n) sequential operations, cannot parallelize

Transformer training:
  All positions compute attention simultaneously:
  Output = softmax(QKᵀ/√d) V  ← Single parallel operation

  Training time: O(1) parallel operations (given enough compute)
```

**Practical impact:**
- Transformer processes 10K tokens in similar time to 100 tokens
- LSTM training time scales linearly with sequence length
- GPT-3 (175B parameters) training would be infeasible with LSTMs

---

**b) Two Scenarios Where LSTMs Are Preferable (5 points)**

**Scenario 1: Real-time Streaming Applications**

```
Use case: Live speech transcription with <100ms latency

LSTM advantage:
- Process incrementally: Each new audio frame → update hidden state → output
- Constant memory: O(1) hidden state regardless of history
- No recomputation: Previous steps don't need revisiting

Transformer challenge:
- Must recompute attention over growing context
- KV cache helps but still O(n) memory
- Latency increases with sequence length
```

**Scenario 2: Resource-Constrained Edge Deployment**

```
Use case: NLP on mobile device or IoT sensor

LSTM advantage:
- Smaller model footprint (fewer parameters for simple tasks)
- O(1) inference memory (just hidden state)
- Well-optimized implementations on edge hardware

Transformer challenge:
- Attention requires O(n²) operations even for inference
- KV cache requires O(n) memory
- Larger minimum model size for good performance
```

---

**c) Vanishing Gradient and LSTM Cell State (5 points)**

**The Vanishing Gradient Problem:**

```
Standard RNN gradient through T time steps:
∂L/∂h₁ = ∂L/∂hₜ × ∏ᵢ₌₂ᵀ ∂hᵢ/∂hᵢ₋₁

Each factor: ∂hᵢ/∂hᵢ₋₁ = Wₕₕ × diag(tanh'(·))

Problem: If ||Wₕₕ|| < 1 or tanh' small:
  Product of T small numbers → exponentially small gradient
  For T=100: 0.9¹⁰⁰ ≈ 10⁻⁵ → gradient vanishes
```

**How LSTM Cell State Solves This:**

```
LSTM cell state update:
cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ

Gradient through cell state:
∂cₜ/∂cₜ₋₁ = fₜ  (forget gate value, typically 0.7-0.99)

Key insight: ADDITIVE update, not multiplicative!

Through T steps:
∂cₜ/∂c₁ = ∏ᵢ₌₂ᵀ fᵢ

If forget gates ≈ 0.9:
  0.9¹⁰⁰ ≈ 10⁻⁵ (still small, but...)

Critical difference:
- Forget gate is LEARNED and can be ~1 when needed
- Model learns: "keep this information" → f ≈ 1 → gradient preserved
- vs. RNN where Wₕₕ is fixed for all contexts
```

**Mathematical intuition:**
```
RNN:  h_t = tanh(Wₕₕ h_{t-1} + ...)  ← Multiplicative, always squished
LSTM: c_t = f_t × c_{t-1} + ...       ← Additive, can be identity

When f_t = 1 and i_t = 0: c_t = c_{t-1} (perfect preservation)
Gradient flows unchanged through "gradient highway"
```

---

### Question 5: Modern NLP System Design

**Rubric (30 points total):**

| Component | Excellent (Full) | Adequate (Half) | Insufficient (Minimal) |
|-----------|------------------|-----------------|------------------------|
| Architecture Design (8) | Complete pipeline with justified model choices | Reasonable but incomplete design | Missing major components |
| Long Documents (7) | Concrete chunking strategy with trade-offs | General approach without specifics | Ignores length challenge |
| Domain Adaptation (7) | Clear pre-training/fine-tuning plan with data requirements | General strategy without details | No domain consideration |
| Evaluation & Deployment (8) | Task-specific metrics + legal accuracy measures | Basic metrics only | Missing evaluation plan |

**Model Answer:**

**1. Architecture Design**

The contract analysis system requires a multi-stage pipeline integrating document understanding, information extraction, and generation capabilities.

**Document Processing Pipeline:**
```
PDF/Word Input → Document Parsing (layout-aware) → Section Segmentation
    → Clause Extraction → Cross-reference Resolution → Task-Specific Heads
```

**Model Selection:**
- **Document Understanding:** LayoutLMv3 or Donut for handling mixed text/table/formatting
- **Clause Extraction:** Legal-BERT + NER head for identifying clause types
- **Clause Comparison:** Sentence-BERT for semantic similarity against templates
- **Summarization:** Long-form T5 (LED or Longformer-Encoder-Decoder)
- **Question Answering:** Legal-BERT fine-tuned on extractive QA

The pipeline uses a shared legal domain encoder (Legal-BERT pre-trained on contracts and case law) as the backbone, with task-specific heads for each function. This maximizes parameter sharing while allowing specialization.

**2. Handling Long Documents**

Contracts of 10-100 pages (10K-100K tokens) exceed standard Transformer context limits (512-4096 tokens). Three complementary strategies address this:

**Strategy 1: Hierarchical Processing**
```
Document → Pages → Paragraphs → Clauses
Each level: Encode with Longformer → Aggregate representations
```

**Strategy 2: Sliding Window with Overlap**
```
Chunk documents into 1024-token windows with 256-token overlap
Run extraction on each chunk → merge predictions
Challenge: Clauses spanning chunk boundaries
Solution: Increase overlap at section boundaries (detected via formatting)
```

**Strategy 3: Retrieval-Augmented Approach for QA**
```
Index all clause embeddings → Given question, retrieve relevant chunks
Run QA model only on retrieved context
Scales to any document length with O(log n) retrieval
```

**Trade-off Analysis:**
- Hierarchical: Best for summarization (needs global view), more complex
- Sliding window: Simplest for extraction, may miss cross-references
- Retrieval: Best for QA, requires index maintenance

Recommendation: Hierarchical for summarization, sliding window for extraction, retrieval for QA—each task uses optimal strategy.

**3. Domain Adaptation**

Legal text differs substantially from general corpora in vocabulary, structure, and reasoning patterns.

**Pre-training Strategy:**
Continue pre-training BERT on 50M tokens of contracts, legal opinions, and regulatory text using MLM objective. This "legal language modeling" teaches:
- Legal terminology ("indemnification," "force majeure")
- Contract structure (recitals, definitions, operative clauses)
- Jurisdiction-specific variations

**Fine-tuning Data Requirements:**
| Task | Annotation Need | Data Source |
|------|-----------------|-------------|
| Clause extraction | 5K contracts with clause labels | Historical reviewed contracts |
| Comparison | 1K deviation annotations | Legal team quality samples |
| Summarization | 2K contract-summary pairs | Existing executive summaries |
| QA | 3K question-answer pairs | Generate from clause extractions |

**Annotation Strategy:**
Use legal professionals for initial 500 examples per task, then active learning to expand. Two annotators with legal background; third for adjudication. Inter-annotator agreement target: κ > 0.80.

**Jurisdiction Handling:**
Train single multilingual model on US/UK/EU contracts with jurisdiction tokens. Alternative: Jurisdiction-specific adapters (LoRA) sharing base model. The shared base captures universal contract concepts while adapters learn jurisdiction-specific terminology.

**4. Evaluation and Deployment**

**Task-Specific Metrics:**
- Clause Extraction: Precision 95%+, Recall 90%+ (higher precision for risk—missing a clause can be caught in review; hallucinating one is dangerous)
- Comparison: Semantic similarity F1 > 85%; flag rate precision > 90%
- Summarization: ROUGE-L > 0.4; legal accuracy via human eval (factual consistency score)
- QA: Exact match > 70%; passage-level F1 > 85%

**Legal Accuracy Assurance:**
1. **Citation verification:** Every extracted clause must map to document location
2. **Hallucination detection:** Cross-check generated summaries against source
3. **Confidence thresholds:** Flag low-confidence extractions for human review (< 0.9)
4. **Audit trail:** Log all model decisions with timestamps for legal defensibility

**Production Considerations:**
- Latency: Process 100-page contract in < 60 seconds (batch processing acceptable)
- Throughput: 1000 contracts/day with 4x A10 GPUs
- Security: On-premise deployment (contracts are confidential)
- Monitoring: Track extraction accuracy weekly via sampling; alert on distribution drift
- Human-in-the-loop: 10% random sample review; all high-stakes clauses (liability, termination) require human confirmation before final output

The system prioritizes precision over recall given legal stakes, uses multi-stage validation, and maintains human oversight for high-risk decisions—balancing AI efficiency with legal accountability.

---

## Performance Interpretation Guide

| Score | Mastery Level | Recommended Action |
|-------|---------------|-------------------|
| 90-100% | **Mastery** | Ready for NLP research and production systems |
| 75-89% | **Proficient** | Review specific gaps, implement NLP pipelines |
| 60-74% | **Developing** | Re-study core NLP concepts |
| Below 60% | **Foundational** | Complete re-review of Lesson 9 |

---

## Review Recommendations by Question

| If You Struggled With | Review These Sections |
|----------------------|----------------------|
| Question 1 | Lesson 9: Pre-trained models (BERT vs GPT) |
| Question 2 | Lesson 9: Attention mechanism, Transformer architecture |
| Question 3 | Lesson 9: Tokenization, text preprocessing |
| Question 4 | Lesson 9: RNNs, LSTMs, Transformers comparison |
| Question 5 | Lesson 9: NLP tasks, modern techniques, system design |

---

*Generated from Lesson 9: Natural Language Processing | Quiz Skill*
