# Assessment Quiz: Natural Language Processing

**Source Material:** notes/natural-language-processing/flashcards/natural-language-processing-flashcards.md
**Practice Problems:** notes/natural-language-processing/practice/natural-language-processing-practice-problems.md
**Concept Map:** notes/natural-language-processing/concept-maps/natural-language-processing-concept-map.md
**Original Study Notes:** notes/natural-language-processing/natural-language-processing-study-notes.md
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
**Concept Tested:** Attention Mechanism
**Source Section:** Core Concepts 7
**Concept Map Node:** Attention (12 connections)
**Related Flashcard:** Card 2
**Related Practice Problem:** P2

In the scaled dot-product attention formula `Attention(Q,K,V) = softmax(QK^T / √d_k)V`, what is the purpose of dividing by √d_k?

A) To reduce the number of parameters in the model and improve training speed

B) To prevent softmax saturation when dot products become large, which would cause vanishing gradients

C) To normalize the output vectors to unit length for stable layer normalization

D) To enable multi-head attention by scaling down each head's contribution

---

### Question 2 | Multiple Choice
**Cognitive Level:** Remember/Understand
**Concept Tested:** BERT vs GPT Architecture
**Source Section:** Core Concepts 9
**Concept Map Node:** BERT (7), GPT (6)
**Related Flashcard:** Card 4
**Related Practice Problem:** P3

Which statement best describes the fundamental architectural difference between BERT and GPT?

A) BERT uses recurrent layers while GPT uses only attention layers, making GPT faster for inference

B) BERT is an encoder-only model with bidirectional attention trained via masked language modeling, while GPT is a decoder-only model with causal attention trained via next-token prediction

C) BERT uses absolute positional encodings while GPT uses relative positional encodings, affecting their ability to handle long sequences

D) BERT has more parameters than GPT, making it more accurate but slower for fine-tuning tasks

---

### Question 3 | Short Answer
**Cognitive Level:** Apply/Analyze
**Concept Tested:** Word Embeddings
**Source Section:** Core Concepts 3
**Concept Map Node:** Embeddings (8 connections)
**Related Flashcard:** Card 1
**Related Practice Problem:** P1
**Expected Response Length:** 3-4 sentences

Word2Vec embeddings are called "static" while BERT embeddings are called "contextual." Explain what this distinction means with a concrete example, and describe one significant limitation of static embeddings that contextual embeddings address.

---

### Question 4 | Short Answer
**Cognitive Level:** Apply/Analyze
**Concept Tested:** Transformer Architecture
**Source Section:** Core Concepts 8
**Concept Map Node:** Transformer (10), Positional Encoding (3)
**Related Flashcard:** Card 3
**Related Practice Problem:** P5
**Expected Response Length:** 3-4 sentences

Unlike RNNs, the Transformer's self-attention mechanism is permutation-invariant, meaning it produces the same output regardless of input order. Explain why this is problematic for language understanding and how positional encodings solve this issue.

---

### Question 5 | Essay
**Cognitive Level:** Evaluate/Synthesize
**Concepts Tested:** Complete NLP Pipeline
**Source Sections:** All Core Concepts
**Concept Map:** Full pathway traversal
**Related Flashcard:** Card 5
**Related Practice Problem:** P3, P4
**Expected Response Length:** 1-2 paragraphs

You are building a customer feedback analysis system for an e-commerce company that must: (1) classify reviews into sentiment categories (positive, neutral, negative), (2) extract product aspects mentioned (price, quality, shipping, customer service), (3) handle reviews in English, Spanish, and French, and (4) process 50,000 reviews daily with results available within 1 hour.

Design a complete solution addressing: (a) model architecture selection with justification for the multilingual requirement; (b) how you would handle the dual task of classification and aspect extraction; (c) training data strategy given limited labeled multilingual data; (d) how to ensure the system meets the throughput requirement; and (e) what metrics you would monitor in production to ensure quality.

**Evaluation Criteria:**
- [ ] Selects appropriate multilingual model with reasoning
- [ ] Addresses multi-task architecture design
- [ ] Proposes realistic training data strategy
- [ ] Calculates throughput feasibility
- [ ] Identifies meaningful quality metrics

---

## Answer Key

### Question 1 | Multiple Choice
**Correct Answer:** B

**Explanation:**
The scaling factor √d_k prevents softmax saturation when the dimension d_k is large:

**Why scaling is needed:**
- Dot products QK^T grow in magnitude with d_k (variance ≈ d_k for random vectors)
- Large values cause softmax to produce near-one-hot distributions
- Near-one-hot → gradients approach zero (saturation)
- Dividing by √d_k keeps variance ≈ 1, maintaining healthy gradient flow

**Mathematical illustration:**
```
d_k = 64:  dot products could be ~8 (√64)
Without scaling: softmax([8, 0, 0]) ≈ [0.9997, 0.0001, 0.0001]
With scaling:    softmax([1, 0, 0]) ≈ [0.58, 0.21, 0.21]
```
The scaled version maintains gradient flow for learning.

**Why Other Options Are Incorrect:**
- A) Scaling doesn't reduce parameters; it's a mathematical operation during forward pass
- C) Output normalization is handled by layer normalization, not attention scaling
- D) Multi-head attention concatenates heads; scaling is independent of multi-head design

**Understanding Gap Indicator:**
If answered incorrectly, review the attention mechanism mathematics in Card 2 and Practice Problem 2.

---

### Question 2 | Multiple Choice
**Correct Answer:** B

**Explanation:**
BERT and GPT represent two fundamental paradigms in pre-trained language models:

| Aspect | BERT | GPT |
|--------|------|-----|
| Architecture | Encoder-only Transformer | Decoder-only Transformer |
| Attention | Bidirectional (sees all tokens) | Causal (left-to-right only) |
| Pre-training | Masked LM (predict [MASK] tokens) | Causal LM (predict next token) |
| Best for | Understanding (classification, NER, QA) | Generation (text completion, dialogue) |

**Bidirectional vs Causal:**
```
BERT:  "The [MASK] sat on the mat" → sees both "The" and "sat on the mat"
GPT:   "The cat" → predicts "sat" seeing only "The cat"
```

**Why Other Options Are Incorrect:**
- A) Both use only attention layers (no recurrence); the difference is attention direction
- C) Both can use learned or sinusoidal positional encodings; this isn't the key difference
- D) Size varies by version (BERT-large: 340M, GPT-3: 175B); size isn't the architectural difference

**Understanding Gap Indicator:**
If answered incorrectly, review BERT/GPT comparison in Card 4 and the pre-training section in study notes.

---

### Question 3 | Short Answer
**Model Answer:**

"Static" embeddings like Word2Vec assign a single, fixed vector to each word regardless of context—"bank" has the same representation whether referring to a financial institution or a river bank. "Contextual" embeddings from BERT generate different vectors for the same word depending on its surrounding context, so "bank" in "I deposited money at the bank" would have a different representation than "bank" in "We walked along the river bank."

The key limitation of static embeddings is their inability to handle polysemy (words with multiple meanings). Since static embeddings collapse all meanings into one vector, they cannot distinguish between different word senses, which leads to errors in tasks requiring semantic understanding. Contextual embeddings solve this by computing representations dynamically based on the full sentence, allowing the model to disambiguate word meanings from context.

**Key Components Required:**
- [ ] Explains static = same vector regardless of context
- [ ] Explains contextual = different vectors based on context
- [ ] Provides concrete polysemy example (bank, bat, etc.)
- [ ] Identifies polysemy/word sense disambiguation as limitation

**Partial Credit Guidance:**
- Full credit: All four components with clear explanation
- Partial credit: Understands the distinction but vague on limitation
- No credit: Confuses static/contextual or misidentifies the limitation

**Understanding Gap Indicator:**
If answered poorly, review Word Embeddings (Concept 3) and BERT representations (Concept 9).

---

### Question 4 | Short Answer
**Model Answer:**

Self-attention computes relationships between all pairs of positions simultaneously without any inherent notion of sequence order—mathematically, `Attention(Q,K,V)` produces identical outputs whether the input is "the cat sat" or "sat cat the." This is catastrophic for language understanding because word order fundamentally determines meaning ("dog bites man" vs "man bites dog"), and grammar depends on sequential structure.

Positional encodings solve this by adding position-dependent vectors to token embeddings before they enter the Transformer. These encodings (either learned or sinusoidal functions of position) ensure that each token carries information about its absolute position in the sequence. The model can then learn to use this positional information within attention, allowing it to distinguish "the cat sat" from any permutation while still benefiting from parallel computation.

**Key Components Required:**
- [ ] Explains permutation invariance = same output regardless of order
- [ ] Explains why order matters for language (meaning, grammar)
- [ ] Describes positional encoding mechanism (added to embeddings)
- [ ] Notes that encodings provide position information to attention

**Partial Credit Guidance:**
- Full credit: Clear explanation of problem and solution with mechanism
- Partial credit: Understands one of problem or solution but not both
- No credit: Misunderstands permutation invariance or positional encoding

**Understanding Gap Indicator:**
If answered poorly, review Transformer Architecture (Concept 8) and Practice Problem 5.

---

### Question 5 | Essay
**Model Answer:**

**(a) Model Architecture Selection:**

I would use **XLM-RoBERTa** (or mBERT) as the base model, specifically the `xlm-roberta-base` variant with 270M parameters. This choice is justified because:
- Pre-trained on 100 languages including English, Spanish, and French
- Cross-lingual transfer: can train primarily on English data and transfer to other languages
- Single model handles all three languages, simplifying deployment
- Strong performance on both classification and token-level tasks

**(b) Multi-Task Architecture:**

I would implement a **multi-task learning** approach with a shared XLM-RoBERTa backbone and two task-specific heads:

```
XLM-RoBERTa Backbone (shared)
         │
    ┌────┴────┐
    ▼         ▼
[CLS] Head   Token Head
(Sentiment)  (Aspect NER)
 3-way       BIO tagging
```

- **Sentiment classification:** Use [CLS] token representation → linear layer → 3 classes
- **Aspect extraction:** Sequence labeling with BIO tags for each aspect category → BiLSTM-CRF on top of token representations

Joint training with weighted loss: `L = L_sentiment + λ × L_aspect` where λ balances the tasks.

**(c) Training Data Strategy:**

Given limited multilingual labeled data:

1. **English-first approach:** Collect and label 10K English reviews (most available)
2. **Zero-shot transfer:** XLM-RoBERTa's cross-lingual abilities enable Spanish/French without labeled data
3. **Translation augmentation:** Machine-translate labeled English data to Spanish/French
4. **Active learning:** Deploy initial model, collect low-confidence predictions for human labeling in each language
5. **Few-shot fine-tuning:** Add 500-1000 labeled examples per non-English language to calibrate

Target: 10K English + 1K each Spanish/French for production quality.

**(d) Throughput Calculation:**

Requirements: 50,000 reviews in 1 hour = 14 reviews/second

**Single GPU throughput:**
- XLM-RoBERTa-base: ~50ms per review (batch size 1)
- With batching (batch=32): ~10ms per review = 100 reviews/second

**Architecture:**
- Use ONNX Runtime for 2× speedup → 200 reviews/second per GPU
- Single GPU provides 14× headroom for the requirement
- Add second GPU for redundancy and burst handling

**Optimization:**
- Quantize to INT8 for additional 2× speedup
- Use dynamic batching to maximize throughput
- Cache results for duplicate/similar reviews

**(e) Production Quality Metrics:**

| Metric | Threshold | Purpose |
|--------|-----------|---------|
| Sentiment accuracy | >85% | Core task quality |
| Aspect F1 | >80% | Extraction quality |
| Per-language accuracy | Within 5% of English | Cross-lingual fairness |
| Latency p99 | <200ms | User experience |
| Confidence distribution | Monitor drift | Model degradation |
| Low-confidence rate | <15% | Human review volume |

**Monitoring dashboard:**
- Daily accuracy sampling (human review of 100 random predictions)
- Alert on >10% drop in confidence scores (distribution shift)
- A/B testing for model updates
- Language-specific breakdowns to catch per-language degradation

**Evaluation Rubric:**

| Criterion | Excellent (4) | Proficient (3) | Developing (2) | Beginning (1) |
|-----------|---------------|----------------|----------------|---------------|
| Model Selection | XLM-RoBERTa with multilingual justification | Reasonable multilingual choice | Generic BERT mentioned | No model specified |
| Multi-Task Design | Shared backbone + specific heads with loss | Two-model or basic multi-task | Mentions both tasks | Single task only |
| Data Strategy | Transfer + augmentation + active learning | 2 strategies mentioned | Vague "need more data" | No strategy |
| Throughput | Calculation with batching/optimization | Addresses requirement generally | Mentions speed | Ignores constraint |
| Quality Metrics | 5+ specific metrics with thresholds | 3-4 metrics | 1-2 generic metrics | No metrics |

**Understanding Gap Indicator:**
If response lacks depth, this may indicate:
- Limited experience with multilingual NLP
- Weak understanding of multi-task learning
- Insufficient production deployment knowledge

---

## Diagnostic Feedback & Review Recommendations

### Performance-Based Review Guide

| If Incorrect On... | Gap Identified | Review Section | Priority |
|--------------------|----------------|----------------|----------|
| Question 1 | Attention mechanics | Core Concepts 7 + Practice P2 | High |
| Question 2 | Pre-trained model architectures | Core Concepts 9 + Flashcard 4 | High |
| Question 3 | Embedding types | Core Concepts 3 + Flashcard 1 | Medium |
| Question 4 | Transformer fundamentals | Core Concepts 8 + Practice P5 | Medium |
| Question 5 | System integration | All sections | Low |

### Targeted Review Recommendations

#### For Multiple Choice Errors (Questions 1-2)
**Indicates:** Foundational concept gaps
**Action:** Review:
- Study Notes: Core Concepts 7 (Attention) and 9 (BERT/GPT)
- Flashcards: Cards 2, 3, and 4
- Practice Problems: P2 (attention calculation) and P3 (fine-tuning)
**Focus On:** Understanding the mechanics, not just memorizing architectures

#### For Short Answer Errors (Questions 3-4)
**Indicates:** Application difficulties
**Action:** Practice:
- Practice Problems: P1 (tokenization) and P5 (Transformer debugging)
- Concept Map: Representation and Transformer clusters
**Focus On:** Connecting concepts to their practical implications

#### For Essay Weakness (Question 5)
**Indicates:** Integration challenges
**Action:** Review interconnections:
- Concept Map: Full pathway traversal
- Practice Problem P4 (NER system design)
- Study Notes: All practical applications
**Focus On:** Building complete systems from components

### Mastery Indicators

- **5/5 Correct:** Strong mastery; ready for NLP projects
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
| flashcards | 5 cards (2E/2M/1H) | Memorization + critical flags |
| practice-problems | 5 problems (1W/2S/1C/1D) | Procedural fluency + common mistakes |
| quiz | 5 questions (2MC/2SA/1E) | Assessment + diagnostic feedback |
