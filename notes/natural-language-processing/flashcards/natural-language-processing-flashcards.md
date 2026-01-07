# Flashcards: Natural Language Processing

**Source:** notes/natural-language-processing/natural-language-processing-study-notes.md
**Concept Map:** notes/natural-language-processing/concept-maps/natural-language-processing-concept-map.md
**Date Generated:** 2026-01-07
**Total Cards:** 5
**Distribution:** 2 Easy | 2 Medium | 1 Hard

---

## Card Distribution Strategy

| Difficulty | Count | Concepts Covered | Selection Rationale |
|------------|-------|------------------|---------------------|
| Easy | 2 | Embeddings, Attention | High centrality (8-12 connections); foundational understanding |
| Medium | 2 | Transformer, BERT/GPT | Critical architecture; pre-training paradigm |
| Hard | 1 | Full NLP Pipeline | Integration across all concepts; system design |

---

## Easy Cards

### Card 1 of 5 | Easy
**Concept:** Word Embeddings
**Centrality:** High (8 connections)
**Related Concepts:** Word2Vec, GloVe, FastText, Dense Representations

#### Front
What are word embeddings, and how do they differ from sparse representations like Bag-of-Words? What key property do they capture?

#### Back
**Word embeddings** are dense, low-dimensional vector representations (typically 100-300 dimensions) of words learned from large text corpora.

**Key differences from Bag-of-Words:**

| Property | Bag-of-Words | Word Embeddings |
|----------|--------------|-----------------|
| Dimensionality | Vocabulary size (10K-100K) | 100-300 |
| Sparsity | Mostly zeros | Dense (all non-zero) |
| Similarity | Orthogonal (no similarity) | Geometric distance = semantic similarity |
| Order | Ignored | Context-aware learning |

**Key property captured:** Semantic similarity through the distributional hypothesis—words occurring in similar contexts have similar meanings.

**Example:** vector("king") - vector("man") + vector("woman") ≈ vector("queen")

**Methods:**
- **Word2Vec:** Skip-gram (predict context) or CBOW (predict target)
- **GloVe:** Global co-occurrence matrix factorization
- **FastText:** Subword embeddings for handling rare/OOV words

#### Mnemonic
**"Sparse to Dense, Far to Close"** — Embeddings compress vocabulary into dense space where similar words cluster together.

#### Common Misconceptions
- ❌ Embeddings capture word order (they don't—static embeddings give same vector regardless of context)
- ❌ Larger dimensions are always better (diminishing returns; 300 often sufficient)
- ❌ Pre-trained embeddings work perfectly for all domains (domain adaptation often needed)

---

### Card 2 of 5 | Easy
**Concept:** Attention Mechanism
**Centrality:** Critical (12 connections)
**Related Concepts:** Self-Attention, Query-Key-Value, Context Vector

#### Front
What is the attention mechanism in NLP? Explain its purpose and the Query-Key-Value framework.

#### Back
**Attention** is a mechanism that allows models to dynamically focus on relevant parts of the input when producing each output, computing weighted combinations based on relevance.

**Purpose:** Solves the bottleneck problem in seq2seq where a fixed-size context vector must compress the entire input sequence.

**Query-Key-Value Framework:**
```
Attention(Q, K, V) = softmax(score(Q, K)) × V
```

| Component | Role | Analogy |
|-----------|------|---------|
| **Query (Q)** | What we're looking for | Search query |
| **Key (K)** | What we match against | Index/tags |
| **Value (V)** | What we retrieve | Actual content |

**Process:**
1. Compute compatibility scores between Query and each Key
2. Apply softmax to get attention weights (sum to 1)
3. Weighted sum of Values = context vector

**Score Functions:**
- **Dot product:** Q · K
- **Scaled dot product:** (Q · K) / √d_k
- **Additive:** v^T tanh(W_q Q + W_k K)

#### Mnemonic
**"QKV = Question, Keywords, Values"** — Ask a question, match keywords, retrieve values.

#### Common Misconceptions
- ❌ Attention replaces RNNs entirely (originally used with RNNs; Transformers later removed RNNs)
- ❌ High attention weight means the model "understands" (correlation, not causation)
- ❌ Attention weights are always interpretable (often distributed across many positions)

---

## Medium Cards

### Card 3 of 5 | Medium
**Concept:** Transformer Architecture
**Centrality:** Critical (10 connections)
**Related Concepts:** Self-Attention, Multi-Head Attention, Positional Encoding, Layer Normalization

#### Front
Describe the Transformer architecture. What are its key components, and why can it process sequences in parallel unlike RNNs?

#### Back
The **Transformer** is a neural architecture that relies entirely on self-attention mechanisms, processing all positions simultaneously without recurrence.

**Key Components:**

```
Input Embeddings + Positional Encoding
              ↓
    ┌─────────────────┐
    │  Multi-Head     │
    │  Self-Attention │ ←── Parallel attention across all positions
    └────────┬────────┘
             ↓ (Add & Norm)
    ┌─────────────────┐
    │  Feed-Forward   │ ←── Position-wise FFN
    │     Network     │
    └────────┬────────┘
             ↓ (Add & Norm)
         [Repeat N×]
```

**Why Parallel (vs. RNN Sequential):**

| RNN | Transformer |
|-----|-------------|
| h_t depends on h_{t-1} | All positions computed simultaneously |
| O(n) sequential steps | O(1) parallel steps |
| Information must flow step-by-step | Direct connections between any positions |

**Critical Components:**
- **Self-Attention:** Relates each position to all others in same sequence
- **Multi-Head:** 8-16 parallel attention operations with different projections
- **Positional Encoding:** Sinusoidal or learned; injects position since attention is permutation-invariant
- **Residual Connections:** Skip connections around each sublayer
- **Layer Normalization:** Stabilizes training

**Scaled Dot-Product Attention:**
```
Attention(Q,K,V) = softmax(QK^T / √d_k) V
```
Scaling by √d_k prevents softmax saturation for large dimensions.

#### Mnemonic
**"STAMP: Self-attention, Transformer, Add-norm, Multi-head, Positional"**

#### Common Misconceptions
- ❌ Transformers understand position inherently (they don't—positional encoding is required)
- ❌ More heads always better (diminishing returns; 8-16 typical)
- ❌ Transformers are only for NLP (now used in vision, audio, proteins, etc.)

#### Critical Flag
⚠️ **Quadratic complexity:** Self-attention is O(n²) in sequence length—problematic for very long documents. This motivates efficient variants (Longformer, BigBird, etc.).

---

### Card 4 of 5 | Medium
**Concept:** BERT and GPT Pre-training
**Centrality:** High (7 connections each)
**Related Concepts:** Masked Language Modeling, Causal Language Modeling, Fine-tuning, Transfer Learning

#### Front
Compare BERT and GPT architectures. What are their pre-training objectives, and for what types of tasks is each better suited?

#### Back
**BERT** (Bidirectional Encoder Representations from Transformers) and **GPT** (Generative Pre-trained Transformer) represent two paradigms for pre-trained language models.

| Aspect | BERT | GPT |
|--------|------|-----|
| **Architecture** | Encoder-only Transformer | Decoder-only Transformer |
| **Attention** | Bidirectional (sees all tokens) | Causal/unidirectional (left-to-right) |
| **Pre-training** | Masked LM + NSP | Causal LM (next token) |
| **Best for** | Understanding tasks | Generation tasks |

**Pre-training Objectives:**

**BERT - Masked Language Modeling (MLM):**
```
Input:  "The [MASK] sat on the mat"
Output: "cat" (predict masked token from bidirectional context)
```
- 15% of tokens masked; predict original token
- Also: Next Sentence Prediction (NSP) - binary classification

**GPT - Causal Language Modeling (CLM):**
```
Input:  "The cat sat on"
Output: "the" (predict next token from left context only)
```
- Standard language modeling objective
- Enables text generation

**Task Suitability:**

| Task Type | Better Model | Reason |
|-----------|--------------|--------|
| Classification | BERT | Bidirectional context |
| NER / Tagging | BERT | Full context at each position |
| Text Generation | GPT | Autoregressive design |
| Summarization | GPT (or BART/T5) | Generation capability |
| Question Answering | BERT | Span extraction |
| Few-shot Learning | GPT-3+ | In-context learning |

**Fine-tuning:** Both use task-specific heads on top of pre-trained base; typically few epochs with lower learning rate.

#### Mnemonic
**"BERT Bids Both ways, GPT Goes forward"** — Bidirectional vs. Generative/autoregressive.

#### Common Misconceptions
- ❌ BERT can generate text well (it can, but not designed for it—use GPT)
- ❌ GPT cannot do classification (it can, but BERT often better)
- ❌ Larger models always win (task and data matter; smaller models can be better for specific domains)

---

## Hard Cards

### Card 5 of 5 | Hard
**Concept:** Complete NLP Pipeline Design
**Centrality:** Integration (spans all concepts)
**Related Concepts:** All core concepts

#### Front
Design a complete NLP pipeline for a customer support ticket classification system that must:
1. Handle multiple languages (English, Spanish, French)
2. Classify tickets into 15 categories
3. Extract key entities (product names, error codes, dates)
4. Process 10,000 tickets per hour
5. Provide confidence scores for human review routing

Address: model selection, preprocessing, training strategy, and production considerations.

#### Back
**Complete Pipeline Design:**

**1. Architecture Selection:**
```
┌─────────────────────────────────────────────────────────┐
│                  Multilingual BERT (mBERT)              │
│         or XLM-RoBERTa (better cross-lingual)           │
└────────────────────────┬────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ↓                ↓                ↓
   ┌─────────┐    ┌───────────┐    ┌──────────────┐
   │ CLS Head│    │ NER Head  │    │ Confidence   │
   │ (15-way)│    │ (Token)   │    │ Calibration  │
   └─────────┘    └───────────┘    └──────────────┘
```

**Model Choice Justification:**
- **XLM-RoBERTa:** Pre-trained on 100 languages; handles code-switching
- **Multi-task:** Shared backbone for classification + NER = efficiency
- **Distilled version:** DistilmBERT for 2× speed with ~95% accuracy

**2. Preprocessing Pipeline:**
```python
# Tokenization
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

# Steps:
1. Language detection (fastText lid.176.bin)
2. Unicode normalization (NFKC)
3. Handle customer-specific entities (product codes → [PRODUCT])
4. Truncation/padding to 256 tokens (covers 95% of tickets)
```

**3. Training Strategy:**

| Phase | Data | Strategy |
|-------|------|----------|
| Pre-training | Use existing XLM-R | No additional pre-training |
| Fine-tuning | 50K labeled tickets | Multi-task: classification + NER jointly |
| Calibration | 5K held-out | Temperature scaling for reliable confidence |

**Loss Function:**
```
L = L_classification + λ × L_NER
  = CrossEntropy(15 classes) + λ × CRF_NER_loss
```

**4. Production Architecture:**
```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Kafka     │───►│  Inference   │───►│  Decision   │
│   Queue     │    │   Service    │    │   Router    │
└─────────────┘    └──────────────┘    └─────────────┘
                          │
                   ┌──────┴──────┐
                   │ ONNX Runtime│  ← Optimized inference
                   │ + Batching  │  ← Batch size 32
                   └─────────────┘
```

**Throughput Calculation:**
- 10,000 tickets/hour = 2.8 tickets/second
- Batch 32, inference ~50ms = 640 tickets/second capacity
- **10× headroom** for spikes

**5. Confidence-Based Routing:**

| Confidence | Action |
|------------|--------|
| > 0.95 | Auto-route to category |
| 0.70 - 0.95 | Auto-route + flag for QA sampling |
| < 0.70 | Human review queue |

**Calibration:** Temperature scaling on validation set ensures confidence reflects true accuracy.

**6. Monitoring & Feedback Loop:**
- Track accuracy by category, language, confidence band
- Log low-confidence predictions for active learning
- Weekly model refresh with newly labeled data
- A/B test model updates before full deployment

#### Production Considerations

| Concern | Solution |
|---------|----------|
| Latency | ONNX + batching; p99 < 100ms |
| Cold start | Model pre-loaded in memory |
| Versioning | MLflow for model registry |
| Explainability | Attention visualization for audits |
| Bias monitoring | Track accuracy by language, check for disparities |

#### Common Misconceptions
- ❌ One model per language is better (multilingual models benefit from transfer)
- ❌ Higher confidence = correct (calibration essential; overconfident models common)
- ❌ Deploy and forget (drift detection and retraining critical)

---

## Anki Export Format

```
# Card 1 - Easy - Word Embeddings
What are word embeddings, and how do they differ from sparse representations like Bag-of-Words? What key property do they capture?	Word embeddings are dense, low-dimensional vectors (100-300 dims) capturing semantic similarity through distributional hypothesis. Unlike BoW (sparse, vocabulary-sized, no similarity), embeddings place similar words geometrically close. Methods: Word2Vec (Skip-gram/CBOW), GloVe (co-occurrence), FastText (subword).	nlp embeddings

# Card 2 - Easy - Attention Mechanism
What is the attention mechanism in NLP? Explain its purpose and the Query-Key-Value framework.	Attention dynamically focuses on relevant input parts using QKV framework: Query (what we seek), Key (what we match), Value (what we retrieve). Attention(Q,K,V) = softmax(score(Q,K)) × V. Solves seq2seq bottleneck by allowing variable context instead of fixed vector.	nlp attention

# Card 3 - Medium - Transformer Architecture
Describe the Transformer architecture. What are its key components, and why can it process sequences in parallel unlike RNNs?	Transformer uses self-attention without recurrence, enabling parallel processing (O(1) vs O(n)). Components: Multi-head self-attention, positional encoding (since attention is position-agnostic), layer norm, residual connections, FFN. Attention(Q,K,V) = softmax(QK^T/√d_k)V. O(n²) complexity in sequence length.	nlp transformer

# Card 4 - Medium - BERT vs GPT
Compare BERT and GPT architectures. What are their pre-training objectives, and for what types of tasks is each better suited?	BERT: encoder-only, bidirectional attention, masked LM + NSP pre-training. Best for understanding (classification, NER, QA). GPT: decoder-only, causal attention, next-token prediction. Best for generation (summarization, dialogue). Both fine-tune with task-specific heads.	nlp bert gpt pretrained

# Card 5 - Hard - NLP Pipeline Design
Design a complete NLP pipeline for multilingual ticket classification with entity extraction at scale.	Use XLM-RoBERTa with multi-task heads (classification + NER). Preprocessing: language detection, normalization, subword tokenization. Training: joint loss, temperature-scaled calibration. Production: ONNX + batching for throughput, confidence routing for human review, monitoring for drift.	nlp pipeline production
```

---

## Review Schedule

| Card | First Review | Second Review | Third Review | Mastery Review |
|------|--------------|---------------|--------------|----------------|
| Card 1 (Easy) | Day 1 | Day 3 | Day 7 | Day 14 |
| Card 2 (Easy) | Day 1 | Day 3 | Day 7 | Day 14 |
| Card 3 (Medium) | Day 1 | Day 4 | Day 10 | Day 21 |
| Card 4 (Medium) | Day 2 | Day 5 | Day 12 | Day 25 |
| Card 5 (Hard) | Day 3 | Day 7 | Day 14 | Day 30 |

---

## Cross-References

| Card | Study Notes Section | Concept Map Node | Practice Problem |
|------|---------------------|------------------|------------------|
| Card 1 | Concept 3: Word Embeddings | Embeddings (8 connections) | Problem 1 |
| Card 2 | Concept 7: Attention Mechanism | Attention (12 connections) | Problem 2 |
| Card 3 | Concept 8: Transformer Architecture | Transformer (10 connections) | Problem 2 |
| Card 4 | Concept 9: Pre-trained Models | BERT (7), GPT (6) | Problem 3 |
| Card 5 | All Concepts | Full pipeline traversal | Problem 4, 5 |
