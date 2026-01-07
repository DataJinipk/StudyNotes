# Flashcards: Lesson 9 - Natural Language Processing

**Source:** Lessons/Lesson_9.md
**Subject Area:** AI Learning - Natural Language Processing: From Text Representation to Neural Language Understanding
**Date Generated:** 2026-01-08
**Total Cards:** 5 (2 Easy, 2 Medium, 1 Hard)

---

## Card Distribution

| Difficulty | Count | Bloom's Level | Focus Area |
|------------|-------|---------------|------------|
| Easy | 2 | Remember/Understand | Core concepts, representation methods |
| Medium | 2 | Apply/Analyze | Architecture selection, attention analysis |
| Hard | 1 | Evaluate/Synthesize | Complete NLP system design |

---

## Easy Cards

### Card 1: Text Representation Comparison

**[FRONT]**
Compare the four main text representation methods (Bag-of-Words, TF-IDF, Word2Vec, Contextual Embeddings) in terms of their dimensions, semantic awareness, and handling of polysemy.

**[BACK]**
**Text Representation Methods Comparison:**

| Method | Dimensions | Semantic Similarity | Polysemy Handling | Context |
|--------|-----------|---------------------|-------------------|---------|
| **Bag-of-Words** | Vocabulary size (~100K) | None | None | None |
| **TF-IDF** | Vocabulary size (~100K) | None | None | Document-level |
| **Word2Vec/GloVe** | 100-300 | Yes (cosine similarity) | None (one vector per word) | Training context |
| **BERT Embeddings** | 768-1024 | Yes | Yes (different vectors) | Full sentence |

**Key Distinctions:**

| Aspect | Sparse (BoW/TF-IDF) | Static Dense (Word2Vec) | Contextual (BERT) |
|--------|---------------------|------------------------|-------------------|
| "good" vs "excellent" | Orthogonal vectors | Similar vectors (~0.7 cosine) | Similar in context |
| "bank" (river vs finance) | One dimension | One vector | Different vectors |
| Out-of-vocabulary | UNK or zero | UNK (FastText: subwords) | Subword handling |

**Evolution Insight:** Each generation addresses limitations of the previous:
- BoW → TF-IDF: Weight important terms
- TF-IDF → Word2Vec: Capture semantics
- Word2Vec → BERT: Handle context-dependent meaning

**Difficulty:** Easy | **Bloom's Level:** Remember

---

### Card 2: Attention Mechanism Fundamentals

**[FRONT]**
Explain the attention mechanism in sequence-to-sequence models. What problem does it solve, what are its components (Query, Key, Value), and how does it improve translation of long sentences?

**[BACK]**
**Attention Mechanism Explained:**

**Problem Solved:** The bottleneck problem in basic Seq2Seq
```
Without attention: Entire source sentence compressed into single fixed vector
                   → Information loss, especially for long sentences

With attention: Decoder can "look back" at any source position
              → Direct access to relevant information
```

**Components:**

| Component | Source | Purpose | Analogy |
|-----------|--------|---------|---------|
| Query (Q) | Decoder state | "What am I looking for?" | Search query |
| Key (K) | Encoder states | "What do I contain?" | Index entries |
| Value (V) | Encoder states | "What information to return?" | Actual content |

**Computation:**
```
Score:    e_ij = score(s_{i-1}, h_j)    # Query-Key compatibility
Weights:  α_ij = softmax(e_i)_j         # Normalized to sum to 1
Context:  c_i = Σ_j α_ij · h_j          # Weighted sum of Values
```

**Long Sentence Improvement:**
```
Without attention (sentence length vs. BLEU):
  Length 10: 0.35 BLEU
  Length 30: 0.20 BLEU  ← Severe degradation
  Length 50: 0.10 BLEU

With attention:
  Length 10: 0.40 BLEU
  Length 30: 0.38 BLEU  ← Maintains quality
  Length 50: 0.35 BLEU
```

**Why It Works:**
1. No fixed bottleneck—access all source information
2. Learns soft alignment between languages
3. Provides interpretability through attention weights

**Difficulty:** Easy | **Bloom's Level:** Understand

---

## Medium Cards

### Card 3: LSTM vs Transformer Selection

**[FRONT]**
A team is choosing between LSTM and Transformer architectures for three NLP tasks. For each task, recommend the better architecture and justify your choice considering training data, sequence length, and computational resources.

**Tasks:**
a) Sentiment classification on 10K labeled tweets (max 280 chars)
b) Machine translation with 10M parallel sentences (up to 100 tokens)
c) Real-time streaming speech transcription with 50ms latency requirement

**[BACK]**
**Architecture Selection Analysis:**

**a) Sentiment Classification (10K tweets, 280 chars)**

**Recommendation: Fine-tuned BERT (Transformer)**

| Factor | LSTM | Transformer (BERT) |
|--------|------|-------------------|
| Data size | Sufficient for training | Pre-trained; 10K enough for fine-tuning |
| Sequence length | 280 chars manageable | Easily handles 512 tokens |
| Performance | Good (~85% accuracy) | Excellent (~92% accuracy) |
| Training time | 1-2 hours | 15-30 min fine-tuning |

**Justification:** Pre-trained BERT provides massive advantage for limited labeled data. Transfer learning outweighs LSTM's efficiency.

---

**b) Machine Translation (10M sentences, 100 tokens)**

**Recommendation: Transformer (encoder-decoder)**

| Factor | LSTM Seq2Seq | Transformer |
|--------|-------------|-------------|
| Parallelization | Sequential (slow) | Fully parallel (fast) |
| Long dependencies | Struggles at 100 tokens | Excellent |
| Training throughput | ~10K sentences/hour | ~100K sentences/hour |
| BLEU score | ~25-30 | ~35-40 |

**Justification:** At scale (10M sentences), Transformer's parallel training is essential. Quality advantage is also substantial for this task.

---

**c) Real-time Streaming Transcription (50ms latency)**

**Recommendation: Streaming LSTM/GRU or Transformer with chunking**

| Factor | LSTM | Transformer |
|--------|------|-------------|
| Streaming | Natural (process token by token) | Requires chunking |
| Latency | Low (single forward pass) | Higher (attention over chunk) |
| Causal processing | Built-in | Requires masking |
| Memory | O(1) for hidden state | O(n) for KV cache |

**Justification:** Real-time constraints favor recurrent models. LSTMs process incrementally without recomputing. Transformers need specialized streaming adaptations (chunked attention, caching).

**Hybrid Option:** Use streaming Conformer (convolution + attention) for best of both worlds.

**Difficulty:** Medium | **Bloom's Level:** Apply

---

### Card 4: BERT Pre-training Analysis

**[FRONT]**
Analyze BERT's Masked Language Modeling (MLM) objective. Why does BERT mask 15% of tokens with the 80/10/10 strategy (80% [MASK], 10% random, 10% unchanged)? What would happen with different masking strategies?

**[BACK]**
**BERT Masking Strategy Analysis:**

**The 80/10/10 Strategy:**
```
When a token is selected for masking (15% of tokens):
- 80%: Replace with [MASK] token
- 10%: Replace with random vocabulary token
- 10%: Keep unchanged
```

**Rationale for Each Component:**

| Strategy | Purpose | Without It |
|----------|---------|------------|
| **80% [MASK]** | Primary training signal; model learns to predict | Insufficient training signal |
| **10% Random** | Prevents reliance on [MASK] token; adds noise | Model overfits to [MASK] cue |
| **10% Unchanged** | Teaches model to reconstruct even correct tokens | Model ignores non-[MASK] positions |

**The Train-Test Mismatch Problem:**

```
Training:  "The [MASK] sat on the mat"
Fine-tune: "The cat sat on the mat" (no [MASK] tokens!)

If 100% [MASK]:
- Model learns: "If I see [MASK], predict something"
- At fine-tuning: No [MASK] → model confused
- Embeddings for real words underutilized

With 80/10/10:
- Model learns: "Any position might need prediction"
- All token embeddings remain useful
- Smoother transfer to downstream tasks
```

**Alternative Strategies and Consequences:**

| Strategy | Effect | Quality Impact |
|----------|--------|---------------|
| 100% [MASK] | Train/test mismatch | -2-3% on downstream |
| 50% [MASK], 50% random | Too much noise | -1-2% on downstream |
| 15% unchanged, 85% [MASK] | Similar to 100% [MASK] | -1-2% on downstream |
| Higher mask rate (25%) | Harder task, less context | Mixed results |
| Lower mask rate (5%) | Easier task, less learning | Slower convergence |

**Why 15% Masking Rate:**
- Higher: Too much information loss per forward pass
- Lower: Too few training signals per example
- 15%: Empirically optimal balance

**Mathematical Perspective:**
```
Training efficiency ∝ mask_rate × (1 - mask_rate)
- mask_rate: more predictions per example
- (1 - mask_rate): more context for prediction

Optimal near 15-25% range
```

**Difficulty:** Medium | **Bloom's Level:** Analyze

---

## Hard Cards

### Card 5: Complete NLP Pipeline Design

**[FRONT]**
Design a complete NLP system for a customer support platform that must:
1. Classify incoming tickets into 50 categories
2. Extract customer information (name, order ID, product, issue type)
3. Route urgent tickets with priority scoring
4. Generate suggested responses for agents
5. Support 10 languages with 100K tickets/day

Specify: architecture choices, model selection, training strategy, and production considerations.

**[BACK]**
**Customer Support NLP System Architecture:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CUSTOMER SUPPORT NLP SYSTEM                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐    │
│  │ Language Detect  │   │ Text Normalizer  │   │ PII Redaction    │    │
│  │ (fastText)       │──▶│ (language-aware) │──▶│ (regex + NER)    │    │
│  └──────────────────┘   └──────────────────┘   └──────────────────┘    │
│                                    │                                     │
│                                    ▼                                     │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │            SHARED ENCODER: XLM-RoBERTa-base (multilingual)      │    │
│  │                     (12 languages, 270M params)                  │    │
│  └───────┬─────────────────┬───────────────────┬──────────────────┘    │
│          │                 │                   │                        │
│          ▼                 ▼                   ▼                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │ Classifier   │  │ NER Module   │  │ Priority     │                  │
│  │ (50 classes) │  │ (entities)   │  │ Scorer       │                  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                  │
│         │                 │                 │                           │
│         └─────────────────┼─────────────────┘                           │
│                           ▼                                             │
│              ┌────────────────────────┐                                 │
│              │  Response Generator    │                                 │
│              │  (mT5-base + RAG)      │                                 │
│              └────────────────────────┘                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

**Component Specifications:**

**1. Classification (50 categories)**
```yaml
model: XLM-RoBERTa-base + classification head
architecture:
  - [CLS] embedding → Dense(768, 256) → ReLU → Dropout(0.1)
  - → Dense(256, 50) → softmax
training:
  - Hierarchical labels: 10 top-level → 50 fine-grained
  - Focal loss for class imbalance
  - Temperature scaling for calibration
performance_target: 92% accuracy, <50ms latency
```

**2. Entity Extraction (NER)**
```yaml
entities:
  - CUSTOMER_NAME: "John Smith contacted us..."
  - ORDER_ID: "#ORD-12345"
  - PRODUCT: "iPhone 15 Pro"
  - ISSUE_TYPE: "refund request", "delivery delay"

model: XLM-RoBERTa + CRF layer
training_data: 5K annotated tickets per language (active learning)
approach:
  - BIO tagging with language-specific patterns
  - Regex post-processing for structured fields (order ID)
  - Confidence thresholds: 0.9 for auto-fill, else human review
```

**3. Priority Scoring**
```yaml
model: Multi-task head on shared encoder
features:
  - Urgency keywords: "immediately", "cancel", "legal"
  - Sentiment score: Negative → higher priority
  - Customer tier: Premium customers prioritized
  - Entity presence: Order ID → easier to resolve

output: Priority score [0-1] → High (>0.8), Medium (0.4-0.8), Low (<0.4)
training: Ranking loss with human-labeled priority pairs
```

**4. Response Generation**
```yaml
model: mT5-base (580M params) with RAG
architecture:
  - Retriever: Dense passage retrieval from FAQ/response database
  - Generator: mT5 conditioned on ticket + retrieved context

response_types:
  - Template-based: Order status, refund policy (80% of cases)
  - Generated: Complex issues requiring synthesis

safeguards:
  - Template fallback if generation confidence < 0.7
  - Never auto-send; agent review required
  - Blocked patterns: Promises, legal claims, personal info
```

---

**Training Strategy:**

```yaml
phase_1_pretraining:
  - Continue pre-training XLM-RoBERTa on 1M unlabeled tickets
  - Domain adaptation: customer support vocabulary
  - 3 epochs, learning rate 1e-5

phase_2_multitask:
  - Joint training: classification + NER + priority
  - Shared encoder, task-specific heads
  - Sample balancing across tasks
  - 10 epochs, learning rate 2e-5

phase_3_generation:
  - Fine-tune mT5 on (ticket, response) pairs
  - Filter to high-quality agent responses
  - RLHF for response quality (optional)

data_requirements:
  - Classification: 50K labeled tickets (1K per class)
  - NER: 5K per language (active learning augmentation)
  - Generation: 100K (ticket, response) pairs
```

---

**Multilingual Strategy:**

```yaml
approach: Single multilingual model (not per-language)
model: XLM-RoBERTa (100 languages pre-trained)

language_handling:
  - Auto-detect language (fastText classifier, 99% accuracy)
  - Respond in detected language
  - Translation fallback for low-resource languages

challenges:
  - Code-mixing: "My bestellung is not here" (German+English)
  - Script variations: Simplified vs Traditional Chinese

solution:
  - Language ID at segment level
  - Unified tokenizer handles all scripts
```

---

**Production Architecture (100K tickets/day):**

```yaml
infrastructure:
  latency_requirements:
    - Classification: <50ms (real-time routing)
    - NER: <100ms (UI display)
    - Generation: <500ms (agent assistance)

  deployment:
    - Model serving: Triton Inference Server
    - Hardware: 4x T4 GPUs (classification/NER), 2x A10 (generation)
    - Batching: Dynamic batching for throughput
    - Caching: Redis for repeated queries

  scaling:
    - 100K/day = ~1.2 tickets/second average
    - Peak: 10x during business hours
    - Horizontal scaling with Kubernetes

  monitoring:
    - Latency P50/P95/P99
    - Classification confidence distribution
    - Drift detection on input distribution
    - A/B testing for model updates
```

---

**Quality Assurance:**

```yaml
evaluation:
  classification:
    - Weekly accuracy audit on 500 random samples
    - Confusion matrix analysis for problematic categories
    - Alert if accuracy drops >2%

  ner:
    - Precision/recall per entity type
    - False positive rate for auto-filled fields

  generation:
    - Human rating (1-5) on suggested responses
    - Agent acceptance rate as implicit feedback
    - Safety audit for problematic outputs

continuous_improvement:
  - Active learning: Flag low-confidence predictions for labeling
  - Feedback loop: Agent corrections improve training data
  - Monthly model refresh with new data
```

**Difficulty:** Hard | **Bloom's Level:** Synthesize

---

## Critical Knowledge Flags

The following concepts appear across multiple cards and represent essential knowledge:

| Concept | Cards | Significance |
|---------|-------|--------------|
| Attention mechanism | 2, 3, 5 | Foundation of modern NLP |
| Pre-training/fine-tuning | 3, 4, 5 | Dominant NLP paradigm |
| Contextual embeddings | 1, 4, 5 | Key advancement over static embeddings |
| Sequence modeling | 2, 3, 5 | Core NLP architecture pattern |

---

## Study Recommendations

### Before These Cards
- Review Lesson 4 (Transformers) for architecture details
- Review Lesson 8 (Neural Network Architectures) for building blocks

### After Mastering These Cards
- Implement a sentiment classifier with BERT
- Experiment with different tokenization strategies
- Fine-tune a model for your own NLP task

### Spaced Repetition Schedule
| Session | Focus |
|---------|-------|
| Day 1 | Cards 1-2 (foundations) |
| Day 3 | Cards 3-4 (analysis) |
| Day 7 | Card 5 (synthesis), review 1-4 |
| Day 14 | Full review |

---

*Generated from Lesson 9: Natural Language Processing | Flashcard Skill*
