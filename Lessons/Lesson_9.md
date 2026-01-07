# Lesson 9: Natural Language Processing

**Topic:** Natural Language Processing: From Text Representation to Neural Language Understanding
**Prerequisites:** Lesson 8 (Neural Network Architectures), Lesson 4 (Transformers)
**Estimated Study Time:** 3-4 hours
**Difficulty:** Advanced

---

## Learning Objectives

Upon completion of this lesson, learners will be able to:

1. **Apply** text preprocessing and tokenization techniques appropriate for different NLP applications
2. **Compare** text representation methods from sparse vectors (BoW, TF-IDF) to dense embeddings (Word2Vec, GloVe)
3. **Analyze** sequence modeling architectures (RNNs, LSTMs, Transformers) and their suitability for language tasks
4. **Evaluate** pre-trained language models (BERT, GPT) and their transfer learning capabilities
5. **Design** complete NLP pipelines for core tasks including classification, NER, and sequence-to-sequence applications

---

## Introduction

Natural Language Processing (NLP) represents the intersection of linguistics, computer science, and machine learning, focused on enabling machines to understand, interpret, and generate human language. As the dominant medium for human communication and knowledge representation, text presents unique challenges: ambiguity, context-dependence, implicit knowledge, and the infinite generativity of language.

The field has undergone a revolutionary transformation—from rule-based systems and hand-crafted features through statistical methods to the current era dominated by deep learning and pre-trained language models. Understanding this progression, from basic text preprocessing through embedding representations to modern Transformer-based models, provides the foundation for building systems that can classify documents, extract information, translate languages, answer questions, and engage in natural conversation.

---

## Core Concepts

### Concept 1: Text Preprocessing and Tokenization

Text preprocessing transforms raw, noisy text into a clean, standardized format, while tokenization segments text into the fundamental units that serve as model inputs.

**Preprocessing Pipeline:**

```
Raw Text → Normalization → Tokenization → Numericalization → Model Input

Normalization steps:
- Unicode normalization (NFC, NFKC)
- Lowercasing (optional; loses "US" vs "us")
- Accent/diacritic handling
- Whitespace normalization
- HTML/markup removal
```

**Tokenization Approaches:**

| Method | Description | Vocabulary | OOV Handling | Example |
|--------|-------------|------------|--------------|---------|
| Word-level | Split on whitespace/punctuation | Large (100K+) | UNK token | "running" → ["running"] |
| Character-level | Individual characters | Small (< 300) | None | "cat" → ["c", "a", "t"] |
| BPE | Iteratively merge frequent pairs | Medium (30K-50K) | Subwords | "running" → ["run", "ning"] |
| WordPiece | Likelihood-based merging | Medium (30K) | Subwords | "running" → ["run", "##ning"] |
| SentencePiece | Language-agnostic BPE/Unigram | Medium | Subwords | Handles any language |

**Modern Tokenizer Example (BPE):**

```
Training corpus: "low lower lowest"

Initial vocabulary: all characters + end-of-word token
Iteration 1: Most frequent pair "l"+"o" → merge to "lo"
Iteration 2: Most frequent pair "lo"+"w" → merge to "low"
Iteration 3: Most frequent pair "e"+"r" → merge to "er"
...continue until vocabulary size reached

Result: "lowest" → ["low", "est"]
        "newer" (unseen) → ["new", "er"]
```

**Critical Design Decisions:**

| Decision | Tradeoff |
|----------|----------|
| Vocabulary size | Larger → fewer UNK, more parameters; Smaller → more subwords per text |
| Lowercasing | Loses named entity cues ("Apple" vs "apple") |
| Stop word removal | Loses grammatical structure; hurts contextual models |
| Stemming/Lemmatization | May conflate distinct meanings (better/good) |

---

### Concept 2: Sparse Text Representations

Traditional NLP represents text as high-dimensional sparse vectors based on word occurrence statistics.

**Bag-of-Words (BoW):**

```
Document: "The cat sat on the mat"
Vocabulary: [cat, dog, mat, on, sat, the]

BoW vector: [1, 0, 1, 1, 1, 2]
            (cat appears 1x, dog 0x, mat 1x, on 1x, sat 1x, the 2x)

Properties:
- Ignores word order: "dog bites man" = "man bites dog"
- Dimension = vocabulary size (typically 10K-100K)
- Very sparse: most entries are 0
```

**TF-IDF (Term Frequency - Inverse Document Frequency):**

```
TF-IDF(t, d) = TF(t, d) × IDF(t)

TF(t, d) = count(t in d) / total_words(d)    # Term frequency in document
IDF(t) = log(N / df(t))                       # Inverse document frequency

Where:
- N = total documents in corpus
- df(t) = documents containing term t

Effect: High for distinctive terms; low for common words
```

**TF-IDF Example:**

| Term | TF (doc) | df | IDF (N=1000) | TF-IDF |
|------|----------|-----|--------------|--------|
| "the" | 0.05 | 950 | 0.02 | 0.001 |
| "neural" | 0.02 | 50 | 1.30 | 0.026 |
| "backpropagation" | 0.01 | 10 | 2.00 | 0.020 |

**N-grams:**

```
Text: "I love natural language processing"

Unigrams: ["I", "love", "natural", "language", "processing"]
Bigrams:  ["I love", "love natural", "natural language", "language processing"]
Trigrams: ["I love natural", "love natural language", "natural language processing"]

Trade-off: Higher n → captures more context, but exponentially larger vocabulary
```

**Limitations of Sparse Representations:**

1. No semantic similarity: "good" and "excellent" are orthogonal
2. High dimensionality: 100K+ dimensions common
3. No generalization: unseen words have no representation
4. Order-agnostic: loses sequential information

---

### Concept 3: Dense Word Embeddings

Word embeddings learn dense, low-dimensional vectors where semantic relationships are encoded in geometric properties.

**Word2Vec Architecture:**

```
Skip-gram: Predict context words from target
    Input: "language"
    Predict: ["natural", "processing"]

    Objective: maximize P(context | target)
    P(wo | wi) = softmax(v'_wo · v_wi)

CBOW: Predict target from context
    Input: ["natural", "processing"]
    Predict: "language"

    Objective: maximize P(target | context)
```

**Training Objective (Skip-gram with Negative Sampling):**

```
Loss = -log σ(v'_pos · v_target) - Σ log σ(-v'_neg · v_target)

Where:
- v_target: embedding of target word
- v'_pos: output embedding of positive context word
- v'_neg: output embeddings of negative samples
- σ: sigmoid function
```

**GloVe (Global Vectors):**

```
Objective: Learn vectors where dot product equals log co-occurrence

v_i · v_j + b_i + b_j = log(X_ij)

Where X_ij = co-occurrence count of words i and j

Combines:
- Global statistics (like LSA)
- Local context windows (like Word2Vec)
```

**Embedding Properties:**

| Property | Example | Implication |
|----------|---------|-------------|
| Semantic similarity | cos(king, queen) high | Similar words cluster |
| Analogical reasoning | king - man + woman ≈ queen | Relationships as vectors |
| Antonymy | good and bad both similar to "quality" | Context similarity ≠ meaning similarity |
| Polysemy limitation | "bank" has one vector | Same vector for river/financial |

**FastText Extension:**

```
Word = sum of character n-gram embeddings

"where" = <wh + whe + her + ere + re> + <where>

Advantages:
- Handles OOV words through subword sharing
- Captures morphology ("unhappy" shares with "happy")
- Better for morphologically rich languages
```

---

### Concept 4: Recurrent Neural Networks for Language

RNNs process sequential data by maintaining hidden states that carry information across time steps.

**Vanilla RNN:**

```
h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b_h)
y_t = W_hy · h_t + b_y

Where:
- h_t: hidden state at time t
- x_t: input at time t
- W_hh, W_xh, W_hy: weight matrices
```

**Unrolled RNN for "hello":**

```
x₁(h) → [RNN] → h₁ → [RNN] → h₂ → [RNN] → h₃ → [RNN] → h₄ → [RNN] → h₅
         ↑          ↑          ↑          ↑          ↑
        x₂(e)      x₃(l)      x₄(l)      x₅(o)
```

**Vanishing Gradient Problem:**

```
Gradient through T steps:
∂L/∂h₁ = ∂L/∂h_T × ∏_{t=2}^{T} ∂h_t/∂h_{t-1}

Each factor ∂h_t/∂h_{t-1} = W_hh × diag(tanh'(·))

If ||W_hh|| < 1 or tanh' small: gradient vanishes exponentially
If ||W_hh|| > 1: gradient explodes exponentially

Practical limit: ~10-20 steps for effective learning
```

**Bidirectional RNN:**

```
Forward:  h_t^→ = RNN(h_{t-1}^→, x_t)
Backward: h_t^← = RNN(h_{t+1}^←, x_t)
Combined: h_t = [h_t^→; h_t^←]

Benefits:
- Full context at each position
- Essential for classification/tagging tasks
- Cannot be used for generation (needs future)
```

---

### Concept 5: LSTM and GRU Architectures

Gated architectures solve the vanishing gradient problem through learned gates that control information flow.

**LSTM Architecture:**

```
Forget Gate:    f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
Input Gate:     i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
Candidate:      c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)
Cell Update:    c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
Output Gate:    o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
Hidden State:   h_t = o_t ⊙ tanh(c_t)

Where ⊙ = element-wise multiplication
```

**Gate Intuitions:**

| Gate | Function | When High | When Low |
|------|----------|-----------|----------|
| Forget (f_t) | What to discard from memory | Keep previous info | Reset memory |
| Input (i_t) | What new info to store | Store new info | Ignore input |
| Output (o_t) | What to expose from memory | Output memory | Hide memory |

**Why LSTM Solves Vanishing Gradients:**

```
Cell state gradient:
∂c_t/∂c_{t-1} = f_t

If f_t ≈ 1: gradient flows unchanged (additive, not multiplicative)
The forget gate learns when to preserve gradient flow

Compare to RNN: ∂h_t/∂h_{t-1} = W_hh × tanh'(...) — always multiplied
```

**GRU (Simplified Gating):**

```
Reset Gate:  r_t = σ(W_r · [h_{t-1}, x_t])
Update Gate: z_t = σ(W_z · [h_{t-1}, x_t])
Candidate:   h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])
Hidden:      h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t

Differences from LSTM:
- No separate cell state
- 2 gates instead of 3
- Often similar performance with fewer parameters
```

**Architecture Comparison:**

| Model | Parameters (per layer) | Gates | Performance |
|-------|----------------------|-------|-------------|
| Vanilla RNN | 3d² | 0 | Poor on long sequences |
| GRU | 9d² | 2 | Good, efficient |
| LSTM | 12d² | 3 | Best for long sequences |

---

### Concept 6: Sequence-to-Sequence and Attention

Encoder-decoder architectures map variable-length input sequences to variable-length output sequences.

**Basic Seq2Seq:**

```
Encoder:
  Input: x₁, x₂, ..., x_n (source sentence)
  Process: h_t = LSTM(h_{t-1}, x_t)
  Output: context vector c = h_n (final hidden state)

Decoder:
  Input: c, y₁, y₂, ..., y_{m-1} (previously generated)
  Process: s_t = LSTM(s_{t-1}, [y_{t-1}, c])
  Output: P(y_t | y_{<t}, c)
```

**Bottleneck Problem:**

```
"The cat sat on the mat" → [single fixed-size vector] → "Le chat s'est assis sur le tapis"

Problem: All source information compressed into one vector
- Information loss for long sentences
- Decoder can't "look back" at source
- Performance degrades with sentence length
```

**Attention Mechanism:**

```
Instead of single context vector, compute weighted sum at each decode step:

Score:    e_{ij} = score(s_{i-1}, h_j)     # Alignment between decoder state and encoder state
Weights:  α_{ij} = softmax(e_i)_j          # Normalized attention weights
Context:  c_i = Σ_j α_{ij} · h_j           # Weighted sum of encoder states

Decoder:  s_i = LSTM(s_{i-1}, [y_{i-1}, c_i])

Score functions:
- Dot product:    score(s, h) = s^T h
- Additive:       score(s, h) = v^T tanh(W_s·s + W_h·h)
- Scaled dot:     score(s, h) = s^T h / √d
```

**Attention Visualization:**

```
Source: "The cat sat"
Target: "Le chat"

Attention weights when generating "chat":
  "The"  →  0.05
  "cat"  →  0.90  ← high attention
  "sat"  →  0.05

The model learns to align "chat" with "cat"
```

**Teacher Forcing:**

```
Training:  Use ground truth y_{t-1} as input (not model's prediction)
Inference: Use model's own predictions (no ground truth available)

Benefits:
- Faster convergence
- More stable training

Drawback:
- Exposure bias: train/test mismatch
```

---

### Concept 7: Transformer Architecture for NLP

The Transformer architecture replaces recurrence with self-attention, enabling parallel processing and capturing long-range dependencies effectively.

**Self-Attention for Language:**

```
Input: "The cat sat on the mat"
       [x₁, x₂, x₃, x₄, x₅, x₆]

For each position i:
  Q_i = x_i · W_Q  (What am I looking for?)
  K_i = x_i · W_K  (What do I contain?)
  V_i = x_i · W_V  (What do I contribute?)

Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

**Self-Attention Example:**

```
Query "cat" attends to:
  "The"  →  0.15  (determiner relationship)
  "cat"  →  0.20  (self)
  "sat"  →  0.40  (subject-verb)
  "on"   →  0.05
  "the"  →  0.05
  "mat"  →  0.15  (cat is on mat - semantic relation)
```

**Transformer Block:**

```
Input
  ↓
[Multi-Head Attention]
  ↓
Add & LayerNorm ←── (residual connection)
  ↓
[Feed-Forward Network]
  ↓
Add & LayerNorm ←── (residual connection)
  ↓
Output
```

**Positional Encoding:**

```
Since self-attention is permutation-equivariant, position must be injected:

PE(pos, 2i)   = sin(pos / 10000^{2i/d})
PE(pos, 2i+1) = cos(pos / 10000^{2i/d})

Properties:
- Unique encoding for each position
- Allows extrapolation to unseen lengths
- Relative positions can be computed from absolute
```

**Encoder vs. Decoder:**

| Component | Attention Type | Mask | Use Case |
|-----------|---------------|------|----------|
| Encoder | Bidirectional | None | Understanding (BERT) |
| Decoder | Causal | Future tokens masked | Generation (GPT) |
| Cross-attention | Encoder → Decoder | None | Translation |

**Causal Masking:**

```
When generating position t, can only attend to positions ≤ t:

     1   2   3   4   5
1  [ 1   0   0   0   0 ]
2  [ 1   1   0   0   0 ]
3  [ 1   1   1   0   0 ]
4  [ 1   1   1   1   0 ]
5  [ 1   1   1   1   1 ]

Prevents "cheating" by looking at future tokens during training
```

---

### Concept 8: Pre-trained Language Models

Pre-training on massive unlabeled text creates powerful general-purpose language representations that transfer to downstream tasks.

**BERT (Bidirectional Encoder Representations):**

```
Architecture: Encoder-only Transformer
Pre-training objectives:
  1. Masked Language Modeling (MLM):
     - Randomly mask 15% of tokens
     - Predict masked tokens from bidirectional context
     - "[MASK]" token 80%, random 10%, unchanged 10%

  2. Next Sentence Prediction (NSP):
     - Given sentence A, is sentence B the actual next sentence?
     - Binary classification task
```

**BERT Input Format:**

```
[CLS] Sentence A [SEP] Sentence B [SEP]

Token embeddings + Segment embeddings + Position embeddings

[CLS] embedding used for classification tasks
Each token embedding used for sequence labeling
```

**GPT (Generative Pre-trained Transformer):**

```
Architecture: Decoder-only Transformer
Pre-training objective:
  - Causal Language Modeling
  - Predict next token given all previous tokens
  - P(x_t | x_1, ..., x_{t-1})
```

**BERT vs GPT:**

| Aspect | BERT | GPT |
|--------|------|-----|
| Direction | Bidirectional | Left-to-right |
| Pre-training | MLM + NSP | Causal LM |
| Architecture | Encoder | Decoder |
| Best for | Understanding tasks | Generation tasks |
| Fine-tuning | Add task head | Prompting or fine-tune |

**Fine-tuning Patterns:**

```
Classification:
  [CLS] + Sentence → BERT → [CLS] hidden → Linear → softmax

Sequence Labeling:
  Token₁ Token₂ ... → BERT → Hidden₁ Hidden₂ ... → Linear per token

Question Answering:
  [CLS] Question [SEP] Passage → BERT → Start/End span prediction
```

**Scale Progression:**

| Model | Parameters | Year | Key Innovation |
|-------|------------|------|----------------|
| BERT-base | 110M | 2018 | Bidirectional pre-training |
| BERT-large | 340M | 2018 | Larger scale |
| GPT-2 | 1.5B | 2019 | Zero-shot capabilities |
| GPT-3 | 175B | 2020 | In-context learning |
| GPT-4 | ~1.7T (est.) | 2023 | Multimodal, reasoning |

---

### Concept 9: Core NLP Tasks and Architectures

Different NLP tasks require specific architectural patterns and evaluation metrics.

**Text Classification:**

```
Task: Assign category to document
Examples: Sentiment (pos/neg), topic, spam detection

Architecture:
  Text → Encoder (BERT) → [CLS] → Dense → softmax(classes)

Metrics:
  - Accuracy: correct / total
  - F1: harmonic mean of precision and recall
  - Macro F1: average F1 across classes (handles imbalance)
```

**Named Entity Recognition (NER):**

```
Task: Identify and classify named entities
Example: "[PER Obama] visited [LOC Paris]"

BIO Tagging:
  Barack → B-PER (Begin Person)
  Obama  → I-PER (Inside Person)
  visited → O (Outside)
  Paris  → B-LOC (Begin Location)

Architecture:
  Tokens → BERT → Per-token hidden → Dense → CRF → BIO tags

CRF layer models transition constraints (B-PER can't follow I-LOC)
```

**Machine Translation:**

```
Task: Convert text from source language to target
Example: "Hello" → "Bonjour"

Architecture: Encoder-Decoder Transformer
  Source → Encoder → Memory
  Memory + Target prefix → Decoder → Next token

Decoding strategies:
  - Greedy: argmax at each step
  - Beam search: maintain top-k hypotheses
  - Sampling: sample from distribution (more diverse)

Metric: BLEU (n-gram precision with brevity penalty)
```

**Question Answering:**

```
Extractive QA:
  Task: Find answer span in passage
  Input: Question + Passage → Model → Start/End positions

Reading Comprehension:
  Question: "Where was Obama born?"
  Passage: "Barack Obama was born in Honolulu, Hawaii..."
  Answer: "Honolulu, Hawaii" (span extraction)

Generative QA:
  Task: Generate answer text
  More flexible but harder to evaluate
```

**Text Summarization:**

```
Extractive: Select important sentences
  - Score sentences, select top-k
  - No new text generated

Abstractive: Generate new summary text
  - Encoder-decoder architecture
  - Can paraphrase and combine information

Metrics:
  - ROUGE-N: n-gram overlap with reference
  - ROUGE-L: longest common subsequence
```

---

### Concept 10: Modern NLP Techniques and Challenges

Contemporary NLP systems employ sophisticated techniques while facing important challenges.

**Contextual Embeddings:**

```
Static (Word2Vec): "bank" always has same vector
Contextual (BERT): "bank" has different vectors in:
  - "river bank" vs "bank account"

Layer-wise representations:
  - Lower layers: syntax, POS information
  - Higher layers: semantics, task-specific features
```

**Transfer Learning Strategies:**

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| Feature extraction | Freeze encoder, train classifier | Limited data, quick deployment |
| Fine-tuning | Update all parameters | Sufficient data, best quality |
| Adapter tuning | Add small trainable modules | Multiple tasks, parameter efficient |
| Prompt tuning | Prepend learnable tokens | Very large models |

**Few-Shot and In-Context Learning:**

```
Zero-shot: Task description only
  "Classify sentiment: 'Great movie!' → "

One-shot: Single example
  "Classify sentiment:
   'Terrible service' → Negative
   'Great movie!' → "

Few-shot: Multiple examples
  Provide 3-5 examples before query
  No gradient updates needed (in-context learning)
```

**Key Challenges:**

| Challenge | Description | Mitigation |
|-----------|-------------|------------|
| Hallucination | Models generate plausible but false information | RAG, fact verification |
| Bias | Training data biases reflected in outputs | Debiasing, diverse data |
| Long context | Quadratic attention complexity | Efficient attention, hierarchical |
| Evaluation | Automatic metrics don't capture quality | Human evaluation, task-specific |
| Multilinguality | Performance varies across languages | Multilingual models, translation |

**Retrieval-Augmented Generation (RAG):**

```
Query → Retrieve relevant documents → Augment prompt → Generate

Benefits:
- Grounded in retrieved facts (reduces hallucination)
- Updatable knowledge (no retraining)
- Transparent source attribution

Components:
1. Document encoder (embed passages)
2. Query encoder (embed question)
3. Retriever (find similar passages)
4. Generator (produce answer with retrieved context)
```

---

## Summary

Natural Language Processing enables machines to understand and generate human language through a sophisticated pipeline of techniques. **Text preprocessing and tokenization** (Concept 1) prepare raw text, with subword tokenization (BPE, WordPiece) now dominating. **Sparse representations** (Concept 2) like TF-IDF capture word statistics but lack semantics. **Dense embeddings** (Concept 3) from Word2Vec and GloVe encode semantic relationships in continuous vector spaces.

**RNNs** (Concept 4) process sequences through hidden states but suffer from vanishing gradients. **LSTMs and GRUs** (Concept 5) introduce gating mechanisms that enable learning long-range dependencies. **Sequence-to-sequence models** (Concept 6) with **attention** enable tasks like translation by allowing the decoder to focus on relevant source positions.

The **Transformer architecture** (Concept 7) revolutionized NLP by replacing recurrence with self-attention, enabling parallel processing and effective long-range modeling. **Pre-trained language models** (Concept 8) like BERT and GPT learn general language understanding from massive corpora, then transfer to downstream tasks through fine-tuning.

**Core NLP tasks** (Concept 9) include classification, NER, translation, QA, and summarization, each with specific architectures and metrics. **Modern challenges** (Concept 10) include hallucination, bias, and long context handling, with techniques like RAG and efficient attention providing solutions.

The progression from tokenization through embeddings, sequence models, attention, and pre-training represents the evolution of increasingly powerful language understanding systems that now approach human-level performance on many benchmarks.

---

## References

- Mikolov, T., et al. (2013). "Efficient Estimation of Word Representations in Vector Space" (Word2Vec)
- Pennington, J., et al. (2014). "GloVe: Global Vectors for Word Representation"
- Hochreiter, S. & Schmidhuber, J. (1997). "Long Short-Term Memory"
- Bahdanau, D., et al. (2015). "Neural Machine Translation by Jointly Learning to Align and Translate"
- Vaswani, A., et al. (2017). "Attention Is All You Need"
- Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers"
- Brown, T., et al. (2020). "Language Models are Few-Shot Learners" (GPT-3)

---

*Generated from Natural Language Processing Study Notes | Lesson Skill*
