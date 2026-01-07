# Transformers

**Topic:** Transformer Architecture: Attention Mechanisms, Architectures, and Applications
**Date:** 2026-01-07
**Complexity Level:** Advanced
**Discipline:** Computer Science / Deep Learning / Natural Language Processing

---

## Learning Objectives

Upon completion of this material, learners will be able to:
- **Analyze** the self-attention mechanism and its computational properties (complexity, expressiveness)
- **Evaluate** architectural choices between encoder-only, decoder-only, and encoder-decoder transformers
- **Apply** positional encoding schemes and understand their impact on sequence modeling
- **Design** transformer-based systems for various tasks (classification, generation, sequence-to-sequence)
- **Critique** transformer scaling properties, efficiency techniques, and computational tradeoffs

---

## Executive Summary

The Transformer architecture, introduced in "Attention Is All You Need" (Vaswani et al., 2017), revolutionized deep learning by replacing recurrent and convolutional operations with self-attention mechanisms. This architecture enables parallel processing of sequences and captures long-range dependencies more effectively than RNNs, becoming the foundation for modern language models (GPT, BERT, T5), vision transformers (ViT), and multimodal systems.

The key innovation is self-attention: each position in a sequence attends to all other positions, computing relevance-weighted representations. Multi-head attention allows the model to jointly attend to information from different representation subspaces. Combined with position encodings, layer normalization, and feed-forward networks, transformers achieve state-of-the-art performance across NLP, computer vision, speech, and beyond.

Understanding transformers is essential for modern AI practitioners. This material covers the attention mechanism, architectural components, major variants (BERT, GPT, T5), and practical considerations including efficiency, scaling laws, and implementation details.

---

## Core Concepts

### Concept 1: Self-Attention Mechanism

**Definition:**
Self-attention (or intra-attention) computes a representation of each position in a sequence by attending to all positions, weighting their contributions based on learned compatibility functions.

**Explanation:**
Given input sequence X = [x1, x2, ..., xn], self-attention computes:
1. **Queries (Q)**, **Keys (K)**, **Values (V)** via linear projections: Q = XWQ, K = XWK, V = XWV
2. **Attention scores:** score(i,j) = qi . kj / sqrt(dk) (scaled dot-product)
3. **Attention weights:** aij = softmax(scores)ij
4. **Output:** outi = sum_j aij . vj

Each position's output is a weighted sum of all values, where weights are determined by query-key compatibility. The sqrt(dk) scaling prevents dot products from growing too large with dimension.

**Key Points:**
- **Query, Key, Value:** Three projections enabling flexible attention patterns
- **Scaled dot-product:** Prevents gradient issues from large dot products
- **Softmax:** Normalizes attention weights to sum to 1
- **Complexity:** O(n^2 * d) for sequence length n, dimension d
- **Global receptive field:** Every position can attend to every other position

### Concept 2: Multi-Head Attention

**Definition:**
Multi-head attention runs multiple attention operations in parallel with different learned projections, allowing the model to jointly attend to information from different representation subspaces.

**Explanation:**
Instead of single attention with d-dimensional keys/queries/values:
1. Project to h different subspaces: Qi = XWQi, Ki = XWKi, Vi = XWVi (each dk = d/h dimensional)
2. Compute attention independently in each head
3. Concatenate outputs: [head1; head2; ...; headh]
4. Final projection: output = concat * WO

Different heads can learn different attention patterns—some may attend to nearby tokens (local), others to syntactically related tokens (structural), others to semantically similar tokens (semantic).

**Key Points:**
- **Parallel attention:** h independent attention operations
- **Subspace projections:** Each head operates in d/h dimensions
- **Diverse patterns:** Different heads learn different relationship types
- **Concatenation:** Combine information from all heads
- **Same complexity:** O(n^2 * d) total, same as single-head with full dimension

### Concept 3: Positional Encoding

**Definition:**
Positional encodings inject sequence order information into the transformer, which otherwise treats input as an unordered set due to the permutation-equivariant nature of self-attention.

**Explanation:**
Self-attention is permutation equivariant: shuffling inputs shuffles outputs identically. To encode position, we add positional information to input embeddings.

**Sinusoidal (Original):**
```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

Different frequencies for different dimensions; enables extrapolation to longer sequences.

**Learned:** Trainable embedding per position; works well but limited to training length.

**Relative (RoPE, ALiBi):** Encode relative distances rather than absolute positions; better length generalization.

**Key Points:**
- **Why needed:** Self-attention is position-agnostic
- **Sinusoidal:** Fixed, generalizes to longer sequences, no parameters
- **Learned:** More flexible but fixed maximum length
- **Relative:** Modern approach; RoPE (rotary), ALiBi (attention bias)
- **Added to embeddings:** PE + token embedding = input to transformer

### Concept 4: Transformer Encoder

**Definition:**
The transformer encoder processes input sequences bidirectionally through stacked layers of multi-head self-attention and feed-forward networks, producing contextualized representations.

**Explanation:**
Each encoder layer consists of:
1. **Multi-head self-attention:** Each position attends to all positions
2. **Add & Norm:** Residual connection + layer normalization
3. **Feed-forward network:** Two linear layers with activation (typically GELU)
4. **Add & Norm:** Another residual + normalization

The encoder sees all positions simultaneously (bidirectional), making it ideal for understanding tasks where full context is available. BERT uses encoder-only architecture for classification, NER, question answering.

**Key Points:**
- **Bidirectional:** All positions attend to all positions (no masking)
- **Stacked layers:** Typically 6-24 layers in standard models
- **Residual connections:** Enable gradient flow; output = sublayer(x) + x
- **Layer normalization:** Stabilizes training; Pre-LN vs Post-LN variants
- **Use cases:** Classification, encoding for retrieval, feature extraction

### Concept 5: Transformer Decoder

**Definition:**
The transformer decoder generates sequences autoregressively, using masked self-attention to prevent attending to future positions and (optionally) cross-attention to encoder outputs.

**Explanation:**
Decoder layers contain:
1. **Masked self-attention:** Position i can only attend to positions <= i
2. **Add & Norm**
3. **Cross-attention (encoder-decoder):** Attend to encoder outputs
4. **Add & Norm**
5. **Feed-forward network**
6. **Add & Norm**

The causal mask ensures autoregressive generation: when predicting token t, the model cannot see tokens > t. For decoder-only models (GPT), there is no cross-attention—only masked self-attention.

**Key Points:**
- **Causal masking:** Prevents attending to future tokens
- **Autoregressive:** Generates one token at a time
- **Cross-attention:** Queries from decoder, keys/values from encoder
- **Decoder-only (GPT):** No encoder, no cross-attention
- **Use cases:** Text generation, language modeling, code completion

### Concept 6: Feed-Forward Networks and Layer Norm

**Definition:**
Feed-forward networks (FFN) apply position-wise transformations, while layer normalization stabilizes activations across the feature dimension at each position.

**Explanation:**
**Feed-Forward Network:**
```
FFN(x) = W2 * activation(W1 * x + b1) + b2
```

Typically expands dimension (d -> 4d -> d). Applied independently to each position. Recent work suggests FFN stores factual knowledge (key-value memories).

**Layer Normalization:**
```
LN(x) = gamma * (x - mu) / sigma + beta
```

Normalizes across feature dimension (not batch). Two variants:
- **Post-LN (original):** LN after residual addition
- **Pre-LN:** LN before sublayer; more stable training, now standard

**Key Points:**
- **FFN:** Two layers, expansion ratio typically 4x
- **Activation:** GELU preferred over ReLU in modern transformers
- **Position-wise:** Same FFN applied to each position independently
- **Layer norm:** Stabilizes training; critical for deep transformers
- **Pre-LN:** Default in modern architectures; better gradient flow

### Concept 7: BERT and Encoder-Only Models

**Definition:**
BERT (Bidirectional Encoder Representations from Transformers) uses encoder-only architecture with masked language modeling pre-training to learn bidirectional contextualized representations.

**Explanation:**
**Pre-training objectives:**
1. **Masked Language Modeling (MLM):** Mask 15% of tokens, predict original
2. **Next Sentence Prediction (NSP):** Predict if two sentences are consecutive (later shown less useful)

**Architecture:** Encoder-only transformer (no decoder, no causal masking)

**Fine-tuning:** Add task-specific head; fine-tune entire model on downstream task.

BERT excels at understanding tasks where full context is available: classification, NER, question answering, semantic similarity.

**Key Points:**
- **Bidirectional:** Attends to both left and right context
- **MLM:** Predict masked tokens; enables deep bidirectional representations
- **[CLS] token:** Special token for classification; its representation used for sentence-level tasks
- **Variants:** RoBERTa (better training), ALBERT (parameter sharing), DistilBERT (distilled)
- **Limitations:** Not suitable for generation; fixed max length

### Concept 8: GPT and Decoder-Only Models

**Definition:**
GPT (Generative Pre-trained Transformer) uses decoder-only architecture with causal language modeling, trained to predict the next token, enabling powerful text generation and few-shot learning.

**Explanation:**
**Pre-training:** Causal language modeling—predict next token given previous tokens:
```
P(x1, x2, ..., xn) = Product of P(xt | x1, ..., x(t-1))
```

**Architecture:** Decoder-only (no encoder, no cross-attention), causal self-attention mask.

**Scaling:** GPT-2 (1.5B) -> GPT-3 (175B) -> GPT-4 demonstrated emergent capabilities at scale: in-context learning, instruction following, reasoning.

**Key Points:**
- **Autoregressive:** Each token predicted from previous tokens only
- **Causal mask:** Upper triangular mask prevents future attention
- **In-context learning:** Learn tasks from examples in prompt
- **Emergent abilities:** Capabilities appearing at scale
- **Variants:** GPT-2, GPT-3, GPT-4, Llama, Mistral, Claude

### Concept 9: Encoder-Decoder Models

**Definition:**
Encoder-decoder transformers process input with bidirectional encoder and generate output with autoregressive decoder connected via cross-attention, ideal for sequence-to-sequence tasks.

**Explanation:**
**Architecture:**
- Encoder: Processes input sequence bidirectionally
- Decoder: Generates output autoregressively
- Cross-attention: Decoder attends to encoder outputs

**Models:**
- **T5:** Text-to-text framework; all tasks as text generation
- **BART:** Denoising autoencoder; corrupted input -> original
- **mT5, mBART:** Multilingual variants

**Use cases:** Translation, summarization, question answering, data-to-text.

**Key Points:**
- **Sequence-to-sequence:** Input sequence -> output sequence
- **Cross-attention:** Bridges encoder and decoder
- **T5 paradigm:** "Translate English to German: [text]" — unified format
- **Flexible:** Handles variable-length input and output
- **Applications:** Translation, summarization, code generation

### Concept 10: Efficient Transformers and Scaling

**Definition:**
Efficient transformer variants reduce the O(n^2) attention complexity through sparse patterns, low-rank approximations, or linear attention, enabling longer sequences and larger scale.

**Explanation:**
**Efficiency techniques:**
- **Sparse attention (Longformer, BigBird):** Attend to local windows + global tokens
- **Linear attention (Performer, Linear Transformer):** Approximate softmax with kernels; O(n)
- **Flash Attention:** Exact attention with memory-efficient GPU kernels
- **Sliding window (Mistral):** Limited context window that slides

**Scaling laws:** Performance scales predictably with compute, data, and parameters:
```
Loss proportional to C^(-alpha) where C is compute budget
```

**Key Points:**
- **Quadratic bottleneck:** O(n^2) limits sequence length
- **Sparse patterns:** Trade expressiveness for efficiency
- **Flash Attention:** Not approximate; just memory-efficient implementation
- **Scaling laws:** Chinchilla-optimal; balance model size and data
- **Context length:** Modern models: 4K -> 32K -> 128K+ tokens

---

## Theoretical Framework

### Attention as Soft Dictionary Lookup

Attention can be viewed as differentiable dictionary lookup: queries search for relevant keys, retrieving associated values. This perspective explains why attention excels at tasks requiring retrieval of relevant information from context.

### Universal Approximation

Transformers are universal approximators for sequence-to-sequence functions. The combination of attention (mixing across positions) and FFN (transformation at each position) provides sufficient expressiveness.

### Inductive Biases

Unlike CNNs (locality) or RNNs (sequential), transformers have minimal inductive bias—they learn patterns from data. This flexibility enables transfer across domains but requires more data.

---

## Practical Applications

### Application 1: Language Understanding (BERT-style)
Sentiment analysis, named entity recognition, question answering, semantic similarity. Encoder models excel when full input context is available for understanding.

### Application 2: Text Generation (GPT-style)
Creative writing, code generation, chatbots, summarization. Decoder models generate coherent, contextually appropriate text autoregressively.

### Application 3: Translation and Summarization (T5-style)
Machine translation, document summarization, data-to-text generation. Encoder-decoder excels at transforming one sequence to another.

### Application 4: Vision and Multimodal
Vision Transformer (ViT) applies transformers to image patches. Multimodal models (CLIP, Flamingo, GPT-4V) combine vision and language transformers.

---

## Critical Analysis

### Strengths
- **Parallelization:** Unlike RNNs, all positions computed simultaneously
- **Long-range dependencies:** Direct connections between any positions
- **Transfer learning:** Pre-trained models transfer remarkably well
- **Scalability:** Performance improves predictably with scale

### Limitations
- **Quadratic complexity:** O(n^2) attention limits sequence length
- **Data hungry:** Requires massive pre-training data
- **Compute intensive:** Training large models requires significant resources
- **Position encoding:** Still an active research area; extrapolation challenges

### Current Debates
- **Efficient attention:** Can we achieve O(n) without sacrificing quality?
- **Architecture search:** Are there better architectures than standard transformer?
- **Emergent abilities:** Real phenomena or measurement artifacts?
- **Context length:** How to effectively use 100K+ token contexts?

---

## Key Terminology

| Term | Definition | Context |
|------|------------|---------|
| Self-Attention | Attending to all positions in same sequence | Core mechanism |
| Multi-Head | Parallel attention with different projections | Diverse patterns |
| Query, Key, Value | Three projections for attention computation | Attention components |
| Positional Encoding | Injecting position information | Sequence order |
| Causal Mask | Preventing attention to future positions | Autoregressive |
| Layer Normalization | Normalizing across features | Training stability |
| Encoder | Bidirectional transformer stack | Understanding |
| Decoder | Autoregressive transformer stack | Generation |
| Cross-Attention | Decoder attending to encoder | Seq2seq connection |
| Pre-training | Training on large unlabeled data | Transfer learning |
| Fine-tuning | Adapting to specific task | Task specialization |
| Flash Attention | Memory-efficient attention kernel | Efficiency |

---

## Review Questions

1. **Comprehension:** Explain why the sqrt(dk) scaling factor is necessary in scaled dot-product attention. What happens without it as dimension increases?

2. **Application:** Design a transformer architecture for a code completion task. Should you use encoder-only, decoder-only, or encoder-decoder? Justify your choice.

3. **Analysis:** Compare how BERT and GPT-3 would approach a sentiment classification task. What are the fundamental differences in their approaches?

4. **Synthesis:** You need to process documents with 50,000 tokens. Design an efficient attention mechanism that maintains quality while being computationally tractable.

---

## Further Reading

- Vaswani, A., et al. - "Attention Is All You Need" (Original Transformer)
- Devlin, J., et al. - "BERT: Pre-training of Deep Bidirectional Transformers"
- Radford, A., et al. - "Language Models are Unsupervised Multitask Learners" (GPT-2)
- Raffel, C., et al. - "Exploring the Limits of Transfer Learning with T5"
- Dao, T., et al. - "FlashAttention: Fast and Memory-Efficient Exact Attention"
- Hoffmann, J., et al. - "Training Compute-Optimal Large Language Models" (Chinchilla)

---

## Summary

The Transformer architecture revolutionized deep learning through the self-attention mechanism, which computes representations by attending to all positions in a sequence with learned query-key-value projections. Multi-head attention enables parallel attention patterns across different subspaces. Positional encodings inject sequence order into the otherwise permutation-equivariant architecture.

Three major variants dominate: encoder-only (BERT) for understanding tasks using bidirectional attention and masked language modeling; decoder-only (GPT) for generation using causal attention and autoregressive prediction; encoder-decoder (T5) for sequence-to-sequence tasks with cross-attention. Each architecture suits different use cases based on whether full context is available and whether generation is required.

Practical transformers combine attention with feed-forward networks, residual connections, and layer normalization in repeated blocks. Efficiency techniques—sparse attention, Flash Attention, sliding windows—address the O(n^2) complexity bottleneck. Scaling laws demonstrate predictable performance improvements with compute, motivating ever-larger models while raising questions about optimal resource allocation.

Understanding transformers is fundamental to modern AI: they underpin language models, vision models, and multimodal systems that are transforming how we interact with artificial intelligence.
