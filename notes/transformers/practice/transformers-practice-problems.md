# Practice Problems: Transformers

**Source:** notes/transformers/transformers-study-notes.md
**Concept Map:** notes/transformers/concept-maps/transformers-concept-map.md
**Date Generated:** 2026-01-07
**Total Problems:** 5
**Distribution:** 1 Warm-Up | 2 Skill-Builder | 1 Challenge | 1 Debug/Fix

---

## Problem 1 | Warm-Up
**Concept:** Self-Attention Computation
**Centrality:** Critical (11 connections)
**Estimated Time:** 10-15 minutes

### Problem Statement

Given the following input sequence with 3 tokens and embedding dimension d=4:

```
X = [[1, 0, 1, 0],    # Token 1: "The"
     [0, 1, 0, 1],    # Token 2: "cat"
     [1, 1, 0, 0]]    # Token 3: "sat"
```

And the following learned weight matrices (simplified for illustration):

```
Wq = [[1, 0],    Wk = [[0, 1],    Wv = [[1, 0],
      [0, 1],          [1, 0],          [0, 1],
      [1, 0],          [0, 1],          [1, 0],
      [0, 1]]          [1, 0]]          [0, 1]]
```

Calculate:
1. The Query (Q), Key (K), and Value (V) matrices
2. The attention scores (before softmax)
3. The attention weights (after softmax, assuming temperature=1)
4. The output for the first token

### Solution

**Step 1: Compute Q, K, V**

```
Q = X @ Wq                    K = X @ Wk                    V = X @ Wv
  = [[1,0,1,0],                 = [[1,0,1,0],                 = [[1,0,1,0],
     [0,1,0,1],                    [0,1,0,1],                    [0,1,0,1],
     [1,1,0,0]]                    [1,1,0,0]]                    [1,1,0,0]]
    @ [[1,0],                     @ [[0,1],                     @ [[1,0],
       [0,1],                        [1,0],                        [0,1],
       [1,0],                        [0,1],                        [1,0],
       [0,1]]                        [1,0]]                        [0,1]]

Q = [[2, 0],                  K = [[0, 2],                  V = [[2, 0],
     [0, 2],                       [2, 0],                       [0, 2],
     [1, 1]]                       [1, 1]]                       [1, 1]]
```

**Step 2: Attention Scores (Q @ K^T / sqrt(dk))**

```
scores = Q @ K^T / sqrt(2)

Q @ K^T = [[2,0],      [[0, 2],      [[0, 4, 2],
           [0,2],   @   [2, 0],   =   [4, 0, 2],
           [1,1]]       [1, 1]]       [2, 2, 2]]

scores = [[0, 4, 2],     / sqrt(2)  =  [[0.00, 2.83, 1.41],
          [4, 0, 2],                     [2.83, 0.00, 1.41],
          [2, 2, 2]]                     [1.41, 1.41, 1.41]]
```

**Step 3: Attention Weights (Softmax per row)**

```
Row 1: softmax([0.00, 2.83, 1.41]) = [0.05, 0.76, 0.19]
Row 2: softmax([2.83, 0.00, 1.41]) = [0.76, 0.05, 0.19]
Row 3: softmax([1.41, 1.41, 1.41]) = [0.33, 0.33, 0.33]

weights = [[0.05, 0.76, 0.19],
           [0.76, 0.05, 0.19],
           [0.33, 0.33, 0.33]]
```

**Step 4: Output for First Token**

```
output[0] = 0.05 * V[0] + 0.76 * V[1] + 0.19 * V[2]
          = 0.05 * [2,0] + 0.76 * [0,2] + 0.19 * [1,1]
          = [0.10, 0] + [0, 1.52] + [0.19, 0.19]
          = [0.29, 1.71]
```

**Interpretation:** Token 1 ("The") attends mostly to Token 2 ("cat") with weight 0.76, picking up cat's representation.

---

## Problem 2 | Skill-Builder
**Concept:** Architecture Selection
**Centrality:** Encoder (7), Decoder (7)
**Estimated Time:** 20-25 minutes

### Problem Statement

For each of the following tasks, recommend whether to use an encoder-only (BERT-style), decoder-only (GPT-style), or encoder-decoder (T5-style) architecture. Justify your choice based on the task requirements.

1. **Sentiment Analysis:** Classify movie reviews as positive/negative
2. **Machine Translation:** Translate English to French
3. **Code Generation:** Generate Python code from natural language description
4. **Named Entity Recognition:** Identify person/organization/location entities in text
5. **Summarization:** Generate a short summary of a long article
6. **Semantic Similarity:** Determine if two sentences have the same meaning

### Solution

| Task | Architecture | Justification |
|------|--------------|---------------|
| **Sentiment Analysis** | Encoder-only (BERT) | Full input available; need to understand entire review; classification head on [CLS] token; bidirectional context captures sentiment modifiers ("not bad") |
| **Machine Translation** | Encoder-Decoder (T5) | Input and output are different sequences; encoder processes source bidirectionally; decoder generates target autoregressively; cross-attention aligns source-target |
| **Code Generation** | Decoder-only (GPT) | Autoregressive generation required; code has strong left-to-right dependencies; modern code models (Codex, StarCoder) are decoder-only; in-context learning with examples |
| **Named Entity Recognition** | Encoder-only (BERT) | Token-level classification; need bidirectional context to disambiguate entities; no generation required; add classification head per token |
| **Summarization** | Decoder-only OR Encoder-Decoder | Both valid. Decoder-only: treat as continuation "TL;DR:". Encoder-decoder: encode full article, generate summary. T5/BART traditionally used; modern LLMs do well with prompting |
| **Semantic Similarity** | Encoder-only (BERT) | Need embeddings for comparison; bidirectional encoding captures meaning; compute similarity between [CLS] embeddings or use cross-encoder architecture |

**Key Decision Framework:**
1. Is generation required? -> Decoder needed
2. Is input fully available? -> Encoder beneficial
3. Are input/output different lengths/languages? -> Encoder-decoder ideal
4. Is in-context learning important? -> Decoder-only scales better

---

## Problem 3 | Skill-Builder
**Concept:** BERT vs GPT Deep Dive
**Centrality:** BERT (6), GPT (6)
**Estimated Time:** 25-30 minutes

### Problem Statement

You're building a customer support system that needs to:
1. Classify incoming tickets into categories (billing, technical, general)
2. Generate automated responses for common questions
3. Extract key information (product name, issue type, urgency)

Design a system using BERT and/or GPT components. For each sub-task:
- Specify which model type to use
- Explain the input/output format
- Describe any fine-tuning needed

### Solution

**System Architecture:**

```
Incoming Ticket
       ↓
┌──────────────────────────────────────────────────────────┐
│  Stage 1: Classification (BERT)                          │
│  Input: "[CLS] ticket text [SEP]"                        │
│  Output: billing/technical/general                       │
└────────────────────────┬─────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│  Stage 2: Information Extraction (BERT)                  │
│  Input: "[CLS] ticket text [SEP]"                        │
│  Output: Token-level tags (product, issue, urgency)      │
└────────────────────────┬─────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│  Stage 3: Response Generation (GPT)                      │
│  Input: "Category: {cat}\nInfo: {extracted}\nTicket:     │
│          {text}\nResponse:"                              │
│  Output: Generated response text                         │
└──────────────────────────────────────────────────────────┘
```

**Stage 1: Classification (BERT)**
```python
# Fine-tuning setup
model = BertForSequenceClassification.from_pretrained('bert-base')
model.classifier = nn.Linear(768, 3)  # 3 categories

# Input format
input = tokenizer("[CLS] My bill is incorrect [SEP]", return_tensors="pt")
logits = model(**input).logits
category = logits.argmax()  # -> "billing"
```

**Stage 2: Information Extraction (BERT)**
```python
# Token classification (NER-style)
model = BertForTokenClassification.from_pretrained('bert-base')
model.classifier = nn.Linear(768, num_tags)  # BIO tags

# Input/Output
input: "My iPhone 15 won't charge after update"
output: ["O", "B-PRODUCT", "I-PRODUCT", "O", "B-ISSUE", "O", "O"]
```

**Stage 3: Response Generation (GPT)**
```python
# Few-shot or fine-tuned GPT
prompt = f"""Category: technical
Extracted: Product=iPhone 15, Issue=charging
Ticket: My iPhone 15 won't charge after update

Response:"""

response = gpt.generate(prompt, max_tokens=150)
# "Thank you for contacting support. For charging issues after
#  an update, please try: 1) Restart your iPhone 15..."
```

**Why This Architecture:**
- **BERT for understanding:** Bidirectional context essential for accurate classification and extraction
- **GPT for generation:** Autoregressive generation produces fluent, coherent responses
- **Pipeline approach:** Each component optimized for its task; easier to debug and improve individually

---

## Problem 4 | Challenge
**Concept:** Efficient Transformers and Scaling
**Centrality:** Flash Attention (4), Scaling (3)
**Estimated Time:** 35-40 minutes

### Problem Statement

You need to deploy a transformer-based document analysis system with the following constraints:
- Documents up to 32,768 tokens
- Latency requirement: < 500ms per document
- Hardware: Single A100 GPU (80GB)
- Task: Extract all dates, monetary amounts, and named entities

Design the system addressing:
1. Attention mechanism choice (standard, sparse, flash)
2. Memory budget analysis
3. Optimization techniques to meet latency requirements
4. Trade-offs in your design choices

### Solution

**1. Attention Mechanism Analysis:**

| Mechanism | Memory | Compute | Quality | Recommendation |
|-----------|--------|---------|---------|----------------|
| Standard | O(n^2) = 4GB for 32K | O(n^2) | Full | Too slow |
| Sparse (Longformer) | O(n*w) ~100MB | O(n*w) | Good | Viable |
| Flash Attention | O(n) ~50MB | O(n^2) but fast | Full | Best choice |

**Recommendation: Flash Attention**
- Exact attention (no approximation)
- Memory efficient (tiled computation)
- 2-4x faster than standard attention
- Enables 32K context on A100

**2. Memory Budget Analysis:**

```
Model: BERT-large or similar encoder (340M params)
- Model weights (fp16): 340M * 2 bytes = 680MB
- Activations (32K seq, batch=1): ~4GB with Flash Attention
- KV cache: Not needed for encoder-only
- Overhead: ~1GB

Total: ~6GB << 80GB available

Can increase batch size or model size!
```

**Optimized Configuration:**
```python
config = {
    "model": "longformer-base-4096",  # or custom model with Flash Attention
    "attention": "flash_attention_2",
    "precision": "fp16",  # or bf16
    "batch_size": 8,  # Process multiple docs
    "max_length": 32768,
    "chunk_strategy": "sliding_window"  # For >32K docs
}
```

**3. Latency Optimization:**

| Technique | Latency Reduction | Implementation |
|-----------|-------------------|----------------|
| Flash Attention | 2-4x | `attn_implementation="flash_attention_2"` |
| FP16/BF16 | 2x | `torch.autocast("cuda", dtype=torch.float16)` |
| Torch Compile | 1.3-2x | `model = torch.compile(model)` |
| Batching | Linear | Process 8 docs together |
| CUDA Graphs | 1.2x | Capture and replay computation |

**Latency Estimate:**
```
Base (fp32, standard attn): ~2000ms
+ Flash Attention: ~600ms
+ FP16: ~350ms
+ Torch Compile: ~280ms
+ Optimized CUDA: ~220ms

Final: ~220ms < 500ms requirement
```

**4. System Architecture:**

```python
import torch
from transformers import AutoModel, AutoTokenizer

class DocumentAnalyzer:
    def __init__(self):
        self.model = AutoModel.from_pretrained(
            "allenai/longformer-base-4096",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16
        ).cuda()
        self.model = torch.compile(self.model)
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

        # NER head for entity extraction
        self.ner_head = nn.Linear(768, num_entity_types).cuda().half()

    @torch.inference_mode()
    def analyze(self, documents: list[str]) -> list[dict]:
        # Batch tokenization
        inputs = self.tokenizer(
            documents,
            max_length=32768,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to("cuda")

        # Forward pass
        with torch.autocast("cuda", dtype=torch.float16):
            outputs = self.model(**inputs)
            entity_logits = self.ner_head(outputs.last_hidden_state)

        # Post-process to extract entities
        return self.extract_entities(entity_logits, inputs)
```

**Trade-offs:**

| Choice | Pro | Con |
|--------|-----|-----|
| Flash Attention | Exact, fast, memory efficient | Requires Ampere+ GPU |
| FP16 | 2x speed | Slight precision loss |
| Longformer base | Fits memory | Less capacity than large |
| Encoder-only | Fast inference | Cannot generate explanations |

---

## Problem 5 | Debug/Fix
**Concept:** Transformer Training Issues
**Centrality:** Multiple components
**Estimated Time:** 25-30 minutes

### Problem Statement

A colleague is training a transformer model and encounters the following issues. Diagnose each problem and propose solutions.

**Issue 1:** Training loss is NaN after a few hundred steps
```
Step 100: loss = 2.34
Step 200: loss = 1.89
Step 300: loss = nan
Step 400: loss = nan
```

**Issue 2:** BERT fine-tuning achieves 95% training accuracy but only 60% validation accuracy
```
Epoch 1: train_acc=0.72, val_acc=0.68
Epoch 5: train_acc=0.89, val_acc=0.65
Epoch 10: train_acc=0.95, val_acc=0.60
```

**Issue 3:** GPT generation produces repetitive text
```
Input: "The weather today is"
Output: "The weather today is nice and nice and nice and nice and..."
```

**Issue 4:** Attention patterns show all heads attending to [CLS] or [SEP] tokens only
```
Attention visualization shows >90% weight on special tokens
```

### Solution

**Issue 1: NaN Loss - Gradient Explosion**

**Diagnosis:** Learning rate too high or missing gradient clipping causing exploding gradients.

**Solutions:**
```python
# Solution 1: Lower learning rate
optimizer = AdamW(model.parameters(), lr=1e-5)  # Was probably 1e-3

# Solution 2: Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Solution 3: Use learning rate warmup
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,
    num_training_steps=total_steps
)

# Solution 4: Check for inf/nan in inputs
assert not torch.isnan(inputs).any()
assert not torch.isinf(inputs).any()

# Solution 5: Use mixed precision properly
scaler = GradScaler()
with autocast():
    loss = model(inputs)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Issue 2: Overfitting - BERT Fine-tuning**

**Diagnosis:** Model memorizing training data; insufficient regularization.

**Solutions:**
```python
# Solution 1: Add dropout
model.config.hidden_dropout_prob = 0.3  # Default 0.1
model.config.attention_probs_dropout_prob = 0.3

# Solution 2: Reduce model capacity
model = BertForSequenceClassification.from_pretrained('bert-base')  # Not bert-large

# Solution 3: Data augmentation
from nlpaug import Augmenter
augmenter = naw.SynonymAug()

# Solution 4: Early stopping
early_stopping = EarlyStopping(patience=3, metric='val_accuracy')

# Solution 5: Weight decay
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

# Solution 6: Freeze early layers
for param in model.bert.embeddings.parameters():
    param.requires_grad = False
for layer in model.bert.encoder.layer[:6]:
    for param in layer.parameters():
        param.requires_grad = False
```

**Issue 3: Repetitive Generation - GPT**

**Diagnosis:** Greedy decoding or temperature too low; model stuck in repetition loop.

**Solutions:**
```python
# Solution 1: Use sampling with temperature
output = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.7,  # Add randomness
    top_p=0.9,        # Nucleus sampling
    top_k=50,         # Limit vocabulary
)

# Solution 2: Repetition penalty
output = model.generate(
    input_ids,
    repetition_penalty=1.2,  # Penalize repeated tokens
    no_repeat_ngram_size=3,  # Block repeated 3-grams
)

# Solution 3: Length penalty for beam search
output = model.generate(
    input_ids,
    num_beams=5,
    length_penalty=1.0,
    early_stopping=True
)
```

**Issue 4: Degenerate Attention - All to Special Tokens**

**Diagnosis:** Model taking shortcut; special tokens become "sinks" for unused attention.

**Solutions:**
```python
# Solution 1: Remove or reduce special token attention
# In attention, mask out CLS/SEP from being attended to (except when needed)

# Solution 2: Use attention entropy regularization
def attention_entropy_loss(attention_weights):
    # Encourage diverse attention patterns
    entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-9), dim=-1)
    return -entropy.mean()  # Maximize entropy

loss = task_loss + 0.1 * attention_entropy_loss(attentions)

# Solution 3: Initialize attention more carefully
# Ensure attention weights don't collapse during initialization

# Solution 4: Check training data
# Special tokens shouldn't dominate important information

# Solution 5: Use relative positional encoding
# Absolute positions can create biases toward certain positions
```

**Debugging Checklist:**

| Issue | Quick Check | Tool |
|-------|-------------|------|
| NaN loss | `torch.isnan(loss)` | TensorBoard |
| Overfitting | Train/val gap | Learning curves |
| Repetition | Sample multiple outputs | Manual inspection |
| Attention collapse | Visualize attention | BertViz |
