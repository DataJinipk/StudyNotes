# Practice Problems: Natural Language Processing

**Source:** notes/natural-language-processing/natural-language-processing-study-notes.md
**Concept Map:** notes/natural-language-processing/concept-maps/natural-language-processing-concept-map.md
**Flashcards:** notes/natural-language-processing/flashcards/natural-language-processing-flashcards.md
**Date Generated:** 2026-01-07
**Total Problems:** 5
**Distribution:** 1 Warm-Up | 2 Skill-Builder | 1 Challenge | 1 Debug/Fix

---

## Problem Distribution

| Type | Count | Purpose | Time Estimate |
|------|-------|---------|---------------|
| Warm-Up | 1 | Activate prior knowledge; build confidence | 10-15 min |
| Skill-Builder | 2 | Develop core procedural fluency | 20-30 min each |
| Challenge | 1 | Extend to complex scenarios | 40-50 min |
| Debug/Fix | 1 | Identify and correct common errors | 20-25 min |

---

## Problem 1 | Warm-Up
**Concept:** Tokenization and Text Preprocessing
**Difficulty:** ⭐☆☆☆☆
**Estimated Time:** 15 minutes
**Prerequisites:** Basic Python, string manipulation

### Problem Statement

You're building a sentiment analysis system for social media. Given the following tweet, perform preprocessing and compare different tokenization approaches:

**Input Tweet:**
```
"OMG!!! 😍 This new iPhone 15 Pro Max is AMAZING!!! Best $1,199 I've ever spent 💰💰💰 #Apple #iPhone15 @AppleSupport you rock!! 🎉🎉"
```

**Tasks:**
1. Apply standard preprocessing (lowercasing, punctuation handling)
2. Tokenize using:
   - Word-level tokenization (split on whitespace/punctuation)
   - Subword tokenization (simulate BPE with common subwords)
3. Calculate the vocabulary size difference between approaches
4. Identify which tokens would be OOV (out-of-vocabulary) in a standard English vocabulary

### Hints

<details>
<summary>Hint 1 (Conceptual)</summary>
Consider what information is lost with each preprocessing step. Emojis carry sentiment information!
</details>

<details>
<summary>Hint 2 (Technical)</summary>
For BPE simulation, common subwords include: "##ing", "##ed", "##ly", "##tion", and root words.
</details>

<details>
<summary>Hint 3 (Approach)</summary>
Standard vocabulary typically excludes: hashtags, mentions, emojis, product names, and prices.
</details>

### Solution

**Step 1: Standard Preprocessing**

```python
original = "OMG!!! 😍 This new iPhone 15 Pro Max is AMAZING!!! Best $1,199 I've ever spent 💰💰💰 #Apple #iPhone15 @AppleSupport you rock!! 🎉🎉"

# Lowercasing
lowercased = original.lower()
# "omg!!! 😍 this new iphone 15 pro max is amazing!!! best $1,199 i've ever spent 💰💰💰 #apple #iphone15 @applesupport you rock!! 🎉🎉"

# Punctuation handling (keep some for meaning)
import re
# Option A: Remove all punctuation
no_punct = re.sub(r'[^\w\s]', '', lowercased)
# "omg  this new iphone 15 pro max is amazing best 1199 ive ever spent   apple iphone15 applesupport you rock "

# Option B: Preserve emojis and hashtags (recommended for sentiment)
preserved = re.sub(r'[!.,;:?]+', ' ', lowercased)  # Remove only standard punctuation
# "omg  😍 this new iphone 15 pro max is amazing  best $1 199 i've ever spent 💰💰💰 #apple #iphone15 @applesupport you rock  🎉🎉"
```

**Information lost with aggressive preprocessing:**
- Emoji sentiment (😍, 💰, 🎉 = positive)
- Emphasis from repeated punctuation (!!!)
- Social signals (#hashtags, @mentions)
- Price information ($1,199)

**Step 2: Tokenization Comparison**

**Word-level tokenization:**
```python
word_tokens = [
    "omg", "😍", "this", "new", "iphone", "15", "pro", "max",
    "is", "amazing", "best", "$1,199", "i've", "ever", "spent",
    "💰", "💰", "💰", "#apple", "#iphone15", "@applesupport",
    "you", "rock", "🎉", "🎉"
]
# Vocabulary needed: 19 unique tokens
```

**Subword tokenization (BPE-style):**
```python
subword_tokens = [
    "om", "##g", "😍", "this", "new", "i", "##phone", "15",
    "pro", "max", "is", "amaz", "##ing", "best", "$", "1",
    ",", "199", "i", "'", "ve", "ever", "spent", "💰", "💰",
    "💰", "#", "apple", "#", "iphone", "##15", "@", "apple",
    "##support", "you", "rock", "🎉", "🎉"
]
# Base vocabulary: ~30K subwords covers most words
```

**Step 3: Vocabulary Size Comparison**

| Approach | Tokens for this tweet | Typical Vocabulary Size |
|----------|----------------------|-------------------------|
| Word-level | 19 unique | 50,000-100,000+ |
| Subword (BPE) | ~25 subwords | 30,000-50,000 |

**Step 4: OOV Analysis (Standard English Vocabulary)**

Likely OOV tokens:
- **Product names:** "iphone", "iphone15"
- **Social tokens:** "#apple", "#iphone15", "@applesupport"
- **Emojis:** "😍", "💰", "🎉"
- **Informal:** "omg"
- **Price format:** "$1,199"

**Subword advantage:** "iphone" → ["i", "##phone"] — both subwords exist in vocabulary!

### Key Takeaways

1. **Preprocessing tradeoffs:** Removing punctuation/emojis loses sentiment signals
2. **Subword tokenization:** Handles OOV words by breaking into known subwords
3. **Domain matters:** Social media requires preserving emojis, hashtags, mentions
4. **Modern approach:** Use pre-trained tokenizers (e.g., `BertTokenizer`) that handle these automatically

### Common Mistakes to Avoid

| Mistake | Why It's Wrong | Correct Approach |
|---------|----------------|------------------|
| Removing all emojis | Loses sentiment information | Keep or map to sentiment labels |
| Case-folding product names | "iPhone" → "iphone" loses brand recognition | Use case-aware models or preserve |
| Splitting contractions wrong | "I've" → "I", "ve" | Use tokenizer-aware splitting: "I", "'ve" |

---

## Problem 2 | Skill-Builder
**Concept:** Self-Attention Calculation
**Difficulty:** ⭐⭐⭐☆☆
**Estimated Time:** 25 minutes
**Prerequisites:** Linear algebra, softmax function

### Problem Statement

Compute the self-attention output for a simple sequence. Given a sequence of 3 tokens with embedding dimension d=4:

**Input embeddings X:**
```
Token 1 (The):   [1.0, 0.0, 1.0, 0.0]
Token 2 (cat):   [0.0, 1.0, 0.0, 1.0]
Token 3 (sat):   [0.5, 0.5, 0.5, 0.5]
```

**Weight matrices (d_k = d_v = 2):**
```
W_Q = [[1, 0],      W_K = [[0, 1],      W_V = [[1, 0],
       [0, 1],             [1, 0],             [0, 1],
       [1, 0],             [0, 1],             [1, 0],
       [0, 1]]             [1, 0]]             [0, 1]]
```

**Tasks:**
1. Compute Q, K, V matrices by projecting X through the weight matrices
2. Calculate attention scores using scaled dot-product: QK^T / √d_k
3. Apply softmax to get attention weights
4. Compute the final attention output
5. Interpret: Which tokens does "cat" attend to most?

### Hints

<details>
<summary>Hint 1 (Setup)</summary>
Q = X × W_Q gives a (3×2) matrix. Same for K and V.
</details>

<details>
<summary>Hint 2 (Scaling)</summary>
√d_k = √2 ≈ 1.414. Divide all scores by this before softmax.
</details>

<details>
<summary>Hint 3 (Softmax)</summary>
softmax([a, b, c]) = [e^a, e^b, e^c] / (e^a + e^b + e^c)
</details>

### Solution

**Step 1: Compute Q, K, V**

```
X (3×4):                W_Q (4×2):              Q = X × W_Q (3×2):
[[1, 0, 1, 0],          [[1, 0],                [[2, 0],     # "The"
 [0, 1, 0, 1],    ×      [0, 1],          =      [0, 2],     # "cat"
 [0.5, 0.5, 0.5, 0.5]]   [1, 0],                 [1, 1]]     # "sat"
                         [0, 1]]

K = X × W_K (3×2):                V = X × W_V (3×2):
[[0, 2],     # "The"              [[2, 0],     # "The"
 [2, 0],     # "cat"               [0, 2],     # "cat"
 [1, 1]]     # "sat"               [1, 1]]     # "sat"
```

**Step 2: Compute QK^T (attention scores)**

```
Q (3×2) × K^T (2×3):

       K^T:  [[0, 2, 1],
              [2, 0, 1]]

Q × K^T = [[2×0 + 0×2, 2×2 + 0×0, 2×1 + 0×1],    [[0, 4, 2],
           [0×0 + 2×2, 0×2 + 2×0, 0×1 + 2×1],  =  [4, 0, 2],
           [1×0 + 1×2, 1×2 + 1×0, 1×1 + 1×1]]     [2, 2, 2]]
```

**Step 3: Scale by √d_k**

```
√d_k = √2 ≈ 1.414

Scaled scores = QK^T / √2:
[[0.00, 2.83, 1.41],
 [2.83, 0.00, 1.41],
 [1.41, 1.41, 1.41]]
```

**Step 4: Apply softmax (row-wise)**

```
Row 1 ("The" attending): softmax([0.00, 2.83, 1.41])
  e^0 = 1.00, e^2.83 = 16.95, e^1.41 = 4.10
  sum = 22.05
  = [0.05, 0.77, 0.19]  # "The" attends mostly to "cat"

Row 2 ("cat" attending): softmax([2.83, 0.00, 1.41])
  = [0.77, 0.05, 0.19]  # "cat" attends mostly to "The"

Row 3 ("sat" attending): softmax([1.41, 1.41, 1.41])
  = [0.33, 0.33, 0.33]  # "sat" attends equally to all
```

**Attention weights matrix:**
```
         The   cat   sat
The    [[0.05, 0.77, 0.19],
cat     [0.77, 0.05, 0.19],
sat     [0.33, 0.33, 0.33]]
```

**Step 5: Compute output (Attention × V)**

```
Attention (3×3) × V (3×2):

V = [[2, 0],
     [0, 2],
     [1, 1]]

Output for "The":  0.05×[2,0] + 0.77×[0,2] + 0.19×[1,1] = [0.29, 1.73]
Output for "cat":  0.77×[2,0] + 0.05×[0,2] + 0.19×[1,1] = [1.73, 0.29]
Output for "sat":  0.33×[2,0] + 0.33×[0,2] + 0.33×[1,1] = [1.00, 1.00]

Final output (3×2):
[[0.29, 1.73],   # "The" - influenced mainly by "cat"
 [1.73, 0.29],   # "cat" - influenced mainly by "The"
 [1.00, 1.00]]   # "sat" - balanced influence
```

**Step 6: Interpretation**

| Token | Attends most to | Weight | Interpretation |
|-------|-----------------|--------|----------------|
| "The" | "cat" | 0.77 | Determiner connects to its noun |
| "cat" | "The" | 0.77 | Noun connects to its determiner |
| "sat" | All equally | 0.33 each | Verb relates to full context |

**Key insight:** The attention mechanism learned to connect related tokens (determiner-noun relationship) even with random-looking weight matrices!

### Key Takeaways

1. **Scaling prevents saturation:** Without √d_k, large dot products → extreme softmax → vanishing gradients
2. **Attention is learned:** Weight matrices W_Q, W_K, W_V are trained to produce useful attention patterns
3. **Interpretability:** Attention weights show which tokens influence each other
4. **Symmetric patterns:** In self-attention, if A attends to B, often B attends to A

---

## Problem 3 | Skill-Builder
**Concept:** BERT Fine-tuning for Classification
**Difficulty:** ⭐⭐⭐☆☆
**Estimated Time:** 30 minutes
**Prerequisites:** PyTorch basics, Transformers library

### Problem Statement

You're fine-tuning BERT for a 3-class sentiment classification task (positive, neutral, negative) on product reviews. Your training shows the following metrics over epochs:

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|------------|-----------|----------|---------|
| 1 | 0.92 | 58% | 0.85 | 62% |
| 2 | 0.45 | 81% | 0.52 | 78% |
| 3 | 0.18 | 94% | 0.61 | 76% |
| 4 | 0.08 | 98% | 0.89 | 74% |
| 5 | 0.03 | 99% | 1.21 | 72% |

**Tasks:**
1. Diagnose the training problem evident in these metrics
2. Identify the optimal checkpoint to use
3. Propose 3 specific solutions to address the issue
4. Write the PyTorch code snippet to implement one solution
5. Explain why BERT fine-tuning is particularly prone to this issue

### Hints

<details>
<summary>Hint 1 (Diagnosis)</summary>
Compare training vs validation trends after epoch 2. What's happening to the gap?
</details>

<details>
<summary>Hint 2 (Solutions)</summary>
Consider: learning rate, regularization, data augmentation, early stopping.
</details>

<details>
<summary>Hint 3 (BERT-specific)</summary>
BERT has 110M+ parameters. How does this relate to the problem with small datasets?
</details>

### Solution

**Step 1: Diagnosis**

**Problem: Overfitting**

Evidence:
- Training loss: 0.92 → 0.03 (continuously decreasing)
- Validation loss: 0.85 → 0.52 → 0.61 → 0.89 → 1.21 (minimum at epoch 2, then increasing)
- Train/Val accuracy gap: 58/62% → 99/72% (widening gap)

```
Training curve visualization:
Loss
│
1.0├─●─────────────────────────── Train
   │  ╲                    ●─────Val
0.8├───●─────────────────●
   │    ╲              ●
0.6├─────●───────────●
   │      ╲
0.4├───────●─────●
   │        ╲  ↑ Optimal point
0.2├─────────●
   │          ╲
0.0├───────────●──●──●
   └──────────────────────────── Epoch
       1    2    3    4    5
```

**Step 2: Optimal Checkpoint**

**Epoch 2** - lowest validation loss (0.52) and best validation accuracy (78%)

```python
# Model selection based on validation loss
best_model = model_epoch_2
best_val_loss = 0.52
best_val_acc = 0.78
```

**Step 3: Three Solutions**

**Solution A: Learning Rate Reduction with Warmup**
```python
from transformers import get_linear_schedule_with_warmup

# Lower base learning rate (BERT default 5e-5 → 2e-5)
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

# Warmup for first 10% of steps
num_training_steps = len(train_dataloader) * num_epochs
num_warmup_steps = int(0.1 * num_training_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)
```

**Solution B: Dropout and Weight Decay**
```python
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=3,
    hidden_dropout_prob=0.3,      # Increase from 0.1 default
    attention_probs_dropout_prob=0.3,
    classifier_dropout=0.5        # High dropout on classifier head
)

# Weight decay in optimizer
optimizer = AdamW(
    model.parameters(),
    lr=2e-5,
    weight_decay=0.1  # Increase from 0.01
)
```

**Solution C: Early Stopping**
```python
class EarlyStopping:
    def __init__(self, patience=2, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

# Usage in training loop
early_stopping = EarlyStopping(patience=2)
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_dataloader)
    val_loss = evaluate(model, val_dataloader)

    if early_stopping(val_loss):
        print(f"Early stopping at epoch {epoch}")
        break
```

**Step 4: Complete Implementation (Solution A)**

```python
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

# Setup
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=3
)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Hyperparameters to prevent overfitting
LEARNING_RATE = 2e-5  # Reduced from 5e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 3  # Fewer epochs
WARMUP_RATIO = 0.1

# Optimizer with weight decay
optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    eps=1e-8
)

# Scheduler with warmup
total_steps = len(train_dataloader) * NUM_EPOCHS
warmup_steps = int(WARMUP_RATIO * total_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# Training loop with gradient clipping
for epoch in range(NUM_EPOCHS):
    model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        # Gradient clipping - prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

    # Validation
    model.eval()
    val_loss = evaluate(model, val_dataloader)
    print(f"Epoch {epoch+1}: Val Loss = {val_loss:.4f}")
```

**Step 5: Why BERT is Prone to Overfitting**

| Factor | Explanation |
|--------|-------------|
| **Model size** | 110M parameters (BERT-base) vs. typically small fine-tuning datasets |
| **Pre-training** | Already learned rich representations; fine-tuning can quickly memorize new data |
| **Capacity** | 12 layers, 768 hidden size = massive capacity for small datasets |
| **Few epochs needed** | BERT fine-tuning typically needs 2-4 epochs; more leads to overfitting |

**Rule of thumb:**
- < 1K samples: High overfitting risk; consider freezing lower layers
- 1K-10K samples: Standard fine-tuning with regularization
- > 10K samples: Can train more aggressively

### Key Takeaways

1. **Monitor validation loss**, not training loss, for model selection
2. **BERT overfits quickly** on small datasets; 2-4 epochs usually sufficient
3. **Combine strategies:** Learning rate warmup + weight decay + early stopping
4. **Layer-wise learning rates:** Lower layers (general features) can use smaller learning rates

---

## Problem 4 | Challenge
**Concept:** End-to-End NER System Design
**Difficulty:** ⭐⭐⭐⭐☆
**Estimated Time:** 45 minutes
**Prerequisites:** All NLP concepts, system design

### Problem Statement

A healthcare company needs an NER system to extract medical entities from clinical notes. The system must identify:

- **MEDICATION:** Drug names, dosages (e.g., "Lisinopril 10mg")
- **CONDITION:** Diseases, symptoms (e.g., "type 2 diabetes", "chest pain")
- **PROCEDURE:** Medical procedures (e.g., "coronary angiography")
- **ANATOMY:** Body parts (e.g., "left ventricle", "lumbar spine")
- **LAB_VALUE:** Test results (e.g., "HbA1c 7.2%", "WBC 12,000")

**Constraints:**
- Must handle abbreviations (e.g., "HTN" = hypertension, "CABG" = coronary artery bypass graft)
- Must process 10,000 notes per day
- Requires >90% F1 score for medication extraction (safety critical)
- Must provide confidence scores for human review
- Notes contain PHI (Protected Health Information) - HIPAA compliance required

**Tasks:**
1. Design the complete NER pipeline architecture
2. Select and justify the base model
3. Design the annotation strategy for creating training data
4. Implement the entity extraction with confidence scoring
5. Propose evaluation methodology and error analysis framework

### Solution

**Task 1: Pipeline Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                    Clinical Notes NER Pipeline                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Raw Note   │───►│ Preprocessor │───►│  Tokenizer   │       │
│  │              │    │              │    │   (Clinical) │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                 │                │
│                                                 ▼                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              ClinicalBERT / PubMedBERT                    │   │
│  │         (Domain-specific pre-trained model)               │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                      │
│            ┌──────────────┼──────────────┐                      │
│            ▼              ▼              ▼                       │
│     ┌───────────┐  ┌───────────┐  ┌───────────┐                 │
│     │  NER Head │  │Confidence │  │Abbreviation│                │
│     │  (BiLSTM  │  │Calibration│  │  Expander  │                │
│     │   + CRF)  │  │           │  │            │                │
│     └─────┬─────┘  └─────┬─────┘  └─────┬──────┘                │
│           │              │              │                        │
│           └──────────────┼──────────────┘                        │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │               Post-processing Layer                       │   │
│  │  • Entity linking to medical ontologies (UMLS, RxNorm)   │   │
│  │  • Confidence-based routing                               │   │
│  │  • PHI detection and flagging                            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          │                                       │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Output Layer                           │   │
│  │  • Structured JSON with entities, spans, confidence      │   │
│  │  • Low-confidence queue for human review                 │   │
│  │  • Audit logging for HIPAA compliance                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Task 2: Model Selection**

**Primary Choice: PubMedBERT + BiLSTM-CRF**

| Model Option | Pros | Cons | Decision |
|--------------|------|------|----------|
| **PubMedBERT** | Pre-trained on biomedical text; understands medical terminology | Requires fine-tuning | ✅ Selected |
| ClinicalBERT | Trained on clinical notes (MIMIC-III) | May have PHI exposure concerns | Alternative |
| BioBERT | Broad biomedical coverage | Less clinical-specific | Backup |
| General BERT | Widely available | Poor on medical terms | ❌ Rejected |

**Architecture Justification:**
```python
class ClinicalNERModel(nn.Module):
    def __init__(self, num_labels=11):  # BIO tagging: 5 entities × 2 + O
        super().__init__()
        self.bert = AutoModel.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(768, 256, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(512, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        lstm_out, _ = self.lstm(sequence_output)
        emissions = self.classifier(lstm_out)

        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask.bool())
            return loss
        else:
            predictions = self.crf.decode(emissions, mask=attention_mask.bool())
            return predictions
```

**Why BiLSTM-CRF on top of BERT:**
- CRF enforces valid BIO sequences (no I-MEDICATION after B-CONDITION)
- BiLSTM captures local dependencies complementing BERT's global attention
- Proven effective for clinical NER benchmarks

**Task 3: Annotation Strategy**

**Phase 1: Data Collection (Week 1-2)**
```
Source: De-identified clinical notes from EHR
Volume: 5,000 notes (target: 500 fully annotated)
Selection: Stratified by department (cardiology, oncology, general medicine)
```

**Phase 2: Annotation Guidelines**
```markdown
## Entity Annotation Rules

### MEDICATION
- Include drug name + dosage + route + frequency
- Span: "Lisinopril 10mg PO daily" → MEDICATION
- Nested: If dosage is separate, annotate together

### CONDITION
- Include modifiers: "uncontrolled type 2 diabetes"
- Negated conditions: "denies chest pain" → Still annotate "chest pain" as CONDITION
- Family history: "family history of MI" → Annotate "MI" with attribute

### Abbreviation Handling
- Annotate the abbreviation, not expansion
- "HTN" → CONDITION (with note that it means hypertension)
```

**Phase 3: Quality Control**
```python
# Inter-annotator agreement calculation
from sklearn.metrics import cohen_kappa_score

def calculate_iaa(annotator1, annotator2):
    """
    Calculate Cohen's Kappa for entity boundary + type agreement
    Target: κ > 0.8 for each entity type
    """
    # Token-level labels comparison
    kappa = cohen_kappa_score(annotator1, annotator2)
    return kappa

# Double-annotation on 20% of data
# Adjudication for disagreements
# Target: 90%+ agreement before scaling
```

**Annotation Workflow:**
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Annotator A │    │ Annotator B │    │ Adjudicator │
│  (Nurse)    │    │  (Doctor)   │    │  (Expert)   │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │
       ▼                  ▼                  │
   ┌──────────────────────────┐              │
   │  Label Studio Platform   │              │
   │  (Overlap: 20% of docs)  │              │
   └────────────┬─────────────┘              │
                │                            │
                ▼                            │
       ┌────────────────┐                    │
       │ IAA Calculation│                    │
       │   κ > 0.8?     │                    │
       └───────┬────────┘                    │
               │ No                          │
               ▼                             │
       ┌────────────────┐                    │
       │  Disagreement  │◄───────────────────┘
       │  Adjudication  │
       └────────────────┘
```

**Task 4: Confidence Scoring Implementation**

```python
import torch
import torch.nn.functional as F
from torchcrf import CRF

class ClinicalNERWithConfidence(nn.Module):
    def __init__(self, num_labels=11):
        super().__init__()
        self.bert = AutoModel.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
        self.lstm = nn.LSTM(768, 256, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(512, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.temperature = nn.Parameter(torch.ones(1))  # Learned temperature

    def forward_with_confidence(self, input_ids, attention_mask):
        # Get emissions
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.lstm(outputs.last_hidden_state)
        emissions = self.classifier(lstm_out)

        # CRF decoding
        predictions = self.crf.decode(emissions, mask=attention_mask.bool())

        # Confidence calculation using marginal probabilities
        # Method: Monte Carlo dropout sampling
        confidences = self.calculate_confidence(
            input_ids, attention_mask, n_samples=10
        )

        return predictions, confidences

    def calculate_confidence(self, input_ids, attention_mask, n_samples=10):
        """
        Monte Carlo Dropout for uncertainty estimation
        """
        self.train()  # Enable dropout
        samples = []

        with torch.no_grad():
            for _ in range(n_samples):
                outputs = self.bert(input_ids, attention_mask=attention_mask)
                lstm_out, _ = self.lstm(outputs.last_hidden_state)
                emissions = self.classifier(lstm_out)
                probs = F.softmax(emissions / self.temperature, dim=-1)
                samples.append(probs)

        self.eval()

        # Stack and calculate variance
        stacked = torch.stack(samples)  # (n_samples, batch, seq_len, num_labels)
        mean_probs = stacked.mean(dim=0)
        variance = stacked.var(dim=0)

        # Confidence = max probability - uncertainty
        max_probs, _ = mean_probs.max(dim=-1)
        uncertainty = variance.sum(dim=-1)  # Total variance across labels

        confidence = max_probs - 0.5 * uncertainty
        return confidence.clamp(0, 1)


def extract_entities_with_confidence(text, model, tokenizer, threshold=0.85):
    """
    Extract entities with confidence-based routing
    """
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt', return_offsets_mapping=True)
    offset_mapping = inputs.pop('offset_mapping')

    # Predict with confidence
    predictions, confidences = model.forward_with_confidence(**inputs)

    # Extract entities
    entities = []
    current_entity = None

    for idx, (pred, conf) in enumerate(zip(predictions[0], confidences[0])):
        label = id2label[pred]

        if label.startswith('B-'):
            if current_entity:
                entities.append(current_entity)
            entity_type = label[2:]
            start, end = offset_mapping[0][idx]
            current_entity = {
                'type': entity_type,
                'start': start.item(),
                'end': end.item(),
                'confidence': conf.item(),
                'needs_review': conf.item() < threshold
            }
        elif label.startswith('I-') and current_entity:
            _, end = offset_mapping[0][idx]
            current_entity['end'] = end.item()
            current_entity['confidence'] = min(current_entity['confidence'], conf.item())
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)

    # Add text spans
    for entity in entities:
        entity['text'] = text[entity['start']:entity['end']]

    return entities


# Example output
"""
{
    "text": "Patient prescribed Lisinopril 10mg for HTN",
    "entities": [
        {
            "type": "MEDICATION",
            "text": "Lisinopril 10mg",
            "start": 19,
            "end": 34,
            "confidence": 0.94,
            "needs_review": false
        },
        {
            "type": "CONDITION",
            "text": "HTN",
            "start": 39,
            "end": 42,
            "confidence": 0.78,
            "needs_review": true  # Abbreviation - lower confidence
        }
    ]
}
"""
```

**Task 5: Evaluation Framework**

```python
from seqeval.metrics import classification_report, f1_score
from collections import defaultdict

class NERErrorAnalyzer:
    def __init__(self):
        self.error_types = defaultdict(list)

    def evaluate(self, y_true, y_pred, texts):
        """
        Comprehensive evaluation with error analysis
        """
        # Standard metrics
        report = classification_report(y_true, y_pred, output_dict=True)

        # Per-entity F1
        entity_f1 = {}
        for entity_type in ['MEDICATION', 'CONDITION', 'PROCEDURE', 'ANATOMY', 'LAB_VALUE']:
            entity_f1[entity_type] = report.get(entity_type, {}).get('f1-score', 0)

        # Error categorization
        self.categorize_errors(y_true, y_pred, texts)

        return {
            'overall_f1': report['micro avg']['f1-score'],
            'entity_f1': entity_f1,
            'medication_f1': entity_f1['MEDICATION'],  # Safety-critical
            'error_analysis': dict(self.error_types)
        }

    def categorize_errors(self, y_true, y_pred, texts):
        """
        Categorize errors for targeted improvement
        """
        for true_seq, pred_seq, text in zip(y_true, y_pred, texts):
            true_entities = self.extract_entities(true_seq)
            pred_entities = self.extract_entities(pred_seq)

            # False negatives (missed entities)
            for entity in true_entities:
                if entity not in pred_entities:
                    self.error_types['false_negative'].append({
                        'entity': entity,
                        'context': text,
                        'type': self.classify_fn_error(entity, text)
                    })

            # False positives (spurious entities)
            for entity in pred_entities:
                if entity not in true_entities:
                    self.error_types['false_positive'].append({
                        'entity': entity,
                        'context': text,
                        'type': self.classify_fp_error(entity, text)
                    })

    def classify_fn_error(self, entity, text):
        """Categorize why entity was missed"""
        if len(entity['text']) <= 3:
            return 'abbreviation'
        if ' ' in entity['text']:
            return 'multi_word'
        return 'rare_term'

    def generate_report(self):
        """
        Generate actionable error report
        """
        report = """
        NER Error Analysis Report
        =========================

        False Negatives by Category:
        - Abbreviations: {abbrev_count} ({abbrev_pct}%)
        - Multi-word entities: {multi_count} ({multi_pct}%)
        - Rare terms: {rare_count} ({rare_pct}%)

        Recommendations:
        1. Add abbreviation expansion module
        2. Increase training data for multi-word entities
        3. Consider entity linking to medical ontologies
        """
        return report

# Evaluation script
def run_evaluation(model, test_dataloader, threshold=0.90):
    """
    Run comprehensive evaluation with safety-critical check
    """
    analyzer = NERErrorAnalyzer()

    all_true, all_pred, all_texts = [], [], []

    for batch in test_dataloader:
        predictions = model.predict(batch)
        all_true.extend(batch['labels'])
        all_pred.extend(predictions)
        all_texts.extend(batch['texts'])

    results = analyzer.evaluate(all_true, all_pred, all_texts)

    # Safety check for medications
    if results['medication_f1'] < threshold:
        print(f"⚠️ WARNING: Medication F1 ({results['medication_f1']:.2%}) below threshold ({threshold:.0%})")
        print("Model NOT approved for production deployment")
        return results, False

    print(f"✅ Medication F1: {results['medication_f1']:.2%} - Meets safety threshold")
    return results, True
```

### Key Takeaways

1. **Domain-specific pre-training matters:** PubMedBERT significantly outperforms general BERT on medical text
2. **CRF layer enforces valid sequences:** Essential for NER to prevent invalid BIO transitions
3. **Confidence calibration enables safe deployment:** Route uncertain predictions to humans
4. **Annotation quality determines ceiling:** Invest in clear guidelines and expert annotators
5. **Error analysis drives improvement:** Categorize errors to target specific weaknesses

---

## Problem 5 | Debug/Fix
**Concept:** Transformer Training Issues
**Difficulty:** ⭐⭐⭐☆☆
**Estimated Time:** 25 minutes
**Prerequisites:** Transformer architecture, training dynamics

### Problem Statement

A colleague is training a Transformer model for machine translation and encounters several issues. Review their code and training logs, then identify and fix the problems.

**Code with Issues:**

```python
import torch
import torch.nn as nn
from torch.optim import Adam

class TransformerMT(nn.Module):
    def __init__(self, vocab_size=30000, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # ISSUE 1: Something missing here
        self.transformer = nn.Transformer(d_model, nhead, num_layers, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        # ISSUE 2: Embedding scaling
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)

        # ISSUE 3: Missing something in transformer call
        output = self.transformer(src_emb, tgt_emb)
        return self.fc_out(output)

# Training setup
model = TransformerMT()
optimizer = Adam(model.parameters(), lr=0.001)  # ISSUE 4: Learning rate
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(100):
    for src, tgt in dataloader:
        optimizer.zero_grad()

        # ISSUE 5: Target handling
        output = model(src, tgt)
        loss = criterion(output.view(-1, 30000), tgt.view(-1))

        loss.backward()
        optimizer.step()  # ISSUE 6: Missing gradient handling

    print(f"Epoch {epoch}: Loss = {loss.item()}")
```

**Training Log:**
```
Epoch 0: Loss = 10.31
Epoch 1: Loss = 10.29
Epoch 2: Loss = 10.28
Epoch 3: Loss = nan
Epoch 4: Loss = nan
...
```

**Tasks:**
1. Identify all 6 issues in the code
2. Explain why each issue causes problems
3. Provide the corrected code
4. Explain why the loss becomes NaN at epoch 3

### Solution

**Issue 1: Missing Positional Encoding**

```python
# PROBLEM: Transformer has no notion of position without explicit encoding
# The self-attention is permutation invariant!

# FIX: Add positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Add to model
self.pos_encoder = PositionalEncoding(d_model)
```

**Issue 2: Missing Embedding Scaling**

```python
# PROBLEM: Embeddings have variance that doesn't match positional encoding scale
# Without scaling, positional info gets overwhelmed

# FIX: Scale embeddings by sqrt(d_model)
src_emb = self.embedding(src) * math.sqrt(self.d_model)
tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
```

**Issue 3: Missing Masks**

```python
# PROBLEM:
# - Decoder can see future tokens (no causal mask)
# - Padding tokens affect attention (no padding mask)

# FIX: Add proper masks
def generate_square_subsequent_mask(sz):
    """Causal mask for decoder"""
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

def create_padding_mask(seq, pad_idx=0):
    """Mask padding tokens"""
    return (seq == pad_idx)

# In forward:
tgt_mask = generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
src_key_padding_mask = create_padding_mask(src)
tgt_key_padding_mask = create_padding_mask(tgt)

output = self.transformer(
    src_emb, tgt_emb,
    tgt_mask=tgt_mask,
    src_key_padding_mask=src_key_padding_mask,
    tgt_key_padding_mask=tgt_key_padding_mask
)
```

**Issue 4: Learning Rate Too High**

```python
# PROBLEM: lr=0.001 is way too high for Transformers
# Causes unstable training and NaN losses

# FIX: Use warmup schedule (as in "Attention is All You Need")
class TransformerScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5)
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

# Setup
optimizer = Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
scheduler = TransformerScheduler(optimizer, d_model=512)
```

**Issue 5: Target Shifting (Teacher Forcing)**

```python
# PROBLEM: Feeding full target sequence including last token
# Model should predict token[i+1] given token[0:i]

# FIX: Shift target for input vs. labels
# Input to decoder: <sos> token1 token2 ... token_n-1
# Labels: token1 token2 ... token_n <eos>

tgt_input = tgt[:, :-1]  # Everything except last token
tgt_labels = tgt[:, 1:]  # Everything except first token (<sos>)

output = model(src, tgt_input)
loss = criterion(output.view(-1, vocab_size), tgt_labels.view(-1))
```

**Issue 6: Missing Gradient Clipping**

```python
# PROBLEM: Exploding gradients cause NaN loss
# Transformer attention scores can produce very large gradients

# FIX: Clip gradients before optimizer step
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
scheduler.step()  # Also update learning rate
```

**Why NaN at Epoch 3:**

The loss becomes NaN due to a cascade of issues:

```
High Learning Rate (0.001)
        ↓
Large Parameter Updates
        ↓
Attention Scores Explode (no scaling, no masks)
        ↓
Softmax Saturation (exp(very_large) → inf)
        ↓
NaN in Attention Weights
        ↓
NaN Propagates Through Network
        ↓
Loss = NaN
```

**Complete Corrected Code:**

```python
import torch
import torch.nn as nn
import math
from torch.optim import Adam

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerMT(nn.Module):
    def __init__(self, vocab_size=30000, d_model=512, nhead=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)  # FIX 1
        self.transformer = nn.Transformer(
            d_model, nhead, num_layers, num_layers,
            dim_feedforward=2048, dropout=dropout
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None):
        # FIX 2: Scale embeddings
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)

        # Add positional encoding
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_encoder(tgt_emb)

        # Transpose for PyTorch Transformer (seq_len, batch, d_model)
        src_emb = src_emb.transpose(0, 1)
        tgt_emb = tgt_emb.transpose(0, 1)

        # FIX 3: Pass masks
        output = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        output = output.transpose(0, 1)
        return self.fc_out(output)

def generate_square_subsequent_mask(sz, device):
    mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

# Training setup
model = TransformerMT()
# FIX 4: Proper optimizer with warmup
optimizer = Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
scheduler = TransformerScheduler(optimizer, d_model=512, warmup_steps=4000)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding

# Training loop
for epoch in range(100):
    model.train()
    total_loss = 0

    for src, tgt in dataloader:
        optimizer.zero_grad()

        # FIX 5: Shift targets
        tgt_input = tgt[:, :-1]
        tgt_labels = tgt[:, 1:]

        # Generate masks
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(1), tgt_input.device)
        src_padding_mask = (src == 0)
        tgt_padding_mask = (tgt_input == 0)

        output = model(
            src, tgt_input,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )

        loss = criterion(output.reshape(-1, 30000), tgt_labels.reshape(-1))
        loss.backward()

        # FIX 6: Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, LR = {scheduler.get_lr():.6f}")
```

### Key Takeaways

1. **Positional encoding is mandatory:** Without it, Transformer cannot distinguish word order
2. **Embedding scaling matters:** Balances embedding magnitude with positional encoding
3. **Masks are essential:** Causal mask prevents future leakage; padding mask prevents attending to padding
4. **Warmup scheduling:** Transformers need careful learning rate warmup
5. **Gradient clipping:** Prevents exploding gradients in attention layers
6. **Teacher forcing shift:** Input and labels must be offset by one position

---

## Problem Summary

| Problem | Type | Concepts | Difficulty | Key Learning |
|---------|------|----------|------------|--------------|
| P1 | Warm-Up | Tokenization, Preprocessing | ⭐ | Preprocessing tradeoffs |
| P2 | Skill-Builder | Self-Attention | ⭐⭐⭐ | QKV computation |
| P3 | Skill-Builder | BERT Fine-tuning | ⭐⭐⭐ | Overfitting diagnosis |
| P4 | Challenge | NER System Design | ⭐⭐⭐⭐ | End-to-end pipeline |
| P5 | Debug/Fix | Transformer Training | ⭐⭐⭐ | Common training issues |

---

## Cross-References

| Problem | Study Notes Section | Concept Map Node | Flashcard |
|---------|---------------------|------------------|-----------|
| P1 | Concept 1: Tokenization | Tokenization (6) | Card 1 |
| P2 | Concept 7: Attention | Attention (12) | Card 2 |
| P3 | Concept 9: Pre-trained Models | BERT (7), Fine-tuning (7) | Card 4 |
| P4 | Concepts 8-10 | Full pipeline | Card 5 |
| P5 | Concept 8: Transformer | Transformer (10) | Card 3 |
