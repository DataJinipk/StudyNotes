# Practice Problems: Lesson 9 - Natural Language Processing

**Source:** Lessons/Lesson_9.md
**Subject Area:** AI Learning - Natural Language Processing: From Text Representation to Neural Language Understanding
**Date Generated:** 2026-01-08
**Total Problems:** 5

---

## Problem Distribution

| Difficulty | Count | Type | Focus Area |
|------------|-------|------|------------|
| Warm-Up | 1 | Direct concept application | TF-IDF computation |
| Skill-Builder | 2 | Multi-step procedural | Attention weights, LSTM analysis |
| Challenge | 1 | Complex synthesis | NLP pipeline design |
| Debug/Fix | 1 | Identify and correct errors | Fine-tuning debugging |

---

## Problem 1: TF-IDF Computation (Warm-Up)

**Difficulty:** Warm-Up
**Estimated Time:** 15 minutes
**Concepts:** Sparse representations, TF-IDF weighting

### Problem Statement

Given the following corpus of 4 documents, compute the TF-IDF vectors for Document 1.

**Corpus:**
- Doc 1: "neural networks learn patterns from data"
- Doc 2: "deep neural networks are powerful"
- Doc 3: "machine learning uses data and patterns"
- Doc 4: "data science applies machine learning"

**Tasks:**
a) Compute the Term Frequency (TF) for each unique word in Document 1
b) Compute the Inverse Document Frequency (IDF) for each word (use log base 10)
c) Calculate the final TF-IDF score for each word
d) Which word has the highest TF-IDF score and why?

---

### Hints

<details>
<summary>Hint 1 (Conceptual)</summary>
TF(t, d) = count of term t in document d / total terms in document d
IDF(t) = log₁₀(N / df(t)), where N = total documents, df(t) = documents containing term t
</details>

<details>
<summary>Hint 2 (Process)</summary>
First identify unique words in Doc 1: "neural", "networks", "learn", "patterns", "from", "data"
Then count how many of the 4 documents contain each word.
</details>

<details>
<summary>Hint 3 (Calculation)</summary>
For "neural": appears in Doc 1, 2 → df = 2, IDF = log₁₀(4/2) = 0.301
For "data": appears in Doc 1, 3, 4 → df = 3, IDF = log₁₀(4/3) = 0.125
</details>

---

### Solution

**a) Term Frequency (TF) for Document 1:**

Document 1: "neural networks learn patterns from data" (6 words)

| Term | Count | TF = count/total |
|------|-------|------------------|
| neural | 1 | 1/6 = 0.167 |
| networks | 1 | 1/6 = 0.167 |
| learn | 1 | 1/6 = 0.167 |
| patterns | 1 | 1/6 = 0.167 |
| from | 1 | 1/6 = 0.167 |
| data | 1 | 1/6 = 0.167 |

**b) Inverse Document Frequency (IDF):**

| Term | Documents Containing | df | IDF = log₁₀(4/df) |
|------|---------------------|-----|-------------------|
| neural | Doc 1, Doc 2 | 2 | log₁₀(4/2) = 0.301 |
| networks | Doc 1, Doc 2 | 2 | log₁₀(4/2) = 0.301 |
| learn | Doc 1 only | 1 | log₁₀(4/1) = 0.602 |
| patterns | Doc 1, Doc 3 | 2 | log₁₀(4/2) = 0.301 |
| from | Doc 1 only | 1 | log₁₀(4/1) = 0.602 |
| data | Doc 1, Doc 3, Doc 4 | 3 | log₁₀(4/3) = 0.125 |

**c) TF-IDF Scores:**

| Term | TF | IDF | TF-IDF = TF × IDF |
|------|-----|-----|-------------------|
| neural | 0.167 | 0.301 | 0.050 |
| networks | 0.167 | 0.301 | 0.050 |
| learn | 0.167 | 0.602 | 0.100 |
| patterns | 0.167 | 0.301 | 0.050 |
| from | 0.167 | 0.602 | 0.100 |
| data | 0.167 | 0.125 | 0.021 |

**d) Highest TF-IDF Words: "learn" and "from" (both 0.100)**

**Explanation:** These words appear ONLY in Document 1 (df=1), making them highly distinctive for this document. The IDF formula gives maximum weight to terms that are unique to a document. In contrast, "data" appears in 3 of 4 documents, giving it a low IDF (0.125) and thus low TF-IDF despite having the same TF.

**Key Insight:** TF-IDF identifies words that are both frequent in a document AND rare across the corpus—these are the most discriminative features for document classification.

---

## Problem 2: Attention Weight Analysis (Skill-Builder)

**Difficulty:** Skill-Builder
**Estimated Time:** 25 minutes
**Concepts:** Attention mechanism, alignment in machine translation

### Problem Statement

You are analyzing a neural machine translation system translating from English to French. The encoder has processed the source sentence, and you have the following attention weights when the decoder generates each French word.

**Source (English):** "The black cat sat on the mat"
**Target (French):** "Le chat noir était assis sur le tapis"

**Attention Weights Matrix:**

|  | The | black | cat | sat | on | the | mat |
|--|-----|-------|-----|-----|-----|-----|-----|
| Le | 0.70 | 0.05 | 0.10 | 0.05 | 0.05 | 0.03 | 0.02 |
| chat | 0.10 | 0.15 | **0.60** | 0.05 | 0.03 | 0.04 | 0.03 |
| noir | 0.05 | **0.75** | 0.10 | 0.03 | 0.02 | 0.03 | 0.02 |
| était | 0.05 | 0.05 | 0.10 | **0.55** | 0.10 | 0.10 | 0.05 |
| assis | 0.03 | 0.02 | 0.05 | **0.70** | 0.10 | 0.05 | 0.05 |
| sur | 0.02 | 0.02 | 0.03 | 0.05 | **0.80** | 0.03 | 0.05 |
| le | 0.02 | 0.02 | 0.03 | 0.05 | 0.05 | **0.75** | 0.08 |
| tapis | 0.02 | 0.02 | 0.03 | 0.03 | 0.05 | 0.05 | **0.80** |

**Tasks:**
a) Identify the word alignments learned by the attention mechanism
b) Note any interesting linguistic phenomena captured by the attention
c) Calculate the context vector for generating "noir" given encoder hidden states h = [h₁, h₂, h₃, h₄, h₅, h₆, h₇] where each hᵢ is a 4-dimensional vector
d) Explain why "noir" (black) appears AFTER "chat" (cat) in French but BEFORE "cat" in English

---

### Hints

<details>
<summary>Hint 1 (Alignment)</summary>
Look for the highest attention weight in each row—this indicates which source word is most relevant when generating each target word.
</details>

<details>
<summary>Hint 2 (Linguistic)</summary>
French adjectives often follow nouns (adjective postposition), while English adjectives precede nouns. The attention mechanism must handle this word order difference.
</details>

<details>
<summary>Hint 3 (Context Vector)</summary>
Context vector c = Σᵢ αᵢ × hᵢ where αᵢ is the attention weight for source position i.
</details>

---

### Solution

**a) Word Alignments:**

| French Word | Aligned English Word | Attention Weight |
|-------------|---------------------|------------------|
| Le | The | 0.70 |
| chat | cat | 0.60 |
| noir | black | 0.75 |
| était | sat | 0.55 |
| assis | sat | 0.70 |
| sur | on | 0.80 |
| le | the | 0.75 |
| tapis | mat | 0.80 |

**b) Interesting Linguistic Phenomena:**

1. **Adjective Reordering:** "black cat" → "chat noir"
   - "chat" attends to "cat" (0.60)
   - "noir" attends to "black" (0.75)
   - The model correctly handles English adj-noun → French noun-adj reordering

2. **One-to-Many Alignment:** "sat" → "était assis"
   - Both "était" (0.55) and "assis" (0.70) attend primarily to "sat"
   - English simple past requires two French words (auxiliary + past participle)

3. **Article Alignment:** "The" → "Le", "the" → "le"
   - Each article aligns to its corresponding English article
   - Attention distinguishes position even for repeated words

4. **Preposition Alignment:** "on" → "sur" (0.80 attention)
   - Direct correspondence with high confidence

**c) Context Vector Calculation for "noir":**

Given attention weights for "noir": [0.05, 0.75, 0.10, 0.03, 0.02, 0.03, 0.02]

```
c_noir = 0.05×h₁ + 0.75×h₂ + 0.10×h₃ + 0.03×h₄ + 0.02×h₅ + 0.03×h₆ + 0.02×h₇

If h₁ = [1,0,0,0], h₂ = [0,1,0,0], h₃ = [0,0,1,0], etc.:

c_noir = 0.05×[1,0,0,0] + 0.75×[0,1,0,0] + 0.10×[0,0,1,0] + ...
       = [0.05, 0.75, 0.10, 0.03, ...]  (truncated)

The context vector is dominated by h₂ (the hidden state for "black")
```

**d) Word Order Difference Explanation:**

```
English: "The [black] [cat]"  (Adjective-Noun order)
French:  "Le [chat] [noir]"   (Noun-Adjective order)

Why attention handles this:
1. Attention has no inherent position constraint
2. When generating "noir", decoder can attend anywhere in source
3. The model learns that French adjectives align with English adjectives
   regardless of position
4. This is why attention revolutionized MT—it naturally handles
   non-monotonic alignments that were difficult for RNNs
```

**Key Insight:** The attention mechanism learns soft alignments that capture complex linguistic correspondences including word reordering, one-to-many mappings, and structural differences between languages—all without explicit linguistic rules.

---

## Problem 3: LSTM Gate Analysis (Skill-Builder)

**Difficulty:** Skill-Builder
**Estimated Time:** 25 minutes
**Concepts:** LSTM architecture, gate functions, long-term dependencies

### Problem Statement

You are analyzing an LSTM processing the sentence: "The movie was bad, but the ending was surprisingly good"

The LSTM should capture that the overall sentiment shifts from negative to positive. You observe the following gate activations at key positions:

**Position 4 ("bad"):**
- Forget gate (f): 0.9
- Input gate (i): 0.8
- Output gate (o): 0.7
- Candidate value (c̃): -0.6 (negative sentiment)

**Position 7 ("surprisingly"):**
- Forget gate (f): 0.3
- Input gate (i): 0.9
- Output gate (o): 0.5
- Candidate value (c̃): 0.2 (mild positive signal)

**Position 8 ("good"):**
- Forget gate (f): 0.4
- Input gate (i): 0.95
- Output gate (o): 0.9
- Candidate value (c̃): 0.8 (strong positive sentiment)

**Tasks:**
a) Trace the cell state evolution from position 4 to position 8 (assume c₃ = 0)
b) Explain why the forget gate is low at "surprisingly" and "good"
c) Why is the output gate high at "good" but low at "surprisingly"?
d) How would a vanilla RNN struggle with this sentence?

---

### Hints

<details>
<summary>Hint 1 (Cell State)</summary>
Cell state update: c_t = f_t × c_{t-1} + i_t × c̃_t
The forget gate controls how much old memory to keep; input gate controls how much new information to add.
</details>

<details>
<summary>Hint 2 (Forget Gate)</summary>
Low forget gate means "forget the previous context." At "surprisingly," the model needs to reset from the negative sentiment to prepare for new information.
</details>

<details>
<summary>Hint 3 (Output Gate)</summary>
Output gate controls how much of the cell state to expose. High output at important words ensures the final representation captures key sentiment.
</details>

---

### Solution

**a) Cell State Evolution:**

```
Position 4 ("bad"):
c₃ = 0 (given)
c₄ = f₄ × c₃ + i₄ × c̃₄
c₄ = 0.9 × 0 + 0.8 × (-0.6)
c₄ = 0 + (-0.48)
c₄ = -0.48  (negative sentiment stored)

Positions 5-6 (assume pass-through with f≈0.9, i≈0.2):
c₅ ≈ 0.9 × (-0.48) + 0.2 × 0 ≈ -0.43
c₆ ≈ 0.9 × (-0.43) + 0.2 × 0 ≈ -0.39

Position 7 ("surprisingly"):
c₇ = f₇ × c₆ + i₇ × c̃₇
c₇ = 0.3 × (-0.39) + 0.9 × 0.2
c₇ = -0.117 + 0.18
c₇ = 0.063  (sentiment shifts toward neutral/positive!)

Position 8 ("good"):
c₈ = f₈ × c₇ + i₈ × c̃₈
c₈ = 0.4 × 0.063 + 0.95 × 0.8
c₈ = 0.025 + 0.76
c₈ = 0.785  (strong positive sentiment!)
```

**Cell State Trajectory:**
```
Position:  4 (bad)  →  5-6  →  7 (surprisingly)  →  8 (good)
Cell:      -0.48   →  -0.39  →     0.063         →    0.785
Sentiment: Negative → Negative → Neutral          → Positive
```

**b) Why Forget Gate is Low at "surprisingly" and "good":**

```
At "surprisingly" (f = 0.3):
- The word signals a contradiction/reversal ("but...surprisingly")
- Low forget gate (0.3) means: "forget 70% of previous context"
- This clears the negative sentiment to make room for new information
- The model learned: contrast words → reset memory

At "good" (f = 0.4):
- Continues the sentiment shift
- Still low (0.4) to further diminish negative context
- Allows strong positive signal to dominate
```

**c) Output Gate Behavior:**

```
At "surprisingly" (o = 0.5, moderate):
- "Surprisingly" is a modifier, not the main sentiment
- Moderate output = "this information exists but isn't the final answer"
- Hidden state: h₇ = 0.5 × tanh(0.063) ≈ 0.03 (subtle signal)

At "good" (o = 0.9, high):
- "Good" is the key sentiment-bearing word
- High output = "expose this to downstream layers"
- Hidden state: h₈ = 0.9 × tanh(0.785) ≈ 0.59 (strong signal)
- This ensures the final sentiment representation is positive
```

**d) Vanilla RNN Limitations:**

```
Problem 1: Vanishing gradient across 8 positions
- Gradient at "bad" from loss at "good": ∂L/∂h₄ = (W_hh)⁴ × ...
- If ||W_hh|| < 1, gradient vanishes
- Model can't learn: "bad" → "good" relationship

Problem 2: No selective forgetting
- RNN: h_t = tanh(W_hh × h_{t-1} + W_xh × x_t)
- Information from "bad" persists and interferes
- No mechanism to "reset" when seeing "surprisingly"

Problem 3: Fixed transformation
- Same W_hh applied regardless of content
- Can't learn: "but" means "forget previous"
- vs. "and" means "continue previous"

Result: Vanilla RNN would likely output negative sentiment,
influenced by earlier "bad" which gradient couldn't unlearn.
```

**Key Insight:** LSTM gates learn contextual operations: the forget gate learns that contrast words signal memory reset, while the output gate learns to emphasize sentiment-bearing words for the final representation.

---

## Problem 4: NLP Pipeline Design (Challenge)

**Difficulty:** Challenge
**Estimated Time:** 40 minutes
**Concepts:** End-to-end NLP system, architecture selection, evaluation

### Problem Statement

You are designing an NLP system for a medical records company that must:

1. **Extract clinical entities** from doctor notes (medications, dosages, symptoms, diagnoses)
2. **Classify documents** into 15 medical specialties
3. **Identify temporal relationships** (when was medication started? symptom duration?)
4. **Flag urgent cases** requiring immediate attention
5. **Support both structured forms AND free-text narratives**

**Constraints:**
- Must achieve >95% precision on medication extraction (patient safety)
- Must handle medical abbreviations and misspellings
- Must process 50K documents/day
- Must provide explainability for clinical decisions
- Must work in a HIPAA-compliant offline environment (no cloud APIs)

**Design a complete system specifying:**
a) Text preprocessing and tokenization strategy
b) Model architecture for each task
c) How to handle the mixed structured/unstructured input
d) Training data requirements and annotation strategy
e) Evaluation metrics and safety guarantees

---

### Hints

<details>
<summary>Hint 1 (Tokenization)</summary>
Medical text requires domain-specific tokenization. Consider: BioWordPiece, medical abbreviation expansion, handling of dosage units (e.g., "500mg" should be "500 mg").
</details>

<details>
<summary>Hint 2 (Architecture)</summary>
Consider domain-specific pre-trained models like PubMedBERT, ClinicalBERT, or BioBERT. These are pre-trained on medical literature and clinical notes.
</details>

<details>
<summary>Hint 3 (Safety)</summary>
For patient safety, consider ensemble methods, confidence thresholds, and human-in-the-loop for low-confidence predictions.
</details>

---

### Solution

**a) Text Preprocessing and Tokenization:**

```yaml
preprocessing_pipeline:
  step_1_format_handling:
    - Detect document type (structured form vs free-text)
    - Extract text from forms, preserve field associations
    - Handle PDF/scanned images with OCR if needed

  step_2_medical_normalization:
    - Abbreviation expansion dictionary (500+ medical abbreviations)
      - "pt" → "patient", "hx" → "history", "dx" → "diagnosis"
    - Dosage normalization: "500mg" → "500 mg", "q4h" → "every 4 hours"
    - Misspelling correction: Medical spell-checker with drug name focus

  step_3_tokenization:
    model: BioWordPiece (from PubMedBERT)
    rationale: Pre-trained on 21M PubMed abstracts, knows medical vocabulary
    special_handling:
      - Preserve dosage units as single tokens
      - Handle hyphenated drug names (e.g., "co-amoxiclav")
      - Split CamelCase medical terms

  step_4_section_detection:
    - Identify note sections: "Chief Complaint", "Assessment", "Plan"
    - Enables section-specific processing
```

**b) Model Architecture:**

```yaml
shared_encoder:
  model: ClinicalBERT-base (110M params)
  source: Pre-trained on MIMIC-III clinical notes
  strategy: Fine-tune on company's historical data

task_1_entity_extraction:
  architecture: ClinicalBERT + BiLSTM + CRF
  entities:
    - MEDICATION (drug names)
    - DOSAGE (amounts, frequencies)
    - SYMPTOM (patient complaints)
    - DIAGNOSIS (ICD codes, disease names)
    - TEMPORAL (dates, durations)

  safety_measures:
    - Confidence threshold: 0.95 for auto-extraction
    - Below threshold → flag for human review
    - Ensemble: 3 models, majority vote required
    - Drug name verification against RxNorm database

task_2_document_classification:
  architecture: ClinicalBERT + classification head
  classes: 15 medical specialties
  approach: Hierarchical classification
    - Level 1: Medical vs Surgical vs Diagnostic
    - Level 2: Specific specialty
  output: Top-3 predictions with confidence scores

task_3_temporal_extraction:
  architecture: ClinicalBERT + relation classification head
  approach:
    - Extract temporal expressions (SUTime parser)
    - Classify relations: BEFORE, AFTER, DURING, OVERLAP
    - Link medications/symptoms to temporal expressions
  example: "Started amoxicillin [MEDICATION] on March 1 [TEMPORAL]"
           → Relation: (amoxicillin, START_DATE, March 1)

task_4_urgency_detection:
  architecture: Multi-task head on shared encoder
  features:
    - Critical keyword detection ("chest pain", "difficulty breathing")
    - Vital sign extraction and threshold checking
    - Symptom severity classification
  output: Urgency score [0-1], > 0.8 triggers alert
  explainability: Highlight phrases contributing to urgency score
```

**c) Handling Mixed Input:**

```yaml
structured_input_handling:
  form_fields:
    - Direct mapping: Field name → entity type
    - "Medication" field → MEDICATION entity (high confidence)
    - Validate against drug database

  strategy:
    - Process structured fields first (reliable)
    - Use as distant supervision for free-text
    - Cross-reference: If form says "aspirin" but note mentions different drug, flag inconsistency

free_text_handling:
  - Full NLP pipeline (tokenization → encoding → task heads)
  - Section-aware processing (Plan section more likely to have medications)

fusion:
  - Combine structured and extracted entities
  - Conflict resolution: Structured > High-confidence extracted > Low-confidence
  - Present unified view with provenance tracking
```

**d) Training Data and Annotation:**

```yaml
data_requirements:
  entity_extraction:
    - 10K annotated documents (diverse specialties)
    - Active learning: Start with 1K, expand based on model uncertainty
    - Annotation guide with medical SME validation

  classification:
    - 1K documents per specialty (15K total)
    - Can bootstrap from billing codes (ICD-10 → specialty mapping)

  temporal:
    - 5K documents with temporal annotations
    - Time expressions + relation labels

  urgency:
    - 3K documents with urgency labels
    - Calibrated against actual patient outcomes

annotation_strategy:
  annotators: 2 medical professionals per document
  adjudication: 3rd annotator for disagreements
  quality_metrics:
    - Inter-annotator agreement > 0.85 (Cohen's kappa)
    - Regular calibration sessions

  tools:
    - Prodigy or Label Studio with medical entity schemes
    - Pre-annotation with model suggestions (speeds annotation 3x)
```

**e) Evaluation and Safety:**

```yaml
evaluation_metrics:
  entity_extraction:
    - Precision (primary): > 95% for medications (safety requirement)
    - Recall: > 85% (catch most entities)
    - F1 by entity type
    - Exact match AND partial match scoring

  classification:
    - Macro F1: > 90%
    - Top-3 accuracy: > 98%
    - Confusion matrix analysis per specialty

  temporal:
    - Relation F1: > 80%
    - Temporal expression detection: > 90%

  urgency:
    - Sensitivity: > 99% (never miss urgent case)
    - Specificity: > 80% (minimize false alarms)
    - Calibration: Predicted probabilities match actual rates

safety_guarantees:
  medication_extraction:
    - Dual-model verification (two independent models must agree)
    - Drug-drug interaction checking against extracted medications
    - Dosage range validation (flag impossible values)

  confidence_thresholds:
    - Auto-accept: Confidence > 0.95 AND ensemble agreement
    - Human review: Confidence 0.7-0.95 OR model disagreement
    - Reject/re-process: Confidence < 0.7

  audit_trail:
    - Log all extractions with model version, confidence, input hash
    - Enable retroactive analysis if issues discovered

  monitoring:
    - Daily accuracy sampling (50 documents manually verified)
    - Drift detection on input distribution
    - Alert on significant accuracy drops

production_architecture:
  throughput: 50K docs/day = ~0.6 docs/second
  hardware: 2x T4 GPUs (handles 5 docs/second with batching)
  latency: < 2 seconds per document
  deployment: On-premise Kubernetes (HIPAA compliance)
  failover: Active-passive with automatic switchover
```

**Key Insight:** Medical NLP requires layered safety: domain-specific pre-training, confidence thresholds, ensemble methods, and human-in-the-loop for edge cases. The cost of false positives/negatives in healthcare justifies this additional complexity.

---

## Problem 5: Fine-tuning Debugging (Debug/Fix)

**Difficulty:** Debug/Fix
**Estimated Time:** 20 minutes
**Concepts:** BERT fine-tuning, common issues, debugging strategies

### Problem Statement

A colleague is fine-tuning BERT for sentiment classification but getting poor results. Review their code and training logs to identify and fix the issues.

**Code:**
```python
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Freeze all BERT layers (only train classifier)
for param in model.bert.parameters():
    param.requires_grad = False

# Tokenize data
def tokenize_function(examples):
    return tokenizer(examples['text'], padding=True, truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=30,
    per_device_train_batch_size=4,
    learning_rate=1e-3,
    warmup_steps=0,
    weight_decay=0.0,
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()
```

**Training Logs:**
```
Epoch 1/30: loss = 1.09, accuracy = 0.35
Epoch 5/30: loss = 1.05, accuracy = 0.37
Epoch 10/30: loss = 1.02, accuracy = 0.38
Epoch 15/30: loss = 0.99, accuracy = 0.39
Epoch 30/30: loss = 0.95, accuracy = 0.41
```

**Tasks:**
a) Identify at least 5 issues in the code
b) Explain why each issue causes poor performance
c) Provide corrected code
d) What accuracy would you expect after fixes?

---

### Hints

<details>
<summary>Hint 1 (Frozen Layers)</summary>
Freezing all BERT layers is usually only done for feature extraction with very limited data. For fine-tuning, you want the encoder to adapt to your domain.
</details>

<details>
<summary>Hint 2 (Learning Rate)</summary>
1e-3 is way too high for BERT fine-tuning. Pre-trained models need gentle updates—typical range is 1e-5 to 5e-5.
</details>

<details>
<summary>Hint 3 (Training Dynamics)</summary>
The loss barely decreasing over 30 epochs suggests the model isn't learning effectively. Check learning rate, frozen parameters, and batch size.
</details>

---

### Solution

**a) Issues Identified:**

| Issue | Location | Severity |
|-------|----------|----------|
| 1. All BERT layers frozen | `param.requires_grad = False` | Critical |
| 2. Learning rate too high | `learning_rate=1e-3` | Critical |
| 3. No warmup | `warmup_steps=0` | Medium |
| 4. Batch size too small | `per_device_train_batch_size=4` | Medium |
| 5. Too many epochs | `num_train_epochs=30` | Low |
| 6. No weight decay | `weight_decay=0.0` | Low |
| 7. No evaluation set | Missing `eval_dataset` | Medium |
| 8. Padding not deferred | `padding=True` in tokenize | Low |

**b) Why Each Issue Causes Poor Performance:**

**Issue 1: Frozen BERT Layers (Critical)**
```
Problem: Only the classification head (768 → 3) is trainable
         = 768 × 3 + 3 = 2,307 parameters out of 110M

Impact: Cannot adapt BERT's rich representations to sentiment task
        Random classifier head on fixed features → limited accuracy

Why loss barely decreases: Only 2K params can change; model is essentially
doing linear probing with random head initialization
```

**Issue 2: Learning Rate 1e-3 (Critical)**
```
Problem: Standard LR for training from scratch, not fine-tuning
         Pre-trained weights are already near good solution

Impact: Large updates destroy pre-trained knowledge ("catastrophic forgetting")
        Even with frozen BERT, 1e-3 is too high for classifier head

Correct range: 1e-5 to 5e-5 for fine-tuning
```

**Issue 3: No Warmup (Medium)**
```
Problem: Large initial learning rate on randomly initialized classifier

Impact: Classifier head starts random; needs gentle initial updates
        Without warmup, early gradients may be unstable

Fix: warmup_ratio=0.1 (10% of training as warmup)
```

**Issue 4: Batch Size 4 (Medium)**
```
Problem: Very small batches → noisy gradient estimates

Impact: High variance in updates, unstable training
        BertAdam paper recommends batch sizes of 16-32

Fix: Increase to 16-32, or use gradient accumulation
     gradient_accumulation_steps=4 with batch_size=4 = effective 16
```

**Issue 5: 30 Epochs (Low)**
```
Problem: BERT fine-tuning typically needs only 2-4 epochs

Impact: With proper settings, 30 epochs would overfit
        With current settings, model isn't learning anyway

Note: The flat accuracy curve suggests underfitting, not overfitting
```

**Issue 6: No Weight Decay (Low)**
```
Problem: No regularization on classifier weights

Impact: Minor; classifier head is small
        More important if unfreezing BERT layers

Fix: weight_decay=0.01 is standard
```

**c) Corrected Code:**

```python
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# DON'T freeze BERT layers - allow fine-tuning
# (Remove the freezing loop entirely)

# Tokenize with deferred padding for efficiency
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',  # or use DataCollatorWithPadding
        truncation=True,
        max_length=128  # Explicit max length
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)  # Add evaluation

# Corrected training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,                    # Reduced from 30
    per_device_train_batch_size=16,        # Increased from 4
    per_device_eval_batch_size=32,
    learning_rate=2e-5,                    # Reduced from 1e-3
    warmup_ratio=0.1,                      # Added warmup
    weight_decay=0.01,                     # Added regularization
    evaluation_strategy='epoch',           # Added evaluation
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
)

# Add compute_metrics for evaluation
from sklearn.metrics import accuracy_score
import numpy as np

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': accuracy_score(labels, predictions)}

# Train with evaluation
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,            # Added
    compute_metrics=compute_metrics,       # Added
)
trainer.train()
```

**d) Expected Accuracy After Fixes:**

```
Original: 41% accuracy (barely above random for 3 classes = 33%)

After fixes:
- Sentiment classification on typical datasets: 85-92% accuracy
- With BERT fine-tuning: Expect 88-91% on balanced 3-class sentiment

Training curve should look like:
Epoch 1: loss = 0.8, accuracy = 75%
Epoch 2: loss = 0.4, accuracy = 85%
Epoch 3: loss = 0.2, accuracy = 89%

Key difference: Loss should decrease rapidly in first epoch
(pre-trained representations adapt quickly)
```

**Summary of Critical Fixes:**

| Parameter | Original | Fixed | Impact |
|-----------|----------|-------|--------|
| Frozen layers | All BERT frozen | None frozen | +40% accuracy |
| Learning rate | 1e-3 | 2e-5 | +20% accuracy |
| Warmup | None | 10% | +5% accuracy |
| Batch size | 4 | 16 | +3% accuracy |
| Epochs | 30 | 3 | Prevents overfit |

---

## Common Mistakes Summary

| Mistake | Why It's Wrong | Prevention |
|---------|---------------|------------|
| High learning rate for fine-tuning | Destroys pre-trained knowledge | Always use 1e-5 to 5e-5 |
| Freezing pre-trained layers | Prevents adaptation | Only freeze for feature extraction |
| Small batch sizes | Noisy gradients | Use 16-32 or gradient accumulation |
| No warmup | Unstable initial training | Use warmup_ratio=0.1 |
| Too many epochs | Overfitting | BERT needs only 2-4 epochs |
| No evaluation set | Can't monitor overfitting | Always include validation |

---

## Self-Assessment Guide

| Score | Mastery Level | Recommended Action |
|-------|---------------|-------------------|
| 5/5 problems | **Mastery** | Ready for production NLP work |
| 4/5 problems | **Proficient** | Review specific gaps |
| 3/5 problems | **Developing** | Re-study core concepts |
| 1-2/5 problems | **Foundational** | Complete Lesson 9 review |

---

*Generated from Lesson 9: Natural Language Processing | Practice Problems Skill*
