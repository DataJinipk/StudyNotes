# Assessment Quiz: Neural Networks

**Source Material:** notes/neural-networks/flashcards/neural-networks-flashcards.md
**Practice Problems:** notes/neural-networks/practice/neural-networks-practice-problems.md
**Concept Map:** notes/neural-networks/concept-maps/neural-networks-concept-map.md
**Original Study Notes:** notes/neural-networks/neural-networks-study-notes.md
**Date Generated:** 2026-01-06
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
**Concept Tested:** Activation Functions
**Source Section:** Core Concepts 3
**Concept Map Node:** Activation Function (5 connections)
**Related Flashcard:** Card 1
**Related Practice Problem:** P3

Why are activation functions essential in neural networks, and what would happen without them?

A) Activation functions speed up training; without them, training would be slower but produce the same results

B) Activation functions introduce non-linearity; without them, any deep network would collapse to a single linear transformation regardless of depth

C) Activation functions prevent overfitting; without them, the network would memorize training data

D) Activation functions normalize outputs; without them, values would grow unboundedly through layers

---

### Question 2 | Multiple Choice
**Cognitive Level:** Remember/Understand
**Concept Tested:** Training Loop and Backpropagation
**Source Section:** Core Concepts 4, 5, 6, 7
**Concept Map Node:** Backpropagation (Central - 8 connections)
**Related Flashcard:** Card 2
**Related Practice Problem:** P2

During backpropagation, what mathematical technique is used to compute gradients, and in what direction do gradients flow through the network?

A) Matrix inversion; gradients flow forward from input to output

B) Chain rule of calculus; gradients flow backward from output to input

C) Gradient approximation; gradients flow in both directions simultaneously

D) Eigenvalue decomposition; gradients flow through skip connections only

---

### Question 3 | Short Answer
**Cognitive Level:** Apply/Analyze
**Concept Tested:** CNN vs Dense Networks
**Source Section:** Core Concepts 8
**Concept Map Node:** CNN (5 connections)
**Related Flashcard:** Card 3
**Related Practice Problem:** P4
**Expected Response Length:** 3-4 sentences

A colleague suggests using a fully-connected (dense) neural network for classifying 256×256 RGB images. Explain why this is problematic and how CNNs address these specific issues. Include at least two concrete advantages of CNNs for image data.

---

### Question 4 | Short Answer
**Cognitive Level:** Apply/Analyze
**Concept Tested:** Training Diagnosis
**Source Section:** Critical Analysis (Limitations)
**Concept Map Node:** Training Process cluster
**Related Flashcard:** Card 5
**Related Practice Problem:** P5
**Expected Response Length:** 3-4 sentences

You observe the following during training: training accuracy is 98%, but validation accuracy is only 62%. The validation loss has been increasing for the last 10 epochs while training loss continues to decrease. Diagnose this situation and explain the underlying mechanism causing this behavior. Propose one specific technique to address it.

---

### Question 5 | Essay
**Cognitive Level:** Evaluate/Synthesize
**Concepts Tested:** Architecture Selection, Full Pipeline
**Source Sections:** All Core Concepts, Practical Applications
**Concept Map:** Full pathway traversal
**Related Flashcard:** Card 4, Card 5
**Related Practice Problem:** P4, P5
**Expected Response Length:** 1-2 paragraphs

You are designing a neural network system to analyze customer support chat transcripts. The system must: (1) classify the primary issue category (billing, technical, account, other), and (2) detect the customer's sentiment (positive, neutral, negative).

Design a complete solution addressing: (a) architecture selection with justification comparing RNN/LSTM vs. Transformer approaches, (b) how you would handle the dual-output requirement (category + sentiment), (c) appropriate loss functions and output activations for each task, (d) a potential challenge specific to text data and how you would address it, and (e) how you would detect if your model is overfitting during training.

**Evaluation Criteria:**
- [ ] Selects and justifies architecture choice
- [ ] Correctly designs dual-output structure
- [ ] Specifies appropriate loss/activation per task
- [ ] Identifies text-specific challenge with solution
- [ ] Describes overfitting detection strategy

---

## Answer Key

### Question 1 | Multiple Choice
**Correct Answer:** B

**Explanation:**
Activation functions introduce non-linearity into the network. Without them, each layer computes a linear transformation: `y = Wx + b`. Stacking multiple linear layers results in another linear transformation: `W2(W1x + b1) + b2 = (W2W1)x + (W2b1 + b2) = W'x + b'`. No matter how many layers, the result is always a single linear function. Non-linear activations break this collapse, enabling networks to approximate complex, non-linear functions.

**Why Other Options Are Incorrect:**
- A) Activations don't primarily affect speed; they affect what functions can be learned
- C) Activations don't prevent overfitting; regularization techniques do
- D) While activations bound outputs, normalization is a separate concern; the core purpose is non-linearity

**Understanding Gap Indicator:**
If answered incorrectly, review the mathematical proof that stacked linear layers collapse. Practice Problem 3 also covers activation selection.

---

### Question 2 | Multiple Choice
**Correct Answer:** B

**Explanation:**
Backpropagation uses the **chain rule** of calculus to decompose the gradient of the loss with respect to each weight into a product of local gradients. Gradients flow **backward** from the output layer toward the input layer: first computing ∂Loss/∂output_weights, then using those gradients to compute ∂Loss/∂hidden_weights, and so on. This backward flow is why it's called "back"-propagation.

**Why Other Options Are Incorrect:**
- A) Matrix inversion is computationally expensive and not used; gradients flow backward, not forward
- C) Gradients are computed exactly via chain rule, not approximated; flow is strictly backward
- D) Eigenvalue decomposition is unrelated; gradients flow through all connections, not just skip connections

**Understanding Gap Indicator:**
If answered incorrectly, review the chain rule derivation in Practice Problem 2. Understanding the backward flow is fundamental to debugging training issues.

---

### Question 3 | Short Answer
**Model Answer:**

Using a dense network for 256×256 RGB images is problematic for two main reasons. First, **parameter explosion**: the input would be 256×256×3 = 196,608 features; a hidden layer of just 1,000 neurons would require ~197 million parameters, making training slow, memory-intensive, and prone to overfitting. Second, **loss of spatial structure**: flattening the image destroys the 2D spatial relationships; pixels that are neighbors (and thus related) become arbitrary positions in a 1D vector.

CNNs address these issues through: (1) **parameter sharing**—the same small filter (e.g., 3×3) is applied across all positions, reducing parameters from millions to thousands; (2) **local connectivity**—each neuron connects only to a local patch, preserving spatial relationships; and (3) **translation invariance**—features are detected regardless of position, so a cat in the corner is recognized the same as a cat in the center.

**Key Components Required:**
- [ ] Identifies parameter explosion problem with numbers
- [ ] Identifies spatial structure loss problem
- [ ] Explains at least two CNN advantages (parameter sharing, local connectivity, or translation invariance)

**Partial Credit Guidance:**
- Full credit: Both problems + two CNN advantages with clear explanation
- Partial credit: One problem + one advantage, or both problems without clear CNN explanation
- No credit: Claims dense networks work fine, or incorrect CNN explanation

**Understanding Gap Indicator:**
If answered poorly, review CNN fundamentals and Practice Problem 4 on CNN architecture design.

---

### Question 4 | Short Answer
**Model Answer:**

This is a clear case of **overfitting**. The model has memorized the training data (98% accuracy) rather than learning generalizable patterns, causing it to fail on unseen validation data (62% accuracy). The mechanism is that the model has enough capacity to fit even the noise in training data; it learns training-specific patterns that don't transfer. The increasing validation loss while training loss decreases is the definitive signal—the model is getting worse at generalizing while getting better at memorization.

To address this, I would implement **early stopping**: monitor validation loss and stop training when it stops improving (or starts increasing), then restore the model weights from the best validation epoch. This prevents the model from continuing to memorize after it has learned the generalizable patterns. Alternative techniques include dropout, L2 regularization, or data augmentation.

**Key Components Required:**
- [ ] Correctly diagnoses overfitting
- [ ] Explains mechanism (memorization vs. generalization)
- [ ] Proposes specific technique with explanation of how it helps

**Partial Credit Guidance:**
- Full credit: Correct diagnosis + mechanism explanation + specific technique with rationale
- Partial credit: Correct diagnosis but weak mechanism explanation or vague solution
- No credit: Incorrect diagnosis (e.g., underfitting) or no technique proposed

**Understanding Gap Indicator:**
If answered poorly, review Practice Problem 5 (Debug/Fix) which covers this exact scenario in depth.

---

### Question 5 | Essay
**Model Answer:**

**Architecture Selection:**
For analyzing chat transcripts, I would choose a **Transformer-based architecture** (such as fine-tuning DistilBERT or a small BERT variant) over RNN/LSTM. Transformers offer several advantages for this task: (1) **parallel processing** enables faster training on modern GPUs; (2) **self-attention** directly captures relationships between any words regardless of distance, important for understanding context like "I loved everything EXCEPT the billing process"; (3) **pre-trained models** have already learned language structure from massive corpora, critical when customer support data may be limited. RNN/LSTM would be viable for resource-constrained environments but would struggle with long transcripts due to vanishing gradients and sequential processing bottlenecks.

**Dual-Output Architecture:**
The model would have a shared encoder (Transformer layers processing the transcript) with two separate output heads branching from the final representation: one head for **category classification** (4-class: Dense → Softmax) and one for **sentiment** (3-class: Dense → Softmax). Both heads use **Categorical Cross-Entropy loss**. Alternatively, sentiment could use 3 independent sigmoids if sentiments could co-occur, but typically they're mutually exclusive. The total loss is the sum (or weighted sum) of both task losses, enabling joint training with shared representations.

**Text-Specific Challenge:**
A key challenge is **class imbalance**—"billing" issues might be 60% of tickets while "account" issues are only 5%. This would cause the model to over-predict majority classes. Solutions include: class-weighted loss functions (penalize mistakes on rare classes more heavily), oversampling minority classes, or using focal loss which down-weights easy examples.

**Overfitting Detection:**
I would monitor training and validation loss/accuracy curves for each task. Overfitting signals include: (1) validation loss increasing while training loss decreases; (2) growing gap between train and validation accuracy; (3) validation metrics plateauing while training metrics continue improving. I would implement early stopping with patience of 3-5 epochs, saving the model checkpoint with best validation performance. Given limited customer support data, I would also use dropout (0.1-0.3) in the classification heads.

**Evaluation Rubric:**

| Criterion | Excellent (4) | Proficient (3) | Developing (2) | Beginning (1) |
|-----------|---------------|----------------|----------------|---------------|
| Architecture | Selects Transformer with specific advantages; acknowledges RNN tradeoffs | Selects appropriate architecture with some justification | Names architecture without clear justification | Wrong architecture or no justification |
| Dual-Output | Correct shared encoder + separate heads; correct activations both | Correct structure for one task, minor error on other | Partially correct structure | Fundamental misunderstanding |
| Loss/Activation | Correct for both (CCE + Softmax for multi-class) | Correct for one task | Partially correct | Wrong choices |
| Text Challenge | Identifies real challenge (imbalance, OOV, etc.) with specific solution | Identifies challenge with generic solution | Vague challenge identification | No challenge identified |
| Overfitting | Multiple specific detection methods + mitigation | Detection method described | Vague overfitting awareness | No overfitting consideration |

**Understanding Gap Indicator:**
If response lacks depth, this may indicate:
- Difficulty selecting architectures based on task requirements
- Weak understanding of multi-task learning structures
- Limited awareness of real-world text processing challenges

---

## Diagnostic Feedback & Review Recommendations

### Performance-Based Review Guide

| If Incorrect On... | Gap Identified | Review Section | Priority |
|--------------------|----------------|----------------|----------|
| Question 1 | Activation function purpose | Core Concepts 3 + Practice P3 | High |
| Question 2 | Backpropagation mechanism | Core Concepts 6 + Practice P2 | High |
| Question 3 | CNN advantages | Core Concepts 8 + Practice P4 | Medium |
| Question 4 | Overfitting diagnosis | Critical Analysis + Practice P5 | Medium |
| Question 5 | Full pipeline synthesis | All sections | Low |

### Targeted Review Recommendations

#### For Multiple Choice Errors (Questions 1-2)
**Indicates:** Foundational concept gaps
**Action:** Review:
- Study Notes: Core Concepts 3 (Activation) and 6 (Backpropagation)
- Flashcards: Cards 1 and 2
- Practice Problems: P2 (manual backprop) and P3 (activation selection)
**Focus On:** Understanding WHY these mechanisms exist, not just WHAT they are

#### For Short Answer Errors (Questions 3-4)
**Indicates:** Application difficulties
**Action:** Practice:
- Practice Problems: P4 (CNN design) and P5 (training diagnosis)
- Concept Map: Architecture and Training Process pathways
**Focus On:** Connecting symptoms to underlying mechanisms

#### For Essay Weakness (Question 5)
**Indicates:** Integration challenges
**Action:** Review interconnections:
- Concept Map: Full Critical Path traversal
- All Practice Problems for procedural fluency
- Study Notes: Practical Applications section
**Focus On:** Building mental models that connect architecture → training → evaluation

### Mastery Indicators

- **5/5 Correct:** Strong mastery; ready for implementation projects
- **4/5 Correct:** Good understanding; review indicated gap
- **3/5 Correct:** Moderate understanding; systematic review + more practice problems
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
    │ 20 concepts│ 5 cards   │ 5 problems│ 5 questions                     │
    │ 32 rels    │ 2E/2M/1H  │ 1W/2S/1C/1D│ 2MC/2SA/1E                     │
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
| concept-map | 20 nodes, 32 relationships, 4 pathways | Structure + learning paths |
| flashcards | 5 cards (2E/2M/1H) | Memorization + critical flags |
| practice-problems | 5 problems (1W/2S/1C/1D) | Procedural fluency + common mistakes |
| quiz | 5 questions (2MC/2SA/1E) | Assessment + diagnostic feedback |
