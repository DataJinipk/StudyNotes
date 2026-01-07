# Flashcard Set: Machine Learning

**Source:** notes/machine-learning/machine-learning-study-notes.md
**Concept Map Reference:** notes/machine-learning/concept-maps/machine-learning-concept-map.md
**Original Source Path:** C:\agentic_ai\StudyNotes\notes\machine-learning\machine-learning-study-notes.md
**Date Generated:** 2026-01-06
**Total Cards:** 5
**Distribution:** 2 Easy | 2 Medium | 1 Hard

---

## Critical Knowledge Summary

Concepts appearing in multiple cards (prioritize for review):
- **Bias-Variance Tradeoff**: Appears in Cards 2, 3, 5 (core theoretical concept)
- **Overfitting**: Appears in Cards 2, 3, 5 (primary failure mode)
- **Model Evaluation**: Appears in Cards 1, 4, 5 (central concept per concept map)
- **Feature Engineering**: Appears in Cards 4, 5 (high practical impact)

---

## Flashcards

---
### Card 1 | Easy
**Cognitive Level:** Remember
**Concept:** Supervised vs. Unsupervised Learning
**Source Section:** Core Concepts - Concepts 1 & 2
**Concept Map Centrality:** High (Supervised: 5 connections)

**FRONT (Question):**
What distinguishes supervised learning from unsupervised learning, and what are the primary task types for each?

**BACK (Answer):**
**Supervised Learning:** Learns from labeled input-output pairs; the algorithm knows the "correct answers" during training.
- **Classification:** Predicts discrete categories (e.g., spam/not spam)
- **Regression:** Predicts continuous values (e.g., house prices)

**Unsupervised Learning:** Discovers patterns in unlabeled data; no "correct answers" provided.
- **Clustering:** Groups similar data points (e.g., customer segments)
- **Dimensionality Reduction:** Compresses features while preserving information

**Key distinction:** Supervised requires labeled data (expensive to obtain); unsupervised works with raw, unlabeled data.

**Critical Knowledge Flag:** Yes - Foundational paradigm distinction

---

---
### Card 2 | Easy
**Cognitive Level:** Understand
**Concept:** Bias-Variance Tradeoff
**Source Section:** Core Concepts - Concept 3
**Concept Map Centrality:** High (6 connections)

**FRONT (Question):**
What is the bias-variance tradeoff, and how do overfitting and underfitting relate to it?

**BACK (Answer):**
The **bias-variance tradeoff** describes the fundamental tension in model complexity:

| Term | Meaning | Model State | Problem |
|------|---------|-------------|---------|
| **High Bias** | Overly simplistic assumptions | **Underfitting** | Misses real patterns |
| **High Variance** | Sensitive to training data noise | **Overfitting** | Memorizes noise |

**Total Error = Bias² + Variance + Irreducible Noise**

The goal is finding optimal complexity:
- Too simple → high bias → underfitting
- Too complex → high variance → overfitting
- Just right → balances both → good generalization

**Critical Knowledge Flag:** Yes - Core theoretical concept appearing in multiple contexts

---

---
### Card 3 | Medium
**Cognitive Level:** Apply
**Concept:** Regularization and Overfitting Prevention
**Source Section:** Core Concepts - Concept 3; Key Terminology
**Concept Map Centrality:** Regularization connects Bias-Variance to Overfitting (bridge concept)

**FRONT (Question):**
Your model achieves 99% accuracy on training data but only 72% on test data. Diagnose the problem and explain how regularization addresses it. What is the tradeoff of applying regularization?

**BACK (Answer):**
**Diagnosis:** The large gap between training (99%) and test (72%) accuracy indicates **overfitting**—the model has memorized training data noise rather than learning generalizable patterns.

**How Regularization Helps:**
Regularization constrains model complexity by adding a penalty term to the loss function:
- **L1 (Lasso):** Adds sum of absolute weights → encourages sparsity (some weights → 0)
- **L2 (Ridge):** Adds sum of squared weights → shrinks all weights toward zero
- **Dropout (Neural Networks):** Randomly drops neurons during training

**The Tradeoff:**
Regularization **reduces variance** (less overfitting) but **increases bias** (model becomes simpler). Too much regularization causes underfitting. The regularization strength hyperparameter (λ) must be tuned via cross-validation to find the optimal balance.

**Critical Knowledge Flag:** Yes - Connects Overfitting, Bias-Variance, and practical solutions

---

---
### Card 4 | Medium
**Cognitive Level:** Analyze
**Concept:** Feature Engineering vs. Deep Learning
**Source Section:** Core Concepts - Concepts 4 & 7
**Concept Map Centrality:** Feature Engineering (4 connections); Deep Learning automates FE

**FRONT (Question):**
Compare manual feature engineering with deep learning's representation learning. Under what conditions should you prefer each approach, and what are the resource tradeoffs?

**BACK (Answer):**
**Manual Feature Engineering:**
- Humans design features using domain expertise
- Selection, transformation, creation of variables
- *Example:* Creating "time since last purchase" from timestamp

**Deep Learning Representation Learning:**
- Network automatically learns hierarchical features
- Early layers: simple patterns → Later layers: complex abstractions
- *Example:* CNN learns edge detectors → shapes → objects

**When to Prefer Each:**

| Condition | Prefer Manual FE | Prefer Deep Learning |
|-----------|-----------------|---------------------|
| Data size | Small-medium datasets | Large datasets (millions+) |
| Domain expertise | Strong domain knowledge available | Patterns unknown/complex |
| Compute resources | Limited resources | GPU clusters available |
| Interpretability | Required for decisions | Less critical |
| Data type | Tabular, structured | Images, text, audio |

**Key Insight:** Deep learning doesn't eliminate FE—it automates the *learned* features but still benefits from thoughtful data preprocessing and augmentation.

**Critical Knowledge Flag:** Yes - Connects Feature Engineering to Deep Learning and Model Evaluation

---

---
### Card 5 | Hard
**Cognitive Level:** Synthesize
**Concept:** End-to-End ML Pipeline Design
**Source Section:** All Core Concepts; Practical Applications; Critical Analysis
**Concept Map Centrality:** Integrates all high-centrality concepts

**FRONT (Question):**
Synthesize a complete machine learning pipeline for a fraud detection system. Address: (1) why this is a challenging supervised learning problem, (2) how bias-variance tradeoff manifests, (3) appropriate evaluation metrics (not accuracy), (4) feature engineering considerations, and (5) strategies for production deployment. Justify each design decision.

**BACK (Answer):**
**1. Supervised Learning Challenge:**
Fraud detection is classification with severe **class imbalance** (~99% legitimate, ~1% fraud). The model must learn from very few positive examples while the cost of false negatives (missed fraud) far exceeds false positives (blocked legitimate transactions).

**2. Bias-Variance in Fraud Detection:**
- **High bias risk:** Simple models miss subtle fraud patterns that criminals specifically design to evade detection
- **High variance risk:** Complex models overfit to specific fraud patterns that quickly evolve
- **Solution:** Ensemble methods (Random Forest, XGBoost) balance both; regularly retrain as fraud patterns shift

**3. Evaluation Metrics (NOT Accuracy):**
Accuracy is misleading with 99% legitimate—a model predicting "always legitimate" achieves 99% accuracy but catches zero fraud.
- **Primary:** Precision-Recall AUC (focuses on positive class)
- **Business-aligned:** Cost-weighted F1 where FN cost >> FP cost
- **Threshold tuning:** Adjust classification threshold based on business tolerance

**4. Feature Engineering:**
- **Temporal:** Transaction frequency, time since last transaction, unusual hours
- **Behavioral:** Deviation from user's normal patterns, location changes
- **Network:** Connections to known fraud accounts, graph-based features
- **Velocity:** Rapid transactions, sudden spending spikes
- *Deep learning alternative:* Sequence models (LSTM) on transaction histories

**5. Production Deployment:**
- **Real-time inference:** Latency requirements (~100ms) constrain model complexity
- **Monitoring:** Track precision/recall drift; fraud patterns evolve
- **Feedback loop:** Human review of flagged transactions updates training data
- **A/B testing:** Compare model versions on live traffic subset
- **Fallback:** Rule-based system for when model confidence is low

**Synthesis:** The pipeline integrates supervised learning on imbalanced data, careful bias-variance management through ensembles, business-aligned evaluation beyond accuracy, domain-driven feature engineering, and production considerations for a continuously evolving adversarial problem.

**Critical Knowledge Flag:** Yes - Integrates Supervised Learning, Bias-Variance, Evaluation, Feature Engineering, and practical deployment

---

---

## Export Formats

### Anki-Compatible (Tab-Separated)
```
What distinguishes supervised from unsupervised learning?	Supervised: labeled pairs, classification/regression. Unsupervised: unlabeled, clustering/reduction. Key: supervised needs labeled data.	easy::paradigms::ml
What is the bias-variance tradeoff?	Tension between simplicity (high bias→underfit) and flexibility (high variance→overfit). Total error = bias² + variance + noise. Goal: optimal complexity.	easy::theory::ml
Diagnose: 99% train accuracy, 72% test. How does regularization help?	Overfitting (memorized noise). Regularization adds penalty constraining complexity. Tradeoff: reduces variance but increases bias. Tune λ via CV.	medium::regularization::ml
Compare manual feature engineering vs deep learning representation.	Manual: domain expertise designs features, small data. DL: auto-learns hierarchical features, large data + GPU needed. DL doesn't eliminate FE, automates learned part.	medium::features::ml
Design fraud detection pipeline addressing class imbalance, metrics, features.	Imbalanced classification; use precision-recall not accuracy; ensemble for bias-variance; temporal/behavioral/network features; real-time inference + drift monitoring.	hard::pipeline::ml
```

### CSV Format
```csv
"Front","Back","Difficulty","Concept","Centrality"
"Supervised vs Unsupervised Learning?","Supervised: labeled, classification/regression. Unsupervised: unlabeled, clustering/reduction.","Easy","Learning Paradigms","High"
"What is bias-variance tradeoff?","High bias=underfit, high variance=overfit. Total error = bias² + variance + noise.","Easy","Bias-Variance","High"
"99% train, 72% test - diagnose and fix","Overfitting. Regularization constrains complexity. Trades variance for bias.","Medium","Regularization","Medium"
"Manual FE vs Deep Learning features?","Manual: domain expertise, small data. DL: auto-learns, needs large data + compute.","Medium","Feature Engineering","High"
"Design fraud detection pipeline","Imbalanced classification, precision-recall metrics, ensemble methods, temporal features, drift monitoring","Hard","Pipeline Design","Critical"
```

---

## Source Mapping

| Card | Source Section | Concept Map Node | Key Terminology |
|------|----------------|------------------|-----------------|
| 1 | Core Concepts 1 & 2 | Supervised Learning, Unsupervised Learning | Classification, regression, clustering |
| 2 | Core Concepts 3 | Bias-Variance Tradeoff | Overfitting, underfitting, variance, bias |
| 3 | Core Concepts 3 + Terms | Regularization, Overfitting | L1, L2, dropout, cross-validation |
| 4 | Core Concepts 4 & 7 | Feature Engineering, Deep Learning | Representation learning, CNN, domain expertise |
| 5 | All + Applications | All high-centrality nodes | Pipeline, deployment, metrics, ensemble |
