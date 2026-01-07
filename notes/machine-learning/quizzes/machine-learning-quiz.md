# Assessment Quiz: Machine Learning

**Source Material:** notes/machine-learning/flashcards/machine-learning-flashcards.md
**Concept Map Reference:** notes/machine-learning/concept-maps/machine-learning-concept-map.md
**Original Study Notes:** notes/machine-learning/machine-learning-study-notes.md
**Date Generated:** 2026-01-06
**Total Questions:** 5
**Estimated Completion Time:** 25-35 minutes
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
**Concept Tested:** Learning Paradigms
**Source Section:** Core Concepts - Concepts 1 & 2
**Concept Map Node:** Supervised Learning (High Centrality)
**Related Flashcard:** Card 1

A data scientist has a dataset of customer purchase histories but no labels indicating customer segments. They want to discover natural groupings in the data. Which approach should they use?

A) Supervised learning with regression, since they are predicting continuous customer values

B) Supervised learning with classification, since they want to assign customers to categories

C) Unsupervised learning with clustering, since they have no labels and want to discover structure

D) Reinforcement learning, since the algorithm must learn through trial and error

---

### Question 2 | Multiple Choice
**Cognitive Level:** Remember/Understand
**Concept Tested:** Bias-Variance Tradeoff
**Source Section:** Core Concepts - Concept 3
**Concept Map Node:** Bias-Variance Tradeoff (High Centrality - 6 connections)
**Related Flashcard:** Card 2

A model performs well on training data but poorly on test data. This is an example of:

A) Underfitting, caused by high bias—the model is too simple to capture the underlying patterns

B) Overfitting, caused by high variance—the model has memorized training noise rather than learning generalizable patterns

C) Irreducible error, caused by noise inherent in the data that no model can eliminate

D) Data leakage, caused by test data information inadvertently influencing training

---

### Question 3 | Short Answer
**Cognitive Level:** Apply/Analyze
**Concept Tested:** Model Evaluation and Metrics
**Source Section:** Core Concepts - Concept 5
**Concept Map Node:** Model Evaluation (Central - 7 connections)
**Related Flashcard:** Cards 1, 5
**Expected Response Length:** 2-4 sentences

A hospital is building a model to predict whether patients have a rare disease (affecting 1% of patients). The model achieves 99% accuracy. Explain why accuracy is misleading here and propose two alternative metrics that would better evaluate this model's performance. Justify your metric choices.

---

### Question 4 | Short Answer
**Cognitive Level:** Apply/Analyze
**Concept Tested:** Gradient Descent and Optimization
**Source Section:** Core Concepts - Concept 6
**Concept Map Node:** Gradient Descent (High Centrality - 5 connections)
**Related Flashcard:** Implicit in Card 5 (training mechanism)
**Expected Response Length:** 2-4 sentences

During neural network training, you observe that the loss oscillates wildly without decreasing. Later, with different settings, the loss decreases extremely slowly. Diagnose the likely cause of each behavior and explain what hyperparameter adjustment would address both issues.

---

### Question 5 | Essay
**Cognitive Level:** Evaluate/Synthesize
**Concepts Tested:** All core concepts integrated
**Source Sections:** All Core Concepts, Practical Applications, Critical Analysis
**Concept Map:** Full pathway traversal (Critical Path)
**Related Flashcard:** Card 5
**Expected Response Length:** 1-2 paragraphs

You are leading the ML team at an e-commerce company tasked with building a product recommendation system. The system must suggest products to users based on their browsing and purchase history.

Design a comprehensive ML approach addressing: (1) whether this is fundamentally a supervised or unsupervised problem (or both), with justification; (2) how you would handle the bias-variance tradeoff given that user preferences evolve over time; (3) your feature engineering strategy for user behavior data; (4) appropriate evaluation metrics beyond simple accuracy; and (5) key deployment considerations for a production recommendation system serving millions of users.

**Evaluation Criteria:**
- [ ] Correctly classifies the learning paradigm(s) with justification
- [ ] Addresses bias-variance in context of evolving preferences
- [ ] Proposes specific, actionable feature engineering strategies
- [ ] Selects appropriate metrics for recommendation quality
- [ ] Considers production deployment challenges

---

## Answer Key

### Question 1 | Multiple Choice
**Correct Answer:** C

**Explanation:**
This scenario describes a classic unsupervised learning problem: the data scientist has data but no labels, and wants to discover inherent structure (natural customer groupings). Clustering algorithms like K-means, hierarchical clustering, or DBSCAN can identify these groupings based on feature similarity without requiring labeled examples.

**Why Other Options Are Incorrect:**
- A) Regression requires labeled continuous outputs; here there are no labels at all
- B) Classification requires labeled categorical outputs; again, no labels exist
- D) Reinforcement learning involves an agent learning through environmental rewards; this is a static data analysis problem

**Understanding Gap Indicator:**
If answered incorrectly, this may indicate confusion about what "supervised" means (requires labels) versus "unsupervised" (discovers structure without labels). Review the key distinction: supervised = learning from answers, unsupervised = finding patterns.

---

### Question 2 | Multiple Choice
**Correct Answer:** B

**Explanation:**
The pattern of good training performance but poor test performance is the hallmark of overfitting. The model has high variance—it is so flexible that it has memorized specifics of the training data, including noise, rather than learning the underlying generalizable patterns. This causes it to fail on new data.

**Why Other Options Are Incorrect:**
- A) Underfitting (high bias) would show poor performance on BOTH training and test data—the model is too simple to fit even the training data well
- C) Irreducible error affects both training and test equally; it doesn't cause a gap between them
- D) Data leakage typically causes overly optimistic test performance, not poor test performance

**Understanding Gap Indicator:**
If answered incorrectly, this may indicate confusion about how overfitting and underfitting manifest in train/test performance gaps. Review: overfitting = train >> test; underfitting = both poor.

---

### Question 3 | Short Answer
**Model Answer:**
Accuracy is misleading because with 99% negative cases, a model predicting "no disease" for every patient achieves 99% accuracy while being completely useless—it catches zero actual disease cases. For this **imbalanced classification** problem, better metrics include:

1. **Recall (Sensitivity):** Measures what proportion of actual disease cases were detected. Critical here because missing a disease (false negative) has severe consequences for the patient.

2. **Precision-Recall AUC:** Evaluates the tradeoff across all thresholds, focusing on the rare positive class rather than the dominant negative class.

Alternative valid metrics: F1-score (balances precision and recall), specificity at a fixed sensitivity threshold, or cost-weighted accuracy where false negatives carry higher penalty.

**Key Components Required:**
- [ ] Explains why accuracy fails (class imbalance, trivial 99% baseline)
- [ ] Proposes at least two alternative metrics
- [ ] Justifies metrics in context of disease detection (FN cost)

**Partial Credit Guidance:**
- Full credit: Correctly identifies imbalance problem + two justified metrics
- Partial credit: Identifies problem but weak metric justification, or only one appropriate metric
- No credit: Suggests accuracy is fine, or proposes inappropriate metrics

**Understanding Gap Indicator:**
If answered incorrectly, this may indicate insufficient understanding of class imbalance and metric selection. Review Model Evaluation section and the relationship between business objectives and metric choice.

---

### Question 4 | Short Answer
**Model Answer:**
**Oscillating loss:** The learning rate is **too high**. Large steps overshoot the optimum, bouncing back and forth across the loss landscape without converging.

**Extremely slow decrease:** The learning rate is **too low**. Tiny steps make progress imperceptibly slow, requiring excessive iterations to approach the optimum.

**Solution:** The **learning rate** hyperparameter must be tuned. Start with a moderate value (e.g., 0.001 for Adam), then adjust based on training curves. Learning rate schedulers can help—start higher for fast early progress, decay over time for fine convergence. Alternatively, adaptive optimizers like Adam automatically adjust per-parameter learning rates.

**Key Components Required:**
- [ ] Correctly diagnoses oscillation as learning rate too high
- [ ] Correctly diagnoses slow progress as learning rate too low
- [ ] Identifies learning rate as the hyperparameter to adjust
- [ ] Optionally mentions schedulers or adaptive optimizers

**Partial Credit Guidance:**
- Full credit: Both diagnoses correct + learning rate identified + solution strategy
- Partial credit: One diagnosis correct, or correct diagnoses but vague solution
- No credit: Attributes to wrong causes (e.g., "bad data" or "wrong architecture")

**Understanding Gap Indicator:**
If answered incorrectly, this may indicate weak understanding of gradient descent mechanics. Review how learning rate controls step size and affects convergence behavior.

---

### Question 5 | Essay
**Model Answer:**

**1. Learning Paradigm (Supervised, Unsupervised, or Both):**
Recommendation systems typically combine both paradigms. The core prediction task—"will user U like product P?"—is **supervised learning** using implicit labels (purchases=positive, views without purchase=weak positive, no interaction=implicit negative). However, **unsupervised learning** plays supporting roles: collaborative filtering uses matrix factorization to discover latent user-item factors without explicit labels, and clustering identifies user segments for cold-start handling. Many modern systems use hybrid approaches combining both.

**2. Bias-Variance with Evolving Preferences:**
User preferences evolve (fashion trends, life changes), creating concept drift. High-variance models memorizing historical patterns fail on shifted preferences; high-bias models miss nuanced taste. **Solutions:** (1) Regular retraining on recent data with time-decayed weighting (older interactions matter less); (2) Online learning for continuous adaptation; (3) Ensemble methods combining long-term preference models with short-term trend detectors; (4) Explicit drift detection triggering model refresh. The temporal dimension means the optimal bias-variance point shifts over time.

**3. Feature Engineering Strategy:**
- **User features:** Demographics, tenure, average session length, purchase frequency, category affinity vectors
- **Item features:** Category, price range, popularity, recency, seasonal flags
- **Interaction features:** Click-through history, dwell time, cart additions, purchase-to-view ratio
- **Contextual features:** Time of day, device type, referral source, current session path
- **Collaborative features:** Similar users' preferences (user-user CF), similar items (item-item CF)
- **Sequence features:** For deep learning, encode browsing sequences with RNNs/Transformers to capture intent patterns

**4. Evaluation Metrics:**
- **Offline:** Precision@K, Recall@K, NDCG (ranking quality), catalog coverage (diversity)
- **Online (A/B tests):** Click-through rate, add-to-cart rate, conversion rate, revenue per session
- **Beyond accuracy:** Diversity (avoiding filter bubbles), novelty (surfacing new items), serendipity (unexpected good recommendations)
- Accuracy alone is insufficient—a system recommending only bestsellers may have high precision but poor user experience

**5. Production Deployment Considerations:**
- **Latency:** Real-time inference requires <100ms; may need candidate generation (fast, approximate) + ranking (slower, precise) two-stage architecture
- **Scale:** Millions of users × millions of products; precompute recommendations where possible, cache heavily
- **Cold start:** New users/products lack history; use content-based fallbacks, popularity, or exploration strategies
- **Feedback loops:** Recommendations influence behavior; monitor for filter bubbles and position bias
- **A/B testing:** Rigorous experimentation infrastructure for model comparison
- **Explainability:** "Because you bought X" explanations increase trust and engagement

**Evaluation Rubric:**

| Criterion | Excellent (4) | Proficient (3) | Developing (2) | Beginning (1) |
|-----------|---------------|----------------|----------------|---------------|
| Paradigm Analysis | Correctly identifies hybrid approach with specific examples | Identifies at least one paradigm correctly | Vague or partial paradigm discussion | Incorrect paradigm classification |
| Bias-Variance | Addresses temporal drift explicitly with mitigation strategies | Acknowledges evolving preferences | Generic bias-variance discussion | No connection to recommendation context |
| Feature Engineering | Specific, diverse features across user/item/context | Several relevant feature categories | Few features, lacks specificity | Generic or irrelevant features |
| Metrics | Offline and online metrics + beyond-accuracy considerations | Multiple relevant metrics | One or two basic metrics | Only accuracy mentioned |
| Production | Multiple specific challenges with solutions | Addresses 2-3 deployment issues | Acknowledges production complexity | No deployment considerations |

**Understanding Gap Indicator:**
If response lacks depth, this may indicate:
- Difficulty applying theoretical concepts to domain-specific problems
- Weak integration of multiple ML concepts into coherent system design
- Limited awareness of production ML challenges beyond model training

---

## Diagnostic Feedback & Review Recommendations

### Performance-Based Review Guide

| If Incorrect On... | Gap Identified | Review Section | Priority |
|--------------------|----------------|----------------|----------|
| Question 1 | Supervised vs unsupervised distinction | Core Concepts 1 & 2 | High |
| Question 2 | Overfitting/underfitting recognition | Core Concepts 3 | High |
| Question 3 | Imbalanced class evaluation | Core Concepts 5 (Evaluation) | Medium |
| Question 4 | Gradient descent mechanics | Core Concepts 6 | Medium |
| Question 5 | End-to-end ML pipeline synthesis | All sections + Applications | Low |

### Targeted Review Recommendations

#### For Multiple Choice Errors (Questions 1-2)
**Indicates:** Foundational concept gaps
**Action:** Review definitions and key distinctions in:
- Study Notes: Core Concepts 1, 2, 3
- Concept Map: Paradigms subgraph, Bias-Variance cluster
- Flashcards: Cards 1 and 2
**Focus On:** Memorizing the defining characteristics that distinguish concepts

#### For Short Answer Errors (Questions 3-4)
**Indicates:** Application or analysis difficulties
**Action:** Practice applying concepts through:
- Study Notes: Practical Applications section
- Concept Map: Learning Pathway 2 (Practitioner)
- Flashcards: Cards 3 and 4
**Focus On:** Connecting symptoms to causes and selecting appropriate solutions

#### For Essay Weakness (Question 5)
**Indicates:** Integration or synthesis challenges
**Action:** Review interconnections between:
- Concept Map: Critical Path traversal
- Study Notes: All Core Concepts + Critical Analysis
- Flashcard: Card 5 (pipeline design pattern)
**Focus On:** Building mental models that connect concepts into coherent system designs

### Mastery Indicators

- **5/5 Correct:** Strong mastery demonstrated; proceed to hands-on implementation projects
- **4/5 Correct:** Good understanding; review indicated gap area before advanced topics
- **3/5 Correct:** Moderate understanding; systematic review of Core Concepts recommended
- **2/5 or below:** Foundational gaps; comprehensive re-study starting from Concept Map Critical Path

---

## Skill Chain Traceability

```
Study Notes (Source) ─────────────────────────────────────────────────────┐
    │                                                                     │
    │  Extracted: 7 Core Concepts, 12 Key Terms, 4 Applications           │
    │                                                                     │
    ├────────────────────┬────────────────────┐                           │
    │                    │                    │                           │
    ▼                    ▼                    ▼                           │
Concept Map         Flashcards            Quiz                            │
    │                    │                    │                           │
    │  18 concepts       │  5 cards           │  5 questions              │
    │  28 relationships  │  2E/2M/1H          │  2MC/2SA/1E               │
    │  4 pathways        │                    │                           │
    │                    │                    │                           │
    └────────┬───────────┘                    │                           │
             │                                │                           │
             │  Centrality → Card difficulty  │                           │
             │  Pathways → Review recs        │                           │
             │                                │                           │
             └────────────────────────────────┤                           │
                                              │                           │
                                              ▼                           │
                                    Quiz integrates all ◄─────────────────┘
                                    with full traceability
```
