# Assessment Quiz: Lesson 11 - Model Deployment and MLOps

**Source Material:** Lessons/Lesson_11.md
**Date Generated:** 2026-01-08
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

---

### Question 1 | Multiple Choice
**Cognitive Level:** Remember/Understand
**Concept Tested:** MLOps Maturity Levels
**Source Section:** Concept 1 - The ML Lifecycle and MLOps Principles

An organization has implemented an automated training pipeline that runs nightly, but deployment to production is still performed manually by the ML team after reviewing training metrics. According to the MLOps maturity model, what level is this organization at?

A) Level 0 - Manual: All processes including training are performed manually in notebooks

B) Level 1 - ML Pipeline: Automated training pipeline with manual deployment

C) Level 2 - CI/CD for ML: Automated testing and deployment with basic monitoring

D) Level 3 - Full MLOps: Continuous training with A/B testing and automated retraining

---

### Question 2 | Multiple Choice
**Cognitive Level:** Remember/Understand
**Concept Tested:** Model Optimization Techniques
**Source Section:** Concept 3 - Model Optimization for Deployment

A team needs to reduce their PyTorch model's inference latency for deployment on edge devices. They want to reduce the model size by approximately 4× while maintaining reasonable accuracy. Which optimization technique would best achieve this goal?

A) Layer fusion - combining multiple layers into single optimized operations

B) Knowledge distillation - training a smaller student model to mimic the larger teacher model

C) INT8 quantization - reducing weight precision from 32-bit floating point to 8-bit integers

D) Structured pruning - removing entire convolutional filters or attention heads

---

### Question 3 | Short Answer
**Cognitive Level:** Apply/Analyze
**Concept Tested:** Feature Store Architecture and Training-Serving Skew
**Source Section:** Concept 10 - Feature Stores and Data Management
**Expected Response Length:** 2-4 sentences

Your ML team has discovered that their fraud detection model performs well in offline evaluation but poorly in production. Investigation reveals that the features used during training are computed differently than those used during real-time inference. Explain what this problem is called, why it occurs, and how a feature store architecture would prevent it.

---

### Question 4 | Short Answer
**Cognitive Level:** Apply/Analyze
**Concept Tested:** Canary Deployment and Quality Gates
**Source Section:** Concepts 6, 9 - CI/CD for ML and Canary Deployments
**Expected Response Length:** 2-4 sentences

You are deploying a new version of a recommendation model and want to minimize risk. Describe how you would configure a canary deployment with quality gates. Include specific metrics you would monitor and conditions that would trigger either promotion to full rollout or automatic rollback.

---

### Question 5 | Essay
**Cognitive Level:** Evaluate/Synthesize
**Concepts Tested:** Complete MLOps Lifecycle (Monitoring, Drift Detection, CI/CD, Model Registry)
**Source Sections:** Concepts 1, 6, 7, 8
**Expected Response Length:** 1-2 paragraphs

A retail company has been running a demand forecasting model in production for 6 months. Recently, prediction accuracy has degraded significantly, but the existing infrastructure alerts (latency, error rate, CPU usage) show no problems. The team suspects that pandemic-related changes in consumer behavior have made the training data outdated.

Analyze this scenario and propose a comprehensive solution that addresses: (1) how to detect such issues earlier in the future, (2) how to implement an automated response when degradation is detected, and (3) how to safely deploy retrained models. Your response should demonstrate integration of monitoring, model registry, and CI/CD concepts.

**Evaluation Criteria:**
- [ ] Identifies monitoring gaps and proposes specific data/model drift detection metrics
- [ ] Describes automated retraining trigger mechanisms
- [ ] Explains model registry role in versioning and rollback capability
- [ ] Outlines safe deployment strategy (canary/blue-green) for retrained models
- [ ] Demonstrates understanding of the interconnected MLOps components

---

## Answer Key

---

### Question 1 | Multiple Choice
**Correct Answer:** B

**Explanation:**
Level 1 - ML Pipeline describes organizations that have automated the training process but still rely on manual intervention for deployment decisions. The scenario describes exactly this: nightly automated training pipelines with human-in-the-loop deployment. This is a significant step above Level 0 (where training itself is manual) but has not yet achieved the full automation of Level 2 (automated deployment with testing) or Level 3 (continuous training and automated retraining).

**Why Other Options Are Incorrect:**
- A) Level 0 would mean training is done manually in Jupyter notebooks, but the scenario explicitly mentions automated nightly training pipelines
- C) Level 2 requires automated testing AND deployment, but the scenario states deployment is manual
- D) Level 3 requires continuous training with automated retraining triggers, which goes beyond what's described

**Understanding Gap Indicator:**
If answered incorrectly, this may indicate:
- Confusion between automation of training vs. deployment processes
- Misunderstanding of the progression through MLOps maturity levels

---

### Question 2 | Multiple Choice
**Correct Answer:** C

**Explanation:**
INT8 quantization reduces the precision of model weights from 32-bit floating point (FP32) to 8-bit integers (INT8), achieving exactly a 4× reduction in model size (32 bits ÷ 8 bits = 4×). This technique is particularly effective for edge deployment because: (1) it reduces memory footprint significantly, (2) INT8 operations are faster on most hardware including mobile CPUs, and (3) modern frameworks like TensorRT and ONNX Runtime have optimized INT8 kernels.

**Why Other Options Are Incorrect:**
- A) Layer fusion improves inference speed (1.2-1.5×) but does not reduce model size (1×)
- B) Knowledge distillation can achieve variable size reduction but doesn't guarantee 4× reduction and requires retraining
- D) Structured pruning at 50% achieves approximately 2× reduction, not 4×

**Understanding Gap Indicator:**
If answered incorrectly, this may indicate:
- Confusion about the relationship between bit precision and model size
- Misunderstanding of optimization technique tradeoffs

---

### Question 3 | Short Answer
**Model Answer:**
This problem is called **training-serving skew** (or train-test skew). It occurs when the feature computation code differs between offline training pipelines and real-time inference systems—for example, using batch SQL aggregations during training but different streaming calculations during serving. A feature store prevents this by enforcing a **single feature definition** that is used for both the offline store (historical data for training) and online store (low-latency serving). Features are computed once by the feature engineering pipeline and materialized to both stores, ensuring consistency. Additionally, feature stores enable point-in-time correctness, preventing future data leakage during training.

**Key Components Required:**
- [ ] Correctly identifies the term "training-serving skew"
- [ ] Explains the root cause (different code paths for feature computation)
- [ ] Describes how feature store's unified definition prevents the problem
- [ ] Mentions offline/online store architecture

**Partial Credit Guidance:**
- Full credit: All four components addressed accurately
- Partial credit: Identifies the problem and feature store solution but lacks architectural detail
- No credit: Does not identify training-serving skew or proposes unrelated solutions

**Understanding Gap Indicator:**
If answered incompletely or incorrectly, this may indicate:
- Unfamiliarity with feature store architecture
- Confusion about where skew originates in ML pipelines

---

### Question 4 | Short Answer
**Model Answer:**
For canary deployment, I would initially route **5-10% of traffic** to the new model version using Istio traffic splitting, while 90-95% continues to the stable version. Quality gates would monitor: (1) **prediction latency** (p99 should not increase >10% vs stable), (2) **error rate** (should remain <1%), (3) **prediction distribution** (KL divergence from stable <0.1), and (4) **business metrics** like click-through rate if available. The canary would run for 30-60 minutes before evaluation. **Promotion criteria**: all metrics within thresholds, proceed to 50% then 100% rollout. **Rollback trigger**: if latency p99 increases >20%, error rate exceeds 2%, or significant prediction distribution shift is detected, automatically route 100% traffic back to stable and alert the team.

**Key Components Required:**
- [ ] Specifies traffic split percentage (e.g., 10% canary)
- [ ] Lists specific metrics to monitor (latency, errors, prediction distribution)
- [ ] Defines promotion criteria with thresholds
- [ ] Defines rollback trigger conditions

**Partial Credit Guidance:**
- Full credit: All four components with specific thresholds
- Partial credit: Describes canary concept correctly but lacks specific metrics or thresholds
- No credit: Confuses canary with blue-green or A/B testing concepts

**Understanding Gap Indicator:**
If answered incompletely or incorrectly, this may indicate:
- Unfamiliarity with traffic splitting mechanisms
- Difficulty translating monitoring concepts into actionable deployment criteria

---

### Question 5 | Essay
**Model Answer:**

The scenario describes a classic case of **concept drift**—where the underlying relationship between inputs and outputs has changed due to external factors (pandemic behavior changes), even though the model infrastructure is functioning correctly. To address this comprehensively:

**(1) Early Detection:** The monitoring stack must extend beyond infrastructure metrics to include **data drift detection** (Kolmogorov-Smirnov test or Population Stability Index comparing current input distributions to training data) and **model performance proxy metrics** (prediction confidence distribution, output distribution shifts). Since ground truth for demand forecasting arrives with delay, implementing **weekly accuracy tracking** against actual sales would catch degradation within 1-2 weeks instead of waiting for business complaints. Prometheus gauges for feature drift scores with alerting rules (e.g., `feature_drift_score > 0.15` triggers warning) would provide proactive notification.

**(2) Automated Response:** When drift is detected, an **automated retraining pipeline** should trigger. This pipeline would: fetch recent data from the feature store, retrain the model using the existing architecture, evaluate against holdout data, and register the new model version in **MLflow** with full lineage (training data version, code commit, hyperparameters, evaluation metrics). Quality gates would validate that the retrained model meets accuracy thresholds before it becomes a deployment candidate. The model registry provides version control and enables instant rollback if needed.

**(3) Safe Deployment:** The retrained model should deploy via **canary release**—starting with 10% traffic, monitoring prediction quality and business metrics for 1-2 days, then gradually increasing to full rollout. Istio VirtualService configurations enable precise traffic splitting. If the canary shows degraded performance compared to the current model, automatic rollback routes all traffic back to the previous version stored in the model registry. This champion-challenger framework ensures new models prove their value before full promotion.

The integration is critical: **monitoring** detects drift → triggers **CI/CD pipeline** for retraining → **model registry** stores new version with lineage → **canary deployment** validates in production → monitoring continues the feedback loop. This creates a sustainable, self-correcting MLOps system rather than relying on manual intervention.

**Evaluation Rubric:**

| Criterion | Excellent (4) | Proficient (3) | Developing (2) | Beginning (1) |
|-----------|---------------|----------------|----------------|---------------|
| **Monitoring & Detection** | Specific drift detection methods (KS, PSI) with concrete metrics and thresholds | Identifies need for drift detection without specific methods | Mentions monitoring generally without drift focus | No clear detection strategy |
| **Automated Response** | Complete pipeline: trigger → retrain → validate → register with lineage | Describes retraining automation but missing components | Mentions retraining but manual process | No automated response mechanism |
| **Safe Deployment** | Canary with specific traffic percentages, metrics, and rollback conditions | Describes gradual rollout without specific configuration | Mentions deployment strategy without details | No safe deployment consideration |
| **Integration & Synthesis** | Clearly explains how all components connect in feedback loop | Shows understanding of component relationships | Components mentioned but not connected | Treats components as isolated |

**Understanding Gap Indicator:**
If response lacks depth, this may indicate:
- Isolated understanding of MLOps components without seeing the integrated system
- Confusion between data drift (input changes) and concept drift (relationship changes)
- Unfamiliarity with how model registry enables operational workflows

---

## Diagnostic Feedback & Review Recommendations

### Performance-Based Review Guide

| If Incorrect On... | Gap Identified | Review Section | Priority |
|--------------------|----------------|----------------|----------|
| Question 1 | MLOps maturity model comprehension | Concept 1: ML Lifecycle and MLOps Principles | High |
| Question 2 | Model optimization technique selection | Concept 3: Model Optimization for Deployment | High |
| Question 3 | Feature store architecture | Concept 10: Feature Stores and Data Management | Medium |
| Question 4 | Deployment strategy configuration | Concept 9: A/B Testing and Canary Deployments | Medium |
| Question 5 | Integrated MLOps system design | Concepts 1, 6, 7, 8 (Synthesis) | Low |

### Targeted Review Recommendations

#### For Multiple Choice Errors (Questions 1-2)
**Indicates:** Foundational concept gaps in MLOps principles or optimization techniques
**Action:** Review definitions and core principles in:
- Concept 1: MLOps Maturity Levels table and comparison with traditional software
- Concept 3: Optimization Techniques Overview table with size/speed/accuracy tradeoffs
**Focus On:** Understanding the distinctions between maturity levels and when to apply each optimization technique

#### For Short Answer Errors (Questions 3-4)
**Indicates:** Application difficulties with feature management and deployment strategies
**Action:** Practice applying concepts through:
- Concept 10: Feature Store Architecture diagram and Feast code examples
- Concept 9: Canary Deployment traffic splitting configuration and automated analysis code
**Focus On:** Translating architectural concepts into concrete configurations with specific thresholds

#### For Essay Weakness (Question 5)
**Indicates:** Integration or synthesis challenges across the MLOps lifecycle
**Action:** Review interconnections between:
- How Monitoring (Concept 8) triggers CI/CD (Concept 6)
- How Model Registry (Concept 7) enables deployment strategies (Concept 9)
- The complete MLOps Lifecycle diagram in Concept 1
**Focus On:** Understanding feedback loops and component dependencies in production ML systems

### Mastery Indicators

| Score | Mastery Level | Interpretation | Next Steps |
|-------|---------------|----------------|------------|
| 5/5 | Expert | Strong mastery of MLOps concepts and integration | Proceed to hands-on implementation projects |
| 4/5 | Proficient | Good understanding with minor gap | Review indicated gap area, then proceed |
| 3/5 | Developing | Moderate understanding | Systematic review of weak areas recommended |
| 2/5 | Foundational | Significant gaps present | Re-study core concepts before proceeding |
| 1/5 or below | Beginning | Fundamental gaps | Comprehensive re-study of entire lesson advised |

### Cross-Reference to Practice Problems

| Quiz Question | Related Practice Problem | Skill Reinforcement |
|--------------|-------------------------|---------------------|
| Q1 (Maturity Levels) | P3 (CI/CD Pipeline Design) | Understanding automation levels |
| Q2 (Optimization) | P1 (Model Export and Optimization) | Hands-on quantization implementation |
| Q3 (Feature Store) | P5 (Monitoring Gap Diagnosis) | Data consistency in production |
| Q4 (Canary Deployment) | P4 (Debug Faulty Canary) | Configuration troubleshooting |
| Q5 (Integrated MLOps) | P3, P5 (Combined) | End-to-end pipeline thinking |

---

*Generated from Lesson 11: Model Deployment and MLOps | Quiz Skill*
