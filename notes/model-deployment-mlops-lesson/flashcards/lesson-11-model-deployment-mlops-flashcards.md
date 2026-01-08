# Flashcard Set: Lesson 11 - Model Deployment and MLOps

**Source:** Lessons/Lesson_11.md
**Date Generated:** 2026-01-08
**Total Cards:** 5
**Distribution:** 2 Easy | 2 Medium | 1 Hard

---

## Critical Knowledge Summary

Concepts appearing in multiple cards (prioritize for review):
- **Model Serving:** Appears in Cards 1, 3, 5
- **Data Drift:** Appears in Cards 2, 4, 5
- **Quantization:** Appears in Cards 1, 3
- **Containerization/Kubernetes:** Appears in Cards 3, 5

---

## Flashcards

---

### Card 1 | Easy
**Cognitive Level:** Remember
**Concept:** Quantization
**Source Section:** Concept 3 - Model Optimization for Deployment

**FRONT (Question):**
What is quantization in the context of model deployment, and what is the typical memory reduction when converting from FP32 to INT8?

**BACK (Answer):**
Quantization is the process of reducing the numerical precision of model weights and activations from floating-point (FP32) to lower-bit representations (INT8 or INT4). Converting from FP32 to INT8 provides a **4× reduction** in model size (32 bits → 8 bits per weight) and enables faster inference on most hardware since INT8 operations are computationally cheaper than FP32 operations.

**Critical Knowledge Flag:** Yes

---

### Card 2 | Easy
**Cognitive Level:** Understand
**Concept:** Data Drift
**Source Section:** Concept 8 - Monitoring and Observability

**FRONT (Question):**
Define data drift and explain why it is a critical concern for production ML systems.

**BACK (Answer):**
Data drift refers to changes in the statistical distribution of input data over time compared to the training data distribution. It is critical because:
1. Models trained on historical data may produce degraded predictions when input patterns change
2. Drift can indicate changes in user behavior, market conditions, or upstream data pipelines
3. Undetected drift leads to silent model failures where the system continues serving poor predictions

Common detection methods include the Kolmogorov-Smirnov test (KS statistic) and Population Stability Index (PSI).

**Critical Knowledge Flag:** Yes

---

### Card 3 | Medium
**Cognitive Level:** Apply
**Concept:** Model Serving Patterns
**Source Section:** Concept 4 - Model Serving Architectures

**FRONT (Question):**
A fintech company needs to deploy a fraud detection model. The model must make predictions in under 50ms for real-time transaction approval, handle traffic spikes during peak shopping hours, and run on a Kubernetes cluster. Which serving pattern should be used, and what architectural components are required?

**BACK (Answer):**
**Serving Pattern:** Online/Real-time Serving

**Required Architectural Components:**
1. **Model Server:** TorchServe, TensorFlow Serving, or Triton with gRPC for low latency
2. **Load Balancer:** Distribute requests across model replicas
3. **Kubernetes Deployment:** Multiple replicas with GPU resources allocated
4. **Horizontal Pod Autoscaler (HPA):** Scale replicas based on CPU/latency metrics during traffic spikes
5. **Health Checks:** Readiness and liveness probes for reliability
6. **Dynamic Batching:** Optional; aggregate requests to maximize GPU throughput while meeting latency SLA

**Optimization:** Apply INT8 quantization to reduce inference latency below 50ms threshold.

**Critical Knowledge Flag:** Yes

---

### Card 4 | Medium
**Cognitive Level:** Analyze
**Concept:** Canary Deployment vs. Blue-Green Deployment
**Source Section:** Concept 9 - A/B Testing and Canary Deployments

**FRONT (Question):**
Compare canary deployment and blue-green deployment strategies for ML models. When would you choose each approach, and what are the tradeoffs?

**BACK (Answer):**
| Aspect | Canary Deployment | Blue-Green Deployment |
|--------|-------------------|----------------------|
| **Traffic Split** | Gradual (e.g., 5% → 25% → 100%) | Instant 100% switch |
| **Risk Level** | Lower (limited blast radius) | Higher (all users affected) |
| **Resource Cost** | ~1.1× (small canary + stable) | 2× (full duplicate environment) |
| **Rollback Speed** | Gradual reduction | Instant switch back |
| **Monitoring Time** | Extended (observe canary metrics) | Minimal |

**When to Choose Canary:**
- High-stakes models where errors are costly
- When you need statistical confidence in the new model
- When you want to monitor real production behavior before full rollout

**When to Choose Blue-Green:**
- Urgent deployments requiring immediate full rollout
- Environments with sufficient resources for duplication
- When model changes are well-tested in staging

**Critical Knowledge Flag:** Yes

---

### Card 5 | Hard
**Cognitive Level:** Evaluate/Synthesize
**Concept:** End-to-End MLOps Architecture
**Source Section:** Concepts 1, 6, 7, 8, 10 (Synthesis)

**FRONT (Question):**
Your organization is at MLOps Maturity Level 1 (automated training pipeline, manual deployment). Design an upgrade path to Level 3 (full MLOps) by describing the key components needed across CI/CD, monitoring, feature management, and deployment. Identify the critical dependencies between these components.

**BACK (Answer):**
**Level 1 → Level 2 (CI/CD for ML):**
1. **Data Validation:** Implement Great Expectations for schema/distribution checks
2. **Quality Gates:** Define accuracy thresholds, latency requirements, fairness checks
3. **Model Registry:** Deploy MLflow for artifact storage, versioning, and lineage
4. **Automated Testing:** Unit tests (code), integration tests (pipeline), model validation tests
5. **Staging Environment:** Kubernetes namespace for pre-production validation

**Level 2 → Level 3 (Full MLOps):**
1. **Continuous Training:** Trigger retraining on data drift detection or scheduled intervals
2. **Feature Store:** Implement Feast with offline (training) and online (serving) stores
3. **Canary Deployment:** Istio traffic splitting for gradual rollout
4. **A/B Testing Framework:** Statistical analysis of business metrics
5. **Automated Rollback:** Alert-triggered rollback when drift/errors exceed thresholds
6. **Comprehensive Monitoring:** Prometheus metrics across 5 layers (infrastructure, service, model, data quality, business)

**Critical Dependencies:**
```
Feature Store → Consistent training/serving (prevents skew)
Model Registry → Versioning enables rollback
Data Validation → Quality gates depend on validated data
Monitoring → Triggers continuous training and rollback
CI/CD Pipeline → Orchestrates all automated workflows
```

**Key Insight:** The transition requires both technical infrastructure (tools) and process changes (ownership, review, response procedures). Without monitoring, continuous training cannot be safely triggered. Without feature stores, training-serving skew will undermine model quality.

**Critical Knowledge Flag:** Yes

---

## Export Formats

### Anki-Compatible (Tab-Separated)

```
What is quantization in the context of model deployment, and what is the typical memory reduction when converting from FP32 to INT8?	Quantization reduces numerical precision from FP32 to INT8, providing 4× memory reduction (32 bits → 8 bits per weight) and faster inference.	mlops::optimization::easy
Define data drift and explain why it is a critical concern for production ML systems.	Data drift is changes in input distribution over time compared to training data. Critical because: (1) degrades predictions, (2) indicates environmental changes, (3) causes silent failures. Detected via KS test or PSI.	mlops::monitoring::easy
A fintech company needs fraud detection with <50ms latency, traffic spike handling, on Kubernetes. Which serving pattern and components?	Online/Real-time serving with: Model Server (Triton/TorchServe + gRPC), Load Balancer, K8s Deployment with replicas, HPA for autoscaling, health checks, optional dynamic batching, INT8 quantization.	mlops::serving::medium
Compare canary vs blue-green deployment for ML models. When choose each?	Canary: gradual rollout, lower risk, ~1.1× resources, extended monitoring - use for high-stakes models. Blue-Green: instant switch, 2× resources, instant rollback - use for urgent deploys with good staging tests.	mlops::deployment::medium
Design upgrade from MLOps Level 1 to Level 3. What components across CI/CD, monitoring, features, deployment? Dependencies?	L1→L2: Data validation, quality gates, MLflow registry, automated testing, staging. L2→L3: Continuous training, Feast feature store, Istio canary, A/B testing, auto-rollback, 5-layer Prometheus monitoring. Dependencies: Feature store→consistency, Registry→rollback, Monitoring→triggers.	mlops::architecture::hard
```

### CSV Format

```csv
"Front","Back","Difficulty","Concept"
"What is quantization in the context of model deployment, and what is the typical memory reduction when converting from FP32 to INT8?","Quantization reduces numerical precision from FP32 to INT8, providing 4× memory reduction and faster inference.","Easy","Quantization"
"Define data drift and explain why it is a critical concern for production ML systems.","Data drift is changes in input distribution over time. Critical because it degrades predictions, indicates environmental changes, and causes silent failures.","Easy","Data Drift"
"A fintech company needs fraud detection with <50ms latency, traffic spike handling, on Kubernetes. Which serving pattern and components?","Online/Real-time serving with Model Server, Load Balancer, K8s Deployment, HPA, health checks, dynamic batching, INT8 quantization.","Medium","Model Serving Patterns"
"Compare canary vs blue-green deployment for ML models. When choose each?","Canary: gradual, lower risk, extended monitoring for high-stakes. Blue-Green: instant switch, 2× resources, for urgent deploys.","Medium","Deployment Strategies"
"Design upgrade from MLOps Level 1 to Level 3. Components and dependencies?","L1→L2: validation, gates, registry, testing. L2→L3: continuous training, feature store, canary, A/B, auto-rollback, monitoring. Dependencies chain through feature store, registry, and monitoring.","Hard","MLOps Architecture"
```

### Plain Text Review

```
Q: What is quantization in the context of model deployment, and what is the typical memory reduction when converting from FP32 to INT8?
A: Quantization reduces numerical precision from FP32 to INT8, providing 4× memory reduction (32 bits → 8 bits per weight) and faster inference on most hardware.

---

Q: Define data drift and explain why it is a critical concern for production ML systems.
A: Data drift is changes in input data distribution over time compared to training data. Critical because: (1) degrades predictions, (2) indicates environmental changes, (3) causes silent failures. Detected via KS test or PSI.

---

Q: A fintech company needs fraud detection with <50ms latency, traffic spike handling, on Kubernetes. Which serving pattern and components?
A: Online/Real-time serving with: Model Server (Triton/TorchServe + gRPC), Load Balancer, K8s Deployment with replicas, HPA for autoscaling, health checks, optional dynamic batching, INT8 quantization.

---

Q: Compare canary vs blue-green deployment for ML models. When choose each?
A: Canary: gradual rollout, lower risk, ~1.1× resources, extended monitoring - use for high-stakes models. Blue-Green: instant switch, 2× resources, instant rollback - use for urgent deploys with good staging tests.

---

Q: Design upgrade from MLOps Level 1 to Level 3. Components and dependencies?
A: L1→L2: Data validation, quality gates, MLflow registry, automated testing, staging. L2→L3: Continuous training, Feast feature store, Istio canary, A/B testing, auto-rollback, 5-layer Prometheus monitoring. Key dependencies: Feature store enables consistency, Registry enables rollback, Monitoring triggers automation.
```

---

## Source Mapping

| Card | Source Section | Key Terminology Used |
|------|----------------|---------------------|
| 1 | Concept 3: Model Optimization | Quantization, FP32, INT8, PTQ, QAT |
| 2 | Concept 8: Monitoring | Data drift, KS test, PSI, distribution |
| 3 | Concept 4: Model Serving | Online serving, REST API, gRPC, HPA, dynamic batching |
| 4 | Concept 9: Deployment Strategies | Canary, blue-green, traffic splitting, rollback |
| 5 | Concepts 1, 6, 7, 8, 10 | MLOps maturity, CI/CD, model registry, monitoring, feature store |

---

## Study Recommendations

### Review Schedule (Spaced Repetition)
- **Day 1:** All 5 cards (initial learning)
- **Day 3:** Cards marked incorrect + Hard card
- **Day 7:** All 5 cards (consolidation)
- **Day 14:** Random selection of 3 cards
- **Day 30:** All 5 cards (long-term retention test)

### Prerequisite Check
Before studying these cards, ensure familiarity with:
- Basic ML model training concepts (Lesson 5)
- Container fundamentals (Docker basics)
- REST API concepts

### Extension Topics
After mastering these cards, explore:
- Specific tools (MLflow, Feast, Prometheus) in hands-on labs
- Kubernetes deployment YAML configurations
- Statistical tests for drift detection (implementation)

---

*Generated from Lesson 11: Model Deployment and MLOps | Flashcards Skill*
