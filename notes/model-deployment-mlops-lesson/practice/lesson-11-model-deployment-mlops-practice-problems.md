# Practice Problems: Lesson 11 - Model Deployment and MLOps

**Source:** Lessons/Lesson_11.md
**Original Source Path:** C:\agentic_ai\StudyNotes\Lessons\Lesson_11.md
**Date Generated:** 2026-01-08
**Total Problems:** 5
**Estimated Total Time:** 75-100 minutes
**Distribution:** 1 Warm-Up | 2 Skill-Builder | 1 Challenge | 1 Debug/Fix

---

## Overview

### Concepts Practiced

| Concept | Problems | Mastery Indicator |
|---------|----------|-------------------|
| Model Serialization (ONNX) | P1, P3 | Can export and validate models |
| Quantization | P1, P4 | Can apply PTQ and measure impact |
| Kubernetes Deployment | P2, P4 | Can write deployment manifests |
| Data Drift Detection | P3, P5 | Can implement and interpret drift tests |
| CI/CD Pipelines | P3, P4 | Can design quality gates |
| Canary Deployment | P4 | Can configure traffic splitting |
| Monitoring/Alerting | P5 | Can identify monitoring gaps |

### Recommended Approach

1. Attempt each problem before looking at hints
2. Use hints progressively—don't skip to solution
3. After solving, read solution to compare approaches
4. Review Common Mistakes even if you solved correctly
5. Attempt Extension Challenges for deeper mastery

### Self-Assessment Guide

| Problems Solved (no hints) | Mastery Level | Recommendation |
|---------------------------|---------------|----------------|
| 5/5 | Expert | Proceed to advanced MLOps tooling |
| 4/5 | Proficient | Review one gap area |
| 3/5 | Developing | More practice recommended |
| 2/5 or below | Foundational | Re-review study notes first |

---

## Problems

---

## Problem 1: Model Export and Optimization

**Type:** Warm-Up
**Concepts Practiced:** Model Serialization, ONNX Export, Post-Training Quantization
**Estimated Time:** 15 minutes
**Prerequisites:** PyTorch basics, understanding of model inference

### Problem Statement

You have a trained PyTorch image classification model that needs to be deployed to a production server. The model currently uses FP32 precision and takes 45ms per inference on CPU. Your latency budget is 25ms.

Given the following model class:

```python
import torch
import torch.nn as nn

class SimpleClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Assume model is trained and loaded
model = SimpleClassifier()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
```

### Requirements

- [ ] Export the model to ONNX format with dynamic batch size
- [ ] Apply post-training dynamic quantization to the model
- [ ] Write code to validate that the quantized model produces similar outputs to the original

### Hints (Progressive)

<details>
<summary>Hint 1: Getting Started</summary>

For ONNX export, you need `torch.onnx.export()`. The key is specifying `dynamic_axes` for batch size flexibility. For quantization, look at `torch.quantization.quantize_dynamic()`.

</details>

<details>
<summary>Hint 2: Key Insight</summary>

Dynamic quantization in PyTorch targets specific layer types. For this model, `nn.Linear` layers are good candidates. The `dtype=torch.qint8` parameter specifies INT8 quantization.

</details>

<details>
<summary>Hint 3: Nearly There</summary>

For validation, compare outputs using `torch.allclose()` with appropriate tolerance (atol=1e-3 for quantization). Create a sample input tensor with shape `(1, 3, 32, 32)` for a 32x32 RGB image.

</details>

### Solution

<details>
<summary>Click to reveal full solution</summary>

**Approach:**
Export to ONNX for cross-platform deployment, then apply dynamic quantization for INT8 inference speed improvement.

**Step-by-Step Solution:**

```python
import torch
import torch.onnx
import torch.quantization
import numpy as np

# Step 1: Export to ONNX with dynamic batch size
example_input = torch.randn(1, 3, 32, 32)

torch.onnx.export(
    model,
    example_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    },
    opset_version=11
)
print("ONNX export complete: model.onnx")

# Step 2: Apply post-training dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # Quantize Linear layers
    dtype=torch.qint8
)
print("Quantization complete")

# Step 3: Validate outputs are similar
test_input = torch.randn(1, 3, 32, 32)

with torch.no_grad():
    original_output = model(test_input)
    quantized_output = quantized_model(test_input)

# Check similarity
is_similar = torch.allclose(original_output, quantized_output, atol=1e-2)
max_diff = (original_output - quantized_output).abs().max().item()

print(f"Outputs similar (atol=1e-2): {is_similar}")
print(f"Maximum difference: {max_diff:.6f}")

# Bonus: Compare model sizes
import os
torch.save(model.state_dict(), 'original.pth')
torch.save(quantized_model.state_dict(), 'quantized.pth')
original_size = os.path.getsize('original.pth') / 1024
quantized_size = os.path.getsize('quantized.pth') / 1024
print(f"Original size: {original_size:.1f} KB")
print(f"Quantized size: {quantized_size:.1f} KB")
print(f"Compression ratio: {original_size/quantized_size:.2f}x")
```

**Final Answer:**
The ONNX export enables deployment on any ONNX-compatible runtime. Dynamic quantization reduces Linear layer precision from FP32 to INT8, typically achieving 2-4x speedup while maintaining accuracy within 1-2%.

**Why This Works:**
Dynamic quantization converts weights to INT8 at load time and computes activations in INT8 during inference. This is ideal for models with large Linear layers (like classifiers) where compute is weight-bound. The ONNX format provides interoperability across frameworks and hardware.

</details>

### Common Mistakes

- ❌ **Mistake:** Forgetting to set `model.eval()` before export
  - **Why it happens:** Training mode includes dropout and batch norm in training state
  - **How to avoid:** Always call `model.eval()` before any inference or export operation

- ❌ **Mistake:** Not specifying `dynamic_axes` in ONNX export
  - **Why it happens:** Default export assumes fixed batch size
  - **How to avoid:** Always define dynamic axes for dimensions that vary (batch size, sequence length)

- ❌ **Mistake:** Quantizing Conv2d layers with dynamic quantization
  - **Why it happens:** Assuming all layers benefit equally
  - **How to avoid:** Dynamic quantization works best on Linear/LSTM; use static quantization for Conv2d

### Extension Challenge

Implement static quantization for the Conv2d layers using a calibration dataset. Compare the inference speed and accuracy of dynamic vs. static quantization.

---

## Problem 2: Kubernetes Deployment Configuration

**Type:** Skill-Builder
**Concepts Practiced:** Containerization, Kubernetes Manifests, Resource Management, Health Checks
**Estimated Time:** 20 minutes
**Prerequisites:** Basic Docker knowledge, YAML syntax, REST API concepts

### Problem Statement

You need to deploy a model serving container to Kubernetes. The container image `myregistry/model-server:v1.0.0` serves predictions via HTTP on port 8080. The endpoint `/predict` handles POST requests, and `/health` returns health status.

**Requirements:**
- The model requires 2GB memory minimum, 4GB maximum
- Each pod needs 1 CPU core, can burst to 2 cores
- Need 3 replicas for high availability
- Pods should only receive traffic after model is loaded (takes ~30 seconds)
- Pods should be restarted if they become unresponsive
- Expose the service externally via LoadBalancer

### Requirements

- [ ] Write a Kubernetes Deployment manifest
- [ ] Configure appropriate resource requests and limits
- [ ] Add readiness and liveness probes
- [ ] Create a Service to expose the deployment

### Hints (Progressive)

<details>
<summary>Hint 1: Getting Started</summary>

A Kubernetes Deployment needs `apiVersion: apps/v1`, a `spec` with `replicas`, and a `template` containing the pod specification. Resources go under `containers[].resources`.

</details>

<details>
<summary>Hint 2: Key Insight</summary>

Readiness probes determine when a pod can receive traffic—use `initialDelaySeconds: 30` since model loading takes 30 seconds. Liveness probes detect hung processes—use a longer interval to avoid unnecessary restarts.

</details>

<details>
<summary>Hint 3: Nearly There</summary>

For probes, use `httpGet` with `path: /health` and `port: 8080`. The Service needs `type: LoadBalancer` and `selector` matching the Deployment's pod labels.

</details>

### Solution

<details>
<summary>Click to reveal full solution</summary>

**Approach:**
Create a Deployment with resource constraints and health probes, then expose it via a LoadBalancer Service.

**Step-by-Step Solution:**

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-serving
  labels:
    app: model-serving
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model-serving
  template:
    metadata:
      labels:
        app: model-serving
    spec:
      containers:
      - name: model-server
        image: myregistry/model-server:v1.0.0
        ports:
        - containerPort: 8080
          name: http

        # Resource management
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"

        # Readiness probe - determines when pod can receive traffic
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30  # Wait for model to load
          periodSeconds: 10
          timeoutSeconds: 5
          successThreshold: 1
          failureThreshold: 3

        # Liveness probe - restarts unresponsive pods
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 60  # Give more time before checking liveness
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3

---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: model-serving
spec:
  type: LoadBalancer
  selector:
    app: model-serving
  ports:
  - name: http
    port: 80
    targetPort: 8080
    protocol: TCP
```

**Deployment Commands:**
```bash
# Apply configurations
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# Verify deployment
kubectl get deployments
kubectl get pods
kubectl get services

# Check pod readiness
kubectl describe pod <pod-name>

# View logs
kubectl logs -f deployment/model-serving
```

**Why This Works:**
- **Resource requests** ensure pods are scheduled on nodes with sufficient capacity
- **Resource limits** prevent runaway pods from affecting others
- **Readiness probe** prevents traffic to pods still loading models
- **Liveness probe** automatically restarts hung pods
- **LoadBalancer** service provides external access with load distribution

</details>

### Common Mistakes

- ❌ **Mistake:** Setting liveness probe `initialDelaySeconds` too low
  - **Why it happens:** Copy-paste from readiness probe settings
  - **How to avoid:** Liveness delay should be > readiness delay + startup time buffer

- ❌ **Mistake:** Using same values for requests and limits
  - **Why it happens:** Misunderstanding the difference
  - **How to avoid:** Requests = guaranteed minimum; Limits = hard cap. Allow burst headroom.

- ❌ **Mistake:** Forgetting to match Service selector with pod labels
  - **Why it happens:** Labels are defined in multiple places
  - **How to avoid:** Verify `spec.selector.matchLabels` in Deployment matches `spec.selector` in Service

### Extension Challenge

Add a Horizontal Pod Autoscaler (HPA) that scales between 3-10 replicas based on CPU utilization (target 70%) and custom metric `requests_per_second` (target 100 RPS per pod).

---

## Problem 3: CI/CD Pipeline Design

**Type:** Skill-Builder
**Concepts Practiced:** CI/CD, Data Validation, Quality Gates, Model Registry
**Estimated Time:** 20 minutes
**Prerequisites:** Git workflows, basic ML training concepts, YAML

### Problem Statement

Design a GitHub Actions CI/CD pipeline for an ML project. The pipeline should:

1. Run unit tests on code changes
2. Validate training data against expected schema
3. Train the model (simulated with a script)
4. Evaluate model performance
5. Only deploy if accuracy > 0.90 and latency < 100ms
6. Register the model in MLflow if gates pass

**Given scripts:**
- `pytest tests/` - runs unit tests
- `python scripts/validate_data.py` - validates data (returns exit code 0 if valid)
- `python scripts/train.py` - trains model, outputs to `outputs/model/`
- `python scripts/evaluate.py` - outputs `outputs/metrics.json` with `{"accuracy": 0.XX, "latency_ms": XX}`
- `python scripts/register_model.py` - registers model with MLflow

### Requirements

- [ ] Create a GitHub Actions workflow YAML file
- [ ] Define jobs with appropriate dependencies
- [ ] Implement quality gate logic based on metrics
- [ ] Upload model artifacts between jobs

### Hints (Progressive)

<details>
<summary>Hint 1: Getting Started</summary>

GitHub Actions workflows use `jobs` that can depend on each other via `needs`. Each job has `steps` that run sequentially. Artifacts are shared between jobs using `actions/upload-artifact` and `actions/download-artifact`.

</details>

<details>
<summary>Hint 2: Key Insight</summary>

To implement quality gates, read `metrics.json` in a step and use shell conditionals or a dedicated script. You can fail a step with `exit 1` if gates don't pass. Use `jq` to parse JSON in bash.

</details>

<details>
<summary>Hint 3: Nearly There</summary>

Structure as: `test` → `train` (needs test) → `evaluate` (needs train) → `deploy` (needs evaluate). The evaluate job should check metrics and fail if gates don't pass, preventing deploy from running.

</details>

### Solution

<details>
<summary>Click to reveal full solution</summary>

**Approach:**
Create a sequential pipeline where each stage gates the next, with explicit quality checks before deployment.

**Step-by-Step Solution:**

```yaml
# .github/workflows/ml-pipeline.yaml
name: ML Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run unit tests
        run: pytest tests/ -v

      - name: Validate training data
        run: python scripts/validate_data.py

  train:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Train model
        run: python scripts/train.py

      - name: Upload model artifact
        uses: actions/upload-artifact@v4
        with:
          name: trained-model
          path: outputs/model/

  evaluate:
    needs: train
    runs-on: ubuntu-latest
    outputs:
      gates_passed: ${{ steps.quality_gates.outputs.passed }}
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Download model artifact
        uses: actions/download-artifact@v4
        with:
          name: trained-model
          path: outputs/model/

      - name: Evaluate model
        run: python scripts/evaluate.py

      - name: Check quality gates
        id: quality_gates
        run: |
          # Read metrics
          ACCURACY=$(jq -r '.accuracy' outputs/metrics.json)
          LATENCY=$(jq -r '.latency_ms' outputs/metrics.json)

          echo "Accuracy: $ACCURACY"
          echo "Latency: $LATENCY ms"

          # Quality gate checks
          ACCURACY_PASS=$(echo "$ACCURACY > 0.90" | bc -l)
          LATENCY_PASS=$(echo "$LATENCY < 100" | bc -l)

          if [ "$ACCURACY_PASS" -eq 1 ] && [ "$LATENCY_PASS" -eq 1 ]; then
            echo "✅ Quality gates PASSED"
            echo "passed=true" >> $GITHUB_OUTPUT
          else
            echo "❌ Quality gates FAILED"
            echo "  Accuracy gate (>0.90): $([ $ACCURACY_PASS -eq 1 ] && echo PASS || echo FAIL)"
            echo "  Latency gate (<100ms): $([ $LATENCY_PASS -eq 1 ] && echo PASS || echo FAIL)"
            echo "passed=false" >> $GITHUB_OUTPUT
            exit 1
          fi

      - name: Upload metrics
        uses: actions/upload-artifact@v4
        with:
          name: metrics
          path: outputs/metrics.json

  deploy:
    needs: evaluate
    if: needs.evaluate.outputs.gates_passed == 'true'
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Download model artifact
        uses: actions/download-artifact@v4
        with:
          name: trained-model
          path: outputs/model/

      - name: Register model with MLflow
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
        run: python scripts/register_model.py

      - name: Deploy to staging
        run: |
          echo "Deploying model to staging environment..."
          # kubectl apply -f k8s/staging/
```

**Pipeline Flow:**
```
test (unit tests + data validation)
    ↓ (pass)
train (model training)
    ↓ (always after train)
evaluate (performance check + quality gates)
    ↓ (only if gates pass)
deploy (MLflow registration + staging deployment)
```

**Why This Works:**
- **Sequential dependencies** ensure each stage validates before proceeding
- **Quality gates** with explicit thresholds prevent bad models from deploying
- **Artifact passing** shares model files between jobs without retraining
- **Environment protection** on deploy job enables approval workflows

</details>

### Common Mistakes

- ❌ **Mistake:** Not using artifacts to pass model between jobs
  - **Why it happens:** Assuming filesystem persists across jobs
  - **How to avoid:** Each job runs in a fresh environment; use artifacts for file sharing

- ❌ **Mistake:** Quality gate check doesn't fail the pipeline
  - **Why it happens:** Missing `exit 1` when gates fail
  - **How to avoid:** Always exit with non-zero code on failure to stop downstream jobs

- ❌ **Mistake:** Hardcoding secrets in workflow file
  - **Why it happens:** Quick testing shortcuts
  - **How to avoid:** Use GitHub Secrets (`${{ secrets.NAME }}`) for sensitive values

### Extension Challenge

Extend the pipeline to include:
1. A canary deployment stage that deploys to 10% of traffic
2. A monitoring step that waits 30 minutes and checks error rates
3. Automatic rollback if error rate exceeds 1%

---

## Problem 4: Debug Faulty Canary Deployment

**Type:** Debug/Fix
**Concepts Practiced:** Canary Deployment, Istio Traffic Management, Monitoring
**Estimated Time:** 15 minutes
**Prerequisites:** Kubernetes basics, understanding of traffic splitting

### Problem Statement

Your team deployed a canary configuration for a new model version, but users are reporting that the canary is receiving too much traffic and some requests are failing. Review the following Istio configuration and identify the bugs:

```yaml
# virtual-service.yaml (BUGGY)
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: model-serving
spec:
  hosts:
  - model-serving
  http:
  - route:
    - destination:
        host: model-serving
        subset: stable
      weight: 90
    - destination:
        host: model-serving
        subset: canary
      weight: 20  # BUG 1: Weights should sum to 100
---
# destination-rule.yaml (BUGGY)
apiVersion: networking.istio.io/v1alpha3
kind: DestinationRule
metadata:
  name: model-serving
spec:
  host: model-serving
  subsets:
  - name: stable
    labels:
      version: v1.0.0
  - name: canary
    labels:
      version: v1.0.0  # BUG 2: Same version label as stable
---
# deployment-canary.yaml (BUGGY)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-serving-canary
spec:
  replicas: 1
  selector:
    matchLabels:
      app: model-serving
  template:
    metadata:
      labels:
        app: model-serving
        version: v1.1.0
    spec:
      containers:
      - name: model-server
        image: myregistry/model-server:v1.1.0
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          # BUG 3: No limits defined - canary could consume excessive resources
        # BUG 4: No readiness probe - traffic sent before model loads
```

### Requirements

- [ ] Identify all bugs in the configuration
- [ ] Explain the impact of each bug
- [ ] Provide corrected configurations

### Hints (Progressive)

<details>
<summary>Hint 1: Getting Started</summary>

Look at the VirtualService weights—what should they sum to? Check the DestinationRule labels against the Deployment labels.

</details>

<details>
<summary>Hint 2: Key Insight</summary>

Istio routes traffic to pods matching the subset labels. If the canary DestinationRule has `version: v1.0.0` but the canary Deployment has `version: v1.1.0`, traffic won't reach the canary pods.

</details>

<details>
<summary>Hint 3: Nearly There</summary>

There are 4 bugs total: (1) weights sum, (2) version label mismatch, (3) missing resource limits, (4) missing readiness probe. The label mismatch is why requests might fail—traffic is being sent to non-existent pods.

</details>

### Solution

<details>
<summary>Click to reveal full solution</summary>

**Approach:**
Systematically review each configuration for consistency, resource safety, and operational readiness.

**Bug Analysis:**

| Bug | Location | Issue | Impact |
|-----|----------|-------|--------|
| 1 | VirtualService weights | 90 + 20 = 110, not 100 | 20% of traffic unaccounted or rejected |
| 2 | DestinationRule canary label | `v1.0.0` instead of `v1.1.0` | No pods match canary subset, 503 errors |
| 3 | Canary Deployment resources | No limits | Canary could starve stable pods |
| 4 | Canary Deployment probes | No readiness probe | Traffic before model loads, initial failures |

**Corrected Configurations:**

```yaml
# virtual-service.yaml (FIXED)
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: model-serving
spec:
  hosts:
  - model-serving
  http:
  - route:
    - destination:
        host: model-serving
        subset: stable
      weight: 90  # FIX: 90 + 10 = 100
    - destination:
        host: model-serving
        subset: canary
      weight: 10  # FIX: Changed from 20 to 10
---
# destination-rule.yaml (FIXED)
apiVersion: networking.istio.io/v1alpha3
kind: DestinationRule
metadata:
  name: model-serving
spec:
  host: model-serving
  subsets:
  - name: stable
    labels:
      version: v1.0.0
  - name: canary
    labels:
      version: v1.1.0  # FIX: Match canary deployment label
---
# deployment-canary.yaml (FIXED)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-serving-canary
spec:
  replicas: 1
  selector:
    matchLabels:
      app: model-serving
      version: v1.1.0  # FIX: Add version to selector for clarity
  template:
    metadata:
      labels:
        app: model-serving
        version: v1.1.0
    spec:
      containers:
      - name: model-server
        image: myregistry/model-server:v1.1.0
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:  # FIX: Add resource limits
            memory: "4Gi"
            cpu: "2"
        readinessProbe:  # FIX: Add readiness probe
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        livenessProbe:  # BONUS: Add liveness probe too
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 30
```

**Why These Fixes Work:**
1. **Weights summing to 100** ensures all traffic is properly distributed
2. **Matching labels** allows Istio to correctly route to canary pods
3. **Resource limits** prevent the canary from affecting stable performance
4. **Readiness probe** ensures traffic only reaches fully initialized pods

</details>

### Common Mistakes

- ❌ **Mistake:** Only checking VirtualService when debugging routing issues
  - **Why it happens:** Forgetting that DestinationRule defines the actual pod selection
  - **How to avoid:** Always verify the chain: VirtualService → DestinationRule → Pod labels

- ❌ **Mistake:** Copying stable deployment without updating all version references
  - **Why it happens:** Manual copy-paste errors
  - **How to avoid:** Use templating (Helm, Kustomize) with variables for versions

### Extension Challenge

Add retry and timeout policies to the VirtualService to handle transient failures during canary rollout. Configure 2 retries with exponential backoff and a 30-second overall timeout.

---

## Problem 5: Diagnose Production Monitoring Gap

**Type:** Challenge
**Concepts Practiced:** Monitoring, Data Drift, Alerting, Observability
**Estimated Time:** 20 minutes
**Prerequisites:** Understanding of ML metrics, basic statistics

### Problem Statement

A fraud detection model has been in production for 3 months. Recently, the business team reported a 15% increase in undetected fraud, but your monitoring dashboard shows no alerts. Review the current monitoring setup and design a comprehensive monitoring strategy.

**Current Monitoring (incomplete):**

```python
# Current metrics being collected
from prometheus_client import Counter, Histogram

PREDICTION_COUNTER = Counter(
    'model_predictions_total',
    'Total predictions made',
    ['model_version']
)

LATENCY_HISTOGRAM = Histogram(
    'model_latency_seconds',
    'Inference latency',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0]
)

# Current alerts
# - Alert if latency p99 > 500ms
# - Alert if error rate > 1%
```

**Available Data:**
- Labeled fraud outcomes (with 1-week delay)
- Input feature distributions from training
- Real-time prediction outputs

### Requirements

- [ ] Identify what monitoring gaps allowed the issue to go undetected
- [ ] Design metrics to detect similar issues in the future
- [ ] Write Prometheus metrics and alerting rules for the solution
- [ ] Specify what data drift tests should run and at what frequency

### Hints (Progressive)

<details>
<summary>Hint 1: Getting Started</summary>

The current monitoring only tracks infrastructure metrics (latency, errors). Consider what ML-specific metrics are missing: prediction distribution, feature distributions, model confidence, and actual performance.

</details>

<details>
<summary>Hint 2: Key Insight</summary>

A 15% increase in undetected fraud with no latency/error alerts suggests the model is running fine technically but making worse predictions. This points to either data drift (inputs changed) or concept drift (fraud patterns changed).

</details>

<details>
<summary>Hint 3: Nearly There</summary>

You need 4 types of monitoring: (1) prediction distribution (are outputs shifting?), (2) confidence distribution (is model less certain?), (3) feature drift (are inputs changing?), (4) ground truth feedback (actual accuracy when labels arrive). The 1-week label delay is workable for weekly accuracy reports.

</details>

### Solution

<details>
<summary>Click to reveal full solution</summary>

**Approach:**
Build a comprehensive monitoring stack covering all 5 layers: infrastructure, service, model, data quality, and business metrics.

**Gap Analysis:**

| Monitoring Layer | Current State | Gap | Impact |
|-----------------|---------------|-----|--------|
| Infrastructure | ✅ Latency, errors | None | - |
| Service | ❌ Missing | No throughput tracking | Can't correlate load with issues |
| Model | ❌ Missing | No prediction/confidence tracking | Can't detect model behavior change |
| Data Quality | ❌ Missing | No drift detection | Can't detect input distribution shift |
| Business | ❌ Missing | No ground truth feedback | Can't measure actual performance |

**Comprehensive Monitoring Solution:**

```python
from prometheus_client import Counter, Histogram, Gauge
import numpy as np
from scipy import stats

# ===== EXISTING (Infrastructure) =====
PREDICTION_COUNTER = Counter(
    'model_predictions_total',
    'Total predictions made',
    ['model_version']
)

LATENCY_HISTOGRAM = Histogram(
    'model_latency_seconds',
    'Inference latency',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0]
)

# ===== NEW: Model Metrics =====
PREDICTION_HISTOGRAM = Histogram(
    'model_prediction_score',
    'Fraud probability score distribution',
    ['model_version'],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
)

CONFIDENCE_HISTOGRAM = Histogram(
    'model_prediction_confidence',
    'Model confidence (max class probability)',
    ['model_version'],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
)

FRAUD_PREDICTED_RATE = Gauge(
    'model_fraud_predicted_rate',
    'Rolling rate of fraud predictions',
    ['model_version']
)

# ===== NEW: Data Quality Metrics =====
FEATURE_DRIFT_GAUGE = Gauge(
    'feature_drift_score',
    'KS statistic for feature drift',
    ['feature_name']
)

MISSING_VALUE_RATE = Gauge(
    'feature_missing_rate',
    'Missing value rate per feature',
    ['feature_name']
)

# ===== NEW: Business Metrics (Ground Truth) =====
WEEKLY_ACCURACY = Gauge(
    'model_weekly_accuracy',
    'Accuracy computed from delayed labels',
    ['model_version']
)

WEEKLY_RECALL = Gauge(
    'model_weekly_recall',
    'Fraud recall computed from delayed labels',
    ['model_version']
)

FALSE_NEGATIVE_RATE = Gauge(
    'model_false_negative_rate',
    'Rate of missed fraud (undetected)',
    ['model_version']
)


# ===== Drift Detection Implementation =====
class DriftDetector:
    def __init__(self, reference_data):
        self.reference = reference_data

    def check_drift(self, current_data, threshold=0.05):
        drift_report = {}
        for feature in self.reference.columns:
            stat, p_value = stats.ks_2samp(
                self.reference[feature],
                current_data[feature]
            )
            drift_detected = p_value < threshold
            drift_report[feature] = {
                'ks_statistic': stat,
                'p_value': p_value,
                'drift_detected': drift_detected
            }
            # Update Prometheus gauge
            FEATURE_DRIFT_GAUGE.labels(feature_name=feature).set(stat)
        return drift_report


# ===== Inference Instrumentation =====
def instrumented_predict(model, features):
    # Track feature statistics
    for feature_name, value in features.items():
        if value is None:
            MISSING_VALUE_RATE.labels(feature_name=feature_name).inc()

    # Get prediction
    with LATENCY_HISTOGRAM.time():
        prediction = model.predict(features)
        confidence = model.predict_proba(features).max()

    # Track prediction metrics
    PREDICTION_COUNTER.labels(model_version='v1.0.0').inc()
    PREDICTION_HISTOGRAM.labels(model_version='v1.0.0').observe(prediction)
    CONFIDENCE_HISTOGRAM.labels(model_version='v1.0.0').observe(confidence)

    return prediction, confidence
```

**Alerting Rules:**

```yaml
# prometheus-alerts.yaml
groups:
  - name: ml-model-alerts
    rules:
      # Existing infrastructure alerts
      - alert: HighLatency
        expr: histogram_quantile(0.99, model_latency_seconds) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Model latency p99 exceeds 500ms"

      # NEW: Model behavior alerts
      - alert: PredictionDistributionShift
        expr: |
          abs(
            avg_over_time(model_fraud_predicted_rate[1h]) -
            avg_over_time(model_fraud_predicted_rate[24h] offset 7d)
          ) > 0.05
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "Fraud prediction rate shifted >5% from last week"

      - alert: LowModelConfidence
        expr: histogram_quantile(0.5, model_prediction_confidence) < 0.7
        for: 1h
        labels:
          severity: warning
        annotations:
          summary: "Median prediction confidence below 70%"

      # NEW: Data drift alerts
      - alert: FeatureDrift
        expr: feature_drift_score > 0.15
        for: 1h
        labels:
          severity: critical
        annotations:
          summary: "Significant feature drift detected"

      # NEW: Business metric alerts
      - alert: AccuracyDegradation
        expr: model_weekly_accuracy < 0.90
        for: 1d
        labels:
          severity: critical
        annotations:
          summary: "Weekly accuracy dropped below 90%"

      - alert: HighFalseNegativeRate
        expr: model_false_negative_rate > 0.10
        for: 1d
        labels:
          severity: critical
        annotations:
          summary: "False negative rate exceeds 10%"
```

**Drift Detection Schedule:**

| Check | Frequency | Test | Threshold | Action |
|-------|-----------|------|-----------|--------|
| Feature drift | Hourly | KS test per feature | p < 0.05 | Alert + investigate |
| Prediction distribution | Hourly | Compare to 7-day baseline | >5% shift | Alert |
| Confidence distribution | Hourly | Median confidence | <70% | Alert |
| Accuracy (ground truth) | Weekly | Compare to baseline | <90% | Alert + retrain |
| Recall (ground truth) | Weekly | Compare to baseline | <85% | Critical alert |

**Why This Would Have Caught the Issue:**
1. **Feature drift detection** would show if fraudster behavior changed
2. **Prediction distribution monitoring** would show if model outputs shifted
3. **Confidence monitoring** might show increased uncertainty
4. **Weekly accuracy from ground truth** would definitively show degradation after 1-week label delay

</details>

### Common Mistakes

- ❌ **Mistake:** Only monitoring infrastructure metrics for ML systems
  - **Why it happens:** Applying traditional software monitoring practices
  - **How to avoid:** ML systems need all 5 layers: infrastructure, service, model, data quality, business

- ❌ **Mistake:** Not computing ground truth metrics even with delayed labels
  - **Why it happens:** Delayed labels seem "too late" to be useful
  - **How to avoid:** Weekly accuracy trends catch gradual degradation before it becomes severe

- ❌ **Mistake:** Setting drift thresholds too sensitive
  - **Why it happens:** Using textbook thresholds without calibration
  - **How to avoid:** Calibrate thresholds on historical data to balance signal vs. noise

### Extension Challenge

Design a feedback loop that automatically triggers model retraining when:
1. Feature drift exceeds threshold for 3+ consecutive hours
2. Weekly accuracy drops below 88%
3. Include a champion-challenger framework to validate the retrained model before promotion

---

## Summary

### Key Takeaways

1. **Model serialization and optimization** (ONNX, quantization) are prerequisites for efficient deployment—these directly impact latency SLAs
2. **Infrastructure configuration** (Kubernetes resources, probes) determines reliability—missing probes cause cascading failures
3. **CI/CD quality gates** are the last line of defense—they must include ML-specific checks, not just code tests
4. **Canary deployments** require careful label matching across multiple Kubernetes resources—a single mismatch breaks routing
5. **Comprehensive monitoring** covers 5 layers—the business team often detects ML issues before infrastructure alerts fire

### Next Steps

- If struggled with **Problem 1**: Review Lesson 11 Concepts 2-3 on serialization and optimization
- If struggled with **Problem 2**: Practice with Kubernetes documentation on Deployments and Services
- If struggled with **Problem 3**: Review Lesson 11 Concept 6 on CI/CD for ML
- If struggled with **Problem 4**: Study Istio traffic management documentation
- If struggled with **Problem 5**: Review Lesson 11 Concept 8 on monitoring and observability
- **Ready for assessment**: Proceed to quiz skill

---

*Generated from Lesson 11: Model Deployment and MLOps | Practice Problems Skill*
