# Lesson 11: Model Deployment and MLOps

**Topic:** Model Deployment and MLOps: From Training to Production at Scale
**Prerequisites:** Lesson 5 (Deep Learning), Lesson 8 (Neural Network Architectures)
**Estimated Study Time:** 3-4 hours
**Difficulty:** Advanced

---

## Learning Objectives

Upon completion of this lesson, learners will be able to:

1. **Design** end-to-end ML pipelines from data ingestion through model serving
2. **Implement** model serving strategies including REST APIs, batch inference, and edge deployment
3. **Apply** model optimization techniques (quantization, pruning, distillation) for production
4. **Evaluate** monitoring and observability strategies for detecting model drift and degradation
5. **Architect** CI/CD pipelines for continuous training and deployment of ML models

---

## Introduction

Training a machine learning model is only the beginning—the real challenge lies in deploying it reliably to production where it delivers value. MLOps (Machine Learning Operations) bridges the gap between ML development and operations, applying DevOps principles to the unique challenges of ML systems: data dependencies, model versioning, continuous training, and performance monitoring.

Unlike traditional software where code is the primary artifact, ML systems have three interacting components: code, data, and models. Changes to any of these can affect system behavior, requiring specialized tooling for versioning, testing, and deployment. Production ML systems must handle model serving at scale, monitor for data drift, enable A/B testing, and support rapid iteration.

This lesson covers the complete MLOps lifecycle: from model serialization and optimization through serving infrastructure and monitoring, providing the foundation for deploying ML systems that are reliable, scalable, and maintainable.

---

## Core Concepts

### Concept 1: The ML Lifecycle and MLOps Principles

MLOps extends DevOps practices to address the unique challenges of machine learning systems.

**Traditional Software vs. ML Systems:**

| Aspect | Traditional Software | ML Systems |
|--------|---------------------|------------|
| Primary artifact | Code | Code + Data + Model |
| Testing | Unit, integration, E2E | + Data validation, model validation |
| Versioning | Code versions | Code + Data + Model versions |
| Deployment | Deploy code | Deploy model + inference code |
| Monitoring | Latency, errors | + Prediction quality, data drift |

**MLOps Lifecycle:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MLOps Lifecycle                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │   Data   │───▶│ Feature  │───▶│  Model   │───▶│  Model   │      │
│  │ Ingestion│    │Engineering│    │ Training │    │Evaluation│      │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘      │
│       │                                               │              │
│       │              ┌─────────────────┐              │              │
│       │              │                 │              │              │
│       ▼              ▼                 │              ▼              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │   Data   │    │  Model   │◀───│ Continuous│◀───│  Model   │      │
│  │Validation│    │ Serving  │    │Monitoring │    │ Registry │      │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘      │
│                       │                 │                            │
│                       └────────────────┘                            │
│                         Feedback Loop                                │
└─────────────────────────────────────────────────────────────────────┘
```

**MLOps Maturity Levels:**

| Level | Description | Characteristics |
|-------|-------------|-----------------|
| **Level 0** | Manual | Jupyter notebooks, manual deployment, no automation |
| **Level 1** | ML Pipeline | Automated training pipeline, manual deployment |
| **Level 2** | CI/CD for ML | Automated testing, deployment, basic monitoring |
| **Level 3** | Full MLOps | Continuous training, A/B testing, automated retraining |

**Key MLOps Principles:**

```
1. Reproducibility
   - Version control for code, data, and models
   - Deterministic training (fixed seeds, pinned dependencies)
   - Environment reproducibility (containers)

2. Automation
   - Automated pipelines for training, validation, deployment
   - Trigger-based retraining (schedule, drift detection)
   - Automated rollback on degradation

3. Continuous Monitoring
   - Model performance metrics
   - Data distribution monitoring
   - System health (latency, throughput, errors)

4. Collaboration
   - Experiment tracking and sharing
   - Model registry for versioning
   - Documentation and lineage tracking
```

---

### Concept 2: Model Serialization and Formats

Converting trained models to portable formats is the first step toward deployment.

**Common Model Formats:**

| Format | Framework | Use Case | Characteristics |
|--------|-----------|----------|-----------------|
| **SavedModel** | TensorFlow | Production serving | Complete graph + weights + signatures |
| **TorchScript** | PyTorch | Production/mobile | JIT-compiled, Python-free |
| **ONNX** | Cross-framework | Interoperability | Open standard, wide runtime support |
| **Pickle** | Python | Quick prototyping | Python-only, security risks |
| **HDF5** | Keras | Legacy Keras | Weights + architecture |

**TensorFlow SavedModel:**

```python
import tensorflow as tf

# Save model
model.save('saved_model/my_model')

# Directory structure:
# saved_model/my_model/
#   ├── assets/
#   ├── saved_model.pb        # Graph definition
#   └── variables/
#       ├── variables.data-00000-of-00001
#       └── variables.index

# Load model
loaded_model = tf.saved_model.load('saved_model/my_model')

# Serving signature
infer = loaded_model.signatures['serving_default']
output = infer(input_tensor)
```

**PyTorch TorchScript:**

```python
import torch

# Method 1: Tracing (for models with fixed control flow)
example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)
traced_model.save('model.pt')

# Method 2: Scripting (for models with dynamic control flow)
scripted_model = torch.jit.script(model)
scripted_model.save('model.pt')

# Load and run
loaded = torch.jit.load('model.pt')
output = loaded(input_tensor)
```

**ONNX (Open Neural Network Exchange):**

```python
import torch.onnx

# Export PyTorch to ONNX
torch.onnx.export(
    model,
    example_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}}  # Dynamic batching
)

# Run with ONNX Runtime
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
output = session.run(None, {'input': input_array})
```

**Model Serialization Best Practices:**

```
1. Include metadata
   - Model version
   - Training date
   - Input/output specifications
   - Performance metrics

2. Validate after serialization
   - Compare outputs: original vs. serialized
   - Test edge cases
   - Verify input preprocessing is included

3. Version artifacts
   - Use content-addressable storage (hash-based)
   - Link to training code and data versions
   - Store in model registry
```

---

### Concept 3: Model Optimization for Deployment

Production models often require optimization to meet latency, memory, and power constraints.

**Optimization Techniques Overview:**

| Technique | Size Reduction | Speed Improvement | Accuracy Impact |
|-----------|---------------|-------------------|-----------------|
| Quantization (INT8) | 4× | 2-4× | 0.5-2% drop |
| Pruning (50%) | 2× | 1.5-2× | 0.5-1% drop |
| Knowledge Distillation | Variable | Variable | Can improve |
| Layer Fusion | 1× | 1.2-1.5× | None |

**Quantization:**

```
Quantization reduces precision from FP32 to INT8/INT4

FP32: 32 bits per weight → INT8: 8 bits per weight
Memory: 4× reduction
Computation: INT8 ops are faster on most hardware

Types:
├── Post-Training Quantization (PTQ)
│   - No retraining needed
│   - Uses calibration dataset
│   - Quick but may lose accuracy
│
└── Quantization-Aware Training (QAT)
    - Simulates quantization during training
    - Model learns to be robust to reduced precision
    - Better accuracy, requires training
```

**Quantization Example (PyTorch):**

```python
import torch.quantization

# Post-Training Dynamic Quantization
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.LSTM},
    dtype=torch.qint8
)

# Post-Training Static Quantization
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)

# Calibrate with representative data
for data in calibration_loader:
    model(data)

torch.quantization.convert(model, inplace=True)
```

**Pruning:**

```
Pruning removes unnecessary weights/neurons

Types:
├── Unstructured Pruning
│   - Remove individual weights (sparse matrix)
│   - Higher compression, needs sparse hardware
│
├── Structured Pruning
│   - Remove entire channels/filters
│   - Dense matrix, works on any hardware
│
└── Magnitude-Based Pruning
    - Remove weights with smallest absolute values
    - Simple and effective
```

**Pruning Example:**

```python
import torch.nn.utils.prune as prune

# Prune 30% of weights by magnitude
prune.l1_unstructured(model.fc1, name='weight', amount=0.3)

# Prune entire channels
prune.ln_structured(model.conv1, name='weight', amount=0.2, n=2, dim=0)

# Make pruning permanent
prune.remove(model.fc1, 'weight')
```

**Knowledge Distillation:**

```
Train small "student" model to mimic large "teacher" model

Loss = α × CE(student_logits, labels) +
       (1-α) × KL(student_soft, teacher_soft)

Where:
- student_soft = softmax(student_logits / T)
- teacher_soft = softmax(teacher_logits / T)
- T = temperature (higher = softer distribution)

Benefits:
- Student learns "dark knowledge" from teacher
- Soft labels provide more information than hard labels
- Can achieve teacher-like performance with fewer parameters
```

**Hardware-Specific Optimization:**

| Hardware | Optimization | Tool |
|----------|--------------|------|
| NVIDIA GPU | TensorRT | FP16/INT8, layer fusion, kernel auto-tuning |
| Intel CPU | OpenVINO | INT8 quantization, graph optimization |
| Mobile | TFLite, CoreML | Quantization, delegation to accelerators |
| Edge | ONNX Runtime | Cross-platform inference optimization |

---

### Concept 4: Model Serving Architectures

Model serving infrastructure must balance latency, throughput, and cost.

**Serving Patterns:**

```
1. Online/Real-time Serving
   - Synchronous request-response
   - Low latency requirement (< 100ms)
   - Examples: Search ranking, recommendation, fraud detection

2. Batch Inference
   - Process large datasets offline
   - Throughput-optimized
   - Examples: Nightly scoring, report generation

3. Streaming Inference
   - Process continuous data streams
   - Near real-time with some latency tolerance
   - Examples: IoT sensor analysis, log anomaly detection

4. Edge Inference
   - Run on device (mobile, IoT, browser)
   - No network latency, privacy-preserving
   - Examples: Mobile apps, autonomous vehicles
```

**REST API Serving Architecture:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    Model Serving Architecture                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Client Request                                                 │
│        │                                                         │
│        ▼                                                         │
│   ┌─────────┐    ┌─────────┐    ┌─────────────────────┐        │
│   │  Load   │───▶│   API   │───▶│   Model Server      │        │
│   │Balancer │    │ Gateway │    │  ┌─────────────┐    │        │
│   └─────────┘    └─────────┘    │  │   Model A   │    │        │
│                       │          │  │  (v1.2.3)   │    │        │
│                       │          │  └─────────────┘    │        │
│                       │          │  ┌─────────────┐    │        │
│                       │          │  │   Model B   │    │        │
│                       │          │  │  (v2.0.1)   │    │        │
│                       │          │  └─────────────┘    │        │
│                       │          └─────────────────────┘        │
│                       │                    │                     │
│                       ▼                    ▼                     │
│              ┌─────────────┐    ┌─────────────────┐             │
│              │   Metrics   │    │  Feature Store  │             │
│              │  Collector  │    │   (optional)    │             │
│              └─────────────┘    └─────────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

**Model Server Options:**

| Server | Framework | Features |
|--------|-----------|----------|
| **TensorFlow Serving** | TensorFlow | gRPC/REST, batching, versioning |
| **TorchServe** | PyTorch | REST, model management, metrics |
| **Triton** | Multi-framework | Dynamic batching, ensemble, GPU sharing |
| **Seldon Core** | Kubernetes-native | Canary, A/B testing, explainability |
| **BentoML** | Framework-agnostic | Easy packaging, cloud deployment |

**FastAPI Model Server Example:**

```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI()

# Load model at startup
model = torch.jit.load("model.pt")
model.eval()

class PredictionRequest(BaseModel):
    features: list[float]

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    # Preprocess
    tensor = torch.tensor(request.features).unsqueeze(0)

    # Inference
    with torch.no_grad():
        output = model(tensor)
        prob = torch.softmax(output, dim=1)

    # Postprocess
    prediction = output.argmax().item()
    confidence = prob.max().item()

    return PredictionResponse(
        prediction=prediction,
        confidence=confidence
    )

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

**Batching Strategies:**

```
Dynamic Batching:
  - Collect requests over time window
  - Batch together for GPU efficiency
  - Trade latency for throughput

Configuration:
  max_batch_size: 32
  batch_timeout_ms: 10  # Wait up to 10ms for more requests

Throughput improvement: 3-10× depending on model
Latency impact: +10-50ms (configurable)
```

---

### Concept 5: Containerization and Orchestration

Containers ensure reproducible deployments across environments.

**Docker for ML:**

```dockerfile
# Dockerfile for model serving
FROM python:3.10-slim

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and code
COPY model/ /app/model/
COPY src/ /app/src/

WORKDIR /app

# Environment variables
ENV MODEL_PATH=/app/model/model.pt
ENV PORT=8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:${PORT}/health || exit 1

# Run server
EXPOSE ${PORT}
CMD ["python", "src/server.py"]
```

**Multi-Stage Build for Smaller Images:**

```dockerfile
# Build stage
FROM python:3.10 AS builder
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# Production stage
FROM python:3.10-slim
COPY --from=builder /wheels /wheels
RUN pip install --no-cache /wheels/*

COPY model/ /app/model/
COPY src/ /app/src/

WORKDIR /app
CMD ["python", "src/server.py"]
```

**Kubernetes Deployment:**

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-serving
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
        image: myregistry/model-server:v1.2.3
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
            nvidia.com/gpu: 1
          limits:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: 1
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 30
---
apiVersion: v1
kind: Service
metadata:
  name: model-serving
spec:
  selector:
    app: model-serving
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
```

**Horizontal Pod Autoscaler:**

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: model-serving-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-serving
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: 100
```

---

### Concept 6: CI/CD for Machine Learning

ML CI/CD extends traditional pipelines with data validation, model testing, and staged rollouts.

**ML Pipeline Stages:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ML CI/CD Pipeline                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐          │
│  │  Code   │───▶│  Data   │───▶│ Model   │───▶│ Model   │          │
│  │  Tests  │    │Validation│    │Training │    │Evaluation│          │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘          │
│       │              │              │              │                 │
│       ▼              ▼              ▼              ▼                 │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐          │
│  │  Lint   │    │ Schema  │    │Experiment│    │ Quality  │          │
│  │  Type   │    │  Check  │    │ Tracking │    │  Gates   │          │
│  │  Check  │    │  Stats  │    │          │    │          │          │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘          │
│                                                     │                │
│                      Pass Quality Gates?            │                │
│                              │                      │                │
│              ┌───────────────┼───────────────┐     │                │
│              │               │               │     │                │
│              ▼               ▼               ▼     ▼                │
│         ┌─────────┐    ┌─────────┐    ┌─────────────┐              │
│         │ Staging │───▶│  Canary │───▶│ Production  │              │
│         │ Deploy  │    │  Deploy │    │   Rollout   │              │
│         └─────────┘    └─────────┘    └─────────────┘              │
│                              │                                       │
│                              ▼                                       │
│                       ┌─────────────┐                               │
│                       │  Monitoring │                               │
│                       │  & Rollback │                               │
│                       └─────────────┘                               │
└─────────────────────────────────────────────────────────────────────┘
```

**Data Validation:**

```python
import great_expectations as ge

# Define data expectations
expectation_suite = ge.core.ExpectationSuite(
    expectation_suite_name="training_data_validation"
)

# Add expectations
expectation_suite.add_expectation(
    ge.core.ExpectationConfiguration(
        expectation_type="expect_column_values_to_not_be_null",
        kwargs={"column": "feature_1"}
    )
)

expectation_suite.add_expectation(
    ge.core.ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={"column": "feature_1", "min_value": 0, "max_value": 100}
    )
)

# Validate data
validation_result = ge.validate(dataframe, expectation_suite)
if not validation_result.success:
    raise ValueError("Data validation failed")
```

**Model Quality Gates:**

```python
def evaluate_model_quality(model, test_data, baseline_metrics):
    """
    Quality gates that must pass before deployment
    """
    metrics = evaluate_model(model, test_data)

    gates = {
        # Accuracy should not drop more than 1%
        "accuracy_gate": metrics['accuracy'] >= baseline_metrics['accuracy'] - 0.01,

        # Latency should be under 100ms p99
        "latency_gate": metrics['latency_p99'] < 100,

        # No bias increase
        "fairness_gate": metrics['demographic_parity'] >= 0.8,

        # Minimum sample size for statistical significance
        "sample_size_gate": metrics['test_samples'] >= 1000,
    }

    return all(gates.values()), gates
```

**GitHub Actions ML Pipeline:**

```yaml
# .github/workflows/ml-pipeline.yml
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
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run unit tests
        run: pytest tests/unit/

      - name: Run data validation
        run: python scripts/validate_data.py

  train:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Train model
        run: python scripts/train.py

      - name: Evaluate model
        run: python scripts/evaluate.py

      - name: Upload model artifact
        uses: actions/upload-artifact@v3
        with:
          name: model
          path: outputs/model/

  deploy-staging:
    needs: train
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - name: Deploy to staging
        run: |
          kubectl apply -f k8s/staging/
          kubectl rollout status deployment/model-serving

  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment: production
    steps:
      - name: Deploy canary (10%)
        run: |
          kubectl apply -f k8s/production/canary.yaml

      - name: Monitor canary
        run: python scripts/monitor_canary.py --duration=30m

      - name: Full rollout
        run: kubectl apply -f k8s/production/deployment.yaml
```

---

### Concept 7: Model Registry and Versioning

A model registry provides centralized management of model artifacts and metadata.

**Model Registry Concepts:**

```
Model Registry Structure:
├── Model (logical entity)
│   ├── Version 1.0.0
│   │   ├── Artifacts (model files)
│   │   ├── Metrics (accuracy, F1, etc.)
│   │   ├── Parameters (hyperparameters)
│   │   ├── Tags (production, staging)
│   │   └── Lineage (training data, code commit)
│   │
│   ├── Version 1.1.0
│   │   └── ...
│   │
│   └── Version 2.0.0
│       └── ...
│
└── Another Model
    └── ...
```

**MLflow Model Registry Example:**

```python
import mlflow
from mlflow.tracking import MlflowClient

# Log model during training
with mlflow.start_run():
    # Train model
    model = train_model(data)

    # Log parameters
    mlflow.log_params({
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100
    })

    # Log metrics
    mlflow.log_metrics({
        "accuracy": 0.95,
        "f1": 0.93,
        "auc": 0.97
    })

    # Log model
    mlflow.pytorch.log_model(
        model,
        "model",
        registered_model_name="fraud_detection"
    )

# Promote model to production
client = MlflowClient()
client.transition_model_version_stage(
    name="fraud_detection",
    version=3,
    stage="Production"
)

# Load production model
model = mlflow.pytorch.load_model(
    "models:/fraud_detection/Production"
)
```

**Model Versioning Strategies:**

```
Semantic Versioning for Models:
  MAJOR.MINOR.PATCH

  MAJOR: Breaking changes (input/output schema change)
  MINOR: New features (additional outputs, improved accuracy)
  PATCH: Bug fixes, minor improvements

Examples:
  1.0.0 → 1.0.1: Fixed preprocessing bug
  1.0.1 → 1.1.0: Added confidence scores to output
  1.1.0 → 2.0.0: Changed input format from JSON to protobuf
```

**Model Lineage:**

```python
# Track full lineage
lineage = {
    "model_version": "2.1.0",
    "training_data": {
        "source": "s3://data/training/2024-01/",
        "version": "abc123",
        "rows": 1_000_000,
        "features": ["f1", "f2", "f3"]
    },
    "code": {
        "repo": "github.com/company/ml-models",
        "commit": "def456",
        "branch": "main"
    },
    "training": {
        "started": "2024-01-15T10:00:00Z",
        "completed": "2024-01-15T14:30:00Z",
        "environment": "gpu-cluster-prod",
        "gpu_type": "A100",
        "framework": "pytorch==2.0"
    },
    "evaluation": {
        "test_accuracy": 0.952,
        "test_f1": 0.941,
        "fairness_score": 0.89
    }
}
```

---

### Concept 8: Monitoring and Observability

Production ML systems require comprehensive monitoring to detect issues and maintain quality.

**Monitoring Layers:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                      ML Monitoring Stack                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Layer 1: Infrastructure Metrics                                     │
│  ├── CPU/GPU utilization                                            │
│  ├── Memory usage                                                   │
│  ├── Network I/O                                                    │
│  └── Disk usage                                                     │
│                                                                      │
│  Layer 2: Service Metrics                                           │
│  ├── Request latency (p50, p95, p99)                               │
│  ├── Throughput (requests/second)                                  │
│  ├── Error rate                                                    │
│  └── Queue depth                                                   │
│                                                                      │
│  Layer 3: Model Metrics                                             │
│  ├── Prediction distribution                                       │
│  ├── Confidence scores                                             │
│  ├── Feature value distributions                                   │
│  └── Model-specific metrics (accuracy proxy)                       │
│                                                                      │
│  Layer 4: Data Quality Metrics                                      │
│  ├── Input data distribution                                       │
│  ├── Missing value rates                                           │
│  ├── Schema violations                                             │
│  └── Data drift detection                                          │
│                                                                      │
│  Layer 5: Business Metrics                                          │
│  ├── Conversion rate (if applicable)                               │
│  ├── User engagement                                               │
│  └── Revenue impact                                                │
└─────────────────────────────────────────────────────────────────────┘
```

**Data Drift Detection:**

```python
from scipy import stats
import numpy as np

def detect_drift(reference_data, current_data, threshold=0.05):
    """
    Detect data drift using statistical tests
    """
    drift_report = {}

    for feature in reference_data.columns:
        # Kolmogorov-Smirnov test for continuous features
        statistic, p_value = stats.ks_2samp(
            reference_data[feature],
            current_data[feature]
        )

        drift_report[feature] = {
            "ks_statistic": statistic,
            "p_value": p_value,
            "drift_detected": p_value < threshold
        }

    return drift_report

# Population Stability Index (PSI) for categorical
def calculate_psi(expected, actual, bins=10):
    """
    PSI < 0.1: No significant change
    PSI 0.1-0.2: Moderate change
    PSI > 0.2: Significant change
    """
    expected_percents = np.histogram(expected, bins=bins)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=bins)[0] / len(actual)

    psi = np.sum(
        (actual_percents - expected_percents) *
        np.log(actual_percents / expected_percents + 1e-10)
    )
    return psi
```

**Prometheus Metrics Example:**

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
PREDICTION_COUNTER = Counter(
    'model_predictions_total',
    'Total predictions made',
    ['model_version', 'prediction_class']
)

LATENCY_HISTOGRAM = Histogram(
    'model_inference_latency_seconds',
    'Model inference latency',
    ['model_version'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

CONFIDENCE_HISTOGRAM = Histogram(
    'model_prediction_confidence',
    'Prediction confidence distribution',
    ['model_version'],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
)

DRIFT_GAUGE = Gauge(
    'feature_drift_score',
    'Data drift score per feature',
    ['feature_name']
)

# Use in inference
@LATENCY_HISTOGRAM.labels(model_version='1.2.3').time()
def predict(input_data):
    prediction = model(input_data)
    confidence = prediction.max()

    PREDICTION_COUNTER.labels(
        model_version='1.2.3',
        prediction_class=str(prediction.argmax())
    ).inc()

    CONFIDENCE_HISTOGRAM.labels(model_version='1.2.3').observe(confidence)

    return prediction
```

**Alerting Rules:**

```yaml
# prometheus-alerts.yaml
groups:
  - name: ml-model-alerts
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.99, model_inference_latency_seconds) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Model latency p99 exceeds 500ms"

      - alert: LowConfidence
        expr: histogram_quantile(0.5, model_prediction_confidence) < 0.7
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Median prediction confidence below 70%"

      - alert: DataDrift
        expr: feature_drift_score > 0.2
        for: 1h
        labels:
          severity: critical
        annotations:
          summary: "Significant data drift detected"

      - alert: ErrorRateHigh
        expr: rate(model_predictions_total{status="error"}[5m]) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Model error rate exceeds 1%"
```

---

### Concept 9: A/B Testing and Canary Deployments

Controlled rollouts enable safe model updates and performance comparison.

**Deployment Strategies:**

```
1. Blue-Green Deployment
   ┌─────────────┐     ┌─────────────┐
   │  Blue (Old) │     │ Green (New) │
   │   100%      │ →   │    100%     │
   └─────────────┘     └─────────────┘

   - Instant switch
   - Easy rollback
   - Requires 2× resources

2. Canary Deployment
   ┌─────────────┐     ┌─────────────┐
   │  Old Model  │     │ New Model   │
   │    90%      │     │    10%      │
   └─────────────┘     └─────────────┘

   - Gradual rollout
   - Monitor before full deployment
   - Lower risk

3. A/B Testing
   ┌─────────────┐     ┌─────────────┐
   │  Model A    │     │  Model B    │
   │   Control   │     │  Treatment  │
   │    50%      │     │    50%      │
   └─────────────┘     └─────────────┘

   - Statistical comparison
   - User-level assignment
   - Measures business impact
```

**Traffic Splitting with Istio:**

```yaml
# virtual-service.yaml
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: model-serving
spec:
  hosts:
  - model-serving
  http:
  - match:
    - headers:
        canary:
          exact: "true"
    route:
    - destination:
        host: model-serving
        subset: canary
  - route:
    - destination:
        host: model-serving
        subset: stable
      weight: 90
    - destination:
        host: model-serving
        subset: canary
      weight: 10
---
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
      version: v1.1.0
```

**A/B Test Analysis:**

```python
from scipy import stats
import numpy as np

def analyze_ab_test(control_conversions, control_total,
                    treatment_conversions, treatment_total,
                    confidence_level=0.95):
    """
    Analyze A/B test results with statistical significance
    """
    # Conversion rates
    control_rate = control_conversions / control_total
    treatment_rate = treatment_conversions / treatment_total

    # Pooled proportion for z-test
    pooled = (control_conversions + treatment_conversions) / \
             (control_total + treatment_total)

    # Standard error
    se = np.sqrt(pooled * (1 - pooled) *
                 (1/control_total + 1/treatment_total))

    # Z-score
    z_score = (treatment_rate - control_rate) / se

    # P-value (two-tailed)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    # Relative lift
    lift = (treatment_rate - control_rate) / control_rate

    # Confidence interval
    z_critical = stats.norm.ppf((1 + confidence_level) / 2)
    ci_lower = (treatment_rate - control_rate) - z_critical * se
    ci_upper = (treatment_rate - control_rate) + z_critical * se

    return {
        "control_rate": control_rate,
        "treatment_rate": treatment_rate,
        "relative_lift": lift,
        "p_value": p_value,
        "significant": p_value < (1 - confidence_level),
        "confidence_interval": (ci_lower, ci_upper)
    }
```

**Automated Canary Analysis:**

```python
def analyze_canary(stable_metrics, canary_metrics, thresholds):
    """
    Automated canary analysis for promotion decision
    """
    analysis = {}

    # Latency comparison
    latency_increase = (canary_metrics['latency_p99'] -
                        stable_metrics['latency_p99']) / \
                       stable_metrics['latency_p99']
    analysis['latency'] = {
        'passed': latency_increase < thresholds['max_latency_increase'],
        'increase': latency_increase
    }

    # Error rate comparison
    error_increase = canary_metrics['error_rate'] - \
                     stable_metrics['error_rate']
    analysis['error_rate'] = {
        'passed': error_increase < thresholds['max_error_increase'],
        'increase': error_increase
    }

    # Business metric comparison
    conversion_lift = (canary_metrics['conversion'] -
                       stable_metrics['conversion']) / \
                      stable_metrics['conversion']
    analysis['conversion'] = {
        'passed': conversion_lift > thresholds['min_conversion_lift'],
        'lift': conversion_lift
    }

    # Overall decision
    analysis['promote'] = all(
        check['passed'] for check in analysis.values()
    )

    return analysis
```

---

### Concept 10: Feature Stores and Data Management

Feature stores provide centralized feature management for training and serving consistency.

**Feature Store Architecture:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Feature Store Architecture                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Data Sources                                                        │
│  ├── Databases                                                      │
│  ├── Data Warehouses                                                │
│  ├── Streaming (Kafka)                                              │
│  └── APIs                                                           │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Feature Engineering Pipelines                    │   │
│  │  ├── Batch transformations (Spark, dbt)                      │   │
│  │  └── Streaming transformations (Flink, Spark Streaming)      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                     Feature Store                             │   │
│  │  ┌─────────────────┐     ┌─────────────────┐                │   │
│  │  │  Offline Store  │     │  Online Store   │                │   │
│  │  │  (Historical)   │     │  (Low Latency)  │                │   │
│  │  │  - Parquet      │     │  - Redis        │                │   │
│  │  │  - Delta Lake   │     │  - DynamoDB     │                │   │
│  │  └────────┬────────┘     └────────┬────────┘                │   │
│  │           │                       │                          │   │
│  │           │    ┌──────────────────┘                          │   │
│  │           │    │                                             │   │
│  │           ▼    ▼                                             │   │
│  │  ┌─────────────────────────────────────────────────────┐    │   │
│  │  │              Feature Registry (Metadata)             │    │   │
│  │  │  - Feature definitions                               │    │   │
│  │  │  - Schemas and types                                 │    │   │
│  │  │  - Lineage and documentation                         │    │   │
│  │  └─────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│           │                       │                                  │
│           ▼                       ▼                                  │
│      ┌─────────┐            ┌─────────┐                            │
│      │Training │            │ Online  │                            │
│      │ (Batch) │            │Inference│                            │
│      └─────────┘            └─────────┘                            │
└─────────────────────────────────────────────────────────────────────┘
```

**Feast Feature Store Example:**

```python
from feast import Entity, Feature, FeatureView, FileSource, ValueType
from feast.feature_store import FeatureStore
from datetime import timedelta

# Define entity
customer = Entity(
    name="customer_id",
    value_type=ValueType.INT64,
    description="Customer identifier"
)

# Define feature source
customer_features_source = FileSource(
    path="data/customer_features.parquet",
    timestamp_field="event_timestamp"
)

# Define feature view
customer_features = FeatureView(
    name="customer_features",
    entities=["customer_id"],
    ttl=timedelta(days=1),
    features=[
        Feature(name="total_purchases", dtype=ValueType.FLOAT),
        Feature(name="avg_purchase_value", dtype=ValueType.FLOAT),
        Feature(name="days_since_last_purchase", dtype=ValueType.INT64),
        Feature(name="purchase_frequency", dtype=ValueType.FLOAT),
    ],
    online=True,
    source=customer_features_source
)

# Use feature store
store = FeatureStore(repo_path=".")

# Training: Get historical features
training_df = store.get_historical_features(
    entity_df=entity_df,  # DataFrame with customer_id and event_timestamp
    features=[
        "customer_features:total_purchases",
        "customer_features:avg_purchase_value",
    ]
).to_df()

# Serving: Get online features
feature_vector = store.get_online_features(
    features=[
        "customer_features:total_purchases",
        "customer_features:avg_purchase_value",
    ],
    entity_rows=[{"customer_id": 12345}]
).to_dict()
```

**Training-Serving Skew Prevention:**

```
Training-Serving Skew Sources:
1. Feature computation differences
   - Different code paths for training vs. serving
   - Solution: Single feature definition used by both

2. Data distribution differences
   - Training on historical data, serving on current
   - Solution: Monitor distribution shifts

3. Time travel issues
   - Using future information during training
   - Solution: Point-in-time joins with timestamp awareness

4. Missing features at serving time
   - Feature not available with low latency
   - Solution: Feature availability SLAs, fallback values

Best Practices:
- Same feature computation code for training and serving
- Materialize features to online store for serving
- Log serving features for validation against training
- Monitor feature distributions in both pipelines
```

---

## Summary

Model Deployment and MLOps bridge the gap between ML development and production systems. The **MLOps lifecycle** (Concept 1) extends DevOps principles to handle the unique challenges of ML: versioning data/models, continuous training, and monitoring model quality. **Model serialization** (Concept 2) converts trained models to portable formats (SavedModel, TorchScript, ONNX) for deployment.

**Model optimization** (Concept 3) through quantization, pruning, and distillation reduces model size and improves inference speed while maintaining accuracy. **Model serving architectures** (Concept 4) provide the infrastructure for real-time, batch, and edge inference with appropriate latency and throughput characteristics.

**Containerization** (Concept 5) with Docker and Kubernetes ensures reproducible deployments and enables horizontal scaling. **CI/CD for ML** (Concept 6) automates testing, validation, and deployment with quality gates specific to ML systems. **Model registries** (Concept 7) provide centralized versioning and lineage tracking.

**Monitoring and observability** (Concept 8) detect data drift, model degradation, and system issues through comprehensive metrics collection. **A/B testing and canary deployments** (Concept 9) enable safe, data-driven model updates. **Feature stores** (Concept 10) ensure consistency between training and serving while managing feature engineering at scale.

Together, these practices enable ML systems that are reliable, scalable, and maintainable—moving beyond prototype notebooks to production-grade machine learning.

---

## References

- Sculley, D., et al. (2015). "Hidden Technical Debt in Machine Learning Systems" (Google)
- Amershi, S., et al. (2019). "Software Engineering for Machine Learning" (Microsoft)
- Polyzotis, N., et al. (2018). "Data Management Challenges in Production Machine Learning" (Google)
- MLflow Documentation - https://mlflow.org/docs/latest/index.html
- Kubeflow Documentation - https://www.kubeflow.org/docs/
- Feast Documentation - https://docs.feast.dev/

---

*Generated for Model Deployment and MLOps | Lesson Skill*
