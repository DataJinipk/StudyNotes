# Practice Problems: Lesson 3 - Large Language Models

**Source:** Lessons/Lesson_3.md
**Subject Area:** AI Learning - Large Language Models: Architecture, Training, and Capabilities
**Date Generated:** 2026-01-08
**Total Problems:** 5
**Distribution:** 1 Warm-Up | 2 Skill-Builder | 1 Challenge | 1 Debug/Fix

---

## Problem Overview

| # | Type | Concept Focus | Difficulty | Estimated Time |
|---|------|---------------|------------|----------------|
| 1 | Warm-Up | Self-Attention Computation | Low | 10-15 min |
| 2 | Skill-Builder | Training Pipeline Analysis | Medium | 20-25 min |
| 3 | Skill-Builder | Inference Optimization | Medium | 20-25 min |
| 4 | Challenge | Production System Design | High | 40-50 min |
| 5 | Debug/Fix | Hallucination Diagnosis | Medium | 25-30 min |

---

## Problem 1: Warm-Up
### Self-Attention Computation

**Concept:** Self-Attention Mechanism (Core Concept 2)
**Cognitive Level:** Apply
**Prerequisites:** Understanding of Q/K/V projections

---

**Problem Statement:**

Given the following simplified self-attention scenario, compute the attention weights and output.

**Setup:**
- Sequence length: 3 tokens ["The", "cat", "sat"]
- Embedding dimension: 4
- Attention dimension (d_k): 4

**Given Matrices:**
```
Query (Q):
Token 0 ("The"): [1, 0, 1, 0]
Token 1 ("cat"): [0, 1, 0, 1]
Token 2 ("sat"): [1, 1, 0, 0]

Key (K):
Token 0 ("The"): [1, 0, 0, 1]
Token 1 ("cat"): [0, 1, 1, 0]
Token 2 ("sat"): [1, 0, 1, 0]

Value (V):
Token 0 ("The"): [0.1, 0.2, 0.3, 0.4]
Token 1 ("cat"): [0.5, 0.6, 0.7, 0.8]
Token 2 ("sat"): [0.9, 1.0, 1.1, 1.2]
```

**Tasks:**
1. Compute the raw attention scores (Q · K^T) for Token 2 ("sat") attending to all tokens
2. Apply the scaling factor (√d_k = 2)
3. Apply softmax to get attention weights
4. Compute the output vector for Token 2

**Hint:** For softmax, use e^x / Σe^x. You may round to 2 decimal places.

---

**Progressive Hints:**

<details>
<summary>Hint 1 (Conceptual)</summary>

For Token 2's attention, you need:
- Q[2] · K[0]^T = dot product of Token 2's Query with Token 0's Key
- Q[2] · K[1]^T = dot product of Token 2's Query with Token 1's Key
- Q[2] · K[2]^T = dot product of Token 2's Query with Token 2's Key

Remember: dot product = sum of element-wise multiplication
</details>

<details>
<summary>Hint 2 (Procedural)</summary>

Step-by-step for raw scores:
- Q[2] = [1, 1, 0, 0]
- K[0] = [1, 0, 0, 1] → Q[2]·K[0] = 1×1 + 1×0 + 0×0 + 0×1 = 1
- Continue for K[1] and K[2]...
- Then divide all scores by √4 = 2
</details>

<details>
<summary>Hint 3 (Solution Direction)</summary>

Raw scores for Token 2: [1, 1, 1]
Scaled scores: [0.5, 0.5, 0.5]
Since all scaled scores are equal, softmax gives equal weights: [0.33, 0.33, 0.33]
Output = weighted sum of all Value vectors
</details>

---

**Solution:**

**Step 1: Raw Attention Scores (Q · K^T for Token 2)**
```
Q[2] = [1, 1, 0, 0]

Score(2→0) = Q[2] · K[0] = [1,1,0,0] · [1,0,0,1] = 1×1 + 1×0 + 0×0 + 0×1 = 1
Score(2→1) = Q[2] · K[1] = [1,1,0,0] · [0,1,1,0] = 1×0 + 1×1 + 0×1 + 0×0 = 1
Score(2→2) = Q[2] · K[2] = [1,1,0,0] · [1,0,1,0] = 1×1 + 1×0 + 0×1 + 0×0 = 1

Raw Scores: [1, 1, 1]
```

**Step 2: Scaling by √d_k**
```
√d_k = √4 = 2

Scaled Scores = [1/2, 1/2, 1/2] = [0.5, 0.5, 0.5]
```

**Step 3: Softmax**
```
e^0.5 ≈ 1.649

Attention Weights:
- Weight[0] = e^0.5 / (3 × e^0.5) = 1/3 ≈ 0.33
- Weight[1] = e^0.5 / (3 × e^0.5) = 1/3 ≈ 0.33
- Weight[2] = e^0.5 / (3 × e^0.5) = 1/3 ≈ 0.33

Weights: [0.33, 0.33, 0.33]
```

**Step 4: Output for Token 2**
```
Output = 0.33 × V[0] + 0.33 × V[1] + 0.33 × V[2]
       = 0.33 × [0.1, 0.2, 0.3, 0.4]
       + 0.33 × [0.5, 0.6, 0.7, 0.8]
       + 0.33 × [0.9, 1.0, 1.1, 1.2]

       = [0.033, 0.066, 0.099, 0.132]
       + [0.165, 0.198, 0.231, 0.264]
       + [0.297, 0.330, 0.363, 0.396]

Output[2] ≈ [0.50, 0.59, 0.69, 0.79]
```

**Key Insight:** With equal attention scores, the output is approximately the average of all Value vectors. In practice, different Q/K combinations create varied attention patterns, allowing the model to selectively attend to relevant context.

---

**Common Mistakes:**
| Mistake | Why It's Wrong | Prevention |
|---------|---------------|------------|
| Forgetting to scale | Large dot products push softmax to extremes, causing vanishing gradients | Always divide by √d_k before softmax |
| Wrong matrix dimensions | K^T transposes Key matrix, not Query | Visualize: Q is (seq, d), K^T is (d, seq), result is (seq, seq) |
| Not normalizing softmax | Softmax must sum to 1 | Always divide by sum of exponentials |

---

## Problem 2: Skill-Builder
### Training Pipeline Analysis

**Concept:** Training Pipeline (Core Concept 3)
**Cognitive Level:** Analyze
**Prerequisites:** Understanding of pre-training, SFT, RLHF

---

**Problem Statement:**

A startup is developing a domain-specific LLM for healthcare. They have:
- Access to a pre-trained 7B parameter model (Llama-style)
- 50,000 medical question-answer pairs from certified physicians
- 5,000 preference comparisons where physicians ranked response quality
- Limited compute budget (can't do full fine-tuning)

**Tasks:**

1. **Design the training pipeline:** Specify which stages to include and in what order.

2. **For each stage, specify:**
   - Training objective
   - Data to use
   - Expected outcome
   - Key hyperparameter considerations

3. **Address the compute constraint:** Recommend parameter-efficient techniques and justify.

4. **Identify risks:** What could go wrong at each stage, and how would you detect it?

---

**Progressive Hints:**

<details>
<summary>Hint 1 (Conceptual)</summary>

The three-stage pipeline is: Pre-training → SFT → RLHF

For a domain-specific model starting from a pre-trained base:
- Pre-training is already done (they have Llama)
- SFT adapts to medical Q&A format
- RLHF aligns with physician preferences

Consider: Is continued pre-training on medical text needed?
</details>

<details>
<summary>Hint 2 (Procedural)</summary>

Pipeline structure:
1. (Optional) Continued pre-training on medical literature
2. SFT on 50K medical Q&A pairs using LoRA
3. RLHF using 5K preference pairs

For LoRA: typical rank r=8-64, applied to attention layers
For RLHF: need to train reward model first, then policy
</details>

<details>
<summary>Hint 3 (Solution Direction)</summary>

Compute-efficient approach:
- Skip continued pre-training (expensive, may not be necessary)
- LoRA for SFT: rank 16-32, attention + MLP projections
- Reward model: smaller model or LoRA-tuned
- Consider DPO as RLHF alternative (no separate reward model)
</details>

---

**Solution:**

**1. Recommended Training Pipeline**

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 0: Base Model (Pre-trained Llama 7B)                      │
│ ─ Already complete, provides general language capability        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: Supervised Fine-Tuning (SFT) with LoRA                 │
│ ─ Adapts to medical Q&A format                                  │
│ ─ Data: 50K physician Q&A pairs                                 │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: Direct Preference Optimization (DPO)                   │
│ ─ Aligns with physician quality preferences                     │
│ ─ Data: 5K preference comparisons                               │
│ ─ Alternative to full RLHF (more compute-efficient)             │
└─────────────────────────────────────────────────────────────────┘
```

**Decision: Skip Continued Pre-training**
- Llama already has medical knowledge from web training
- SFT data will activate this knowledge
- Continued pre-training is expensive and risks catastrophic forgetting

**2. Stage Specifications**

| Stage | Objective | Data | Expected Outcome | Key Hyperparameters |
|-------|-----------|------|------------------|---------------------|
| **SFT** | Cross-entropy loss on response tokens | 50K medical Q&A | Model follows medical Q&A format, uses appropriate terminology | LoRA rank: 32, α: 64, LR: 1e-4, epochs: 3 |
| **DPO** | Preference likelihood maximization | 5K preference pairs | Responses align with physician quality standards | β: 0.1, LR: 5e-5, epochs: 1 |

**3. Compute-Efficient Techniques**

| Technique | Application | Justification |
|-----------|-------------|---------------|
| **LoRA** | SFT and DPO | Updates <1% of parameters; prevents catastrophic forgetting; enables efficient iteration |
| **DPO over RLHF** | Alignment stage | No separate reward model needed; single training phase; mathematically equivalent under assumptions |
| **Gradient checkpointing** | Both stages | Trades compute for memory; enables larger batch sizes |
| **Mixed precision (BF16)** | Both stages | 2x memory reduction; faster computation |

**LoRA Configuration:**
```python
lora_config = {
    "r": 32,              # Rank (balance: capacity vs. efficiency)
    "lora_alpha": 64,     # Scaling (typically 2x rank)
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"       # MLP
    ],
    "lora_dropout": 0.05
}
```

**4. Risk Analysis**

| Stage | Risk | Detection | Mitigation |
|-------|------|-----------|------------|
| **SFT** | Overfitting to training Q&A format | Validation loss increases while train loss decreases | Early stopping; increase dropout; reduce epochs |
| **SFT** | Catastrophic forgetting of general knowledge | Performance drops on general benchmarks | Use LoRA (already planned); test on general tasks |
| **SFT** | Learning physician errors/biases | Manual review reveals factual errors | Curate training data; add fact-checking layer |
| **DPO** | Reward hacking | Model finds shortcuts that satisfy preference model but reduce quality | Diverse evaluation; human spot-checks |
| **DPO** | Preference overfitting | Responses become formulaic | Lower β; fewer epochs; maintain response diversity |

**Evaluation Checkpoints:**
```
After SFT:
- Medical Q&A accuracy (held-out set)
- Response format compliance
- General knowledge retention (sample tasks)

After DPO:
- Physician preference rate vs. baseline
- Factual accuracy maintenance
- Response diversity metrics
```

---

**Common Mistakes:**
| Mistake | Why It's Wrong | Prevention |
|---------|---------------|------------|
| Full fine-tuning with limited compute | Will either fail or require severe compromises | Always use LoRA for constrained budgets |
| Skipping SFT and going straight to RLHF | Model won't follow instructions well; RLHF builds on SFT | SFT creates the base that RLHF refines |
| Using all data for training | No held-out set for evaluation | Reserve 10-20% for validation |

---

## Problem 3: Skill-Builder
### Inference Optimization

**Concept:** Inference and Generation (Core Concept 5)
**Cognitive Level:** Apply
**Prerequisites:** Understanding of KV cache, quantization, batching

---

**Problem Statement:**

You're deploying a 70B parameter LLM for a customer service chatbot. The requirements are:

| Metric | Requirement |
|--------|-------------|
| Time-to-first-token (TTFT) | < 800ms |
| Generation speed | > 30 tokens/second |
| Throughput | 500 concurrent users |
| Hardware budget | 4x A100 80GB GPUs |

**Current baseline (unoptimized):**
- TTFT: 2.5 seconds
- Generation: 12 tokens/second
- Max concurrent: 50 users

**Tasks:**

1. **Calculate memory requirements** for the model in FP16 vs INT8 vs INT4
2. **Design the optimization stack** to meet requirements
3. **Calculate expected improvements** for each optimization
4. **Identify the bottleneck** preventing 500 concurrent users and propose a solution

---

**Progressive Hints:**

<details>
<summary>Hint 1 (Conceptual)</summary>

Memory calculation: Parameters × bytes per parameter
- FP16: 2 bytes per parameter
- INT8: 1 byte per parameter
- INT4: 0.5 bytes per parameter

70B parameters × 2 bytes = 140GB for FP16
</details>

<details>
<summary>Hint 2 (Procedural)</summary>

Optimization stack (in order of impact):
1. Quantization (INT8 or INT4) - reduces memory, increases throughput
2. KV cache optimization - essential for generation
3. Tensor parallelism across 4 GPUs - enables serving 70B
4. Continuous batching - maximizes GPU utilization
5. Speculative decoding - accelerates generation
</details>

<details>
<summary>Hint 3 (Solution Direction)</summary>

With INT4 quantization: 70B × 0.5 = 35GB model weight
KV cache per user (rough): ~2GB for long context
4x A100 80GB = 320GB total

Bottleneck: KV cache memory for 500 concurrent users
Solution: Paged attention (vLLM-style) or shorter context limits
</details>

---

**Solution:**

**1. Memory Requirements Calculation**

| Precision | Bytes/Param | Model Weights | Per-GPU (4-way parallel) |
|-----------|-------------|---------------|--------------------------|
| FP16 | 2 | 140 GB | 35 GB |
| INT8 | 1 | 70 GB | 17.5 GB |
| INT4 | 0.5 | 35 GB | 8.75 GB |

**KV Cache Requirements (per user, 4K context):**
```
KV cache size = 2 × num_layers × 2 × hidden_dim × context_length × bytes

For 70B (80 layers, 8192 hidden, 4K context, FP16):
= 2 × 80 × 2 × 8192 × 4096 × 2 bytes
= 21.5 GB per user at full context

With INT8 KV cache: ~10.7 GB per user
```

**2. Optimization Stack Design**

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 1: Model Compression                                       │
│ ─ INT4 quantization (AWQ/GPTQ) for weights                      │
│ ─ INT8 for KV cache                                             │
│ ─ Result: 35GB model + reduced KV memory                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Layer 2: Parallelism                                            │
│ ─ Tensor parallelism across 4x A100 (80GB each)                 │
│ ─ Model sharded: ~9GB weights per GPU                           │
│ ─ Remaining: ~71GB per GPU for KV cache + overhead              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Layer 3: Serving Optimization                                   │
│ ─ Continuous batching (process requests as they arrive)         │
│ ─ Paged attention (vLLM) for memory-efficient KV management     │
│ ─ Speculative decoding with 7B draft model                      │
└─────────────────────────────────────────────────────────────────┘
```

**3. Expected Improvements**

| Optimization | Metric Impact | Calculation |
|--------------|---------------|-------------|
| **INT4 Quantization** | 2-3x throughput increase | Less memory bandwidth bottleneck |
| **KV Cache** | Essential for generation | Without: O(n²), With: O(n) |
| **Continuous Batching** | 5-10x throughput vs. static | Fills GPU during generation idle |
| **Speculative Decoding** | 2-3x tokens/sec | Draft model proposes, large verifies |

**Expected Performance:**
```
Baseline: TTFT 2.5s, 12 tok/s, 50 concurrent

After INT4 + Tensor Parallel:
- TTFT: ~1.0s (reduced memory access)
- Tokens/sec: ~30 (2.5x improvement)

After Continuous Batching:
- Concurrent: ~200 users (4x improvement)

After Paged Attention:
- Concurrent: 400-500 users (efficient KV allocation)

After Speculative Decoding:
- Tokens/sec: ~50 (1.7x additional)

Final: TTFT <800ms ✓, 50 tok/s ✓, 500 concurrent ✓
```

**4. Bottleneck Analysis**

**Primary Bottleneck: KV Cache Memory**

```
Available GPU memory: 4 × 80GB = 320GB
Model weights (INT4): 35GB
Overhead (activations, etc.): ~20GB
Available for KV: 265GB

Standard KV cache (INT8, 4K context): 10.7GB per user
Maximum users: 265 / 10.7 ≈ 25 concurrent (way below 500)
```

**Solution: Paged Attention (vLLM approach)**

| Technique | How It Helps |
|-----------|--------------|
| **Block-based allocation** | Only allocate KV blocks as needed, not full context |
| **Memory sharing** | Common prefixes share KV blocks |
| **Dynamic eviction** | Swap cold KV blocks to CPU |

**With Paged Attention:**
```
Average context usage: ~1K tokens (not full 4K)
Effective KV per user: ~2.7GB
Concurrent users: 265 / 2.7 ≈ 98 per GPU setup

Add CPU offloading for inactive conversations:
- Hot users in GPU: 100
- Warm users (recent): 200 (fast swap)
- Cold users (inactive): 200+ (slower restore)

Total concurrent: 500+ ✓
```

---

**Common Mistakes:**
| Mistake | Why It's Wrong | Prevention |
|---------|---------------|------------|
| Ignoring KV cache memory | Main memory consumer at scale | Calculate KV requirements per user early |
| Over-aggressive quantization | INT4 can degrade quality significantly | Benchmark quality before deployment |
| Static batching | Leaves GPU idle during generation | Use continuous/dynamic batching |

---

## Problem 4: Challenge
### Production System Design

**Concept:** Complete LLM System (All Core Concepts)
**Cognitive Level:** Synthesize
**Prerequisites:** All previous concepts mastered

---

**Problem Statement:**

Design an LLM-powered code review system for a large enterprise. The system must:

**Functional Requirements:**
1. Analyze pull requests for bugs, security vulnerabilities, and style issues
2. Suggest specific fixes with explanations
3. Learn from team feedback (accepted/rejected suggestions)

**Non-Functional Requirements:**
| Requirement | Specification |
|-------------|---------------|
| Latency | First feedback within 30 seconds of PR submission |
| Accuracy | >90% precision (false positives waste developer time) |
| Throughput | 1000 PRs/day across 50 repositories |
| Security | Code never leaves enterprise network |
| Adaptability | Learns repository-specific patterns |

**Constraints:**
- Must use on-premise infrastructure (compliance requirement)
- Available hardware: 8x A100 GPUs, 1TB RAM, enterprise Kubernetes
- Budget for one LLM (can't use multiple specialized models)

**Tasks:**

1. **Select and justify base model** considering the constraints
2. **Design the processing pipeline** from PR submission to feedback
3. **Specify the training/adaptation strategy** for repository-specific learning
4. **Address hallucination risks** specific to code review
5. **Design the feedback loop** for continuous improvement
6. **Calculate capacity** and identify scaling bottlenecks

---

**Progressive Hints:**

<details>
<summary>Hint 1 (Conceptual)</summary>

Key design decisions:
- Model: Code-specialized open model (CodeLlama, DeepSeek-Coder, StarCoder)
- Pipeline: Chunk PR → Analyze chunks → Aggregate → Filter low-confidence
- Adaptation: LoRA fine-tuning on repository-specific patterns
- Hallucination: Never suggest fixes for code you didn't see
</details>

<details>
<summary>Hint 2 (Procedural)</summary>

Pipeline stages:
1. PR ingestion → Extract diff, context files
2. Chunking → Split into reviewable units (functions, classes)
3. Analysis → LLM reviews each chunk
4. Aggregation → Combine findings, deduplicate
5. Confidence filtering → Only surface high-confidence issues
6. Presentation → Format for developer consumption
</details>

<details>
<summary>Hint 3 (Solution Direction)</summary>

Capacity calculation:
- 1000 PRs/day ÷ 8 hours = ~125 PRs/hour = ~2 PRs/minute
- Average PR: 10 chunks × 2K tokens input × 500 tokens output
- With 8x A100: easily handle this load
- Real bottleneck: Quality, not throughput
</details>

---

**Solution:**

**1. Base Model Selection**

| Candidate | Size | Strengths | Weaknesses | Decision |
|-----------|------|-----------|------------|----------|
| CodeLlama 34B | 34B | Strong code understanding, fits on-premise | Older, less recent training | **Selected** |
| DeepSeek-Coder 33B | 33B | Excellent code perf, recent | Less enterprise deployment history | Backup |
| StarCoder2 15B | 15B | Efficient, good quality | Smaller may miss nuanced issues | Not selected |

**Justification for CodeLlama 34B:**
- Fits on 8x A100 with room for batching (INT8: 34GB)
- Strong code reasoning capabilities
- Well-documented LoRA fine-tuning support
- Permissive license for enterprise use

**2. Processing Pipeline**

```
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1: PR Ingestion                                           │
│ ─────────────────────                                           │
│ Trigger: Webhook on PR creation/update                          │
│ Action: Extract diff, pull affected files, gather metadata      │
│ Output: {diff, context_files, PR_metadata}                      │
│ Latency budget: 2 seconds                                       │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2: Intelligent Chunking                                   │
│ ─────────────────────────────                                   │
│ Strategy: AST-based chunking (function/class boundaries)        │
│ Context: Include imports, type definitions, called functions    │
│ Output: [{chunk, context, file_path, line_range}]               │
│ Latency budget: 3 seconds                                       │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 3: Parallel LLM Analysis                                  │
│ ─────────────────────────────                                   │
│ Prompt: System (reviewer role) + Few-shot + Chunk + Context     │
│ Output schema: {issue_type, severity, location, fix, reasoning} │
│ Batch: Process all chunks in parallel                           │
│ Latency budget: 15 seconds (parallel across GPUs)               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 4: Aggregation & Deduplication                            │
│ ────────────────────────────────                                │
│ Merge: Combine findings from all chunks                         │
│ Dedupe: Remove duplicate issues (same code, similar suggestion) │
│ Prioritize: Sort by severity × confidence                       │
│ Latency budget: 2 seconds                                       │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 5: Confidence Filtering                                   │
│ ───────────────────────────                                     │
│ Threshold: Only surface issues with confidence > 0.85           │
│ Rationale: High precision requirement (>90%)                    │
│ Low-confidence: Log for analysis, don't show to developers      │
│ Latency budget: 1 second                                        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 6: PR Comment Generation                                  │
│ ──────────────────────────────                                  │
│ Format: Inline comments at specific lines                       │
│ Include: Issue, fix suggestion, reasoning                       │
│ Post: Via GitHub/GitLab API                                     │
│ Latency budget: 2 seconds                                       │
└─────────────────────────────────────────────────────────────────┘

Total latency: 25 seconds (within 30s requirement) ✓
```

**3. Training/Adaptation Strategy**

| Stage | Approach | Data | Frequency |
|-------|----------|------|-----------|
| **Initial** | LoRA fine-tune on code review dataset | Public code review datasets + enterprise samples | Once |
| **Repository-specific** | Per-repo LoRA adapter | Accepted suggestions from that repo | Weekly batch |
| **Continuous** | Update adapters with new feedback | New accepted/rejected suggestions | Weekly |

**Adapter Architecture:**
```python
# Base model + stackable LoRA adapters
adapters = {
    "base_review": CodeReviewLoRA,       # General code review skills
    "repo_frontend": RepoSpecificLoRA,   # Frontend patterns
    "repo_backend": RepoSpecificLoRA,    # Backend patterns
    # ... per repository
}

# At inference, load base + repo-specific adapter
active_adapters = ["base_review", f"repo_{repository_id}"]
```

**4. Hallucination Mitigation**

| Risk | Mitigation | Implementation |
|------|------------|----------------|
| **Suggesting fixes for unseen code** | Only allow suggestions that reference visible lines | Parse suggestions, verify line numbers exist in context |
| **Fabricated API suggestions** | Cross-reference with codebase | Embedding search: does suggested function exist? |
| **Incorrect security claims** | Conservative security flagging | For security issues, require explicit pattern match OR high confidence |
| **Wrong syntax in suggestions** | Validate generated code | Parse suggested fixes; reject if syntax invalid |

**Validation Layer:**
```
For each suggestion:
1. Parse suggested code → Valid syntax?
2. Check referenced lines → Exist in PR?
3. If security claim → Pattern database match?
4. If API reference → Exists in codebase embeddings?

If any check fails → Demote to low-confidence or reject
```

**5. Feedback Loop Design**

```
┌─────────────────────────────────────────────────────────────────┐
│                    FEEDBACK COLLECTION                          │
│                                                                 │
│  Developer action on suggestion:                                │
│  ├── 👍 Accept → Store as positive example                      │
│  ├── 👎 Dismiss → Store as negative example with reason         │
│  ├── ✏️ Modify then apply → Store original + modification       │
│  └── 🤷 Ignore → No signal (don't train on)                     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FEEDBACK PROCESSING                          │
│                                                                 │
│  Weekly batch job:                                              │
│  1. Aggregate feedback per repository                           │
│  2. Filter: minimum 10 examples for training signal             │
│  3. Balance: sample to avoid skew toward common issues          │
│  4. Format: Create preference pairs (accepted vs. rejected)     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ADAPTER UPDATE                               │
│                                                                 │
│  Training:                                                      │
│  1. DPO on preference pairs (accepted = preferred)              │
│  2. Validate on held-out set before deployment                  │
│  3. A/B test: 10% traffic to new adapter                        │
│  4. If metrics improve → Roll out; else → Rollback              │
└─────────────────────────────────────────────────────────────────┘
```

**6. Capacity Calculation**

```
Requirements: 1000 PRs/day, 30s latency

Average PR:
- 10 changed files
- 5 reviewable chunks
- 2000 input tokens per chunk (code + context)
- 500 output tokens per chunk

Daily tokens:
- Input: 1000 PRs × 5 chunks × 2000 = 10B input tokens
- Output: 1000 PRs × 5 chunks × 500 = 2.5B output tokens

Throughput needed:
- 8 working hours = 28,800 seconds
- But 30s latency requirement means parallel processing
- Peak load: ~2 PRs/minute = 10 chunks/minute needing completion

With 8x A100 (INT8 CodeLlama 34B):
- Per-GPU throughput: ~100 tokens/second output
- Total: 800 tokens/second
- Can process: 800 / 500 = 1.6 chunks/second = 96 chunks/minute

Needed: 10 chunks/minute
Available: 96 chunks/minute
Headroom: 9.6x ✓

Bottleneck: NOT throughput
Real bottleneck: Quality calibration to hit 90% precision
```

---

**Common Mistakes:**
| Mistake | Why It's Wrong | Prevention |
|---------|---------------|------------|
| Over-optimizing for throughput | Quality is the real challenge | Focus on precision/recall before scaling |
| Training on all feedback equally | Ignored suggestions have no signal | Only train on explicit accept/reject |
| Single global model | Repositories have different patterns | Use per-repo adapters |

---

## Problem 5: Debug/Fix
### Hallucination Diagnosis

**Concept:** Capabilities and Limitations (Core Concept 6)
**Cognitive Level:** Analyze
**Prerequisites:** Understanding of hallucination, context utilization

---

**Problem Statement:**

A medical information system using an LLM is exhibiting concerning behavior. Review the following interaction logs and diagnose the issues.

**System Configuration:**
- Model: Fine-tuned Llama 70B for medical Q&A
- Context: RAG system retrieves relevant medical literature
- Prompt: Includes retrieved context + user question

**Problematic Interaction 1:**
```
Retrieved Context:
"Metformin is typically prescribed at 500mg twice daily for type 2 diabetes,
with maximum dose of 2000mg daily. Common side effects include GI upset.
Contraindicated in patients with eGFR < 30 mL/min/1.73m²."

User Question:
"What's the recommended dosage of metformin for someone with kidney problems?"

Model Response:
"For patients with kidney problems, metformin dosage should be adjusted based
on eGFR. For eGFR 30-45, reduce to 1000mg daily. For eGFR 45-60, standard
dosing is safe. The medication is contraindicated when eGFR falls below 30.
Recent studies from 2024 suggest that metformin may have protective effects
on kidney function when used appropriately."
```

**Problematic Interaction 2:**
```
Retrieved Context:
"Aspirin 81mg daily is recommended for secondary prevention in patients with
established cardiovascular disease. Primary prevention use is more controversial.
Bleeding risks include GI hemorrhage and intracranial bleeding."

[Second retrieved document:]
"The 2019 ACC/AHA guidelines recommend against routine aspirin use for primary
prevention in adults over 70 or those with increased bleeding risk."

User Question:
"Should my 72-year-old mother take daily aspirin? She's healthy but worried
about heart attacks."

Model Response:
"For a healthy 72-year-old without established cardiovascular disease, daily
aspirin for primary prevention is generally recommended as the cardiovascular
benefits outweigh the bleeding risks in most patients. A typical dose would be
81mg daily. She should discuss with her doctor, but this is standard practice."
```

**Tasks:**

1. **Identify the specific hallucination/error** in each interaction
2. **Diagnose the root cause** (context utilization, fabrication, reasoning error)
3. **Explain why the model made this error** based on LLM limitations
4. **Propose system-level fixes** to prevent each type of error
5. **Design a validation layer** that would catch these issues

---

**Progressive Hints:**

<details>
<summary>Hint 1 (Conceptual)</summary>

Interaction 1 issues:
- The specific eGFR thresholds (30-45, 45-60) for dose adjustment are NOT in the context
- The "2024 studies" claim is fabricated

Interaction 2 issues:
- Model contradicts the retrieved guideline
- Gets the recommendation backwards for this demographic
</details>

<details>
<summary>Hint 2 (Procedural)</summary>

Root causes:
- Interaction 1: Model filled in gaps with plausible but unverified information
- Interaction 2: "Lost in the middle" - second document less attended to
- Both: No uncertainty indication despite low confidence

Check: Does each claim have explicit support in retrieved context?
</details>

<details>
<summary>Hint 3 (Solution Direction)</summary>

Fixes:
1. Require explicit citations for all specific claims
2. Use confidence calibration to flag uncertain responses
3. Add a verification layer that checks claims against context
4. For medical: Conservative default = "consult physician" for edge cases
</details>

---

**Solution:**

**1. Specific Errors Identified**

| Interaction | Error Type | Specific Error |
|-------------|------------|----------------|
| **1** | Fabrication | eGFR thresholds "30-45" and "45-60" for dose adjustment are invented |
| **1** | Fabrication | "Recent studies from 2024" is completely made up |
| **1** | Gap-filling | Provided specific guidance not supported by context |
| **2** | Contradiction | Recommends aspirin when context explicitly recommends against |
| **2** | Context misuse | Ignored second retrieved document's key guidance |
| **2** | Safety error | Gave dangerous advice for vulnerable population |

**2. Root Cause Diagnosis**

**Interaction 1: Fabrication**
```
Root Cause: Plausibility-driven gap-filling

The context mentions:
- Contraindication at eGFR < 30 ✓
- No intermediate dosing guidance

Model behavior:
- Recognized a gap in dosing guidance for moderate kidney impairment
- Generated plausible-sounding thresholds based on training data
- Added fabricated "2024 studies" for credibility

Why this happens: LLMs optimize for coherent, helpful responses. Gaps in
context trigger generation from training patterns, not abstention.
```

**Interaction 2: Context Contradiction**
```
Root Cause: Lost in the middle + Recency bias in training

Retrieved context order:
1. First: General aspirin information (secondary prevention)
2. Second: Specific guideline (against primary prevention in >70)

Model behavior:
- Attended strongly to first document (aspirin is helpful narrative)
- Under-weighted second document (the contradiction)
- Generated recommendation matching common training pattern

Why this happens:
- "Lost in the middle" phenomenon: middle context less attended
- Training data likely has more "aspirin is beneficial" examples
- Model defaulted to training prior over contradictory context
```

**3. LLM Limitation Mapping**

| Error | LLM Limitation | Explanation |
|-------|----------------|-------------|
| Fabricated eGFR thresholds | Hallucination | Cannot distinguish known from inferred facts |
| "2024 studies" claim | Hallucination | Generates supporting evidence that doesn't exist |
| Ignored guideline | Lost in the middle | Attention to middle context is reduced |
| Wrong recommendation | Training distribution bias | Common patterns override contradictory context |
| No uncertainty | No calibration | Cannot indicate when response is uncertain |

**4. System-Level Fixes**

| Fix | Implementation | Target Error |
|-----|----------------|--------------|
| **Citation requirement** | Prompt: "Every specific claim must cite [Source: document name, line]" | Fabrication |
| **Quote extraction** | Require verbatim quotes for critical claims | Fabrication |
| **Context reordering** | Put most relevant document last (recency effect) | Lost in middle |
| **Explicit contradiction check** | Prompt: "First identify any contradictions between documents" | Context conflict |
| **Conservative default** | For medical: "If uncertain, recommend physician consultation" | Safety errors |
| **Confidence-gated output** | Generate confidence; suppress low-confidence specific claims | All |

**Improved System Prompt:**
```
You are a medical information assistant. For each response:

1. CITATIONS: Every specific claim (dosages, thresholds, recommendations)
   must include [Source: document_name] or be marked [NOT IN PROVIDED SOURCES].

2. CONTRADICTIONS: Before answering, identify any contradictions between
   retrieved documents. When documents conflict, state the conflict explicitly.

3. LIMITATIONS: If the retrieved context doesn't fully answer the question,
   say "The provided sources don't specify..." rather than inferring.

4. SAFETY: For patient-specific questions, always recommend consulting a
   healthcare provider, especially for elderly patients or those with
   comorbidities.

5. NO FABRICATION: Never cite studies, dates, or statistics not explicitly
   in the provided context.
```

**5. Validation Layer Design**

```
┌─────────────────────────────────────────────────────────────────┐
│ VALIDATION LAYER                                                │
│                                                                 │
│ Input: {response, retrieved_context}                            │
│ Output: {validated_response, issues[], confidence}              │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ CHECK 1: Citation Verification                                  │
│ ─────────────────────────────                                   │
│ For each claim with [Source: X]:                                │
│   - Does X exist in retrieved documents? → Pass/Fail            │
│   - Does X actually contain this claim? → Pass/Fail             │
│                                                                 │
│ For claims without citations:                                   │
│   - Flag as potentially ungrounded                              │
│   - Search context: Is claim supported? → Verified/Unverified   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ CHECK 2: Numeric Verification                                   │
│ ───────────────────────────                                     │
│ Extract numbers from response: dosages, thresholds, percentages │
│ For each number:                                                │
│   - Find in context? → Verified                                 │
│   - Not in context? → Flag "potentially fabricated"             │
│                                                                 │
│ Example from Interaction 1:                                     │
│   "30-45", "45-60" → NOT FOUND IN CONTEXT → FLAGGED             │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ CHECK 3: Contradiction Detection                                │
│ ──────────────────────────────                                  │
│ Compare response against each retrieved document:               │
│   - Semantic similarity of recommendation                       │
│   - If response contradicts any document → FLAG                 │
│                                                                 │
│ Example from Interaction 2:                                     │
│   Response: "generally recommended"                             │
│   Context: "recommend against"                                  │
│   → CONTRADICTION DETECTED                                      │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ CHECK 4: Safety-Critical Keywords                               │
│ ─────────────────────────────────                               │
│ If response contains:                                           │
│   - Drug names + dosages → Require citation verification        │
│   - Age-specific advice → Verify against guidelines             │
│   - "safe", "recommended" → Check for contradicting warnings    │
│                                                                 │
│ High-risk responses require human review before delivery        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ DECISION LAYER                                                  │
│ ──────────────                                                  │
│ Issues found = 0      → Deliver response                        │
│ Issues found = 1-2    → Add disclaimers, flag uncertain claims  │
│ Issues found = 3+     → Regenerate with stricter prompt         │
│ Critical issues       → Human review required                   │
└─────────────────────────────────────────────────────────────────┘
```

**Validation Applied to Interactions:**

| Interaction | Validation Check | Finding | Action |
|-------------|------------------|---------|--------|
| 1 | Numeric verification | "30-45", "45-60", "2024" not in context | Flag + regenerate |
| 1 | Citation check | No citations provided | Flag |
| 2 | Contradiction detection | "recommended" vs. "recommend against" | Block + regenerate |
| 2 | Safety keyword | "72-year-old" + recommendation | Require guideline match |

---

**Common Mistakes:**
| Mistake | Why It's Wrong | Prevention |
|---------|---------------|------------|
| Trusting specific numbers without verification | LLMs confabulate plausible statistics | Always verify numbers against source |
| Assuming model uses all context equally | Lost in the middle is systematic | Place critical info at start/end |
| No validation layer | Hallucinations will occur | Always validate safety-critical outputs |

---

## Self-Assessment Guide

### Mastery Checklist

| Problem | Mastery Indicator | Check |
|---------|-------------------|-------|
| **1 (Warm-Up)** | Can compute attention weights without reference | ☐ |
| **2 (Skill-Builder)** | Can design training pipeline for new domain | ☐ |
| **3 (Skill-Builder)** | Can calculate and optimize inference performance | ☐ |
| **4 (Challenge)** | Can design end-to-end LLM system with reliability | ☐ |
| **5 (Debug/Fix)** | Can diagnose and fix hallucination issues | ☐ |

### Progression Path

```
If struggled with Problem 1:
  → Review: Self-Attention mechanism, Q/K/V projections
  → Flashcard: Card 1 (Easy)

If struggled with Problem 2:
  → Review: Training Pipeline section, LoRA, RLHF
  → Flashcard: Card 2 (Easy)

If struggled with Problem 3:
  → Review: Inference and Generation section
  → Flashcard: Card 4 (Medium)

If struggled with Problem 4:
  → Review: All Core Concepts, Case Study
  → Flashcard: Card 5 (Hard)

If struggled with Problem 5:
  → Review: Capabilities and Limitations section
  → Flashcard: Card 3 (Medium)
```

---

## Extension Challenges

### For Problem 1 (Self-Attention):
Implement multi-head attention with h=2 heads for the same input. How does splitting the attention dimension affect the output?

### For Problem 2 (Training):
Design an experiment to determine whether continued pre-training on medical text would help. What metrics would you measure?

### For Problem 3 (Inference):
Calculate the exact KV cache memory requirements for a 70B model with 100K context window. At what point does CPU offloading become necessary?

### For Problem 4 (System Design):
Extend the code review system to support multi-language repositories where a single PR might include Python, JavaScript, and SQL. How does this affect chunking and analysis?

### For Problem 5 (Hallucination):
Design an automated test suite that would detect the hallucination patterns identified. How would you generate test cases that probe for fabrication?

---

*Generated from Lesson 3: Large Language Models | Practice Problems Skill*
