# Practice Problems: Lesson 7 - Generative AI

**Source:** Lessons/Lesson_7.md
**Subject Area:** AI Learning - Generative AI: Foundations, Architectures, and Applications
**Date Generated:** 2026-01-08
**Total Problems:** 5
**Estimated Completion Time:** 90-120 minutes

---

## Problem Distribution

| Type | Count | Difficulty | Focus |
|------|-------|------------|-------|
| Warm-Up | 1 | Foundation | Direct concept application |
| Skill-Builder | 2 | Intermediate | Multi-step procedures |
| Challenge | 1 | Advanced | Complex synthesis |
| Debug/Fix | 1 | Diagnostic | Error identification |

---

## Problem 1: Warm-Up - Sampling Parameter Configuration

**Difficulty:** Foundation
**Estimated Time:** 15 minutes
**Concepts:** Temperature, top-p, sampling strategies

### Problem Statement

You are configuring an LLM API for three different applications. For each scenario, specify the sampling parameters and justify your choices.

**Scenarios:**

A) **Medical Symptom Checker:** Patients describe symptoms and receive possible conditions. Accuracy is critical; the system should not speculate.

B) **Poetry Generator:** Users provide a theme, and the system generates creative poems. Uniqueness and artistic expression are valued.

C) **SQL Query Generator:** Users describe what data they need in natural language, and the system generates SQL queries. Queries must be syntactically valid.

**For each scenario, specify:**
- Temperature (0.0 - 2.0)
- Top-p (0.0 - 1.0)
- Any additional parameters or constraints
- Reasoning for your choices

### Hints

<details>
<summary>Hint 1 (Conceptual)</summary>
Temperature controls randomness: low = deterministic, high = creative.
Top-p (nucleus sampling) limits the token pool to cumulative probability p.
</details>

<details>
<summary>Hint 2 (Procedural)</summary>
Consider the consequences of errors in each domain:
- Medical: false information could harm patients
- Poetry: "wrong" answers are actually features
- SQL: syntax errors make outputs unusable
</details>

<details>
<summary>Hint 3 (Structural)</summary>
Also consider: stop sequences, max tokens, and post-processing validation for each use case.
</details>

### Solution

**A) Medical Symptom Checker**

```
Temperature: 0.1-0.2
Top-p: 0.9
Max tokens: 500
Stop sequences: None specific

Additional constraints:
- System prompt: "Only mention conditions with strong symptom matches"
- Post-processing: Filter speculative language
- Require confidence levels or citations
```

**Reasoning:**
- Very low temperature ensures consistent, grounded responses
- Slightly above 0 allows minor variation without hallucination
- Medical domain requires maximal accuracy; creativity is dangerous
- Should never guess or speculate about serious conditions

---

**B) Poetry Generator**

```
Temperature: 0.9-1.2
Top-p: 0.85-0.92
Max tokens: 200-400
Stop sequences: ["\n\n\n"] (end after poem)

Additional constraints:
- System prompt: "Be creative, use varied vocabulary and structures"
- No accuracy constraints needed
- May benefit from presence_penalty: 0.5 to avoid repetition
```

**Reasoning:**
- High temperature encourages unexpected word choices and metaphors
- Top-p slightly restricted to avoid completely nonsensical tokens
- Poetry benefits from creative exploration
- Repetition penalty ensures varied language

---

**C) SQL Query Generator**

```
Temperature: 0.3-0.5
Top-p: 0.95
Max tokens: 300
Stop sequences: [";", "```"]

Additional constraints:
- System prompt: Include schema information
- Post-processing: SQL syntax validation before execution
- Consider: few-shot examples of correct queries
```

**Reasoning:**
- Moderate temperature allows alternative valid approaches (JOINs vs subqueries)
- SQL has strict syntax; too much creativity causes errors
- Higher top-p because SQL vocabulary is constrained
- Must validate syntax before executing on database
- Stop at semicolon to get clean, single queries

---

**Summary Table:**

| Application | Temperature | Top-p | Key Constraint |
|-------------|-------------|-------|----------------|
| Medical | 0.1-0.2 | 0.9 | Accuracy critical |
| Poetry | 0.9-1.2 | 0.85-0.92 | Creativity valued |
| SQL | 0.3-0.5 | 0.95 | Syntax validation |

---

## Problem 2: Skill-Builder - Diffusion Process Analysis

**Difficulty:** Intermediate
**Estimated Time:** 25 minutes
**Concepts:** Forward/reverse diffusion, noise schedules, CFG

### Problem Statement

You are analyzing a diffusion model for image generation with the following configuration:

- **Noise schedule:** Linear β from β₁ = 0.0001 to β_T = 0.02
- **Total timesteps:** T = 1000
- **Image size:** 64×64×3

**Tasks:**

a) Calculate the cumulative noise parameter ᾱ_t at t=500 (halfway through diffusion). The formula is:
   - αₜ = 1 - βₜ
   - ᾱₜ = ∏ᵢ₌₁ᵗ αᵢ

b) If the original image x₀ has pixel values in [0, 1], describe qualitatively what x₅₀₀ looks like (the noised image at t=500).

c) The model uses Classifier-Free Guidance with scale w=7.5. Given:
   - Unconditional noise prediction: ε_uncond = 0.3
   - Conditional noise prediction: ε_cond = 0.8

   Calculate the guided noise prediction ε̂ and explain what this means for generation.

d) Why would increasing the guidance scale to w=15 potentially degrade image quality?

### Hints

<details>
<summary>Hint 1 (Part a)</summary>
With linear schedule, βₜ increases linearly. For approximation, you can use:
ᾱₜ ≈ exp(-Σᵢ₌₁ᵗ βᵢ) since log(1-β) ≈ -β for small β
</details>

<details>
<summary>Hint 2 (Part c)</summary>
CFG formula: ε̂ = ε_uncond + w × (ε_cond - ε_uncond)
This amplifies the "direction" from unconditional toward conditional.
</details>

<details>
<summary>Hint 3 (Part d)</summary>
Consider what happens when you over-amplify a signal. Think about image artifacts, color saturation, and loss of fine details.
</details>

### Solution

**Part a) Calculate ᾱ₅₀₀**

Linear schedule from β₁ = 0.0001 to β₁₀₀₀ = 0.02:
```
βₜ = β₁ + (t-1)/(T-1) × (β_T - β₁)
   = 0.0001 + (t-1)/999 × (0.02 - 0.0001)
   = 0.0001 + (t-1)/999 × 0.0199
```

At t=500:
```
β₅₀₀ = 0.0001 + 499/999 × 0.0199 ≈ 0.0001 + 0.00994 ≈ 0.01004
```

For ᾱₜ calculation, use the approximation:
```
ᾱₜ ≈ exp(-Σᵢ₌₁ᵗ βᵢ)

Sum of βᵢ from 1 to 500 (arithmetic series):
≈ 500 × (β₁ + β₅₀₀)/2
≈ 500 × (0.0001 + 0.01)/2
≈ 500 × 0.00505
≈ 2.525

ᾱ₅₀₀ ≈ exp(-2.525) ≈ 0.08
```

**Answer:** ᾱ₅₀₀ ≈ **0.08** (approximately 8% of original signal remains)

---

**Part b) Qualitative description of x₅₀₀**

The noised image formula is:
```
x_t = √ᾱₜ × x₀ + √(1-ᾱₜ) × ε
```

With ᾱ₅₀₀ ≈ 0.08:
- Signal coefficient: √0.08 ≈ 0.28
- Noise coefficient: √0.92 ≈ 0.96

**Description:**
- The image is heavily obscured by noise (~96% noise, ~28% signal)
- Very rough structure may be barely visible (large color blobs)
- Fine details completely destroyed
- Looks like "TV static with a hint of color bias"
- Human would not recognize the original content
- Still distinguishable from pure noise (t=1000)

---

**Part c) CFG Calculation**

Given:
- ε_uncond = 0.3
- ε_cond = 0.8
- w = 7.5

Apply CFG formula:
```
ε̂ = ε_uncond + w × (ε_cond - ε_uncond)
  = 0.3 + 7.5 × (0.8 - 0.3)
  = 0.3 + 7.5 × 0.5
  = 0.3 + 3.75
  = 4.05
```

**Answer:** ε̂ = **4.05**

**Interpretation:**
- The conditional "direction" (ε_cond - ε_uncond = 0.5) is amplified 7.5×
- Result (4.05) is much larger than either original prediction
- This strongly pushes generation toward the text prompt
- The model will denoise more aggressively in the "conditional direction"
- Trade-off: Better prompt adherence, but may reduce diversity and naturalness

---

**Part d) Why w=15 degrades quality**

**Over-amplification effects:**

1. **Color saturation:** Guided predictions push colors to extremes, causing oversaturated, unnatural hues

2. **Loss of fine details:** Strong guidance overwhelms subtle features; textures become simplified or repetitive

3. **Artifacts:** Mathematical instabilities from large prediction values cause visual glitches, especially at edges

4. **Mode collapse:** Very high guidance forces all generations toward a narrow "hyper-conditional" mode, losing variety

5. **Numerical issues:** When ε̂ becomes very large (4.05 vs typical ~1), subsequent denoising steps may produce out-of-range values

**Example at w=15:**
```
ε̂ = 0.3 + 15 × 0.5 = 7.8
```
This extreme value causes the model to "overcorrect" during denoising, producing unrealistic images.

**Recommended range:** w ∈ [5, 10] for most applications, with 7.5 as a robust default.

---

## Problem 3: Skill-Builder - Fine-Tuning Strategy Design

**Difficulty:** Intermediate
**Estimated Time:** 25 minutes
**Concepts:** LoRA, QLoRA, instruction tuning, domain adaptation

### Problem Statement

A healthcare startup wants to fine-tune an open-source 13B parameter LLM for clinical note summarization. They have:

- **Hardware:** 2× NVIDIA A10 GPUs (24GB each)
- **Data:** 50,000 clinical note + summary pairs
- **Requirements:**
  - Must handle HIPAA-compliant data (cannot use external APIs)
  - Summaries must be medically accurate
  - Inference latency < 2 seconds per note

**Tasks:**

a) Can they perform full fine-tuning with their hardware? Calculate the memory requirements.

b) Design a LoRA fine-tuning configuration. Specify:
   - Target modules
   - Rank (r)
   - Alpha
   - Dropout
   - Estimated trainable parameters

c) Should they use QLoRA? Analyze the tradeoffs for this specific use case.

d) Design the training data format and any data augmentation strategies for medical accuracy.

### Hints

<details>
<summary>Hint 1 (Memory)</summary>
Full fine-tuning memory ≈ 4 × model_size (weights + gradients + optimizer states)
13B × 4 bytes (FP32) = 52GB for weights alone
</details>

<details>
<summary>Hint 2 (LoRA Config)</summary>
For summarization tasks, target attention modules (q_proj, v_proj) and possibly MLP layers.
Rank 8-64 typically sufficient, alpha = 2×rank is common.
</details>

<details>
<summary>Hint 3 (Medical Domain)</summary>
Medical accuracy requires: maintaining terminology, not hallucinating conditions, preserving critical information. Consider validation strategies.
</details>

### Solution

**Part a) Full Fine-Tuning Memory Analysis**

Memory breakdown for 13B model:
```
Model weights (FP16):        13B × 2 bytes = 26 GB
Gradients (FP16):            13B × 2 bytes = 26 GB
Optimizer states (Adam FP32): 13B × 8 bytes = 104 GB
Activations (batch-dependent): ~10-20 GB

Total minimum: 26 + 26 + 104 + 15 = ~171 GB
```

**Available:** 2 × 24 GB = 48 GB

**Answer:** **No**, full fine-tuning is not possible. They would need ~4× more GPU memory (8× A10s or 2× A100-80GB).

---

**Part b) LoRA Configuration**

```yaml
# LoRA Configuration for Clinical Summarization
base_model: "meta-llama/Llama-2-13b-hf"

lora_config:
  r: 32                    # Rank - higher for complex task
  lora_alpha: 64           # Alpha = 2 × r (standard)
  lora_dropout: 0.05       # Light dropout for regularization

  target_modules:
    - q_proj              # Query projection (attention)
    - v_proj              # Value projection (attention)
    - k_proj              # Key projection (attention)
    - o_proj              # Output projection (attention)
    - gate_proj           # MLP gate (for complex reasoning)
    - up_proj             # MLP up projection

  bias: "none"            # Don't train biases
  task_type: "CAUSAL_LM"

training_args:
  learning_rate: 2e-4
  batch_size: 4           # Per GPU
  gradient_accumulation: 4
  epochs: 3
  warmup_ratio: 0.1
```

**Parameter calculation:**
```
For each target module in 13B Llama-2:
- Hidden dim (d): 5120
- Rank (r): 32

Params per module: 2 × d × r = 2 × 5120 × 32 = 327,680

Target modules: 6 per layer × 40 layers = 240 modules
Total LoRA params: 240 × 327,680 ≈ 78.6M parameters

Percentage: 78.6M / 13B = 0.6%
```

**Estimated trainable parameters:** ~79M (0.6% of base model)

**Memory estimate with LoRA:**
```
Base model (FP16 frozen):    26 GB
LoRA weights:                ~0.15 GB
LoRA gradients + optimizer:  ~0.6 GB
Activations:                 ~10 GB
─────────────────────────────────────
Total:                       ~37 GB
```

This fits within 2× A10 (48 GB) with room for batching.

---

**Part c) QLoRA Analysis**

**QLoRA would:**
- Quantize base model to 4-bit: 26 GB → 6.5 GB
- Total memory: ~17-20 GB
- Enable single-GPU training

**Tradeoffs for healthcare use case:**

| Factor | QLoRA Advantage | QLoRA Disadvantage |
|--------|-----------------|---------------------|
| Memory | Fits on single A10 | — |
| Cost | Fewer GPUs needed | — |
| Quality | — | Slight degradation from quantization |
| Medical accuracy | — | Risk of subtle errors in terminology |
| Inference | — | May need FP16 for production |

**Recommendation:** **Use standard LoRA, not QLoRA**

**Rationale:**
1. They have sufficient memory (48GB) for LoRA
2. Medical domain requires maximum accuracy
3. Quantization introduces small errors that could affect clinical terminology
4. 50K samples is enough data to benefit from full-precision adaptation
5. Production inference at FP16 is already fast enough (< 2s requirement)

QLoRA would be recommended if they only had one GPU or needed to fine-tune a larger model (70B).

---

**Part d) Training Data Format and Augmentation**

**Data Format:**
```json
{
  "instruction": "Summarize the following clinical note, preserving all diagnoses, medications, and key findings.",
  "input": "[Full clinical note text...]",
  "output": "[Target summary...]",
  "metadata": {
    "note_type": "discharge_summary",
    "specialty": "cardiology",
    "length_category": "long"
  }
}
```

**Data Augmentation Strategies:**

1. **Terminology preservation training:**
   ```
   Create variants where key medical terms must be preserved exactly.
   Add explicit instruction: "Ensure the following terms appear in summary: [list]"
   ```

2. **Negative examples:**
   ```
   Include examples where summary incorrectly omits critical information.
   Label these as "incorrect" for contrastive learning.
   ```

3. **Multi-length summaries:**
   ```
   For each note, create brief (1-2 sentences) and detailed (paragraph) summaries.
   Train model to follow length instructions.
   ```

4. **Template augmentation:**
   ```
   Vary instruction phrasing:
   - "Summarize this clinical note"
   - "Create a brief summary of the patient encounter"
   - "Extract key clinical findings from this note"
   ```

**Validation Strategy:**

```python
def validate_summary(note, summary):
    # 1. Medical entity extraction
    note_entities = extract_medical_entities(note)
    summary_entities = extract_medical_entities(summary)

    # 2. Critical entity coverage
    critical = ["diagnoses", "medications", "allergies"]
    for entity_type in critical:
        if not all(e in summary_entities for e in note_entities[entity_type]):
            return False, f"Missing {entity_type}"

    # 3. Hallucination check
    for entity in summary_entities:
        if entity not in note_entities:
            return False, f"Hallucinated entity: {entity}"

    return True, "Valid"
```

---

## Problem 4: Challenge - Multimodal System Architecture

**Difficulty:** Advanced
**Estimated Time:** 30 minutes
**Concepts:** Multimodal models, RAG, system design, evaluation

### Problem Statement

A law firm wants to build an AI system for contract analysis that can:

1. Process contracts (PDFs with text, tables, signatures, stamps)
2. Answer questions about specific clauses
3. Compare clauses across multiple contracts
4. Flag potential issues or unusual terms
5. Generate summary reports

**Constraints:**
- Must work with confidential documents (no external APIs for document content)
- Need to process 100+ page contracts
- Response time < 30 seconds for queries
- Must cite specific pages/sections for all answers

**Design a complete system architecture addressing:**
- Document processing pipeline
- Model selection (open-source where privacy required)
- Retrieval strategy for long documents
- Generation with citations
- Evaluation approach

### Hints

<details>
<summary>Hint 1 (Document Processing)</summary>
Consider: OCR for scanned pages, table extraction, hierarchical chunking by sections, visual element handling.
</details>

<details>
<summary>Hint 2 (Long Documents)</summary>
100+ pages exceeds context limits. Need retrieval strategy: hierarchical indexing, multi-stage retrieval, or summarization layers.
</details>

<details>
<summary>Hint 3 (Citations)</summary>
Track metadata through the pipeline: page numbers, section headers, bounding boxes. Include in generation prompt.
</details>

### Solution

**Complete Contract Analysis System Architecture:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DOCUMENT INGESTION                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  PDF Upload → Document AI    →  ┌─────────────────────────────────┐    │
│               (on-premise)      │ Extracted Components:           │    │
│                                 │ • Text (by section/page)        │    │
│                                 │ • Tables (as markdown)          │    │
│                                 │ • Visual elements (described)   │    │
│                                 │ • Metadata (dates, parties)     │    │
│                                 └─────────────┬───────────────────┘    │
│                                               │                         │
│                                               ▼                         │
│                              ┌─────────────────────────────────┐       │
│                              │ HIERARCHICAL CHUNKING           │       │
│                              │ L1: Full contract summary       │       │
│                              │ L2: Section summaries           │       │
│                              │ L3: Clause-level chunks (512t)  │       │
│                              └─────────────┬───────────────────┘       │
│                                            │                            │
└────────────────────────────────────────────┼────────────────────────────┘
                                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         INDEXING & STORAGE                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │ Vector Store     │    │ BM25 Index       │    │ Metadata Store   │  │
│  │ (Milvus/Qdrant)  │    │ (Elasticsearch)  │    │ (PostgreSQL)     │  │
│  │                  │    │                  │    │                  │  │
│  │ - Embeddings     │    │ - Keyword search │    │ - Page numbers   │  │
│  │ - Hierarchical   │    │ - Legal terms    │    │ - Section IDs    │  │
│  │   levels         │    │ - Entity names   │    │ - Contracts DB   │  │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         QUERY PROCESSING                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  User Query → Query       → Multi-Stage Retrieval                       │
│               Analysis                                                   │
│                              │                                           │
│               ┌──────────────┼──────────────┐                           │
│               ▼              ▼              ▼                           │
│          L1 Summary    L2 Sections    L3 Clauses                        │
│          (context)     (relevance)    (precision)                       │
│               │              │              │                           │
│               └──────────────┼──────────────┘                           │
│                              ▼                                           │
│                         Reranker (cross-encoder)                        │
│                              ▼                                           │
│                    Top-10 chunks with citations                         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         GENERATION & RESPONSE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Retrieved Context + Query → Local LLM        → Response with Citations │
│                              (Llama-2-70B or  │                         │
│                               Mixtral 8x22B)  │  Format:                │
│                                               │  "According to Section  │
│  System Prompt:                               │   4.2 (page 23), the    │
│  "You are a legal assistant.                  │   termination clause    │
│   Always cite specific                        │   states..."            │
│   sections and pages.                         │                         │
│   If uncertain, say so."                      │                         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Component Details:**

**1. Document Processing (On-Premise)**

| Component | Tool | Purpose |
|-----------|------|---------|
| PDF parsing | PyMuPDF + Unstructured | Extract text preserving layout |
| OCR | Tesseract / PaddleOCR | Handle scanned documents |
| Table extraction | Table Transformer | Convert tables to markdown |
| Visual analysis | Local LLaVA | Describe stamps, signatures |

**2. Model Selection**

| Function | Model | Rationale |
|----------|-------|-----------|
| Embeddings | BGE-large-en-v1.5 | Open source, strong retrieval |
| Reranker | bge-reranker-large | Cross-encoder precision |
| Generation | Mixtral-8x22B-Instruct | Strong reasoning, open source |
| Fallback | Llama-2-70B-chat | Proven, well-documented |

**3. Retrieval Strategy**

```python
def multi_stage_retrieval(query, contract_id, k=10):
    # Stage 1: Get contract context
    contract_summary = get_l1_summary(contract_id)

    # Stage 2: Identify relevant sections
    query_embedding = embed(query)
    relevant_sections = vector_search(
        query_embedding,
        level="L2",
        contract_id=contract_id,
        top_k=5
    )

    # Stage 3: Fine-grained clause retrieval
    clause_candidates = []
    for section in relevant_sections:
        clauses = vector_search(
            query_embedding,
            level="L3",
            section_id=section.id,
            top_k=10
        )
        clause_candidates.extend(clauses)

    # Stage 4: Hybrid search for keyword coverage
    keyword_results = bm25_search(query, contract_id, top_k=10)

    # Stage 5: Rerank combined results
    all_candidates = deduplicate(clause_candidates + keyword_results)
    reranked = cross_encoder_rerank(query, all_candidates)

    return reranked[:k], contract_summary
```

**4. Citation Generation**

```python
GENERATION_PROMPT = """You are a legal contract analyst. Answer the question based ONLY on the provided contract excerpts.

Contract Summary:
{contract_summary}

Relevant Excerpts:
{formatted_excerpts}

Rules:
1. ALWAYS cite specific sections and page numbers
2. Use format: "According to Section X.Y (page N), ..."
3. If information is not in the excerpts, say "This information is not found in the provided sections"
4. Never infer or assume terms not explicitly stated

Question: {query}

Answer with citations:"""

def format_excerpt(chunk):
    return f"[Section {chunk.section} | Page {chunk.page}]\n{chunk.text}\n"
```

**5. Evaluation Approach**

| Metric | Method | Target |
|--------|--------|--------|
| Retrieval recall | Human-labeled relevant clauses | >90% |
| Citation accuracy | Verify cited page contains claim | >95% |
| Answer correctness | Legal expert review (sample) | >85% |
| Hallucination rate | Claims not in source | <5% |
| Latency | End-to-end timing | <30s |

**Evaluation Pipeline:**
```python
def evaluate_response(query, response, contract):
    scores = {}

    # 1. Citation verification
    citations = extract_citations(response)
    for citation in citations:
        page_text = get_page_text(contract, citation.page)
        scores['citation_accuracy'] = verify_claim_in_text(
            citation.claim, page_text
        )

    # 2. Completeness check
    relevant_clauses = human_labels[query]
    retrieved = extract_retrieved_sections(response)
    scores['recall'] = len(set(relevant_clauses) & set(retrieved)) / len(relevant_clauses)

    # 3. Hallucination detection
    claims = extract_claims(response)
    for claim in claims:
        if not claim_grounded_in_contract(claim, contract):
            scores['hallucination_detected'] = True

    return scores
```

---

## Problem 5: Debug/Fix - Generation Quality Issues

**Difficulty:** Diagnostic
**Estimated Time:** 20 minutes
**Concepts:** Prompt engineering, sampling, fine-tuning diagnostics

### Problem Statement

You are debugging a text-to-image generation system using Stable Diffusion. Users report various quality issues. For each scenario, identify the cause and propose a fix.

**Scenario A: Prompt Ignored**
```
Prompt: "A red sports car parked in front of the Eiffel Tower"
Result: Images consistently show blue or silver cars, tower is correctly rendered
Settings: CFG scale = 3.0, steps = 20
```

**Scenario B: Oversaturated/Artificial**
```
Prompt: "A professional headshot of a business executive"
Result: Colors are extremely vibrant, skin looks plastic, background is surreal
Settings: CFG scale = 15.0, steps = 50
```

**Scenario C: Inconsistent Faces**
```
Prompt: "A woman with blonde hair smiling" (same prompt, multiple generations)
Result: Every generation produces completely different facial features, ages vary wildly
Settings: CFG scale = 7.5, steps = 30, seed = random
```

**Scenario D: Blurry Output**
```
Prompt: "A detailed macro photograph of a butterfly wing"
Result: Image is soft and lacks fine details, looks like a watercolor
Settings: CFG scale = 7.5, steps = 10, negative prompt = ""
```

### Hints

<details>
<summary>Hint A</summary>
CFG scale determines how strongly the prompt influences generation. Very low values make the model ignore the prompt.
</details>

<details>
<summary>Hint B</summary>
Very high CFG causes over-amplification. Consider what happens to the guided prediction with extreme weights.
</details>

<details>
<summary>Hint C</summary>
Inconsistency without a fixed seed is expected. But for human faces, standard diffusion has known limitations.
</details>

<details>
<summary>Hint D</summary>
Consider the relationship between denoising steps and image quality. Fewer steps = less refinement.
</details>

### Solution

**Scenario A: Prompt Ignored - Diagnosis & Fix**

**Problem:** CFG scale = 3.0 is too low

**Explanation:**
```
CFG formula: ε̂ = ε_uncond + w × (ε_cond - ε_uncond)

With w = 3.0, the conditional signal is only amplified 3×.
This is insufficient for detailed attribute control like "red car".
Model defaults to common training distribution (silver/gray cars more common).
```

**Fix:**
```yaml
# Before
cfg_scale: 3.0
steps: 20

# After
cfg_scale: 7.5      # Standard, good prompt adherence
steps: 30           # Slightly more for better quality
```

**Additional tip:** Add to negative prompt: "blue car, silver car, gray car"

---

**Scenario B: Oversaturated/Artificial - Diagnosis & Fix**

**Problem:** CFG scale = 15.0 is too high

**Explanation:**
```
With w = 15.0, the guided prediction is heavily amplified:
ε̂ = ε_uncond + 15 × (ε_cond - ε_uncond)

This causes:
1. Color saturation pushed to extremes
2. Fine details overwhelmed by strong conditioning
3. Unnatural/artificial appearance
4. Loss of photorealism
```

**Fix:**
```yaml
# Before
cfg_scale: 15.0
steps: 50

# After
cfg_scale: 7.0-8.0    # Professional photos need moderate guidance
steps: 50             # Steps are fine

# Add specific negative prompts
negative_prompt: "oversaturated, cartoon, plastic, unrealistic, painting"

# Consider model choice
model: "realistic-vision" or "photon" (photorealistic fine-tunes)
```

---

**Scenario C: Inconsistent Faces - Diagnosis & Fix**

**Problem:** Multiple interacting issues

**Explanation:**
1. **Random seed:** Each generation starts from different noise
2. **Inherent diffusion limitation:** Standard models don't have face consistency
3. **Prompt lacks specificity:** "A woman" is too generic

**Fix (Multiple approaches):**

```yaml
# Approach 1: Fixed seed for reproducibility
seed: 42              # Same seed = same face
cfg_scale: 7.5
steps: 30

# Approach 2: More specific prompt
prompt: "A woman with blonde hair smiling, age 35-40, oval face shape,
         professional headshot style, consistent lighting"

# Approach 3: Use face-specialized model or LoRA
model: "realistic-vision-v5"
lora: "consistent-character-lora"

# Approach 4: Reference image (img2img or IP-Adapter)
# Use a reference face to maintain consistency
ip_adapter_scale: 0.5
reference_image: "target_face.jpg"
```

**Best practice for character consistency:**
- Use LoRA trained on specific person (with permission)
- Or use IP-Adapter with reference image
- Always fix seed when consistency is required

---

**Scenario D: Blurry Output - Diagnosis & Fix**

**Problem:** steps = 10 is far too few

**Explanation:**
```
Diffusion denoising is iterative refinement:
- Steps 1-10: Establish basic composition and colors
- Steps 11-25: Add medium details, proper shapes
- Steps 26-40: Fine details, textures, sharpness
- Steps 40+: Diminishing returns

With only 10 steps:
- Denoising is incomplete
- Fine details (butterfly wing patterns) never form
- Result looks "unfinished" or blurry
```

**Fix:**
```yaml
# Before
cfg_scale: 7.5
steps: 10
negative_prompt: ""

# After
cfg_scale: 7.5
steps: 40-50          # Macro photos need high detail
negative_prompt: "blurry, soft focus, low quality, watercolor, painting"

# Additional for macro photography
# Use high-res upscaler or detail-focused model
upscale: true
upscale_model: "4x-UltraSharp"

# Better sampler for fine details
sampler: "DPM++ 2M Karras"  # Better than Euler for details
```

**Step guidelines:**
| Image Type | Recommended Steps |
|------------|-------------------|
| Quick preview | 15-20 |
| Standard quality | 25-30 |
| High detail (macro, landscapes) | 40-50 |
| Maximum quality | 50-75 |

---

## Self-Assessment Guide

After completing these problems, evaluate your understanding:

| Level | Criteria |
|-------|----------|
| **Mastery** | Completed all problems correctly, identified multiple valid approaches |
| **Proficient** | Solved 4/5 problems, minor errors in complex scenarios |
| **Developing** | Solved warm-up and skill-builders, struggled with challenge/debug |
| **Foundational** | Need review of core concepts before attempting problems |

### Common Mistakes to Avoid

1. **Ignoring privacy constraints:** Using cloud APIs when data must stay local
2. **CFG extremes:** Too low ignores prompts, too high causes artifacts
3. **Insufficient steps:** Under 20 steps rarely produces quality images
4. **Memory miscalculation:** Forgetting optimizer states in VRAM estimation
5. **Missing citations:** RAG systems must track and verify sources

---

*Generated from Lesson 7: Generative AI | Practice Problems Skill*
