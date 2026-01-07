# Assessment Quiz: Lesson 3 - Large Language Models

**Source:** Lessons/Lesson_3.md
**Subject Area:** AI Learning - Large Language Models: Architecture, Training, and Capabilities
**Date Generated:** 2026-01-08
**Total Questions:** 5
**Distribution:** 2 Multiple Choice | 2 Short Answer | 1 Essay

---

## Quiz Overview

| # | Type | Concept | Cognitive Level | Points |
|---|------|---------|-----------------|--------|
| 1 | Multiple Choice | Self-Attention Mechanism | Remember/Understand | 10 |
| 2 | Multiple Choice | Training Pipeline | Understand | 10 |
| 3 | Short Answer | Context Windows & Inference | Apply/Analyze | 20 |
| 4 | Short Answer | Hallucination & Limitations | Analyze | 20 |
| 5 | Essay | Complete LLM System Design | Synthesize/Evaluate | 40 |

**Total Points:** 100
**Recommended Time:** 45-60 minutes

---

## Questions

---

### Question 1 | Multiple Choice
**Concept:** Self-Attention Mechanism
**Cognitive Level:** Remember/Understand
**Points:** 10

In the self-attention mechanism, what is the purpose of scaling the Query-Key dot product by √d_k before applying softmax?

**A.** To increase the magnitude of attention weights for better gradient flow

**B.** To prevent large dot products from pushing softmax into regions with extremely small gradients

**C.** To normalize the attention weights so they sum to exactly 1.0

**D.** To reduce computational complexity from O(n²) to O(n log n)

---

### Question 2 | Multiple Choice
**Concept:** Training Pipeline (Pre-training → SFT → RLHF)
**Cognitive Level:** Understand
**Points:** 10

A research team has a pre-trained LLM that performs well on general language tasks but gives unhelpful responses to user instructions. They want to make it more helpful while maintaining its broad knowledge. Which training approach should they apply FIRST?

**A.** RLHF with human preference comparisons to optimize for helpfulness

**B.** Continued pre-training on a larger, more diverse text corpus

**C.** Supervised fine-tuning (SFT) on high-quality instruction-response pairs

**D.** LoRA fine-tuning directly on user feedback from production logs

---

### Question 3 | Short Answer
**Concept:** Context Windows and Inference Optimization
**Cognitive Level:** Apply/Analyze
**Points:** 20

**Scenario:** You are deploying a 13B parameter LLM for a document summarization service. Each document is approximately 50,000 tokens, but your model has a 4,096 token context window.

**Questions:**

**(a)** Explain why you cannot simply feed the entire document to the model. What is the fundamental constraint? (5 points)

**(b)** Describe TWO different architectural approaches to handle documents exceeding the context window. For each approach, explain the mechanism and identify one significant tradeoff. (10 points)

**(c)** The service must handle 100 documents per hour with a maximum latency of 30 seconds per document. Your single GPU can generate 50 tokens per second. Calculate whether this is achievable for 500-token summaries, and identify the primary bottleneck if it is not. (5 points)

---

### Question 4 | Short Answer
**Concept:** Capabilities and Limitations (Hallucination)
**Cognitive Level:** Analyze
**Points:** 20

**Scenario:** A legal technology company deployed an LLM to help lawyers research case law. After two weeks, they discovered the following issues:

1. The model cited "Smith v. Johnson (2019)" with a specific ruling—but this case does not exist
2. For questions about recent 2024 cases, the model confidently provided incorrect holdings
3. When asked about a 1954 case, the model gave accurate information

**Questions:**

**(a)** For each of the three issues, identify the specific LLM limitation at play. Use precise terminology from the lesson. (6 points)

**(b)** Explain why issue #3 was accurate while issues #1 and #2 were not. What does this reveal about the nature of LLM "knowledge"? (6 points)

**(c)** Design a two-layer validation system that would catch issues #1 and #2 before the response reaches the user. Be specific about what each layer checks and how. (8 points)

---

### Question 5 | Essay
**Concept:** Complete LLM System Design
**Cognitive Level:** Synthesize/Evaluate
**Points:** 40

**Prompt:**

Design a production-ready LLM system for a healthcare organization that needs to automatically generate clinical documentation from doctor-patient conversation transcripts.

**Requirements:**
- Input: Audio transcripts of 15-30 minute patient consultations
- Output: Structured clinical notes (chief complaint, history, assessment, plan)
- Accuracy: Must not hallucinate symptoms, diagnoses, or treatments
- Compliance: Must be auditable (show what information came from where)
- Scale: 500 consultations per day

**Your essay must address:**

1. **Architecture Design (10 points)**
   - Model selection and deployment considerations
   - Pipeline stages from transcript to clinical note
   - How you handle transcripts exceeding context windows

2. **Reliability Engineering (10 points)**
   - Specific hallucination mitigation strategies for medical context
   - How you ensure auditability (tracing outputs to transcript sources)
   - Validation mechanisms before clinical note delivery

3. **Training and Adaptation (10 points)**
   - How you would adapt a base LLM for medical documentation
   - Feedback loop for continuous improvement
   - How you handle specialty-specific terminology and formats

4. **Critical Evaluation (10 points)**
   - What are the remaining risks even with your mitigations?
   - Where would you require human review vs. automated delivery?
   - How would you evaluate whether this system is safe to deploy?

**Evaluation Criteria:**
- Technical accuracy and depth of LLM understanding
- Practical feasibility of proposed solutions
- Appropriate concern for safety in medical context
- Integration of concepts across all lesson sections

---

## Answer Key

---

### Question 1 | Answer

**Correct Answer: B**

**Explanation:**
The scaling factor √d_k prevents the dot products from becoming too large. When dot products are very large, the softmax function pushes most of its probability mass onto a single position (approaching a one-hot vector). In this regime:
- Gradients become extremely small (vanishing gradient problem)
- The model cannot effectively learn to adjust attention patterns
- Attention becomes "hard" rather than "soft," reducing expressiveness

**Why other options are wrong:**
- **A** is backwards—scaling *reduces* magnitude, not increases it
- **C** describes what softmax does naturally, not the purpose of scaling
- **D** is incorrect—scaling doesn't change the asymptotic complexity

**Source Reference:** Core Concepts - Concept 2: Self-Attention Mechanism

**Understanding Gap Indicator:** If this was missed, review the mathematical formulation of attention: Attention(Q, K, V) = softmax(QK^T / √d_k) V, paying attention to why each component exists.

---

### Question 2 | Answer

**Correct Answer: C**

**Explanation:**
The training pipeline follows a specific order for good reason:
1. **Pre-training** (already done) → Develops broad language capability
2. **SFT** (needed first) → Teaches the model to follow instructions and be helpful
3. **RLHF** (after SFT) → Refines helpfulness based on nuanced preferences

SFT must come before RLHF because:
- RLHF optimizes *how* the model responds, but it needs a baseline that can already respond to instructions
- Without SFT, the model is a "completion" model that continues text rather than responding helpfully
- RLHF on an unaligned model would be optimizing the wrong behavior

**Why other options are wrong:**
- **A** (RLHF first) skips the necessary instruction-following foundation
- **B** (more pre-training) won't help—the problem is format, not knowledge
- **D** (LoRA on production logs) conflates SFT and RLHF, and production logs are noisy

**Source Reference:** Core Concepts - Concept 3: Training Pipeline

**Understanding Gap Indicator:** If this was missed, review why each training stage exists and what specific capability it develops. The key insight is that stages are sequential and build on each other.

---

### Question 3 | Answer

**(a) Context Window Constraint (5 points)**

**Full Credit Answer:**
The fundamental constraint is the O(n²) complexity of self-attention with respect to sequence length. Self-attention computes pairwise interactions between all tokens, meaning:
- 50,000 tokens would require 50,000 × 50,000 = 2.5 billion attention computations per layer
- Memory requirements grow quadratically—storing attention matrices for 50K tokens is infeasible
- The 4,096 token limit is an architectural constraint, not arbitrary

Additionally, even if computation were possible, the "lost in the middle" phenomenon means attention quality degrades for very long contexts.

**Partial Credit:** Mentioning memory/compute constraints without O(n²) explanation (3 points)

---

**(b) Two Approaches to Handle Long Documents (10 points)**

**Full Credit Answer:**

| Approach | Mechanism | Tradeoff |
|----------|-----------|----------|
| **1. Chunking with Hierarchical Summarization** | Split document into overlapping chunks within context limit. Summarize each chunk. Then summarize the summaries. Repeat until single output. | **Tradeoff:** Information loss at each summarization level. Cannot capture relationships between distant chunks. Final summary may miss nuances present in original. |
| **2. Retrieval-Augmented Generation (RAG)** | Embed all chunks into vector store. When generating summary, retrieve most relevant chunks for each section. Summarize with retrieved context. | **Tradeoff:** Retrieval may miss relevant information not semantically similar to query. More complex pipeline. Requires good embedding model and retrieval tuning. |

**Alternative acceptable approaches:**
- Sparse attention mechanisms (Longformer, BigBird)
- Sliding window attention with global tokens
- Map-reduce summarization

**Partial Credit:** One complete approach with tradeoff (5 points); two approaches without tradeoffs (6 points)

---

**(c) Throughput Calculation (5 points)**

**Full Credit Answer:**

```
Requirements:
- 100 documents/hour = 100/60 ≈ 1.67 documents/minute
- Max latency: 30 seconds/document
- Summary length: 500 tokens
- GPU speed: 50 tokens/second

Calculation for single document (chunking approach):
- 50,000 tokens ÷ 4,096 = ~13 chunks (with overlap, ~15 chunks)
- Per chunk summary: ~100 tokens generation = 2 seconds
- 15 chunks × 2 seconds = 30 seconds for first level
- Hierarchical pass: 15 × 100 = 1,500 tokens → 4 chunks → 8 seconds
- Final summary: ~6 seconds
- Total: ~44 seconds per document

Result: NOT achievable with single GPU

Bottleneck: Generation throughput. Need either:
- Multiple GPUs (2 would suffice with some margin)
- Parallel chunk processing
- Faster model (smaller or quantized)
```

**Partial Credit:** Correct methodology with minor calculation errors (3-4 points)

**Source Reference:** Core Concepts - Concepts 4 (Context Windows) and 5 (Inference)

---

### Question 4 | Answer

**(a) LLM Limitations Identification (6 points)**

| Issue | LLM Limitation | Explanation |
|-------|----------------|-------------|
| **#1: Fake citation** | **Hallucination** | Model generated plausible-sounding but entirely fabricated case citation with confident specificity |
| **#2: Wrong 2024 info** | **Knowledge cutoff** + **Hallucination** | Model's training data doesn't include 2024 cases; it fabricated an answer rather than indicating uncertainty |
| **#3: Accurate 1954 info** | (Not a limitation) | 1954 case was in training data; demonstrates model can recall training information accurately |

**Full Credit:** Correctly identifies hallucination for #1, knowledge cutoff + hallucination for #2, and correctly notes #3 is not a limitation (2 points each)

---

**(b) Understanding LLM "Knowledge" (6 points)**

**Full Credit Answer:**

The contrast between #3 (accurate) and #1/#2 (inaccurate) reveals crucial insights about LLM knowledge:

1. **Pattern Reproduction vs. True Knowledge:** The model doesn't "know" law—it reproduces patterns from training data. The 1954 case was likely discussed in training texts; the model can retrieve those patterns. "Smith v. Johnson 2019" was fabricated by combining common legal name patterns.

2. **No Uncertainty Awareness:** The model cannot distinguish between:
   - Information it saw repeatedly in training (high confidence appropriate)
   - Information it saw rarely (low confidence appropriate)
   - Information it never saw (should abstain entirely)

   It generates with equal confidence regardless.

3. **Confabulation over Abstention:** When information is outside training, LLMs default to generating plausible content rather than admitting ignorance. This is because training optimized for coherent, helpful responses—not for calibrated uncertainty.

4. **Temporal Blindness:** The model has no concept of "when" it learned information or whether that information is current. It processes all queries as if asking about timeless facts.

**Partial Credit:** Explains pattern matching (2 points), mentions lack of uncertainty (2 points), discusses confabulation (2 points)

---

**(c) Two-Layer Validation System (8 points)**

**Full Credit Answer:**

```
┌─────────────────────────────────────────────────────────────────┐
│ LAYER 1: Citation Verification                                  │
│ ──────────────────────────────                                  │
│                                                                 │
│ For each case citation in response:                             │
│ 1. Extract: case name, year, court, citation number            │
│ 2. Query authoritative legal database (Westlaw, LexisNexis)    │
│ 3. Verification outcomes:                                       │
│    ├── FOUND + MATCHES → Citation verified ✓                   │
│    ├── FOUND + DIFFERS → Flag discrepancy ⚠                    │
│    └── NOT FOUND → Flag as potentially fabricated ✗            │
│                                                                 │
│ Action: Block response if any citation fails verification       │
│ Catches: Issue #1 (fake citations)                              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ LAYER 2: Temporal Validity Check                                │
│ ─────────────────────────────────                               │
│                                                                 │
│ For questions about recent events/cases:                        │
│ 1. Detect temporal references in query (dates, "recent", etc.) │
│ 2. Compare against model's known training cutoff               │
│ 3. If query references post-cutoff date:                       │
│    ├── Flag response as potentially outdated                   │
│    ├── Add disclaimer: "Model knowledge may not include..."    │
│    └── Require: RAG with current database before answering     │
│                                                                 │
│ Action: Supplement with current data or add clear disclaimer   │
│ Catches: Issue #2 (wrong recent information)                   │
└─────────────────────────────────────────────────────────────────┘

Combined System Flow:
Query → LLM Response → Layer 1 (citation check) → Layer 2 (temporal check) →
  → If all pass: Deliver with confidence
  → If Layer 1 fails: Block + regenerate with RAG
  → If Layer 2 flags: Add disclaimer + supplement with current data
```

**Partial Credit:** One layer fully described (4 points); both layers mentioned but lacking detail (5-6 points)

**Source Reference:** Core Concepts - Concept 6: Capabilities and Limitations, Critical Analysis

---

### Question 5 | Essay Rubric

**Total: 40 points**

---

**Section 1: Architecture Design (10 points)**

| Points | Criteria |
|--------|----------|
| 9-10 | Comprehensive architecture with appropriate model selection (medical-capable open model or fine-tuned), complete pipeline from audio to note, practical chunking strategy for long transcripts, deployment considerations (on-premise for HIPAA if applicable) |
| 7-8 | Solid architecture with most components; may lack depth in one area |
| 5-6 | Basic architecture present but missing key considerations (e.g., no chunking strategy, unclear model choice) |
| 3-4 | Incomplete architecture; significant gaps in pipeline design |
| 0-2 | Minimal or incorrect architectural thinking |

**Key Elements for Full Credit:**
- Model: Code/medical-specialized (e.g., Med-PaLM, fine-tuned Llama with medical data)
- Pipeline: Audio → Transcript → Chunking by speaker turns → Extraction per section → Aggregation → Structured output
- Chunking: By consultation phases (chief complaint, history, exam, assessment) or fixed speaker turns with overlap
- Context handling: Either RAG or hierarchical processing for 30-minute transcripts

---

**Section 2: Reliability Engineering (10 points)**

| Points | Criteria |
|--------|----------|
| 9-10 | Multiple specific hallucination mitigations appropriate for medical context, clear auditability with source tracing, multi-layer validation before delivery |
| 7-8 | Good reliability design with minor gaps |
| 5-6 | Basic reliability considerations but lacking medical-specific concerns |
| 3-4 | Mentions reliability but no concrete mechanisms |
| 0-2 | No meaningful reliability discussion |

**Key Elements for Full Credit:**
- Hallucination mitigation: Extraction-only mode (no generation of new clinical info), citation requirements, confidence thresholds
- Auditability: Every output field linked to transcript timestamp/quote, audit log of all processing steps
- Validation: Structured output parsing, medical terminology verification, contradiction detection within note

---

**Section 3: Training and Adaptation (10 points)**

| Points | Criteria |
|--------|----------|
| 9-10 | Realistic adaptation strategy (LoRA on medical transcription data), well-designed feedback loop, handles specialty variation |
| 7-8 | Solid training approach with minor gaps |
| 5-6 | Basic adaptation mentioned but lacking practical detail |
| 3-4 | Vague training discussion |
| 0-2 | No meaningful training strategy |

**Key Elements for Full Credit:**
- Adaptation: SFT on (transcript, clinical note) pairs from existing documentation, LoRA for efficiency
- Feedback: Physician correction interface, accepted/rejected edits, periodic adapter updates
- Specialties: Either specialty-specific adapters or specialty-aware prompting with examples

---

**Section 4: Critical Evaluation (10 points)**

| Points | Criteria |
|--------|----------|
| 9-10 | Thoughtful analysis of remaining risks, appropriate human review criteria, comprehensive evaluation plan including safety |
| 7-8 | Good critical analysis with minor omissions |
| 5-6 | Some critical thinking but missing key concerns |
| 3-4 | Superficial evaluation |
| 0-2 | No meaningful critical analysis |

**Key Elements for Full Credit:**
- Remaining risks: Rare conditions, transcript errors propagating, liability concerns, physician over-reliance
- Human review: All notes requiring medication changes, any flagged uncertainties, random sampling for quality
- Evaluation: Agreement with physician-written notes, error rate measurement, clinical outcome tracking, pilot before full deployment

---

**Sample High-Scoring Essay Excerpt:**

> "For the architecture, I propose a three-stage pipeline using a fine-tuned Llama-3 70B model deployed on-premise to satisfy HIPAA requirements. Stage 1 segments the transcript by speaker and consultation phase using a specialized diarization model. Stage 2 processes each segment with extraction prompts—one per clinical note section—requiring verbatim quotes from the transcript for every clinical claim. Stage 3 aggregates extractions into the final structured note with cross-reference validation.
>
> The critical reliability challenge is ensuring no hallucinated clinical information. Unlike general summarization where plausible-sounding content is merely inconvenient, fabricated symptoms or diagnoses could harm patients. My mitigation strategy operates at three levels: (1) Prompt-level: The model is instructed to ONLY extract, never infer—if information isn't explicitly stated, the field is marked 'Not documented in transcript.' (2) Output-level: Every clinical term must include a timestamp reference (e.g., 'patient reports chest pain [3:42]'), enabling one-click verification. (3) System-level: A validation layer checks that all referenced timestamps exist and that extracted quotes actually appear in the transcript.
>
> Even with these mitigations, risks remain. The system cannot catch errors in the original transcript (speech recognition mistakes), may miss implied information that physicians would understand from context, and creates liability ambiguity (who is responsible for the note—the AI or the reviewing physician?). I would require human review for all notes before finalization during an initial 6-month pilot, with graduated autonomy only after demonstrating <2% clinically significant error rate..."

---

## Performance Interpretation

### Score Ranges

| Score | Level | Interpretation |
|-------|-------|----------------|
| 90-100 | Mastery | Ready to design production LLM systems; understands architecture, training, and limitations deeply |
| 80-89 | Proficient | Strong understanding; minor gaps in integration or critical analysis |
| 70-79 | Competent | Solid foundational knowledge; needs more practice with complex system design |
| 60-69 | Developing | Understands core concepts but struggles with application and integration |
| Below 60 | Foundational | Needs to review core concepts before attempting system-level thinking |

### Recommended Review by Question

| If you struggled with... | Review these sections... |
|--------------------------|--------------------------|
| Q1 (Self-Attention) | Core Concept 2, Flashcard 1 |
| Q2 (Training Pipeline) | Core Concept 3, Flashcard 2 |
| Q3 (Context/Inference) | Core Concepts 4 & 5, Practice Problems 1 & 3 |
| Q4 (Hallucination) | Core Concept 6, Practice Problem 5 |
| Q5 (System Design) | All Core Concepts, Case Study, Practice Problem 4 |

---

## Cross-Lesson Connections

This quiz assesses readiness to apply LLM knowledge in:

| Downstream Application | Required Understanding |
|------------------------|------------------------|
| **Prompt Engineering (Lesson 2)** | Why prompt structure affects attention (Q1), why instructions work (Q2) |
| **Agent Skills (Lesson 1)** | How to build reliable capabilities on unreliable LLMs (Q4, Q5) |
| **Production Systems** | How to deploy, scale, and validate LLM applications (Q3, Q5) |

---

*Generated from Lesson 3: Large Language Models | Quiz Skill*
