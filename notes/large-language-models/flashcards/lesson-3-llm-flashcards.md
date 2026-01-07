# Flashcard Set: Lesson 3 - Large Language Models

**Source:** Lessons/Lesson_3.md
**Subject Area:** AI Learning - Large Language Models: Architecture, Training, and Capabilities
**Date Generated:** 2026-01-08
**Total Cards:** 5
**Distribution:** 2 Easy | 2 Medium | 1 Hard

---

## Critical Knowledge Summary

Concepts appearing in multiple cards (prioritize for review):
- **Self-Attention**: Appears in Cards 1, 2, 4 (computational foundation)
- **Training Pipeline**: Appears in Cards 2, 3, 5 (capability development)
- **Context Windows**: Appears in Cards 1, 4, 5 (fundamental constraint)
- **Hallucination**: Appears in Cards 3, 5 (critical limitation)

---

## Flashcards

---
### Card 1 | Easy
**Cognitive Level:** Remember
**Concept:** Transformer Architecture and Self-Attention
**Source Section:** Core Concepts - Concepts 1 & 2

**FRONT (Question):**
What are the three learned projections in the self-attention mechanism, and what question does each projection conceptually answer?

**BACK (Answer):**
The three learned projections in self-attention are **Query (Q)**, **Key (K)**, and **Value (V)**:

| Projection | Conceptual Question | Role |
|------------|---------------------|------|
| **Query (Q)** | "What information am I looking for?" | Represents what the current position needs |
| **Key (K)** | "What information do I contain?" | Represents what each position offers |
| **Value (V)** | "What information do I provide when attended to?" | Contains the actual information to retrieve |

**Computation Flow:**
1. Query-Key dot product → compatibility scores
2. Scale by √d_k → prevents vanishing gradients in softmax
3. Softmax → probability distribution (attention weights)
4. Weighted sum of Values → output representation

**Key Formula:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**Why This Matters:** Understanding Q/K/V is essential for interpreting attention patterns and diagnosing model behavior.

**Critical Knowledge Flag:** Yes - Foundation of all Transformer-based models

---

---
### Card 2 | Easy
**Cognitive Level:** Understand
**Concept:** Training Pipeline Stages
**Source Section:** Core Concepts - Concept 3

**FRONT (Question):**
Describe the three stages of the LLM training pipeline (Pre-training → SFT → RLHF). What specific capability does each stage develop?

**BACK (Answer):**
**Three-Stage Training Pipeline:**

| Stage | Method | Data | Capability Developed |
|-------|--------|------|---------------------|
| **Pre-training** | Self-supervised next-token prediction | Trillions of tokens (web, books, code) | Language patterns, factual knowledge, reasoning foundations |
| **SFT** | Supervised learning on curated pairs | (instruction, response) examples | Instruction following, helpful responses, dialogue capability |
| **RLHF** | Reward model + policy optimization | Human preference comparisons | Alignment with human preferences (helpful, honest, harmless) |

**Why This Ordering:**
1. Pre-training creates raw capability but produces a "completion" model
2. SFT teaches *how* to respond helpfully to users
3. RLHF optimizes for nuanced preferences hard to specify in examples

**Key Insight:** Each stage builds on the previous—you cannot skip stages. A model without pre-training has no knowledge; without SFT it cannot follow instructions; without RLHF it may be helpful but misaligned.

**Parameter-Efficient Option:** LoRA (Low-Rank Adaptation) enables fine-tuning by updating small matrices, reducing compute and preventing catastrophic forgetting.

**Critical Knowledge Flag:** Yes - Understanding training explains model behavior

---

---
### Card 3 | Medium
**Cognitive Level:** Apply
**Concept:** Hallucination Mitigation Strategies
**Source Section:** Core Concepts - Concept 6, Practical Applications

**FRONT (Question):**
You're building a legal document analysis system using an LLM. The system must extract contract clauses and identify potential risks. Given that hallucination could have serious legal consequences, design three specific mitigation strategies and explain how each addresses hallucination risk.

**BACK (Answer):**
**Three Hallucination Mitigation Strategies:**

| Strategy | Implementation | How It Addresses Hallucination |
|----------|----------------|-------------------------------|
| **1. RAG with Citation** | Retrieve relevant contract sections before generation; require model to cite specific line numbers for each extracted clause | Grounds outputs in source text; enables verification; ungrounded claims become obvious |
| **2. Structured Extraction** | Use strict JSON schema requiring `clause_text`, `source_location`, and `confidence` fields; reject outputs that don't match schema | Forces model to extract rather than generate; malformed outputs filtered automatically |
| **3. Cross-Validation** | Run extraction multiple times with temperature variation; flag clauses appearing in <80% of runs; require unanimous agreement for high-risk items | Exploits that hallucinations are inconsistent while true extractions are stable |

**Additional Layer - Human Oversight:**
```
Classification         Review Requirement
─────────────────────────────────────────
High-risk clauses  →   Always human review
Inconsistent items →   Human verification
High confidence    →   Spot-check sampling
```

**Why Multiple Layers:** No single strategy eliminates hallucination. Defense in depth catches different failure modes:
- RAG catches fabrication of facts
- Schema catches structural violations
- Cross-validation catches inconsistent generations

**Critical Knowledge Flag:** Yes - Hallucination mitigation is essential for production systems

---

---
### Card 4 | Medium
**Cognitive Level:** Analyze
**Concept:** Inference Optimization Tradeoffs
**Source Section:** Core Concepts - Concept 5

**FRONT (Question):**
Analyze the tradeoffs between three inference optimizations: KV caching, quantization, and batching. For a customer-facing chatbot requiring <500ms time-to-first-token and 1000 requests/minute throughput, recommend which optimizations to prioritize and justify your choices.

**BACK (Answer):**
**Optimization Analysis:**

| Optimization | Benefit | Cost | Tradeoff |
|--------------|---------|------|----------|
| **KV Caching** | Reduces generation from O(n²) to O(n) complexity | Memory: must store K/V for full context per request | Essential—without it, generation is impractically slow |
| **Quantization** | 2-4x faster inference, reduced memory | Quality degradation (varies: INT8 minimal, INT4 noticeable) | Good ROI—INT8 typically has <1% quality loss |
| **Batching** | Higher throughput (concurrent requests) | Increased latency per request | Tension with TTFT requirement |

**Recommendation for Given Requirements:**

| Priority | Optimization | Rationale |
|----------|--------------|-----------|
| **1st** | KV Caching | Non-negotiable—enables acceptable generation speed |
| **2nd** | Quantization (INT8) | Achieves latency target with minimal quality loss |
| **3rd** | Dynamic Batching | Batch small groups (4-8) when queue builds; prioritize single requests when queue empty |

**Key Metrics Mapping:**
- **TTFT (<500ms):** KV cache + quantization directly reduce; batching increases
- **Throughput (1000 req/min):** Batching directly increases; quantization helps by freeing GPU memory

**Architectural Decision:**
```
IF queue_depth < 4:
    Process immediately (minimize TTFT)
ELSE:
    Batch up to 8 requests (maximize throughput)
```

**Critical Knowledge Flag:** Yes - Production deployment requires understanding these tradeoffs

---

---
### Card 5 | Hard
**Cognitive Level:** Synthesize
**Concept:** Complete LLM System Design
**Source Section:** All Core Concepts, Theoretical Framework, Critical Analysis

**FRONT (Question):**
Synthesize a complete design for an LLM-powered research assistant that helps scientists review academic literature. Your design must address:

1. **Architecture:** How will the system handle papers exceeding context window limits?
2. **Training/Customization:** What approach for domain adaptation without full fine-tuning?
3. **Reliability:** How will you handle hallucination for scientific claims?
4. **User Experience:** How will you communicate uncertainty to researchers?
5. **Evaluation:** How will you measure system quality?

**BACK (Answer):**
**1. Architecture for Long Documents:**

```
┌────────────────────────────────────────────────────────────────┐
│                      DOCUMENT PROCESSING                        │
├────────────────────────────────────────────────────────────────┤
│  Paper → Semantic Chunking → Vector Store → Retrieval          │
│                                                                 │
│  Chunking Strategy:                                             │
│  - Section-aware splitting (Abstract, Methods, Results, etc.)  │
│  - Overlap windows preserve cross-chunk context                 │
│  - Metadata preserved: section type, citations, figures         │
│                                                                 │
│  Query Flow:                                                    │
│  User Query → Embed → Retrieve top-k chunks → LLM synthesis    │
└────────────────────────────────────────────────────────────────┘
```

**Why Semantic Chunking:** Scientific papers have structured sections with distinct information types. Respecting boundaries prevents retrieving fragments that lack necessary context.

**2. Domain Adaptation Without Fine-Tuning:**

| Approach | Implementation | Rationale |
|----------|----------------|-----------|
| **RAG** | Index domain literature; retrieve at inference | Adds factual grounding without training |
| **System Prompt** | Scientific conventions, citation format, skeptical stance | Shapes behavior without parameter updates |
| **Few-Shot Examples** | 2-3 high-quality analysis demonstrations | Teaches domain-specific reasoning patterns |
| **LoRA** (if needed) | Train on curated scientific Q&A pairs | Parameter-efficient if RAG insufficient |

**Preference Order:** RAG + System Prompt → Few-Shot → LoRA (escalate only if simpler approaches fail)

**3. Hallucination Mitigation for Scientific Claims:**

| Claim Type | Mitigation Strategy |
|------------|---------------------|
| **Factual (statistics, results)** | Require verbatim quotes with page numbers; verify against source |
| **Methodological** | Cross-reference Methods section; flag if not explicitly stated |
| **Interpretive (implications)** | Label as "interpretation" with supporting evidence |
| **Comparative (vs. other work)** | Require both papers in context; flag if comparison paper not retrieved |

**Architecture:**
```
User Query → Retrieve Sources → Generate with Citations
                                       ↓
                            Verification Layer
                         (Check quotes exist in source)
                                       ↓
                   Confidence Classification (High/Medium/Low)
                                       ↓
                            Response + Uncertainty Signals
```

**4. Communicating Uncertainty:**

| Confidence | Visual Signal | Behavior |
|------------|---------------|----------|
| **High** | Green indicator | Verbatim extraction with source link |
| **Medium** | Yellow indicator | "The paper suggests..." + source |
| **Low** | Orange indicator | "Based on limited context..." + explicit caveats |
| **Cannot Determine** | Red indicator | "I couldn't find this in the retrieved papers. This claim requires verification." |

**Key Principle:** Never present uncertain information with the same formatting as grounded information. Visual differentiation enables appropriate researcher trust calibration.

**5. Evaluation Framework:**

| Dimension | Metric | Measurement |
|-----------|--------|-------------|
| **Retrieval Quality** | Recall@10, MRR | Does retrieval find relevant sections? |
| **Extraction Accuracy** | Token-level F1 against ground truth | Are quotes accurate? |
| **Hallucination Rate** | % claims without source grounding | Manual audit sample |
| **Usefulness** | Researcher task completion time, satisfaction | User studies |
| **Calibration** | Correlation between confidence and accuracy | Binned accuracy analysis |

**Evaluation Protocol:**
1. Curate test set: 50 papers across domains with expert annotations
2. Automated metrics: retrieval, extraction, response time
3. Human evaluation: 100 randomly sampled responses rated by domain experts
4. Calibration analysis: group by confidence level, measure accuracy per group

**Critical Knowledge Flag:** Yes - Integrates all LLM concepts into production system design

---

---

## Export Formats

### Anki-Compatible (Tab-Separated)
```
What are Q/K/V projections in self-attention?	Query (what am I looking for?), Key (what do I contain?), Value (what do I provide?). Attention = softmax(QK^T/√d_k)V	easy::architecture::llm
Describe the three-stage LLM training pipeline	Pre-training (next-token, trillions tokens) → SFT (instruction pairs) → RLHF (preference alignment). Each stage: capability → helpfulness → alignment.	easy::training::llm
Design hallucination mitigation for legal document analysis	1. RAG with citations (ground in source), 2. Structured extraction (schema enforcement), 3. Cross-validation (consistency checking). Defense in depth.	medium::reliability::llm
Analyze inference optimization tradeoffs for chatbot	KV cache (essential), Quantization (good ROI), Batching (throughput vs latency). Dynamic batching: batch when queue builds, single when empty.	medium::deployment::llm
Design LLM research assistant for scientific literature	Semantic chunking for long docs, RAG+prompting for adaptation, multi-layer hallucination mitigation, visual confidence signals, 5-dimension evaluation.	hard::synthesis::llm
```

### CSV Format
```csv
"Front","Back","Difficulty","Concept","Cognitive_Level"
"What are Q/K/V projections in self-attention?","Query, Key, Value projections. Q=what I seek, K=what I offer, V=what I provide. Attention = softmax(QK^T/√d_k)V","Easy","Self-Attention","Remember"
"Describe the three-stage LLM training pipeline","Pre-training → SFT → RLHF. Develops: capability, helpfulness, alignment respectively. Each builds on previous.","Easy","Training Pipeline","Understand"
"Design hallucination mitigation for legal analysis","RAG+citations, structured extraction, cross-validation. Defense in depth catches different failure modes.","Medium","Hallucination","Apply"
"Analyze inference optimizations for production chatbot","KV cache (essential), quantization (good ROI), dynamic batching (balance TTFT vs throughput).","Medium","Inference","Analyze"
"Design complete LLM research assistant","Semantic chunking, RAG+prompting, multi-layer verification, visual confidence, 5D evaluation framework.","Hard","System Design","Synthesize"
```

---

## Source Mapping

| Card | Source Section | Key Terminology | Bloom's Level |
|------|----------------|-----------------|---------------|
| 1 | Core Concepts - Concepts 1 & 2 | Query, Key, Value, self-attention, √d_k scaling | Remember |
| 2 | Core Concepts - Concept 3 | Pre-training, SFT, RLHF, LoRA | Understand |
| 3 | Core Concepts - Concept 6, Applications | Hallucination, RAG, validation, cross-validation | Apply |
| 4 | Core Concepts - Concept 5 | KV cache, quantization, batching, TTFT | Analyze |
| 5 | All Core Concepts + Framework | Chunking, RAG, confidence calibration, evaluation | Synthesize |

---

## Spaced Repetition Schedule

| Card | Initial Interval | Difficulty Multiplier | Recommended Review |
|------|------------------|----------------------|-------------------|
| 1 (Easy) | 1 day | 2.5x | Foundation - review first |
| 2 (Easy) | 1 day | 2.5x | Foundation - review with Card 1 |
| 3 (Medium) | 3 days | 2.0x | After mastering Cards 1-2 |
| 4 (Medium) | 3 days | 2.0x | Requires production context |
| 5 (Hard) | 7 days | 1.5x | Review after all others mastered |

---

## Connection to Previous Lessons

| LLM Concept | Prompt Engineering (Lesson 2) | Agent Skills (Lesson 1) |
|-------------|-------------------------------|-------------------------|
| Self-Attention | Explains why context placement matters (primacy/recency) | Skills must respect attention patterns |
| Context Windows | Fundamental constraint on prompt length | Skills must chunk within limits |
| Training Pipeline | Explains why instruction formatting works | Skills leverage SFT conventions |
| Hallucination | Motivates validation in prompt chains | Skills need verification layers |
| Inference | Explains latency considerations | Skills must optimize for production |

---

*Generated from Lesson 3: Large Language Models | Flashcards Skill*
