# Flashcard Set: Large Language Models

**Source:** notes/large-language-models/large-language-models-study-notes.md
**Concept Map Reference:** notes/large-language-models/concept-maps/large-language-models-concept-map.md
**Date Generated:** 2026-01-06
**Total Cards:** 5
**Distribution:** 2 Easy | 2 Medium | 1 Hard

---

## Critical Knowledge Summary

Concepts appearing in multiple cards (prioritize for review):
- **Transformer/Self-Attention**: Appears in Cards 1, 4, 5 (core architecture)
- **Pre-training/RLHF**: Appears in Cards 2, 5 (training pipeline)
- **Prompting**: Appears in Cards 3, 5 (interaction method)
- **Hallucination**: Appears in Cards 2, 5 (critical limitation)
- **Context Window**: Appears in Cards 4, 5 (deployment constraint)

---

## Flashcards

---
### Card 1 | Easy
**Cognitive Level:** Remember/Understand
**Concept:** Transformer Architecture and Self-Attention
**Source Section:** Core Concepts 1, 2, 3
**Concept Map Centrality:** Transformer (9), Self-Attention (7)

**FRONT (Question):**
What is the Transformer architecture, and how does self-attention enable it to process sequences differently from RNNs?

**BACK (Answer):**
**Transformer Architecture:**
A neural network that processes sequences using **self-attention** mechanisms instead of recurrence, enabling parallel computation and direct modeling of relationships between any positions.

**Self-Attention Mechanism:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```
- **Query (Q):** "What am I looking for?"
- **Key (K):** "What do I contain?"
- **Value (V):** "What information do I provide?"

**Key Differences from RNNs:**

| Aspect | RNN | Transformer |
|--------|-----|-------------|
| Processing | Sequential (one token at a time) | Parallel (all tokens simultaneously) |
| Long-range deps | Information decays over distance | Direct attention to any position |
| Gradients | Vanishing/exploding over steps | Stable (no sequential path) |
| GPU efficiency | Poor (sequential bottleneck) | Excellent (parallelizable) |

**Additional Components:**
- **Positional Encoding:** Injects sequence order (attention is position-agnostic)
- **Multi-Head Attention:** Multiple parallel attention operations
- **Layer Normalization + Residuals:** Training stability

**Critical Knowledge Flag:** Yes - Foundation of all modern LLMs

---

---
### Card 2 | Easy
**Cognitive Level:** Understand
**Concept:** Training Pipeline: Pre-training → SFT → RLHF
**Source Section:** Core Concepts 5, 6, 7
**Concept Map Centrality:** Pre-training (6), RLHF (5)

**FRONT (Question):**
Describe the three-stage training pipeline for modern LLMs: pre-training, supervised fine-tuning (SFT), and RLHF. What does each stage accomplish?

**BACK (Answer):**
**Three-Stage Training Pipeline:**

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │  Pre-train   │ ──► │     SFT      │ ──► │    RLHF      │    │
│  │              │     │              │     │              │    │
│  │ (Foundation) │     │ (Specialize) │     │   (Align)    │    │
│  └──────────────┘     └──────────────┘     └──────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

| Stage | Objective | Data | Output |
|-------|-----------|------|--------|
| **Pre-training** | Learn language patterns, facts, reasoning | Trillions of tokens (web, books, code) | Foundation model with broad capabilities |
| **SFT** | Learn to follow instructions | Curated (instruction, response) pairs | Helpful assistant behavior |
| **RLHF** | Align with human preferences | Human preference comparisons (A > B) | Safe, helpful, honest responses |

**RLHF Components:**
1. **Reward Model:** Trained on human preferences to predict which response is better
2. **Policy Optimization:** PPO adjusts LLM to maximize reward
3. **KL Constraint:** Prevents reward hacking by staying near SFT model

**Why This Pipeline?**
- Pre-training alone produces capable but unaligned "completion engines"
- SFT teaches instruction-following format
- RLHF optimizes subjective qualities (helpfulness, harmlessness, honesty)

**Critical Knowledge Flag:** Yes - Understanding training explains model behavior

---

---
### Card 3 | Medium
**Cognitive Level:** Apply
**Concept:** Prompting Strategies and In-Context Learning
**Source Section:** Core Concepts 8
**Concept Map Centrality:** Prompting (5), In-Context Learning (3)

**FRONT (Question):**
You need to build a system that extracts product information (name, price, category) from unstructured e-commerce descriptions. Compare zero-shot, few-shot, and chain-of-thought prompting approaches. Which would you recommend and why?

**BACK (Answer):**
**Prompting Strategy Comparison:**

| Strategy | Approach | Pros | Cons |
|----------|----------|------|------|
| **Zero-shot** | Instruction only: "Extract product name, price, category from this text" | Simple, minimal tokens | May miss edge cases, inconsistent format |
| **Few-shot** | Include 2-3 examples showing input→output | Better format consistency, handles ambiguity | Uses context tokens, need good examples |
| **Chain-of-Thought** | "First identify the product, then find price indicators, then categorize..." | Better reasoning, debuggable | Slower, more tokens, may overthink |

**Recommended Approach: Few-shot with structured output**

```
Extract product information in JSON format.

Example 1:
Input: "Amazing wireless headphones, now just $79.99! Perfect for music lovers."
Output: {"name": "Wireless Headphones", "price": 79.99, "category": "Electronics"}

Example 2:
Input: "Organic cotton t-shirt, men's large, blue - $24"
Output: {"name": "Organic Cotton T-Shirt", "price": 24.00, "category": "Clothing"}

Now extract from:
[user's product description]
```

**Why Few-shot:**
1. **Format consistency:** Examples demonstrate exact JSON structure expected
2. **Edge case handling:** Examples show how to handle price formats, abbreviations
3. **Reliability:** More consistent than zero-shot, simpler than CoT for extraction
4. **Efficiency:** 2-3 examples sufficient; CoT overkill for structured extraction

**When to use CoT instead:** Complex reasoning tasks, multi-step problems, when you need to audit the reasoning process.

**Critical Knowledge Flag:** Yes - Primary method for adapting LLMs to tasks

---

---
### Card 4 | Medium
**Cognitive Level:** Analyze
**Concept:** Context Windows and Inference Optimization
**Source Section:** Core Concepts 9, 10
**Concept Map Centrality:** Context Window (4), Generation (5), KV Cache (3)

**FRONT (Question):**
Analyze the tradeoffs in LLM inference for a customer service chatbot that needs to (a) maintain conversation history, (b) access a 50-page product manual, and (c) respond within 2 seconds. Address context limits, KV caching, and potential solutions.

**BACK (Answer):**
**Challenge Analysis:**

| Requirement | Constraint | Token Estimate |
|-------------|------------|----------------|
| Conversation history | Growing with each turn | 500-2000 tokens/conversation |
| Product manual | 50 pages | ~40,000 tokens |
| Response latency | 2 seconds | Limits processing time |

**Problem:** Total context (~42K tokens) may exceed limits or cause:
- **O(n²) attention cost:** Long contexts are computationally expensive
- **Lost in the middle:** Information buried in middle of context is poorly recalled
- **Latency issues:** Long contexts slow generation

**Solution Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│                       User Query                             │
│                           │                                  │
│                           ▼                                  │
│              ┌────────────────────────┐                     │
│              │  Retrieval (RAG)       │                     │
│              │  - Embed query         │                     │
│              │  - Search manual chunks│                     │
│              │  - Return top 3-5 chunks│                    │
│              └───────────┬────────────┘                     │
│                          ▼                                   │
│    ┌─────────────────────────────────────────┐              │
│    │ Context Assembly (~4K tokens)           │              │
│    │ - System prompt (500)                   │              │
│    │ - Recent conversation (1000)            │              │
│    │ - Retrieved manual chunks (2000)        │              │
│    │ - Current query (500)                   │              │
│    └─────────────────────────────────────────┘              │
│                          │                                   │
│                          ▼                                   │
│              ┌────────────────────────┐                     │
│              │  LLM with KV Cache     │◄── Cached from      │
│              │  (Optimized inference) │    previous turns   │
│              └───────────┬────────────┘                     │
│                          ▼                                   │
│                      Response                                │
└─────────────────────────────────────────────────────────────┘
```

**Optimization Techniques:**

| Technique | Benefit | Tradeoff |
|-----------|---------|----------|
| **RAG** | Only load relevant sections | Retrieval quality matters |
| **KV Caching** | Reuse attention computations across turns | Memory usage |
| **Quantization** | Faster inference (INT8/INT4) | Slight quality loss |
| **Context truncation** | Fit within limits | May lose important history |
| **Summarization** | Compress old conversation | Adds latency |

**Critical Knowledge Flag:** Yes - Essential for production deployment

---

---
### Card 5 | Hard
**Cognitive Level:** Evaluate/Synthesize
**Concept:** Complete LLM System Design with Safety Considerations
**Source Section:** All Core Concepts, Critical Analysis
**Concept Map Centrality:** Integrates all high-centrality nodes

**FRONT (Question):**
Synthesize a complete LLM-powered system for a healthcare company that wants to help patients understand their medical records. Address: (1) architecture selection and training approach, (2) prompting and interaction design, (3) hallucination mitigation, (4) inference optimization for cost, and (5) safety guardrails specific to medical context.

**BACK (Answer):**
**1. Architecture and Training:**

| Decision | Choice | Justification |
|----------|--------|---------------|
| Base model | Fine-tuned LLM (e.g., Llama, Claude API) | Pre-trained medical knowledge + controllability |
| Training approach | SFT on medical Q&A + domain adaptation | Specialized vocabulary, appropriate hedging |
| Parameter efficiency | LoRA fine-tuning | Cost-effective, preserves base capabilities |

**Avoid full RLHF** unless extensive medical expert feedback available—risk of optimizing for perceived helpfulness over accuracy.

**2. Prompting and Interaction:**

```
System Prompt:
You are a medical information assistant helping patients understand their
records. You MUST:
- Explain terms in plain language
- Never diagnose or recommend treatments
- Say "consult your doctor" for medical decisions
- Cite specific sections of the patient's record
- Acknowledge uncertainty explicitly

Format responses with:
- Summary of what the document says
- Plain-language explanation
- Questions to ask their healthcare provider
```

**3. Hallucination Mitigation:**

| Strategy | Implementation |
|----------|----------------|
| **Retrieval grounding** | RAG from patient's actual records; cite sources |
| **Constrained generation** | Only reference information present in provided documents |
| **Confidence calibration** | Require explicit uncertainty markers |
| **Verification layer** | Cross-check extracted facts against source |
| **Human-in-the-loop** | Flag low-confidence responses for review |

**4. Inference Optimization:**

```
Cost Structure:
┌─────────────────────────────────────────────┐
│ Input tokens: ~2K (record excerpt + query)  │
│ Output tokens: ~500 (explanation)           │
│ Requests/day: ~10,000                       │
└─────────────────────────────────────────────┘

Optimizations:
- Quantized model (INT8): 2x cost reduction
- Prompt caching: Reuse system prompt processing
- Batching: Group concurrent requests
- Tiered approach: Simple queries → smaller model
                   Complex queries → larger model
```

**5. Safety Guardrails (Medical-Specific):**

| Layer | Guardrail | Purpose |
|-------|-----------|---------|
| Input | PII detection | Ensure proper data handling |
| Input | Query classification | Block diagnosis-seeking queries |
| Output | Claim verification | Check generated claims against source |
| Output | Disclaimer injection | Ensure "consult your doctor" present |
| Output | Confidence thresholds | Reject low-confidence medical explanations |
| Monitoring | Human review queue | Flag edge cases for expert review |
| Audit | Full logging | Compliance, debugging, improvement |

**Red Lines (Never Allow):**
- Diagnosis or treatment recommendations
- Medication dosage advice
- Predictions about patient outcomes
- Contradicting physician instructions

**Critical Knowledge Flag:** Yes - Integrates architecture, training, prompting, deployment, and safety

---

---

## Export Formats

### Anki-Compatible (Tab-Separated)
```
What is Transformer architecture and how does self-attention differ from RNNs?	Transformer uses self-attention for parallel processing of all positions; RNNs process sequentially. Attention: Q,K,V mechanism with softmax(QK^T/√d)V. Enables direct long-range connections without vanishing gradients.	easy::architecture::llm
Describe the three-stage LLM training pipeline	1. Pre-training: Next-token prediction on trillions of tokens → foundation model. 2. SFT: Instruction-following on curated pairs → helpful assistant. 3. RLHF: Human preferences via reward model → aligned behavior.	easy::training::llm
Compare zero-shot, few-shot, and chain-of-thought prompting	Zero-shot: instruction only. Few-shot: include examples (best for consistent format). CoT: step-by-step reasoning (best for complex logic). For extraction tasks, few-shot recommended.	medium::prompting::llm
Analyze context window tradeoffs for chatbot with large manual	RAG for dynamic retrieval, KV caching for conversation, quantization for speed. Avoid full context—O(n²) cost, "lost in middle" problem. Assemble ~4K context from retrieved chunks.	medium::deployment::llm
Design LLM system for medical record explanation	RAG-grounded generation, SFT fine-tuning, confidence thresholds, output verification, mandatory disclaimers, human review queue. Never allow diagnosis/treatment recommendations.	hard::system-design::llm
```

### CSV Format
```csv
"Front","Back","Difficulty","Concept","Centrality"
"Transformer and self-attention?","Self-attention: Q,K,V mechanism for parallel sequence processing. Differs from RNN: parallel vs sequential, stable gradients, direct long-range attention.","Easy","Architecture","Critical"
"Three-stage training pipeline?","Pre-train (foundation) → SFT (instruction-following) → RLHF (alignment). Each stage adds capabilities.","Easy","Training","Critical"
"Zero-shot vs few-shot vs CoT?","Zero: instruction only. Few: with examples. CoT: step-by-step reasoning. Choose based on task complexity.","Medium","Prompting","High"
"Context window tradeoffs?","RAG + KV cache + quantization. Avoid full context—cost and recall issues.","Medium","Deployment","High"
"Medical LLM system design?","RAG grounding, verification layers, safety guardrails, never allow diagnosis.","Hard","System Design","Integration"
```

---

## Source Mapping

| Card | Source Sections | Concept Map Nodes | Key Terms |
|------|-----------------|-------------------|-----------|
| 1 | Concepts 1, 2, 3 | Transformer, Self-Attention, Positional Encoding | Q/K/V, multi-head, parallel processing |
| 2 | Concepts 5, 6, 7 | Pre-training, SFT, RLHF, Reward Model | Causal LM, instruction tuning, PPO |
| 3 | Concept 8 | Prompting, In-Context Learning, Chain-of-Thought | Zero-shot, few-shot, system prompt |
| 4 | Concepts 9, 10 | Context Window, Generation, KV Cache | O(n²), RAG, quantization |
| 5 | All concepts | Full integration | Hallucination, alignment, deployment |
