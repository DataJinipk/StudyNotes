# Assessment Quiz: Large Language Models

**Source Material:** notes/large-language-models/flashcards/large-language-models-flashcards.md
**Practice Problems:** notes/large-language-models/practice/large-language-models-practice-problems.md
**Concept Map:** notes/large-language-models/concept-maps/large-language-models-concept-map.md
**Original Study Notes:** notes/large-language-models/large-language-models-study-notes.md
**Date Generated:** 2026-01-06
**Total Questions:** 5
**Estimated Completion Time:** 30-40 minutes
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
**Concept Tested:** Self-Attention Mechanism
**Source Section:** Core Concepts 2
**Concept Map Node:** Self-Attention (7 connections)
**Related Flashcard:** Card 1
**Related Practice Problem:** P2

In the Transformer self-attention mechanism, what is the purpose of dividing by √d_k (the square root of the key dimension) before applying softmax?

A) To normalize the output vectors to unit length for consistent gradient flow

B) To prevent dot products from becoming too large, which would cause softmax to produce extremely peaked distributions with vanishing gradients

C) To reduce the computational cost of the attention operation by a factor of √d_k

D) To ensure the attention weights sum to exactly 1.0 across all positions

---

### Question 2 | Multiple Choice
**Cognitive Level:** Remember/Understand
**Concept Tested:** Training Pipeline
**Source Section:** Core Concepts 5, 6, 7
**Concept Map Node:** Pre-training (6), RLHF (5), SFT (4)
**Related Flashcard:** Card 2
**Related Practice Problem:** P4

Which statement correctly describes the relationship between pre-training, supervised fine-tuning (SFT), and RLHF in modern LLM development?

A) Pre-training creates task-specific capabilities, SFT adds general knowledge, and RLHF improves computational efficiency

B) Pre-training creates a foundation model with broad capabilities, SFT teaches instruction-following format, and RLHF aligns behavior with human preferences

C) Pre-training uses human feedback, SFT uses reward models, and RLHF uses large unlabeled corpora

D) All three stages can be performed in any order, as they optimize independent aspects of model behavior

---

### Question 3 | Short Answer
**Cognitive Level:** Apply/Analyze
**Concept Tested:** Context Windows and RAG
**Source Section:** Core Concepts 9
**Concept Map Node:** Context Window (4), Generation (5)
**Related Flashcard:** Card 4
**Related Practice Problem:** P1, P5
**Expected Response Length:** 3-4 sentences

A company wants to build a Q&A system over a 500-page technical manual using an LLM with a 4,096 token context window. The manual contains approximately 250,000 tokens. Explain why simply inserting the entire manual into the context is not feasible, and describe the recommended architectural pattern to address this limitation. Include at least one specific challenge that may arise with your proposed solution.

---

### Question 4 | Short Answer
**Cognitive Level:** Apply/Analyze
**Concept Tested:** Hallucination and Alignment
**Source Section:** Critical Analysis (Limitations)
**Concept Map Node:** Hallucination (3), Alignment (2)
**Related Flashcard:** Card 5
**Related Practice Problem:** P3, P4
**Expected Response Length:** 3-4 sentences

You're deploying an LLM to assist medical researchers in summarizing clinical trial results. During testing, you observe that the model occasionally generates plausible-sounding statistics that do not appear in the source documents. Identify this failure mode, explain why it occurs, and propose two specific mitigation strategies appropriate for a high-stakes medical context.

---

### Question 5 | Essay
**Cognitive Level:** Evaluate/Synthesize
**Concepts Tested:** Full LLM Pipeline, Deployment Considerations
**Source Sections:** All Core Concepts, Practical Applications
**Concept Map:** Full pathway traversal
**Related Flashcard:** Card 3, Card 4, Card 5
**Related Practice Problem:** P3, P4, P5
**Expected Response Length:** 1-2 paragraphs

You are the lead AI engineer tasked with building an intelligent code review assistant for a software company. The assistant must: (1) analyze pull request diffs to identify bugs, security issues, and style violations, (2) provide explanations for each issue found, (3) suggest specific code fixes, and (4) operate within a 2-second response time budget for reviews under 500 lines.

Design a complete solution addressing: (a) architecture selection—would you fine-tune a code-specific model or use a general LLM with prompting? Justify your choice; (b) prompting or fine-tuning strategy for the code review task; (c) how you would structure the output to make issues actionable; (d) latency optimization techniques to meet the 2-second target; and (e) how you would evaluate the quality of the code reviews beyond simple accuracy metrics.

**Evaluation Criteria:**
- [ ] Justifies architecture choice with specific reasoning
- [ ] Describes appropriate prompting/fine-tuning approach
- [ ] Specifies structured, actionable output format
- [ ] Addresses latency optimization concretely
- [ ] Proposes meaningful quality evaluation methodology

---

## Answer Key

### Question 1 | Multiple Choice
**Correct Answer:** B

**Explanation:**
The scaling factor √d_k is applied because dot products between queries and keys grow with the dimension d_k. For high-dimensional vectors, the dot products can become very large, which pushes the softmax function into regions where its gradients are extremely small (the function becomes nearly flat). This "softmax saturation" causes vanishing gradients during training, making learning difficult. Dividing by √d_k keeps the variance of dot products at approximately 1, ensuring the softmax operates in a region with meaningful gradients.

**Why Other Options Are Incorrect:**
- A) Scaling affects the softmax input magnitude, not output vector length; normalization is handled separately by layer norm
- C) The division is a scalar operation and does not reduce computational cost; it's purely for numerical stability
- D) Softmax already ensures weights sum to 1.0; the scaling addresses gradient flow, not normalization

**Understanding Gap Indicator:**
If answered incorrectly, review the mathematical properties of softmax and how input magnitude affects gradient flow. Practice Problem 2 provides hands-on computation of attention scores.

---

### Question 2 | Multiple Choice
**Correct Answer:** B

**Explanation:**
The three-stage pipeline follows a logical progression:
1. **Pre-training** on trillions of tokens creates a foundation model with broad language understanding, factual knowledge, and reasoning patterns—but no specific behavior alignment
2. **Supervised Fine-tuning (SFT)** on curated (instruction, response) pairs teaches the model the format of helpful assistant behavior—following instructions, maintaining conversation structure
3. **RLHF** uses human preference comparisons to train a reward model, then optimizes the policy to align with subjective qualities (helpfulness, harmlessness, honesty) that are difficult to specify directly

**Why Other Options Are Incorrect:**
- A) Inverts the relationship: pre-training provides general knowledge, not task-specific capabilities
- C) Reverses the training data: pre-training uses unlabeled corpora, RLHF uses human feedback
- D) The stages must be performed in sequence; each builds on the previous stage's outputs

**Understanding Gap Indicator:**
If answered incorrectly, review the training pipeline in Flashcard 2 and the detailed RLHF design in Practice Problem 4.

---

### Question 3 | Short Answer
**Model Answer:**

Inserting the 250,000-token manual into a 4,096-token context is not feasible because **the document is 60× larger than the context window limit**—self-attention's O(n²) complexity makes such long contexts computationally prohibitive, and the model cannot process tokens beyond its window. The recommended solution is **Retrieval-Augmented Generation (RAG)**: chunk the manual into 500-token segments, create vector embeddings for each chunk, and at query time retrieve the most relevant chunks using semantic similarity search. These retrieved chunks (e.g., top 5-10) are then inserted into the context alongside the user's question.

A key challenge with RAG is **retrieval quality**—if the relevant chunks are not retrieved (due to vocabulary mismatch, ambiguous queries, or information spanning multiple chunks), the model cannot provide accurate answers. Mitigation includes hybrid search (combining semantic and keyword retrieval) and chunk overlap to avoid splitting critical information.

**Key Components Required:**
- [ ] Identifies context window limit as fundamental constraint
- [ ] Describes RAG architecture (chunking, embedding, retrieval)
- [ ] Names at least one specific challenge (retrieval quality, lost context, chunk boundaries)

**Partial Credit Guidance:**
- Full credit: Clear constraint explanation + RAG description + specific challenge with mitigation
- Partial credit: Mentions context limit and RAG but vague on challenges
- No credit: Suggests the context window can be exceeded, or misunderstands RAG

**Understanding Gap Indicator:**
If answered poorly, review Context Window concepts in Flashcard 4 and Practice Problem 1.

---

### Question 4 | Short Answer
**Model Answer:**

This failure mode is **hallucination**—the model generates plausible but fabricated information not present in the source material. Hallucinations occur because LLMs are trained to produce fluent, likely text based on patterns learned during pre-training; they cannot reliably distinguish between recalled knowledge and generated fabrications. In the medical domain, this is particularly dangerous as fabricated statistics could lead to incorrect clinical decisions.

Two mitigation strategies for high-stakes medical use:
1. **Retrieval grounding with citation enforcement:** Require the model to cite specific passages from the source document for every statistic or claim, then programmatically verify that cited passages exist and contain the claimed information. Reject any response with unverifiable claims.
2. **Constrained generation with human review:** Use the LLM only to extract and rephrase information explicitly present in the source, with instructions to state "not found in document" rather than infer. Flag all outputs for expert review before clinical use, treating the AI as a draft assistant rather than authoritative source.

**Key Components Required:**
- [ ] Correctly identifies hallucination as the failure mode
- [ ] Explains why LLMs hallucinate (pattern completion, lack of grounding)
- [ ] Proposes two specific, appropriate mitigations for medical context

**Partial Credit Guidance:**
- Full credit: Correct identification + mechanism explanation + two concrete mitigations
- Partial credit: Correct identification but generic mitigations
- No credit: Misidentifies the failure mode or proposes inappropriate solutions

**Understanding Gap Indicator:**
If answered poorly, review hallucination discussions in Critical Analysis and the medical system design in Flashcard 5.

---

### Question 5 | Essay
**Model Answer:**

**Architecture Selection:**
I would **use a code-specialized LLM** (such as CodeLlama, StarCoder, or Codestral) **with few-shot prompting** rather than fine-tuning. Code-specialized models are pre-trained on code repositories and already understand syntax, common bug patterns, and security vulnerabilities. Fine-tuning would require a large labeled dataset of code reviews (expensive to create and maintain), risks overfitting to specific coding styles, and introduces deployment complexity. Few-shot prompting with a well-designed system prompt provides flexibility to update review criteria without retraining, works well for this classification + generation task, and allows rapid iteration. For larger teams with substantial labeled data and specific review standards, domain-adapted LoRA fine-tuning could be considered as a second phase.

**Prompting Strategy:**
The system prompt would define the role ("expert code reviewer"), specify review categories (bugs, security, style, performance), and require structured output. Few-shot examples would demonstrate identifying real bugs with explanations and fixes:

```
System: You are an expert code reviewer. Analyze the provided diff and identify issues.
For each issue: (1) specify type [Bug|Security|Style|Performance], (2) identify the location (file:line),
(3) explain why it's a problem, (4) suggest a fix. If no issues found, state "No issues identified."

Example:
[Example diff showing unchecked null pointer]
Review: {"issues": [{"type": "Bug", "location": "auth.py:45", "explanation": "user object may be None if login fails, causing AttributeError on next line", "fix": "Add 'if user is None: return error' before accessing user.id"}]}
```

**Output Structure:**
Responses would be JSON-formatted for parseability:
```json
{
  "summary": "Found 2 issues: 1 security, 1 style",
  "issues": [
    {
      "type": "Security",
      "severity": "High",
      "location": "api/handlers.py:127",
      "code_snippet": "query = f\"SELECT * FROM users WHERE id = {user_id}\"",
      "explanation": "SQL injection vulnerability: user_id is inserted directly into query string without sanitization",
      "suggested_fix": "query = \"SELECT * FROM users WHERE id = ?\"; cursor.execute(query, (user_id,))"
    }
  ],
  "approval_recommendation": "Request changes"
}
```

**Latency Optimization:**
To meet the 2-second budget for 500-line diffs:
1. **Streaming responses:** Begin displaying issues as they're generated rather than waiting for completion
2. **Context optimization:** Only include relevant file sections (±10 lines around changes), not entire files
3. **Quantization:** Use INT8 quantized model (50% memory, faster inference, minimal quality loss on code tasks)
4. **Caching:** Cache embeddings of common code patterns; use prefix caching for shared system prompts
5. **Parallel chunking:** For very large PRs, split into independent files and review in parallel, then aggregate

Target token budget: ~1000 input (diff + prompt) + ~500 output, achievable in <2s with optimized 7B model.

**Quality Evaluation:**
Beyond accuracy, I would measure:
1. **Precision/Recall on labeled test set:** What percentage of identified issues are real? What percentage of real issues are found?
2. **Developer acceptance rate:** What percentage of AI-suggested issues do reviewers mark as valid?
3. **Fix quality:** Do developers use the suggested fixes, modify them, or write entirely different solutions?
4. **A/B testing:** Compare PR merge rates, bug escape rates, and review time between AI-assisted and manual reviews
5. **Qualitative feedback:** Regular surveys on explanation clarity, false positive frustration, and trust calibration

**Evaluation Rubric:**

| Criterion | Excellent (4) | Proficient (3) | Developing (2) | Beginning (1) |
|-----------|---------------|----------------|----------------|---------------|
| Architecture | Justifies code-specialized model + prompting with clear reasoning; acknowledges tradeoffs | Chooses appropriate architecture with some justification | Names architecture without justification | Inappropriate choice or no reasoning |
| Prompting/Fine-tuning | Detailed prompt design with examples; clear review criteria | Describes approach with some specifics | Vague strategy | No prompting strategy |
| Output Structure | Actionable JSON with all required fields; developer-friendly | Structured output with most elements | Partially structured | Unstructured text output |
| Latency | Multiple concrete techniques (quantization, streaming, caching) | 2-3 specific optimization techniques | Generic "make it faster" | No latency consideration |
| Evaluation | Multi-faceted (precision/recall, human acceptance, A/B, qualitative) | 2-3 meaningful metrics | Single metric (accuracy) | No evaluation plan |

**Understanding Gap Indicator:**
If response lacks depth, this may indicate:
- Limited experience with code-specialized LLMs and their capabilities
- Weak understanding of latency/throughput tradeoffs in production
- Insufficient appreciation for evaluation beyond simple accuracy

---

## Diagnostic Feedback & Review Recommendations

### Performance-Based Review Guide

| If Incorrect On... | Gap Identified | Review Section | Priority |
|--------------------|----------------|----------------|----------|
| Question 1 | Self-attention mechanism details | Core Concepts 2 + Practice P2 | High |
| Question 2 | Training pipeline understanding | Core Concepts 5-7 + Flashcard 2 | High |
| Question 3 | Context limitations and RAG | Core Concepts 9 + Practice P1 | Medium |
| Question 4 | Hallucination and safety | Critical Analysis + Flashcard 5 | Medium |
| Question 5 | Full pipeline synthesis | All sections | Low |

### Targeted Review Recommendations

#### For Multiple Choice Errors (Questions 1-2)
**Indicates:** Foundational concept gaps
**Action:** Review:
- Study Notes: Core Concepts 2 (Self-Attention) and 5-7 (Training Pipeline)
- Flashcards: Cards 1 and 2
- Practice Problems: P2 (attention computation) and P4 (RLHF design)
**Focus On:** Understanding WHY mechanisms work, not just WHAT they are

#### For Short Answer Errors (Questions 3-4)
**Indicates:** Application difficulties
**Action:** Practice:
- Practice Problems: P1 (tokenization/context) and P3 (prompt engineering)
- Concept Map: Context Window and Hallucination pathways
**Focus On:** Connecting limitations to system design decisions

#### For Essay Weakness (Question 5)
**Indicates:** Integration challenges
**Action:** Review interconnections:
- Concept Map: Full Critical Path traversal
- All Practice Problems for procedural fluency
- Study Notes: Practical Applications section
**Focus On:** Building mental models that connect architecture → training → deployment → evaluation

### Mastery Indicators

- **5/5 Correct:** Strong mastery; ready for production LLM projects
- **4/5 Correct:** Good understanding; review indicated gap
- **3/5 Correct:** Moderate understanding; systematic review + more practice problems
- **2/5 or below:** Foundational gaps; restart from Concept Map Critical Path

---

## Skill Chain Traceability

```
Study Notes (Source) ──────────────────────────────────────────────────────┐
    │                                                                      │
    │  10 Core Concepts, 12 Key Terms, 4 Applications                      │
    │                                                                      │
    ├────────────┬────────────┬────────────┬────────────┐                  │
    │            │            │            │            │                  │
    ▼            ▼            ▼            ▼            ▼                  │
Concept Map  Flashcards   Practice    Quiz                                 │
    │            │        Problems      │                                  │
    │ 24 concepts│ 5 cards   │ 5 problems│ 5 questions                     │
    │ 38 rels    │ 2E/2M/1H  │ 1W/2S/1C/1D│ 2MC/2SA/1E                     │
    │ 4 pathways │           │           │                                 │
    │            │           │           │                                 │
    └─────┬──────┴─────┬─────┴─────┬─────┘                                 │
          │            │           │                                       │
          │ Centrality │ Practice  │                                       │
          │ → Card     │ → Quiz    │                                       │
          │ difficulty │ distractors│                                      │
          │            │           │                                       │
          └────────────┴───────────┴───────────────────────────────────────┘
                                   │
                          Quiz integrates ALL
                          upstream materials
```

---

## Complete 5-Skill Chain Summary

| Skill | Output | Key Contribution to Chain |
|-------|--------|---------------------------|
| study-notes-creator | 10 concepts, theory, applications | Foundation content |
| concept-map | 24 nodes, 38 relationships, 4 pathways | Structure + learning paths |
| flashcards | 5 cards (2E/2M/1H) | Memorization + critical flags |
| practice-problems | 5 problems (1W/2S/1C/1D) | Procedural fluency + common mistakes |
| quiz | 5 questions (2MC/2SA/1E) | Assessment + diagnostic feedback |
