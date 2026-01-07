# Assessment Quiz: Prompt Engineering

**Source Material:** notes/prompt-engineering/flashcards/prompt-engineering-flashcards.md
**Original Source Path:** notes/prompt-engineering/prompt-engineering-study-notes.md
**Date Generated:** 2026-01-06
**Total Questions:** 5
**Estimated Completion Time:** 20-30 minutes
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
**Concept Tested:** Chain-of-Thought Prompting
**Source Section:** Core Concepts - Concept 3
**Source Flashcard:** Card 2

Chain-of-Thought (CoT) prompting improves model performance on complex reasoning tasks primarily because:

A) It increases the model's parameter count during inference, enabling more sophisticated computation

B) It forces the model to generate intermediate reasoning steps as tokens, enabling decomposition and self-correction

C) It automatically fine-tunes the model on the specific reasoning task before generating output

D) It bypasses the model's attention mechanism to access deeper knowledge representations

---

### Question 2 | Multiple Choice
**Cognitive Level:** Remember/Understand
**Concept Tested:** Prompt Anatomy and Structure
**Source Section:** Core Concepts - Concept 1
**Source Flashcard:** Card 1

When structuring an effective prompt, which ordering of components is recommended and why?

A) Instruction → Context → Examples → Constraints, because the model needs to know the task before understanding the domain

B) Examples → Context → Instruction → Constraints, because in-context learning requires examples first

C) Context → Instruction → Constraints → Output Format, because context primes attention mechanisms before task specification

D) Output Format → Constraints → Instruction → Context, because the model processes tokens in reverse order

---

### Question 3 | Short Answer
**Cognitive Level:** Apply/Analyze
**Concept Tested:** Few-Shot Prompting Design
**Source Section:** Core Concepts - Concept 2
**Source Flashcard:** Card 3
**Expected Response Length:** 2-4 sentences

You are building a prompt to classify customer support tickets into categories (Billing, Technical, Account, Other). You have limited context window space. Should you use zero-shot, few-shot (2 examples per category), or many-shot (10 examples per category) prompting? Justify your choice by analyzing the trade-offs for this specific task.

---

### Question 4 | Short Answer
**Cognitive Level:** Apply/Analyze
**Concept Tested:** Prompt Chaining vs. Monolithic Approaches
**Source Section:** Core Concepts - Concept 5
**Source Flashcard:** Card 4
**Expected Response Length:** 2-4 sentences

A development team is building a feature that takes a user's natural language description and generates a database query. They're debating between a single sophisticated prompt versus a three-step chain (parse intent → identify entities → construct query). Given that incorrect queries could corrupt production data, which approach should they choose and why?

---

### Question 5 | Essay
**Cognitive Level:** Evaluate/Synthesize
**Concepts Tested:** Prompt Structure, Few-Shot, CoT, System Prompts, Chaining, Brittleness
**Source Sections:** All Core Concepts, Critical Analysis
**Source Flashcard:** Card 5
**Expected Response Length:** 1-2 paragraphs

You are the technical lead for an AI-powered legal document analysis platform. The system must extract key clauses from contracts, identify potential risks, and generate summary reports—all while maintaining high reliability since outputs inform legal decisions.

Design a comprehensive prompt engineering architecture for this system. Your response should address: (1) how you would structure the system prompt to establish appropriate behavioral guardrails, (2) which components would use few-shot examples versus zero-shot approaches, (3) where chain-of-thought reasoning is essential, (4) how you would decompose the workflow using prompt chaining, and (5) specific strategies to address the brittleness limitation given the high-stakes nature of the application.

**Evaluation Criteria:**
- [ ] Addresses all five architectural components requested
- [ ] Demonstrates integration of multiple prompting techniques
- [ ] Provides specific, actionable design decisions (not generic advice)
- [ ] Shows critical evaluation of trade-offs given the legal domain requirements

---

## Answer Key

### Question 1 | Multiple Choice
**Correct Answer:** B

**Explanation:**
Chain-of-Thought prompting works by instructing the model to generate intermediate reasoning steps as output tokens before arriving at a final answer. Because these steps are generated as tokens, the model can decompose complex problems into simpler sub-problems, and the generated reasoning creates context that enables self-correction. This token-level generation of reasoning is the mechanism that improves performance.

**Why Other Options Are Incorrect:**
- A) CoT does not affect parameter count; the model's weights remain unchanged during inference
- C) CoT is an inference-time technique, not fine-tuning; no weight updates occur
- D) CoT works with the attention mechanism, not around it; the generated tokens become part of the attention context

**Understanding Gap Indicator:**
If answered incorrectly, this may indicate confusion between inference-time prompting techniques and training-time modifications. Review the distinction between prompting (input manipulation) and fine-tuning (weight modification).

---

### Question 2 | Multiple Choice
**Correct Answer:** C

**Explanation:**
The recommended ordering places context before instruction because context "primes" the model's attention mechanisms, activating relevant knowledge domains before the task is specified. This ordering mirrors how humans process instructions—understanding the domain before understanding the specific ask. Constraints and output format follow instruction to bound and shape the response.

**Why Other Options Are Incorrect:**
- A) While instruction is important, providing it before context means the model lacks domain framing when interpreting the task
- B) Examples can come after instruction; placing them first without context reduces their interpretability
- D) Models process tokens in forward order, not reverse; this ordering makes no practical sense

**Understanding Gap Indicator:**
If answered incorrectly, this may indicate incomplete understanding of how attention mechanisms process sequential input. Review the theoretical framework on "Attention as Resource Allocation."

---

### Question 3 | Short Answer
**Model Answer:**
Few-shot prompting (2 examples per category = 8 total examples) is the optimal choice for this classification task. Zero-shot would likely work for common categories but may struggle with edge cases or company-specific category definitions. Many-shot (40 examples) would consume excessive context window space while providing diminishing returns—research shows example quality and diversity matter more than quantity. With 2 diverse examples per category, the model can learn the classification boundaries while preserving context space for the actual tickets to classify. The key is selecting examples that represent typical cases and boundary cases for each category.

**Key Components Required:**
- [ ] Selects few-shot with clear reasoning
- [ ] Addresses context window trade-off
- [ ] References quality over quantity principle
- [ ] Considers task characteristics (well-defined categories)

**Partial Credit Guidance:**
- Full credit: Selects few-shot with reasoning addressing both context limits and task requirements
- Partial credit: Correct choice but reasoning incomplete or generic
- No credit: Selects many-shot without acknowledging context limits, or zero-shot without acknowledging category ambiguity risks

**Understanding Gap Indicator:**
If answered incorrectly, this may indicate difficulty applying shot-selection principles to specific task characteristics. Review the conditions favoring each approach in Core Concepts - Concept 2.

---

### Question 4 | Short Answer
**Model Answer:**
The three-step chain approach is strongly recommended given the high stakes of incorrect queries potentially corrupting production data. Chaining provides critical advantages here: (1) each step can be validated before proceeding—intent parsing can be verified, entities can be confirmed against schema, and the final query can be reviewed before execution; (2) failures can be isolated to specific chain links, enabling targeted debugging; (3) the intermediate outputs create an audit trail for accountability. While chaining introduces latency (three API calls vs. one), this trade-off is acceptable when incorrect outputs have severe consequences. The monolithic approach's "black box" nature is inappropriate for high-risk database operations.

**Key Components Required:**
- [ ] Selects chaining approach
- [ ] References validation/verification capability at each step
- [ ] Addresses error isolation and debugging benefits
- [ ] Acknowledges latency trade-off but justifies given risk profile

**Partial Credit Guidance:**
- Full credit: Clear selection with multiple justifications tied to risk profile
- Partial credit: Correct selection with limited justification
- No credit: Selects monolithic approach, or selects chaining but fails to connect to data integrity concerns

**Understanding Gap Indicator:**
If answered incorrectly, this may indicate difficulty evaluating trade-offs based on application requirements. Review the analysis of when to use chaining in Core Concepts - Concept 5.

---

### Question 5 | Essay
**Model Answer:**

For a legal document analysis platform, I would design a layered architecture combining all major prompting techniques with rigorous validation:

**System Prompt Architecture:** The system prompt establishes the model as a "legal document analyst assistant" with explicit constraints: never provide legal advice (only analysis), always cite specific clause locations, flag uncertainty explicitly, and refuse to process documents outside contract types it's trained for. Negative constraints are critical here—the model must know what NOT to do given legal liability concerns.

**Shot Strategy by Component:** Clause extraction uses few-shot prompting (3-4 examples per clause type) because contract language follows patterns but has firm-specific variations—examples anchor expected extractions. Risk identification uses zero-shot with detailed criteria definitions because risks are domain-specific and examples might anchor the model too narrowly, missing novel risk patterns. Summary generation uses zero-shot with strong format constraints since summarization is a general capability.

**Chain-of-Thought Integration:** CoT is essential for risk identification where the model must reason about clause implications. The prompt explicitly requires: "For each identified clause, reason through potential risks by considering: party obligations, ambiguous language, missing standard protections, and unusual terms. Show your reasoning before concluding." This creates auditable reasoning traces for legal review.

**Prompt Chain Architecture:** The workflow decomposes into: (1) Document parsing and clause segmentation → (2) Per-clause extraction with structured output → (3) Risk analysis with CoT reasoning → (4) Cross-clause dependency analysis → (5) Summary generation. Each step validates output schema before proceeding. Failed validation triggers human review rather than chain continuation.

**Brittleness Mitigation:** Given legal stakes, I implement: (a) confidence scoring on all extractions with low-confidence items flagged for human review; (b) dual-prompt validation where critical extractions run through two differently-worded prompts and discrepancies trigger review; (c) version-controlled prompt registry with regression tests against labeled contract corpus; (d) production monitoring tracking extraction accuracy metrics with automated alerts on drift; (e) graceful degradation where validation failures return "requires human review" rather than potentially incorrect output.

**Evaluation Rubric:**

| Criterion | Excellent (4) | Proficient (3) | Developing (2) | Beginning (1) |
|-----------|---------------|----------------|----------------|---------------|
| Concept Integration | All 5 components addressed with specific, integrated design decisions | 4-5 components addressed, mostly specific | 3 components addressed, some generic | <3 components, generic advice |
| Critical Analysis | Trade-offs explicitly evaluated for legal domain; justified departures from defaults | Trade-offs acknowledged, mostly justified | Some trade-offs noted | No trade-off analysis |
| Actionability | Specific, implementable decisions (e.g., "3-4 examples per clause type") | Mostly specific with some vague areas | Mix of specific and generic | Generic advice only |
| Domain Awareness | Design clearly shaped by legal requirements (liability, auditability, uncertainty) | Domain considered in most decisions | Domain mentioned but not integrated | Domain not reflected in design |

**Understanding Gap Indicator:**
If response lacks depth, this may indicate:
- Difficulty synthesizing multiple techniques into coherent architecture
- Weak connection between technique selection and domain requirements
- Surface understanding of brittleness without concrete mitigation strategies

---

## Diagnostic Feedback & Review Recommendations

### Performance-Based Review Guide

| If Incorrect On... | Gap Identified | Review Section | Priority |
|--------------------|----------------|----------------|----------|
| Question 1 | CoT mechanism misunderstanding | Core Concepts: Concept 3 | High |
| Question 2 | Prompt structure principles unclear | Core Concepts: Concept 1 | High |
| Question 3 | Shot-selection application weak | Core Concepts: Concept 2 | Medium |
| Question 4 | Chaining trade-off analysis | Core Concepts: Concept 5; Critical Analysis | Medium |
| Question 5 | Integration/synthesis difficulty | All sections + Theoretical Framework | Low |

### Targeted Review Recommendations

#### For Multiple Choice Errors (Questions 1-2)
**Indicates:** Foundational concept gaps
**Action:** Review definitions and mechanisms in:
- Study Notes: Core Concepts sections 1 and 3
- Key Terminology table (CoT, Prompt Anatomy)
**Focus On:** Understanding *why* techniques work, not just *what* they are

#### For Short Answer Errors (Questions 3-4)
**Indicates:** Application or analysis difficulties
**Action:** Practice applying concepts through:
- Study Notes: Practical Applications section
- Review Questions 2 and 3 from study notes
**Focus On:** Matching technique selection to specific task requirements

#### For Essay Weakness (Question 5)
**Indicates:** Integration or synthesis challenges
**Action:** Review interconnections between:
- All Core Concepts—trace how each relates to others
- Critical Analysis: Strengths, Limitations, and brittleness discussion
- Theoretical Framework: Understanding underlying principles enables flexible application
**Focus On:** Building mental models that connect techniques rather than memorizing isolated facts

### Mastery Indicators

- **5/5 Correct:** Strong mastery demonstrated; proceed to hands-on prompt engineering practice
- **4/5 Correct:** Good understanding; review indicated gap area before advanced applications
- **3/5 Correct:** Moderate understanding; systematic review of Core Concepts recommended
- **2/5 or below:** Foundational gaps; comprehensive re-study of source material advised

---

## Skill Chain Traceability

```
Study Notes (Source)
    │
    │  Extracted: 5 Core Concepts + Critical Analysis + Theoretical Framework
    ▼
Flashcards (Intermediate)
    │
    │  Cards: 5 (2 Easy, 2 Medium, 1 Hard)
    │  Critical Knowledge Flagged: CoT, Few-Shot, Prompt Structure
    ▼
Quiz (Output)
    │
    │  Questions: 5 (2 MC, 2 SA, 1 Essay)
    │  Mapped: Flashcard → Study Notes sections
    │  Added: Detailed explanations, gap indicators, review recommendations
    ▼
Complete Learning Module
```
