# Flashcard Set: Prompt Engineering

**Source:** notes/prompt-engineering/prompt-engineering-study-notes.md
**Original Source Path:** C:\agentic_ai\StudyNotes\notes\prompt-engineering\prompt-engineering-study-notes.md
**Date Generated:** 2026-01-06
**Total Cards:** 5
**Distribution:** 2 Easy | 2 Medium | 1 Hard

---

## Critical Knowledge Summary

Concepts appearing in multiple cards (prioritize for review):
- **Chain-of-Thought (CoT)**: Appears in Cards 2, 4, 5 (core reasoning technique)
- **Few-Shot Prompting**: Appears in Cards 1, 3, 5 (foundational technique)
- **Prompt Structure**: Appears in Cards 1, 3, 5 (architectural principle)

---

## Flashcards

---
### Card 1 | Easy
**Cognitive Level:** Remember
**Concept:** Prompt Anatomy and Structure
**Source Section:** Core Concepts - Concept 1

**FRONT (Question):**
What are the five key structural components of an effective prompt?

**BACK (Answer):**
1. **Context** - Background information that frames the task domain
2. **Instruction** - The precise action or task specification
3. **Constraints** - Boundaries that limit undesired responses
4. **Output Format** - Specifications for how results should be structured
5. **Examples** - Demonstrations of expected input-output mappings (when included)

**Critical Knowledge Flag:** Yes - Foundational structure referenced throughout prompting techniques

---

---
### Card 2 | Easy
**Cognitive Level:** Understand
**Concept:** Chain-of-Thought Prompting
**Source Section:** Core Concepts - Concept 3

**FRONT (Question):**
What is Chain-of-Thought (CoT) prompting, and why does it improve model performance on complex tasks?

**BACK (Answer):**
Chain-of-Thought prompting instructs models to articulate intermediate reasoning steps before arriving at a final answer. It improves performance because:
- Decomposing problems into steps reduces complexity at each stage
- Generated intermediate tokens enable self-correction
- The model can catch errors in reasoning before producing final output
- Even zero-shot CoT ("Let's think step by step") provides significant accuracy gains

**Critical Knowledge Flag:** Yes - Core technique for reasoning tasks

---

---
### Card 3 | Medium
**Cognitive Level:** Apply
**Concept:** Few-Shot Prompting Design
**Source Section:** Core Concepts - Concept 2; Practical Applications

**FRONT (Question):**
You need to extract sentiment and key features from product reviews. Design a few-shot prompt structure that would maximize extraction accuracy. What elements must you include?

**BACK (Answer):**
**Required Elements:**
1. **Context:** "You are a product review analyzer extracting structured data."
2. **Output Format Specification:** Define exact schema (JSON/table) for extracted data
3. **2-3 Diverse Examples:** Include positive, negative, and mixed sentiment reviews demonstrating the extraction pattern
4. **Edge Case Handling:** Show how to handle missing information or ambiguous sentiment
5. **Explicit Field Definitions:** Clarify what counts as a "feature" vs. opinion

**Key Principle:** Example quality and diversity matter more than quantity. Each example should demonstrate a distinct pattern the model may encounter.

**Critical Knowledge Flag:** Yes - Applies both Few-Shot and Prompt Structure concepts

---

---
### Card 4 | Medium
**Cognitive Level:** Analyze
**Concept:** Prompt Chaining vs. Monolithic Prompts
**Source Section:** Core Concepts - Concept 5; Critical Analysis

**FRONT (Question):**
Compare monolithic (single-prompt) approaches versus prompt chaining for complex tasks. When should you use each approach, and what are the key trade-offs?

**BACK (Answer):**
**Use Monolithic Prompts When:**
- Task is well-defined and single-focused
- Context requirements are minimal
- Latency is critical (single API call)
- Task doesn't benefit from intermediate verification

**Use Prompt Chaining When:**
- Task requires multiple distinct operations (extract → transform → generate)
- Intermediate outputs need validation before proceeding
- Debugging and error isolation are priorities
- Task complexity exceeds reliable single-prompt execution

**Trade-offs:**
| Aspect | Monolithic | Chaining |
|--------|-----------|----------|
| Latency | Lower | Higher (multiple calls) |
| Reliability | Variable | Higher (isolated steps) |
| Debuggability | Difficult | Easy (pinpoint failures) |
| Flexibility | Limited | High (swap chain components) |

**Critical Knowledge Flag:** Yes - Connects Prompt Chaining with CoT decomposition principles

---

---
### Card 5 | Hard
**Cognitive Level:** Synthesize
**Concept:** Integrated Prompt Engineering Strategy
**Source Section:** Core Concepts (all); Theoretical Framework; Critical Analysis

**FRONT (Question):**
Synthesize an integrated prompt engineering strategy for building a reliable production application. How do prompt structure, few-shot learning, chain-of-thought, system prompts, and chaining work together? Address the brittleness limitation in your design.

**BACK (Answer):**
**Integrated Strategy:**

1. **System Prompt (Foundation):** Establish persistent behavioral framework—role, constraints, output conventions. This provides stable context that reduces variability across interactions.

2. **Prompt Structure (Per-Request):** Each user request processed through structured prompts with explicit context, instruction, constraints, and format. Standardized structure improves consistency.

3. **Few-Shot Examples (Pattern Establishment):** Embed canonical examples in system prompt or inject dynamically based on task type. Examples anchor expected behavior, reducing brittleness.

4. **Chain-of-Thought (Reasoning Quality):** For complex reasoning, explicit CoT instruction ensures transparent, verifiable reasoning paths. Intermediate steps enable quality checks.

5. **Prompt Chaining (Complex Workflows):** Decompose multi-step tasks into focused chain links. Each link validated before proceeding, enabling graceful failure handling.

**Addressing Brittleness:**
- **Version Control:** Treat prompts as code; track changes, run regression tests
- **Validation Gates:** Verify outputs at chain boundaries against expected schemas
- **Fallback Strategies:** Define degradation paths when validation fails
- **Ensemble Approaches:** Use multiple prompt variants, aggregate/vote on outputs
- **Continuous Monitoring:** Track output quality metrics in production; alert on drift

**The synthesis recognizes that individual techniques reinforce each other—system prompts reduce per-request variability, few-shot reduces interpretation ambiguity, CoT improves reasoning reliability, and chaining isolates failures.**

**Critical Knowledge Flag:** Yes - Integrates all major concepts: Structure, Few-Shot, CoT, System Prompts, Chaining

---

---

## Export Formats

### Anki-Compatible (Tab-Separated)
```
What are the five key structural components of an effective prompt?	1. Context 2. Instruction 3. Constraints 4. Output Format 5. Examples	easy::structure::prompt-engineering
What is Chain-of-Thought (CoT) prompting and why does it improve performance?	CoT instructs models to show reasoning steps, enabling decomposition, self-correction, and error catching before final output.	easy::cot::prompt-engineering
Design a few-shot prompt for extracting sentiment from reviews. What elements are required?	Context, output format spec, 2-3 diverse examples, edge case handling, explicit field definitions. Quality > quantity for examples.	medium::few-shot::prompt-engineering
Compare monolithic vs. prompt chaining approaches. When use each?	Monolithic: simple tasks, low latency needs. Chaining: complex multi-step tasks, need validation, debugging priority. Trade-off: latency vs reliability.	medium::chaining::prompt-engineering
Synthesize an integrated prompt engineering strategy addressing brittleness.	Layer: system prompt (foundation) + structured per-request prompts + few-shot examples + CoT for reasoning + chaining for workflows. Address brittleness via version control, validation gates, fallbacks, ensembles, monitoring.	hard::synthesis::prompt-engineering
```

### CSV Format
```csv
"Front","Back","Difficulty","Concept"
"What are the five key structural components of an effective prompt?","Context, Instruction, Constraints, Output Format, Examples","Easy","Prompt Structure"
"What is CoT prompting and why does it improve performance?","Instructs models to show reasoning steps; enables decomposition and self-correction","Easy","Chain-of-Thought"
"Design a few-shot prompt for sentiment extraction - what's required?","Context, format spec, diverse examples, edge cases, field definitions","Medium","Few-Shot Design"
"When use monolithic vs chaining approaches?","Monolithic for simple/low-latency; Chaining for complex/validation needs","Medium","Prompt Chaining"
"Synthesize integrated prompt strategy addressing brittleness","Layer system prompts, structure, few-shot, CoT, chaining; add version control, validation, fallbacks","Hard","Integration"
```

---

## Source Mapping

| Card | Source Section | Key Terminology Used |
|------|----------------|---------------------|
| 1 | Core Concepts: Concept 1 | Prompt anatomy, context, constraints, output format |
| 2 | Core Concepts: Concept 3 | Chain-of-thought, reasoning steps, zero-shot CoT |
| 3 | Core Concepts: Concept 2; Applications | Few-shot, in-context learning, examples |
| 4 | Core Concepts: Concept 5; Critical Analysis | Prompt chaining, decomposition, debugging |
| 5 | All Core Concepts + Framework + Analysis | System prompt, brittleness, integration |
