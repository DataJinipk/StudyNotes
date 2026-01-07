# Assessment Quiz: Creating Agent Skills

**Source Material:** notes/agent-skills/flashcards/creating-agent-skills-flashcards.md
**Concept Map Reference:** notes/agent-skills/concept-maps/creating-agent-skills-concept-map.md
**Original Study Notes:** notes/agent-skills/creating-agent-skills-study-notes.md
**Source Lesson:** Lessions/Lesson_1.md
**Date Generated:** 2026-01-06
**Total Questions:** 5
**Estimated Completion Time:** 25-35 minutes
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
**Concept Tested:** Skill Definition Components
**Source Section:** Core Concepts - Concepts 1 & 2
**Concept Map Node:** Skill Definition (Critical - 8 connections)
**Related Flashcard:** Card 1

Which of the following is NOT a core structural component of a complete agent skill definition?

A) Trigger specifications defining when the skill activates

B) Input/output contracts specifying data formats and requirements

C) Training data for fine-tuning the underlying language model

D) Chain metadata declaring upstream and downstream skill relationships

---

### Question 2 | Multiple Choice
**Cognitive Level:** Remember/Understand
**Concept Tested:** Input/Output Contracts and Chaining
**Source Section:** Core Concepts - Concepts 4 & 6
**Concept Map Node:** Output Contract (High - 5 connections)
**Related Flashcard:** Card 2

Two skills need to be chained: Skill A produces structured JSON with fields {summary, key_points, references}. Skill B requires input with fields {summary, key_points, sentiment_score}. What will happen when these skills are chained?

A) The chain will work perfectly since both skills share some common fields

B) The chain will fail because Skill A's output contract doesn't include the required sentiment_score field that Skill B needs

C) The chain will work because the language model can infer the missing sentiment_score field

D) The chain will work but Skill B will ignore the references field it doesn't need

---

### Question 3 | Short Answer
**Cognitive Level:** Apply/Analyze
**Concept Tested:** Trigger Mechanism Design
**Source Section:** Core Concepts - Concept 3
**Concept Map Node:** Triggers (Medium - 3 connections)
**Related Flashcard:** Card 3
**Expected Response Length:** 2-4 sentences

You are creating a skill that generates test cases from function signatures. Design three trigger specifications (mix of keywords and patterns) for this skill. Then explain one potential over-triggering scenario your triggers might cause and how you would mitigate it.

---

### Question 4 | Short Answer
**Cognitive Level:** Apply/Analyze
**Concept Tested:** Skill Chain Analysis
**Source Section:** Core Concepts - Concept 6
**Concept Map Node:** Skill Chain (High - 5 connections)
**Related Flashcard:** Card 4
**Expected Response Length:** 2-4 sentences

Consider a three-skill chain: `data-fetcher → data-transformer → report-generator`. The report-generator produces poor quality output. Using the concept of chain debugging, explain how you would systematically identify whether the problem originates in skill 1, skill 2, or skill 3. What role do output contracts play in this debugging process?

---

### Question 5 | Essay
**Cognitive Level:** Evaluate/Synthesize
**Concepts Tested:** All core concepts integrated
**Source Sections:** All Core Concepts, Practical Applications, Critical Analysis
**Concept Map:** Full pathway traversal
**Related Flashcard:** Card 5
**Expected Response Length:** 1-2 paragraphs

You are tasked with designing a skill ecosystem for automated technical writing. The ecosystem must transform rough technical notes into polished documentation.

Design a complete skill chain (minimum 3 skills) that accomplishes this goal. For each skill, specify: (1) its name and single responsibility, (2) key trigger specifications, (3) input and output contracts, and (4) its position in the chain (upstream/downstream relationships). Additionally, address: (5) how metadata should pass through the chain to maintain document coherence, (6) potential failure points and mitigation strategies, and (7) how the design achieves the quality properties of idempotency, reusability, and composability.

**Evaluation Criteria:**
- [ ] Designs coherent 3+ skill chain with clear single responsibilities
- [ ] Specifies triggers, contracts, and chain relationships for each skill
- [ ] Addresses metadata propagation for document coherence
- [ ] Identifies failure points with mitigation strategies
- [ ] Demonstrates understanding of quality properties

---

## Answer Key

### Question 1 | Multiple Choice
**Correct Answer:** C

**Explanation:**
Agent skills are runtime constructs that define how an agent executes capabilities—they do not involve training or fine-tuning the underlying model. Skills work with the model as-is, providing structured instructions, triggers, contracts, and metadata. The seven core components are: name/description, triggers, arguments, input contract, output contract, instructions, and chain metadata.

**Why Other Options Are Incorrect:**
- A) Triggers ARE a core component—they define activation conditions (keywords, patterns)
- B) Input/output contracts ARE core components—they specify data interfaces
- D) Chain metadata IS a core component—it declares skill relationships

**Understanding Gap Indicator:**
If answered incorrectly, this may indicate confusion between skill definition (runtime prompt engineering) and model fine-tuning (training-time modification). Skills modify agent behavior without changing model weights.

---

### Question 2 | Multiple Choice
**Correct Answer:** B

**Explanation:**
For skills to chain successfully, the upstream skill's output contract must satisfy the downstream skill's input contract. Skill B explicitly requires a `sentiment_score` field that Skill A does not produce. This contract mismatch will cause the chain to fail—Skill B cannot execute properly without its required input field.

**Why Other Options Are Incorrect:**
- A) Partial field overlap is insufficient; ALL required fields must be provided
- C) Language models should not "infer" missing structured data; this introduces unreliability
- D) The issue isn't extra fields (references); it's the MISSING required field (sentiment_score)

**Understanding Gap Indicator:**
If answered incorrectly, this may indicate incomplete understanding of contract strictness. Review the principle: output contracts must fully satisfy input contracts for chain compatibility.

---

### Question 3 | Short Answer
**Model Answer:**

**Trigger Specifications:**
```yaml
trigger:
  - keyword: generate test cases
  - keyword: create unit tests
  - pattern: "(generate|create|write) .* tests? (for|from)"
```

**Over-Triggering Scenario:**
The pattern `"(generate|create|write) .* tests? (for|from)"` could over-trigger on requests like "write documentation tests for the API" (documentation task, not test generation) or "create tests for evaluating student performance" (educational context, not code testing).

**Mitigation:**
Add negative constraints or more specific patterns:
```yaml
  - pattern: "(generate|create|write) .* (unit |integration )?tests? (for|from) .* (function|method|class|code)"
```
This adds code-specific terms to increase specificity while maintaining flexibility.

**Key Components Required:**
- [ ] Provides at least 3 triggers (mix of keywords and patterns)
- [ ] Identifies realistic over-triggering scenario
- [ ] Proposes specific mitigation approach

**Partial Credit Guidance:**
- Full credit: 3 appropriate triggers + specific over-trigger example + concrete mitigation
- Partial credit: Triggers provided but over-triggering discussion weak
- No credit: Triggers inappropriate for task or no over-triggering discussion

**Understanding Gap Indicator:**
If answered incorrectly, this may indicate difficulty translating task requirements into trigger specifications. Review the sensitivity vs. specificity tradeoff in trigger design.

---

### Question 4 | Short Answer
**Model Answer:**

**Systematic Chain Debugging:**
Isolate each skill by examining intermediate outputs at chain boundaries. First, run `data-fetcher` alone and validate its output against its declared output contract—does it produce all required fields in the correct format? If valid, run `data-transformer` with known-good input and validate its output. If both upstream skills produce contract-compliant outputs, the problem is in `report-generator`. If an earlier skill violates its contract, that's the failure point.

**Role of Output Contracts:**
Output contracts provide the validation schema for debugging. At each chain boundary, compare actual output against the declared contract:
- Are all required fields present?
- Are data types correct?
- Is the structure as specified?

Contract violations pinpoint where data quality degrades. Without explicit contracts, debugging requires inspecting actual content rather than structural validation.

**Key Components Required:**
- [ ] Describes systematic isolation approach (test each skill independently)
- [ ] Uses contracts as validation criteria at boundaries
- [ ] Explains how contracts enable structural rather than content-based debugging

**Partial Credit Guidance:**
- Full credit: Clear debugging methodology + contracts as validation schema
- Partial credit: General debugging concept but weak contract connection
- No credit: No systematic approach or misunderstands contract role

**Understanding Gap Indicator:**
If answered incorrectly, this may indicate difficulty applying contracts to practical debugging scenarios. Review how output contracts serve as testable specifications.

---

### Question 5 | Essay
**Model Answer:**

**Technical Writing Skill Chain Design:**

**Skill 1: content-extractor**
- **Responsibility:** Parse rough technical notes, identify key information, extract structured data
- **Triggers:** `keyword: extract content`, `pattern: "parse .* (notes|draft)"`
- **Input Contract:** Raw markdown/text files with technical content
- **Output Contract:** Structured JSON with {title, sections[], code_blocks[], key_concepts[], todos[]}
- **Chain Position:** `upstream: null`, `downstream: [content-organizer]`

**Skill 2: content-organizer**
- **Responsibility:** Organize extracted content into logical documentation structure
- **Triggers:** `keyword: organize content`, `pattern: "structure .* documentation"`
- **Input Contract:** Structured JSON from content-extractor (requires sections[], key_concepts[])
- **Output Contract:** Markdown outline with {hierarchy, section_order, cross_references[], style_notes}
- **Chain Position:** `upstream: [content-extractor]`, `downstream: [doc-polisher]`

**Skill 3: doc-polisher**
- **Responsibility:** Transform organized content into polished, publication-ready documentation
- **Triggers:** `keyword: polish documentation`, `pattern: "finalize .* docs"`
- **Input Contract:** Markdown outline from content-organizer + original content reference
- **Output Contract:** Final markdown documentation with consistent style, proper formatting, complete sections
- **Chain Position:** `upstream: [content-organizer]`, `downstream: null`

**Metadata Propagation:**
```yaml
metadata_pass:
  - original_source_path    # Traceability to source notes
  - document_title          # Consistent naming throughout
  - style_guide_reference   # Consistent formatting rules
  - key_concepts            # Terms requiring consistent treatment
```
The `key_concepts` metadata ensures technical terms are treated consistently across all skills. `style_guide_reference` maintains formatting coherence.

**Failure Points and Mitigation:**
| Transition | Failure Risk | Mitigation |
|------------|--------------|------------|
| Notes → Extractor | Unstructured notes missing clear sections | Fallback to heuristic parsing; flag for human review |
| Extractor → Organizer | Missing required fields | Validation gate; insert placeholder sections with TODOs |
| Organizer → Polisher | Style inconsistencies | Style guide in metadata; polisher applies normalization pass |

**Quality Properties:**
- **Idempotency:** Each skill produces deterministic output from same input—no randomness in extraction rules or organization logic
- **Reusability:** Skills are content-agnostic; content-extractor works for any technical notes (API docs, architecture docs, tutorials)
- **Composability:** Clear contracts enable substitution—a different polisher (e.g., `api-doc-polisher`) could replace `doc-polisher` for API-specific output

**Evaluation Rubric:**

| Criterion | Excellent (4) | Proficient (3) | Developing (2) | Beginning (1) |
|-----------|---------------|----------------|----------------|---------------|
| Chain Design | 3+ skills with clear single responsibilities, logical flow | 3 skills but some responsibility overlap | 2 skills or unclear responsibilities | <2 skills or incoherent flow |
| Specifications | Complete triggers, contracts, chain for each skill | Most specifications present | Partial specifications | Missing key specifications |
| Metadata | Specific metadata fields with coherence rationale | Metadata mentioned but generic | Vague metadata discussion | No metadata consideration |
| Failure Handling | Specific failures per transition with mitigations | General failure discussion | Acknowledges failures exist | No failure consideration |
| Quality Properties | Demonstrates all 3 properties with skill-specific examples | Addresses 2-3 properties | Mentions properties generically | No quality discussion |

**Understanding Gap Indicator:**
If response lacks depth, this may indicate:
- Difficulty decomposing complex workflows into single-responsibility skills
- Weak understanding of how contracts enable chain composition
- Limited awareness of production concerns (failure handling, metadata propagation)

---

## Diagnostic Feedback & Review Recommendations

### Performance-Based Review Guide

| If Incorrect On... | Gap Identified | Review Section | Priority |
|--------------------|----------------|----------------|----------|
| Question 1 | Skill component confusion | Core Concepts 1 & 2 | High |
| Question 2 | Contract matching principles | Core Concepts 4 & 6 | High |
| Question 3 | Trigger design application | Core Concepts 3 | Medium |
| Question 4 | Chain debugging methodology | Core Concepts 6 | Medium |
| Question 5 | Full skill synthesis | All sections | Low |

### Targeted Review Recommendations

#### For Multiple Choice Errors (Questions 1-2)
**Indicates:** Foundational concept gaps
**Action:** Review definitions and relationships in:
- Study Notes: Core Concepts 1, 2, 4
- Concept Map: Skill Definition cluster, Contract nodes
- Flashcards: Cards 1 and 2
**Focus On:** Distinguishing skill components from model training concepts

#### For Short Answer Errors (Questions 3-4)
**Indicates:** Application difficulties
**Action:** Practice applying concepts through:
- Study Notes: Practical Applications section
- Concept Map: Learning Pathway 2 (Contract-First)
- Flashcards: Cards 3 and 4
**Focus On:** Translating abstract concepts into concrete specifications

#### For Essay Weakness (Question 5)
**Indicates:** Integration challenges
**Action:** Review interconnections between:
- Concept Map: Critical Path traversal
- Study Notes: All Core Concepts + Case Study
- Flashcard: Card 5 (complete skill synthesis)
**Focus On:** Building skills from components rather than memorizing templates

### Mastery Indicators

- **5/5 Correct:** Strong mastery; proceed to implementing actual skills
- **4/5 Correct:** Good understanding; review indicated gap before complex skill design
- **3/5 Correct:** Moderate understanding; systematic review recommended
- **2/5 or below:** Foundational gaps; re-study from Concept Map Critical Path

---

## Skill Chain Traceability

```
Source Lesson (Lesson_1.md)
    │
    │  Topic: Creating Agent Skills
    │
    ▼
Study Notes ──────────────────────────────────────────────────────────┐
    │                                                                 │
    │  Extracted: 7 Core Concepts, 10 Key Terms                       │
    │                                                                 │
    ├────────────────────┬────────────────────┐                       │
    │                    │                    │                       │
    ▼                    ▼                    ▼                       │
Concept Map         Flashcards            Quiz                        │
    │                    │                    │                       │
    │  16 concepts       │  5 cards           │  5 questions          │
    │  24 relationships  │  2E/2M/1H          │  2MC/2SA/1E           │
    │  3 pathways        │                    │                       │
    │                    │                    │                       │
    └────────┬───────────┘                    │                       │
             │                                │                       │
             │  Centrality → Difficulty       │                       │
             │  Pathways → Review recs        │                       │
             │                                │                       │
             └────────────────────────────────┤                       │
                                              │                       │
                                    Quiz integrates all ◄─────────────┘
                                    Source: Lesson_1.md
```
