---
name: quiz
description: Generate comprehensive assessment quizzes from study notes with answer keys, diagnostic feedback, and targeted review recommendations
trigger:
  - keyword: quiz
  - keyword: assessment
  - keyword: practice test
  - keyword: self-evaluation
  - keyword: test questions
  - keyword: exam questions
  - pattern: "create .* quiz (from|for|on)"
  - pattern: "generate .* (quiz|test|assessment)"
  - pattern: "test me on"
  - pattern: "assess .* (knowledge|understanding)"
arguments:
  - name: source
    description: Path to study notes or flashcards file to create quiz from
    required: true
  - name: question_count
    description: Total number of questions to generate
    required: false
    default: 5
  - name: distribution
    description: "Question type distribution (e.g., '2-2-1' for 2 MC, 2 SA, 1 Essay)"
    required: false
    default: "2-2-1"
  - name: include_answers
    description: Whether to include answer key in output
    required: false
    default: true
inputs:
  type: markdown
  source: study-notes-creator output OR flashcards output (with original notes reference)
  required_sections:
    - Core Concepts OR Flashcards
  recommended_sections:
    - Key Terminology
    - Review Questions
    - Critical Knowledge Summary
outputs:
  type: markdown
  location: notes/[subject]/quizzes/[topic]-quiz.md
  format: structured quiz with questions, answer key, and diagnostic feedback
  sections:
    - questions (2 MC, 2 SA, 1 Essay)
    - answer_key
    - diagnostic_feedback
    - review_recommendations
tools:
  - Read
  - Write
  - Grep
  - Glob
chain:
  upstream:
    - study-notes-creator
    - flashcards
  downstream: null
  dual_input_mode:
    primary: flashcards
    supplementary: study-notes-creator
    fallback: Use original study notes for answer explanations when flashcards lack depth
tags:
  - education
  - assessment
  - evaluation
  - study-materials
---

# Quiz

## Description
A professional skill for generating comprehensive assessment quizzes from study notes. Creates varied question types aligned with cognitive taxonomy levels, complete with answer keys, diagnostic feedback, and targeted review recommendations.

## Trigger
Use this skill when the user requests quizzes, assessments, practice tests, or self-evaluation materials from existing study notes or educational content.

## Instructions

When creating assessment quizzes from study notes, follow this systematic methodology:

### 1. Content Analysis Phase

Analyze the source study notes to:
- Identify core concepts, principles, and terminology
- Map conceptual relationships and dependencies
- Determine testable knowledge components
- Classify content by cognitive complexity (Bloom's Taxonomy)
- Note section boundaries for review recommendations

### 2. Question Type Distribution

Generate exactly **5 quiz questions** distributed as follows:

| Question Type | Count | Cognitive Level | Assessment Focus |
|---------------|-------|-----------------|------------------|
| Multiple Choice | 2 | Remember/Understand | Conceptual knowledge, definitions, factual recall |
| Short Answer | 2 | Apply/Analyze | Application of concepts, problem-solving, analysis |
| Essay | 1 | Evaluate/Synthesize | Integration, critical evaluation, original synthesis |

### 3. Quiz Structure

Generate the quiz using this format:

```markdown
# Assessment Quiz: [Topic Title]

**Source Material:** [Study notes reference]
**Original Source Path:** [Full path for chain traceability]
**Date Generated:** [YYYY-MM-DD]
**Total Questions:** 5
**Estimated Completion Time:** 20-30 minutes
**Distribution:** 2 Multiple Choice | 2 Short Answer | 1 Essay

---

## Instructions
- Multiple Choice: Select the single best answer
- Short Answer: Respond in 2-4 sentences
- Essay: Provide a comprehensive response (1-2 paragraphs)

---

## Questions

[Questions follow]

---

## Answer Key

[Answer key with explanations]

---

## Diagnostic Feedback

[Gap analysis and review recommendations]
```

### 4. Question Generation Guidelines

#### Multiple Choice Questions (Conceptual)

```markdown
### Question 1 | Multiple Choice
**Cognitive Level:** Remember/Understand
**Concept Tested:** [Primary concept]
**Source Section:** [Section reference from study notes]

[Clear, unambiguous question stem]

A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]

---
```

**Design Principles:**
- One clearly correct answer with plausible distractors
- Distractors should reflect common misconceptions
- Avoid "all of the above" or "none of the above"
- Question stem should be complete and self-contained

#### Short Answer Questions (Application)

```markdown
### Question 3 | Short Answer
**Cognitive Level:** Apply/Analyze
**Concept Tested:** [Primary concept]
**Source Section:** [Section reference from study notes]
**Expected Response Length:** 2-4 sentences

[Question requiring application or analysis of concepts]

---
```

**Design Principles:**
- Require demonstration of understanding, not mere recall
- Present scenarios or problems requiring concept application
- Allow for partial credit through multi-component responses
- Clear parameters for expected response scope

#### Essay Question (Synthesis)

```markdown
### Question 5 | Essay
**Cognitive Level:** Evaluate/Synthesize
**Concepts Tested:** [Multiple concepts integrated]
**Source Sections:** [Multiple section references]
**Expected Response Length:** 1-2 paragraphs

[Question requiring integration of multiple concepts and critical evaluation]

**Evaluation Criteria:**
- [ ] Addresses all components of the question
- [ ] Demonstrates integration of multiple concepts
- [ ] Provides evidence-based reasoning
- [ ] Shows critical evaluation or original synthesis

---
```

**Design Principles:**
- Require synthesis of multiple concepts from different sections
- Demand evaluation, comparison, or original analysis
- Provide clear evaluation criteria
- Allow demonstration of deep understanding

### 5. Answer Key Structure

```markdown
## Answer Key

### Question 1 | Multiple Choice
**Correct Answer:** [Letter]

**Explanation:**
[Detailed explanation of why this answer is correct]

**Why Other Options Are Incorrect:**
- A) [If incorrect: explanation of misconception]
- B) [If incorrect: explanation of misconception]
- C) [If incorrect: explanation of misconception]
- D) [If incorrect: explanation of misconception]

**Understanding Gap Indicator:**
If answered incorrectly, this may indicate:
- [Specific knowledge gap or misconception]

---

### Question 3 | Short Answer
**Model Answer:**
[Comprehensive model response]

**Key Components Required:**
- [ ] [Component 1]
- [ ] [Component 2]
- [ ] [Component 3]

**Partial Credit Guidance:**
- Full credit: All key components addressed accurately
- Partial credit: [Criteria for partial credit]
- No credit: [Criteria indicating fundamental misunderstanding]

**Understanding Gap Indicator:**
If answered incompletely or incorrectly, this may indicate:
- [Specific knowledge gap]

---

### Question 5 | Essay
**Model Answer:**
[Comprehensive model response demonstrating synthesis]

**Evaluation Rubric:**

| Criterion | Excellent (4) | Proficient (3) | Developing (2) | Beginning (1) |
|-----------|---------------|----------------|----------------|---------------|
| Concept Integration | [Description] | [Description] | [Description] | [Description] |
| Critical Analysis | [Description] | [Description] | [Description] | [Description] |
| Evidence & Reasoning | [Description] | [Description] | [Description] | [Description] |
| Communication | [Description] | [Description] | [Description] | [Description] |

**Understanding Gap Indicator:**
If response lacks depth, this may indicate:
- [Specific conceptual gaps]

---
```

### 6. Diagnostic Feedback Section

```markdown
## Diagnostic Feedback & Review Recommendations

### Performance-Based Review Guide

| If Incorrect On... | Gap Identified | Review Section | Priority |
|--------------------|----------------|----------------|----------|
| Question 1 | [Knowledge gap] | [Specific section] | High |
| Question 2 | [Knowledge gap] | [Specific section] | High |
| Question 3 | [Knowledge gap] | [Specific section] | Medium |
| Question 4 | [Knowledge gap] | [Specific section] | Medium |
| Question 5 | [Knowledge gap] | [Specific sections] | Low |

### Targeted Review Recommendations

#### For Multiple Choice Errors (Questions 1-2)
**Indicates:** Foundational concept gaps
**Action:** Review definitions and core principles in:
- [Section reference 1]
- [Section reference 2]
**Focus On:** [Specific concepts to reinforce]

#### For Short Answer Errors (Questions 3-4)
**Indicates:** Application or analysis difficulties
**Action:** Practice applying concepts through:
- [Section reference with examples]
- [Section reference with case studies]
**Focus On:** [Specific skills to develop]

#### For Essay Weakness (Question 5)
**Indicates:** Integration or synthesis challenges
**Action:** Review interconnections between:
- [Section reference 1] and [Section reference 2]
- [Theoretical framework section]
**Focus On:** [Higher-order thinking development]

### Mastery Indicators

- **5/5 Correct:** Strong mastery demonstrated; proceed to advanced material
- **4/5 Correct:** Good understanding; review indicated gap area
- **3/5 Correct:** Moderate understanding; systematic review recommended
- **2/5 or below:** Foundational gaps; comprehensive re-study advised
```

### 7. Quality Standards

Ensure all quiz questions meet these criteria:

- **Validity:** Questions accurately measure intended knowledge
- **Reliability:** Clear, unambiguous wording produces consistent interpretation
- **Alignment:** Questions map directly to study note content
- **Fairness:** No trick questions or misleading phrasing
- **Discrimination:** Questions differentiate levels of understanding
- **Actionability:** Feedback provides clear remediation pathways

## Tools to Utilize

- Read: For analyzing source study notes
- Write: For saving generated quizzes to files
- Grep: For locating specific concepts within notes
- Glob: For finding relevant study note files

## Example Invocation

User: "Create a quiz from my cognitive psychology study notes"

Response should include:
- 5 structured questions (2 MC, 2 SA, 1 Essay)
- Complete answer key with explanations
- Gap indicators for each question
- Section-specific review recommendations
- Performance interpretation guide

## Output Location

Save generated quizzes to:
```
StudyNotes/
└── notes/
    └── [subject]/
        └── quizzes/
            └── [topic]-quiz.md
```
