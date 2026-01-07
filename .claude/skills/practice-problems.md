---
name: practice-problems
description: Generate hands-on practice problems with scaffolded difficulty for skill-building and application mastery
trigger:
  - keyword: practice problems
  - keyword: exercises
  - keyword: practice exercises
  - keyword: hands-on practice
  - keyword: coding exercises
  - keyword: worked examples
  - pattern: "create .* (practice|exercises|problems)"
  - pattern: "generate .* (practice|exercises|problems)"
  - pattern: "give me .* (practice|problems|exercises)"
  - pattern: "(practice|exercise) (set|problems)"
arguments:
  - name: source
    description: Path to study notes or topic to create practice problems from
    required: true
  - name: count
    description: Total number of problems to generate
    required: false
    default: 5
  - name: type
    description: Problem type focus (conceptual, procedural, debugging, design)
    required: false
    default: mixed
  - name: include_solutions
    description: Whether to include full worked solutions
    required: false
    default: true
inputs:
  type: markdown
  source: study-notes-creator output OR concept-map output
  required_sections:
    - Core Concepts
  recommended_sections:
    - Practical Applications
    - Key Terminology
    - Review Questions
outputs:
  type: markdown
  location: notes/[subject]/practice/[topic]-practice-problems.md
  format: structured problem set with scaffolded hints and solutions
  sections:
    - problem_statements
    - hint_progressions
    - worked_solutions
    - common_mistakes
    - extension_challenges
tools:
  - Read
  - Write
  - Grep
  - Glob
chain:
  upstream:
    - study-notes-creator
    - concept-map
  downstream:
    - quiz
  parallel_with:
    - flashcards
    - concept-map
  metadata_pass:
    - original_source_path
    - concepts_practiced
    - difficulty_distribution
    - skill_gaps_targeted
tags:
  - education
  - practice
  - skill-building
  - hands-on
  - exercises
---

# Practice Problems

## Description
A professional skill for generating hands-on practice problems that build procedural fluency and application mastery. Creates scaffolded exercises with progressive hints, worked solutions, and common mistake warnings, targeting the gap between conceptual understanding and practical competence.

## Trigger
Use this skill when the user requests practice problems, exercises, hands-on practice, worked examples, or skill-building activities from study materials.

## Instructions

When creating practice problems from study notes, follow this systematic methodology:

### 1. Content Analysis Phase

Analyze the source study notes to:
- Identify procedural skills that require practice (not just recall)
- Extract application scenarios from Practical Applications section
- Map concepts to hands-on tasks that demonstrate mastery
- Determine prerequisite knowledge for problem scaffolding
- Identify common misconceptions to address through problem design

### 2. Problem Type Classification

Generate **5 practice problems** distributed across types:

| Problem Type | Count | Focus | Cognitive Demand |
|--------------|-------|-------|------------------|
| **Warm-Up** | 1 | Direct application of single concept | Apply |
| **Skill-Builder** | 2 | Multi-step procedural tasks | Apply/Analyze |
| **Challenge** | 1 | Complex scenario requiring synthesis | Analyze/Synthesize |
| **Debug/Fix** | 1 | Identify and correct errors | Evaluate |

### 3. Problem Structure

Generate each problem using this format:

```markdown
---
## Problem [N]: [Descriptive Title]

**Type:** [Warm-Up | Skill-Builder | Challenge | Debug/Fix]
**Concepts Practiced:** [List of concepts from study notes]
**Estimated Time:** [X minutes]
**Prerequisites:** [Required prior knowledge]

### Problem Statement

[Clear, unambiguous problem description]

[If applicable: starter code, data, or scenario setup]

### Requirements

- [ ] [Specific requirement 1]
- [ ] [Specific requirement 2]
- [ ] [Specific requirement 3]

### Hints (Progressive)

<details>
<summary>Hint 1: Getting Started</summary>

[Gentle nudge toward approach without revealing solution]

</details>

<details>
<summary>Hint 2: Key Insight</summary>

[More specific guidance on the core technique]

</details>

<details>
<summary>Hint 3: Nearly There</summary>

[Final hint that stops just short of the solution]

</details>

### Solution

<details>
<summary>Click to reveal full solution</summary>

**Approach:**
[Explanation of solution strategy]

**Step-by-Step Solution:**
[Detailed walkthrough]

**Final Answer/Code:**
[Complete solution]

**Why This Works:**
[Conceptual explanation connecting to study notes]

</details>

### Common Mistakes

- ❌ **Mistake:** [Common error]
  - **Why it happens:** [Root cause]
  - **How to avoid:** [Prevention strategy]

### Extension Challenge

[Optional harder variant for additional practice]

---
```

### 4. Problem Design Guidelines

#### Warm-Up Problems
- Single concept, direct application
- Builds confidence and activates prior knowledge
- Should be solvable in 5-10 minutes
- Success rate target: 90% of prepared learners

#### Skill-Builder Problems
- Combine 2-3 concepts in realistic scenario
- Require procedural execution, not just understanding
- Include intermediate checkpoints for self-assessment
- Success rate target: 70% of prepared learners

#### Challenge Problems
- Integrate multiple concepts with novel twist
- Require strategic thinking about approach
- May have multiple valid solutions
- Success rate target: 50% of prepared learners

#### Debug/Fix Problems
- Present flawed solution or broken implementation
- Require diagnosis before correction
- Target common misconceptions explicitly
- Build critical evaluation skills

### 5. Hint Progression Design

Hints should follow a scaffolded progression:

| Hint Level | Purpose | Example Pattern |
|------------|---------|-----------------|
| **Hint 1** | Orient to approach | "Consider what [concept] tells us about..." |
| **Hint 2** | Key technique | "The key insight is to use [technique] because..." |
| **Hint 3** | Almost solution | "Start by [specific step], then [next step]..." |

**Hint Principles:**
- Each hint should enable progress without full revelation
- Hints accumulate—reading all three approaches 80% solution
- Design so most learners need only 1-2 hints

### 6. Solution Quality Standards

Solutions must include:

```markdown
### Solution

**Approach:**
[1-2 sentences on overall strategy]

**Step-by-Step Solution:**

1. **[Step Name]:** [What to do]
   - [Why this step matters]

2. **[Step Name]:** [What to do]
   - [Connection to concept from notes]

3. **[Step Name]:** [What to do]
   - [Verification that step is correct]

**Final Answer:**
[Complete solution with all details]

**Conceptual Connection:**
This problem demonstrates [concept] from the study notes because [explanation].
The key insight is [principle] which we covered in [section reference].
```

### 7. Common Mistakes Section

For each problem, identify 2-3 common mistakes:

```markdown
### Common Mistakes

- ❌ **Mistake:** [What learners often do wrong]
  - **Why it happens:** [Misconception or oversight causing this]
  - **How to avoid:** [Specific strategy or check]
  - **Related concept:** [Link to study notes section]

- ❌ **Mistake:** [Another common error]
  - **Why it happens:** [Root cause]
  - **How to avoid:** [Prevention approach]
```

### 8. Problem Set Structure

```markdown
# Practice Problems: [Topic Title]

**Source:** [Study notes reference]
**Original Source Path:** [Full path for traceability]
**Date Generated:** [YYYY-MM-DD]
**Total Problems:** 5
**Estimated Total Time:** 60-90 minutes
**Distribution:** 1 Warm-Up | 2 Skill-Builder | 1 Challenge | 1 Debug/Fix

---

## Overview

### Concepts Practiced
| Concept | Problems | Mastery Indicator |
|---------|----------|-------------------|
| [Concept 1] | P1, P3 | Can apply without hints |
| [Concept 2] | P2, P4, P5 | Can identify errors |
| [Concept 3] | P3, P4 | Can synthesize with other concepts |

### Recommended Approach
1. Attempt each problem before looking at hints
2. Use hints progressively—don't skip to solution
3. After solving, read solution to compare approaches
4. Review Common Mistakes even if you solved correctly
5. Attempt Extension Challenges for deeper mastery

### Self-Assessment Guide
| Problems Solved (no hints) | Mastery Level | Recommendation |
|---------------------------|---------------|----------------|
| 5/5 | Expert | Proceed to advanced material |
| 4/5 | Proficient | Review one gap area |
| 3/5 | Developing | More practice recommended |
| 2/5 or below | Foundational | Re-review study notes first |

---

## Problems

[Individual problems follow]

---

## Summary

### Key Takeaways
- [Insight 1 reinforced by problems]
- [Insight 2 reinforced by problems]
- [Insight 3 reinforced by problems]

### Next Steps
- If struggled with [concept]: Review [section] in study notes
- For more practice: [Related problems or resources]
- Ready for assessment: Proceed to quiz skill
```

### 9. Quality Standards

Ensure all practice problems meet these criteria:

- **Authenticity:** Problems reflect real-world application scenarios
- **Scaffolding:** Difficulty progression is appropriate
- **Completeness:** Solutions are fully worked with explanations
- **Diagnosticity:** Mistakes reveal specific misconceptions
- **Engagement:** Problems are interesting, not just mechanical
- **Traceability:** Clear links to source study note concepts

### 10. Integration with Skill Chain

**Receiving from Concept Map:**
- High-centrality concepts become Challenge problem focus
- Relationship clusters inform multi-concept Skill-Builders
- Learning pathways suggest problem sequencing

**Feeding into Quiz:**
- Problems reveal which concepts need assessment emphasis
- Common mistakes inform quiz distractor design
- Mastery indicators guide question difficulty

## Tools to Utilize

- Read: For analyzing source study notes and concept maps
- Write: For saving generated problem sets to files
- Grep: For locating specific concepts and applications
- Glob: For finding relevant source files

## Example Invocation

User: "Create practice problems from my machine learning study notes"

Response should include:
- 5 scaffolded problems (1 warm-up, 2 skill-builder, 1 challenge, 1 debug)
- Progressive hints for each problem
- Fully worked solutions with conceptual connections
- Common mistakes with prevention strategies
- Self-assessment guide for mastery evaluation

## Output Location

Save generated problem sets to:
```
StudyNotes/
└── notes/
    └── [subject]/
        └── practice/
            └── [topic]-practice-problems.md
```
