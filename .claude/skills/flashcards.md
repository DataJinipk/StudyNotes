---
name: flashcards
description: Transform study notes into structured flashcards optimized for spaced repetition learning systems
trigger:
  - keyword: flashcards
  - keyword: spaced repetition
  - keyword: review cards
  - keyword: anki cards
  - keyword: memory cards
  - pattern: "create .* flashcards (from|for)"
  - pattern: "generate .* cards (from|for)"
  - pattern: "turn .* into flashcards"
arguments:
  - name: source
    description: Path to study notes file or topic name to create flashcards from
    required: true
  - name: count
    description: Total number of flashcards to generate
    required: false
    default: 5
  - name: distribution
    description: "Difficulty distribution (e.g., '2-2-1' for 2 easy, 2 medium, 1 hard)"
    required: false
    default: "2-2-1"
  - name: export_format
    description: Output format (markdown, anki, csv)
    required: false
    default: markdown
inputs:
  type: markdown
  source: study-notes-creator output or any structured study notes
  required_sections:
    - Core Concepts OR Key Terminology
outputs:
  type: markdown
  location: notes/[subject]/flashcards/[topic]-flashcards.md
  format: structured flashcard set with export formats
  exports:
    - anki (tab-separated)
    - csv
    - plain text
tools:
  - Read
  - Write
  - Grep
  - Glob
chain:
  upstream:
    - study-notes-creator
  downstream:
    - quiz
  metadata_pass:
    - original_source_path
    - concepts_extracted
    - critical_knowledge_flags
tags:
  - education
  - memorization
  - spaced-repetition
  - study-materials
---

# Flashcards

## Description
A professional skill for transforming study notes into structured flashcards optimized for spaced repetition learning systems. Generates tiered difficulty cards aligned with Bloom's Taxonomy cognitive levels.

## Trigger
Use this skill when the user requests flashcards, spaced repetition cards, or review cards from existing study notes or educational content.

## Instructions

When creating flashcards from study notes, follow this systematic methodology:

### 1. Content Analysis Phase

Analyze the source study notes to:
- Identify core concepts, definitions, and terminology
- Extract key relationships and dependencies between concepts
- Determine hierarchical importance of information
- Map concepts to appropriate cognitive complexity levels
- Identify concepts that serve as foundational knowledge (critical knowledge indicators)

### 2. Difficulty Classification Framework

Generate exactly **5 flashcards** distributed across cognitive levels:

| Difficulty | Count | Cognitive Level | Card Type |
|------------|-------|-----------------|-----------|
| Easy | 2 | Remember/Understand | Definition-level: terminology, basic facts, simple recall |
| Medium | 2 | Apply/Analyze | Application-level: scenarios, problem-solving, comparisons |
| Hard | 1 | Evaluate/Synthesize | Synthesis-level: integration of concepts, critical evaluation |

### 3. Flashcard Structure

Generate each flashcard using this format:

```markdown
---
### Card [N] | [Difficulty Level]
**Cognitive Level:** [Bloom's Taxonomy Level]
**Concept:** [Primary concept being tested]
**Source Section:** [Section reference from study notes]

**FRONT (Question):**
[Clear, unambiguous question or prompt]

**BACK (Answer):**
[Comprehensive yet concise answer]

**Critical Knowledge Flag:** [Yes/No - if concept appears in multiple cards]
---
```

### 4. Card Generation Guidelines

#### Easy Cards (Definition-Level)
- Focus on "What is...?" and "Define..." questions
- Target single concepts or terms
- Answers should be concise (1-3 sentences)
- Suitable for initial learning and recognition

#### Medium Cards (Application-Level)
- Focus on "How would you...?" and "What happens when...?" questions
- Require application of knowledge to scenarios
- May involve comparison or contrast between concepts
- Answers should demonstrate practical understanding

#### Hard Cards (Synthesis-Level)
- Focus on "Why does...?" and "How do X and Y interact to...?" questions
- Require integration of multiple concepts
- Demand evaluation or critical analysis
- Answers should demonstrate deep conceptual understanding

### 5. Spaced Repetition Optimization

Format output for compatibility with common spaced repetition tools:

```markdown
## Flashcard Set: [Topic Title]

**Source:** [Original study notes reference]
**Original Source Path:** [Full path for chain traceability]
**Date Generated:** [YYYY-MM-DD]
**Total Cards:** 5
**Distribution:** 2 Easy | 2 Medium | 1 Hard

---

## Critical Knowledge Summary
Concepts appearing in multiple cards (prioritize for review):
- [Concept 1]: Appears in Cards [X, Y]
- [Concept 2]: Appears in Cards [X, Z]

---

## Flashcards

[Individual cards follow]

---

## Export Formats

### Anki-Compatible (Tab-Separated)
[Front]\t[Back]\t[Tags]

### CSV Format
"Front","Back","Difficulty","Concept"

### Plain Text Review
Q: [Question]
A: [Answer]

---

## Source Mapping
| Card | Source Section | Key Terminology Used |
|------|----------------|---------------------|
| 1 | [Section] | [Terms] |
```

### 6. Quality Standards

Ensure all flashcards meet these criteria:

- **Atomicity:** Each card tests exactly one piece of knowledge
- **Clarity:** Questions are unambiguous and answers are definitive
- **Relevance:** Cards focus on high-yield, examinable content
- **Consistency:** Uniform formatting across all cards
- **Traceability:** Cards can be linked back to source material
- **Balance:** Appropriate cognitive load distribution

### 7. Critical Knowledge Flagging

After generating all cards, analyze for concept overlap:
- Identify concepts referenced in multiple cards
- Mark these as "Critical Knowledge" requiring priority review
- Provide a summary section listing all flagged concepts
- Note which cards share common underlying principles

## Tools to Utilize

- Read: For analyzing source study notes
- Write: For saving generated flashcard sets to files
- Grep: For searching notes for key terminology
- Glob: For locating relevant study note files

## Example Invocation

User: "Create flashcards from my machine learning study notes"

Response should include:
- Analysis summary of source material
- 5 structured flashcards (2 Easy, 2 Medium, 1 Hard)
- Critical knowledge flags for overlapping concepts
- Export-ready formats for Anki/CSV
- Recommendations for review scheduling

## Output Location

Save generated flashcard sets to:
```
StudyNotes/
└── notes/
    └── [subject]/
        └── flashcards/
            └── [topic]-flashcards.md
```
