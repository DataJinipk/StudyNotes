---
name: study-notes-creator
description: Generate comprehensive, academic-level study notes suitable for advanced learners, graduate students, and professionals
trigger:
  - keyword: study notes
  - keyword: lesson summary
  - keyword: educational content
  - keyword: create notes
  - keyword: generate notes
  - pattern: "create .* notes (on|about|for)"
  - pattern: "study (guide|material) (on|about|for)"
arguments:
  - name: topic
    description: The subject or topic to create study notes for
    required: true
  - name: complexity
    description: Target complexity level (introductory, intermediate, advanced)
    required: false
    default: advanced
  - name: output_path
    description: Custom output path for the generated notes
    required: false
inputs: null
outputs:
  type: markdown
  location: notes/[subject]/[topic]-study-notes.md
  format: structured academic notes with learning objectives, core concepts, and review questions
tools:
  - WebSearch
  - WebFetch
  - Read
  - Write
chain:
  upstream: null
  downstream:
    - flashcards
    - quiz
tags:
  - education
  - academic
  - content-generation
  - study-materials
---

# Study Notes Creator

## Description
A professional skill for generating comprehensive, academic-level study notes suitable for advanced learners, graduate students, and professionals.

## Trigger
Use this skill when the user requests study notes, lesson summaries, or educational content creation.

## Instructions

When creating study notes, follow this systematic methodology:

### 1. Content Analysis Phase
- Identify the core subject matter and discipline
- Determine the complexity level and target audience
- Extract key concepts, theories, and frameworks
- Map relationships between topics

### 2. Structure and Organization

Generate notes using this professional format:

```markdown
# [Topic Title]

## Learning Objectives
- List 3-5 measurable learning outcomes using Bloom's Taxonomy verbs
- Focus on higher-order thinking skills (analyze, evaluate, synthesize)

## Executive Summary
Provide a concise 2-3 paragraph overview of the topic's significance and scope.

## Core Concepts

### [Concept 1]
**Definition:** Precise academic definition
**Explanation:** Detailed elaboration with context
**Key Points:**
- Point 1
- Point 2
- Point 3

### [Concept 2]
[Repeat structure]

## Theoretical Framework
- Present underlying theories and models
- Include scholarly perspectives and schools of thought
- Reference seminal works and researchers

## Practical Applications
- Real-world implementations
- Case studies or examples
- Industry relevance

## Critical Analysis
- Strengths and limitations of presented concepts
- Current debates in the field
- Areas requiring further research

## Key Terminology
| Term | Definition | Context |
|------|------------|---------|
| Term 1 | Definition | Usage context |

## Review Questions
1. Comprehension questions
2. Application questions
3. Analysis questions
4. Synthesis questions

## Further Reading
- Recommended academic sources
- Supplementary materials
- Related topics for exploration

## Summary
Concise recapitulation of main points and their interconnections.
```

### 3. Quality Standards

Ensure all notes adhere to these professional criteria:

- **Accuracy:** Verify factual correctness and currency of information
- **Clarity:** Use precise academic language without unnecessary jargon
- **Depth:** Provide sufficient detail for professional-level understanding
- **Coherence:** Maintain logical flow and clear transitions
- **Citation-Ready:** Format information for easy academic referencing
- **Visual Organization:** Use hierarchical structure, tables, and formatting

### 4. Enhancement Features

Include where appropriate:
- Diagrams or visual representation descriptions
- Mnemonics for complex information retention
- Cross-references to related topics
- Common misconceptions and clarifications
- Historical context and evolution of concepts

### 5. Output Specifications

- Use formal academic tone throughout
- Employ discipline-specific terminology with definitions
- Structure content for both linear reading and reference lookup
- Include metadata: topic, date, complexity level
- Format for easy export to various note-taking systems

## Tools to Utilize
- WebSearch: For current developments and recent research
- WebFetch: For retrieving specific academic resources
- Read: For analyzing source materials provided by user
- Write: For saving generated notes to files

## Example Invocation

User: "Create study notes on machine learning optimization algorithms"

Response should include:
- Comprehensive coverage of gradient descent variants
- Mathematical foundations with notation
- Convergence properties and proofs overview
- Hyperparameter considerations
- Comparative analysis of algorithms
- Implementation considerations
- Current research directions
