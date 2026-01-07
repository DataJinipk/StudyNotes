# Flashcard Set: Creating Agent Skills

**Source:** notes/agent-skills/creating-agent-skills-study-notes.md
**Concept Map Reference:** notes/agent-skills/concept-maps/creating-agent-skills-concept-map.md
**Original Source Path:** C:\agentic_ai\StudyNotes\notes\agent-skills\creating-agent-skills-study-notes.md
**Source Lesson:** Lessions/Lesson_1.md
**Date Generated:** 2026-01-06
**Total Cards:** 5
**Distribution:** 2 Easy | 2 Medium | 1 Hard

---

## Critical Knowledge Summary

Concepts appearing in multiple cards (prioritize for review):
- **Skill Definition**: Appears in Cards 1, 3, 5 (central concept)
- **Input/Output Contracts**: Appears in Cards 2, 4, 5 (interface specification)
- **Skill Chain**: Appears in Cards 4, 5 (composition pattern)
- **Triggers**: Appears in Cards 1, 3 (activation mechanism)

---

## Flashcards

---
### Card 1 | Easy
**Cognitive Level:** Remember
**Concept:** Skill Definition and Structure
**Source Section:** Core Concepts - Concepts 1 & 2
**Concept Map Centrality:** Critical (8 connections)

**FRONT (Question):**
What is an agent skill, and what are the seven core structural components of a complete skill definition?

**BACK (Answer):**
An **agent skill** is a self-contained, declarative module that defines a specific capability an AI agent can invoke, comprising metadata for identification, trigger conditions for activation, procedural instructions for execution, and interface specifications for integration.

**Seven Core Structural Components:**

| # | Component | Purpose |
|---|-----------|---------|
| 1 | **Name/Description** | Identification and discoverability |
| 2 | **Triggers** | Activation conditions (keywords, patterns) |
| 3 | **Arguments** | Parameter specifications |
| 4 | **Input Contract** | What the skill consumes |
| 5 | **Output Contract** | What the skill produces |
| 6 | **Instructions** | Step-by-step execution procedures |
| 7 | **Chain Metadata** | Upstream/downstream relationships |

*(Additional: Tools, Tags)*

**Critical Knowledge Flag:** Yes - Foundation for all skill creation

---

---
### Card 2 | Easy
**Cognitive Level:** Understand
**Concept:** Input/Output Contracts
**Source Section:** Core Concepts - Concept 4
**Concept Map Centrality:** High (Input: 4, Output: 5 connections)

**FRONT (Question):**
What are input/output contracts in skill design, and why are they essential for skill chaining?

**BACK (Answer):**
**Input Contract:** Specifies what data format, structure, and required fields a skill needs to execute successfully.
- Defines source type (markdown, JSON, etc.)
- Lists required sections or fields
- Enables validation before execution

**Output Contract:** Specifies what the skill produces, including format, location, and structure.
- Defines output type and schema
- Specifies file location patterns
- Lists included sections/components

**Why Essential for Chaining:**
Skills can only chain when **Skill A's output contract matches Skill B's input contract**. Without explicit contracts:
- Data format mismatches cause chain failures
- Missing fields break downstream processing
- No automated compatibility verification possible

Contracts enable "plug-and-play" composition of skills into workflows.

**Critical Knowledge Flag:** Yes - Required for any multi-skill workflow

---

---
### Card 3 | Medium
**Cognitive Level:** Apply
**Concept:** Trigger Mechanism Design
**Source Section:** Core Concepts - Concept 3
**Concept Map Centrality:** Medium (3 connections)

**FRONT (Question):**
Design the trigger specifications for a skill that generates API documentation from code comments. Include at least two keyword triggers and one pattern trigger. Explain the tradeoff between trigger sensitivity and specificity.

**BACK (Answer):**
**Trigger Specifications:**

```yaml
trigger:
  - keyword: api documentation
  - keyword: generate docs
  - keyword: document api
  - keyword: swagger docs
  - pattern: "(create|generate|build) .* (api|endpoint) (docs|documentation)"
  - pattern: "document .* (api|endpoints|routes)"
```

**Sensitivity vs. Specificity Tradeoff:**

| Aspect | High Sensitivity | High Specificity |
|--------|-----------------|------------------|
| **Goal** | Catch all relevant requests | Avoid false activations |
| **Risk** | Over-triggering on unrelated | Missing valid requests |
| **Example** | `pattern: ".*docs.*"` (too broad) | `keyword: "generate OpenAPI spec"` (too narrow) |

**Balanced Approach:**
- Use specific keywords for common exact phrases
- Use patterns for phrasal variations
- Avoid single-word triggers (too sensitive)
- Test triggers against corpus of expected AND unexpected inputs

**Critical Knowledge Flag:** Yes - Connects to Skill Definition activation

---

---
### Card 4 | Medium
**Cognitive Level:** Analyze
**Concept:** Skill Chain Design
**Source Section:** Core Concepts - Concept 6
**Concept Map Centrality:** High (5 connections)

**FRONT (Question):**
Analyze the skill chain: `study-notes-creator → concept-map → flashcards → quiz`. For each transition, identify: (1) what data must pass through, (2) potential failure points, and (3) what chain metadata enables the flow.

**BACK (Answer):**
**Chain Analysis:**

| Transition | Data Passed | Failure Points | Enabling Metadata |
|------------|-------------|----------------|-------------------|
| Notes → Concept Map | Full study notes with Core Concepts section | Missing required sections; malformed markdown | `upstream: study-notes-creator`; `required_sections: Core Concepts` |
| Concept Map → Flashcards | Concept hierarchy, centrality metrics, source path | Centrality data missing; path not preserved | `metadata_pass: [original_source_path, concepts_extracted]` |
| Flashcards → Quiz | Cards with difficulty levels, critical knowledge flags, source mappings | Insufficient depth for answer explanations | `upstream: [study-notes-creator, flashcards]`; `dual_input_mode` |

**Critical Chain Metadata:**

```yaml
chain:
  upstream:
    - study-notes-creator    # Primary source
  downstream:
    - flashcards
    - quiz
  metadata_pass:
    - original_source_path   # Traceability
    - concepts_extracted     # Content passed
    - critical_knowledge_flags  # Priority markers
```

**Key Insight:** The quiz skill requires **dual input** (flashcards + original notes) because flashcards lack answer explanation depth—a design decision documented in chain metadata.

**Critical Knowledge Flag:** Yes - Integrates Contracts and Chain concepts

---

---
### Card 5 | Hard
**Cognitive Level:** Synthesize
**Concept:** Complete Skill Design
**Source Section:** All Core Concepts; Practical Applications
**Concept Map Centrality:** Integrates all high-centrality nodes

**FRONT (Question):**
Synthesize a complete skill definition for a "code-reviewer" skill that analyzes pull requests and provides structured feedback. Address: (1) YAML metadata structure with triggers and contracts, (2) how it would chain with upstream/downstream skills, (3) procedural instruction phases, and (4) quality properties (idempotency, reusability, composability).

**BACK (Answer):**
**1. YAML Metadata Structure:**

```yaml
---
name: code-reviewer
description: Analyzes pull requests providing structured quality feedback
trigger:
  - keyword: review code
  - keyword: review pr
  - keyword: code review
  - pattern: "review .* (pr|pull request|changes)"
  - pattern: "check .* code (quality|style)"
arguments:
  - name: source
    description: PR URL, diff file, or code path
    required: true
  - name: focus_areas
    description: Specific aspects to emphasize (security, performance, style)
    required: false
    default: all
inputs:
  type: code_diff
  source: git diff, PR API, or file path
  required_fields:
    - changed_files
    - diff_content
outputs:
  type: markdown
  location: reviews/[pr-id]-review.md
  sections:
    - summary
    - issues_by_severity
    - suggestions
    - approval_recommendation
tools:
  - Read
  - Grep
  - Glob
  - Bash (git commands)
chain:
  upstream:
    - pr-fetcher
    - code-analyzer
  downstream:
    - fix-suggester
    - pr-commenter
  metadata_pass:
    - file_paths_reviewed
    - severity_counts
    - blocking_issues
tags:
  - development
  - code-quality
  - automation
---
```

**2. Chain Integration:**

```
pr-fetcher → code-reviewer → fix-suggester → pr-commenter
     │              │               │              │
     │              │               │              └─ Posts comments to PR
     │              │               └─ Generates fix code for issues
     │              └─ Analyzes and categorizes issues
     └─ Retrieves PR data from GitHub/GitLab
```

**3. Procedural Instruction Phases:**

| Phase | Actions | Output |
|-------|---------|--------|
| **1. Analysis** | Parse diff, identify changed files, extract code blocks | Structured change map |
| **2. Evaluation** | Check against style guide, security patterns, best practices | Issue list with severity |
| **3. Categorization** | Group by severity (critical/major/minor), type (bug/style/security) | Categorized findings |
| **4. Recommendation** | Synthesize findings into approval decision | Summary + recommendation |
| **5. Formatting** | Structure output per contract | Final review document |

**4. Quality Properties:**

| Property | How Achieved |
|----------|--------------|
| **Idempotency** | Same PR diff always produces same review (no randomness in evaluation criteria) |
| **Reusability** | Language-agnostic core with configurable style rules; works across projects |
| **Composability** | Clear input (diff) / output (structured review) contracts; chains with fix-suggester |

**Critical Knowledge Flag:** Yes - Integrates all core skill creation concepts

---

---

## Export Formats

### Anki-Compatible (Tab-Separated)
```
What is an agent skill and its 7 core components?	Self-contained module with: name/description, triggers, arguments, input contract, output contract, instructions, chain metadata.	easy::definition::skills
What are I/O contracts and why essential for chaining?	Input: what skill consumes. Output: what skill produces. Chaining requires output→input contract match for data flow.	easy::contracts::skills
Design triggers for API documentation skill	Keywords: "api documentation", "generate docs". Patterns: "(create|generate) .* api (docs|documentation)". Balance sensitivity vs specificity.	medium::triggers::skills
Analyze study-notes→concept-map→flashcards→quiz chain	Data: notes→concepts+centrality→cards+flags→quiz. Failures: missing sections, depth gaps. Metadata: upstream, metadata_pass, dual_input_mode.	medium::chaining::skills
Synthesize complete code-reviewer skill	YAML: triggers, contracts, tools. Chain: pr-fetcher→reviewer→fix-suggester. Phases: analyze→evaluate→categorize→recommend→format. Quality: idempotent, reusable, composable.	hard::synthesis::skills
```

### CSV Format
```csv
"Front","Back","Difficulty","Concept","Centrality"
"What is an agent skill?","Self-contained module defining agent capability with 7 components","Easy","Skill Definition","Critical"
"What are I/O contracts?","Input: what skill consumes. Output: what it produces. Enable chain compatibility.","Easy","Contracts","High"
"Design API doc skill triggers","Keywords + patterns balancing sensitivity vs specificity","Medium","Triggers","Medium"
"Analyze 4-skill chain","Data flow, failure points, enabling metadata for each transition","Medium","Skill Chain","High"
"Synthesize code-reviewer skill","Complete YAML, chain position, phases, quality properties","Hard","Full Design","Integration"
```

---

## Source Mapping

| Card | Source Section | Concept Map Node | Key Terminology |
|------|----------------|------------------|-----------------|
| 1 | Core Concepts 1 & 2 | Skill Definition, YAML Metadata | Skill, triggers, contracts |
| 2 | Core Concepts 4 | Input Contract, Output Contract | Contracts, chaining, validation |
| 3 | Core Concepts 3 | Triggers | Keywords, patterns, sensitivity |
| 4 | Core Concepts 6 | Skill Chain, Upstream, Downstream | Chain, metadata_pass, dual_input |
| 5 | All + Applications | All high-centrality nodes | Full skill anatomy |
