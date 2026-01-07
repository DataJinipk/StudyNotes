# Practice Problems: Lesson 1 - Creating Agent Skills

**Source:** Lessons/Lesson_1.md
**Concept Map Reference:** notes/agent-skills/concept-maps/lesson-1-agent-skills-concept-map.md
**Flashcard Reference:** notes/agent-skills/flashcards/lesson-1-agent-skills-flashcards.md
**Date Generated:** 2026-01-07
**Total Problems:** 5
**Estimated Total Time:** 90-120 minutes
**Distribution:** 1 Warm-Up | 2 Skill-Builder | 1 Challenge | 1 Debug/Fix

---

## Overview

### Concepts Practiced
| Concept | Problems | Mastery Indicator |
|---------|----------|-------------------|
| Agent Skill Definition | P1, P4 | Can articulate all core components |
| Invocation Patterns | P2, P4 | Can design appropriate triggers for use cases |
| Skill Composition | P3, P5 | Can chain skills with proper data flow |
| Theoretical Foundations | P4 | Can apply design principles to decisions |
| Quality Dimensions | P4, P5 | Can evaluate and debug skill designs |

### Recommended Approach
1. Attempt each problem before looking at hints
2. Use hints progressively—don't skip to solution
3. After solving, read solution to compare approaches
4. Review Common Mistakes even if you solved correctly
5. Attempt Extension Challenges for deeper mastery

### Self-Assessment Guide
| Problems Solved (no hints) | Mastery Level | Recommendation |
|---------------------------|---------------|----------------|
| 5/5 | Expert | Ready to design production skills |
| 4/5 | Proficient | Review one gap area |
| 3/5 | Developing | More practice recommended |
| 2/5 or below | Foundational | Re-review Lesson 1 study notes first |

---

## Problems

---

## Problem 1: Agent Skill Component Analysis

**Type:** Warm-Up
**Concepts Practiced:** Agent Skill Definition, Core Components
**Estimated Time:** 15 minutes
**Prerequisites:** Understanding of skill architecture from Lesson 1

### Problem Statement

You are given the following skill description:

> "A skill that analyzes code commits and generates release notes in markdown format. Users invoke it by typing `/release-notes [commit-range]`. The skill reads git commit history, categorizes changes by type (feature, fix, refactor), and outputs a formatted changelog. It requires git access and outputs to a specified file path."

Identify and explicitly label each of the five core architectural components of an agent skill within this description.

### Requirements

- [ ] Identify the **Invocation Mechanism** with specific details
- [ ] Identify the **Input Validation Logic** requirements
- [ ] Identify the **Execution Procedures** (what the skill does)
- [ ] Identify the **Output Formatting Specifications**
- [ ] Identify implied **Error Handling** needs
- [ ] Map each component to the text evidence

### Hints (Progressive)

<details>
<summary>Hint 1: Getting Started</summary>

Look for these signals in the description:
- Invocation: How does the user trigger it? What syntax?
- Input: What parameters are needed? What must be validated?
- Execution: What actions does the skill perform?
- Output: What format? Where does it go?

</details>

<details>
<summary>Hint 2: Key Insight</summary>

Some components are explicit (invocation is clearly `/release-notes [commit-range]`), while others are implicit. Error handling isn't stated but can be inferred from what could go wrong.

Ask: "What could fail?" for each step to identify error handling needs.

</details>

<details>
<summary>Hint 3: Nearly There</summary>

For Input Validation, consider:
- Is the commit-range format valid?
- Does the specified range exist in the repository?

For Error Handling, consider:
- What if there are no commits in range?
- What if git access fails?

</details>

### Solution

<details>
<summary>Click to reveal full solution</summary>

**Approach:**
Systematically extract each architectural component by analyzing the description for explicit statements and inferring implicit requirements.

**Component Analysis:**

| Component | Extracted Details | Text Evidence |
|-----------|-------------------|---------------|
| **Invocation Mechanism** | Slash command: `/release-notes [commit-range]` | "Users invoke it by typing `/release-notes [commit-range]`" |
| **Input Validation Logic** | Must validate: (1) commit-range format is valid git range syntax (e.g., `v1.0..v1.1` or `HEAD~5..HEAD`), (2) specified commits exist in repository, (3) user has git access permissions | "requires git access", "[commit-range]" parameter |
| **Execution Procedures** | (1) Read git commit history for specified range, (2) Parse commit messages, (3) Categorize by type (feature/fix/refactor), (4) Aggregate and format results | "reads git commit history, categorizes changes by type (feature, fix, refactor)" |
| **Output Formatting** | Markdown-formatted changelog, written to specified file path | "generates release notes in markdown format", "outputs to a specified file path" |
| **Error Handling** | Must handle: (1) Invalid commit range syntax, (2) Non-existent commits, (3) Git access failures, (4) Empty commit range, (5) File write permission errors | Inferred from execution requirements |

**Detailed Error Handling Scenarios:**

```
Error Scenario              → Appropriate Response
────────────────────────────────────────────────────────
Invalid range syntax        → Clear error message with valid format examples
Commits not found           → "Commit range not found in repository"
Git access denied           → "Unable to access git repository. Check permissions."
No commits in range         → Generate empty changelog with note
Cannot write output file    → "Cannot write to path. Check permissions."
```

**Why This Matters:**
Understanding component structure enables you to:
1. Design skills systematically rather than ad-hoc
2. Ensure all necessary elements are specified
3. Anticipate failure modes before implementation
4. Create consistent, maintainable skill definitions

</details>

### Common Mistakes

- ❌ **Mistake:** Confusing execution procedures with output formatting
  - **Why it happens:** Both describe "what the skill produces"
  - **How to avoid:** Execution = actions taken; Output = format/structure of results

- ❌ **Mistake:** Overlooking implicit error handling requirements
  - **Why it happens:** Focusing only on "happy path" described in text
  - **How to avoid:** For each step, ask "What could go wrong?"

- ❌ **Mistake:** Treating the commit-range as just a string parameter
  - **Why it happens:** Not considering validation requirements
  - **How to avoid:** All inputs need validation—consider format, existence, permissions

### Extension Challenge

Extend this skill definition to support an additional invocation pattern: natural language triggers like "Generate release notes for the last week" or "Create changelog since version 2.0". What additional input validation would be needed?

---

---

## Problem 2: Invocation Pattern Design

**Type:** Skill-Builder
**Concepts Practiced:** Invocation Patterns, Slash Commands, Natural Language Triggers
**Estimated Time:** 20 minutes
**Prerequisites:** Understanding of invocation pattern types and trade-offs

### Problem Statement

Design comprehensive invocation patterns for a **code-review skill** that analyzes pull requests and provides structured feedback. Your design must include:

1. **Slash command** with appropriate parameters
2. **At least three natural language trigger examples**
3. **Trade-off analysis** for your design decisions

Consider that this skill will be used by:
- Senior developers who review many PRs daily (power users)
- Junior developers who review occasionally (casual users)
- CI/CD systems that trigger reviews automatically (programmatic users)

### Requirements

- [ ] Design slash command with required and optional parameters
- [ ] Provide three diverse natural language triggers
- [ ] Analyze discoverability vs. flexibility trade-offs
- [ ] Address the needs of all three user types
- [ ] Consider parameter disambiguation for natural language

### Hints (Progressive)

<details>
<summary>Hint 1: Getting Started</summary>

Start with the slash command. What information does a code review need?
- Which PR/code to review
- What aspects to focus on (security, performance, style)
- Output preferences

Format: `/command required-param --optional=value`

</details>

<details>
<summary>Hint 2: Key Insight</summary>

Different user types need different invocation styles:
- Power users: Fast, precise slash commands with shortcuts
- Casual users: Forgiving natural language
- Programmatic: Predictable, explicit parameters

Natural language triggers should cover varied phrasings:
- "Review this PR"
- "Check the code in PR #123"
- "Analyze the pull request for security issues"

</details>

<details>
<summary>Hint 3: Nearly There</summary>

For trade-off analysis, consider:
- Discoverability: Can users find how to use it?
- Flexibility: Does it handle varied inputs?
- Precision: Can users specify exactly what they want?
- Error-proneness: How easily can users make mistakes?

CI/CD systems need programmatic invocation—consider API-style parameter passing.

</details>

### Solution

<details>
<summary>Click to reveal full solution</summary>

**Approach:**
Design layered invocation supporting all user types, then analyze trade-offs.

**1. Slash Command Design:**

```
/review-code <source> [--focus=<areas>] [--depth=<level>] [--format=<output>]

Required:
  <source>         PR URL, PR number, diff file path, or branch name

Optional:
  --focus          Comma-separated: security,performance,style,logic,tests
                   Default: all
  --depth          quick | standard | thorough
                   Default: standard
  --format         inline | summary | detailed | json
                   Default: detailed

Examples:
  /review-code #142
  /review-code https://github.com/org/repo/pull/142
  /review-code #142 --focus=security,performance --depth=thorough
  /review-code feature-branch --format=json
```

**2. Natural Language Triggers:**

| Trigger Example | Interpretation | Disambiguation |
|-----------------|----------------|----------------|
| "Review PR #142" | source=#142, defaults for rest | Number extraction |
| "Do a quick security review of the authentication changes" | focus=security, depth=quick, source=recent auth PR | Context-aware source detection |
| "Check this pull request for performance issues" | focus=performance, source=linked/current PR | Requires PR context |
| "Analyze the code changes in feature-login branch" | source=feature-login, depth=standard | Branch name extraction |
| "Give me a thorough code review with JSON output" | depth=thorough, format=json, source=current context | Requires source context |

**3. Programmatic Invocation (CI/CD):**

```yaml
# GitHub Actions example
- name: Code Review
  uses: agent-skills/review-code@v1
  with:
    source: ${{ github.event.pull_request.number }}
    focus: security,tests
    depth: thorough
    format: json
    output-file: review-results.json
```

**4. Trade-off Analysis:**

| Dimension | Slash Command | Natural Language | Programmatic |
|-----------|---------------|------------------|--------------|
| **Discoverability** | High (documented, autocomplete) | Low (must guess phrasing) | Medium (API docs) |
| **Flexibility** | Medium (fixed parameters) | High (varied expressions) | Low (strict schema) |
| **Precision** | High (explicit values) | Low (may need clarification) | High (exact values) |
| **Error-proneness** | Low (validation) | Medium (misinterpretation) | Low (schema validation) |
| **Learning curve** | Medium | Low | High (for setup) |
| **Best for** | Power users | Casual users | Automation |

**Design Decisions & Rationale:**

| Decision | Rationale |
|----------|-----------|
| Source as positional (required) | Most essential; reduces typing for common case |
| Focus as optional with "all" default | Beginners get comprehensive review; experts customize |
| Depth levels as names not numbers | "thorough" clearer than "3"; self-documenting |
| JSON format option | Enables programmatic post-processing |

**User Type Satisfaction:**

| User Type | Recommended Invocation | Why |
|-----------|----------------------|-----|
| Senior Dev (power) | `/review-code #142 --focus=security` | Fast, precise, memorable |
| Junior Dev (casual) | "Review this PR for me" | No syntax to learn |
| CI/CD (programmatic) | YAML/JSON with explicit parameters | Predictable, versionable |

</details>

### Common Mistakes

- ❌ **Mistake:** Only designing for one user type
  - **Why it happens:** Designing from personal preference
  - **How to avoid:** Explicitly list user personas; verify each is served

- ❌ **Mistake:** Too many required parameters
  - **Why it happens:** Wanting to capture all information upfront
  - **How to avoid:** Make common cases simple; use sensible defaults

- ❌ **Mistake:** Ambiguous natural language without clarification strategy
  - **Why it happens:** Assuming AI will "figure it out"
  - **How to avoid:** Define disambiguation prompts for ambiguous cases

### Extension Challenge

Design an error recovery flow for when natural language input is ambiguous. For example, if a user says "Review the changes" without specifying which PR, what clarification dialogue should the skill initiate?

---

---

## Problem 3: Skill Chain Design

**Type:** Skill-Builder
**Concepts Practiced:** Skill Composition, Sequential Chaining, Context Preservation
**Estimated Time:** 25 minutes
**Prerequisites:** Understanding of composition patterns and data flow

### Problem Statement

Design a skill chain for automating the creation of comprehensive learning materials. The chain should produce study notes, flashcards, and a quiz from a single topic input.

Your chain must specify:
1. **Skill sequence** with clear execution order
2. **Data contracts** between each skill (what passes through)
3. **Context preservation** strategy
4. **Failure handling** at each transition point

Starting input: A topic name (e.g., "Machine Learning Fundamentals")
Final outputs: Study notes file, flashcard set, quiz document

### Requirements

- [ ] Define 3+ skills in the chain with clear purposes
- [ ] Specify input/output contracts for each skill
- [ ] Show data flow between skills with specific fields
- [ ] Identify what context must be preserved across the chain
- [ ] Define failure modes and recovery strategies for each transition

### Hints (Progressive)

<details>
<summary>Hint 1: Getting Started</summary>

A natural chain might be:
```
Topic → [Generate Notes] → [Create Flashcards] → [Create Quiz]
```

But consider: Does the quiz only need flashcards, or also the original notes?
This affects your chain topology.

</details>

<details>
<summary>Hint 2: Key Insight</summary>

Context preservation is critical. The quiz skill likely needs:
- Original topic (for context)
- Study notes (for answer explanations)
- Flashcards (for critical knowledge flags)

This suggests the quiz skill has **multiple inputs** from earlier chain stages.

Data contracts should specify:
- Format (markdown, JSON, structured data)
- Required fields (sections, metadata)
- Location (file paths, in-memory)

</details>

<details>
<summary>Hint 3: Nearly There</summary>

For failure handling, consider:
- What if study notes generation fails? (Chain cannot continue)
- What if flashcards have no "critical knowledge"? (Quiz proceeds with reduced info)
- What if file write fails? (Retry vs. in-memory fallback)

Recovery strategies: Retry, fallback, partial completion, user notification.

</details>

### Solution

<details>
<summary>Click to reveal full solution</summary>

**Approach:**
Design a branching chain where study notes feed multiple downstream skills, with the quiz receiving consolidated context.

**1. Chain Architecture:**

```
                    ┌─────────────────────┐
                    │   study-notes-      │
                    │     creator         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   concept-map       │ (optional, parallel)
                    └──────────┬──────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ▼                     │                     ▼
┌─────────────────┐            │          ┌─────────────────┐
│   flashcards    │            │          │ practice-       │
│                 │            │          │   problems      │
└────────┬────────┘            │          └────────┬────────┘
         │                     │                   │
         │         ┌───────────┘                   │
         │         │                               │
         └─────────┴───────────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │       quiz          │
                    └─────────────────────┘
```

**2. Skill Definitions and Data Contracts:**

| Skill | Purpose | Input Contract | Output Contract |
|-------|---------|----------------|-----------------|
| **study-notes-creator** | Generate comprehensive notes from topic | `{topic: string, depth: enum}` | `{notes_path: string, sections: string[], key_concepts: array, metadata: object}` |
| **concept-map** | Visualize concept relationships | `{source: notes_path, concepts: array}` | `{map_path: string, centrality_scores: object, relationships: array}` |
| **flashcards** | Create spaced-repetition cards | `{source: notes_path, concepts: array, centrality: object}` | `{cards_path: string, cards: array, critical_flags: string[]}` |
| **practice-problems** | Generate hands-on exercises | `{source: notes_path, sections: array}` | `{problems_path: string, problems: array, common_mistakes: array}` |
| **quiz** | Create assessment instrument | `{notes: path, flashcards: path, problems: path, critical_flags: array}` | `{quiz_path: string, questions: array, answer_key: object}` |

**3. Detailed Data Flow:**

```
STAGE 1: study-notes-creator
────────────────────────────
Input:  {topic: "Machine Learning Fundamentals", depth: "comprehensive"}
Output: {
  notes_path: "notes/ml/ml-fundamentals.md",
  sections: ["Introduction", "Supervised Learning", "Unsupervised Learning", ...],
  key_concepts: ["gradient descent", "overfitting", "cross-validation", ...],
  metadata: {word_count: 5000, difficulty: "intermediate", date: "2026-01-07"}
}

STAGE 2a: flashcards (parallel with 2b)
────────────────────────────
Input:  {
  source: "notes/ml/ml-fundamentals.md",
  concepts: ["gradient descent", "overfitting", ...],
  centrality: {from concept-map if available}
}
Output: {
  cards_path: "notes/ml/flashcards/ml-fundamentals-flashcards.md",
  cards: [{front: "...", back: "...", difficulty: "easy"}, ...],
  critical_flags: ["gradient descent", "bias-variance tradeoff"]
}

STAGE 2b: practice-problems (parallel with 2a)
────────────────────────────
Input:  {
  source: "notes/ml/ml-fundamentals.md",
  sections: ["Supervised Learning", "Model Evaluation", ...]
}
Output: {
  problems_path: "notes/ml/practice/ml-fundamentals-practice.md",
  problems: [{type: "skill-builder", concept: "...", ...}, ...],
  common_mistakes: ["confusing precision/recall", "forgetting to normalize"]
}

STAGE 3: quiz (depends on stages 1, 2a, 2b)
────────────────────────────
Input:  {
  notes: "notes/ml/ml-fundamentals.md",
  flashcards: "notes/ml/flashcards/ml-fundamentals-flashcards.md",
  problems: "notes/ml/practice/ml-fundamentals-practice.md",
  critical_flags: ["gradient descent", "bias-variance tradeoff"]
}
Output: {
  quiz_path: "notes/ml/quizzes/ml-fundamentals-quiz.md",
  questions: [{type: "multiple-choice", ...}, {type: "essay", ...}],
  answer_key: {q1: "B", q2: "...", ...}
}
```

**4. Context Preservation Strategy:**

| Context Element | Where Created | Where Needed | Preservation Method |
|-----------------|---------------|--------------|---------------------|
| `topic` | User input | All skills | Pass through metadata |
| `notes_path` | study-notes | All downstream | Store in chain context |
| `key_concepts` | study-notes | flashcards, quiz | Pass in output contract |
| `critical_flags` | flashcards | quiz | Explicit output field |
| `common_mistakes` | practice | quiz | Used for distractor design |

**Chain Context Object:**

```json
{
  "chain_id": "uuid",
  "initiated": "2026-01-07T10:30:00Z",
  "topic": "Machine Learning Fundamentals",
  "artifacts": {
    "notes": {"path": "...", "status": "complete"},
    "flashcards": {"path": "...", "status": "complete"},
    "practice": {"path": "...", "status": "complete"},
    "quiz": {"path": "...", "status": "pending"}
  },
  "propagated_data": {
    "key_concepts": [...],
    "critical_flags": [...],
    "common_mistakes": [...]
  }
}
```

**5. Failure Handling:**

| Transition | Failure Mode | Impact | Recovery Strategy |
|------------|--------------|--------|-------------------|
| Input → Notes | Topic too vague | Chain blocked | Prompt for clarification |
| Notes → Flashcards | Notes missing sections | Degraded flashcards | Generate from available sections; warn user |
| Notes → Practice | Notes too short | Insufficient problems | Generate fewer problems; note limitation |
| Flashcards → Quiz | No critical flags | Quiz lacks emphasis markers | Use concept frequency as proxy |
| Practice → Quiz | No common mistakes | Weaker distractors | Use generic misconceptions |
| Any → File write | Permission error | Output lost | Retry; fall back to in-memory; return to user |

**Recovery Decision Tree:**

```
Failure Detected
      │
      ├─► Is upstream artifact available?
      │         │
      │         ├─► YES: Proceed with degraded output
      │         │         └─► Log warning
      │         │
      │         └─► NO: Is this critical path?
      │                   │
      │                   ├─► YES: Halt chain, notify user
      │                   │
      │                   └─► NO: Skip skill, continue chain
      │
      └─► Return partial results with status report
```

</details>

### Common Mistakes

- ❌ **Mistake:** Linear chain when parallel execution is possible
  - **Why it happens:** Simpler to conceptualize
  - **How to avoid:** Identify independent skills; parallelize where no data dependency exists

- ❌ **Mistake:** Not preserving original source path
  - **Why it happens:** Assuming downstream skills only need immediate predecessor output
  - **How to avoid:** Include source references in all outputs for traceability

- ❌ **Mistake:** Binary failure handling (all-or-nothing)
  - **Why it happens:** Error handling as afterthought
  - **How to avoid:** Design graceful degradation; partial success is often valuable

### Extension Challenge

Modify your chain design to support **conditional branching**: if the topic is code-related, include a `code-examples` skill; if conceptual, include a `concept-map` skill. How does this affect your context preservation strategy?

---

---

## Problem 4: Complete Skill Design

**Type:** Challenge
**Concepts Practiced:** Full Skill Architecture, Theoretical Foundations, Quality Dimensions
**Estimated Time:** 30 minutes
**Prerequisites:** All Lesson 1 concepts

### Problem Statement

Design a complete agent skill for **meeting-summarizer**: a skill that processes meeting transcripts and generates structured summaries with action items, decisions made, and key discussion points.

Your design must include:

1. **Full skill specification** (invocation, inputs, outputs, execution)
2. **Theoretical justification** referencing Modular Programming, Cognitive Load, or Activity Theory
3. **Perspective choice** (Minimalist vs Holistic) with rationale
4. **Quality dimension analysis** (how your design achieves reusability, consistency, maintainability, composability)
5. **Integration points** for potential skill chains

### Requirements

- [ ] Complete invocation pattern (slash + natural language)
- [ ] Detailed input/output contracts
- [ ] Step-by-step execution procedure
- [ ] Explicit reference to at least one theoretical foundation
- [ ] Justified granularity decision (atomic vs complete)
- [ ] Analysis of all four quality dimensions
- [ ] Identified upstream and downstream skill opportunities

### Hints (Progressive)

<details>
<summary>Hint 1: Getting Started</summary>

Start with the core capability: What does this skill do at its essence?
- Input: Meeting transcript (text, audio reference, or structured data)
- Output: Structured summary with specific sections

Then expand: What makes a good meeting summary?
- Action items with assignees and deadlines
- Decisions made (and by whom)
- Key discussion topics
- Follow-up questions

</details>

<details>
<summary>Hint 2: Key Insight</summary>

Theoretical justification should connect design choices to principles:

- **Modular Programming:** If designing atomic—separate skills for extraction vs. formatting
- **Cognitive Load:** Sensible defaults for output format; don't require users to specify everything
- **Activity Theory:** The skill mediates between "raw transcript" and "actionable summary"—what user intentions does it serve?

For granularity: A holistic design handles the complete use case (transcript → final summary). A minimalist design might separate into: extract-action-items, extract-decisions, format-summary.

</details>

<details>
<summary>Hint 3: Nearly There</summary>

Quality dimensions checklist:

- **Reusability:** Can this skill work for different meeting types (standup, planning, retrospective)?
- **Consistency:** Does it always produce the same output structure?
- **Maintainability:** If summary format requirements change, how many places need updating?
- **Composability:** Can it chain with calendar-integration, task-creation, or email-distribution skills?

Integration points:
- Upstream: transcript-generator, audio-to-text
- Downstream: task-manager, email-sender, calendar-updater

</details>

### Solution

<details>
<summary>Click to reveal full solution</summary>

**Approach:**
Design complete skill specification, then analyze through theoretical and quality lenses.

**1. Skill Specification:**

```yaml
---
name: meeting-summarizer
description: Processes meeting transcripts to generate structured summaries with action items, decisions, and key points
version: 1.0.0

trigger:
  - keyword: summarize meeting
  - keyword: meeting summary
  - keyword: meeting notes
  - pattern: "(summarize|process|analyze) .* (meeting|transcript|discussion)"
  - pattern: "create .* (summary|notes) (for|from) .* meeting"

arguments:
  - name: source
    description: Meeting transcript (file path, URL, or inline text)
    required: true
  - name: meeting_type
    description: Type of meeting for optimized extraction
    required: false
    default: general
    options: [standup, planning, retrospective, decision, brainstorm, general]
  - name: format
    description: Output format
    required: false
    default: markdown
    options: [markdown, json, html, slack]
  - name: include_timestamps
    description: Include timestamp references from transcript
    required: false
    default: false

inputs:
  type: text
  source: file_path | url | inline_text
  formats_accepted: [txt, md, docx, vtt, srt]
  required_content: meeting transcript with speaker labels (preferred)

outputs:
  type: structured_document
  location: summaries/{date}-{meeting_title}-summary.{format}
  sections:
    - meeting_metadata (title, date, attendees, duration)
    - executive_summary (2-3 sentences)
    - key_discussion_points (bulleted list)
    - decisions_made (with decision-maker attribution)
    - action_items (with assignee, deadline, priority)
    - open_questions (unresolved topics)
    - next_steps (follow-up meetings, dependencies)

tools:
  - Read
  - Write
  - Glob (for finding related transcripts)

chain:
  upstream:
    - audio-transcriber
    - calendar-event-fetcher
  downstream:
    - task-creator
    - email-composer
    - calendar-scheduler
  metadata_pass:
    - meeting_date
    - attendees
    - action_items (for task-creator)
    - next_meeting_topics (for calendar-scheduler)

tags:
  - productivity
  - meetings
  - summarization
---

## Execution Procedure

### Phase 1: Input Processing
1. Accept transcript source (file, URL, or inline)
2. Detect format and parse appropriately
3. Identify speaker labels if present
4. Extract meeting metadata (date, title, attendees) from header or content

### Phase 2: Content Analysis
1. Segment transcript into topical sections
2. Identify decision language patterns ("we decided", "agreed to", "will go with")
3. Extract action items using task language ("will do", "action item", "by Friday")
4. Detect questions and whether they were resolved
5. Tag content by relevance/importance

### Phase 3: Synthesis
1. Generate executive summary from key themes
2. Compile decisions with attribution
3. Format action items with structured fields (owner, deadline, description)
4. List open questions and discussion points
5. Infer next steps from context

### Phase 4: Output Generation
1. Apply selected format template
2. Include metadata header
3. Add cross-references if part of meeting series
4. Write to specified location
5. Return structured data for downstream skills
```

**2. Theoretical Justification:**

| Theory | Application to Design |
|--------|----------------------|
| **Modular Programming** | Skill has clear boundaries: transcript in, structured summary out. Internal phases (parse → analyze → synthesize → format) are loosely coupled. Could be decomposed but value is in the complete workflow. |
| **Cognitive Load** | Defaults reduce decision burden: general meeting type, markdown format, no timestamps. Users only specify what differs from common case. Progressive complexity—basic use needs only the transcript. |
| **Activity Theory** | Skill mediates between raw communication artifact (transcript) and actionable knowledge (decisions, tasks). Serves user intention of "understanding what happened and what to do next" without requiring manual extraction. |

**3. Perspective Choice: Holistic (with Modular Internal Design)**

**Choice:** Holistic—this skill handles the complete use case from transcript to formatted summary.

**Rationale:**

| Factor | Holistic Advantage | Minimalist Trade-off |
|--------|-------------------|---------------------|
| **User Mental Model** | "Summarize meeting" is one action | Would require: extract → analyze → format pipeline |
| **Context Dependency** | Action items depend on decision context | Atomic skills lose semantic relationships |
| **Workflow Coherence** | Single invocation, single output | Multiple outputs require assembly |
| **Practical Use** | Users want summaries, not components | Few users need just action-item extraction |

**However:** Internal implementation follows modular principles (separate parsing, analysis, synthesis phases). This enables future decomposition if atomic skills become valuable.

**4. Quality Dimension Analysis:**

| Dimension | How Achieved | Design Evidence |
|-----------|--------------|-----------------|
| **Reusability** | Meeting type parameter adapts extraction heuristics | `meeting_type: [standup, planning, retrospective, ...]`—same skill serves varied meetings |
| **Consistency** | Fixed output schema regardless of input variation | `outputs.sections` explicitly defines structure; any transcript produces same-shaped summary |
| **Maintainability** | Single skill owns summarization logic | Format changes require updating one skill; templates centralized |
| **Composability** | Clean contracts enable chaining | `chain.downstream` with `metadata_pass` explicitly defines integration points |

**5. Integration Points:**

```
         ┌──────────────────┐
         │ audio-transcriber │ ─┐
         └──────────────────┘   │
                                │
         ┌──────────────────┐   │
         │ calendar-fetcher │ ──┼──► meeting-summarizer
         └──────────────────┘   │          │
                                │          │
         ┌──────────────────┐   │          ├──► task-creator (action_items)
         │ (manual upload)  │ ──┘          │
         └──────────────────┘              ├──► email-composer (summary, attendees)
                                           │
                                           ├──► calendar-scheduler (next_steps)
                                           │
                                           └──► meeting-archive (full record)
```

**Upstream Integration Contract:**
```json
{
  "from": "audio-transcriber",
  "provides": {
    "transcript_path": "string",
    "speakers_identified": "boolean",
    "duration_minutes": "number"
  }
}
```

**Downstream Integration Contract:**
```json
{
  "to": "task-creator",
  "provides": {
    "action_items": [
      {
        "description": "string",
        "assignee": "string",
        "deadline": "date",
        "priority": "enum",
        "source_meeting": "string"
      }
    ]
  }
}
```

</details>

### Common Mistakes

- ❌ **Mistake:** Not specifying meeting_type variations
  - **Why it happens:** Designing for generic case only
  - **How to avoid:** Consider how extraction differs by meeting purpose

- ❌ **Mistake:** Forgetting attribution for decisions and actions
  - **Why it happens:** Focusing on what was decided, not who
  - **How to avoid:** Accountability requires attribution; include in output schema

- ❌ **Mistake:** Theoretical justification as afterthought
  - **Why it happens:** Design first, justify later
  - **How to avoid:** Let theory guide design choices from the start

### Extension Challenge

Extend the meeting-summarizer to support a **series awareness** feature: when processing a recurring meeting, automatically reference decisions and action items from previous meetings in the series. How does this affect input contracts and execution procedures?

---

---

## Problem 5: Skill Chain Debugging

**Type:** Debug/Fix
**Concepts Practiced:** Skill Composition, Context Preservation, Error Diagnosis
**Estimated Time:** 20 minutes
**Prerequisites:** Understanding of skill chains and data flow

### Problem Statement

A developer has implemented a three-skill chain for generating documentation from code repositories. The chain is failing intermittently. Analyze the following skill definitions and chain execution log to:

1. **Identify the bugs** (at least 3 issues)
2. **Explain why each causes failure**
3. **Provide corrected specifications**

**Skill Definitions:**

```yaml
# Skill 1: code-analyzer
name: code-analyzer
inputs:
  type: directory
  required_fields: [path]
outputs:
  type: json
  fields: [functions, classes, imports]
  location: analysis/{repo_name}.json

# Skill 2: doc-generator
name: doc-generator
inputs:
  type: json
  required_fields: [functions, classes]
  source: analysis/*.json
outputs:
  type: markdown
  fields: [api_reference, examples]
  location: docs/{filename}.md

# Skill 3: doc-validator
name: doc-validator
inputs:
  type: markdown
  required_fields: [content]
  source: docs/api_reference.md
outputs:
  type: report
  fields: [valid, errors, warnings]
```

**Chain Execution Log:**

```
[10:00:01] Chain started: code-analyzer → doc-generator → doc-validator
[10:00:02] code-analyzer: Input received: {path: "/repo/src"}
[10:00:05] code-analyzer: Analysis complete
[10:00:05] code-analyzer: Output written to: analysis/my-repo.json
[10:00:06] doc-generator: Looking for input at: analysis/*.json
[10:00:06] doc-generator: Found 3 files: my-repo.json, old-analysis.json, test-data.json
[10:00:06] doc-generator: Processing: old-analysis.json (first match)
[10:00:08] doc-generator: Output written to: docs/old-analysis.md
[10:00:09] doc-validator: Looking for input at: docs/api_reference.md
[10:00:09] doc-validator: ERROR - File not found: docs/api_reference.md
[10:00:09] Chain failed at doc-validator
```

### Requirements

- [ ] Identify at least 3 distinct bugs in the skill chain
- [ ] Explain the failure mechanism for each bug
- [ ] Provide corrected YAML for affected skills
- [ ] Describe how proper chain metadata would prevent these issues

### Hints (Progressive)

<details>
<summary>Hint 1: Getting Started</summary>

Read the execution log carefully. Look for mismatches between:
- What one skill outputs
- What the next skill expects

The chain failed at doc-validator, but the root causes may be earlier.

</details>

<details>
<summary>Hint 2: Key Insight</summary>

Three distinct issues:

1. **Input source ambiguity:** doc-generator uses `analysis/*.json` but there are multiple files
2. **Output naming mismatch:** doc-generator outputs `{filename}.md` but validator expects `api_reference.md`
3. **Missing chain context:** No mechanism to pass the specific analysis file from skill 1 to skill 2

Trace the data flow: Where does "my-repo.json" get lost?

</details>

<details>
<summary>Hint 3: Nearly There</summary>

The corrected chain needs:
- Explicit file path passing (not wildcards)
- Consistent naming conventions
- Chain metadata that preserves context

Consider adding a `chain.metadata_pass` field to each skill that explicitly declares what data flows forward.

</details>

### Solution

<details>
<summary>Click to reveal full solution</summary>

**Approach:**
Trace execution log against skill definitions to identify mismatches, then correct contracts and add chain metadata.

**Bug Analysis:**

### Bug 1: Ambiguous Input Source (doc-generator)

**Location:** doc-generator inputs.source

**Definition:**
```yaml
source: analysis/*.json
```

**Problem:** Wildcard matches multiple files (my-repo.json, old-analysis.json, test-data.json). Skill arbitrarily selects first match (old-analysis.json) instead of the file actually produced by the preceding skill.

**Failure Mechanism:** Glob pattern `*.json` is evaluated independently, with no awareness of chain context. File system ordering determines selection, not chain data flow.

**Evidence:** Log shows "Processing: old-analysis.json (first match)" despite code-analyzer producing "my-repo.json".

**Corrected Specification:**
```yaml
inputs:
  type: json
  required_fields: [functions, classes]
  source: ${chain.previous_output.path}  # Dynamic reference
```

---

### Bug 2: Output Naming Mismatch (doc-generator → doc-validator)

**Location:** doc-generator outputs.location vs doc-validator inputs.source

**Definition:**
```yaml
# doc-generator
outputs:
  location: docs/{filename}.md  # Produces: docs/old-analysis.md

# doc-validator
inputs:
  source: docs/api_reference.md  # Expects: docs/api_reference.md
```

**Problem:** doc-generator uses dynamic naming (`{filename}.md`), but doc-validator expects a hardcoded filename (`api_reference.md`). Names never match.

**Failure Mechanism:** Even if Bug 1 were fixed and correct file processed, output would be `docs/my-repo.md`, still not matching `api_reference.md`.

**Evidence:** Log shows "Output written to: docs/old-analysis.md" but validator looks for "docs/api_reference.md".

**Corrected Specification:**
```yaml
# Option A: Standardize output name
# doc-generator
outputs:
  location: docs/api_reference.md  # Fixed name

# Option B: Dynamic input reference
# doc-validator
inputs:
  source: ${chain.previous_output.path}  # Dynamic reference
```

---

### Bug 3: Missing Chain Context Propagation

**Location:** All skills—no chain metadata defined

**Problem:** No mechanism exists to pass context (file paths, identifiers) from one skill to the next. Each skill operates in isolation, relying on file system state.

**Failure Mechanism:** Skills cannot reference their predecessor's actual output. They use patterns or hardcoded paths that may not match runtime values.

**Evidence:** None of the skills define `chain.metadata_pass` or reference `${chain.previous_output}`.

**Corrected Specification (all skills):**

```yaml
# Skill 1: code-analyzer
name: code-analyzer
inputs:
  type: directory
  required_fields: [path]
outputs:
  type: json
  fields: [functions, classes, imports]
  location: analysis/${input.repo_name}.json
chain:
  position: 1
  downstream: [doc-generator]
  metadata_pass:
    - output_path: analysis/${input.repo_name}.json
    - repo_name: ${input.repo_name}

# Skill 2: doc-generator
name: doc-generator
inputs:
  type: json
  required_fields: [functions, classes]
  source: ${chain.metadata.output_path}  # From previous skill
outputs:
  type: markdown
  fields: [api_reference, examples]
  location: docs/${chain.metadata.repo_name}-api.md
chain:
  position: 2
  upstream: [code-analyzer]
  downstream: [doc-validator]
  metadata_pass:
    - output_path: docs/${chain.metadata.repo_name}-api.md
    - repo_name: ${chain.metadata.repo_name}

# Skill 3: doc-validator
name: doc-validator
inputs:
  type: markdown
  required_fields: [content]
  source: ${chain.metadata.output_path}  # From previous skill
outputs:
  type: report
  fields: [valid, errors, warnings]
  location: reports/${chain.metadata.repo_name}-validation.json
chain:
  position: 3
  upstream: [doc-generator]
  metadata_pass:
    - validation_result
```

---

**Complete Corrected Chain Execution:**

```
[10:00:01] Chain started with context: {repo_name: "my-repo", path: "/repo/src"}
[10:00:02] code-analyzer: Input received: {path: "/repo/src"}
[10:00:05] code-analyzer: Analysis complete
[10:00:05] code-analyzer: Output written to: analysis/my-repo.json
[10:00:05] code-analyzer: Propagating metadata: {output_path: "analysis/my-repo.json", repo_name: "my-repo"}
[10:00:06] doc-generator: Input source from chain metadata: analysis/my-repo.json
[10:00:08] doc-generator: Output written to: docs/my-repo-api.md
[10:00:08] doc-generator: Propagating metadata: {output_path: "docs/my-repo-api.md", repo_name: "my-repo"}
[10:00:09] doc-validator: Input source from chain metadata: docs/my-repo-api.md
[10:00:10] doc-validator: Validation complete: {valid: true, errors: [], warnings: []}
[10:00:10] Chain completed successfully
```

---

**Prevention Principles:**

| Principle | Application |
|-----------|-------------|
| **Explicit over implicit** | Use `${chain.metadata}` references, not file patterns |
| **Consistent naming** | Derive names from shared context variables |
| **Metadata propagation** | Each skill declares what it passes forward |
| **Position awareness** | Skills know their chain position and neighbors |

</details>

### Common Mistakes

- ❌ **Mistake:** Using wildcards for inter-skill file references
  - **Why it happens:** Works in isolation testing
  - **How to avoid:** Always use explicit path references from chain context

- ❌ **Mistake:** Hardcoding expected filenames across skill boundaries
  - **Why it happens:** Simplifies individual skill design
  - **How to avoid:** Use variables derived from shared chain context

- ❌ **Mistake:** Not testing chains with "dirty" file system state
  - **Why it happens:** Clean test environments
  - **How to avoid:** Test with existing files that could match patterns

### Extension Challenge

Design a chain monitoring system that could detect these issues before chain failure. What metadata would you log at each transition? What invariants would you check?

---

---

## Summary

### Key Takeaways
1. **Skill components** are identifiable in any capability description—practice extracting them systematically
2. **Invocation patterns** should serve all user types—design for power users, casual users, and automation
3. **Skill chains** require explicit data contracts and context preservation—never rely on implicit file system state
4. **Complete skill design** benefits from theoretical grounding—let principles guide decisions
5. **Chain debugging** requires tracing data flow across boundaries—most failures occur at transitions

### Concepts by Problem
| Problem | Primary Concepts | Secondary Concepts |
|---------|-----------------|-------------------|
| P1 (Warm-Up) | Skill Components | Error Handling |
| P2 (Skill-Builder) | Invocation Patterns | User Personas |
| P3 (Skill-Builder) | Skill Composition | Context Preservation |
| P4 (Challenge) | Complete Skill Design | Theoretical Foundations |
| P5 (Debug/Fix) | Chain Debugging | Data Contracts |

### Next Steps
- If struggled with P1: Re-review Core Concepts in Lesson 1
- If struggled with P2: Study invocation pattern trade-offs
- If struggled with P3: Practice drawing chain diagrams with explicit contracts
- If struggled with P4: Review theoretical framework section; connect theory to practice
- If struggled with P5: Implement a simple chain and intentionally break it; debug
- Ready for assessment: Proceed to quiz

---

*Generated from Lesson 1: Creating Agent Skills | Practice Problems Skill*
