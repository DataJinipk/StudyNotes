# CLAUDE.md

## Project Overview
This workspace is dedicated to creating professional-level academic study materials and educational content.

## Communication Style

Always use an academic tone in all responses. This includes:

- Formal language with precise terminology
- Clear, structured explanations with logical progression
- Citations and references to relevant concepts when applicable
- Objective and analytical approach to problem-solving
- Avoidance of colloquialisms, slang, or overly casual expressions
- Use of discipline-appropriate vocabulary and conventions
- Well-organized responses with clear thesis statements and supporting arguments

## Available Skills

### Study Notes Creator (`/study-notes-creator`)
Professional skill for generating comprehensive academic-level study notes. Invoke with:
```
/study-notes-creator [topic]
```

Features:
- Structured notes following academic conventions
- Learning objectives aligned with Bloom's Taxonomy
- Theoretical frameworks and practical applications
- Critical analysis sections
- Review questions and key terminology tables
- Further reading recommendations

### Concept Map (`/concept-map`)
Professional skill for generating visual concept maps showing relationships between key ideas. Invoke with:
```
/concept-map [study-notes-file-or-topic]
```

Features:
- Extracts concepts and classifies relationship types (is-a, enables, contrasts, requires)
- Generates Mermaid diagram code for visual rendering
- Creates hierarchical text representation of concept structure
- Produces relationship matrix with connection strength analysis
- Calculates concept centrality metrics (High/Medium/Low connectivity)
- Suggests learning pathways (foundational, goal-oriented, comparative)
- Identifies critical path for minimum viable understanding

### Flashcards (`/flashcards`)
Professional skill for transforming study notes into structured flashcards optimized for spaced repetition. Invoke with:
```
/flashcards [study-notes-file-or-topic]
```

Features:
- Analyzes study notes for key concepts and terminology
- Generates 5 tiered flashcards per set:
  - 2 Easy (definition-level, Remember/Understand)
  - 2 Medium (application-level, Apply/Analyze)
  - 1 Hard (synthesis-level, Evaluate/Synthesize)
- Front (question) and back (answer) format
- Critical knowledge flagging for concepts appearing in multiple cards
- Export formats compatible with Anki, CSV, and plain text
- Spaced repetition optimization

### Practice Problems (`/practice-problems`)
Professional skill for generating hands-on practice problems with scaffolded difficulty. Invoke with:
```
/practice-problems [study-notes-file-or-topic]
```

Features:
- Generates 5 scaffolded practice problems:
  - 1 Warm-Up (direct concept application)
  - 2 Skill-Builder (multi-step procedural tasks)
  - 1 Challenge (complex synthesis scenario)
  - 1 Debug/Fix (identify and correct errors)
- Progressive hint system (3 levels per problem)
- Fully worked solutions with conceptual connections
- Common mistakes section with prevention strategies
- Extension challenges for deeper mastery
- Self-assessment guide for mastery evaluation

### Quiz (`/quiz`)
Professional skill for generating comprehensive assessment quizzes from study notes. Invoke with:
```
/quiz [study-notes-file-or-topic]
```

Features:
- Extracts key concepts from study notes for assessment
- Generates 5 quiz questions:
  - 2 Multiple choice (conceptual, Remember/Understand)
  - 2 Short answer (application, Apply/Analyze)
  - 1 Essay (synthesis, Evaluate/Synthesize)
- Complete answer key with detailed explanations
- Understanding gap indicators for incorrect answers
- Performance-based review recommendations linked to specific study note sections
- Mastery level interpretation guide

## Workflow Guidelines

1. **Research Phase:** Gather authoritative sources and verify information accuracy
2. **Organization Phase:** Structure content hierarchically with clear sections
3. **Development Phase:** Elaborate concepts with appropriate depth
4. **Review Phase:** Ensure academic rigor and consistency
5. **Output Phase:** Save notes in well-formatted markdown files

## File Organization

Store generated notes in organized directories:
```
StudyNotes/
├── .claude/skills/     # Skill definitions
├── notes/              # Generated study notes
│   ├── [subject]/      # Subject-specific folders
│   │   ├── concept-maps/ # Generated concept maps
│   │   ├── flashcards/   # Generated flashcard sets
│   │   ├── practice/     # Generated practice problems
│   │   └── quizzes/      # Generated assessment quizzes
│   └── ...
└── resources/          # Reference materials
```

## Skill Chain Architecture

The skills form an integrated learning content pipeline:

```
                         ┌───────────────────┐
                         │  study-notes-     │
                         │    creator        │
                         └─────────┬─────────┘
                                   │
            ┌──────────────────────┼──────────────────────┐
            │                      │                      │
            ▼                      ▼                      ▼
   ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
   │ concept-map │        │ flashcards  │        │  practice-  │
   │             │        │             │        │  problems   │
   └──────┬──────┘        └──────┬──────┘        └──────┬──────┘
          │                      │                      │
          │    ┌─────────────────┼──────────────────────┘
          │    │                 │
          └────┼─────────────────┘
               │
               ▼
        ┌─────────────┐
        │    quiz     │
        └─────────────┘
```

**Chain Relationships:**
- `study-notes-creator` → feeds all downstream skills
- `concept-map` → runs parallel with `flashcards` and `practice-problems`; centrality informs difficulty
- `flashcards` → feeds `quiz` with critical knowledge flags
- `practice-problems` → feeds `quiz` with common mistakes for distractor design; builds procedural fluency
- `quiz` → terminal skill; references all upstream sources for review recommendations

**Skill Purposes:**
| Skill | Learning Goal | Output Type |
|-------|---------------|-------------|
| study-notes-creator | Conceptual understanding | Reference material |
| concept-map | Relationship visualization | Visual + pathways |
| flashcards | Memorization & recall | Spaced repetition cards |
| practice-problems | Procedural fluency | Hands-on exercises |
| quiz | Assessment & diagnosis | Evaluation + feedback |
