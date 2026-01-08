# Flashcard Set: Lesson 12 - AI Agents, Autonomous Systems, and Spec-Driven Development

**Source:** Lessons/Lesson_12.md
**Date Generated:** 2026-01-08
**Total Cards:** 5
**Distribution:** 2 Easy | 2 Medium | 1 Hard

---

## Critical Knowledge Summary

Concepts appearing in multiple cards (prioritize for review):
- **ReAct Pattern:** Appears in Cards 1, 3, 5
- **Spec-Driven Development (SDD):** Appears in Cards 2, 4, 5
- **Tool Use:** Appears in Cards 1, 3
- **Agent Loop:** Appears in Cards 1, 5

---

## Flashcards

---

### Card 1 | Easy
**Cognitive Level:** Remember
**Concept:** ReAct Pattern
**Source Section:** Concept 2 - Reasoning Patterns and Prompting Strategies

**FRONT (Question):**
What is the ReAct pattern in AI agents, and what are its three key components in the execution cycle?

**BACK (Answer):**
ReAct (Reasoning + Acting) is a prompting pattern that combines explicit reasoning traces with tool-based actions. Its three key components are:

1. **Thought:** The agent's reasoning about what to do next
2. **Action:** A tool call or operation to execute (e.g., `search("query")`)
3. **Observation:** The result returned from the action

The cycle repeats (Thought → Action → Observation) until the agent has enough information to produce a **Final Answer**. ReAct grounds reasoning in real-world observations, making it more reliable than pure Chain-of-Thought for tasks requiring external information.

**Critical Knowledge Flag:** Yes

---

### Card 2 | Easy
**Cognitive Level:** Understand
**Concept:** Spec-Kit Framework Phases
**Source Section:** Concept 5 - Specification-Driven Development (SDD)

**FRONT (Question):**
List the five phases of the Spec-Kit SDD framework in order, and briefly describe what each phase accomplishes.

**BACK (Answer):**
The Spec-Kit framework defines five sequential phases:

1. **Constitution** (`/sp.constitution`): Establish governing principles—coding standards, technology constraints, quality requirements, and security policies

2. **Specification** (`/sp.specify`): Define requirements focusing on outcomes—user stories, functional/non-functional requirements, API contracts

3. **Planning** (`/sp.plan`): Create technical implementation strategy—architecture decisions, technology stack, component breakdown

4. **Tasks** (`/sp.tasks`): Generate actionable implementation tasks—ordered list with dependencies, completion criteria, complexity estimates

5. **Implementation** (`/sp.implement`): Execute all tasks according to plan—code generation, test execution, iterative refinement

**Quality commands** like `/sp.clarify`, `/sp.analyze`, and `/sp.checklist` support validation throughout.

**Critical Knowledge Flag:** Yes

---

### Card 3 | Medium
**Cognitive Level:** Apply
**Concept:** Tool Use and Safety
**Source Section:** Concepts 3, 9 - Tool Use and Function Calling; Safety, Control, and Alignment

**FRONT (Question):**
You are building an AI agent that can execute shell commands and modify files. Design the safety validation flow that should occur before any tool execution. What checks should be performed, and what approval levels would you implement?

**BACK (Answer):**
**Safety Validation Flow:**

```
Action Request → Blocklist Check → Scope Validation → Resource Limits → Approval Level → Execution/Reject
```

**Checks to Perform:**
1. **Blocklist Check:** Reject dangerous patterns (e.g., `rm -rf /`, `curl | bash`, `chmod 777`)
2. **Scope Validation:** Ensure file operations stay within allowed workspace; network calls only to approved hosts
3. **Resource Limits:** Enforce timeout limits, memory caps, rate limits
4. **Schema Validation:** Verify tool parameters match expected types/formats

**Approval Levels:**
| Level | Actions | Behavior |
|-------|---------|----------|
| **Autonomous** | Read operations, safe computations | Execute immediately |
| **Notify** | File modifications in workspace | Inform human, continue |
| **Confirm** | External API calls, deletions | Wait for human approval |
| **Blocked** | System modifications, credentials | Always reject |

All actions should be logged to an **audit trail** for review.

**Critical Knowledge Flag:** Yes

---

### Card 4 | Medium
**Cognitive Level:** Analyze
**Concept:** SDD vs Traditional Development
**Source Section:** Concept 5 - Specification-Driven Development (SDD)

**FRONT (Question):**
Compare Specification-Driven Development (SDD) with traditional software development. What are the key differences in artifact ownership, developer role, and iteration cycle? What are the advantages and risks of each approach?

**BACK (Answer):**
| Aspect | Traditional Development | Spec-Driven Development |
|--------|------------------------|-------------------------|
| **Primary Artifact** | Code | Specification |
| **Developer Role** | Write code, interpret requirements | Write specifications, validate output |
| **Iteration Cycle** | Write → Test → Debug manually | Specify → Generate → Validate → Refine spec |
| **Code Ownership** | Developers own and maintain | Code is generated, spec is maintained |
| **Reproducibility** | Varies by developer | Deterministic from spec |

**SDD Advantages:**
- Faster initial implementation
- Consistent code quality
- Specifications serve as documentation
- Easy to regenerate with updates

**SDD Risks:**
- Agent may misinterpret ambiguous specs
- Generated code may not match organizational patterns
- Debugging generated code can be challenging
- Over-reliance may reduce developer skills

**Best Practice:** Use SDD for well-defined tasks with clear specifications; maintain human review for critical systems.

**Critical Knowledge Flag:** Yes

---

### Card 5 | Hard
**Cognitive Level:** Evaluate/Synthesize
**Concept:** Complete Agent Architecture for SDD
**Source Section:** Concepts 1, 2, 3, 4, 5, 9 (Synthesis)

**FRONT (Question):**
Design a complete AI agent architecture for implementing Specification-Driven Development. Your design should include: (1) the agent loop components, (2) required reasoning patterns, (3) essential tools, (4) memory systems, and (5) safety mechanisms. Explain how these components work together through one iteration of implementing a feature from specification.

**BACK (Answer):**
**Complete SDD Agent Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│                    SDD AGENT ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────┤
│  PERCEPTION: Specification parser, codebase indexer          │
│  REASONING: LLM with ReAct + CoT                            │
│  PLANNING: Task decomposition, dependency ordering           │
│  ACTION: Tool executor with safety layer                     │
│  MEMORY: Working (context) + Episodic (past solutions)      │
└─────────────────────────────────────────────────────────────┘
```

**Components:**

**(1) Agent Loop:** Observe (read spec/code) → Reason (analyze requirements) → Plan (create tasks) → Act (execute tools) → Learn (store outcomes)

**(2) Reasoning Patterns:**
- **Chain-of-Thought:** Break down complex requirements
- **ReAct:** Ground reasoning in file reads and test results
- **Self-Debugging:** Fix errors through iterative refinement

**(3) Essential Tools:**
- File operations: `read_file`, `write_file`, `edit_file`
- Search: `grep`, `glob`, `find_symbol`
- Execution: `run_tests`, `run_linter`, `bash`
- Analysis: `get_diagnostics` (LSP)

**(4) Memory Systems:**
- **Working Memory:** Current task, active files, recent errors
- **Episodic Memory:** Past implementations for similar specs (vector retrieval)
- **Semantic Memory:** Codebase index, API documentation

**(5) Safety Mechanisms:**
- Action validation (blocklist, scope constraints)
- Human approval for destructive operations
- Sandboxed code execution
- Comprehensive audit logging

**One Feature Implementation Cycle:**

1. **Parse Specification:** Extract requirements for "Add user authentication"
2. **Retrieve Context:** Query episodic memory for similar past auth implementations
3. **Plan Tasks:** Generate ordered task list (models → middleware → endpoints → tests)
4. **For each task:**
   - **Reason (CoT):** "I need to create a User model with email and hashed password"
   - **Act (Tool):** `write_file("models/user.py", code)`
   - **Observe:** File created successfully
   - **Verify:** `run_tests()` → Observation: 2 tests pass
5. **Self-Debug:** If tests fail, analyze errors, generate fix, retry
6. **Complete:** All tasks done, tests pass, commit changes
7. **Store:** Save successful approach to episodic memory for future retrieval

**Critical Knowledge Flag:** Yes

---

## Export Formats

### Anki-Compatible (Tab-Separated)

```
What is the ReAct pattern in AI agents, and what are its three key components?	ReAct (Reasoning + Acting) combines reasoning with tool actions. Components: (1) Thought - reasoning, (2) Action - tool call, (3) Observation - result. Cycle repeats until Final Answer.	agents::reasoning::easy
List the five phases of the Spec-Kit SDD framework in order.	1. Constitution (principles), 2. Specification (requirements), 3. Planning (architecture), 4. Tasks (actionable items), 5. Implementation (code generation). Quality commands: /sp.clarify, /sp.analyze, /sp.checklist.	agents::sdd::easy
Design safety validation flow for an agent executing shell commands and file modifications.	Flow: Request → Blocklist Check → Scope Validation → Resource Limits → Approval Level → Execute/Reject. Levels: Autonomous (reads), Notify (workspace writes), Confirm (external/deletes), Blocked (system mods). All actions logged.	agents::safety::medium
Compare SDD with traditional development: artifacts, roles, iteration, advantages, risks.	Traditional: code artifact, dev writes code, manual iteration. SDD: spec artifact, dev writes specs, generate-validate cycle. SDD advantages: faster, consistent, documented. Risks: misinterpretation, debugging generated code, skill atrophy.	agents::sdd::medium
Design complete agent architecture for SDD with loop, reasoning, tools, memory, safety.	Loop: Observe→Reason→Plan→Act→Learn. Reasoning: CoT+ReAct+Self-Debug. Tools: file ops, search, execution, analysis. Memory: working (context), episodic (past solutions), semantic (codebase). Safety: validation, HITL, sandbox, audit.	agents::architecture::hard
```

### CSV Format

```csv
"Front","Back","Difficulty","Concept"
"What is the ReAct pattern in AI agents, and what are its three key components?","ReAct combines reasoning with tool actions via Thought→Action→Observation cycle until Final Answer.","Easy","ReAct Pattern"
"List the five phases of the Spec-Kit SDD framework in order.","1. Constitution, 2. Specification, 3. Planning, 4. Tasks, 5. Implementation. Plus quality commands.","Easy","Spec-Kit Framework"
"Design safety validation flow for agent executing shell commands.","Blocklist→Scope→Limits→Approval Level check. Four levels: Autonomous, Notify, Confirm, Blocked.","Medium","Tool Safety"
"Compare SDD with traditional development.","SDD: spec as artifact, generate-validate cycle. Traditional: code as artifact, manual iteration. Trade-offs in speed vs control.","Medium","SDD Comparison"
"Design complete agent architecture for SDD.","Integrate agent loop, CoT/ReAct reasoning, file/search/exec tools, working/episodic/semantic memory, and safety layer with HITL.","Hard","Agent Architecture"
```

### Plain Text Review

```
Q: What is the ReAct pattern in AI agents, and what are its three key components?
A: ReAct (Reasoning + Acting) combines reasoning traces with tool actions. Three components: (1) Thought - agent's reasoning, (2) Action - tool execution, (3) Observation - result from action. Cycle repeats until Final Answer.

---

Q: List the five phases of the Spec-Kit SDD framework in order.
A: 1. Constitution (/sp.constitution) - establish guidelines
   2. Specification (/sp.specify) - define requirements
   3. Planning (/sp.plan) - technical strategy
   4. Tasks (/sp.tasks) - actionable items
   5. Implementation (/sp.implement) - execute and verify

---

Q: Design safety validation flow for an agent executing shell commands and file modifications.
A: Flow: Request → Blocklist Check → Scope Validation → Resource Limits → Approval Level → Execute/Reject
   Levels: Autonomous (reads), Notify (workspace writes), Confirm (external APIs, deletions), Blocked (system modifications)
   Always log to audit trail.

---

Q: Compare SDD with traditional development.
A: Traditional: Code is primary artifact, developers write code, manual iteration
   SDD: Specification is primary artifact, developers write specs, generate-validate cycle
   SDD advantages: faster, consistent, documented. Risks: misinterpretation, debugging challenges.

---

Q: Design complete agent architecture for SDD.
A: Agent Loop: Observe→Reason→Plan→Act→Learn
   Reasoning: CoT + ReAct + Self-Debugging
   Tools: File ops, search, execution, LSP diagnostics
   Memory: Working (context), Episodic (past solutions), Semantic (codebase)
   Safety: Validation, human-in-the-loop, sandboxing, audit logging
```

---

## Source Mapping

| Card | Source Section | Key Terminology Used |
|------|----------------|---------------------|
| 1 | Concept 2: Reasoning Patterns | ReAct, Thought, Action, Observation, Chain-of-Thought |
| 2 | Concept 5: SDD | Spec-Kit, Constitution, Specification, Planning, Tasks, Implementation |
| 3 | Concepts 3, 9: Tool Use, Safety | Blocklist, scope validation, approval levels, HITL, audit |
| 4 | Concept 5: SDD | SDD philosophy, traditional vs SDD, artifact ownership |
| 5 | Concepts 1-5, 9: Synthesis | Agent loop, reasoning, tools, memory, safety integration |

---

## Study Recommendations

### Review Schedule (Spaced Repetition)
- **Day 1:** All 5 cards (initial learning)
- **Day 3:** Cards marked incorrect + Hard card
- **Day 7:** All 5 cards (consolidation)
- **Day 14:** Random selection of 3 cards
- **Day 30:** All 5 cards (long-term retention test)

### Prerequisite Check
Before studying these cards, ensure familiarity with:
- Large Language Models (Lesson 3)
- Prompt Engineering basics (Lesson 2)
- Basic software development concepts

### Extension Topics
After mastering these cards, explore:
- Implement ReAct pattern from scratch
- Use Spec-Kit CLI on a real project
- Build safety layer for coding agent
- Evaluate agent on SWE-bench subset

---

*Generated from Lesson 12: AI Agents, Autonomous Systems, and Spec-Driven Development | Flashcards Skill*
