# Assessment Quiz: AI Agents, Autonomous Systems, and Spec-Driven Development

**Source Material:** Lesson 12 - AI Agents, Autonomous Systems, and Spec-Driven Development
**Original Source Path:** C:\agentic_ai\StudyNotes\Lessons\Lesson_12.md
**Flashcard Reference:** C:\agentic_ai\StudyNotes\notes\ai-agents-sdd-lesson\flashcards\lesson-12-ai-agents-sdd-flashcards.md
**Date Generated:** 2026-01-08
**Total Questions:** 5
**Estimated Completion Time:** 20-30 minutes
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
**Concept Tested:** ReAct Pattern Structure
**Source Section:** Concept 2 - Reasoning Patterns and Prompting Strategies

The ReAct (Reasoning + Acting) pattern is a fundamental prompting strategy for AI agents. Which sequence correctly represents one complete iteration of the ReAct cycle?

A) Action → Observation → Thought → Final Answer

B) Thought → Action → Observation → (repeat or Final Answer)

C) Plan → Execute → Evaluate → Learn

D) Perceive → Reason → Act → Memory Update

---

### Question 2 | Multiple Choice
**Cognitive Level:** Remember/Understand
**Concept Tested:** Specification-Driven Development Phases
**Source Section:** Concept 5 - Specification-Driven Development (SDD)

In the Spec-Kit SDD framework, a developer wants to establish coding standards, security policies, and technology constraints before defining specific feature requirements. Which phase should they complete first?

A) Specification (`/sp.specify`) - to define the outcomes before constraints

B) Planning (`/sp.plan`) - to establish the technical architecture

C) Constitution (`/sp.constitution`) - to establish governing principles

D) Tasks (`/sp.tasks`) - to break down work into actionable items

---

### Question 3 | Short Answer
**Cognitive Level:** Apply/Analyze
**Concept Tested:** Memory Systems Architecture
**Source Section:** Concept 4 - Memory Systems for Agents
**Expected Response Length:** 2-4 sentences

An AI coding agent is working on a complex refactoring task that spans multiple sessions. The user mentions they prefer functional programming patterns, and the agent previously helped implement a similar refactoring three weeks ago.

**Question:** Identify which memory types (working, episodic, semantic) would store each of these pieces of information, and explain how they would be retrieved during the current task.

---

### Question 4 | Short Answer
**Cognitive Level:** Apply/Analyze
**Concept Tested:** Safety Mechanisms and Human-in-the-Loop
**Source Section:** Concept 9 - Safety, Control, and Alignment
**Expected Response Length:** 2-4 sentences

A coding agent receives the following task: "Clean up the project by deleting all files in the `/tmp/build` directory and then deploy the application to the production server."

**Question:** Analyze this request using the four approval levels (Autonomous, Notify, Confirm, Blocked). Assign an appropriate approval level to each action and justify your classification based on the safety principles discussed in the lesson.

---

### Question 5 | Essay
**Cognitive Level:** Evaluate/Synthesize
**Concepts Tested:** Agent Architecture, SDD Methodology, Multi-Agent Systems, Evaluation
**Source Sections:** Concepts 1, 5, 7, 8
**Expected Response Length:** 1-2 paragraphs

A software company is considering adopting Specification-Driven Development (SDD) with AI agents for their development workflow. They are debating between two approaches:

**Approach A:** A single autonomous coding agent that handles the entire SDD pipeline from specification to implementation.

**Approach B:** A multi-agent system with specialized agents (Planner, Coder, Reviewer, Tester) coordinated by a Manager agent.

**Question:** Evaluate both approaches considering: (1) the trade-offs in architecture complexity vs. capability, (2) how each approach would handle error recovery and quality assurance, and (3) what evaluation metrics would be most appropriate for measuring success in each case. Recommend which approach would be more suitable for a team transitioning from traditional development, and justify your recommendation.

**Evaluation Criteria:**
- [ ] Addresses all three components of the question
- [ ] Demonstrates integration of agent architecture and SDD concepts
- [ ] Provides evidence-based reasoning from lesson content
- [ ] Shows critical evaluation of trade-offs for both approaches

---

## Answer Key

### Question 1 | Multiple Choice
**Correct Answer:** B

**Explanation:**
The ReAct pattern explicitly combines reasoning traces (Thoughts) with actions (tool calls) in an interleaved manner. The correct sequence is:

1. **Thought:** The agent reasons about what to do next
2. **Action:** The agent executes a tool call (e.g., `search("query")`)
3. **Observation:** The result from the action is received
4. The cycle repeats until the agent determines it has sufficient information
5. **Final Answer:** The agent provides its response

This is directly illustrated in Concept 2 with the example:
```
Thought: I need to find the current weather in Tokyo...
Action: search("Tokyo current weather")
Observation: Tokyo weather: 18°C, partly cloudy...
```

**Why Other Options Are Incorrect:**
- **A)** Reverses the order - Thought must precede Action, as reasoning determines which action to take
- **C)** Describes a general learning loop (Plan-Execute-Evaluate-Learn), not the specific ReAct prompting pattern
- **D)** Describes the general Agent Execution Loop from Concept 1, which is a higher-level abstraction than the ReAct prompting strategy

**Understanding Gap Indicator:**
If answered incorrectly, this may indicate:
- Confusion between the Agent Execution Loop (architectural) and ReAct (prompting pattern)
- Misunderstanding of the role of explicit reasoning traces in agent behavior

---

### Question 2 | Multiple Choice
**Correct Answer:** C

**Explanation:**
The Spec-Kit framework follows a deliberate five-phase sequence where Constitution comes first because it establishes the foundational guidelines that govern all subsequent phases. According to Concept 5:

> "Phase 1: CONSTITUTION (`/sp.constitution`) - Establish governing principles and development guidelines including coding standards and conventions, technology constraints, quality requirements, and security policies."

This phase creates the "rules of the game" before any specific features are defined, ensuring consistency across all specifications and implementations.

**Why Other Options Are Incorrect:**
- **A) Specification:** Comes after Constitution; defining outcomes without established constraints may lead to specifications that violate standards
- **B) Planning:** Phase 3; technical architecture decisions should align with constitutional principles
- **D) Tasks:** Phase 4; breaking down work requires first having specifications to break down

**Understanding Gap Indicator:**
If answered incorrectly, this may indicate:
- Misunderstanding of the dependency relationship between SDD phases
- Confusion about why constraints must precede specifications in systematic development

---

### Question 3 | Short Answer
**Model Answer:**

The three pieces of information would be stored as follows:

1. **Current refactoring task context** → **Working Memory**: The active files, current progress, and immediate context are maintained in working memory (the LLM's context window) for direct access during execution.

2. **User preference for functional programming** → **Semantic Memory**: User preferences and learned knowledge are stored in semantic memory, which could be implemented as a vector database or knowledge graph. This would be retrieved via query-based lookup when making coding decisions.

3. **Previous similar refactoring from three weeks ago** → **Episodic Memory**: Past interaction episodes are stored with timestamps and embeddings. This would be retrieved via similarity search (embedding the current task description and finding semantically similar past episodes) combined with recency weighting.

During the current task, the agent would: (a) load working memory with the current codebase context, (b) retrieve the functional programming preference from semantic memory when generating code, and (c) use hybrid retrieval (relevance + recency) on episodic memory to find the previous refactoring as a reference pattern.

**Key Components Required:**
- [ ] Correctly identifies all three memory types
- [ ] Explains appropriate storage for each information type
- [ ] Describes retrieval mechanisms (similarity search, query-based, direct access)

**Partial Credit Guidance:**
- **Full credit:** All three memory types correctly identified with appropriate retrieval methods
- **Partial credit:** Two memory types correct, or all three identified but retrieval mechanisms unclear
- **No credit:** Confusion between memory types or inability to distinguish their purposes

**Understanding Gap Indicator:**
If answered incompletely or incorrectly, this may indicate:
- Difficulty distinguishing between episodic (event-based) and semantic (fact-based) memory
- Unclear understanding of how vector embeddings enable similarity-based retrieval

---

### Question 4 | Short Answer
**Model Answer:**

**Action 1: Delete files in `/tmp/build`**
- **Approval Level: Confirm**
- **Justification:** File deletion is a destructive, potentially irreversible operation. While `/tmp/build` is a temporary directory (lower risk than source code), deletions should still require human confirmation because the agent could misunderstand which files are "build" files or the directory path could be misinterpreted. The lesson states that "File deletion" falls under the Confirm level where the system should "Wait for approval."

**Action 2: Deploy to production server**
- **Approval Level: Confirm** (or arguably **Blocked** depending on policy)
- **Justification:** Production deployments are high-stakes, external operations that affect live systems and users. These should never be autonomous because errors can have significant business impact. The lesson categorizes "Production deployments" explicitly under the Confirm level. Some organizations may classify this as Blocked entirely, requiring deployment through separate CI/CD systems with additional safeguards.

The agent should pause and request explicit human approval before executing either action, presenting the specific files to be deleted and the deployment target for verification.

**Key Components Required:**
- [ ] Correctly assigns Confirm (or Blocked) levels to both actions
- [ ] Justifies based on reversibility and impact criteria
- [ ] References the safety principles from the lesson

**Partial Credit Guidance:**
- **Full credit:** Both actions correctly classified with clear justification referencing lesson concepts
- **Partial credit:** Correct levels but weak justification, or one action misclassified
- **No credit:** Assigns Autonomous or Notify to destructive operations

**Understanding Gap Indicator:**
If answered incompletely or incorrectly, this may indicate:
- Underestimating the risks of autonomous destructive actions
- Misunderstanding the criteria for escalating approval levels

---

### Question 5 | Essay
**Model Answer:**

**Approach Evaluation:**

**(1) Architecture Complexity vs. Capability Trade-offs:**

**Approach A (Single Agent)** offers simplicity in deployment and coordination—no inter-agent communication protocols, no message-passing overhead, and a unified context for decision-making. However, a single agent must handle diverse tasks (planning, coding, reviewing, testing), which can overwhelm its context window and lead to capability dilution. The lesson notes that agents have "scope bounded by context window" as a limitation.

**Approach B (Multi-Agent System)** provides specialization advantages: the Coder agent can be optimized for code generation, the Reviewer for static analysis patterns, and the Tester for test case synthesis. This follows the hierarchical "Manager-Worker" architecture described in Concept 7, where the Manager coordinates and workers execute specialized tasks. However, this introduces complexity in orchestration, potential communication failures, and the need for robust message protocols.

**(2) Error Recovery and Quality Assurance:**

**Approach A** relies on self-debugging loops within the single agent—if tests fail, the same agent must analyze failures and generate fixes. This works for simple errors but may struggle with systematic issues that require fresh perspective, as the agent lacks external feedback mechanisms.

**Approach B** enables the **Debate Pattern** for quality assurance: the Coder produces code, the Reviewer critiques it, and iteration continues until approval. This separation provides natural checkpoints and multiple perspectives on errors. The lesson describes how "debate/adversarial" architectures improve reasoning by having a Proposer defend against a Critic, with a Judge making final decisions.

**(3) Evaluation Metrics:**

For **Approach A**, metrics should focus on end-to-end task completion: **SWE-bench resolution rate** (can it solve real issues?), **steps to completion** (efficiency), and **self-recovery rate** (how often does it fix its own errors?).

For **Approach B**, additional metrics are needed for agent coordination: **handoff success rate** (do transitions between agents preserve context?), **inter-agent latency** (coordination overhead), and per-agent **specialization accuracy** (does each agent excel in its domain?). The lesson's Agent Evaluation Framework suggests measuring both capability and efficiency dimensions.

**Recommendation:**

For a team transitioning from traditional development, **Approach B (Multi-Agent)** is more suitable despite higher initial complexity. The multi-agent structure mirrors familiar software team dynamics—developers write code, reviewers provide feedback, QA tests—making the mental model more intuitive for the team. The separation of concerns also provides clearer audit trails (which agent made which decision) and more natural human-in-the-loop integration points: humans can review code at the same stage they would review a developer's pull request. The lesson emphasizes that "Human oversight" should be maintained, and multi-agent systems provide more natural breakpoints for human intervention. The team can incrementally increase automation by gradually reducing human review at specific agent handoff points as trust is established.

**Evaluation Rubric:**

| Criterion | Excellent (4) | Proficient (3) | Developing (2) | Beginning (1) |
|-----------|---------------|----------------|----------------|---------------|
| Concept Integration | Synthesizes 4+ concepts from multiple lesson sections | Connects 3 concepts accurately | References 2 concepts but limited integration | Single concept or inaccurate references |
| Critical Analysis | Evaluates trade-offs with nuanced consideration of context | Identifies key trade-offs with supporting reasoning | Lists trade-offs but limited analysis | Describes only advantages or only disadvantages |
| Evidence & Reasoning | Every claim supported by lesson content | Most claims supported by content | Some claims unsupported | Minimal connection to lesson |
| Communication | Clear structure, precise terminology, logical flow | Organized with minor clarity issues | Understandable but disorganized | Unclear or incomplete response |

**Understanding Gap Indicator:**
If response lacks depth, this may indicate:
- Difficulty synthesizing across multiple lesson concepts
- Limited understanding of multi-agent coordination patterns
- Unclear distinction between evaluation metrics for different architectures

---

## Diagnostic Feedback & Review Recommendations

### Performance-Based Review Guide

| If Incorrect On... | Gap Identified | Review Section | Priority |
|--------------------|----------------|----------------|----------|
| Question 1 | ReAct pattern structure | Concept 2: Reasoning Patterns (ReAct subsection) | High |
| Question 2 | SDD phase ordering | Concept 5: Spec-Kit Framework (Five-Phase Methodology) | High |
| Question 3 | Memory system types and retrieval | Concept 4: Memory Systems for Agents | Medium |
| Question 4 | Safety approval levels | Concept 9: Human-in-the-Loop Patterns | Medium |
| Question 5 | Architecture synthesis and evaluation | Concepts 1, 5, 7, 8 (integration required) | Low |

### Targeted Review Recommendations

#### For Multiple Choice Errors (Questions 1-2)
**Indicates:** Foundational concept gaps in agent reasoning patterns or SDD methodology
**Action:** Review definitions and core structures in:
- Concept 2: Focus on the ReAct Implementation Pattern code example (lines 168-203)
- Concept 5: Study the Spec-Kit SDD Phases diagram (lines 650-709)
**Focus On:** Distinguishing between architectural concepts (Agent Loop) and prompting strategies (ReAct); understanding the sequential dependency of SDD phases

#### For Short Answer Errors (Questions 3-4)
**Indicates:** Application or analysis difficulties with memory systems or safety mechanisms
**Action:** Practice applying concepts through:
- Concept 4: Trace through the Memory-Augmented Agent Loop code (lines 601-624)
- Concept 9: Study the Human-in-the-Loop code pattern (lines 1653-1695)
**Focus On:** Mapping specific information types to appropriate memory stores; classifying actions by their risk profile and reversibility

#### For Essay Weakness (Question 5)
**Indicates:** Integration or synthesis challenges across multiple concepts
**Action:** Review interconnections between:
- Concept 1 (Agent Architectures) and Concept 7 (Multi-Agent Systems)
- Concept 5 (SDD) and Concept 8 (Evaluation and Benchmarking)
**Focus On:** Understanding how individual concepts combine into complete systems; developing frameworks for evaluating architectural trade-offs

### Mastery Indicators

| Score | Interpretation | Recommended Action |
|-------|---------------|-------------------|
| **5/5 Correct** | Strong mastery demonstrated | Proceed to implementation practice; consider building a simple agent |
| **4/5 Correct** | Good understanding | Review the specific gap area; attempt related practice problems |
| **3/5 Correct** | Moderate understanding | Systematic review of all concepts recommended; use flashcard set for reinforcement |
| **2/5 or below** | Foundational gaps | Comprehensive re-study of Lesson 12 advised; ensure prerequisites (Lessons 2, 3) are solid |

### Cross-Reference to Related Materials

| If You Struggled With... | Also Review... |
|--------------------------|----------------|
| ReAct pattern | Flashcard 1 (Critical Knowledge) |
| SDD phases | Flashcard 2 (Critical Knowledge) |
| Safety mechanisms | Flashcard 3 (Application context) |
| Agent architecture synthesis | Flashcard 5 (Complete architecture design) |
| Multi-agent patterns | Practice Problem 4 (Challenge problem on coordination) |

---

## Self-Assessment Checklist

Before considering this quiz complete, verify:

- [ ] I can explain the ReAct pattern without referring to notes
- [ ] I can list all five SDD phases in order and describe each
- [ ] I can classify a given action into the appropriate approval level
- [ ] I can design a memory system for a specific agent use case
- [ ] I can evaluate trade-offs between single-agent and multi-agent approaches
- [ ] I understand how to measure agent performance using appropriate benchmarks

---

*Generated from Lesson 12: AI Agents, Autonomous Systems, and Spec-Driven Development | Quiz Skill*
