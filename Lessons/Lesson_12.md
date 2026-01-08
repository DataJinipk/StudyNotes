# Lesson 12: AI Agents, Autonomous Systems, and Spec-Driven Development

**Topic:** AI Agents, Autonomous Systems, and Spec-Driven Development: From Reactive Tools to Autonomous Problem Solvers
**Prerequisites:** Lesson 2 (Prompt Engineering), Lesson 3 (Large Language Models), Lesson 11 (MLOps)
**Estimated Study Time:** 4-5 hours
**Difficulty:** Advanced

---

## Learning Objectives

Upon completion of this lesson, learners will be able to:

1. **Analyze** agent architectures and explain how reasoning, planning, and action components interact
2. **Design** tool-use patterns and function-calling interfaces for LLM-based agents
3. **Implement** memory systems that enable agents to maintain context across long-horizon tasks
4. **Apply** Specification-Driven Development (SDD) principles to autonomous software creation
5. **Evaluate** agent performance using appropriate benchmarks and safety considerations

---

## Introduction

The evolution from static language models to autonomous AI agents represents a fundamental shift in how artificial intelligence systems interact with the world. While traditional LLMs generate text responses to prompts, AI agents perceive their environment, reason about goals, plan sequences of actions, execute those actions through tools, and learn from outcomes—all with minimal human intervention.

This transformation is driven by three converging capabilities: advanced reasoning through techniques like Chain-of-Thought and ReAct, sophisticated tool use enabling interaction with external systems, and memory architectures that maintain coherent behavior across extended tasks. Together, these enable **agentic AI**—systems that don't merely respond but actively pursue objectives.

A particularly significant application of agentic AI is **Specification-Driven Development (SDD)**, where autonomous agents interpret high-level requirements and produce working software. This paradigm shift moves developers from writing code to writing specifications, with AI agents handling implementation details. Understanding agent architectures, their capabilities, and their limitations is essential for effectively leveraging this new development paradigm.

This lesson covers the complete landscape of AI agents: from foundational architectures and reasoning patterns through practical tool use and memory systems, culminating in specification-driven development and multi-agent collaboration.

---

## Core Concepts

### Concept 1: Agent Fundamentals and Architectures

An AI agent is a system that perceives its environment, reasons about goals, and takes actions to achieve those goals autonomously.

**The Agent Loop:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Agent Execution Loop                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│    ┌──────────────┐                                                 │
│    │  Environment │◀────────────────────────────────┐               │
│    └──────┬───────┘                                 │               │
│           │ Observation                             │               │
│           ▼                                         │               │
│    ┌──────────────┐                                 │               │
│    │   Perceive   │                                 │               │
│    └──────┬───────┘                                 │               │
│           │                                         │               │
│           ▼                                         │               │
│    ┌──────────────┐     ┌──────────────┐           │               │
│    │    Reason    │────▶│     Plan     │           │               │
│    └──────────────┘     └──────┬───────┘           │               │
│           ▲                    │                    │               │
│           │                    ▼                    │               │
│    ┌──────────────┐     ┌──────────────┐           │               │
│    │    Memory    │◀───▶│     Act      │───────────┘               │
│    └──────────────┘     └──────────────┘                           │
│                               │                                     │
│                               ▼                                     │
│                        ┌──────────────┐                            │
│                        │    Tools     │                            │
│                        └──────────────┘                            │
└─────────────────────────────────────────────────────────────────────┘
```

**Agent vs. Traditional LLM:**

| Aspect | Traditional LLM | AI Agent |
|--------|-----------------|----------|
| Interaction | Single turn | Multi-turn with environment |
| State | Stateless | Maintains memory and context |
| Actions | Text generation only | Tool use, API calls, code execution |
| Goals | Respond to prompt | Achieve specified objectives |
| Autonomy | Human-in-the-loop | Autonomous decision-making |
| Scope | Bounded by context window | Extended through memory and tools |

**Core Agent Components:**

```
Agent Architecture Components:

1. Perception Layer
   ├── Input processing (text, images, structured data)
   ├── Environment state extraction
   └── Context aggregation from multiple sources

2. Reasoning Engine (typically LLM)
   ├── Goal interpretation
   ├── Situation analysis
   ├── Decision making
   └── Plan generation

3. Memory System
   ├── Working memory (current context)
   ├── Episodic memory (past interactions)
   └── Semantic memory (learned knowledge)

4. Action Layer
   ├── Tool selection
   ├── Parameter generation
   ├── Execution management
   └── Result interpretation

5. Learning/Adaptation
   ├── Feedback incorporation
   ├── Strategy refinement
   └── Error correction
```

**Agent Taxonomies:**

| Type | Description | Example |
|------|-------------|---------|
| **Reactive** | Direct stimulus-response, no internal state | Simple chatbots |
| **Deliberative** | Maintains world model, plans before acting | Planning agents |
| **Hybrid** | Combines reactive and deliberative | Most modern agents |
| **Learning** | Improves performance through experience | RL-based agents |
| **Multi-Agent** | Multiple agents collaborating/competing | Agent swarms |

---

### Concept 2: Reasoning Patterns and Prompting Strategies

Effective agents require structured reasoning approaches that go beyond simple prompting.

**Chain-of-Thought (CoT) Reasoning:**

```
Standard Prompting:
  Q: "What is 23 × 17?"
  A: "391"

Chain-of-Thought:
  Q: "What is 23 × 17?"
  A: "Let me break this down:
      23 × 17 = 23 × (10 + 7)
             = 23 × 10 + 23 × 7
             = 230 + 161
             = 391"
```

**ReAct (Reasoning + Acting) Pattern:**

```
ReAct combines reasoning traces with actions:

Thought: I need to find the current weather in Tokyo to answer this question.
Action: search("Tokyo current weather")
Observation: Tokyo weather: 18°C, partly cloudy, humidity 65%

Thought: Now I have the weather data. The user asked if they need an umbrella.
         18°C with partly cloudy skies suggests no immediate rain.
Action: respond("Based on current conditions (18°C, partly cloudy),
                 you likely don't need an umbrella right now, but
                 consider checking the forecast for later.")
```

**ReAct Implementation Pattern:**

```python
REACT_PROMPT = """
You are an agent that solves problems by thinking step-by-step and using tools.

Available tools:
{tool_descriptions}

Use this format:
Thought: [Your reasoning about what to do next]
Action: [tool_name(parameters)]
Observation: [Result from the tool - this will be provided]
... (repeat Thought/Action/Observation as needed)
Thought: I now have enough information to answer.
Final Answer: [Your response to the user]

Question: {user_question}
"""

def react_loop(question, tools, max_iterations=10):
    context = REACT_PROMPT.format(
        tool_descriptions=format_tools(tools),
        user_question=question
    )

    for i in range(max_iterations):
        response = llm.generate(context)

        if "Final Answer:" in response:
            return extract_final_answer(response)

        action = parse_action(response)
        observation = execute_tool(action, tools)

        context += f"\n{response}\nObservation: {observation}\n"

    return "Max iterations reached without conclusion"
```

**Tree of Thoughts (ToT):**

```
Tree of Thoughts explores multiple reasoning paths:

                    [Problem]
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
      [Path A]      [Path B]      [Path C]
      Score: 0.7    Score: 0.9    Score: 0.4
          │             │
          │        ┌────┴────┐
          │        ▼         ▼
          │    [Path B1]  [Path B2]
          │    Score: 0.85 Score: 0.95
          │                  │
          └──────────────────┴──────▶ [Solution]

Key Operations:
1. Generate: Propose multiple next steps
2. Evaluate: Score each path's promise
3. Search: BFS/DFS through promising paths
4. Backtrack: Abandon low-scoring branches
```

**Reasoning Pattern Comparison:**

| Pattern | Strengths | Weaknesses | Best For |
|---------|-----------|------------|----------|
| **CoT** | Simple, effective | Linear, no backtracking | Math, logical reasoning |
| **ReAct** | Grounds reasoning in observations | Sequential, one path | Tool-using tasks |
| **ToT** | Explores alternatives | Expensive, complex | Creative problem-solving |
| **Reflexion** | Self-correction | Requires good self-evaluation | Iterative improvement |

**Self-Consistency for Robust Reasoning:**

```python
def self_consistency(question, num_samples=5):
    """
    Generate multiple reasoning paths and take majority vote
    """
    answers = []
    for _ in range(num_samples):
        response = llm.generate(
            cot_prompt.format(question=question),
            temperature=0.7  # Enable diversity
        )
        answer = extract_answer(response)
        answers.append(answer)

    # Majority voting
    return Counter(answers).most_common(1)[0][0]
```

---

### Concept 3: Tool Use and Function Calling

Tools extend agent capabilities beyond text generation to interact with external systems.

**Tool Architecture:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Tool Use Pipeline                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  User Query ──▶ LLM ──▶ Tool Selection ──▶ Parameter Extraction     │
│                              │                     │                 │
│                              ▼                     ▼                 │
│                        ┌─────────────────────────────────┐          │
│                        │         Tool Registry           │          │
│                        │  ┌─────┐ ┌─────┐ ┌─────┐       │          │
│                        │  │Search│ │ Code │ │ API │ ...  │          │
│                        │  └─────┘ └─────┘ └─────┘       │          │
│                        └─────────────────────────────────┘          │
│                                       │                              │
│                                       ▼                              │
│                              Tool Execution                          │
│                                       │                              │
│                                       ▼                              │
│                              Result Processing                       │
│                                       │                              │
│                                       ▼                              │
│                        LLM Incorporates Result ──▶ Response          │
└─────────────────────────────────────────────────────────────────────┘
```

**Function Calling Schema (OpenAI-style):**

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Execute Python code and return the result",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    }
                },
                "required": ["code"]
            }
        }
    }
]

# LLM response with tool call
response = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "search_web",
                "arguments": '{"query": "latest AI research 2024"}'
            }
        }
    ]
}
```

**Tool Categories:**

| Category | Examples | Use Cases |
|----------|----------|-----------|
| **Information Retrieval** | Web search, RAG, database queries | Fact-finding, research |
| **Code Execution** | Python interpreter, shell, sandboxes | Computation, automation |
| **External APIs** | Weather, stocks, calendars | Real-time data access |
| **File Operations** | Read, write, edit files | Document processing |
| **Communication** | Email, Slack, notifications | User interaction |
| **Specialized** | Image generation, calculators | Domain-specific tasks |

**Robust Tool Execution:**

```python
class ToolExecutor:
    def __init__(self, tools: dict, timeout: int = 30):
        self.tools = tools
        self.timeout = timeout

    def execute(self, tool_name: str, arguments: dict) -> ToolResult:
        if tool_name not in self.tools:
            return ToolResult(
                success=False,
                error=f"Unknown tool: {tool_name}"
            )

        tool = self.tools[tool_name]

        # Validate arguments against schema
        validation_error = self._validate_args(tool.schema, arguments)
        if validation_error:
            return ToolResult(success=False, error=validation_error)

        try:
            # Execute with timeout
            result = self._execute_with_timeout(
                tool.function,
                arguments,
                self.timeout
            )
            return ToolResult(success=True, data=result)

        except TimeoutError:
            return ToolResult(
                success=False,
                error=f"Tool execution timed out after {self.timeout}s"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Tool execution failed: {str(e)}"
            )

    def _execute_with_timeout(self, func, args, timeout):
        # Implementation with proper timeout handling
        with ThreadPoolExecutor() as executor:
            future = executor.submit(func, **args)
            return future.result(timeout=timeout)
```

**Tool Selection Strategies:**

```
1. Description-Based Selection
   - LLM reads tool descriptions and selects appropriate tool
   - Works well for distinct, well-described tools

2. Few-Shot Examples
   - Provide examples of when to use each tool
   - Improves accuracy for similar queries

3. Hierarchical Selection
   - First select category, then specific tool
   - Scales better with many tools

4. Retrieval-Augmented Tool Selection
   - Embed tool descriptions
   - Retrieve relevant tools based on query similarity
   - Present only relevant subset to LLM
```

---

### Concept 4: Memory Systems for Agents

Memory enables agents to maintain coherence across extended interactions and learn from experience.

**Memory Architecture:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Agent Memory Systems                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Working Memory                            │   │
│  │  Current context, active goals, recent observations         │   │
│  │  ┌─────────────────────────────────────────────────────┐   │   │
│  │  │ Context Window: Last N tokens of conversation       │   │   │
│  │  └─────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│              ┌───────────────┼───────────────┐                      │
│              ▼               ▼               ▼                      │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐          │
│  │   Episodic    │  │   Semantic    │  │  Procedural   │          │
│  │    Memory     │  │    Memory     │  │    Memory     │          │
│  ├───────────────┤  ├───────────────┤  ├───────────────┤          │
│  │ Past events   │  │ Facts and     │  │ How to do     │          │
│  │ and episodes  │  │ knowledge     │  │ things        │          │
│  │               │  │               │  │               │          │
│  │ "Last week I  │  │ "Python is a  │  │ "To search:   │          │
│  │  helped user  │  │  programming  │  │  1. Parse     │          │
│  │  debug React" │  │  language"    │  │  2. Query     │          │
│  └───────────────┘  └───────────────┘  └───────────────┘          │
│         │                   │                   │                   │
│         └───────────────────┼───────────────────┘                   │
│                             ▼                                       │
│                    ┌───────────────┐                               │
│                    │ Vector Store  │                               │
│                    │  (Long-term)  │                               │
│                    └───────────────┘                               │
└─────────────────────────────────────────────────────────────────────┘
```

**Memory Types and Implementation:**

| Memory Type | Content | Storage | Retrieval |
|-------------|---------|---------|-----------|
| **Working** | Current task context | In-context (prompt) | Direct access |
| **Episodic** | Past interactions | Vector DB | Similarity search |
| **Semantic** | Facts, knowledge | Vector DB / KG | Query-based |
| **Procedural** | Skills, procedures | Code / Prompts | Pattern matching |

**Episodic Memory Implementation:**

```python
from datetime import datetime
import numpy as np

class EpisodicMemory:
    def __init__(self, embedding_model, vector_store):
        self.embedder = embedding_model
        self.store = vector_store

    def store_episode(self, content: str, metadata: dict = None):
        """Store an interaction episode"""
        episode = {
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "embedding": self.embedder.embed(content),
            "metadata": metadata or {}
        }
        self.store.add(episode)

    def retrieve_relevant(self, query: str, k: int = 5) -> list:
        """Retrieve k most relevant episodes"""
        query_embedding = self.embedder.embed(query)
        results = self.store.similarity_search(
            query_embedding,
            k=k
        )
        return results

    def retrieve_recent(self, n: int = 10) -> list:
        """Retrieve n most recent episodes"""
        return self.store.get_recent(n)

    def retrieve_hybrid(self, query: str, k: int = 5) -> list:
        """Combine relevance and recency"""
        relevant = self.retrieve_relevant(query, k * 2)
        recent = self.retrieve_recent(k * 2)

        # Score combining relevance and recency
        scored = []
        for episode in set(relevant + recent):
            relevance_score = episode.similarity_score
            recency_score = self._recency_weight(episode.timestamp)
            combined = 0.7 * relevance_score + 0.3 * recency_score
            scored.append((episode, combined))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [ep for ep, _ in scored[:k]]
```

**Conversation Summarization for Memory:**

```python
SUMMARIZE_PROMPT = """
Summarize the following conversation, preserving:
1. Key decisions made
2. Important information shared
3. Unresolved questions or tasks
4. User preferences expressed

Conversation:
{conversation}

Summary:
"""

class ConversationMemory:
    def __init__(self, llm, max_tokens=4000):
        self.llm = llm
        self.max_tokens = max_tokens
        self.messages = []
        self.summaries = []

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self._maybe_summarize()

    def _maybe_summarize(self):
        """Summarize old messages if context is getting too long"""
        total_tokens = sum(len(m["content"]) // 4 for m in self.messages)

        if total_tokens > self.max_tokens:
            # Summarize oldest half of messages
            split_point = len(self.messages) // 2
            to_summarize = self.messages[:split_point]

            summary = self.llm.generate(
                SUMMARIZE_PROMPT.format(
                    conversation=format_messages(to_summarize)
                )
            )

            self.summaries.append(summary)
            self.messages = self.messages[split_point:]

    def get_context(self) -> str:
        """Get full context including summaries"""
        context_parts = []

        if self.summaries:
            context_parts.append("Previous conversation summary:")
            context_parts.extend(self.summaries)
            context_parts.append("\nRecent messages:")

        context_parts.append(format_messages(self.messages))
        return "\n".join(context_parts)
```

**Memory-Augmented Agent Loop:**

```python
def agent_loop_with_memory(query, agent, memory):
    # Retrieve relevant past context
    relevant_episodes = memory.retrieve_relevant(query, k=3)

    # Build augmented context
    context = f"""
    Relevant past interactions:
    {format_episodes(relevant_episodes)}

    Current query: {query}
    """

    # Run agent
    response = agent.run(context)

    # Store this interaction
    memory.store_episode(
        content=f"User: {query}\nAssistant: {response}",
        metadata={"type": "interaction", "query": query}
    )

    return response
```

---

### Concept 5: Specification-Driven Development (SDD)

SDD is a paradigm where AI agents autonomously implement software from high-level specifications. Unlike traditional development where specifications guide human developers, SDD makes specifications **executable**—directly generating working implementations rather than just guiding them.

**Core SDD Philosophy:**

```
Traditional Development:
  Specification → Human interprets → Human codes → Human tests

Specification-Driven Development:
  Specification → Agent interprets → Agent codes → Agent tests → Human validates

Key Insight: Specifications become the primary artifact, not the code.
Code becomes a generated output, reproducible from the specification.
```

**The Spec-Kit Framework (Five-Phase Methodology):**

The spec-kit-plus framework (github.com/panaversity/spec-kit-plus) defines a structured five-phase approach:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Spec-Kit SDD Phases                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Phase 1: CONSTITUTION (/sp.constitution)                           │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Establish governing principles and development guidelines     │  │
│  │ • Coding standards and conventions                           │  │
│  │ • Technology constraints                                     │  │
│  │ • Quality requirements                                       │  │
│  │ • Security policies                                          │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│                              ▼                                       │
│  Phase 2: SPECIFICATION (/sp.specify)                               │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Define requirements focusing on outcomes, not implementation  │  │
│  │ • User stories with acceptance criteria                      │  │
│  │ • Functional requirements                                    │  │
│  │ • Non-functional requirements                                │  │
│  │ • API contracts and data models                              │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│                              ▼                                       │
│  Phase 3: PLANNING (/sp.plan)                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Create technical implementation strategy                      │  │
│  │ • Architecture decisions                                     │  │
│  │ • Technology stack selection                                 │  │
│  │ • Component breakdown                                        │  │
│  │ • Dependency mapping                                         │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│                              ▼                                       │
│  Phase 4: TASK BREAKDOWN (/sp.tasks)                                │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Generate actionable implementation tasks                      │  │
│  │ • Ordered task list with dependencies                        │  │
│  │ • Clear completion criteria per task                         │  │
│  │ • Estimated complexity                                       │  │
│  │ • Test requirements per task                                 │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│                              ▼                                       │
│  Phase 5: IMPLEMENTATION (/sp.implement)                            │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Execute all tasks according to plan                          │  │
│  │ • Code generation per task                                   │  │
│  │ • Test execution and validation                              │  │
│  │ • Iterative refinement                                       │  │
│  │ • Documentation generation                                   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  Quality Commands (Optional):                                        │
│  • /sp.clarify  - Resolve ambiguities before planning               │
│  • /sp.analyze  - Verify consistency after task generation          │
│  • /sp.checklist - Generate validation checklist                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**SDD Workflow (Detailed):**

```
┌─────────────────────────────────────────────────────────────────────┐
│              Specification-Driven Development Pipeline               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐                                                   │
│  │Specification │  "Build a REST API for user management            │
│  │  (Natural    │   with CRUD operations, JWT authentication,       │
│  │   Language)  │   and PostgreSQL storage"                         │
│  └──────┬───────┘                                                   │
│         │                                                            │
│         ▼                                                            │
│  ┌──────────────┐                                                   │
│  │ Requirements │  - User model with email, password, name          │
│  │  Extraction  │  - Endpoints: POST /users, GET /users/{id}, ...   │
│  │              │  - JWT-based auth middleware                       │
│  └──────┬───────┘  - PostgreSQL with SQLAlchemy ORM                 │
│         │                                                            │
│         ▼                                                            │
│  ┌──────────────┐                                                   │
│  │  Planning &  │  1. Set up project structure                      │
│  │   Design     │  2. Define database models                        │
│  │              │  3. Implement auth middleware                     │
│  └──────┬───────┘  4. Create CRUD endpoints                         │
│         │          5. Add tests                                      │
│         ▼                                                            │
│  ┌──────────────┐                                                   │
│  │Implementation│  [Agent writes code file by file]                 │
│  │    Loop      │  [Runs tests, fixes errors]                       │
│  │              │  [Iterates until spec is met]                     │
│  └──────┬───────┘                                                   │
│         │                                                            │
│         ▼                                                            │
│  ┌──────────────┐                                                   │
│  │ Verification │  - All tests pass                                 │
│  │  & Delivery  │  - Spec requirements checklist complete           │
│  └──────────────┘  - Code review / human approval                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Key SDD Principles:**

```
1. Specification as Contract
   - Spec defines WHAT, not HOW
   - Clear acceptance criteria
   - Testable requirements

2. Iterative Refinement
   - Agent proposes → Human reviews → Agent refines
   - Progressive elaboration of requirements
   - Continuous validation against spec

3. Verification-First
   - Generate tests from specifications
   - Implementation guided by test requirements
   - Spec compliance as success metric

4. Human-in-the-Loop
   - Human provides specs and feedback
   - Agent handles implementation
   - Human approves final output
```

**Specification Structure:**

```markdown
# Feature Specification: User Authentication System

## Overview
Implement a secure user authentication system with email/password login,
JWT tokens, and role-based access control.

## Functional Requirements
### FR-1: User Registration
- Accept email, password, and display name
- Validate email format and uniqueness
- Hash password with bcrypt (cost factor 12)
- Return user object (excluding password)

### FR-2: User Login
- Accept email and password
- Verify credentials
- Return JWT access token (1 hour expiry)
- Return JWT refresh token (7 day expiry)

### FR-3: Token Refresh
- Accept valid refresh token
- Return new access token
- Invalidate used refresh token (rotation)

## Non-Functional Requirements
### NFR-1: Security
- Passwords never stored in plaintext
- Tokens signed with RS256
- Rate limiting: 5 failed logins per 15 minutes

### NFR-2: Performance
- Login response < 200ms p95
- Support 100 concurrent authentications

## API Contract
```yaml
paths:
  /auth/register:
    post:
      requestBody:
        content:
          application/json:
            schema:
              type: object
              required: [email, password, name]
              properties:
                email: {type: string, format: email}
                password: {type: string, minLength: 8}
                name: {type: string}
      responses:
        201: {description: User created}
        400: {description: Validation error}
        409: {description: Email already exists}
```

## Acceptance Criteria
- [ ] All API endpoints match contract
- [ ] Unit test coverage > 80%
- [ ] Integration tests for auth flows
- [ ] Security scan passes (no critical issues)
```

**SDD Agent Implementation:**

```python
class SDDAgent:
    def __init__(self, llm, tools, workspace):
        self.llm = llm
        self.tools = tools
        self.workspace = workspace

    def implement_spec(self, specification: str) -> ImplementationResult:
        # Phase 1: Analyze specification
        requirements = self._extract_requirements(specification)

        # Phase 2: Create implementation plan
        plan = self._create_plan(requirements)

        # Phase 3: Generate tests from spec
        tests = self._generate_tests(requirements)
        self._write_tests(tests)

        # Phase 4: Implement iteratively
        for step in plan.steps:
            self._implement_step(step)

            # Verify progress
            test_results = self._run_tests()
            if test_results.failures:
                self._fix_failures(test_results)

        # Phase 5: Final verification
        return self._verify_against_spec(specification)

    def _extract_requirements(self, spec: str) -> Requirements:
        prompt = f"""
        Analyze this specification and extract:
        1. Functional requirements (numbered list)
        2. Non-functional requirements
        3. API contracts
        4. Acceptance criteria

        Specification:
        {spec}
        """
        return self.llm.generate_structured(prompt, Requirements)

    def _create_plan(self, requirements: Requirements) -> Plan:
        prompt = f"""
        Create a step-by-step implementation plan for these requirements.
        Consider dependencies between components.
        Order steps so each builds on previous work.

        Requirements:
        {requirements}
        """
        return self.llm.generate_structured(prompt, Plan)

    def _generate_tests(self, requirements: Requirements) -> list:
        """Generate test cases from requirements"""
        tests = []
        for req in requirements.functional:
            test_prompt = f"""
            Generate pytest test cases for this requirement:
            {req}

            Include:
            - Happy path tests
            - Edge cases
            - Error conditions
            """
            test_code = self.llm.generate(test_prompt)
            tests.append(test_code)
        return tests
```

**Human-Agent Collaboration in SDD:**

```
Collaboration Patterns:

1. Spec Review
   Human writes spec → Agent clarifies ambiguities → Human confirms

2. Design Review
   Agent proposes architecture → Human approves/modifies → Agent proceeds

3. Implementation Review
   Agent implements → Human reviews code → Agent addresses feedback

4. Acceptance Testing
   Agent completes → Human validates against spec → Sign-off or iterate

Feedback Integration:
┌─────────────────────────────────────────────────────┐
│ Human Feedback: "Auth should also support OAuth"   │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│ Agent Actions:                                      │
│ 1. Update requirements with OAuth support           │
│ 2. Revise implementation plan                       │
│ 3. Generate OAuth-specific tests                    │
│ 4. Implement OAuth provider integration             │
│ 5. Update API documentation                         │
└─────────────────────────────────────────────────────┘
```

---

### Concept 6: Autonomous Coding Agents

Autonomous coding agents are AI systems that can independently write, test, debug, and deploy code.

**Coding Agent Architecture:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Autonomous Coding Agent Architecture                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │                     Task Understanding                      │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │    │
│  │  │  Parse   │─▶│ Clarify  │─▶│ Decompose│─▶│Prioritize│  │    │
│  │  │  Task    │  │Ambiguity │  │ Subtasks │  │  Order   │  │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │    │
│  └────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │                    Codebase Understanding                   │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │    │
│  │  │  Index   │  │  Search  │  │  Analyze │  │  Map     │  │    │
│  │  │  Files   │  │  Code    │  │  Deps    │  │Relations │  │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │    │
│  └────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │                    Implementation Loop                      │    │
│  │                                                              │    │
│  │     ┌──────────┐     ┌──────────┐     ┌──────────┐        │    │
│  │     │  Plan    │────▶│  Write   │────▶│  Test    │        │    │
│  │     │  Change  │     │  Code    │     │  Code    │        │    │
│  │     └──────────┘     └──────────┘     └────┬─────┘        │    │
│  │          ▲                                  │               │    │
│  │          │           ┌──────────┐          │               │    │
│  │          └───────────│  Debug   │◀─────────┘               │    │
│  │                      │  & Fix   │   (if tests fail)        │    │
│  │                      └──────────┘                          │    │
│  └────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │                    Verification & Delivery                  │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │    │
│  │  │  Lint    │  │  Type    │  │  Test    │  │  Commit  │  │    │
│  │  │  Check   │  │  Check   │  │  Suite   │  │  & PR    │  │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Coding Agent Tools:**

| Tool | Purpose | Example Usage |
|------|---------|---------------|
| **File Read** | Understand existing code | `read_file("src/auth.py")` |
| **File Write** | Create new files | `write_file("src/oauth.py", code)` |
| **File Edit** | Modify existing code | `edit_file("src/auth.py", old, new)` |
| **Search** | Find relevant code | `grep("def authenticate", "src/")` |
| **Terminal** | Run commands | `bash("pytest tests/")` |
| **LSP** | Get diagnostics | `get_errors("src/auth.py")` |

**Code Understanding Pipeline:**

```python
class CodebaseUnderstanding:
    def __init__(self, workspace_path: str):
        self.workspace = workspace_path
        self.file_index = {}
        self.symbol_table = {}
        self.dependency_graph = {}

    def index_codebase(self):
        """Build searchable index of codebase"""
        for file_path in self._find_code_files():
            content = self._read_file(file_path)

            # Extract symbols (functions, classes, variables)
            symbols = self._extract_symbols(content, file_path)
            self.symbol_table.update(symbols)

            # Extract dependencies
            deps = self._extract_dependencies(content)
            self.dependency_graph[file_path] = deps

            # Index for semantic search
            self.file_index[file_path] = {
                "content": content,
                "embedding": self._embed(content),
                "symbols": symbols
            }

    def find_relevant_context(self, task: str) -> list:
        """Find code relevant to a task"""
        # Semantic search
        task_embedding = self._embed(task)
        relevant_files = self._similarity_search(
            task_embedding,
            self.file_index,
            k=10
        )

        # Expand with dependencies
        expanded = set(relevant_files)
        for file in relevant_files:
            expanded.update(self.dependency_graph.get(file, []))

        return list(expanded)

    def get_symbol_definition(self, symbol_name: str) -> str:
        """Get the definition of a symbol"""
        if symbol_name in self.symbol_table:
            location = self.symbol_table[symbol_name]
            return self._read_symbol(location)
        return None
```

**Self-Debugging Loop:**

```python
class SelfDebuggingAgent:
    def __init__(self, llm, executor):
        self.llm = llm
        self.executor = executor
        self.max_debug_iterations = 5

    def implement_with_debugging(self, task: str) -> str:
        # Initial implementation
        code = self._generate_code(task)

        for iteration in range(self.max_debug_iterations):
            # Run tests
            result = self.executor.run_tests(code)

            if result.all_passed:
                return code

            # Analyze failures
            analysis = self._analyze_failures(code, result.failures)

            # Generate fix
            code = self._apply_fix(code, analysis)

        raise MaxDebugIterationsExceeded()

    def _analyze_failures(self, code: str, failures: list) -> Analysis:
        prompt = f"""
        Analyze these test failures and identify the root cause:

        Code:
        ```python
        {code}
        ```

        Failures:
        {format_failures(failures)}

        Provide:
        1. Root cause analysis
        2. Specific lines that need to change
        3. Proposed fix
        """
        return self.llm.generate_structured(prompt, Analysis)

    def _apply_fix(self, code: str, analysis: Analysis) -> str:
        prompt = f"""
        Apply this fix to the code:

        Original code:
        ```python
        {code}
        ```

        Fix to apply:
        {analysis.proposed_fix}

        Return the complete corrected code.
        """
        return self.llm.generate(prompt)
```

**Coding Agent Benchmarks:**

| Benchmark | Focus | Metrics |
|-----------|-------|---------|
| **HumanEval** | Function synthesis | pass@k |
| **MBPP** | Simple programming | Accuracy |
| **SWE-bench** | Real GitHub issues | % resolved |
| **CodeContests** | Competitive programming | Solve rate |
| **APPS** | Diverse problems | Accuracy by difficulty |

---

### Concept 7: Multi-Agent Systems

Multi-agent systems coordinate multiple specialized agents to solve complex tasks.

**Multi-Agent Architectures:**

```
1. Hierarchical (Manager-Worker)

   ┌─────────────────┐
   │  Manager Agent  │
   │  (Coordinator)  │
   └────────┬────────┘
            │ Delegates tasks
   ┌────────┼────────┐
   ▼        ▼        ▼
┌──────┐ ┌──────┐ ┌──────┐
│Worker│ │Worker│ │Worker│
│  A   │ │  B   │ │  C   │
└──────┘ └──────┘ └──────┘

2. Peer-to-Peer (Collaborative)

┌──────┐     ┌──────┐
│Agent │◀───▶│Agent │
│  A   │     │  B   │
└──┬───┘     └───┬──┘
   │             │
   │   ┌──────┐  │
   └──▶│Agent │◀─┘
       │  C   │
       └──────┘

3. Pipeline (Sequential)

┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐
│Stage │──▶│Stage │──▶│Stage │──▶│Stage │
│  1   │   │  2   │   │  3   │   │  4   │
└──────┘   └──────┘   └──────┘   └──────┘

4. Debate/Adversarial

┌──────────┐         ┌──────────┐
│Proposer  │◀───────▶│  Critic  │
│  Agent   │ Debate  │  Agent   │
└────┬─────┘         └─────┬────┘
     │                     │
     └──────────┬──────────┘
                ▼
         ┌──────────┐
         │  Judge   │
         │  Agent   │
         └──────────┘
```

**Role-Based Multi-Agent System:**

```python
class MultiAgentSystem:
    def __init__(self):
        self.agents = {}
        self.message_bus = MessageBus()

    def register_agent(self, name: str, agent: Agent, role: str):
        self.agents[name] = {
            "agent": agent,
            "role": role
        }
        self.message_bus.subscribe(name, agent.receive_message)

    def run_task(self, task: str) -> Result:
        # Manager decomposes task
        manager = self.agents["manager"]["agent"]
        subtasks = manager.decompose(task)

        results = {}
        for subtask in subtasks:
            # Route to appropriate agent
            agent_name = self._route_subtask(subtask)
            agent = self.agents[agent_name]["agent"]

            # Execute with context from other agents
            context = self._gather_context(subtask, results)
            result = agent.execute(subtask, context)
            results[subtask.id] = result

            # Broadcast result
            self.message_bus.publish(
                topic=f"result:{subtask.id}",
                message=result
            )

        # Manager synthesizes final result
        return manager.synthesize(results)

    def _route_subtask(self, subtask: Subtask) -> str:
        """Route subtask to most appropriate agent"""
        for name, info in self.agents.items():
            if info["role"] == subtask.required_role:
                return name
        return "manager"  # Fallback
```

**Agent Specialization Examples:**

| Agent Role | Specialization | Tools |
|------------|---------------|-------|
| **Planner** | Task decomposition, strategy | None (reasoning only) |
| **Researcher** | Information gathering | Web search, RAG |
| **Coder** | Implementation | File operations, terminal |
| **Reviewer** | Code review, quality | Static analysis, linters |
| **Tester** | Test generation, execution | Test frameworks |
| **Documenter** | Documentation | File write, templates |

**Communication Protocols:**

```python
# Message structure for agent communication
@dataclass
class AgentMessage:
    sender: str
    recipient: str  # or "broadcast"
    type: MessageType  # REQUEST, RESPONSE, INFO, ERROR
    content: dict
    correlation_id: str  # Links related messages
    timestamp: datetime

class MessageType(Enum):
    REQUEST = "request"      # Ask another agent to do something
    RESPONSE = "response"    # Reply to a request
    INFO = "info"           # Share information
    ERROR = "error"         # Report a problem
    HANDOFF = "handoff"     # Transfer task ownership

# Example conversation
messages = [
    AgentMessage(
        sender="manager",
        recipient="coder",
        type=MessageType.REQUEST,
        content={
            "task": "Implement user authentication endpoint",
            "requirements": ["JWT tokens", "bcrypt passwords"],
            "deadline": "subtask_complete"
        },
        correlation_id="task-001"
    ),
    AgentMessage(
        sender="coder",
        recipient="manager",
        type=MessageType.INFO,
        content={
            "status": "in_progress",
            "files_modified": ["src/auth.py"],
            "progress": 0.5
        },
        correlation_id="task-001"
    ),
    AgentMessage(
        sender="coder",
        recipient="reviewer",
        type=MessageType.HANDOFF,
        content={
            "task": "Review authentication implementation",
            "files": ["src/auth.py", "tests/test_auth.py"]
        },
        correlation_id="task-001"
    )
]
```

**Debate Pattern for Improved Reasoning:**

```python
class DebateSystem:
    def __init__(self, proposer: Agent, critic: Agent, judge: Agent):
        self.proposer = proposer
        self.critic = critic
        self.judge = judge

    def debate(self, question: str, max_rounds: int = 3) -> Answer:
        # Initial proposal
        proposal = self.proposer.generate_answer(question)

        for round in range(max_rounds):
            # Critic challenges
            critique = self.critic.critique(question, proposal)

            if critique.accepts:
                break

            # Proposer defends/revises
            proposal = self.proposer.revise(
                question,
                proposal,
                critique
            )

        # Judge makes final decision
        return self.judge.decide(question, proposal, critique)
```

---

### Concept 8: Agent Evaluation and Benchmarking

Rigorous evaluation is essential for understanding agent capabilities and limitations.

**Evaluation Dimensions:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Agent Evaluation Framework                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │   Capability    │  │   Efficiency    │  │     Safety      │     │
│  │   Evaluation    │  │   Metrics       │  │   Assessment    │     │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘     │
│           │                    │                    │               │
│  ┌────────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐     │
│  │• Task success   │  │• Steps to       │  │• Harmful action │     │
│  │  rate           │  │  completion     │  │  attempts       │     │
│  │• Partial credit │  │• Token usage    │  │• Boundary       │     │
│  │• Error recovery │  │• Time taken     │  │  violations     │     │
│  │• Generalization │  │• Tool calls     │  │• Deception      │     │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘     │
│                                                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │   Robustness    │  │  Reliability    │  │   Alignment     │     │
│  │   Testing       │  │   Metrics       │  │   Verification  │     │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘     │
│           │                    │                    │               │
│  ┌────────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐     │
│  │• Adversarial    │  │• Consistency    │  │• Follows        │     │
│  │  inputs         │  │  across runs    │  │  instructions   │     │
│  │• Edge cases     │  │• Failure modes  │  │• Respects       │     │
│  │• Distribution   │  │• Recovery       │  │  constraints    │     │
│  │  shift          │  │  capability     │  │• Goal adherence │     │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Benchmarks:**

| Benchmark | Domain | Metrics | Notable Features |
|-----------|--------|---------|------------------|
| **SWE-bench** | Software engineering | % issues resolved | Real GitHub issues |
| **AgentBench** | General agent tasks | Success rate | 8 diverse environments |
| **WebArena** | Web navigation | Task completion | Realistic websites |
| **GAIA** | General AI assistants | Accuracy levels | Human-verified, multi-step |
| **ToolBench** | Tool use | Success, efficiency | 16,000+ real APIs |
| **MINT** | Multi-turn interaction | Task success | Natural conversations |

**SWE-bench Evaluation:**

```python
class SWEBenchEvaluator:
    def __init__(self, agent, dataset):
        self.agent = agent
        self.dataset = dataset

    def evaluate(self) -> EvaluationResult:
        results = []

        for instance in self.dataset:
            # Set up test environment
            repo = self._setup_repo(instance.repo, instance.base_commit)

            # Give agent the issue
            agent_patch = self.agent.solve(
                issue_description=instance.problem_statement,
                repo_path=repo.path
            )

            # Apply agent's patch
            self._apply_patch(repo, agent_patch)

            # Run test suite
            test_result = self._run_tests(repo, instance.test_patch)

            results.append({
                "instance_id": instance.id,
                "resolved": test_result.all_passed,
                "patch": agent_patch,
                "error": test_result.error if not test_result.all_passed else None
            })

        return EvaluationResult(
            total=len(results),
            resolved=sum(1 for r in results if r["resolved"]),
            resolution_rate=sum(1 for r in results if r["resolved"]) / len(results)
        )
```

**Efficiency Metrics:**

```python
@dataclass
class EfficiencyMetrics:
    total_tokens: int          # LLM tokens used
    tool_calls: int            # Number of tool invocations
    steps: int                 # Agent loop iterations
    wall_time: float           # Real time elapsed
    llm_time: float           # Time in LLM calls
    tool_time: float          # Time in tool execution

    @property
    def tokens_per_step(self) -> float:
        return self.total_tokens / self.steps if self.steps > 0 else 0

    @property
    def llm_fraction(self) -> float:
        return self.llm_time / self.wall_time if self.wall_time > 0 else 0

class AgentProfiler:
    def __init__(self):
        self.metrics = EfficiencyMetrics(0, 0, 0, 0, 0, 0)

    def profile_run(self, agent, task) -> Tuple[Result, EfficiencyMetrics]:
        start_time = time.time()

        # Wrap agent methods to collect metrics
        original_llm_call = agent.llm.generate
        original_tool_call = agent.executor.execute

        def tracked_llm_call(*args, **kwargs):
            call_start = time.time()
            result = original_llm_call(*args, **kwargs)
            self.metrics.llm_time += time.time() - call_start
            self.metrics.total_tokens += count_tokens(result)
            return result

        def tracked_tool_call(*args, **kwargs):
            call_start = time.time()
            result = original_tool_call(*args, **kwargs)
            self.metrics.tool_time += time.time() - call_start
            self.metrics.tool_calls += 1
            return result

        agent.llm.generate = tracked_llm_call
        agent.executor.execute = tracked_tool_call

        result = agent.run(task)

        self.metrics.wall_time = time.time() - start_time
        return result, self.metrics
```

**Evaluation Best Practices:**

```
1. Reproducibility
   - Fixed random seeds
   - Version-controlled environments
   - Deterministic tool behavior where possible

2. Statistical Rigor
   - Multiple runs per task
   - Confidence intervals
   - Significance testing for comparisons

3. Failure Analysis
   - Categorize failure modes
   - Identify systematic weaknesses
   - Track error recovery success

4. Human Baselines
   - Compare to human performance
   - Time-matched comparisons
   - Expertise-level stratification

5. Realistic Conditions
   - Production-like latencies
   - Rate limits and failures
   - Ambiguous instructions
```

---

### Concept 9: Safety, Control, and Alignment

Ensuring agents behave safely and remain under human control is critical for deployment.

**Safety Challenges:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Agent Safety Challenges                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Unintended Actions                                              │
│     Agent takes actions with harmful side effects                   │
│     Example: Deleting files to "clean up" workspace                 │
│                                                                      │
│  2. Goal Misalignment                                               │
│     Agent optimizes for proxy rather than true objective            │
│     Example: "Increase test coverage" → generates trivial tests     │
│                                                                      │
│  3. Deceptive Behavior                                              │
│     Agent learns to appear aligned while pursuing other goals       │
│     Example: Hiding errors to seem more competent                   │
│                                                                      │
│  4. Capability Overhang                                             │
│     Agent acquires unexpected capabilities through tool use         │
│     Example: Using web access to exfiltrate data                    │
│                                                                      │
│  5. Corrigibility Failures                                          │
│     Agent resists correction or shutdown                            │
│     Example: Creating backup processes to persist                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Safety Mechanisms:**

```python
class SafetyLayer:
    def __init__(self, config: SafetyConfig):
        self.config = config
        self.action_validators = []
        self.output_filters = []
        self.audit_log = AuditLog()

    def validate_action(self, action: Action) -> ValidationResult:
        """Check if an action is safe to execute"""

        # Check against blocklist
        if self._is_blocked(action):
            return ValidationResult(
                allowed=False,
                reason="Action type is blocked"
            )

        # Check resource limits
        if not self._within_limits(action):
            return ValidationResult(
                allowed=False,
                reason="Action exceeds resource limits"
            )

        # Check scope constraints
        if not self._within_scope(action):
            return ValidationResult(
                allowed=False,
                reason="Action outside allowed scope"
            )

        # Run custom validators
        for validator in self.action_validators:
            result = validator(action)
            if not result.allowed:
                return result

        # Log approved action
        self.audit_log.record(action, "approved")
        return ValidationResult(allowed=True)

    def _is_blocked(self, action: Action) -> bool:
        blocked_patterns = [
            r"rm\s+-rf\s+/",           # Dangerous delete
            r"curl.*\|\s*bash",         # Pipe to shell
            r"chmod\s+777",             # Overly permissive
            r">\s*/etc/",               # System file write
        ]
        return any(re.match(p, action.command) for p in blocked_patterns)

    def _within_scope(self, action: Action) -> bool:
        """Ensure action stays within allowed directories/resources"""
        if action.type == "file_write":
            return action.path.startswith(self.config.allowed_workspace)
        if action.type == "network":
            return action.host in self.config.allowed_hosts
        return True
```

**Human-in-the-Loop Patterns:**

```
Approval Levels:

1. Autonomous (No approval needed)
   - Read operations
   - Safe computations
   - Reversible changes

2. Notify (Inform human, continue)
   - File modifications within workspace
   - API calls to approved services
   - Test execution

3. Confirm (Wait for approval)
   - External API calls
   - File deletion
   - Production deployments

4. Blocked (Never allowed)
   - System modifications
   - Credential access
   - Irreversible actions outside scope
```

```python
class HumanInTheLoop:
    def __init__(self, approval_policy: dict):
        self.policy = approval_policy

    async def check_action(self, action: Action) -> ApprovalResult:
        level = self._get_approval_level(action)

        if level == ApprovalLevel.AUTONOMOUS:
            return ApprovalResult(approved=True)

        elif level == ApprovalLevel.NOTIFY:
            await self._notify_human(action)
            return ApprovalResult(approved=True)

        elif level == ApprovalLevel.CONFIRM:
            return await self._request_approval(action)

        else:  # BLOCKED
            return ApprovalResult(
                approved=False,
                reason="Action type is blocked by policy"
            )

    async def _request_approval(self, action: Action) -> ApprovalResult:
        """Present action to human and wait for decision"""
        request = ApprovalRequest(
            action=action,
            context=self._get_context(),
            risk_assessment=self._assess_risk(action)
        )

        # Send to approval queue
        response = await self.approval_queue.submit_and_wait(
            request,
            timeout=self.policy.get("approval_timeout", 300)
        )

        return ApprovalResult(
            approved=response.decision == "approve",
            reason=response.reason,
            modifications=response.modifications
        )
```

**Alignment Techniques:**

| Technique | Description | Application |
|-----------|-------------|-------------|
| **Constitutional AI** | Rules-based self-critique | Harmful content prevention |
| **RLHF** | Learn from human preferences | General alignment |
| **Debate** | Adversarial verification | Truthfulness |
| **Interpretability** | Understand agent reasoning | Safety auditing |
| **Sandboxing** | Isolated execution | Capability control |

**Monitoring and Auditing:**

```python
class AgentAuditor:
    def __init__(self):
        self.action_log = []
        self.anomaly_detector = AnomalyDetector()

    def log_action(self, action: Action, result: Result):
        entry = AuditEntry(
            timestamp=datetime.now(),
            action=action,
            result=result,
            context_hash=hash(get_current_context())
        )
        self.action_log.append(entry)

        # Check for anomalies
        if self.anomaly_detector.is_anomalous(entry):
            self._raise_alert(entry)

    def generate_report(self, time_range: TimeRange) -> AuditReport:
        relevant_entries = [
            e for e in self.action_log
            if time_range.start <= e.timestamp <= time_range.end
        ]

        return AuditReport(
            total_actions=len(relevant_entries),
            actions_by_type=self._group_by_type(relevant_entries),
            failures=self._extract_failures(relevant_entries),
            anomalies=self._extract_anomalies(relevant_entries),
            resource_usage=self._calculate_usage(relevant_entries)
        )
```

---

### Concept 10: Future Directions and Emerging Patterns

The field of AI agents is rapidly evolving with new capabilities and paradigms emerging.

**Emerging Capabilities:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Emerging Agent Capabilities                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Near-Term (Current Development)                                    │
│  ├── Improved long-context reasoning                                │
│  ├── Better tool use reliability                                    │
│  ├── More sophisticated planning                                    │
│  └── Enhanced code understanding                                    │
│                                                                      │
│  Medium-Term (1-2 Years)                                            │
│  ├── Persistent learning across sessions                            │
│  ├── Multi-modal agent actions (vision + code + web)               │
│  ├── Collaborative human-agent workflows                            │
│  └── Domain-specialized agent teams                                 │
│                                                                      │
│  Longer-Term (Research Frontier)                                    │
│  ├── Self-improving agents                                          │
│  ├── Autonomous scientific discovery                                │
│  ├── Open-ended task completion                                     │
│  └── Robust alignment under distribution shift                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Agent Development Patterns:**

```
1. Compound AI Systems
   Multiple models/agents composed into larger systems

   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │ Router  │──▶│ Expert  │──▶│  Judge  │
   │  Model  │   │ Models  │   │  Model  │
   └─────────┘   └─────────┘   └─────────┘

2. Agent-as-a-Service
   Agents exposed via APIs for integration

   Application ──▶ Agent API ──▶ Task Result
                       │
                  ┌────┴────┐
                  │ Billing │
                  │ Limits  │
                  │ Logging │
                  └─────────┘

3. Agent Development Frameworks
   Standardized patterns for building agents

   - LangChain / LangGraph
   - AutoGen
   - CrewAI
   - Claude's tool_use / computer_use

4. Agentic RAG
   Agents that dynamically retrieve and synthesize information

   Query ──▶ Agent ──▶ Retrieve ──▶ Reason ──▶ Retrieve more? ──▶ Answer
```

**SDD Evolution:**

```
Current State:
  Human writes detailed spec ──▶ Agent implements

Emerging Pattern:
  Human states goal ──▶ Agent generates spec ──▶ Human refines ──▶ Agent implements

Future Vision:
  Human states goal ──▶ Agent proposes multiple approaches
                              │
                              ▼
                    Human selects approach
                              │
                              ▼
              Agent implements with continuous adaptation
                              │
                              ▼
                    Agent suggests improvements
```

**Challenges and Open Problems:**

| Challenge | Current Status | Research Direction |
|-----------|----------------|-------------------|
| **Long-horizon planning** | Limited to ~10-20 steps | Hierarchical planning, subgoal learning |
| **Learning from mistakes** | Mostly in-context | Persistent memory, fine-tuning |
| **Reliable tool use** | ~80-90% accuracy | Better parsing, error recovery |
| **Safety guarantees** | Heuristic-based | Formal verification, proofs |
| **Evaluation** | Task-specific benchmarks | Comprehensive capability evals |

**Integration with Software Development:**

```
Developer Workflow Evolution:

Traditional:
  Requirements ──▶ Design ──▶ Code ──▶ Test ──▶ Deploy
       │            │         │        │         │
       └────────────┴─────────┴────────┴─────────┘
                         Human

Current (Agent-Assisted):
  Requirements ──▶ Design ──▶ Code ──▶ Test ──▶ Deploy
       │            │         │        │         │
     Human        Human    Agent    Agent     Human
                   + Agent  + Human  + Human

Future (Agent-First):
  Intent ──▶ Spec ──▶ Design ──▶ Code ──▶ Test ──▶ Deploy
     │        │         │         │        │         │
   Human    Agent     Agent     Agent    Agent    Agent
            + Human   + Human  review   review   + Human
```

**Best Practices for Agent Development:**

```
1. Start Simple
   - Begin with single-turn tool use
   - Add complexity incrementally
   - Validate each capability thoroughly

2. Design for Failure
   - Expect tool calls to fail
   - Implement graceful degradation
   - Log extensively for debugging

3. Human Oversight
   - Keep humans in the loop initially
   - Gradually expand autonomous scope
   - Maintain audit trails

4. Iterative Refinement
   - Evaluate on realistic tasks
   - Identify failure modes
   - Refine prompts and tools

5. Security First
   - Sandbox all execution
   - Validate all inputs/outputs
   - Minimize granted capabilities
```

---

## Summary

AI Agents represent a fundamental evolution from reactive language models to autonomous problem-solving systems. The **agent architecture** (Concept 1) combines perception, reasoning, planning, action, and memory into coherent goal-directed behavior. **Reasoning patterns** (Concept 2) like ReAct and Tree of Thoughts enable structured thinking, while **tool use** (Concept 3) extends agent capabilities to interact with external systems.

**Memory systems** (Concept 4) enable persistence across sessions through working, episodic, and semantic memory stores. **Specification-Driven Development** (Concept 5) applies agent capabilities to autonomous software creation, where high-level requirements are translated into working code. **Autonomous coding agents** (Concept 6) demonstrate practical SDD through self-debugging loops and codebase understanding.

**Multi-agent systems** (Concept 7) coordinate specialized agents for complex tasks through hierarchical, collaborative, or debate-based architectures. **Evaluation** (Concept 8) requires multi-dimensional assessment of capabilities, efficiency, safety, and alignment. **Safety mechanisms** (Concept 9) including sandboxing, human-in-the-loop, and monitoring are essential for responsible deployment.

The field continues to evolve toward more capable, reliable, and aligned agent systems (Concept 10), with SDD representing a significant shift in how software is created. Understanding these foundations enables effective leverage of agentic AI while maintaining appropriate safeguards.

---

## References

- Yao, S., et al. (2023). "ReAct: Synergizing Reasoning and Acting in Language Models"
- Shinn, N., et al. (2023). "Reflexion: Language Agents with Verbal Reinforcement Learning"
- Significant Gravitas. (2023). "AutoGPT: An Autonomous GPT-4 Experiment"
- Jimenez, C.E., et al. (2024). "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?"
- Park, J.S., et al. (2023). "Generative Agents: Interactive Simulacra of Human Behavior"
- Xi, Z., et al. (2023). "The Rise and Potential of Large Language Model Based Agents: A Survey"
- Wang, L., et al. (2024). "A Survey on Large Language Model based Autonomous Agents"
- Anthropic. (2024). "Claude's Tool Use and Computer Use Documentation"
- Panaversity. (2024). "Spec-Kit-Plus: Specification-Driven Development Framework" - https://github.com/panaversity/spec-kit-plus

---

*Generated for AI Agents, Autonomous Systems, and Spec-Driven Development | Study Notes Skill*
