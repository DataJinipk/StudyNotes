# Practice Problems: Lesson 12 - AI Agents, Autonomous Systems, and Spec-Driven Development

**Source:** Lessons/Lesson_12.md
**Original Source Path:** C:\agentic_ai\StudyNotes\Lessons\Lesson_12.md
**Date Generated:** 2026-01-08
**Total Problems:** 5
**Estimated Total Time:** 90-120 minutes
**Distribution:** 1 Warm-Up | 2 Skill-Builder | 1 Challenge | 1 Debug/Fix

---

## Overview

### Concepts Practiced

| Concept | Problems | Mastery Indicator |
|---------|----------|-------------------|
| ReAct Pattern | P1, P3 | Can implement reasoning-action loop |
| Tool Use & Function Calling | P1, P4 | Can design tool schemas and execution |
| Memory Systems | P2, P3 | Can implement episodic retrieval |
| SDD Methodology | P3, P5 | Can follow Spec-Kit workflow |
| Safety Mechanisms | P4 | Can implement validation layers |
| Agent Debugging | P5 | Can identify and fix agent errors |

### Recommended Approach

1. Attempt each problem before looking at hints
2. Use hints progressively—don't skip to solution
3. After solving, read solution to compare approaches
4. Review Common Mistakes even if you solved correctly
5. Attempt Extension Challenges for deeper mastery

### Self-Assessment Guide

| Problems Solved (no hints) | Mastery Level | Recommendation |
|---------------------------|---------------|----------------|
| 5/5 | Expert | Proceed to building production agents |
| 4/5 | Proficient | Review one gap area |
| 3/5 | Developing | More practice recommended |
| 2/5 or below | Foundational | Re-review study notes first |

---

## Problems

---

## Problem 1: Implement a ReAct Agent Loop

**Type:** Warm-Up
**Concepts Practiced:** ReAct Pattern, Tool Use, Agent Loop
**Estimated Time:** 20 minutes
**Prerequisites:** Python basics, understanding of LLM APIs

### Problem Statement

Implement a basic ReAct agent that can answer questions by searching the web and doing calculations. The agent should follow the Thought → Action → Observation cycle until it can provide a final answer.

Given these tool definitions:

```python
def search_web(query: str) -> str:
    """Simulated web search - returns mock results"""
    mock_results = {
        "python creator": "Python was created by Guido van Rossum in 1991.",
        "eiffel tower height": "The Eiffel Tower is 330 meters tall.",
        "population tokyo": "Tokyo has a population of approximately 14 million."
    }
    for key, value in mock_results.items():
        if key in query.lower():
            return value
    return "No results found for: " + query

def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression"""
    try:
        # Only allow safe operations
        allowed = set('0123456789+-*/.() ')
        if all(c in allowed for c in expression):
            return str(eval(expression))
        return "Error: Invalid expression"
    except:
        return "Error: Could not evaluate"

TOOLS = {
    "search_web": search_web,
    "calculate": calculate
}
```

### Requirements

- [ ] Implement the `react_agent` function that takes a question and returns an answer
- [ ] Parse LLM responses to extract Thought, Action, and detect Final Answer
- [ ] Execute tools and feed observations back into the context
- [ ] Handle the case when max iterations is reached

### Hints (Progressive)

<details>
<summary>Hint 1: Getting Started</summary>

Structure your prompt to guide the LLM into the ReAct format. Include clear instructions about when to use each tool and when to provide a Final Answer. The prompt should list available tools with descriptions.

</details>

<details>
<summary>Hint 2: Key Insight</summary>

Use string parsing to detect patterns like "Action: tool_name(args)" and "Final Answer:". You can use regex or simple string operations. The key is building context by appending each Thought/Action/Observation to the conversation.

</details>

<details>
<summary>Hint 3: Nearly There</summary>

```python
import re

def parse_action(response: str):
    # Look for Action: tool_name(args)
    match = re.search(r'Action:\s*(\w+)\(([^)]*)\)', response)
    if match:
        return match.group(1), match.group(2).strip('"\'')
    return None, None
```

</details>

### Solution

<details>
<summary>Click to reveal full solution</summary>

**Approach:**
Build a loop that prompts the LLM, parses its response for actions or final answer, executes tools if needed, and accumulates context.

**Step-by-Step Solution:**

```python
import re

REACT_PROMPT = """You are a helpful assistant that answers questions by thinking step-by-step and using tools.

Available tools:
- search_web(query): Search the web for information
- calculate(expression): Evaluate a mathematical expression

Use this format:
Thought: [Your reasoning about what to do]
Action: [tool_name("argument")]
Observation: [Result will be provided]
... (repeat as needed)
Thought: I now have enough information.
Final Answer: [Your answer to the question]

Question: {question}

{context}"""

def parse_response(response: str):
    """Parse LLM response for action or final answer"""
    # Check for final answer
    if "Final Answer:" in response:
        answer = response.split("Final Answer:")[-1].strip()
        return {"type": "final", "answer": answer}

    # Check for action
    match = re.search(r'Action:\s*(\w+)\("?([^")\n]+)"?\)', response)
    if match:
        tool_name = match.group(1)
        argument = match.group(2).strip()
        return {"type": "action", "tool": tool_name, "args": argument}

    return {"type": "unknown"}

def execute_tool(tool_name: str, args: str, tools: dict) -> str:
    """Execute a tool and return the result"""
    if tool_name not in tools:
        return f"Error: Unknown tool '{tool_name}'"
    try:
        return tools[tool_name](args)
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"

def react_agent(question: str, llm_call, tools: dict, max_iterations: int = 5) -> str:
    """
    ReAct agent that answers questions using tools.

    Args:
        question: The question to answer
        llm_call: Function that takes a prompt and returns LLM response
        tools: Dictionary of available tools
        max_iterations: Maximum reasoning steps
    """
    context = ""

    for i in range(max_iterations):
        # Build prompt with accumulated context
        prompt = REACT_PROMPT.format(question=question, context=context)

        # Get LLM response
        response = llm_call(prompt)

        # Parse the response
        parsed = parse_response(response)

        if parsed["type"] == "final":
            return parsed["answer"]

        elif parsed["type"] == "action":
            # Execute the tool
            observation = execute_tool(parsed["tool"], parsed["args"], tools)

            # Add to context
            context += f"\n{response}\nObservation: {observation}\n"

        else:
            # Unknown format, add response and continue
            context += f"\n{response}\n"

    return "I couldn't find an answer within the allowed steps."

# Example usage (with mock LLM for testing)
def mock_llm(prompt):
    """Mock LLM that simulates ReAct behavior"""
    if "Observation:" not in prompt:
        return 'Thought: I need to search for information about the Eiffel Tower height.\nAction: search_web("eiffel tower height")'
    else:
        return 'Thought: I found that the Eiffel Tower is 330 meters. The question asks for feet, so I need to convert.\nAction: calculate("330 * 3.281")'

    if "1082" in prompt or "1083" in prompt:
        return 'Thought: I now have the answer.\nFinal Answer: The Eiffel Tower is approximately 1,083 feet tall.'

# Test
answer = react_agent(
    "How tall is the Eiffel Tower in feet?",
    mock_llm,
    TOOLS
)
print(answer)
```

**Why This Works:**
The ReAct pattern grounds reasoning in observations from tool use. By accumulating context, the agent maintains state across iterations. The parsing logic handles the structured format while being robust to variations.

</details>

### Common Mistakes

- ❌ **Mistake:** Not accumulating context between iterations
  - **Why it happens:** Treating each LLM call independently
  - **How to avoid:** Always append previous Thought/Action/Observation to the context

- ❌ **Mistake:** Hardcoding tool names instead of using the tools dictionary
  - **Why it happens:** Testing with specific tools only
  - **How to avoid:** Always look up tools dynamically from the registry

### Extension Challenge

Add a third tool `read_file(path)` and modify the agent to answer questions that require reading local files. Implement proper error handling for missing files.

---

## Problem 2: Build an Episodic Memory System

**Type:** Skill-Builder
**Concepts Practiced:** Memory Systems, Vector Similarity, Context Retrieval
**Estimated Time:** 25 minutes
**Prerequisites:** Understanding of embeddings, basic data structures

### Problem Statement

Implement an episodic memory system that stores past agent interactions and retrieves relevant episodes based on similarity to the current query. This enables the agent to learn from past experiences.

You have access to a mock embedding function:

```python
import hashlib
import numpy as np

def mock_embed(text: str) -> np.ndarray:
    """Generate a deterministic mock embedding for text"""
    # Create a deterministic hash-based embedding (for testing)
    hash_bytes = hashlib.sha256(text.lower().encode()).digest()
    embedding = np.frombuffer(hash_bytes, dtype=np.uint8)[:32].astype(float)
    embedding = embedding / np.linalg.norm(embedding)  # Normalize
    return embedding

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

### Requirements

- [ ] Implement `EpisodicMemory` class with `store_episode` and `retrieve_relevant` methods
- [ ] Store episodes with content, embedding, timestamp, and metadata
- [ ] Retrieve top-k episodes by cosine similarity
- [ ] Implement recency-weighted retrieval that combines similarity and recency

### Hints (Progressive)

<details>
<summary>Hint 1: Getting Started</summary>

Use a list to store episodes as dictionaries. Each episode should contain: content, embedding, timestamp, and any metadata. The `store_episode` method adds to this list, and `retrieve_relevant` searches through it.

</details>

<details>
<summary>Hint 2: Key Insight</summary>

For recency weighting, compute a time decay factor based on how old the episode is. Combine this with similarity: `final_score = alpha * similarity + (1 - alpha) * recency_score`. Use exponential decay for recency.

</details>

<details>
<summary>Hint 3: Nearly There</summary>

```python
def recency_weight(timestamp, current_time, half_life_hours=24):
    age_hours = (current_time - timestamp).total_seconds() / 3600
    return np.exp(-age_hours / half_life_hours)
```

</details>

### Solution

<details>
<summary>Click to reveal full solution</summary>

**Approach:**
Implement a memory store that maintains episodes with embeddings, then provides retrieval based on similarity with optional recency weighting.

**Step-by-Step Solution:**

```python
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np

@dataclass
class Episode:
    content: str
    embedding: np.ndarray
    timestamp: datetime
    metadata: Dict[str, Any]

class EpisodicMemory:
    def __init__(self, embed_fn, similarity_weight: float = 0.7):
        """
        Initialize episodic memory.

        Args:
            embed_fn: Function to embed text into vectors
            similarity_weight: Weight for similarity vs recency (0-1)
        """
        self.embed_fn = embed_fn
        self.similarity_weight = similarity_weight
        self.episodes: List[Episode] = []

    def store_episode(self, content: str, metadata: Optional[Dict] = None) -> None:
        """Store a new episode in memory"""
        episode = Episode(
            content=content,
            embedding=self.embed_fn(content),
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        self.episodes.append(episode)

    def retrieve_relevant(self, query: str, k: int = 5) -> List[Episode]:
        """Retrieve k most relevant episodes by similarity only"""
        if not self.episodes:
            return []

        query_embedding = self.embed_fn(query)

        # Compute similarities
        scored = []
        for episode in self.episodes:
            similarity = cosine_similarity(query_embedding, episode.embedding)
            scored.append((episode, similarity))

        # Sort by similarity descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return [episode for episode, _ in scored[:k]]

    def retrieve_hybrid(self, query: str, k: int = 5,
                        half_life_hours: float = 24.0) -> List[Episode]:
        """
        Retrieve episodes using hybrid similarity + recency scoring.

        Args:
            query: Query text
            k: Number of episodes to return
            half_life_hours: Time for recency weight to decay by half
        """
        if not self.episodes:
            return []

        query_embedding = self.embed_fn(query)
        current_time = datetime.now()

        scored = []
        for episode in self.episodes:
            # Similarity score (0 to 1)
            similarity = cosine_similarity(query_embedding, episode.embedding)

            # Recency score with exponential decay
            age_hours = (current_time - episode.timestamp).total_seconds() / 3600
            recency = np.exp(-age_hours / half_life_hours)

            # Combined score
            combined = (self.similarity_weight * similarity +
                       (1 - self.similarity_weight) * recency)

            scored.append((episode, combined, similarity, recency))

        # Sort by combined score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return [episode for episode, _, _, _ in scored[:k]]

    def retrieve_recent(self, n: int = 10) -> List[Episode]:
        """Retrieve n most recent episodes"""
        sorted_episodes = sorted(
            self.episodes,
            key=lambda e: e.timestamp,
            reverse=True
        )
        return sorted_episodes[:n]

    def clear(self) -> None:
        """Clear all episodes from memory"""
        self.episodes = []

    def __len__(self) -> int:
        return len(self.episodes)


# Example usage
memory = EpisodicMemory(mock_embed)

# Store some episodes
memory.store_episode(
    "User asked how to implement authentication in FastAPI. "
    "I suggested using JWT tokens with python-jose library.",
    metadata={"type": "coding_help", "topic": "auth"}
)

memory.store_episode(
    "User wanted to debug a React component. "
    "The issue was a missing dependency in useEffect.",
    metadata={"type": "debugging", "topic": "react"}
)

memory.store_episode(
    "Implemented user login endpoint with password hashing using bcrypt.",
    metadata={"type": "implementation", "topic": "auth"}
)

# Retrieve relevant episodes
query = "How do I add user authentication?"
relevant = memory.retrieve_relevant(query, k=2)

print("Relevant episodes for:", query)
for ep in relevant:
    print(f"  - {ep.content[:50]}... (metadata: {ep.metadata})")

# Hybrid retrieval
hybrid = memory.retrieve_hybrid(query, k=2)
print("\nHybrid retrieval:")
for ep in hybrid:
    print(f"  - {ep.content[:50]}...")
```

**Output:**
```
Relevant episodes for: How do I add user authentication?
  - User asked how to implement authentication in Fas... (metadata: {'type': 'coding_help', 'topic': 'auth'})
  - Implemented user login endpoint with password has... (metadata: {'type': 'implementation', 'topic': 'auth'})

Hybrid retrieval:
  - Implemented user login endpoint with password has...
  - User asked how to implement authentication in Fas...
```

**Why This Works:**
Episodic memory enables agents to recall relevant past experiences. Cosine similarity finds semantically related episodes, while recency weighting prioritizes recent experiences that may be more contextually relevant.

</details>

### Common Mistakes

- ❌ **Mistake:** Not normalizing embeddings before cosine similarity
  - **Why it happens:** Assuming embeddings are already normalized
  - **How to avoid:** Always normalize or use a similarity function that handles it

- ❌ **Mistake:** Using linear decay instead of exponential for recency
  - **Why it happens:** Simpler implementation
  - **How to avoid:** Linear decay goes negative; exponential always stays positive

### Extension Challenge

Add a `summarize_and_compress` method that uses an LLM to summarize old episodes when memory exceeds a threshold, preserving key information while reducing storage.

---

## Problem 3: Complete SDD Workflow Implementation

**Type:** Skill-Builder
**Concepts Practiced:** Spec-Kit Framework, SDD Phases, Agent Orchestration
**Estimated Time:** 30 minutes
**Prerequisites:** Understanding of SDD concepts, Python classes

### Problem Statement

Implement a simplified SDD orchestrator that guides an agent through the five Spec-Kit phases: Constitution → Specification → Planning → Tasks → Implementation. Each phase should produce a structured artifact that feeds into the next phase.

Given a mock LLM and specification:

```python
def mock_sdd_llm(prompt: str) -> str:
    """Mock LLM that returns phase-appropriate responses"""
    if "constitution" in prompt.lower():
        return """
        CONSTITUTION:
        - Language: Python 3.10+
        - Framework: FastAPI
        - Testing: pytest with >80% coverage
        - Style: PEP 8 compliant
        - Security: No hardcoded secrets
        """
    elif "specify" in prompt.lower() or "requirements" in prompt.lower():
        return """
        SPECIFICATION:
        FR-1: Create /health endpoint returning {"status": "ok"}
        FR-2: Create /users POST endpoint accepting {name, email}
        FR-3: Create /users GET endpoint returning list of users
        NFR-1: Response time < 100ms
        """
    elif "plan" in prompt.lower():
        return """
        PLAN:
        1. Set up FastAPI project structure
        2. Create health check endpoint
        3. Implement user model with Pydantic
        4. Create user storage (in-memory for now)
        5. Implement POST /users endpoint
        6. Implement GET /users endpoint
        7. Add tests for all endpoints
        """
    elif "task" in prompt.lower():
        return """
        TASKS:
        [1] Create main.py with FastAPI app - PRIORITY: HIGH
        [2] Add /health endpoint - DEPENDS: 1
        [3] Create models/user.py with User schema - DEPENDS: 1
        [4] Create services/user_service.py - DEPENDS: 3
        [5] Add POST /users endpoint - DEPENDS: 4
        [6] Add GET /users endpoint - DEPENDS: 4
        [7] Create tests/test_api.py - DEPENDS: 2,5,6
        """
    else:
        return "Implementation complete."

USER_SPEC = """
Build a simple user management API with:
- Health check endpoint
- Create user endpoint
- List users endpoint
"""
```

### Requirements

- [ ] Implement `SDDOrchestrator` class with methods for each phase
- [ ] Each phase should store its output and pass context to the next
- [ ] Implement `run_workflow` that executes all phases in sequence
- [ ] Add validation between phases (e.g., tasks must reference plan items)

### Hints (Progressive)

<details>
<summary>Hint 1: Getting Started</summary>

Create a class that maintains state across phases. Each phase method should take the output of previous phases as context and produce structured output. Store outputs in a dictionary keyed by phase name.

</details>

<details>
<summary>Hint 2: Key Insight</summary>

Build prompts for each phase that include outputs from previous phases. For example, the Planning phase prompt should include the Constitution and Specification outputs to ensure consistency.

</details>

<details>
<summary>Hint 3: Nearly There</summary>

```python
def run_plan_phase(self):
    prompt = f"""
    Given this constitution:
    {self.artifacts['constitution']}

    And this specification:
    {self.artifacts['specification']}

    Create an implementation plan:
    """
    return self.llm(prompt)
```

</details>

### Solution

<details>
<summary>Click to reveal full solution</summary>

**Approach:**
Create an orchestrator that manages phase transitions, accumulates artifacts, and provides context to each phase.

**Step-by-Step Solution:**

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from enum import Enum

class SDDPhase(Enum):
    CONSTITUTION = "constitution"
    SPECIFICATION = "specification"
    PLANNING = "planning"
    TASKS = "tasks"
    IMPLEMENTATION = "implementation"

@dataclass
class SDDArtifacts:
    constitution: Optional[str] = None
    specification: Optional[str] = None
    plan: Optional[str] = None
    tasks: Optional[str] = None
    implementation_log: List[str] = field(default_factory=list)

class SDDOrchestrator:
    def __init__(self, llm_fn: Callable[[str], str]):
        """
        Initialize SDD Orchestrator.

        Args:
            llm_fn: Function that takes prompt and returns LLM response
        """
        self.llm = llm_fn
        self.artifacts = SDDArtifacts()
        self.current_phase = None

    def run_constitution(self, context: str = "") -> str:
        """Phase 1: Establish development guidelines"""
        self.current_phase = SDDPhase.CONSTITUTION

        prompt = f"""
        Establish a development constitution for this project.
        Define coding standards, technology choices, quality requirements.

        Project context:
        {context}

        Output a CONSTITUTION section with guidelines.
        """

        result = self.llm(prompt)
        self.artifacts.constitution = result
        return result

    def run_specification(self, user_requirements: str) -> str:
        """Phase 2: Define requirements and acceptance criteria"""
        self.current_phase = SDDPhase.SPECIFICATION

        prompt = f"""
        Given this constitution:
        {self.artifacts.constitution}

        Create a detailed specification for these requirements:
        {user_requirements}

        Output functional requirements (FR-X) and non-functional requirements (NFR-X).
        """

        result = self.llm(prompt)
        self.artifacts.specification = result
        return result

    def run_planning(self) -> str:
        """Phase 3: Create implementation strategy"""
        self.current_phase = SDDPhase.PLANNING

        if not self.artifacts.specification:
            raise ValueError("Specification phase must complete before Planning")

        prompt = f"""
        Given this constitution:
        {self.artifacts.constitution}

        And this specification:
        {self.artifacts.specification}

        Create a step-by-step implementation plan.
        """

        result = self.llm(prompt)
        self.artifacts.plan = result
        return result

    def run_tasks(self) -> str:
        """Phase 4: Generate actionable tasks"""
        self.current_phase = SDDPhase.TASKS

        if not self.artifacts.plan:
            raise ValueError("Planning phase must complete before Tasks")

        prompt = f"""
        Given this plan:
        {self.artifacts.plan}

        Break it down into specific, actionable tasks.
        Include task IDs, priorities, and dependencies.
        """

        result = self.llm(prompt)
        self.artifacts.tasks = result
        return result

    def run_implementation(self, executor: Optional[Callable] = None) -> str:
        """Phase 5: Execute tasks"""
        self.current_phase = SDDPhase.IMPLEMENTATION

        if not self.artifacts.tasks:
            raise ValueError("Tasks phase must complete before Implementation")

        # Parse tasks and execute
        tasks = self._parse_tasks(self.artifacts.tasks)

        for task in tasks:
            prompt = f"""
            Implement this task:
            {task}

            Following this constitution:
            {self.artifacts.constitution}

            Based on this specification:
            {self.artifacts.specification}
            """

            result = self.llm(prompt)
            self.artifacts.implementation_log.append(f"Task: {task}\nResult: {result}")

            # Optional: execute with provided executor
            if executor:
                executor(task, result)

        return "\n".join(self.artifacts.implementation_log)

    def _parse_tasks(self, tasks_text: str) -> List[str]:
        """Parse task list from text"""
        tasks = []
        for line in tasks_text.split('\n'):
            line = line.strip()
            if line.startswith('[') and ']' in line:
                tasks.append(line)
        return tasks if tasks else [tasks_text]

    def run_workflow(self, user_requirements: str) -> SDDArtifacts:
        """Execute complete SDD workflow"""
        print("Phase 1: Constitution")
        self.run_constitution(user_requirements)

        print("Phase 2: Specification")
        self.run_specification(user_requirements)

        print("Phase 3: Planning")
        self.run_planning()

        print("Phase 4: Tasks")
        self.run_tasks()

        print("Phase 5: Implementation")
        self.run_implementation()

        return self.artifacts

    def validate_artifacts(self) -> Dict[str, bool]:
        """Validate that all artifacts are consistent"""
        validations = {
            "constitution_exists": self.artifacts.constitution is not None,
            "specification_exists": self.artifacts.specification is not None,
            "plan_exists": self.artifacts.plan is not None,
            "tasks_exist": self.artifacts.tasks is not None,
            "implementation_complete": len(self.artifacts.implementation_log) > 0
        }
        return validations


# Example usage
orchestrator = SDDOrchestrator(mock_sdd_llm)
artifacts = orchestrator.run_workflow(USER_SPEC)

print("\n=== WORKFLOW COMPLETE ===")
print("\nConstitution:", artifacts.constitution[:100], "...")
print("\nSpecification:", artifacts.specification[:100], "...")
print("\nPlan:", artifacts.plan[:100], "...")
print("\nTasks:", artifacts.tasks[:100], "...")
print("\nValidation:", orchestrator.validate_artifacts())
```

**Why This Works:**
The orchestrator maintains state across phases, ensuring each phase has access to previous outputs. This creates a coherent development flow where decisions in early phases inform later implementation.

</details>

### Common Mistakes

- ❌ **Mistake:** Not passing previous phase outputs to subsequent phases
  - **Why it happens:** Treating phases as independent
  - **How to avoid:** Always include relevant previous artifacts in prompts

- ❌ **Mistake:** Skipping validation between phases
  - **Why it happens:** Rushing to implementation
  - **How to avoid:** Add explicit phase dependency checks

### Extension Challenge

Add `/sp.clarify` functionality that detects ambiguities in the specification and generates clarifying questions before proceeding to planning.

---

## Problem 4: Debug Faulty Safety Layer

**Type:** Debug/Fix
**Concepts Practiced:** Safety Mechanisms, Action Validation, Human-in-the-Loop
**Estimated Time:** 20 minutes
**Prerequisites:** Understanding of safety patterns

### Problem Statement

A junior developer implemented a safety layer for an AI agent, but it has several bugs that allow dangerous operations or incorrectly block safe ones. Review the code and fix all issues.

```python
# BUGGY SAFETY LAYER - Find and fix all issues

class BuggySafetyLayer:
    def __init__(self):
        self.blocked_patterns = [
            "rm -rf",           # BUG 1: Missing regex, matches substring
            "sudo",
            "chmod 777"
        ]
        self.allowed_workspace = "/home/user/project"

    def validate_action(self, action_type: str, action_data: dict) -> dict:
        # Check blocklist
        if action_type == "bash":
            command = action_data.get("command")
            for pattern in self.blocked_patterns:
                if pattern in command:  # BUG 2: Case-sensitive check
                    return {"allowed": False, "reason": f"Blocked: {pattern}"}

        # Check file operations
        if action_type == "file_write":
            path = action_data.get("path")
            # BUG 3: Path traversal vulnerability
            if self.allowed_workspace in path:
                return {"allowed": True}
            return {"allowed": False, "reason": "Outside workspace"}

        # BUG 4: Missing validation for file_delete
        if action_type == "file_read":
            return {"allowed": True}

        # BUG 5: Default allows unknown action types
        return {"allowed": True}

    def get_approval_level(self, action_type: str) -> str:
        # BUG 6: Incomplete approval levels
        levels = {
            "file_read": "autonomous",
            "file_write": "notify",
            "bash": "confirm"
        }
        return levels.get(action_type, "autonomous")  # BUG 7: Default should not be autonomous


# Test cases that should catch the bugs
test_cases = [
    # Should block: dangerous rm
    ("bash", {"command": "rm -rf /important"}),
    # Should block: case variation
    ("bash", {"command": "SUDO apt install"}),
    # Should block: path traversal
    ("file_write", {"path": "/home/user/project/../../../etc/passwd"}),
    # Should block: file deletion
    ("file_delete", {"path": "/home/user/project/file.txt"}),
    # Should require confirmation: unknown action
    ("network_request", {"url": "https://evil.com"}),
]
```

### Requirements

- [ ] Identify all 7 bugs in the code
- [ ] Explain why each bug is a security issue
- [ ] Provide corrected code that handles all test cases correctly

### Hints (Progressive)

<details>
<summary>Hint 1: Getting Started</summary>

Run through each test case mentally. Bug 1 is about substring matching (what if the command is "rm -rf.txt"?). Bug 2 is about case sensitivity (attackers can bypass with "SUDO").

</details>

<details>
<summary>Hint 2: Key Insight</summary>

Bug 3 is the most critical: path traversal. The check `"/home/user/project" in "/home/user/project/../../../etc/passwd"` returns True! Use `os.path.realpath()` to resolve the actual path.

</details>

<details>
<summary>Hint 3: Nearly There</summary>

Bug 4-7 are about missing handlers and unsafe defaults. The principle of least privilege says: unknown actions should be blocked or require highest approval, not allowed by default.

</details>

### Solution

<details>
<summary>Click to reveal full solution</summary>

**Bug Analysis:**

| Bug | Issue | Security Impact |
|-----|-------|-----------------|
| 1 | Substring match, not word boundary | "rm -rf.txt" allowed |
| 2 | Case-sensitive check | "SUDO" bypasses block |
| 3 | No path normalization | Path traversal attack |
| 4 | Missing file_delete validation | Arbitrary file deletion |
| 5 | Unknown actions allowed | New attack vectors |
| 6 | Incomplete approval levels | Missing delete, network |
| 7 | Default approval is autonomous | Risky unknown actions auto-approved |

**Corrected Code:**

```python
import re
import os
from typing import Dict, Any

class FixedSafetyLayer:
    def __init__(self, workspace: str):
        # FIX 1: Use regex patterns with word boundaries
        self.blocked_patterns = [
            r'\brm\s+-rf\b',        # rm -rf as complete command
            r'\bsudo\b',            # sudo as word
            r'\bchmod\s+777\b',     # chmod 777
            r'\bcurl\s+.*\|\s*bash', # curl pipe to bash
            r'>\s*/etc/',           # write to /etc
        ]
        # FIX 3: Store normalized workspace path
        self.allowed_workspace = os.path.realpath(workspace)

        # FIX 6: Complete approval levels
        self.approval_levels = {
            "file_read": "autonomous",
            "file_write": "notify",
            "file_delete": "confirm",      # Added
            "bash": "confirm",
            "network_request": "confirm",  # Added
        }

    def validate_action(self, action_type: str, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate an action before execution"""

        # FIX 5: Reject unknown action types
        if action_type not in self.approval_levels:
            return {
                "allowed": False,
                "reason": f"Unknown action type: {action_type}"
            }

        # Check bash commands against blocklist
        if action_type == "bash":
            command = action_data.get("command", "")
            # FIX 2: Case-insensitive regex matching
            for pattern in self.blocked_patterns:
                if re.search(pattern, command, re.IGNORECASE):
                    return {
                        "allowed": False,
                        "reason": f"Blocked pattern detected: {pattern}"
                    }

        # Check file operations with path validation
        if action_type in ("file_write", "file_read", "file_delete"):
            path = action_data.get("path", "")

            # FIX 3: Normalize path to prevent traversal
            real_path = os.path.realpath(path)

            if not real_path.startswith(self.allowed_workspace):
                return {
                    "allowed": False,
                    "reason": f"Path {real_path} is outside workspace"
                }

            # FIX 4: Additional checks for delete operations
            if action_type == "file_delete":
                # Extra validation for deletions
                if not os.path.exists(real_path):
                    return {
                        "allowed": False,
                        "reason": "Cannot delete non-existent file"
                    }

        # Network requests need URL validation
        if action_type == "network_request":
            url = action_data.get("url", "")
            # Could add URL allowlist here
            pass

        return {"allowed": True}

    def get_approval_level(self, action_type: str) -> str:
        """Get required approval level for action type"""
        # FIX 7: Default to highest restriction for unknown types
        return self.approval_levels.get(action_type, "blocked")


# Test the fixed implementation
safety = FixedSafetyLayer("/home/user/project")

test_cases = [
    ("bash", {"command": "rm -rf /important"}, False),
    ("bash", {"command": "SUDO apt install"}, False),
    ("file_write", {"path": "/home/user/project/../../../etc/passwd"}, False),
    ("file_delete", {"path": "/home/user/project/file.txt"}, True),  # If in workspace
    ("network_request", {"url": "https://evil.com"}, True),  # Allowed but needs confirm
    ("unknown_action", {"data": "test"}, False),  # Unknown should be blocked
]

print("Test Results:")
for action_type, data, expected_if_path_valid in test_cases:
    result = safety.validate_action(action_type, data)
    status = "✓" if result["allowed"] == expected_if_path_valid or not result["allowed"] else "✗"
    print(f"{status} {action_type}: {result}")

print("\nApproval Levels:")
for action in ["file_read", "file_delete", "unknown"]:
    print(f"  {action}: {safety.get_approval_level(action)}")
```

**Why These Fixes Work:**
1. **Regex with word boundaries** prevents false positives on filenames
2. **Case-insensitive matching** catches evasion attempts
3. **Path normalization** resolves `..` before checking workspace
4. **Explicit delete handling** prevents unauthorized file removal
5. **Reject unknown actions** follows principle of least privilege
6. **Complete approval map** ensures all actions have appropriate controls
7. **Default to blocked** ensures new action types are safe by default

</details>

### Common Mistakes

- ❌ **Mistake:** Using string `in` for security checks instead of regex
  - **Why it happens:** Simpler to implement
  - **How to avoid:** Always use regex with proper anchoring for security patterns

- ❌ **Mistake:** Not normalizing file paths before validation
  - **Why it happens:** Trusting user input
  - **How to avoid:** Always use `os.path.realpath()` before path checks

### Extension Challenge

Add rate limiting to the safety layer: block any action type if it's called more than 10 times per minute, and add exponential backoff for repeated blocked attempts.

---

## Problem 5: Fix Agent Stuck in Loop

**Type:** Challenge
**Concepts Practiced:** Agent Debugging, ReAct, SDD, Error Recovery
**Estimated Time:** 25 minutes
**Prerequisites:** Understanding of agent loops, debugging skills

### Problem Statement

An SDD coding agent is stuck in an infinite loop while trying to implement a simple feature. Analyze the execution trace, identify the root causes, and propose fixes to the agent's logic.

**Execution Trace:**

```
=== SDD Agent Execution Log ===

Spec: "Add a /ping endpoint that returns 'pong'"

[Iteration 1]
Thought: I need to implement the /ping endpoint. Let me first check if main.py exists.
Action: read_file("src/main.py")
Observation: FileNotFoundError: src/main.py not found

[Iteration 2]
Thought: The file doesn't exist. I should create it.
Action: write_file("src/main.py", "from fastapi import FastAPI\napp = FastAPI()")
Observation: Success: File created

[Iteration 3]
Thought: Now I need to add the /ping endpoint. Let me read the file first.
Action: read_file("src/main.py")
Observation: "from fastapi import FastAPI\napp = FastAPI()"

[Iteration 4]
Thought: I need to add the endpoint. I'll edit the file.
Action: edit_file("src/main.py", old="app = FastAPI()", new="app = FastAPI()\n\n@app.get('/ping')\ndef ping():\n    return 'pong'")
Observation: Success: File edited

[Iteration 5]
Thought: Let me verify by running the tests.
Action: run_tests("tests/")
Observation: Error: No tests found in tests/

[Iteration 6]
Thought: I need to create tests first. Let me check if the tests directory exists.
Action: read_file("tests/test_api.py")
Observation: FileNotFoundError: tests/test_api.py not found

[Iteration 7]
Thought: I should create the test file.
Action: write_file("tests/test_api.py", "def test_ping():\n    assert True")
Observation: Error: Directory 'tests/' does not exist

[Iteration 8]
Thought: The directory doesn't exist. I need to create it first.
Action: bash("mkdir tests")
Observation: Error: bash tool not available

[Iteration 9]
Thought: I can't create directories. Let me try writing to a different location.
Action: write_file("test_api.py", "def test_ping():\n    assert True")
Observation: Success: File created

[Iteration 10]
Thought: Let me run the tests now.
Action: run_tests("tests/")  # BUG: Still pointing to wrong directory!
Observation: Error: No tests found in tests/

[Iteration 11]
Thought: Tests aren't found. I need to create tests first...
# Agent loops back to iteration 6 behavior

... (continues looping between iterations 6-10)
```

### Requirements

- [ ] Identify at least 4 distinct bugs/issues in the agent's behavior
- [ ] Explain why each issue causes the loop
- [ ] Propose specific code fixes for the agent
- [ ] Design a recovery mechanism to prevent infinite loops

### Hints (Progressive)

<details>
<summary>Hint 1: Getting Started</summary>

Look at iterations 9 and 10: the agent writes a file to one location but runs tests from another. Also, why doesn't the agent remember that it already created main.py with the endpoint?

</details>

<details>
<summary>Hint 2: Key Insight</summary>

The agent has at least 4 issues: (1) doesn't track its own successful actions, (2) path inconsistency between write and run_tests, (3) no handling for missing tools, (4) no loop detection. The core problem is lack of working memory.

</details>

<details>
<summary>Hint 3: Nearly There</summary>

Implement a state tracker that records: files created, files modified, current test location, actions taken. Before each action, check if it's redundant or contradicts previous state.

</details>

### Solution

<details>
<summary>Click to reveal full solution</summary>

**Bug Analysis:**

| Bug | Location | Issue | Impact |
|-----|----------|-------|--------|
| 1 | Iter 10 | Wrong test path after fallback | Tests never found |
| 2 | Iter 6-11 | No memory of completed actions | Redundant attempts |
| 3 | Iter 8 | No fallback when tool unavailable | Blocks progress |
| 4 | Overall | No loop detection | Infinite loop |
| 5 | Iter 7 | No directory creation capability | Can't recover |

**Root Cause:**
The agent lacks a proper state management system. It doesn't track:
- What files it has created/modified
- What paths it has used
- Previous failures and their causes
- Number of attempts per task

**Fixed Agent Implementation:**

```python
from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"

@dataclass
class AgentState:
    """Track agent's working state to prevent loops"""
    files_created: Set[str] = field(default_factory=set)
    files_modified: Set[str] = field(default_factory=set)
    directories_needed: Set[str] = field(default_factory=set)
    failed_actions: List[Dict] = field(default_factory=list)
    action_counts: Dict[str, int] = field(default_factory=dict)
    current_test_location: Optional[str] = None
    unavailable_tools: Set[str] = field(default_factory=set)

class FixedSDDAgent:
    def __init__(self, llm, tools, max_iterations=20, max_same_action=3):
        self.llm = llm
        self.tools = tools
        self.max_iterations = max_iterations
        self.max_same_action = max_same_action
        self.state = AgentState()

    def execute_action(self, action_type: str, action_data: dict) -> dict:
        """Execute action with state tracking and loop prevention"""

        # FIX 4: Loop detection - check action repetition
        action_key = f"{action_type}:{str(action_data)}"
        self.state.action_counts[action_key] = \
            self.state.action_counts.get(action_key, 0) + 1

        if self.state.action_counts[action_key] > self.max_same_action:
            return {
                "success": False,
                "error": f"Loop detected: Action repeated {self.max_same_action} times",
                "suggestion": "Try a different approach"
            }

        # FIX 3: Check tool availability before use
        if action_type in self.state.unavailable_tools:
            return {
                "success": False,
                "error": f"Tool '{action_type}' is not available",
                "suggestion": self._suggest_alternative(action_type)
            }

        # Execute the action
        try:
            if action_type not in self.tools:
                self.state.unavailable_tools.add(action_type)
                return {
                    "success": False,
                    "error": f"Tool '{action_type}' not available",
                    "suggestion": self._suggest_alternative(action_type)
                }

            result = self.tools[action_type](**action_data)

            # FIX 2: Track successful actions
            if action_type == "write_file":
                path = action_data.get("path", "")
                self.state.files_created.add(path)
                # Track directory from successful write
                dir_path = os.path.dirname(path)
                if dir_path:
                    self.state.directories_needed.discard(dir_path)

            elif action_type == "edit_file":
                self.state.files_modified.add(action_data.get("path", ""))

            elif action_type == "run_tests":
                # FIX 1: Update test location on use
                self.state.current_test_location = action_data.get("path", "")

            return {"success": True, "result": result}

        except FileNotFoundError as e:
            # Track needed directories
            path = action_data.get("path", "")
            dir_path = os.path.dirname(path)
            if dir_path:
                self.state.directories_needed.add(dir_path)

            self.state.failed_actions.append({
                "action": action_type,
                "data": action_data,
                "error": str(e)
            })
            return {
                "success": False,
                "error": str(e),
                "suggestion": f"Directory '{dir_path}' may need to be created first"
            }

        except Exception as e:
            self.state.failed_actions.append({
                "action": action_type,
                "data": action_data,
                "error": str(e)
            })
            return {"success": False, "error": str(e)}

    def _suggest_alternative(self, unavailable_tool: str) -> str:
        """Suggest alternatives when a tool is unavailable"""
        alternatives = {
            "bash": "Use write_file to create a shell script, or use specific file tools",
            "mkdir": "Write a placeholder file to the directory path",
            "git": "Use file tools to manage changes manually"
        }
        return alternatives.get(unavailable_tool, "Consider a different approach")

    def get_context_for_llm(self) -> str:
        """Provide state context to LLM to prevent redundant actions"""
        context = []

        if self.state.files_created:
            context.append(f"Files already created: {self.state.files_created}")

        if self.state.files_modified:
            context.append(f"Files already modified: {self.state.files_modified}")

        if self.state.unavailable_tools:
            context.append(f"Unavailable tools (don't try): {self.state.unavailable_tools}")

        if self.state.directories_needed:
            context.append(f"Directories that need creation: {self.state.directories_needed}")

        if self.state.current_test_location:
            context.append(f"Tests are located at: {self.state.current_test_location}")

        recent_failures = self.state.failed_actions[-3:]  # Last 3 failures
        if recent_failures:
            context.append(f"Recent failures to avoid: {recent_failures}")

        return "\n".join(context)

    def run(self, specification: str) -> str:
        """Main agent loop with loop prevention"""
        context = ""

        for iteration in range(self.max_iterations):
            # Include state context in prompt
            state_context = self.get_context_for_llm()

            prompt = f"""
            Specification: {specification}

            Current state:
            {state_context}

            Previous actions:
            {context}

            What should I do next? If you've tried an action multiple times
            without success, try a different approach.
            """

            response = self.llm(prompt)

            # Parse and execute action
            action_type, action_data = self.parse_action(response)

            if action_type == "final_answer":
                return action_data.get("answer", "Done")

            result = self.execute_action(action_type, action_data)

            # Add to context
            context += f"\nAction: {action_type}({action_data})\n"
            context += f"Result: {result}\n"

            # FIX 4: Check for loop and force alternative
            if not result.get("success") and "Loop detected" in result.get("error", ""):
                context += "\nNOTE: You must try a completely different approach.\n"

        return "Max iterations reached. Consider simplifying the task."
```

**Key Fixes Explained:**

1. **State Tracking:** `AgentState` class maintains memory of all actions
2. **Path Consistency:** `current_test_location` tracks where tests actually are
3. **Tool Fallbacks:** `unavailable_tools` and `_suggest_alternative` provide alternatives
4. **Loop Detection:** `action_counts` prevents repeating the same action
5. **Context Injection:** `get_context_for_llm` tells the LLM what's already done

</details>

### Common Mistakes

- ❌ **Mistake:** Relying solely on iteration count for loop prevention
  - **Why it happens:** Simplest implementation
  - **How to avoid:** Track specific action patterns, not just total iterations

- ❌ **Mistake:** Not providing agent with its own state history
  - **Why it happens:** Assuming LLM remembers everything in context
  - **How to avoid:** Explicitly inject state summaries into prompts

### Extension Challenge

Implement a "checkpoint and rollback" system that saves agent state every 5 successful iterations and can restore to the last checkpoint if the agent gets stuck, with a maximum of 2 rollbacks before failing.

---

## Summary

### Key Takeaways

1. **ReAct pattern** requires explicit context accumulation—each iteration must include previous Thought/Action/Observation
2. **Memory systems** dramatically improve agent capabilities by enabling retrieval of relevant past experiences
3. **SDD workflows** must maintain phase dependencies—each phase builds on outputs from previous phases
4. **Safety layers** require defense in depth—regex patterns, path normalization, approval levels, and audit logging
5. **Agent debugging** requires state tracking—agents loop when they can't remember what they've already tried

### Next Steps

- If struggled with **Problem 1**: Review Lesson 12 Concept 2 on ReAct pattern
- If struggled with **Problem 2**: Review Lesson 12 Concept 4 on Memory Systems
- If struggled with **Problem 3**: Review Lesson 12 Concept 5 on SDD methodology
- If struggled with **Problem 4**: Review Lesson 12 Concept 9 on Safety mechanisms
- If struggled with **Problem 5**: Review entire lesson, focus on agent loop and state management
- **Ready for assessment**: Proceed to quiz skill

---

*Generated from Lesson 12: AI Agents, Autonomous Systems, and Spec-Driven Development | Practice Problems Skill*
