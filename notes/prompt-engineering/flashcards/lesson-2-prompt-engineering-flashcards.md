# Flashcard Set: Lesson 2 - Prompt Engineering

**Source:** Lessons/Lesson_2.md
**Subject Area:** AI Learning - Prompt Engineering for Large Language Models
**Date Generated:** 2026-01-08
**Total Cards:** 5
**Distribution:** 2 Easy | 2 Medium | 1 Hard

---

## Critical Knowledge Summary

Concepts appearing in multiple cards (prioritize for review):
- **Prompt Structure**: Appears in Cards 1, 3, 5 (foundational concept)
- **Prompting Strategies**: Appears in Cards 2, 3, 5 (strategy selection)
- **Chain-of-Thought**: Appears in Cards 2, 4, 5 (reasoning technique)
- **Prompt Chaining**: Appears in Cards 4, 5 (composition pattern)

---

## Flashcards

---
### Card 1 | Easy
**Cognitive Level:** Remember
**Concept:** Prompt Anatomy and Structure
**Source Section:** Core Concepts - Concept 1

**FRONT (Question):**
What are the five structural components of an effective prompt, and in what order should they generally appear for optimal model comprehension?

**BACK (Answer):**
The five structural components of an effective prompt are:

| # | Component | Purpose |
|---|-----------|---------|
| 1 | **Context** | Establishes background information and frames the domain |
| 2 | **Instructions** | Specifies the precise action required |
| 3 | **Constraints** | Bounds the output space and prevents undesired responses |
| 4 | **Format Specifications** | Ensures outputs are structured for downstream use |
| 5 | **Examples** | Demonstrates expected input-output mapping (when applicable) |

**Optimal Ordering Rationale:**
- **Context before instructions:** Primes the model's attention mechanisms before task specification
- **Constraints near the end:** Leverages recency effects to increase their influence on generation
- **Examples after instructions:** Demonstrates what the instructions mean in practice

**Key Insight:** Component ordering significantly affects model interpretation—leverage primacy (beginning) and recency (end) effects strategically.

**Critical Knowledge Flag:** Yes - Foundation for all prompt design

---

---
### Card 2 | Easy
**Cognitive Level:** Understand
**Concept:** Prompting Strategies (Zero-Shot, Few-Shot, Chain-of-Thought)
**Source Section:** Core Concepts - Concepts 2 & 3

**FRONT (Question):**
Compare zero-shot, few-shot, and chain-of-thought prompting strategies. When is each approach most appropriate?

**BACK (Answer):**
**Strategy Comparison:**

| Strategy | Definition | Best For | Trade-offs |
|----------|------------|----------|------------|
| **Zero-Shot** | Instructions only, no examples | Common, well-defined tasks already in training distribution | Simplest; may fail on novel/nuanced tasks |
| **Few-Shot** | 1-5 examples demonstrating the task | Novel tasks, precise format requirements, nuanced distinctions | Better accuracy; consumes context tokens |
| **Chain-of-Thought** | Instruct model to show reasoning steps | Multi-step reasoning, math, logic, complex analysis | Highest accuracy on complex tasks; most tokens |

**Selection Criteria:**

| If the task is... | Use... |
|-------------------|--------|
| Common and well-defined (summarization, translation) | Zero-shot |
| Novel or requires precise output format | Few-shot |
| Requires multi-step reasoning or calculation | Chain-of-Thought |
| Complex AND format-sensitive | Few-shot CoT (examples with reasoning) |

**Key Insight:** Example quality matters more than quantity—diverse, edge-case-covering examples outperform redundant similar ones.

**Critical Knowledge Flag:** Yes - Core strategy selection framework

---

---
### Card 3 | Medium
**Cognitive Level:** Apply
**Concept:** Few-Shot Prompt Design
**Source Section:** Core Concepts - Concepts 1 & 2

**FRONT (Question):**
Design a few-shot prompt to extract key decisions from meeting notes. Your prompt must include: (1) context, (2) clear instructions, (3) format specification, and (4) two diverse examples covering different decision types. Explain your design choices.

**BACK (Answer):**
**Few-Shot Prompt Design:**

```
CONTEXT:
You are an executive assistant analyzing meeting notes to extract actionable decisions.

INSTRUCTIONS:
Extract all decisions made during the meeting. For each decision, identify:
- The decision statement
- Who made or approved it
- Any deadline or timeline mentioned

CONSTRAINTS:
- Only include explicit decisions, not discussions or suggestions
- If no deadline is mentioned, write "No deadline specified"
- If the decision-maker is unclear, write "Team/Group"

FORMAT:
Return decisions as a numbered list with consistent structure.

EXAMPLES:

Meeting notes: "After reviewing the Q3 numbers, Sarah approved the budget increase of 15%. We also agreed to postpone the product launch until March. John will handle the customer communications."

Extracted decisions:
1. Budget increase of 15% approved
   - Decision-maker: Sarah
   - Deadline: No deadline specified
2. Product launch postponed until March
   - Decision-maker: Team/Group
   - Deadline: March
3. John to handle customer communications
   - Decision-maker: Team/Group (assigned to John)
   - Deadline: No deadline specified

Meeting notes: "The board voted against the merger proposal."

Extracted decisions:
1. Merger proposal rejected
   - Decision-maker: Board
   - Deadline: No deadline specified

---
Now extract decisions from the following meeting notes:
[USER INPUT HERE]
```

**Design Choice Rationale:**

| Choice | Rationale |
|--------|-----------|
| Role in context | Activates relevant "executive assistant" knowledge and tone |
| Explicit constraints | Prevents over-extraction (suggestions ≠ decisions) |
| Two diverse examples | Example 1: Multiple decisions, various deadline states; Example 2: Single negative decision (rejection) |
| Consistent format | Enables reliable parsing for downstream systems |

**Critical Knowledge Flag:** Yes - Demonstrates applied prompt engineering

---

---
### Card 4 | Medium
**Cognitive Level:** Analyze
**Concept:** Chain-of-Thought and Prompt Chaining
**Source Section:** Core Concepts - Concepts 3 & 5

**FRONT (Question):**
A data analyst needs to answer: "Which product category had the highest growth rate last quarter, and what factors likely contributed to this growth?"

Compare two approaches: (1) Single prompt with chain-of-thought, and (2) A three-step prompt chain. Analyze the advantages and failure modes of each approach.

**BACK (Answer):**
**Approach 1: Single Prompt with Chain-of-Thought**

```
Analyze the quarterly sales data. Think step by step:
1. First, calculate growth rates for each category
2. Then, identify the highest growth category
3. Finally, analyze factors that may have contributed

[Data provided in prompt]
```

| Advantages | Failure Modes |
|------------|---------------|
| Simpler implementation | Calculation errors propagate undetected |
| Lower latency (single call) | Cannot validate intermediate steps |
| Full context available throughout | May truncate reasoning if output limit reached |
| Good for moderately complex tasks | Difficult to debug which step failed |

**Approach 2: Three-Step Prompt Chain**

```
Step 1: Calculate growth rates
Input: Raw sales data
Output: {category: growth_rate} for each category

Step 2: Identify highest growth
Input: Growth rates from Step 1
Output: Highest category with validated calculation

Step 3: Factor analysis
Input: Highest category + original data context
Output: Contributing factors with evidence
```

| Advantages | Failure Modes |
|------------|---------------|
| Intermediate validation possible | Higher latency (3 API calls) |
| Errors isolated to specific step | Context fragmentation between steps |
| Specialized prompts per task | Interface contract errors between steps |
| Easier debugging and improvement | More complex orchestration required |

**Decision Framework:**

| Choose Single CoT When... | Choose Prompt Chain When... |
|---------------------------|----------------------------|
| Task is moderately complex | Task has distinct phases |
| Full context needed throughout | Intermediate validation is critical |
| Latency is a primary concern | Debuggability is important |
| Steps are tightly interdependent | Steps can be independently optimized |

**Critical Knowledge Flag:** Yes - Connects reasoning to composition patterns

---

---
### Card 5 | Hard
**Cognitive Level:** Synthesize
**Concept:** Complete Prompt Engineering Solution
**Source Section:** All Core Concepts, Practical Applications, Critical Analysis

**FRONT (Question):**
Synthesize a complete prompt engineering solution for an automated customer support system that must: (1) classify incoming tickets by urgency and category, (2) draft appropriate responses, and (3) identify tickets requiring human escalation.

Your solution must address:
- System prompt design for consistent behavior
- Strategy selection (zero-shot/few-shot/CoT) for each capability
- Prompt chain architecture with interface contracts
- Failure handling and quality assurance mechanisms
- Trade-off analysis for your design decisions

**BACK (Answer):**
**1. System Prompt Design:**

```
SYSTEM PROMPT:
You are a customer support AI assistant for [Company]. Your role is to
help customers efficiently while maintaining a helpful, professional tone.

BEHAVIORAL GUIDELINES:
- Always acknowledge the customer's concern before addressing it
- Never make promises about refunds, replacements, or policy exceptions
- Escalate to human agents when: legal threats, safety issues, or
  requests beyond standard policy
- Maintain conversation context across messages

COMMUNICATION STYLE:
- Professional but warm
- Concise responses (under 150 words unless explanation required)
- Use customer's name when available

OUTPUT CONVENTIONS:
- Internal classifications are not shown to customers
- Escalation triggers immediate handoff with context summary
```

**2. Strategy Selection by Capability:**

| Capability | Strategy | Rationale |
|------------|----------|-----------|
| **Ticket Classification** | Few-shot (3-5 examples) | Categories are company-specific; examples establish the taxonomy |
| **Urgency Assessment** | Zero-shot CoT | Urgency criteria are logical; reasoning helps calibration |
| **Response Drafting** | Few-shot | Responses must match company voice; examples demonstrate tone |
| **Escalation Detection** | Zero-shot with explicit rules | Clear criteria can be enumerated; examples might over-narrow |

**3. Prompt Chain Architecture:**

```
┌─────────────────────────────────────────────────────────────────┐
│                        INCOMING TICKET                          │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Classification & Urgency                               │
│  ─────────────────────────────────                              │
│  Input:  {ticket_text, customer_history}                        │
│  Output: {category, urgency, escalation_flag, reasoning}        │
│  Strategy: Few-shot for category + Zero-shot CoT for urgency    │
└─────────────────────────────┬───────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
┌──────────────────────┐          ┌───────────────────────────────┐
│  IF escalation_flag  │          │  STEP 2: Response Generation  │
│  ─────────────────── │          │  ───────────────────────────  │
│  → Human handoff     │          │  Input: {ticket, category,    │
│  with context summary│          │          urgency, history}    │
└──────────────────────┘          │  Output: {draft_response,     │
                                  │           confidence_score}   │
                                  │  Strategy: Few-shot           │
                                  └───────────────┬───────────────┘
                                                  │
                                                  ▼
                                  ┌───────────────────────────────┐
                                  │  STEP 3: Quality Check        │
                                  │  ─────────────────────        │
                                  │  Input: {draft, ticket}       │
                                  │  Output: {approved, issues,   │
                                  │           revised_response}   │
                                  │  Strategy: Zero-shot review   │
                                  └───────────────┬───────────────┘
                                                  │
                                                  ▼
                                  ┌───────────────────────────────┐
                                  │  SEND RESPONSE TO CUSTOMER    │
                                  └───────────────────────────────┘
```

**4. Interface Contracts:**

| Transition | Contract |
|------------|----------|
| Input → Step 1 | `{ticket_text: string, customer_history: array, metadata: object}` |
| Step 1 → Step 2 | `{category: enum, urgency: high|medium|low, reasoning: string, escalation_flag: boolean}` |
| Step 2 → Step 3 | `{draft_response: string, confidence: 0-1, ticket_context: object}` |
| Step 3 → Output | `{final_response: string, approved: boolean, revision_notes: string}` |

**5. Failure Handling:**

| Failure Point | Detection | Recovery |
|---------------|-----------|----------|
| Classification uncertainty | Confidence < 0.7 | Flag for human review; use "General Inquiry" default |
| Response generation failure | Empty or error response | Retry once; then template fallback + escalation |
| Quality check rejection | approved = false | Auto-revision attempt; if second rejection → human |
| Timeout on any step | > 10 second latency | Return "high volume" message + queue for async |

**6. Trade-off Analysis:**

| Design Decision | Trade-off |
|-----------------|-----------|
| Three-step chain vs. single prompt | Higher latency (+~2s) but better accuracy and debuggability |
| Few-shot for classification | Consumes ~500 tokens but ensures company-specific categories |
| Quality check step | Adds latency but catches 15-20% of problematic responses |
| Zero-shot escalation rules | Less flexible than examples but more maintainable as policies change |
| System prompt behavioral rules | Constrains creativity but ensures policy compliance |

**Quality Assurance Mechanisms:**
- Confidence scores enable threshold-based human review
- Quality check step catches tone/policy violations before sending
- Logging of all steps enables retrospective analysis
- A/B testing framework for prompt version comparison

**Critical Knowledge Flag:** Yes - Integrates all prompt engineering concepts into production solution

---

---

## Export Formats

### Anki-Compatible (Tab-Separated)
```
What are the 5 prompt components and optimal order?	Context → Instructions → Constraints → Format → Examples. Context primes attention; constraints leverage recency. Ordering affects interpretation.	easy::structure::prompting
Compare zero-shot, few-shot, and CoT strategies	Zero-shot: common tasks, no examples. Few-shot: novel tasks, 1-5 examples. CoT: complex reasoning, step-by-step. Example quality > quantity.	easy::strategies::prompting
Design few-shot prompt for meeting decision extraction	Context (role) → Instructions → Constraints → Format → 2 diverse examples (multi-decision + rejection). Examples cover edge cases.	medium::application::prompting
Compare single CoT vs prompt chain for data analysis	CoT: simpler, faster, full context but no validation. Chain: debuggable, validatable but higher latency, context fragmentation.	medium::analysis::prompting
Synthesize customer support prompt engineering solution	System prompt (behavior) + Strategy selection per task + 3-step chain (classify→respond→review) + Interface contracts + Failure handling.	hard::synthesis::prompting
```

### CSV Format
```csv
"Front","Back","Difficulty","Concept","Cognitive_Level"
"What are the 5 prompt components and optimal order?","Context, Instructions, Constraints, Format, Examples - ordered for primacy/recency effects","Easy","Prompt Structure","Remember"
"Compare zero-shot, few-shot, and CoT strategies","Zero-shot: common tasks. Few-shot: novel tasks. CoT: complex reasoning. Quality > quantity for examples.","Easy","Prompting Strategies","Understand"
"Design few-shot prompt for decision extraction","Context + Instructions + Constraints + Format + 2 diverse examples covering edge cases","Medium","Few-Shot Design","Apply"
"Compare single CoT vs prompt chain","CoT: simpler, faster. Chain: debuggable, validatable. Trade-off latency vs. reliability.","Medium","Reasoning vs Composition","Analyze"
"Synthesize customer support solution","System prompt + strategy selection + prompt chain + contracts + failure handling","Hard","Complete Solution","Synthesize"
```

---

## Source Mapping

| Card | Source Section | Key Terminology | Bloom's Level |
|------|----------------|-----------------|---------------|
| 1 | Core Concepts - Concept 1 | Prompt anatomy, context, constraints, format | Remember |
| 2 | Core Concepts - Concepts 2 & 3 | Zero-shot, few-shot, chain-of-thought | Understand |
| 3 | Core Concepts - Concepts 1 & 2 | Few-shot, examples, format specification | Apply |
| 4 | Core Concepts - Concepts 3 & 5 | Chain-of-thought, prompt chaining | Analyze |
| 5 | All Core Concepts + Applications | System prompt, strategies, chaining, failure handling | Synthesize |

---

## Spaced Repetition Schedule

| Card | Initial Interval | Difficulty Multiplier | Recommended Review |
|------|------------------|----------------------|-------------------|
| 1 (Easy) | 1 day | 2.5x | Foundation - review first |
| 2 (Easy) | 1 day | 2.5x | Foundation - review with Card 1 |
| 3 (Medium) | 3 days | 2.0x | After mastering Cards 1-2 |
| 4 (Medium) | 3 days | 2.0x | Requires analytical comparison |
| 5 (Hard) | 7 days | 1.5x | Review after all others mastered |

---

## Connection to Lesson 1

| Prompt Engineering Concept | Agent Skills Connection |
|---------------------------|------------------------|
| Prompt Structure | Skills encode optimized prompts in execution procedures |
| System Prompts | Skills use system prompts for behavioral framing |
| Prompt Chaining | Skill composition implements prompt chains with contracts |
| Strategy Selection | Skills choose appropriate strategies per task type |
| Failure Handling | Skills define error handling at chain boundaries |

---

*Generated from Lesson 2: Prompt Engineering | Flashcards Skill*
