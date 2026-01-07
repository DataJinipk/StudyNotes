# Practice Problems: Lesson 2 - Prompt Engineering

**Source:** Lessons/Lesson_2.md
**Concept Map Reference:** notes/prompt-engineering/concept-maps/lesson-2-prompt-engineering-concept-map.md
**Flashcard Reference:** notes/prompt-engineering/flashcards/lesson-2-prompt-engineering-flashcards.md
**Date Generated:** 2026-01-08
**Total Problems:** 5
**Estimated Total Time:** 90-120 minutes
**Distribution:** 1 Warm-Up | 2 Skill-Builder | 1 Challenge | 1 Debug/Fix

---

## Overview

### Concepts Practiced
| Concept | Problems | Mastery Indicator |
|---------|----------|-------------------|
| Prompt Structure | P1, P5 | Can identify and construct all five components |
| Few-Shot Prompting | P2, P4 | Can design effective examples with edge case coverage |
| Chain-of-Thought | P3, P4 | Can elicit reasoning for complex tasks |
| Prompt Chaining | P4 | Can decompose and sequence multi-step workflows |
| Prompt Debugging | P5 | Can diagnose and fix common prompt failures |

### Recommended Approach
1. Attempt each problem before looking at hints
2. Use hints progressively—don't skip to solution
3. After solving, read solution to compare approaches
4. Review Common Mistakes even if you solved correctly
5. Attempt Extension Challenges for deeper mastery

### Self-Assessment Guide
| Problems Solved (no hints) | Mastery Level | Recommendation |
|---------------------------|---------------|----------------|
| 5/5 | Expert | Ready for production prompt development |
| 4/5 | Proficient | Review one gap area |
| 3/5 | Developing | More practice recommended |
| 2/5 or below | Foundational | Re-review Lesson 2 study notes first |

---

## Problems

---

## Problem 1: Prompt Structure Analysis and Construction

**Type:** Warm-Up
**Concepts Practiced:** Prompt Anatomy, Five Components
**Estimated Time:** 15 minutes
**Prerequisites:** Understanding of prompt structure from Lesson 2

### Problem Statement

You are given the following poorly structured prompt that a developer wrote to extract product information from customer reviews:

```
Get the product details from this review. I need the name and what features
people mentioned. Also tell me if they liked it or not. Here's the review:
"I bought the XPhone Pro last month. The camera is amazing but battery life
is disappointing. Overall I'd recommend it for photography enthusiasts."
```

**Tasks:**
1. Identify which of the five prompt components are present, missing, or poorly implemented
2. Rewrite the prompt with all five components properly structured
3. Explain how your restructured prompt improves reliability

### Requirements

- [ ] Analyze original prompt identifying present/missing components
- [ ] Construct improved prompt with all five components
- [ ] Include at least one example demonstrating expected output
- [ ] Explain how each added component improves the prompt

### Hints (Progressive)

<details>
<summary>Hint 1: Getting Started</summary>

The five components are: Context, Instructions, Constraints, Format Specification, and Examples.

Check the original prompt against each:
- Is there role/domain context?
- Are instructions clear and specific?
- Are there constraints on what NOT to include?
- Is the output format specified?
- Are there examples?

</details>

<details>
<summary>Hint 2: Key Insight</summary>

The original prompt has:
- ✗ No context (who is extracting this? why?)
- △ Vague instructions ("get details", "what features")
- ✗ No constraints (should it include speculation? only explicit mentions?)
- ✗ No format specification (JSON? bullet points? prose?)
- ✗ No examples

Each missing component introduces ambiguity that causes inconsistent outputs.

</details>

<details>
<summary>Hint 3: Nearly There</summary>

Structure your improved prompt as:

```
CONTEXT: [Role and purpose]

INSTRUCTIONS: [Specific extraction requirements]

CONSTRAINTS:
- [What to exclude]
- [Edge case handling]

FORMAT:
[Exact output structure with field names]

EXAMPLE:
Input: [sample review]
Output: [expected extraction]

---
Now extract from: [actual review]
```

</details>

### Solution

<details>
<summary>Click to reveal full solution</summary>

**Component Analysis of Original Prompt:**

| Component | Status | Issue |
|-----------|--------|-------|
| **Context** | ❌ Missing | No role, domain, or purpose framing |
| **Instructions** | ⚠️ Vague | "Get details" is ambiguous; "what features" unclear scope |
| **Constraints** | ❌ Missing | No guidance on handling implicit vs explicit info |
| **Format** | ❌ Missing | No output structure specified |
| **Examples** | ❌ Missing | No demonstration of expected output |

**Restructured Prompt:**

```
CONTEXT:
You are a product intelligence analyst extracting structured data from
customer reviews for a retail analytics dashboard.

INSTRUCTIONS:
Extract the following information from the customer review:
1. Product name (exact name as mentioned)
2. Features mentioned (positive and negative separately)
3. Overall sentiment (positive, negative, or mixed)
4. Purchase recommendation (if stated)

CONSTRAINTS:
- Only extract explicitly mentioned information; do not infer unstated features
- If a feature sentiment is unclear, classify it as "neutral"
- If product name is not explicitly stated, use "Unknown Product"
- Do not include reviewer opinions about price unless specifically mentioned

FORMAT:
Return a JSON object with this structure:
{
  "product_name": "string",
  "features": {
    "positive": ["feature1", "feature2"],
    "negative": ["feature1"],
    "neutral": []
  },
  "sentiment": "positive|negative|mixed",
  "recommendation": "yes|no|not stated"
}

EXAMPLE:
Input: "Love my new SoundMax headphones! Great bass and comfortable fit,
though the cable is too short. Would definitely buy again."

Output:
{
  "product_name": "SoundMax headphones",
  "features": {
    "positive": ["bass quality", "comfortable fit"],
    "negative": ["cable length"],
    "neutral": []
  },
  "sentiment": "positive",
  "recommendation": "yes"
}

---
Now extract from the following review:
"I bought the XPhone Pro last month. The camera is amazing but battery life
is disappointing. Overall I'd recommend it for photography enthusiasts."
```

**Expected Output:**
```json
{
  "product_name": "XPhone Pro",
  "features": {
    "positive": ["camera"],
    "negative": ["battery life"],
    "neutral": []
  },
  "sentiment": "mixed",
  "recommendation": "yes"
}
```

**How Each Component Improves Reliability:**

| Component | Reliability Improvement |
|-----------|------------------------|
| **Context** | Activates product analysis knowledge; establishes professional tone |
| **Instructions** | Specific numbered tasks eliminate ambiguity about what to extract |
| **Constraints** | "Only explicit" prevents hallucination; edge case handling ensures consistency |
| **Format** | JSON schema ensures parseable, consistent output structure |
| **Example** | Demonstrates exactly how to categorize features and determine sentiment |

</details>

### Common Mistakes

- ❌ **Mistake:** Adding context but making it too generic ("You are a helpful assistant")
  - **Why it happens:** Defaulting to common patterns
  - **How to avoid:** Context should be specific to the domain and task

- ❌ **Mistake:** Example doesn't cover edge cases (e.g., only positive review example)
  - **Why it happens:** Choosing easiest example to write
  - **How to avoid:** Example should demonstrate handling of mixed/complex cases

- ❌ **Mistake:** Format specification is prose instead of schema
  - **Why it happens:** Writing format as description rather than template
  - **How to avoid:** Show exact structure with field names and types

### Extension Challenge

Modify your prompt to handle reviews that mention multiple products. How would you adjust the format specification and what additional constraints would you need?

---

---

## Problem 2: Few-Shot Prompt Design for Classification

**Type:** Skill-Builder
**Concepts Practiced:** Few-Shot Prompting, Example Selection, Edge Cases
**Estimated Time:** 20 minutes
**Prerequisites:** Understanding of few-shot prompting and in-context learning

### Problem Statement

Design a few-shot prompt for classifying customer support tickets into one of five categories:

- **Billing**: Payment issues, refunds, subscription problems
- **Technical**: Product bugs, errors, functionality issues
- **Account**: Login problems, password reset, profile updates
- **Shipping**: Delivery tracking, address changes, lost packages
- **General**: Product questions, feature requests, other inquiries

Your prompt must include 5 diverse examples that:
1. Cover all five categories
2. Include at least one ambiguous case that could fit multiple categories
3. Demonstrate consistent classification logic

### Requirements

- [ ] Complete prompt with context, instructions, and format
- [ ] Five examples covering all categories
- [ ] At least one edge case demonstrating disambiguation
- [ ] Explanation of example selection rationale

### Hints (Progressive)

<details>
<summary>Hint 1: Getting Started</summary>

Structure your prompt with:
1. Context establishing the classification task
2. Category definitions (so model understands boundaries)
3. Clear instruction for single-label classification
4. Format showing input/output pattern
5. Five diverse examples

Start with clear-cut examples for each category before tackling ambiguous cases.

</details>

<details>
<summary>Hint 2: Key Insight</summary>

The hardest part is the ambiguous example. Consider cases like:
- "Can't log in to see my invoice" → Account or Billing?
- "My order says delivered but I didn't get it" → Shipping or Technical?

Your example should show how to resolve ambiguity consistently (e.g., primary user intent, first mentioned issue).

</details>

<details>
<summary>Hint 3: Nearly There</summary>

Example selection rationale should address:
- Why these examples? (Category coverage)
- Why this ambiguous case? (Common confusion pattern)
- How does resolution demonstrate the rule? (Primary intent principle)

Include a brief disambiguation guideline in your instructions.

</details>

### Solution

<details>
<summary>Click to reveal full solution</summary>

**Few-Shot Classification Prompt:**

```
CONTEXT:
You are a customer support ticket router. Your task is to classify incoming
tickets into exactly one category to ensure they reach the right team.

CATEGORIES:
- Billing: Payment processing, charges, refunds, subscription management
- Technical: Product bugs, errors, crashes, feature malfunctions
- Account: Authentication, login, password, profile, security settings
- Shipping: Delivery, tracking, addresses, package status
- General: Product information, feature requests, feedback, other

INSTRUCTIONS:
Classify each ticket into exactly ONE category. When a ticket could fit
multiple categories, classify based on the PRIMARY issue the customer
needs resolved (what must be fixed for them to be satisfied).

FORMAT:
Ticket: [customer message]
Category: [single category name]
Reasoning: [one sentence explaining classification]

EXAMPLES:

Ticket: "I was charged twice for my subscription this month."
Category: Billing
Reasoning: Primary issue is an incorrect charge that needs refund/correction.

Ticket: "The app crashes every time I try to upload a photo."
Category: Technical
Reasoning: Primary issue is a bug causing functionality failure.

Ticket: "I forgot my password and the reset email never arrives."
Category: Account
Reasoning: Primary issue is authentication access, even though email delivery
is involved.

Ticket: "My package shows delivered but I never received it."
Category: Shipping
Reasoning: Primary issue is package delivery status, not a technical tracking
error.

Ticket: "I can't access my billing history because the page won't load."
Category: Technical
Reasoning: Although billing-related, the PRIMARY issue is a technical bug
preventing page load—once fixed, billing access resolves automatically.

---
Now classify the following ticket:
[USER INPUT]
```

**Example Selection Rationale:**

| Example | Purpose | Design Decision |
|---------|---------|-----------------|
| 1 (Billing) | Clear-cut billing case | Establishes baseline for payment issues |
| 2 (Technical) | Clear-cut technical case | Shows bug/crash pattern |
| 3 (Account) | Edge case: Account + email issue | Demonstrates that authentication is primary even when delivery is involved |
| 4 (Shipping) | Edge case: Could seem like tech error | Shows that delivery issues stay in Shipping even if tracking might be wrong |
| 5 (Ambiguous) | **Critical:** Billing + Technical overlap | Demonstrates the "primary issue" principle—what needs fixing for resolution |

**Why Example 5 is Critical:**

This example explicitly shows how to resolve the most common ambiguity pattern: when users can't access something (Technical) related to a specific function (Billing/Account/etc.). The reasoning demonstrates that:
- Technical = something is broken and needs fixing
- Other categories = something is wrong with the user's actual data/account/order

The "page won't load" is a bug; the fact that it's the billing page is incidental.

**Disambiguation Principle Applied:**
> "What must be FIXED for the customer to be satisfied?"
> - Fix a bug → Technical
> - Fix incorrect data/charges → Billing
> - Fix access credentials → Account
> - Fix delivery issue → Shipping

</details>

### Common Mistakes

- ❌ **Mistake:** All examples are clear-cut; no ambiguous cases
  - **Why it happens:** Easier to write; avoids complexity
  - **How to avoid:** Intentionally include the hardest classification case you can think of

- ❌ **Mistake:** Reasoning in examples is too brief or missing
  - **Why it happens:** Assuming model will infer the logic
  - **How to avoid:** Explicit reasoning teaches the decision principle

- ❌ **Mistake:** Examples are too similar in structure/length
  - **Why it happens:** Copy-paste pattern
  - **How to avoid:** Vary ticket lengths, writing styles, and complexity

### Extension Challenge

Extend your prompt to output a confidence score (high/medium/low) for each classification. Add an example where confidence should be "low" and explain how you would handle low-confidence tickets in a production system.

---

---

## Problem 3: Chain-of-Thought for Complex Analysis

**Type:** Skill-Builder
**Concepts Practiced:** Chain-of-Thought Prompting, Reasoning Elicitation
**Estimated Time:** 20 minutes
**Prerequisites:** Understanding of CoT and its benefits for complex tasks

### Problem Statement

A financial analyst needs to determine whether a company is a good investment target based on a summary of financial metrics. The task requires multi-step reasoning:

1. Analyze revenue trend
2. Evaluate profitability metrics
3. Assess debt levels
4. Consider growth indicators
5. Synthesize into recommendation

Design a chain-of-thought prompt that:
1. Elicits explicit step-by-step reasoning
2. Ensures all four analysis areas are covered
3. Produces a justified final recommendation

**Test Input:**
```
Company: TechFlow Inc.
Revenue: $45M (2023), $38M (2022), $29M (2021)
Profit Margin: 12% (industry average: 15%)
Debt-to-Equity: 0.8 (industry average: 0.5)
R&D Spending: 22% of revenue (industry average: 12%)
Customer Growth: 40% YoY
```

### Requirements

- [ ] Prompt elicits explicit reasoning for each analysis area
- [ ] Structure ensures no analysis area is skipped
- [ ] Final recommendation is clearly tied to preceding analysis
- [ ] Demonstrate expected output format

### Hints (Progressive)

<details>
<summary>Hint 1: Getting Started</summary>

You can elicit CoT in two ways:
1. **Zero-shot CoT:** Simply add "Let's analyze this step by step"
2. **Structured CoT:** Provide a template with explicit reasoning sections

For financial analysis, structured CoT is better because it ensures all areas are covered.

</details>

<details>
<summary>Hint 2: Key Insight</summary>

Structure your prompt to require analysis in each area BEFORE the recommendation:

```
For each area, state:
- The data points you're examining
- What they indicate (positive/negative/neutral)
- How they compare to benchmarks

Only after completing all four areas, provide your recommendation with
explicit references to your analysis.
```

This prevents the model from jumping to a conclusion and backfilling justification.

</details>

<details>
<summary>Hint 3: Nearly There</summary>

Add a "synthesis" step before the final recommendation:

```
STEP 5 - SYNTHESIS:
Count: X positive indicators, Y negative indicators, Z neutral
Weight assessment: Which factors are most material for this company type?
Overall profile: [growth/stable/distressed] company with [strengths] but [concerns]

RECOMMENDATION:
Based on the above analysis... [explicit ties to steps 1-4]
```

</details>

### Solution

<details>
<summary>Click to reveal full solution</summary>

**Chain-of-Thought Investment Analysis Prompt:**

```
CONTEXT:
You are a financial analyst evaluating potential investment targets. Your
analysis must be systematic, considering multiple factors before reaching
a conclusion.

INSTRUCTIONS:
Analyze the company data below using the following structured approach.
Complete EACH step fully before proceeding to the next. Your reasoning
in each step should be explicit and reference specific data points.

ANALYSIS FRAMEWORK:

STEP 1 - REVENUE TREND ANALYSIS:
- Calculate year-over-year growth rates
- Identify the trend direction (accelerating, stable, decelerating)
- Assessment: [Positive/Negative/Neutral] with explanation

STEP 2 - PROFITABILITY ANALYSIS:
- Compare profit margin to industry average
- Identify gap and its significance
- Assessment: [Positive/Negative/Neutral] with explanation

STEP 3 - FINANCIAL HEALTH ANALYSIS:
- Evaluate debt-to-equity vs industry benchmark
- Assess risk level implications
- Assessment: [Positive/Negative/Neutral] with explanation

STEP 4 - GROWTH INDICATORS ANALYSIS:
- Evaluate R&D investment relative to industry
- Consider customer growth metrics
- Assessment: [Positive/Negative/Neutral] with explanation

STEP 5 - SYNTHESIS:
- Count: [X] Positive, [Y] Negative, [Z] Neutral assessments
- Identify the MOST MATERIAL factors for this company profile
- Determine if positive factors outweigh concerns or vice versa

STEP 6 - RECOMMENDATION:
State: INVEST / DO NOT INVEST / REQUIRES FURTHER ANALYSIS
Provide 2-3 sentence justification explicitly referencing findings from
Steps 1-4 that most influenced this decision.

---
COMPANY DATA:
Company: TechFlow Inc.
Revenue: $45M (2023), $38M (2022), $29M (2021)
Profit Margin: 12% (industry average: 15%)
Debt-to-Equity: 0.8 (industry average: 0.5)
R&D Spending: 22% of revenue (industry average: 12%)
Customer Growth: 40% YoY

Begin your analysis:
```

**Expected Output:**

```
STEP 1 - REVENUE TREND ANALYSIS:
- YoY Growth: 2022→2023: 18.4%, 2021→2022: 31.0%
- Trend: Decelerating growth (31% → 18%), but still positive double-digits
- Assessment: POSITIVE - Strong revenue growth despite deceleration;
  $16M increase over 2 years represents healthy expansion

STEP 2 - PROFITABILITY ANALYSIS:
- Profit margin 12% vs industry 15% = 3 percentage point gap
- Gap represents 20% lower profitability than peers
- Assessment: NEGATIVE - Below-average profitability suggests operational
  inefficiencies or pricing pressure; concerning but not critical

STEP 3 - FINANCIAL HEALTH ANALYSIS:
- Debt-to-equity 0.8 vs industry 0.5 = 60% higher leverage
- Elevated debt increases financial risk and interest burden
- Assessment: NEGATIVE - Higher than average leverage is concerning,
  especially combined with below-average profitability

STEP 4 - GROWTH INDICATORS ANALYSIS:
- R&D at 22% vs industry 12% = 83% higher investment in innovation
- Customer growth 40% YoY is exceptional
- Assessment: POSITIVE - Heavy R&D investment and strong customer
  acquisition suggest company is prioritizing growth over current profits

STEP 5 - SYNTHESIS:
- Count: 2 Positive, 2 Negative, 0 Neutral
- Most material factors: For a growth-stage tech company, customer
  acquisition (40% YoY) and R&D investment are leading indicators;
  current profitability is less material than trajectory
- Profile: Growth company with strong top-line expansion, deliberately
  trading current margins for market position

STEP 6 - RECOMMENDATION:
INVEST (with monitoring conditions)

Justification: Despite below-average profitability (Step 2) and elevated
debt (Step 3), TechFlow exhibits characteristics of a successful growth
company: strong revenue expansion (Step 1), exceptional customer growth,
and heavy R&D investment (Step 4). The elevated debt is likely funding
growth initiatives. Recommend investment with quarterly monitoring of
debt levels and path to profitability improvement.
```

**Why This Structure Works:**

| Design Element | Benefit |
|----------------|---------|
| Numbered steps | Forces sequential analysis; prevents skipping |
| Assessment per step | Requires commitment before moving on |
| Explicit synthesis | Weighs factors before conclusion |
| Reference requirement | Prevents unsupported recommendations |
| Framework in prompt | Ensures consistent output structure |

</details>

### Common Mistakes

- ❌ **Mistake:** Using zero-shot CoT ("think step by step") for structured analysis
  - **Why it happens:** Zero-shot is simpler
  - **How to avoid:** Structured tasks need structured templates

- ❌ **Mistake:** Steps don't require explicit assessment before proceeding
  - **Why it happens:** Trusting model to self-organize
  - **How to avoid:** Require [Positive/Negative/Neutral] commitment per step

- ❌ **Mistake:** Recommendation doesn't reference prior analysis
  - **Why it happens:** Model generates conclusion independently
  - **How to avoid:** Explicit instruction to reference "findings from Steps 1-4"

### Extension Challenge

Modify your prompt to handle cases where data is missing (e.g., no profit margin provided). How would you adjust the instructions to handle incomplete information gracefully while still producing useful analysis?

---

---

## Problem 4: Complete Prompt Chain Design

**Type:** Challenge
**Concepts Practiced:** Prompt Chaining, Task Decomposition, Interface Contracts
**Estimated Time:** 30 minutes
**Prerequisites:** All Lesson 2 concepts

### Problem Statement

Design a complete prompt chain for **automated research report generation**. The system takes a research topic and produces a structured report with:
- Executive summary
- Key findings (at least 5)
- Evidence analysis
- Conclusions and recommendations

Your chain must:
1. Decompose into 3-4 discrete steps with clear single responsibilities
2. Define interface contracts between each step
3. Specify the prompting strategy for each step (zero-shot, few-shot, CoT)
4. Include failure handling for each transition
5. Address how context is preserved across the chain

### Requirements

- [ ] Design 3-4 step chain with single-responsibility steps
- [ ] Write complete prompts for at least 2 steps
- [ ] Specify input/output contracts for all steps
- [ ] Justify prompting strategy selection per step
- [ ] Define failure modes and recovery strategies

### Hints (Progressive)

<details>
<summary>Hint 1: Getting Started</summary>

Consider this decomposition:
1. **Research Gathering**: Extract key information from provided sources
2. **Analysis**: Identify patterns, themes, and significance
3. **Synthesis**: Organize findings into coherent structure
4. **Report Generation**: Produce final formatted report

Each step should have a clear input and output that enables the next step.

</details>

<details>
<summary>Hint 2: Key Insight</summary>

Strategy selection per step:
- Information extraction → Few-shot (to establish what "key information" looks like)
- Pattern analysis → CoT (requires reasoning about relationships)
- Synthesis/organization → Zero-shot with structure (clear task, no examples needed)
- Report writing → Few-shot (to establish tone and format conventions)

Interface contracts should specify:
- Required fields
- Data types
- Validation criteria

</details>

<details>
<summary>Hint 3: Nearly There</summary>

Failure handling per transition:
- Step 1 fails: Not enough sources → Request additional input
- Step 2 fails: No patterns found → Lower threshold or human review
- Step 3 fails: Insufficient findings → Merge with step 2 output
- Step 4 fails: Format issues → Retry with stricter template

Context preservation:
- Original topic passed through all steps
- Source citations maintained in metadata
- Confidence scores propagate to final report

</details>

### Solution

<details>
<summary>Click to reveal full solution</summary>

**Research Report Generation Chain Architecture:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INPUT: Topic + Source Documents                   │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1: Information Extraction                                      │
│  Strategy: Few-Shot                                                  │
│  Purpose: Extract key facts, claims, and data from sources          │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 2: Pattern Analysis                                            │
│  Strategy: Chain-of-Thought                                          │
│  Purpose: Identify themes, relationships, and significance          │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 3: Synthesis & Organization                                    │
│  Strategy: Zero-Shot with Structure                                  │
│  Purpose: Organize into report sections with logical flow           │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 4: Report Generation                                           │
│  Strategy: Few-Shot                                                  │
│  Purpose: Produce polished, formatted final report                  │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    OUTPUT: Complete Research Report                  │
└─────────────────────────────────────────────────────────────────────┘
```

**Interface Contracts:**

| Transition | Contract |
|------------|----------|
| Input → Step 1 | `{topic: string, sources: [{title, content, url}], requirements: object}` |
| Step 1 → Step 2 | `{topic: string, extractions: [{fact, source_id, confidence, category}], source_metadata: array}` |
| Step 2 → Step 3 | `{topic: string, findings: [{theme, evidence[], significance, confidence}], patterns: array}` |
| Step 3 → Step 4 | `{topic: string, structure: {exec_summary_points[], sections[], conclusions[]}, citations: array}` |
| Step 4 → Output | `{report: markdown_string, metadata: {word_count, citation_count, confidence_score}}` |

---

**STEP 1 PROMPT: Information Extraction (Few-Shot)**

```
CONTEXT:
You are a research analyst extracting key information from source documents.
Your extractions will feed into a larger analysis pipeline, so accuracy and
source attribution are critical.

INSTRUCTIONS:
For each source document, extract:
1. Key facts and claims (with direct quotes where possible)
2. Data points and statistics
3. Expert opinions or conclusions
4. Methodology notes (if applicable)

CONSTRAINTS:
- Only extract explicitly stated information; no inference
- Assign confidence: HIGH (direct quote), MEDIUM (clear paraphrase), LOW (implied)
- Categorize each extraction: fact, statistic, opinion, methodology
- If source contains no relevant information, return empty extraction with note

FORMAT:
{
  "extractions": [
    {
      "content": "extracted text or data",
      "category": "fact|statistic|opinion|methodology",
      "source_id": "source identifier",
      "confidence": "high|medium|low",
      "page_reference": "if applicable"
    }
  ],
  "source_metadata": {
    "source_id": "identifier",
    "relevance_score": 0-1,
    "notes": "any quality concerns"
  }
}

EXAMPLE:
Source: "According to the 2024 Industry Report, cloud adoption increased 34%
year-over-year. Dr. Smith, lead analyst, noted that 'security concerns remain
the primary barrier to adoption for 67% of enterprises surveyed.'"

Extraction:
{
  "extractions": [
    {
      "content": "cloud adoption increased 34% year-over-year",
      "category": "statistic",
      "source_id": "2024-industry-report",
      "confidence": "high",
      "page_reference": null
    },
    {
      "content": "security concerns remain the primary barrier to adoption for 67% of enterprises surveyed",
      "category": "statistic",
      "source_id": "2024-industry-report",
      "confidence": "high",
      "page_reference": null
    },
    {
      "content": "Dr. Smith characterizes security as the 'primary barrier'",
      "category": "opinion",
      "source_id": "2024-industry-report",
      "confidence": "high",
      "page_reference": null
    }
  ],
  "source_metadata": {
    "source_id": "2024-industry-report",
    "relevance_score": 0.9,
    "notes": "Industry report, likely reliable but verify methodology"
  }
}

---
TOPIC: {topic}

SOURCES:
{sources}

Extract key information from each source:
```

---

**STEP 2 PROMPT: Pattern Analysis (Chain-of-Thought)**

```
CONTEXT:
You are a research analyst identifying patterns and themes across extracted
information. Your analysis will form the basis of research findings.

INSTRUCTIONS:
Analyze the extractions below using this structured approach:

PHASE 1 - CATEGORIZATION:
Group extractions by topic/theme. List each theme and its supporting extractions.

PHASE 2 - PATTERN IDENTIFICATION:
For each theme, identify:
- Consensus points (multiple sources agree)
- Contradictions (sources disagree)
- Gaps (important aspects not covered)

PHASE 3 - SIGNIFICANCE ASSESSMENT:
For each pattern, explain:
- Why this pattern matters for the research topic
- Confidence level based on evidence strength
- Any caveats or limitations

PHASE 4 - FINDING FORMULATION:
Convert patterns into 5-7 discrete findings, each with:
- Clear statement
- Supporting evidence (with source references)
- Significance rating (high/medium/low)

FORMAT:
{
  "themes": [
    {
      "name": "theme name",
      "extraction_ids": ["id1", "id2"],
      "consensus": "what sources agree on",
      "contradictions": "where sources disagree",
      "gaps": "what's missing"
    }
  ],
  "findings": [
    {
      "statement": "clear finding statement",
      "evidence": ["supporting extraction references"],
      "significance": "high|medium|low",
      "confidence": "high|medium|low",
      "caveats": "any limitations"
    }
  ]
}

---
TOPIC: {topic}

EXTRACTIONS:
{extractions_from_step_1}

Begin your analysis with Phase 1:
```

---

**Strategy Justification:**

| Step | Strategy | Rationale |
|------|----------|-----------|
| **Step 1** | Few-Shot | Extraction format must be precise and consistent; examples demonstrate the level of granularity expected |
| **Step 2** | Chain-of-Thought | Pattern identification requires reasoning about relationships across sources; explicit phases prevent jumping to conclusions |
| **Step 3** | Zero-Shot + Structure | Organization is a well-defined task; structure template sufficient without examples |
| **Step 4** | Few-Shot | Report tone and format need demonstration; examples establish professional conventions |

---

**Failure Handling:**

| Step | Failure Mode | Detection | Recovery |
|------|--------------|-----------|----------|
| **Step 1** | Sources contain no relevant info | All extractions empty or low-confidence | Request additional sources; lower relevance threshold |
| **Step 1** | Extraction hallucination | Confidence scores inconsistent with content | Re-run with stricter "explicit only" emphasis |
| **Step 2** | No patterns found | Fewer than 3 findings produced | Lower significance threshold; accept weaker patterns |
| **Step 2** | Contradictory findings | Findings conflict with each other | Flag for human review; include contradiction in report |
| **Step 3** | Structure incomplete | Missing required sections | Retry with emphasis on missing sections |
| **Step 4** | Format violation | Output not valid markdown | Retry with stricter template; fall back to structured JSON |

---

**Context Preservation:**

```json
{
  "chain_context": {
    "topic": "original research topic",
    "initiated": "timestamp",
    "source_ids": ["maintained throughout"],
    "confidence_propagation": {
      "step1_confidence": 0.85,
      "step2_confidence": 0.78,
      "step3_confidence": 0.80,
      "final_confidence": 0.76  // product of chain
    }
  },
  "metadata_passthrough": [
    "source_citations",
    "extraction_confidence_scores",
    "finding_evidence_links"
  ]
}
```

</details>

### Common Mistakes

- ❌ **Mistake:** Steps have overlapping responsibilities
  - **Why it happens:** Incomplete decomposition thinking
  - **How to avoid:** Each step should have exactly one clear output type

- ❌ **Mistake:** Interface contracts are too loose ("pass relevant info")
  - **Why it happens:** Avoiding specification effort
  - **How to avoid:** Define exact fields, types, and validation criteria

- ❌ **Mistake:** No strategy justification—same approach for all steps
  - **Why it happens:** Not analyzing each step's requirements
  - **How to avoid:** Ask "what would make this step fail?" for each

### Extension Challenge

Add a "Quality Check" step between Steps 3 and 4 that validates the synthesized structure against the original extractions, ensuring no key findings were lost and all claims are traceable to sources. Design the prompt and interface contracts.

---

---

## Problem 5: Prompt Debugging and Repair

**Type:** Debug/Fix
**Concepts Practiced:** Prompt Failure Diagnosis, Component Analysis, Repair Strategies
**Estimated Time:** 20 minutes
**Prerequisites:** Understanding of prompt structure and common failure modes

### Problem Statement

A developer's prompt for generating product descriptions is producing inconsistent and problematic outputs. Analyze the prompt, identify the issues, and provide a corrected version.

**Original Prompt:**
```
You are a product copywriter. Write a compelling product description.

Product: Wireless Bluetooth Earbuds
Features: Active noise cancellation, 30-hour battery, water resistant

Make it engaging and mention all the features. Keep it professional but fun.
Add a call to action at the end.
```

**Observed Problems:**
1. Output length varies wildly (50 words to 300 words)
2. Sometimes invents features not in the input ("crystal clear calls", "premium drivers")
3. Tone inconsistency—some outputs are very casual, others very formal
4. Call to action sometimes missing or awkwardly placed
5. Sometimes outputs marketing claims without evidence ("best in class")

**Tasks:**
1. Identify which prompt components are causing each problem
2. Explain the failure mechanism for each issue
3. Provide a corrected prompt that addresses all five problems

### Requirements

- [ ] Map each problem to specific prompt deficiency
- [ ] Explain why the deficiency causes that specific failure
- [ ] Provide complete corrected prompt
- [ ] Verify correction addresses each original problem

### Hints (Progressive)

<details>
<summary>Hint 1: Getting Started</summary>

Analyze against the five components:
- Context: Present but minimal ("product copywriter")
- Instructions: Vague ("compelling", "engaging", "professional but fun")
- Constraints: Almost none (no length, no anti-hallucination rules)
- Format: None (no structure specified)
- Examples: None

Each missing/weak component maps to specific failures.

</details>

<details>
<summary>Hint 2: Key Insight</summary>

Problem-to-deficiency mapping:
1. Length varies → No length constraint
2. Invents features → No "only use provided features" constraint
3. Tone inconsistent → "Professional but fun" is ambiguous
4. CTA issues → No format specification for structure
5. Unsubstantiated claims → No constraint against superlatives without evidence

</details>

<details>
<summary>Hint 3: Nearly There</summary>

Your corrected prompt needs:
- Explicit word count range
- "Only include features from the provided list" constraint
- Specific tone guidance (instead of vague "professional but fun")
- Format template showing section structure including CTA position
- Constraint against unverifiable superlatives

Consider including an example to demonstrate the expected output pattern.

</details>

### Solution

<details>
<summary>Click to reveal full solution</summary>

**Problem-to-Deficiency Analysis:**

| Problem | Root Cause | Deficient Component |
|---------|------------|---------------------|
| 1. Length varies (50-300 words) | No length specification | Missing **Constraint** |
| 2. Invents features | No restriction on feature source | Missing **Constraint** |
| 3. Tone inconsistent | "Professional but fun" is subjective | Vague **Instructions** |
| 4. CTA issues | No structural guidance | Missing **Format** |
| 5. Unsubstantiated claims | No superlative/claim restrictions | Missing **Constraint** |

**Failure Mechanisms:**

| Problem | Why It Happens |
|---------|----------------|
| **Length variance** | Without bounds, model optimizes for "compelling" which it interprets variably based on context |
| **Feature invention** | Model associates product type with common features; generates plausible additions without instruction to restrict |
| **Tone variance** | "Professional but fun" has no objective definition; model samples from wide style distribution |
| **CTA issues** | Without structural placement, model decides whether/where to include based on general patterns |
| **Unsubstantiated claims** | Marketing copy training data includes superlatives; model reproduces pattern without evidence requirement |

---

**Corrected Prompt:**

```
CONTEXT:
You are a product copywriter for an e-commerce platform. Your descriptions
appear on product listing pages where customers make purchase decisions.
Descriptions should be informative, accurate, and persuasive without
being hyperbolic.

INSTRUCTIONS:
Write a product description following this structure:
1. Opening hook (1 sentence capturing the key value proposition)
2. Feature highlights (bullet points covering ALL provided features)
3. Use case or benefit statement (who is this product for)
4. Call to action (purchase encouragement)

TONE GUIDELINES:
- Confident but not boastful
- Conversational but not casual (no slang, contractions OK)
- Enthusiastic but grounded in provided facts
- Example tone: "These earbuds deliver..." NOT "These AMAZING earbuds..."

CONSTRAINTS:
- Total length: 80-120 words
- ONLY mention features explicitly provided in the input; do not add features
- Do not use superlatives (best, ultimate, perfect) unless supported by
  specific evidence in the input
- Do not make claims about competitor comparisons
- Call to action must be the final sentence

FORMAT:
[Opening hook - 1 sentence]

Key Features:
• [Feature 1 with brief benefit]
• [Feature 2 with brief benefit]
• [Feature 3 with brief benefit]

[Use case statement - 1-2 sentences about ideal customer]

[Call to action - 1 sentence]

EXAMPLE:
Product: Running Shoes
Features: Lightweight mesh upper, responsive foam sole, reflective details

Output:
Hit the pavement with confidence in shoes designed for serious runners.

Key Features:
• Lightweight mesh upper keeps feet cool and comfortable mile after mile
• Responsive foam sole returns energy with every stride
• Reflective details enhance visibility during early morning or evening runs

Whether you're training for a marathon or enjoying weekend jogs, these shoes
support your running goals.

Add them to your cart and experience the difference on your next run.

---
PRODUCT: Wireless Bluetooth Earbuds
FEATURES: Active noise cancellation, 30-hour battery, water resistant

Write the product description:
```

**Expected Output:**

```
Immerse yourself in your music with earbuds that block out the world and
keep playing all day.

Key Features:
• Active noise cancellation lets you focus on your audio without distractions
• 30-hour battery life powers through your longest days without recharging
• Water resistant design handles workouts and weather with confidence

Perfect for commuters, travelers, and fitness enthusiasts who demand
uninterrupted audio.

Order now and discover what you've been missing.
```

**Verification - Problems Addressed:**

| Problem | How Corrected Prompt Fixes It |
|---------|-------------------------------|
| 1. Length varies | Explicit "80-120 words" constraint |
| 2. Invents features | "ONLY mention features explicitly provided" |
| 3. Tone inconsistent | Specific tone guidelines with examples of good/bad |
| 4. CTA issues | Format template places CTA explicitly at end |
| 5. Unsubstantiated claims | "Do not use superlatives unless supported" |

</details>

### Common Mistakes

- ❌ **Mistake:** Adding constraints without explaining "why" to the model
  - **Why it happens:** Assuming model understands constraint purpose
  - **How to avoid:** Context explains the e-commerce setting, giving constraints meaning

- ❌ **Mistake:** Fixing tone with more vague adjectives ("warm but professional")
  - **Why it happens:** Difficulty articulating tone precisely
  - **How to avoid:** Give examples of acceptable and unacceptable phrasing

- ❌ **Mistake:** Format template without example output
  - **Why it happens:** Assuming template is self-explanatory
  - **How to avoid:** Example demonstrates template applied to real content

### Extension Challenge

The product team now wants A/B testing capability—generating two description variants (one feature-focused, one benefit-focused) from the same input. Modify your prompt to produce both variants while maintaining quality controls.

---

---

## Summary

### Key Takeaways
1. **Prompt structure** requires all five components (context, instructions, constraints, format, examples) for reliable outputs
2. **Few-shot prompting** depends on example quality and diversity more than quantity—cover edge cases
3. **Chain-of-thought** must be structured for complex tasks; zero-shot CoT insufficient for multi-factor analysis
4. **Prompt chains** require explicit interface contracts and failure handling at each transition
5. **Prompt debugging** maps symptoms to component deficiencies; systematic analysis beats trial-and-error

### Concepts by Problem
| Problem | Primary Concepts | Secondary Concepts |
|---------|-----------------|-------------------|
| P1 (Warm-Up) | Prompt Structure | Component analysis |
| P2 (Skill-Builder) | Few-Shot Prompting | Example selection, edge cases |
| P3 (Skill-Builder) | Chain-of-Thought | Structured reasoning |
| P4 (Challenge) | Prompt Chaining | Decomposition, contracts |
| P5 (Debug/Fix) | Failure Diagnosis | Repair strategies |

### Connection to Lesson 1 (Agent Skills)
| Practice Problem | Agent Skills Connection |
|------------------|------------------------|
| P1: Prompt Structure | Skills encode structured prompts in execution procedures |
| P2: Few-Shot Design | Skills include examples in instructions |
| P4: Prompt Chains | Skill composition implements prompt chains |
| P5: Debugging | Skill failure handling follows similar patterns |

### Next Steps
- If struggled with P1: Re-review five components in Core Concepts - Concept 1
- If struggled with P2-P3: Practice with real tasks; iterate on example selection
- If struggled with P4: Review Lesson 1 on skill composition
- If struggled with P5: Build diagnostic checklist from component framework
- Ready for assessment: Proceed to quiz

---

*Generated from Lesson 2: Prompt Engineering | Practice Problems Skill*
