# Assessment Quiz: Lesson 2 - Prompt Engineering

**Source Material:** Lessons/Lesson_2.md
**Flashcard Reference:** notes/prompt-engineering/flashcards/lesson-2-prompt-engineering-flashcards.md
**Concept Map Reference:** notes/prompt-engineering/concept-maps/lesson-2-prompt-engineering-concept-map.md
**Practice Problems Reference:** notes/prompt-engineering/practice/lesson-2-prompt-engineering-practice-problems.md
**Date Generated:** 2026-01-08
**Total Questions:** 5
**Estimated Completion Time:** 30-40 minutes
**Distribution:** 2 Multiple Choice | 2 Short Answer | 1 Essay

---

## Instructions

- **Multiple Choice:** Select the single best answer
- **Short Answer:** Respond in 3-5 sentences
- **Essay:** Provide a comprehensive response (2-3 paragraphs)
- **Open Book:** You may reference the study notes, but attempt questions first from memory

---

## Questions

### Question 1 | Multiple Choice
**Cognitive Level:** Remember/Understand
**Concept Tested:** Prompt Structure and Component Ordering
**Source Section:** Core Concepts - Concept 1
**Concept Map Node:** Prompt Structure (Critical - 10 connections)
**Related Flashcard:** Card 1

Why should context generally precede instructions in a well-structured prompt, and constraints appear near the end?

A) This ordering reduces token count, making prompts more cost-efficient to process

B) Context primes the model's attention mechanisms before task specification, while constraints near the end leverage recency effects for stronger influence on generation

C) Models are trained to expect this exact ordering, and deviating from it causes parsing errors

D) Context must load relevant knowledge bases before instructions can be interpreted, similar to database queries

---

### Question 2 | Multiple Choice
**Cognitive Level:** Remember/Understand
**Concept Tested:** Prompting Strategy Selection
**Source Section:** Core Concepts - Concepts 2 & 3
**Concept Map Node:** Few-Shot (High - 6 connections), Chain-of-Thought (High - 7 connections)
**Related Flashcard:** Card 2

A developer needs to build a prompt for calculating shipping costs based on package dimensions, weight, destination zone, and shipping speed. The calculation involves multiple conditional rules and requires accurate numerical results. Which prompting strategy is most appropriate?

A) Zero-shot prompting, because shipping calculations are common tasks the model has seen during training

B) Few-shot prompting with 3-5 examples covering different shipping scenarios

C) Chain-of-thought prompting that instructs the model to show calculation steps before the final answer

D) Many-shot prompting with 10+ examples to ensure the model learns the exact calculation formula

---

### Question 3 | Short Answer
**Cognitive Level:** Apply/Analyze
**Concept Tested:** Few-Shot Example Design
**Source Section:** Core Concepts - Concept 2
**Concept Map Node:** Few-Shot, Examples, In-Context Learning
**Related Flashcard:** Card 3
**Expected Response Length:** 3-5 sentences

You are designing a few-shot prompt for sentiment classification of customer reviews into three categories: Positive, Negative, and Mixed. You have space for only 3 examples in your prompt.

Describe what characteristics your three examples should have to maximize classification accuracy. Explain why example diversity matters more than example quantity, and identify one specific edge case you would include.

---

### Question 4 | Short Answer
**Cognitive Level:** Apply/Analyze
**Concept Tested:** System Prompts and Behavioral Configuration
**Source Section:** Core Concepts - Concept 4
**Concept Map Node:** System Prompt (High - 5 connections)
**Related Flashcard:** Card 5
**Expected Response Length:** 3-5 sentences

A company is deploying an AI assistant for their customer service platform. The assistant must maintain a professional tone, never promise refunds without manager approval, and escalate to human agents when customers express frustration.

Explain the role of a system prompt in achieving these requirements. Describe at least three specific elements you would include in the system prompt and explain how each element contributes to consistent, safe behavior.

---

### Question 5 | Essay
**Cognitive Level:** Evaluate/Synthesize
**Concepts Tested:** Prompt Chaining, Strategy Selection, Failure Handling
**Source Sections:** All Core Concepts, Practical Applications, Critical Analysis
**Concept Map:** Full pathway traversal
**Related Flashcard:** Card 4, Card 5
**Expected Response Length:** 2-3 paragraphs

**Scenario:** You are building an automated content moderation system for a social media platform. The system must analyze user posts and:
1. Classify content into categories (safe, warning, remove)
2. Identify specific policy violations if any
3. Generate an explanation for moderation decisions (for appeal purposes)
4. Produce a sanitized version of problematic content when possible

Design a prompt engineering solution addressing:

1. **Architecture Decision:** Should this be a single prompt or a prompt chain? Justify your choice with specific reasoning about task complexity and reliability requirements.

2. **Strategy Selection:** For each major capability, specify whether you would use zero-shot, few-shot, or chain-of-thought prompting and explain why.

3. **Critical Constraints:** Identify three constraints that are essential for a content moderation system and explain how you would implement each in your prompt design.

4. **Failure Handling:** What happens when the system is uncertain? Design a specific mechanism for handling edge cases where classification confidence is low.

5. **Evaluation Consideration:** How would you measure whether your prompt engineering solution is working effectively? Identify two metrics and explain what they measure.

**Evaluation Criteria:**
- [ ] Makes justified architecture decision (single vs. chain)
- [ ] Selects appropriate strategies with clear reasoning
- [ ] Identifies meaningful constraints with implementation approach
- [ ] Designs specific uncertainty handling mechanism
- [ ] Proposes relevant evaluation metrics

---

## Answer Key

### Question 1 | Multiple Choice
**Correct Answer:** B

**Explanation:**
The ordering of prompt components is not arbitrary—it leverages how transformer attention mechanisms process sequential input. **Context preceding instructions** ensures the model's attention is primed with relevant domain knowledge and framing before it encounters the task specification. This is analogous to how humans understand requests better when given background first. **Constraints near the end** exploit recency bias (the tendency for later tokens to have stronger influence on generation), making boundaries and restrictions more likely to be respected in the output.

**Why Other Options Are Incorrect:**
- **A)** Incorrect—ordering does not affect token count; the same words are present regardless of arrangement.
- **C)** Incorrect—models are not trained to expect a specific ordering format; they process prompts flexibly. There are no "parsing errors" from reordering.
- **D)** Incorrect—this mischaracterizes how LLMs work. They don't "load knowledge bases" like databases; knowledge is encoded in weights and activated by attention patterns.

**Understanding Gap Indicator:**
If answered incorrectly, this may indicate confusion about how attention mechanisms work in transformers. Review the Theoretical Framework section on "Attention as Resource Allocation."

**Review Recommendation:** Study Notes: Theoretical Framework; Concept Map: Attention Allocation node

---

### Question 2 | Multiple Choice
**Correct Answer:** C

**Explanation:**
Shipping cost calculation involves **multi-step reasoning** with conditional logic (different rates by zone, weight thresholds, speed multipliers). Chain-of-thought prompting is optimal because:
1. It forces the model to decompose the calculation into explicit steps
2. Intermediate steps are generated as tokens, enabling error detection
3. Numerical calculations benefit from showing work (reduces arithmetic errors)
4. The conditional nature of shipping rules requires logical reasoning about which rules apply

**Why Other Options Are Incorrect:**
- **A)** Zero-shot is insufficient—while shipping is a common concept, the specific calculation rules require explicit reasoning. Zero-shot would likely produce inaccurate results.
- **B)** Few-shot demonstrates format but doesn't ensure correct calculation logic. The model might pattern-match to similar examples rather than correctly applying rules to new inputs.
- **D)** Many-shot is overkill and wastes context window. The issue isn't learning a pattern from examples; it's performing accurate multi-step calculation.

**Understanding Gap Indicator:**
If answered incorrectly, this may indicate difficulty matching task characteristics to strategy strengths. Review the Strategy Selection Decision Tree in the Concept Map.

**Review Recommendation:** Study Notes: Core Concepts - Concept 3; Flashcard 2

---

### Question 3 | Short Answer
**Model Answer:**

**Example Characteristics:**
The three examples should cover all three sentiment categories (one Positive, one Negative, one Mixed) and vary in linguistic expression. They should include different review lengths, writing styles (formal vs. casual), and sentiment expression patterns (explicit statements like "I love it" vs. implicit indicators like describing usage frequency).

**Why Diversity Over Quantity:**
Example diversity matters more than quantity because in-context learning works by pattern recognition. Three diverse examples teach the model to recognize sentiment across varied expressions, while three similar examples (e.g., all short reviews with explicit sentiment words) would cause the model to overfit to that specific pattern and fail on reviews with different characteristics.

**Specific Edge Case:**
I would include a **Mixed sentiment example** with the pattern: "Product quality is excellent but customer service was terrible." This demonstrates how to classify reviews containing both strong positive and strong negative elements—the most common source of classification errors. The example teaches that mixed isn't just "lukewarm" but can involve strong opposing sentiments.

**Key Components Required:**
- [ ] Identifies category coverage (all three sentiments)
- [ ] Explains linguistic diversity (length, style, expression patterns)
- [ ] Articulates why diversity enables generalization
- [ ] Provides specific edge case with reasoning

**Partial Credit Guidance:**
- Full credit: All components addressed with specific examples
- Partial credit: General diversity mentioned but weak edge case discussion
- Minimal credit: Only mentions category coverage

**Understanding Gap Indicator:**
If answered incorrectly, this may indicate treating few-shot as "more examples = better" rather than understanding in-context learning mechanics. Review the In-Context Learning Theory section.

**Review Recommendation:** Study Notes: Theoretical Framework - In-Context Learning; Practice Problem 2

---

### Question 4 | Short Answer
**Model Answer:**

**Role of System Prompt:**
A system prompt establishes a persistent behavioral framework that governs the AI assistant's responses across all customer interactions. Unlike user prompts that request specific actions, system prompts configure the assistant's persona, capabilities, constraints, and escalation protocols—functioning as a "constitution" that remains constant regardless of individual customer messages.

**Three Essential Elements:**

1. **Role and Tone Specification:**
   ```
   "You are a professional customer service representative for [Company].
   Maintain a helpful, empathetic, and professional tone in all responses.
   Use clear language and avoid jargon."
   ```
   *Contribution:* Establishes consistent communication style across all agents and interactions, ensuring brand voice is maintained.

2. **Capability Boundaries (Refund Constraint):**
   ```
   "You cannot authorize refunds, exchanges, or credits. If a customer
   requests financial remediation, explain that you will escalate to a
   manager who can review their case within 24 hours."
   ```
   *Contribution:* Prevents the assistant from making unauthorized promises, protecting the company legally and financially while setting clear customer expectations.

3. **Escalation Protocol:**
   ```
   "If a customer expresses frustration, anger, or dissatisfaction three
   or more times, OR uses profanity, OR explicitly requests a human agent,
   immediately acknowledge their feelings and transfer to a human agent
   with a summary of the conversation."
   ```
   *Contribution:* Ensures frustrated customers receive human attention before situations escalate, improving customer satisfaction and reducing complaint severity.

**Key Components Required:**
- [ ] Explains system prompt's persistent/constitutional nature
- [ ] Provides at least 3 specific elements with example language
- [ ] Connects each element to a specific behavioral outcome
- [ ] Addresses all three requirements (tone, refunds, escalation)

**Partial Credit Guidance:**
- Full credit: Clear role explanation + 3 specific elements with behavioral connections
- Partial credit: Elements listed but vague or missing behavioral explanation
- Minimal credit: Generic description without specific elements

**Understanding Gap Indicator:**
If answered incorrectly, this may indicate confusion between system prompts and user prompts, or difficulty translating requirements into prompt constraints. Review Core Concepts - Concept 4.

**Review Recommendation:** Study Notes: Core Concepts - Concept 4; Flashcard 5

---

### Question 5 | Essay
**Model Answer:**

**1. Architecture Decision: Prompt Chain (4 Steps)**

A prompt chain is the appropriate architecture for this content moderation system, not a single prompt. The justification is threefold:

First, **task complexity requires decomposition**. The four requirements (classify, identify violations, explain, sanitize) are distinct operations with different success criteria. A single prompt attempting all four would need to balance competing objectives, likely degrading performance on each.

Second, **reliability requirements demand validation**. Content moderation has high stakes—false positives frustrate users while false negatives expose the platform to harm. A chain enables validation at each step: if classification is uncertain, we can stop before generating explanations for incorrect decisions.

Third, **different capabilities require different strategies**. Classification benefits from few-shot examples, while explanation generation benefits from chain-of-thought. A single prompt cannot optimally serve both needs.

**Proposed Chain:**
```
Step 1: Classification → Step 2: Violation Identification → Step 3: Explanation Generation → Step 4: Content Sanitization (conditional)
```

**2. Strategy Selection:**

| Capability | Strategy | Rationale |
|------------|----------|-----------|
| **Classification** | Few-shot (5 examples) | Policy categories are platform-specific; examples establish the taxonomy and boundary conditions |
| **Violation Identification** | Zero-shot with policy list | Violations are explicitly defined; model matches content against provided policy definitions |
| **Explanation Generation** | Chain-of-thought | Explanations must reference specific content and specific policies; reasoning ensures logical connection |
| **Content Sanitization** | Few-shot | Sanitization patterns vary by violation type; examples demonstrate appropriate transformations |

**3. Critical Constraints:**

**Constraint 1: Bias Mitigation**
```
"Evaluate content based solely on the words and their meaning in context.
Do not consider or infer author demographics. The same content must receive
the same classification regardless of who posted it."
```
*Implementation:* Include in system prompt + few-shot examples deliberately vary implied author demographics while showing consistent classification.

**Constraint 2: Evidence Requirement**
```
"Only classify content as 'remove' if you can identify a specific policy
violation with a direct quote from the content. If no specific violation
can be cited, classify as 'warning' maximum."
```
*Implementation:* Output format requires `violation_quote` field; validation step rejects removals without quotes.

**Constraint 3: Context Consideration**
```
"Consider whether content is quoting, reporting, or condemning problematic
material versus endorsing it. Educational, newsworthy, or critical discussion
of policy-violating topics should generally be classified as 'safe' or 'warning'
rather than 'remove'."
```
*Implementation:* Include few-shot examples demonstrating this distinction (e.g., news article about hate speech vs. hate speech itself).

**4. Uncertainty Handling Mechanism:**

When classification confidence is low (operationalized as model assigning >20% probability to multiple categories), implement a **tiered escalation system:**

```
If confidence_score < 0.7:
    1. Re-run classification with additional context (user history, thread context)
    2. If still uncertain after context enrichment:
       - If highest probability is "remove": Flag for human review (high stakes)
       - If highest probability is "warning": Apply warning + flag for review
       - If highest probability is "safe": Classify as safe (fail-open for low-risk)
    3. Log all uncertain cases for model improvement training data
```

This mechanism ensures:
- High-stakes decisions (removals) never proceed without confidence
- Low-stakes decisions (safe classifications) don't bottleneck on human review
- Uncertainty data feeds continuous improvement

**5. Evaluation Metrics:**

**Metric 1: Precision and Recall by Category**
- *What it measures:* Classification accuracy, specifically the rate of false positives (wrongly removed content) and false negatives (missed violations)
- *Why it matters:* High false positives alienates users; high false negatives exposes platform to harm
- *Target:* >95% precision for "remove" category; >90% recall for severe violations

**Metric 2: Explanation Quality Score (Human Evaluation)**
- *What it measures:* Whether generated explanations accurately cite the relevant policy, quote the violating content, and logically connect the two
- *Why it matters:* Poor explanations undermine trust and fail appeals processes
- *Evaluation method:* Sample 100 moderation decisions weekly; human raters score explanation clarity, accuracy, and completeness on 1-5 scale
- *Target:* Average score >4.0

---

**Evaluation Rubric:**

| Criterion | Excellent (5) | Proficient (4) | Developing (3) | Beginning (2) | Insufficient (1) |
|-----------|---------------|----------------|----------------|---------------|------------------|
| Architecture Decision | Clear choice with 3+ specific justifications tied to task requirements | Choice with 2 solid justifications | Choice stated with weak reasoning | Choice without justification | No clear architecture |
| Strategy Selection | All capabilities addressed with strategy + rationale for each | Most capabilities with reasoning | Strategies listed but weak rationale | Generic strategies | No strategy discussion |
| Constraints | 3 meaningful constraints with implementation details | 2-3 constraints with some implementation | Constraints mentioned generically | 1 vague constraint | No constraints |
| Uncertainty Handling | Specific mechanism with tiered logic and operational thresholds | Mechanism described but missing details | Acknowledges uncertainty exists | Vague mention | No handling |
| Evaluation Metrics | 2 relevant metrics with measurement methodology | 2 metrics with partial methodology | 1 metric or generic mentions | Vague quality mention | No metrics |

**Understanding Gap Indicator:**
If response lacks depth in specific areas:
- Weak architecture decision → Review Prompt Chaining benefits in Core Concepts - Concept 5
- Poor strategy selection → Review Strategy Selection Decision Tree in Concept Map
- Missing constraints → Review System Prompts and Constraints in Core Concepts
- No uncertainty handling → Review Failure Handling in Practice Problem 4
- Weak metrics → Consider what "working" means for each capability

---

## Diagnostic Feedback & Review Recommendations

### Performance-Based Review Guide

| If Incorrect On... | Gap Identified | Review Section | Priority |
|--------------------|----------------|----------------|----------|
| Question 1 | Component ordering rationale | Theoretical Framework | High |
| Question 2 | Strategy selection criteria | Core Concepts 2 & 3 | High |
| Question 3 | Few-shot example design | Core Concepts 2 | Medium |
| Question 4 | System prompt design | Core Concepts 4 | Medium |
| Question 5 | Integrated prompt engineering | All sections | Low (cumulative) |

### Targeted Review Recommendations

#### For Multiple Choice Errors (Questions 1-2)
**Indicates:** Foundational concept gaps
**Action Steps:**
1. Re-read Core Concepts 1-3 in study notes
2. Study Concept Map—trace connections from Prompt Structure to Strategies
3. Review Flashcards 1 and 2 using spaced repetition
4. Attempt Practice Problems 1 and 3 before retaking quiz

**Focus On:** Understanding WHY certain approaches work, not just WHAT they are

#### For Short Answer Errors (Questions 3-4)
**Indicates:** Difficulty applying concepts to scenarios
**Action Steps:**
1. Review Practice Problem 2 (few-shot design) and Problem 5 (debugging)
2. Study the example prompts in Lesson 2 Case Study
3. Practice writing prompts for real tasks—iterate on failures
4. Use Concept Map Pathway 2 (Strategy-First) to connect principles to applications

**Focus On:** Translating requirements into specific prompt elements

#### For Essay Weakness (Question 5)
**Indicates:** Integration and synthesis challenges
**Action Steps:**
1. Complete all five Practice Problems, especially P4 (prompt chain design)
2. Study Concept Map Critical Path for minimum viable understanding
3. Review the connection between Lesson 2 and Lesson 1 (Agent Skills)
4. Practice designing complete systems, not just individual prompts

**Focus On:** Treating prompt engineering as system design, not just text writing

### Mastery Level Interpretation

| Score | Level | Interpretation | Next Steps |
|-------|-------|----------------|------------|
| 5/5 | **Expert** | Strong mastery of prompt engineering | Ready for production systems; proceed to Lesson 1 (Agent Skills) |
| 4/5 | **Proficient** | Good understanding with minor gaps | Review indicated gap; safe to proceed with awareness |
| 3/5 | **Developing** | Moderate understanding; application difficulties | Systematic review recommended; complete all practice problems |
| 2/5 | **Foundational** | Significant gaps in core concepts | Re-study from Concept Map Critical Path |
| 1/5 or below | **Beginning** | Requires comprehensive review | Restart from Lesson 2 study notes; use Foundational pathway |

---

## Cross-Reference Matrix

| Quiz Question | Concept Map Node | Flashcard | Practice Problem |
|---------------|------------------|-----------|------------------|
| Q1 (MC) | Prompt Structure, Attention Allocation | Card 1 | P1 (Warm-Up) |
| Q2 (MC) | Strategies cluster, Chain-of-Thought | Card 2 | P3 (Skill-Builder) |
| Q3 (SA) | Few-Shot, Examples, In-Context Learning | Card 3 | P2 (Skill-Builder) |
| Q4 (SA) | System Prompt | Card 5 | P5 (Debug/Fix) |
| Q5 (Essay) | Full Integration | Cards 4, 5 | P4 (Challenge) |

---

## Skill Chain Traceability

```
Lesson 2 (Source)
    │
    │  Topic: Prompt Engineering
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Study Notes                                  │
│  Sections: 5 Core Concepts, Theoretical Framework,                  │
│  Practical Applications, Critical Analysis                          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────────────────────┐
        │                      │                                      │
        ▼                      ▼                                      ▼
┌───────────────┐    ┌─────────────────┐                    ┌─────────────────┐
│  Concept Map  │    │   Flashcards    │                    │    Practice     │
│               │    │                 │                    │    Problems     │
│  22 concepts  │    │  5 cards        │                    │                 │
│  34 relations │    │  2E/2M/1H       │                    │  5 problems     │
│  4 pathways   │    │                 │                    │  1W/2S/1C/1D    │
└───────┬───────┘    └────────┬────────┘                    └────────┬────────┘
        │                     │                                      │
        └─────────────────────┼──────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │        Quiz         │
                    │                     │
                    │  5 questions        │
                    │  2MC/2SA/1E         │
                    │                     │
                    │  Integrates:        │
                    │  - Concept centrality│
                    │  - Flashcard mapping │
                    │  - Practice skills   │
                    └─────────────────────┘
```

---

## Connection to Lesson 1 (Agent Skills)

| Quiz Question | Agent Skills Connection |
|---------------|------------------------|
| Q1 (Structure) | Skills encode optimized prompt structures in execution procedures |
| Q2 (Strategies) | Skills select appropriate strategies per task type |
| Q3 (Few-Shot) | Skills include examples in instructions for consistency |
| Q4 (System Prompts) | Skills use system prompts for behavioral framing |
| Q5 (Chains) | Skill composition directly implements prompt chains |

**Prerequisite Verification:**
Mastery of Lesson 2 (Prompt Engineering) provides the foundation for understanding how Agent Skills (Lesson 1) encode and execute sophisticated prompting strategies within reusable, composable modules.

---

*Generated from Lesson 2: Prompt Engineering | Quiz Skill*
