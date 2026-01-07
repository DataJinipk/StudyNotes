# Assessment Quiz: Lesson 1 - Creating Agent Skills

**Source Material:** Lessons/Lesson_1.md
**Flashcard Reference:** notes/agent-skills/flashcards/lesson-1-agent-skills-flashcards.md
**Concept Map Reference:** notes/agent-skills/concept-maps/lesson-1-agent-skills-concept-map.md
**Practice Problems Reference:** notes/agent-skills/practice/lesson-1-agent-skills-practice-problems.md
**Date Generated:** 2026-01-07
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
**Concept Tested:** Agent Skill Definition and Architecture
**Source Section:** Core Concepts - Concept 1
**Concept Map Node:** Agent Skill (Critical - 9 connections)
**Related Flashcard:** Card 1

What fundamentally distinguishes an agent skill from ad-hoc prompt engineering?

A) Agent skills require training the underlying language model on domain-specific data, while prompt engineering uses the model as-is

B) Agent skills encapsulate capabilities in structured, reusable modules with defined interfaces, while ad-hoc prompting requires crafting new instructions for each interaction

C) Agent skills can only be invoked through slash commands, while prompt engineering allows natural language interaction

D) Agent skills execute faster than ad-hoc prompts because they are pre-compiled into the model's architecture

---

### Question 2 | Multiple Choice
**Cognitive Level:** Remember/Understand
**Concept Tested:** Skill Composition Patterns
**Source Section:** Core Concepts - Concept 3
**Concept Map Node:** Skill Composition (High - 7 connections)
**Related Flashcard:** Card 2

A workflow requires processing customer feedback data through three independent analysis types: sentiment analysis, topic extraction, and urgency classification. After all three complete, results must be combined into a unified report. Which composition pattern combination is most appropriate?

A) Sequential chaining for all four skills: sentiment → topic → urgency → report

B) Parallel execution for the three analyses, followed by sequential chaining to the report skill

C) Hierarchical nesting where the report skill orchestrates the three analyses as sub-skills

D) Conditional branching that selects the most relevant analysis based on feedback content

---

### Question 3 | Short Answer
**Cognitive Level:** Apply/Analyze
**Concept Tested:** Invocation Pattern Trade-offs
**Source Section:** Core Concepts - Concept 2
**Concept Map Node:** Invocation Pattern (High - 5 connections)
**Related Flashcard:** Card 3
**Expected Response Length:** 3-5 sentences

You are designing a skill for generating database queries from natural language descriptions. This skill will be used by:
- Database administrators (experts who use it daily)
- Business analysts (occasional users with limited SQL knowledge)

Design both a slash command and a natural language trigger for this skill. Then analyze which invocation pattern each user group would prefer and why, referencing the discoverability-flexibility trade-off discussed in Lesson 1.

---

### Question 4 | Short Answer
**Cognitive Level:** Apply/Analyze
**Concept Tested:** Theoretical Foundations
**Source Section:** Theoretical Framework
**Concept Map Node:** Modular Programming Theory, Cognitive Load Theory (Medium - 3 connections each)
**Related Flashcard:** Card 4
**Expected Response Length:** 3-5 sentences

The minimalist and holistic perspectives represent contrasting approaches to skill granularity. Consider a skill that processes expense reports—it extracts line items, validates against policy, calculates totals, and generates approval recommendations.

Using principles from **cognitive load theory** and **modular programming theory**, argue whether this should be designed as one holistic skill or decomposed into multiple atomic skills. Identify at least one advantage and one disadvantage of your chosen approach.

---

### Question 5 | Essay
**Cognitive Level:** Evaluate/Synthesize
**Concepts Tested:** Complete Skill Architecture, Quality Dimensions, Composition
**Source Sections:** All Core Concepts, Critical Analysis, Practical Applications
**Concept Map:** Full pathway traversal
**Related Flashcard:** Card 5
**Expected Response Length:** 2-3 paragraphs

**Scenario:** You are the lead architect for a content management platform. The product team requests an "intelligent publishing workflow" that transforms draft articles into publication-ready content with SEO optimization, image suggestions, and social media snippets.

Design a skill-based solution addressing the following:

1. **Skill Chain Architecture:** Define 3-4 skills with clear single responsibilities, specifying how they connect (sequential, parallel, or hybrid)

2. **Interface Contracts:** For each skill, specify the critical input requirements and output guarantees that enable composition

3. **Quality Analysis:** Evaluate how your design achieves (or sacrifices) each of the four quality dimensions: reusability, consistency, maintainability, and composability

4. **Failure Resilience:** Identify the two most critical failure points in your chain and describe specific recovery strategies

5. **Perspective Justification:** Explain whether your overall design follows a minimalist or holistic philosophy and defend this choice based on the platform's needs

**Evaluation Criteria:**
- [ ] Designs coherent skill chain with clear responsibilities and connections
- [ ] Specifies meaningful interface contracts that enable composition
- [ ] Analyzes all four quality dimensions with specific design evidence
- [ ] Identifies realistic failure points with concrete recovery strategies
- [ ] Provides reasoned justification for granularity decisions

---

## Answer Key

### Question 1 | Multiple Choice
**Correct Answer:** B

**Explanation:**
The fundamental distinction is **structural**: agent skills are self-contained, reusable modules that encapsulate specific capabilities with defined invocation patterns, input/output specifications, and execution procedures. This structure enables skills to be documented, tested, versioned, and composed into larger workflows. Ad-hoc prompt engineering, by contrast, requires crafting unique instructions for each interaction, lacks formal interfaces, and cannot be systematically reused or composed.

**Why Other Options Are Incorrect:**
- **A)** Incorrect—skills do NOT require model training. They are runtime constructs that work with the model as-is, using structured prompts and procedures rather than fine-tuning.
- **C)** Incorrect—skills can support multiple invocation patterns including slash commands, natural language triggers, and programmatic API calls. The lesson explicitly discusses this flexibility.
- **D)** Incorrect—skills are not "pre-compiled" into the model. They execute through the same inference process as any prompt; the advantage is in structure and reusability, not execution speed.

**Understanding Gap Indicator:**
If answered incorrectly, this may indicate confusion between skill definition (structured prompt engineering) and model fine-tuning (training-time modification). Review Core Concepts - Concept 1 and the Key Terminology section.

**Review Recommendation:** Study Notes Section: Core Concepts - Concept 1; Flashcard 1

---

### Question 2 | Multiple Choice
**Correct Answer:** B

**Explanation:**
The workflow has two distinct phases: (1) three independent analyses that share no data dependencies, and (2) a synthesis step that requires all three results. **Parallel execution** is optimal for the independent analyses because they can run concurrently, reducing total execution time. Once all three complete, **sequential chaining** feeds their combined outputs to the report generator. This hybrid pattern matches the workflow's logical structure.

**Why Other Options Are Incorrect:**
- **A)** Incorrect—sequential chaining forces unnecessary serialization. Sentiment analysis doesn't need topic extraction results; they're independent and should run in parallel.
- **C)** Incorrect—hierarchical nesting implies the report skill dynamically orchestrates and invokes sub-skills. While technically possible, this conflates orchestration logic with synthesis logic and complicates the report skill's responsibility.
- **D)** Incorrect—conditional branching selects between alternatives, but this workflow requires ALL three analyses, not choosing one based on conditions.

**Understanding Gap Indicator:**
If answered incorrectly, this may indicate difficulty matching composition patterns to workflow requirements. Review the four composition patterns (sequential, parallel, conditional, hierarchical) and their appropriate use cases.

**Review Recommendation:** Study Notes Section: Core Concepts - Concept 3; Flashcard 2; Concept Map - Composition Patterns cluster

---

### Question 3 | Short Answer
**Model Answer:**

**Slash Command Design:**
```
/query-gen <description> [--dialect=mysql|postgres|sqlite] [--explain]
```
Example: `/query-gen "find all customers who ordered last month" --dialect=postgres`

**Natural Language Trigger:**
"Write a SQL query that..." or "Generate a database query to..." or "How would I query for..."
Example: "Write a SQL query that finds customers who haven't ordered in 90 days"

**User Preference Analysis:**

**Database Administrators (Experts)** would prefer the **slash command** because:
- High discoverability through explicit syntax and autocomplete
- Precision in specifying dialect and options
- Faster invocation once syntax is memorized
- Reduced cognitive load for repeated daily use—no need to phrase naturally

**Business Analysts (Occasional Users)** would prefer **natural language triggers** because:
- Lower learning curve—no syntax to memorize
- Higher flexibility—can express intent in their own words
- More forgiving of imprecise terminology
- Natural fit for their occasional, exploratory use pattern

**Trade-off Connection:** This illustrates the discoverability-flexibility trade-off from Lesson 1. Slash commands maximize discoverability (users know exactly what options exist) at the cost of flexibility. Natural language maximizes flexibility (varied phrasings accepted) at the cost of discoverability (users may not know skill capabilities). Designing for both user groups requires supporting both patterns.

**Key Components Required:**
- [ ] Provides both slash command with parameters and natural language examples
- [ ] Correctly identifies DBA preference for slash commands with valid reasoning
- [ ] Correctly identifies analyst preference for natural language with valid reasoning
- [ ] References discoverability-flexibility trade-off explicitly

**Partial Credit Guidance:**
- Full credit: Both patterns + correct user analysis + explicit trade-off reference
- Partial credit: Patterns provided but weak user analysis or missing trade-off
- Minimal credit: Only one pattern or reversed user preferences

**Understanding Gap Indicator:**
If answered incorrectly, this may indicate difficulty applying theoretical concepts to practical design decisions. Review the invocation pattern types and their trade-offs in Core Concepts - Concept 2.

**Review Recommendation:** Study Notes Section: Core Concepts - Concept 2; Flashcard 3; Practice Problem 2

---

### Question 4 | Short Answer
**Model Answer:**

**Recommended Approach:** This should be designed as **one holistic skill** with modular internal phases.

**Argument from Cognitive Load Theory:**
From the user's perspective, "process expense report" is a single mental task. Requiring users to invoke separate skills for extraction, validation, calculation, and recommendation would increase extraneous cognitive load—they must understand multiple skill interfaces, manage intermediate outputs, and ensure correct sequencing. A holistic skill with sensible defaults (validate against standard policy, use default approval thresholds) minimizes decisions required from users.

**Argument from Modular Programming Theory:**
While modular theory advocates for decomposition, it emphasizes **cohesion**—components should contain strongly related functionality. Expense processing steps are tightly coupled: validation requires extracted line items, totals require validated items, recommendations require all prior data. Decomposing creates artificial boundaries that fragment a naturally cohesive workflow.

**Advantage:** Single invocation, single output, simple user mental model. Users request expense processing and receive complete results without orchestration burden.

**Disadvantage:** Reduced reusability—the line-item extraction logic cannot be reused for other purposes (e.g., invoice processing) without extracting it into a separate skill. If extraction patterns need updating, changes affect the entire expense skill.

**Alternative Justification (if choosing atomic):**
A valid argument could also be made for decomposition if the platform requires extractors to be reused across expense reports, invoices, and receipts. In this case, modular programming's loose coupling principle would favor separate skills with explicit contracts, despite increased orchestration complexity.

**Key Components Required:**
- [ ] Takes clear position (holistic or atomic)
- [ ] References cognitive load theory appropriately
- [ ] References modular programming theory appropriately
- [ ] Identifies at least one advantage of chosen approach
- [ ] Identifies at least one disadvantage of chosen approach

**Partial Credit Guidance:**
- Full credit: Clear position + both theories applied + advantage + disadvantage
- Partial credit: Position taken but only one theory applied or missing trade-off
- Minimal credit: Position without theoretical justification

**Understanding Gap Indicator:**
If answered incorrectly, this may indicate difficulty connecting theoretical principles to design decisions. Review the Theoretical Framework section and the Scholarly Perspectives on granularity.

**Review Recommendation:** Study Notes Sections: Theoretical Framework, Critical Analysis; Flashcard 4; Concept Map - Theory cluster

---

### Question 5 | Essay
**Model Answer:**

**1. Skill Chain Architecture:**

I propose a four-skill chain with hybrid composition:

```
                    ┌─────────────────────┐
                    │   content-analyzer  │
                    │   (extract topics,  │
                    │    structure draft) │
                    └──────────┬──────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ▼                     ▼                     ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  seo-optimizer  │  │ image-suggester │  │ social-snipper  │
│  (keywords,     │  │ (visual recs    │  │ (platform-      │
│   meta tags)    │  │  from content)  │  │  specific clips)│
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         └────────────────────┴────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  content-assembler  │
                    │  (merge all into    │
                    │   publish-ready)    │
                    └─────────────────────┘
```

- **content-analyzer** (sequential first): Extracts topics, structure, and key themes from draft
- **seo-optimizer**, **image-suggester**, **social-snipper** (parallel): Independent optimizations that all depend on analyzer output but not each other
- **content-assembler** (sequential last): Merges all optimizations into final publish-ready package

**2. Interface Contracts:**

| Skill | Input Contract | Output Contract |
|-------|---------------|-----------------|
| **content-analyzer** | `{draft: markdown, metadata: {author, category}}` | `{structure: {title, sections[], word_count}, topics: string[], tone: enum, key_quotes: string[]}` |
| **seo-optimizer** | `{topics: string[], title: string, structure: object}` | `{keywords: {primary, secondary[]}, meta_description: string, heading_suggestions: object}` |
| **image-suggester** | `{topics: string[], sections: array, tone: enum}` | `{suggestions: [{section, image_query, alt_text, placement}]}` |
| **social-snipper** | `{title: string, key_quotes: string[], topics: string[]}` | `{twitter: string, linkedin: string, facebook: string, hashtags: string[]}` |
| **content-assembler** | `{original_draft, seo_data, image_suggestions, social_snippets}` | `{final_draft: markdown, seo_metadata: object, asset_manifest: object, social_kit: object}` |

**3. Quality Analysis:**

| Dimension | Assessment | Evidence |
|-----------|------------|----------|
| **Reusability** | High | Each optimization skill is content-agnostic. `seo-optimizer` works for any content with topics; `social-snipper` works for any content with quotes. Can be reused for blog posts, product pages, newsletters. |
| **Consistency** | High | Fixed output schemas ensure every article receives identical optimization structure. All articles get the same SEO fields, same social platform coverage. |
| **Maintainability** | Medium-High | SEO algorithm changes only affect `seo-optimizer`; social platform changes only affect `social-snipper`. However, if topic extraction logic changes in `content-analyzer`, all downstream skills may need contract updates. |
| **Composability** | High | Explicit contracts enable substitution. A `video-snipper` could replace or supplement `social-snipper`. An `advanced-seo-optimizer` could swap in with same interface. Parallel execution pattern demonstrates composition working. |

**4. Failure Resilience:**

**Critical Failure Point 1: content-analyzer produces incomplete topics**
- *Risk:* Downstream skills receive empty or low-quality topic lists, producing generic/irrelevant optimizations
- *Recovery Strategy:* Implement minimum viability check—if topics.length < 3 or confidence_score < 0.7, fall back to keyword extraction from title + first paragraph. Log warning for human review but allow chain to continue with degraded input rather than failing entirely.

**Critical Failure Point 2: content-assembler receives partial results (one parallel skill times out)**
- *Risk:* Assembly fails or produces incomplete output missing social snippets or images
- *Recovery Strategy:* Implement graceful degradation with configurable requirements. Social snippets are "nice to have," so proceed with placeholder noting "social optimization pending." Images are more critical—implement retry with shorter timeout, then proceed with "image suggestions unavailable" flag. Never block publication entirely for optimization failures.

**5. Perspective Justification:**

This design follows a **minimalist philosophy** with clear single-responsibility skills composed into a workflow. I chose minimalist over holistic for three reasons specific to a content management platform:

First, **reusability across content types**: A CMS handles diverse content—articles, product descriptions, announcements. Atomic skills like `seo-optimizer` and `social-snipper` work identically across content types, maximizing investment. A holistic "article-publisher" skill would need reimplementation for each content type.

Second, **independent evolution**: SEO best practices change frequently; social platforms add/remove features. Minimalist skills enable updating `seo-optimizer` without touching other skills. Platform team can improve one optimization without regression risk to others.

Third, **team scalability**: Different team members can own different skills. The SEO specialist maintains `seo-optimizer`; the social media team maintains `social-snipper`. Holistic design would require coordination for any change.

The trade-off is increased orchestration complexity—the chain configuration must be managed, and context preservation requires explicit metadata passing. For a platform with dedicated engineering resources, this overhead is acceptable given the reusability benefits.

**Evaluation Rubric:**

| Criterion | Excellent (5) | Proficient (4) | Developing (3) | Beginning (2) | Insufficient (1) |
|-----------|---------------|----------------|----------------|---------------|------------------|
| Chain Architecture | 4+ skills with clear responsibilities, logical hybrid composition | 3+ skills with reasonable composition | Skills defined but unclear responsibilities or composition | Fewer than 3 skills or incoherent flow | No coherent architecture |
| Interface Contracts | Specific fields for all skills enabling clear composition | Most contracts complete, minor gaps | Contracts present but vague or incomplete | Generic or missing contracts | No contracts specified |
| Quality Analysis | All 4 dimensions analyzed with specific design evidence | 3-4 dimensions with evidence | 2-3 dimensions, limited evidence | 1-2 dimensions, generic discussion | No quality analysis |
| Failure Resilience | 2+ specific failures with concrete recovery strategies | 2 failures with reasonable recovery | 1 failure with recovery or generic discussion | Acknowledges failures exist | No failure consideration |
| Perspective Justification | Clear choice with multiple platform-specific reasons | Clear choice with some reasoning | Position stated without strong justification | Unclear position | No justification |

**Understanding Gap Indicator:**
If response lacks depth in specific areas:
- Weak chain architecture → Review Core Concepts - Concept 3 on composition patterns
- Vague contracts → Review Flashcard 2 and Practice Problem 3 on data flow
- Missing quality analysis → Review Critical Analysis section
- No failure handling → Review Practice Problem 5 on chain debugging
- Weak perspective → Review Theoretical Framework and Scholarly Perspectives

---

## Diagnostic Feedback & Review Recommendations

### Performance-Based Review Guide

| If Incorrect On... | Gap Identified | Review Section | Priority |
|--------------------|----------------|----------------|----------|
| Question 1 | Core definition confusion | Core Concepts - Concept 1 | Critical |
| Question 2 | Composition pattern selection | Core Concepts - Concept 3 | High |
| Question 3 | Invocation pattern application | Core Concepts - Concept 2 | Medium |
| Question 4 | Theoretical foundation application | Theoretical Framework | Medium |
| Question 5 | Integrated skill design | All sections | Low (cumulative) |

### Targeted Review Recommendations

#### For Multiple Choice Errors (Questions 1-2)
**Indicates:** Foundational concept gaps requiring systematic review
**Action Steps:**
1. Re-read Core Concepts - Concepts 1, 2, and 3 in study notes
2. Study the Concept Map—trace connections from Agent Skill to Composition
3. Review Flashcards 1 and 2 using spaced repetition
4. Attempt Practice Problems 1 and 3 before retaking quiz

**Focus On:** Distinguishing structural characteristics of skills from implementation details

#### For Short Answer Errors (Questions 3-4)
**Indicates:** Difficulty applying concepts to scenarios
**Action Steps:**
1. Review Theoretical Framework section for principle definitions
2. Study Practice Problem 2 (invocation patterns) and Problem 4 (complete design)
3. Use Concept Map Pathway 2 (Theory-First) to connect principles to applications
4. Practice writing skill specifications for simple use cases

**Focus On:** Connecting abstract principles (cognitive load, modularity) to concrete design decisions

#### For Essay Weakness (Question 5)
**Indicates:** Integration and synthesis challenges
**Action Steps:**
1. Complete all five Practice Problems to build component skills
2. Study Concept Map Critical Path for minimum viable understanding
3. Review the Case Study in Practical Applications section
4. Attempt designing a complete skill chain for a different scenario

**Focus On:** Building complete designs from component decisions rather than memorizing templates

### Mastery Level Interpretation

| Score | Level | Interpretation | Next Steps |
|-------|-------|----------------|------------|
| 5/5 | **Expert** | Strong mastery of skill design concepts | Ready for production skill development; consider extending to multi-agent systems |
| 4/5 | **Proficient** | Good understanding with minor gaps | Review indicated gap area; proceed to implementation with mentorship |
| 3/5 | **Developing** | Moderate understanding; application difficulties | Systematic review recommended before complex skill design; complete all practice problems |
| 2/5 | **Foundational** | Significant gaps in core concepts | Re-study from Concept Map Critical Path; focus on definitions before applications |
| 1/5 or below | **Beginning** | Requires comprehensive review | Restart from Lesson 1 study notes; use Pathway 1 (Foundational) from concept map |

---

## Skill Chain Traceability

```
Lesson 1 (Source)
    │
    │  Topic: Creating Agent Skills
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Study Notes                                  │
│  Sections: Executive Summary, 3 Core Concepts, Theoretical          │
│  Framework, Practical Applications, Critical Analysis               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────────────────────┐
        │                      │                                      │
        ▼                      ▼                                      ▼
┌───────────────┐    ┌─────────────────┐                    ┌─────────────────┐
│  Concept Map  │    │   Flashcards    │                    │    Practice     │
│               │    │                 │                    │    Problems     │
│  18 concepts  │    │  5 cards        │                    │                 │
│  28 relations │    │  2E/2M/1H       │                    │  5 problems     │
│  4 pathways   │    │                 │                    │  1W/2S/1C/1D    │
└───────┬───────┘    └────────┬────────┘                    └────────┬────────┘
        │                     │                                      │
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

### Cross-Reference Matrix

| Quiz Question | Concept Map Node | Flashcard | Practice Problem |
|---------------|------------------|-----------|------------------|
| Q1 (MC) | Agent Skill (Critical) | Card 1 | P1 (Warm-Up) |
| Q2 (MC) | Skill Composition (High) | Card 2 | P3 (Skill-Builder) |
| Q3 (SA) | Invocation Pattern (High) | Card 3 | P2 (Skill-Builder) |
| Q4 (SA) | Theory Cluster (Medium) | Card 4 | P4 (Challenge) |
| Q5 (Essay) | Full Integration | Card 5 | P4, P5 (Challenge, Debug) |

---

*Generated from Lesson 1: Creating Agent Skills | Quiz Skill*
