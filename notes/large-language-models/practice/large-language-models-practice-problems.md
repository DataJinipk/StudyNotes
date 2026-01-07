# Practice Problems: Large Language Models

**Source:** notes/large-language-models/large-language-models-study-notes.md
**Concept Map Reference:** notes/large-language-models/concept-maps/large-language-models-concept-map.md
**Date Generated:** 2026-01-06
**Total Problems:** 5
**Distribution:** 1 Warm-Up | 2 Skill-Builder | 1 Challenge | 1 Debug/Fix

---

## Problem Distribution Strategy

| Problem | Type | Concepts Tested | Difficulty | Time Est. |
|---------|------|-----------------|------------|-----------|
| P1 | Warm-Up | Tokenization, Context Limits | Low | 10-15 min |
| P2 | Skill-Builder | Self-Attention Computation | Medium | 20-25 min |
| P3 | Skill-Builder | Prompt Engineering | Medium | 20-25 min |
| P4 | Challenge | RLHF Pipeline Design | High | 35-45 min |
| P5 | Debug/Fix | Inference Performance | Medium | 25-30 min |

---

## Problems

---

### Problem 1 | Warm-Up
**Concept:** Tokenization and Context Windows
**Source Section:** Core Concepts 4, 9
**Concept Map Node:** Tokenization (4), Context Window (4)
**Related Flashcard:** Card 1, Card 4
**Estimated Time:** 10-15 minutes

#### Problem Statement

You are building a document Q&A system using an LLM with the following specifications:
- Context window: 8,192 tokens
- Tokenizer: GPT-style BPE (approximately 4 characters per token for English text)
- System prompt: 200 tokens (fixed)
- Response reservation: 500 tokens

A user wants to upload a 35-page technical report (approximately 17,500 words) and ask questions about it.

**Tasks:**
1. Estimate how many tokens the 35-page report requires
2. Calculate the available context for the document after accounting for system prompt and response
3. Determine what percentage of the document can fit in a single context window
4. Propose a strategy to handle the full document

---

#### Hints

<details>
<summary>Hint 1 (Conceptual)</summary>

For English text, common estimates are:
- 1 token ≈ 4 characters ≈ 0.75 words
- Or equivalently: 1 word ≈ 1.33 tokens

Calculate total tokens needed, then compare to available budget.
</details>

<details>
<summary>Hint 2 (Approach)</summary>

Available context = Total window - System prompt - Response reservation

For the document: ~17,500 words × 1.33 tokens/word ≈ ?

Compare this to available context to find percentage.
</details>

<details>
<summary>Hint 3 (Solution Direction)</summary>

Document ≈ 23,275 tokens, Available ≈ 7,492 tokens

Strategies for exceeding context:
- Chunking with retrieval (RAG)
- Summarization + relevant sections
- Multi-turn conversation with different sections
</details>

---

#### Solution

**1. Token Estimate for Document:**
```
Words: 17,500
Tokens/word: ~1.33 (GPT-style BPE)
Total tokens: 17,500 × 1.33 ≈ 23,275 tokens
```

**2. Available Context:**
```
Total window:       8,192 tokens
System prompt:      - 200 tokens
Response reserve:   - 500 tokens
─────────────────────────────────
Available:          7,492 tokens
```

**3. Coverage Percentage:**
```
Coverage = 7,492 / 23,275 = 32.2%

Only about 1/3 of the document fits in a single context.
```

**4. Strategy for Full Document:**

| Strategy | Approach | Pros | Cons |
|----------|----------|------|------|
| **RAG (Recommended)** | Chunk document, embed chunks, retrieve relevant chunks for each query | Only load relevant context; scales to any document size | Requires embedding infrastructure; retrieval quality matters |
| **Hierarchical Summary** | Create section summaries + keep full relevant sections | Preserves important details | May lose nuance; extra preprocessing |
| **Sliding Window** | Process document in overlapping chunks | Simple implementation | No cross-section reasoning |
| **Map-Reduce** | Ask question to each chunk, combine answers | Comprehensive coverage | Slow; may produce inconsistent answers |

**Recommended Implementation (RAG):**
```
1. Chunk document into ~500 token segments with overlap
2. Generate embeddings for each chunk
3. On query: embed query → find top-k similar chunks → include in context
4. Context assembly:
   - System prompt (200 tokens)
   - Top 5 retrieved chunks (2,500 tokens)
   - User query (100 tokens)
   - Response space (500 tokens)
   Total: 3,300 tokens used (comfortable margin)
```

---

#### Common Mistakes

| Mistake | Why It's Wrong | Correct Approach |
|---------|----------------|------------------|
| Using characters instead of tokens | Tokens ≠ characters; BPE tokenization varies | Use word-to-token ratio (~1.33 for English) |
| Forgetting response reservation | Need space for model to generate output | Reserve 10-20% for response |
| Stuffing max context | Ignoring "lost in the middle" phenomenon | Use retrieval for relevant sections only |
| One-token-per-word assumption | Underestimates actual token count | GPT-style: ~1.33 tokens per word |

---

#### Extension Challenge

Calculate the token requirements for the same document in Chinese, where tokenization typically yields 1.5-2 tokens per character. How does this affect your context management strategy?

---

---

### Problem 2 | Skill-Builder
**Concept:** Self-Attention Mechanism
**Source Section:** Core Concepts 2
**Concept Map Node:** Self-Attention (7), Multi-Head Attention (3)
**Related Flashcard:** Card 1
**Estimated Time:** 20-25 minutes

#### Problem Statement

Given a simplified self-attention scenario with 3 tokens and embedding dimension 4:

**Input Embeddings (X):**
```
Token 1 "The":  [1.0, 0.0, 1.0, 0.0]
Token 2 "cat":  [0.0, 1.0, 0.0, 1.0]
Token 3 "sat":  [1.0, 1.0, 0.0, 0.0]
```

**Weight Matrices (simplified, same dimension):**
```
W_Q = W_K = W_V = Identity matrix (for simplicity)
```

**Tasks:**
1. Compute the Query, Key, and Value matrices
2. Calculate the attention scores (QK^T)
3. Apply scaling by √d_k where d_k = 4
4. Apply softmax to get attention weights (row-wise)
5. Compute the final output (Attention × V)
6. Interpret what the attention weights mean for Token 1

---

#### Hints

<details>
<summary>Hint 1 (Conceptual)</summary>

With identity weight matrices:
- Q = X × W_Q = X (queries equal embeddings)
- K = X × W_K = X (keys equal embeddings)
- V = X × W_V = X (values equal embeddings)

Attention scores = Q × K^T (dot products between all query-key pairs)
</details>

<details>
<summary>Hint 2 (Approach)</summary>

For a 3×4 Q matrix and 4×3 K^T matrix:
- Result is 3×3 attention score matrix
- Entry (i,j) = dot product of query_i and key_j

Softmax formula: softmax(x_i) = e^(x_i) / Σ e^(x_j)
</details>

<details>
<summary>Hint 3 (Solution Direction)</summary>

QK^T before scaling:
```
      The  cat  sat
The [  2    0    1  ]
cat [  0    2    1  ]
sat [  1    1    2  ]
```

After scaling by √4 = 2, divide each by 2.
Then apply softmax to each row.
</details>

---

#### Solution

**1. Compute Q, K, V:**
Since W_Q = W_K = W_V = Identity:
```
Q = K = V = X =
    [1.0, 0.0, 1.0, 0.0]  ← "The"
    [0.0, 1.0, 0.0, 1.0]  ← "cat"
    [1.0, 1.0, 0.0, 0.0]  ← "sat"
```

**2. Attention Scores (QK^T):**
Each entry (i,j) = dot product of row i of Q with row j of K
```
Score(The, The) = 1×1 + 0×0 + 1×1 + 0×0 = 2
Score(The, cat) = 1×0 + 0×1 + 1×0 + 0×1 = 0
Score(The, sat) = 1×1 + 0×1 + 1×0 + 0×0 = 1

Score(cat, The) = 0×1 + 1×0 + 0×1 + 1×0 = 0
Score(cat, cat) = 0×0 + 1×1 + 0×0 + 1×1 = 2
Score(cat, sat) = 0×1 + 1×1 + 0×0 + 1×0 = 1

Score(sat, The) = 1×1 + 1×0 + 0×1 + 0×0 = 1
Score(sat, cat) = 1×0 + 1×1 + 0×0 + 0×1 = 1
Score(sat, sat) = 1×1 + 1×1 + 0×0 + 0×0 = 2

QK^T =
      The  cat  sat
The [  2    0    1  ]
cat [  0    2    1  ]
sat [  1    1    2  ]
```

**3. Scaled Scores (÷ √d_k = √4 = 2):**
```
Scaled =
      The   cat   sat
The [ 1.0   0.0   0.5 ]
cat [ 0.0   1.0   0.5 ]
sat [ 0.5   0.5   1.0 ]
```

**4. Softmax (row-wise):**
```
Row 1 (The): softmax([1.0, 0.0, 0.5])
  e^1.0 = 2.718, e^0.0 = 1.0, e^0.5 = 1.649
  sum = 5.367
  = [0.506, 0.186, 0.307]

Row 2 (cat): softmax([0.0, 1.0, 0.5])
  = [0.186, 0.506, 0.307]

Row 3 (sat): softmax([0.5, 0.5, 1.0])
  e^0.5 = 1.649, e^0.5 = 1.649, e^1.0 = 2.718
  sum = 6.016
  = [0.274, 0.274, 0.452]

Attention Weights =
        The    cat    sat
The [ 0.506  0.186  0.307 ]
cat [ 0.186  0.506  0.307 ]
sat [ 0.274  0.274  0.452 ]
```

**5. Output (Attention × V):**
```
Output[The] = 0.506×[1,0,1,0] + 0.186×[0,1,0,1] + 0.307×[1,1,0,0]
            = [0.506, 0, 0.506, 0] + [0, 0.186, 0, 0.186] + [0.307, 0.307, 0, 0]
            = [0.813, 0.493, 0.506, 0.186]

Output[cat] = 0.186×[1,0,1,0] + 0.506×[0,1,0,1] + 0.307×[1,1,0,0]
            = [0.493, 0.813, 0.186, 0.506]

Output[sat] = 0.274×[1,0,1,0] + 0.274×[0,1,0,1] + 0.452×[1,1,0,0]
            = [0.726, 0.726, 0.274, 0.274]
```

**6. Interpretation for Token 1 ("The"):**
- Attention weights: [0.506, 0.186, 0.307] for [The, cat, sat]
- "The" attends most strongly to itself (50.6%)
- Second strongest attention to "sat" (30.7%)
- Weakest attention to "cat" (18.6%)
- The output for "The" is a weighted blend of all token embeddings, dominated by its own embedding plus contribution from "sat"

**This reflects:** Similar embeddings (The and sat share the first dimension) produce higher attention scores.

---

#### Common Mistakes

| Mistake | Why It's Wrong | Correct Approach |
|---------|----------------|------------------|
| Forgetting to scale by √d_k | Causes gradient issues; softmax too peaked | Always divide by √d_k before softmax |
| Column-wise softmax | Loses interpretability; incorrect aggregation | Softmax is applied row-wise (per query) |
| Matrix multiplication order | K^T comes second in QK^T | Attention = Q × K^T, then × V |
| Ignoring softmax normalization | Weights must sum to 1 per query | Verify each row sums to 1.0 |

---

#### Extension Challenge

Implement causal masking: Modify the attention computation so Token 1 can only attend to Token 1, Token 2 to Tokens 1-2, and Token 3 to all tokens. How do the outputs change?

---

---

### Problem 3 | Skill-Builder
**Concept:** Prompt Engineering
**Source Section:** Core Concepts 8
**Concept Map Node:** Prompting (5), In-Context Learning (3), Chain-of-Thought (2)
**Related Flashcard:** Card 3
**Estimated Time:** 20-25 minutes

#### Problem Statement

You're building a customer support automation system. The LLM needs to:
1. Classify incoming tickets into categories: [Billing, Technical, Account, Shipping, Other]
2. Determine urgency: [Low, Medium, High, Critical]
3. Extract key entities: customer ID, product name, order number (if mentioned)
4. Generate a suggested response template

**Sample Ticket:**
```
Subject: Can't access my account after payment
Body: Hi, I paid for the premium subscription yesterday (Order #ORD-2024-8891)
but I still can't access premium features. My username is john.doe@email.com
and customer ID is CUS-445566. The payment was $99. I need this fixed ASAP
as I have a deadline tomorrow. This is really frustrating!
```

**Tasks:**
1. Design a system prompt that handles all four requirements
2. Create a few-shot prompt with 2 examples showing expected output format
3. Identify potential failure modes and add guardrails
4. Propose how to validate the LLM's outputs programmatically

---

#### Hints

<details>
<summary>Hint 1 (Conceptual)</summary>

Structure your system prompt with:
- Clear role definition
- Explicit output format (JSON for parseability)
- Each requirement as a specific field
- Examples of edge cases
</details>

<details>
<summary>Hint 2 (Approach)</summary>

For few-shot examples:
- Choose examples that demonstrate different categories/urgencies
- Show extraction of entities including "not mentioned" cases
- Include suggested response that addresses the actual issue

For validation: Consider regex for order IDs, enum validation for categories.
</details>

<details>
<summary>Hint 3 (Solution Direction)</summary>

Failure modes to address:
- Hallucinating order numbers not in the ticket
- Inconsistent JSON format
- Overly casual response tone
- Missing urgency indicators (deadline = high urgency)

Add explicit instructions: "Only extract entities actually present in the ticket"
</details>

---

#### Solution

**1. System Prompt:**
```
You are a customer support ticket analyzer. For each ticket, provide analysis
in the exact JSON format specified below.

RULES:
- Category must be exactly one of: Billing, Technical, Account, Shipping, Other
- Urgency must be exactly one of: Low, Medium, High, Critical
  - Critical: Service down, security issue, explicit deadline within 24h
  - High: Explicit urgency language ("ASAP", "urgent"), deadline within 48h
  - Medium: Standard issue affecting functionality
  - Low: Questions, feedback, non-blocking issues
- Only extract entities ACTUALLY MENTIONED in the ticket. Use null if not present.
- Suggested response must be professional, empathetic, and address the specific issue

OUTPUT FORMAT (strict JSON):
{
  "category": "<category>",
  "urgency": "<urgency>",
  "entities": {
    "customer_id": "<id or null>",
    "order_number": "<number or null>",
    "product_name": "<name or null>",
    "email": "<email or null>"
  },
  "key_issue": "<one-sentence summary>",
  "suggested_response": "<professional response template>"
}
```

**2. Few-Shot Examples:**

```
EXAMPLE 1:
Ticket:
Subject: Wrong item delivered
Body: I ordered the blue XL t-shirt but received red medium. Order ORD-2024-1234.

Output:
{
  "category": "Shipping",
  "urgency": "Medium",
  "entities": {
    "customer_id": null,
    "order_number": "ORD-2024-1234",
    "product_name": "blue XL t-shirt",
    "email": null
  },
  "key_issue": "Customer received wrong item (wrong color and size)",
  "suggested_response": "Dear Customer,\n\nThank you for contacting us about order ORD-2024-1234. We sincerely apologize for sending the wrong item. We will ship the correct blue XL t-shirt immediately and include a prepaid return label for the incorrect item.\n\nBest regards,\nSupport Team"
}

---

EXAMPLE 2:
Ticket:
Subject: Question about features
Body: Hi! I'm considering upgrading to premium. What features are included? Thanks!

Output:
{
  "category": "Other",
  "urgency": "Low",
  "entities": {
    "customer_id": null,
    "order_number": null,
    "product_name": "premium",
    "email": null
  },
  "key_issue": "Pre-purchase inquiry about premium features",
  "suggested_response": "Dear Customer,\n\nThank you for your interest in our premium plan! Premium includes [Feature 1], [Feature 2], and [Feature 3]. I'd be happy to answer any specific questions.\n\nBest regards,\nSupport Team"
}

---

NOW ANALYZE THIS TICKET:
[Insert customer ticket here]
```

**3. Failure Modes and Guardrails:**

| Failure Mode | Risk | Guardrail |
|--------------|------|-----------|
| Hallucinated entities | Extract fake order numbers | Add rule: "Only extract entities with exact text match in ticket" |
| Wrong urgency | Miss deadline indicators | Add examples: "ASAP", "urgent", "deadline" → High/Critical |
| Inconsistent JSON | Breaks downstream parsing | Add: "Output ONLY valid JSON, no additional text" |
| Unprofessional tone | Matches customer's frustrated tone | Add: "Response must be calm, professional, empathetic regardless of ticket tone" |
| Category confusion | Technical vs Account overlap | Add: "Account = access/login/permissions; Technical = bugs/features/functionality" |
| Verbose responses | Inefficient, may include errors | Add: "Suggested response should be 3-5 sentences maximum" |

**4. Output Validation Strategy:**

```python
import json
import re

def validate_llm_output(output_text, original_ticket):
    try:
        result = json.loads(output_text)
    except json.JSONDecodeError:
        return {"valid": False, "error": "Invalid JSON"}

    errors = []

    # Category validation
    valid_categories = ["Billing", "Technical", "Account", "Shipping", "Other"]
    if result.get("category") not in valid_categories:
        errors.append(f"Invalid category: {result.get('category')}")

    # Urgency validation
    valid_urgencies = ["Low", "Medium", "High", "Critical"]
    if result.get("urgency") not in valid_urgencies:
        errors.append(f"Invalid urgency: {result.get('urgency')}")

    # Entity validation - check entities exist in original ticket
    entities = result.get("entities", {})

    if entities.get("order_number"):
        if entities["order_number"] not in original_ticket:
            errors.append(f"Hallucinated order number: {entities['order_number']}")

    if entities.get("customer_id"):
        if entities["customer_id"] not in original_ticket:
            errors.append(f"Hallucinated customer ID: {entities['customer_id']}")

    # Email format validation
    if entities.get("email"):
        email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        if not re.match(email_pattern, entities["email"]):
            errors.append(f"Invalid email format: {entities['email']}")

    # Required fields check
    required_fields = ["category", "urgency", "entities", "key_issue", "suggested_response"]
    for field in required_fields:
        if field not in result:
            errors.append(f"Missing required field: {field}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "result": result if len(errors) == 0 else None
    }
```

---

#### Common Mistakes

| Mistake | Why It's Wrong | Correct Approach |
|---------|----------------|------------------|
| Vague category definitions | LLM inconsistently classifies edge cases | Provide explicit rules: "Account = access issues, Technical = bugs" |
| No output format specification | Unparseable responses | Require strict JSON with exact field names |
| Ignoring tone matching | LLM may mirror customer frustration | Explicitly require professional tone |
| No entity hallucination check | LLM invents data not in ticket | Validate entities against source text |

---

#### Extension Challenge

Add support for multi-language tickets. How would you modify the prompt to handle tickets in Spanish or French while maintaining the same JSON output structure in English?

---

---

### Problem 4 | Challenge
**Concept:** RLHF Pipeline Design
**Source Section:** Core Concepts 7
**Concept Map Node:** RLHF (5), Reward Model (2), Alignment (2)
**Related Flashcard:** Card 2, Card 5
**Estimated Time:** 35-45 minutes

#### Problem Statement

Your company is developing an AI assistant for professional email writing. After SFT, the model writes grammatically correct emails but often:
- Uses overly formal/stiff language when casual is appropriate
- Provides generic responses instead of addressing specific points
- Sometimes includes irrelevant information

You've been tasked with designing an RLHF pipeline to improve email quality. You have access to:
- 100 professional raters (experienced business writers)
- The SFT-trained model
- 50,000 email writing prompts with context
- 3 months and reasonable compute budget

**Tasks:**
1. Design the data collection protocol for human preferences
2. Specify the reward model architecture and training approach
3. Describe the policy optimization strategy (addressing potential pitfalls)
4. Define metrics to evaluate improvement
5. Identify risks and propose mitigations

---

#### Hints

<details>
<summary>Hint 1 (Conceptual)</summary>

RLHF pipeline:
1. Generate multiple responses per prompt
2. Collect human preferences (A vs B)
3. Train reward model on preferences
4. Optimize policy with RL (PPO) to maximize reward

Key decisions: How many responses to compare? What criteria for raters? How to prevent reward hacking?
</details>

<details>
<summary>Hint 2 (Approach)</summary>

For comparison collection:
- Generate 2-4 responses per prompt
- Use temperature sampling for diversity
- Provide clear rating guidelines (not just "which is better" but specific criteria)

Reward model training:
- Bradley-Terry model for pairwise preferences
- Split data train/val to detect overfitting
</details>

<details>
<summary>Hint 3 (Solution Direction)</summary>

Pitfalls to address:
- Reward hacking: Model finds shortcuts (e.g., always being verbose)
- KL divergence explosion: Policy moves too far from SFT
- Rater inconsistency: Need calibration and quality control

Metrics: Win rate vs. SFT baseline, task completion rate, user satisfaction in A/B test
</details>

---

#### Solution

**1. Data Collection Protocol:**

**Prompt Sampling:**
```
From 50K prompts, sample:
- 30K for reward model training
- 10K for reward model validation
- 10K for policy evaluation (held out)

Stratify by:
- Email type (request, response, follow-up, introduction)
- Formality level (casual, professional, formal)
- Length requirement (brief, standard, detailed)
```

**Response Generation:**
```
For each prompt, generate 4 responses using SFT model:
- Response A: temperature=0.3 (conservative)
- Response B: temperature=0.7 (moderate)
- Response C: temperature=1.0 (creative)
- Response D: temperature=0.7, different random seed

Present pairs: (A,B), (A,C), (B,D) → 3 comparisons per prompt
Total comparisons: 30K × 3 = 90K training comparisons
```

**Rater Guidelines:**
```
Rating Criteria (in order of importance):
1. Appropriateness: Does the tone match the context? (40%)
2. Specificity: Does it address the specific points raised? (30%)
3. Conciseness: Is information relevant and well-organized? (20%)
4. Professionalism: Grammar, spelling, format (10%)

Instructions:
- Choose which email you would actually send
- If both are unusable, select "Neither" (collect for analysis)
- Consider the recipient relationship indicated in prompt
- Ignore minor typos; focus on substance and tone
```

**Quality Control:**
```
- 10% of comparisons are duplicates (measure consistency)
- 5% are "gold standard" with known correct answer
- Minimum 90% agreement required to remain in rater pool
- Each comparison rated by 2 raters; third tiebreaker if needed
- Weekly calibration sessions with feedback
```

**2. Reward Model Architecture:**

```
Architecture:
┌─────────────────────────────────────────────────────┐
│                    Input: Email Text                 │
│                          ↓                          │
│  ┌─────────────────────────────────────────────┐    │
│  │  Pre-trained LLM Encoder (frozen or LoRA)   │    │
│  │  (Same architecture as SFT model)           │    │
│  └─────────────────────────────────────────────┘    │
│                          ↓                          │
│              [CLS] token representation             │
│                          ↓                          │
│  ┌─────────────────────────────────────────────┐    │
│  │          Linear Layer → Scalar Reward        │    │
│  └─────────────────────────────────────────────┘    │
│                          ↓                          │
│                   Reward Score (r)                  │
└─────────────────────────────────────────────────────┘
```

**Training Objective (Bradley-Terry):**
```
Loss = -log(σ(r_preferred - r_rejected))

Where:
- r_preferred = reward(winning_email)
- r_rejected = reward(losing_email)
- σ = sigmoid function

Intuition: Reward model should score preferred email higher
```

**Training Details:**
```
- Learning rate: 1e-5 (fine-tuning)
- Batch size: 64 comparisons
- Epochs: 1-2 (prevent overfitting)
- Validation accuracy target: >65% (human agreement is ~70-75%)
- Regularization: Weight decay 0.01, dropout 0.1
```

**3. Policy Optimization Strategy:**

```
Algorithm: PPO (Proximal Policy Optimization)

Objective:
maximize E[r(response)] - β × KL(π_new || π_sft)

Where:
- r(response) = reward model score
- KL term prevents diverging too far from SFT model
- β (KL coefficient) = 0.01-0.1, tuned dynamically
```

**Implementation Details:**
```python
# Pseudocode for PPO training loop
for batch in training_batches:
    # Generate responses with current policy
    prompts = sample_prompts(batch_size=256)
    responses = policy.generate(prompts, temperature=0.7)

    # Score with reward model
    rewards = reward_model.score(prompts, responses)

    # Compute KL penalty
    policy_logprobs = policy.log_prob(responses)
    sft_logprobs = sft_model.log_prob(responses)
    kl_penalty = policy_logprobs - sft_logprobs

    # Combined objective
    objective = rewards - beta * kl_penalty

    # PPO update with clipping
    ratio = exp(policy_logprobs - old_policy_logprobs)
    clipped_ratio = clip(ratio, 1-epsilon, 1+epsilon)
    loss = -min(ratio * objective, clipped_ratio * objective)

    policy.update(loss)

    # Adaptive KL coefficient
    if kl_penalty.mean() > target_kl * 1.5:
        beta *= 1.5  # Increase penalty
    elif kl_penalty.mean() < target_kl / 1.5:
        beta /= 1.5  # Decrease penalty
```

**Pitfall Mitigations:**

| Pitfall | Symptom | Mitigation |
|---------|---------|------------|
| Reward hacking | Model exploits reward model quirks (e.g., always verbose) | Monitor response length distribution; add length penalty if needed |
| Mode collapse | All responses become similar | Maintain temperature; add entropy bonus to objective |
| KL explosion | Policy diverges too far from SFT | Adaptive β; hard KL constraint; early stopping |
| Reward model overfitting | Validation accuracy drops during RL | Use held-out prompts; retrain reward model periodically |

**4. Evaluation Metrics:**

| Metric | Measurement Method | Target |
|--------|-------------------|--------|
| Win rate vs. SFT | Human eval: RLHF vs SFT responses | >60% preference for RLHF |
| Reward model score | Average reward on held-out set | Improve from baseline |
| Response diversity | Distinct n-grams, embedding variance | No decrease from SFT |
| Task completion | % of emails rated "would send" | >85% |
| Criteria-specific | Rate tone, specificity, conciseness separately | Improve on each |
| A/B test | Real users choose RLHF vs. SFT assistant | Significant preference |

**Evaluation Protocol:**
```
1. Sample 500 prompts from held-out set
2. Generate responses from both SFT and RLHF models
3. Blind human evaluation (different raters from training)
4. Compute win/loss/tie rates
5. Statistical significance test (p < 0.05)
6. Qualitative analysis of losses (what does RLHF still get wrong?)
```

**5. Risks and Mitigations:**

| Risk | Impact | Mitigation |
|------|--------|------------|
| Rater bias | Model learns specific rater preferences, not general quality | Diverse rater pool; stratified sampling; regular rotation |
| Sycophancy | Model agrees with user too readily; loses objectivity | Include adversarial prompts; train on preference for honest disagreement |
| Reward model errors | Model optimizes flawed proxy; produces high-reward but bad outputs | Ensemble reward models; periodic human eval checkpoints |
| Distribution shift | Good on training prompts, fails on production | Diverse prompt collection; deployment monitoring; fallback to SFT |
| Prompt injection | Users manipulate model via adversarial prompts | Red-teaming; input validation; safety layer |
| Over-optimization | Model "games" the reward model | Early stopping; KL constraint; fresh reward model training |

---

#### Common Mistakes

| Mistake | Why It's Wrong | Correct Approach |
|---------|----------------|------------------|
| Single rater per comparison | High variance; captures individual preference | 2+ raters with tiebreaker |
| Training reward model too long | Overfits to training comparisons | 1-2 epochs max; monitor validation |
| Ignoring KL constraint | Policy diverges; loses SFT capabilities | Adaptive β; monitor KL throughout |
| Using only reward score as metric | Reward model is imperfect proxy | Include human evaluation on held-out |

---

#### Extension Challenge

Design a "Constitutional AI" alternative: Instead of human comparisons, define principles (e.g., "prefer concise over verbose", "match tone to context") and have an LLM rate responses against these principles. What are the tradeoffs compared to human RLHF?

---

---

### Problem 5 | Debug/Fix
**Concept:** Inference Performance Optimization
**Source Section:** Core Concepts 10
**Concept Map Node:** Generation (5), KV Cache (3), Quantization (2)
**Related Flashcard:** Card 4
**Estimated Time:** 25-30 minutes

#### Problem Statement

You've deployed an LLM-based chatbot with the following production issues:

**System Specs:**
- Model: 7B parameter LLM
- GPU: NVIDIA A10 (24GB VRAM)
- Framework: Transformers + vLLM
- Context window: 4096 tokens

**Observed Issues:**
```
Issue 1: High latency
- Average time-to-first-token (TTFT): 2.5 seconds
- Average tokens per second (TPS): 15 tokens/sec
- Target: TTFT < 500ms, TPS > 50

Issue 2: Memory errors during peak
- OOM errors when concurrent requests exceed 8
- Target: Support 20 concurrent requests

Issue 3: Inconsistent latency
- P50 latency: 3 seconds
- P99 latency: 25 seconds
- Some requests timeout (>30s)
```

**Current Configuration:**
```python
model = AutoModelForCausalLM.from_pretrained(
    "model-7b",
    torch_dtype=torch.float32,  # Full precision
    device_map="auto"
)

# Generation config
generation_config = {
    "max_new_tokens": 512,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
}

# Serving: single request at a time
for request in requests:
    output = model.generate(**request, **generation_config)
```

**Tasks:**
1. Diagnose the root causes of each issue
2. Propose specific fixes with implementation details
3. Estimate the expected improvement for each fix
4. Identify tradeoffs or risks of your optimizations
5. Design a monitoring strategy to verify improvements

---

#### Hints

<details>
<summary>Hint 1 (Conceptual)</summary>

Root cause analysis:
- Issue 1: float32 uses 2x memory/compute vs float16; no batching
- Issue 2: No quantization; continuous batching not enabled
- Issue 3: Variable input/output lengths; no request timeout handling
</details>

<details>
<summary>Hint 2 (Approach)</summary>

Key optimizations:
1. Precision: float32 → float16 or int8 quantization
2. Batching: Enable continuous batching (vLLM does this)
3. KV cache: Ensure efficient memory management
4. Request management: Timeouts, priority queuing
</details>

<details>
<summary>Hint 3 (Solution Direction)</summary>

Expected improvements:
- float16: ~2x throughput, 50% memory
- int8 quantization: additional 50% memory, slight quality tradeoff
- Continuous batching: 3-5x throughput on concurrent requests
- KV cache optimization: Enables more concurrent requests
</details>

---

#### Solution

**1. Root Cause Diagnosis:**

| Issue | Symptom | Root Cause |
|-------|---------|------------|
| High TTFT | 2.5s first token | float32 = slow computation; no prefix caching |
| Low TPS | 15 tok/s | float32 = 2x memory bandwidth; sequential processing |
| OOM at 8 requests | Memory errors | float32 (28GB for 7B) > 24GB GPU; no memory optimization |
| Inconsistent latency | P99 >> P50 | No continuous batching; long requests block short ones |
| Timeouts | >30s some requests | No timeout handling; large outputs not limited |

**Memory Analysis:**
```
7B parameters × 4 bytes (float32) = 28GB model weights
24GB GPU VRAM available
→ Model barely fits; no room for KV cache or batch processing

With float16: 7B × 2 bytes = 14GB
→ 10GB available for KV cache and batching
```

**2. Proposed Fixes:**

**Fix A: Precision Reduction**
```python
# Change from float32 to float16
model = AutoModelForCausalLM.from_pretrained(
    "model-7b",
    torch_dtype=torch.float16,  # Half precision
    device_map="auto"
)

# Or use bfloat16 for better numerical stability
model = AutoModelForCausalLM.from_pretrained(
    "model-7b",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

**Expected Improvement:**
- Memory: 28GB → 14GB (50% reduction)
- Throughput: ~2x (memory bandwidth is bottleneck)
- Quality: Negligible loss for inference

---

**Fix B: INT8 Quantization**
```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)

model = AutoModelForCausalLM.from_pretrained(
    "model-7b",
    quantization_config=quantization_config,
    device_map="auto"
)
```

**Expected Improvement:**
- Memory: 28GB → 7GB (75% reduction)
- Throughput: Slight additional improvement
- Quality: Minor degradation (~1-2% on benchmarks)

---

**Fix C: Use vLLM for Continuous Batching**
```python
from vllm import LLM, SamplingParams

# vLLM handles continuous batching, KV cache management, and optimizations
llm = LLM(
    model="model-7b",
    dtype="float16",           # Or "auto" for bfloat16 if supported
    tensor_parallel_size=1,    # Single GPU
    gpu_memory_utilization=0.9,  # Use 90% of GPU memory
    max_num_seqs=32,           # Max concurrent sequences
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
)

# Batch processing
outputs = llm.generate(prompts, sampling_params)
```

**Expected Improvement:**
- Throughput: 3-5x for concurrent requests
- Concurrency: 20+ simultaneous requests supported
- TTFT: <500ms with prefix caching

---

**Fix D: Request Management**
```python
import asyncio
from vllm import AsyncLLMEngine

# Async engine for production
engine = AsyncLLMEngine.from_engine_args(engine_args)

async def generate_with_timeout(prompt, timeout=30):
    try:
        result = await asyncio.wait_for(
            engine.generate(prompt, sampling_params),
            timeout=timeout
        )
        return result
    except asyncio.TimeoutError:
        return {"error": "Request timeout", "partial_result": None}

# Priority queue for different request types
from heapq import heappush, heappop

class PriorityRequestQueue:
    def __init__(self):
        self.queue = []

    def add_request(self, request, priority):
        # Lower number = higher priority
        # Priority 1: Paid users, short prompts
        # Priority 2: Free users
        # Priority 3: Batch/background jobs
        heappush(self.queue, (priority, request))

    def get_next(self):
        return heappop(self.queue)[1]
```

---

**Fix E: Output Length Control**
```python
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256,     # Reduced from 512
    stop=["\n\n", "---", "[END]"],  # Early stopping tokens
)

# Streaming for better perceived latency
async def stream_response(prompt):
    async for output in engine.generate(prompt, sampling_params, stream=True):
        yield output.text  # Send tokens as generated
```

**3. Expected Improvements Summary:**

| Fix | TTFT | TPS | Concurrency | Quality |
|-----|------|-----|-------------|---------|
| Baseline | 2.5s | 15 | 8 | 100% |
| float16 | 1.2s | 30 | 12 | 99.9% |
| +INT8 | 1.0s | 35 | 20 | 98-99% |
| +vLLM | 0.4s | 80 | 25+ | 98-99% |
| +Timeouts | 0.4s | 80 | 25+ | 98-99% (controlled) |

**Final Configuration:**
```
TTFT: 400ms (target: 500ms) ✓
TPS: 80 tokens/sec (target: 50) ✓
Concurrency: 25+ (target: 20) ✓
P99/P50 ratio: ~3x (down from ~8x)
```

**4. Tradeoffs and Risks:**

| Optimization | Tradeoff | Mitigation |
|--------------|----------|------------|
| float16 | Potential numerical instability in edge cases | Use bfloat16; monitor for quality regression |
| INT8 | Quality degradation on complex reasoning | Benchmark on production-like prompts; fallback to float16 for critical requests |
| Continuous batching | More complex deployment; debugging harder | Comprehensive logging; gradual rollout |
| Timeouts | Incomplete responses for complex queries | Inform users; offer retry; queue for later processing |
| Max token limits | May cut off legitimate long responses | Analyze response lengths; adjust per use case |

**5. Monitoring Strategy:**

```python
# Metrics to track
metrics = {
    # Latency
    "ttft_p50": Histogram("ttft_p50_seconds"),
    "ttft_p99": Histogram("ttft_p99_seconds"),
    "tps": Gauge("tokens_per_second"),

    # Throughput
    "requests_per_second": Counter("requests_total"),
    "concurrent_requests": Gauge("concurrent_requests"),

    # Errors
    "oom_errors": Counter("oom_errors_total"),
    "timeout_errors": Counter("timeout_errors_total"),

    # Quality (sample-based)
    "response_length_avg": Gauge("response_length"),
    "user_rating": Gauge("user_rating_avg"),

    # Resource utilization
    "gpu_memory_used": Gauge("gpu_memory_bytes"),
    "gpu_utilization": Gauge("gpu_utilization_percent"),
}

# Alerting thresholds
alerts = {
    "ttft_p99 > 1s": "Critical",
    "tps < 40": "Warning",
    "gpu_memory > 22GB": "Warning",
    "timeout_rate > 5%": "Critical",
}

# A/B test for quality validation
def quality_check(sample_rate=0.01):
    # Sample 1% of requests
    # Run through both old and new pipeline
    # Compare outputs using quality metrics
    pass
```

---

#### Common Mistakes

| Mistake | Why It's Wrong | Correct Approach |
|---------|----------------|------------------|
| Jumping to INT4 quantization | Quality degradation too severe for many use cases | Start with float16, then INT8; test quality |
| Ignoring KV cache | Memory not available for batching | vLLM manages this automatically; ensure memory headroom |
| No timeout handling | One slow request blocks everything | Implement async processing with timeouts |
| Not monitoring quality | Optimizations may degrade output | Sample-based quality checks; user feedback |

---

#### Extension Challenge

The team wants to deploy on a machine with 2× A10 GPUs. Design a tensor parallelism strategy: How would you split the model across GPUs? What additional considerations arise for multi-GPU inference?

---

---

## Skills Integration Summary

This practice problem set integrates with the full skill chain:

```
Study Notes (10 Concepts)
        ↓
Concept Map (24 concepts, 38 relationships)
        ↓
Flashcards (5 cards: 2E/2M/1H)
        ↓
Practice Problems ← YOU ARE HERE
        ↓
Quiz (5 questions: 2MC/2SA/1E)
```

| Problem | Concepts Practiced | Prepares For |
|---------|-------------------|--------------|
| P1 | Tokenization, Context Window | Quiz Q3, Q4 |
| P2 | Self-Attention, Transformer | Quiz Q1 |
| P3 | Prompting, In-Context Learning | Quiz Q2, Q5 |
| P4 | RLHF, Alignment | Quiz Q4, Q5 |
| P5 | Inference, Deployment | Quiz Q3 |
