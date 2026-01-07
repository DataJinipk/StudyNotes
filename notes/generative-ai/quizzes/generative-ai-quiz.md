# Assessment Quiz: Generative AI

**Source Material:** notes/generative-ai/flashcards/generative-ai-flashcards.md
**Practice Problems:** notes/generative-ai/practice/generative-ai-practice-problems.md
**Concept Map:** notes/generative-ai/concept-maps/generative-ai-concept-map.md
**Original Study Notes:** notes/generative-ai/generative-ai-study-notes.md
**Date Generated:** 2026-01-07
**Total Questions:** 5
**Estimated Completion Time:** 30-40 minutes
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
**Concept Tested:** Diffusion Models
**Source Section:** Core Concepts 3, 5
**Concept Map Node:** Diffusion (8 connections)
**Related Flashcard:** Card 3
**Related Practice Problem:** P3

In a diffusion model for image generation, what does the neural network learn to predict during training, and why is this approach preferred over directly predicting the clean image?

A) The network predicts the clean image directly because this provides the clearest training signal and fastest convergence

B) The network predicts the noise added at each timestep because predicting noise is easier than predicting complex image structure, and this formulation connects to score matching theory

C) The network predicts the next timestep's noisy image because this enables autoregressive generation similar to language models

D) The network predicts a binary mask indicating which pixels contain noise because this allows selective denoising of different image regions

---

### Question 2 | Multiple Choice
**Cognitive Level:** Remember/Understand
**Concept Tested:** RAG and Hallucination Mitigation
**Source Section:** Core Concepts 8, 9
**Concept Map Node:** RAG (6 connections), Hallucination (5 connections)
**Related Flashcard:** Card 2
**Related Practice Problem:** P2, P5

A company deploys a customer support chatbot using RAG to answer questions about their products. Users report that the bot sometimes states features that don't exist. Which of the following correctly explains why RAG doesn't fully eliminate hallucination and proposes an appropriate mitigation?

A) RAG fails because the retrieval step always returns irrelevant documents; the fix is to increase top-k to retrieve more documents

B) RAG fails because LLMs can misinterpret or fabricate information even from correct sources; the fix is to add citation verification that checks claims against retrieved documents

C) RAG fails because vector embeddings cannot capture semantic meaning; the fix is to use keyword search instead of semantic search

D) RAG fails because the LLM context window is too small; the fix is to use a larger model with 1M+ context tokens

---

### Question 3 | Short Answer
**Cognitive Level:** Apply/Analyze
**Concept Tested:** Classifier-Free Guidance
**Source Section:** Core Concepts 3, 5
**Concept Map Node:** CFG (4 connections)
**Related Flashcard:** Card 3
**Related Practice Problem:** P3
**Expected Response Length:** 3-4 sentences

A designer using Stable Diffusion notices that setting guidance scale to 3 produces creative but loosely prompt-adherent images, while scale 20 produces over-saturated images with artifacts. Explain the mathematical intuition behind classifier-free guidance and why extreme values produce these effects.

---

### Question 4 | Short Answer
**Cognitive Level:** Apply/Analyze
**Concept Tested:** Fine-tuning Strategy Selection
**Source Section:** Core Concepts 7
**Concept Map Node:** Fine-tuning (7), LoRA (5), RLHF (7)
**Related Flashcard:** Card 4
**Related Practice Problem:** P4
**Expected Response Length:** 3-4 sentences

A startup has 500 high-quality customer service conversations and wants to create a support bot that matches their brand voice. They have limited compute (one consumer GPU with 24GB VRAM) and need to fine-tune Llama 3 8B. Recommend the appropriate fine-tuning approach and explain why full fine-tuning would be problematic in this scenario.

---

### Question 5 | Essay
**Cognitive Level:** Evaluate/Synthesize
**Concepts Tested:** Complete Generative AI System Design
**Source Sections:** All Core Concepts
**Concept Map:** Full pathway traversal
**Related Flashcard:** Card 5
**Related Practice Problem:** P1, P2, P4
**Expected Response Length:** 1-2 paragraphs

A legal technology company wants to build an AI system that helps attorneys draft contracts. The system must: (1) generate contract clauses based on natural language descriptions, (2) ensure all generated text is grounded in the firm's approved clause library (no hallucinated legal language), (3) allow attorneys to control the formality and jurisdiction-specific language, and (4) maintain attorney-client privilege by keeping all data on-premise.

Design a complete solution addressing: (a) model architecture decisions (API vs self-hosted, which models); (b) how to ensure generated clauses are grounded in approved templates; (c) how to provide control over style and jurisdiction; (d) deployment architecture for privacy; and (e) what safety guardrails are essential for legal text generation.

**Evaluation Criteria:**
- [ ] Justifies model selection for legal domain and privacy constraints
- [ ] Designs RAG system for clause grounding
- [ ] Proposes control mechanisms for style/jurisdiction
- [ ] Addresses on-premise deployment requirements
- [ ] Identifies legal-specific safety concerns

---

## Answer Key

### Question 1 | Multiple Choice
**Correct Answer:** B

**Explanation:**
Diffusion models learn to predict the noise ε that was added to create a noisy image at each timestep:

**The Training Process:**
```
Clean image x₀ → Add noise ε → Noisy image xₜ
Model learns: εθ(xₜ, t) ≈ ε  (predict noise from noisy image + timestep)
Loss: ||ε - εθ(xₜ, t)||²
```

**Why Predict Noise Instead of Clean Image:**

| Aspect | Predicting Noise | Predicting Clean Image |
|--------|------------------|------------------------|
| **Difficulty** | Easier—noise is Gaussian with known statistics | Harder—images have complex structure |
| **Theory** | Connects to score matching: ε ∝ -∇log p(xₜ) | No direct theoretical connection |
| **Stability** | Loss is well-behaved across all timesteps | Different timesteps have very different targets |
| **Generation** | Can iteratively subtract predicted noise | Requires single-step reconstruction |

**Mathematical Connection:**
Predicting ε is equivalent to estimating the score function ∇log p(xₜ|x₀), which points toward the data distribution. This enables principled generation via Langevin dynamics.

**Why Other Options Are Incorrect:**
- A) Directly predicting clean images is harder and doesn't leverage score matching theory
- C) Diffusion is NOT autoregressive; it uses parallel spatial generation with iterative temporal refinement
- D) Noise affects all pixels continuously, not as a binary mask

---

### Question 2 | Multiple Choice
**Correct Answer:** B

**Explanation:**
RAG significantly reduces hallucination but doesn't eliminate it because:

**Why RAG Doesn't Fully Prevent Hallucination:**

| Failure Mode | Description | Example |
|--------------|-------------|---------|
| **Misinterpretation** | LLM incorrectly synthesizes retrieved information | Combines two product features that don't exist together |
| **Fabrication from patterns** | LLM generates plausible-sounding additions | Invents a feature that "sounds right" for the product |
| **Wrong document used** | Retrieval returns similar but incorrect docs | Returns competitor's spec sheet |
| **Over-generalization** | Applies retrieved info too broadly | States all models have a feature only one has |

**The Citation Verification Fix:**
```python
def verify_response(response, retrieved_sources):
    # Extract factual claims from response
    claims = extract_claims(response)

    for claim in claims:
        # Check if claim is supported by sources
        if not verify_claim_in_sources(claim, retrieved_sources):
            flag_as_unverified(claim)

    return response_with_verification
```

This post-processing step catches when the LLM generates information not present in the sources.

**Why Other Options Are Incorrect:**
- A) Increasing top-k retrieves more documents but doesn't fix misinterpretation; may introduce more noise
- C) Vector embeddings DO capture semantic meaning; this is why RAG works at all
- D) Context window size isn't the primary cause of hallucination in RAG

---

### Question 3 | Short Answer
**Model Answer:**

Classifier-free guidance (CFG) works by computing two predictions at each denoising step: one conditioned on the prompt (εcond) and one unconditional (εuncond). The final prediction is: ε̃ = εuncond + w × (εcond - εuncond), where w is the guidance scale. This formula amplifies the "direction" of the conditioning—how much the prompt changes the generation compared to unconditional output.

At scale w=3, the conditioning signal is only moderately amplified, so the model retains diversity but follows the prompt loosely. At w=20, the conditioning is extremely amplified, pushing generation so aggressively toward "prompt-like" features that it overshoots, producing oversaturated colors and repetitive patterns. The optimal range (7-8 for most models) balances prompt adherence with image quality. The artifacts at high scales occur because we're extrapolating beyond the natural distribution the model learned.

**Key Components Required:**
- [ ] Explains the ε̃ = εuncond + w × (εcond - εuncond) formula or its intuition
- [ ] Describes guidance scale as amplification of conditioning direction
- [ ] Explains why low values produce loose prompt adherence
- [ ] Explains why extreme values cause artifacts (over-amplification/extrapolation)

**Partial Credit Guidance:**
- Full credit: All components with clear explanation of amplification mechanism
- Partial credit: Understands tradeoff but vague on mathematical mechanism
- No credit: Cannot explain relationship between guidance scale and output quality

---

### Question 4 | Short Answer
**Model Answer:**

The recommended approach is **QLoRA (Quantized LoRA)**, which combines 4-bit quantization of the base model with low-rank adaptation matrices. With QLoRA, the 8B parameter model can be loaded in approximately 4-6GB VRAM (4-bit quantized), leaving room for the small trainable LoRA matrices and training overhead. Only ~0.5-1% of parameters are trained (the LoRA adapters), making it feasible on 24GB consumer hardware.

Full fine-tuning would be problematic for several reasons: (1) Llama 3 8B requires ~32GB just to load in fp16, exceeding the 24GB limit; (2) full fine-tuning needs additional memory for optimizer states (2-4x model size for Adam), gradients, and activations; (3) with only 500 examples, full fine-tuning risks catastrophic forgetting of the model's general capabilities, whereas LoRA preserves the frozen base model's knowledge while learning task-specific adaptations.

**Key Components Required:**
- [ ] Recommends QLoRA or LoRA as the appropriate approach
- [ ] Explains memory savings from quantization + parameter efficiency
- [ ] Identifies VRAM constraint as blocking full fine-tuning
- [ ] Mentions catastrophic forgetting or preservation of base capabilities

**Partial Credit Guidance:**
- Full credit: Correct recommendation with memory and forgetting justification
- Partial credit: Correct recommendation but incomplete justification
- No credit: Recommends full fine-tuning or doesn't address constraints

---

### Question 5 | Essay
**Model Answer:**

**(a) Model Architecture Decisions:**

For privacy and attorney-client privilege requirements, I would deploy a **self-hosted open-weights model** rather than API services. Llama 3 70B or Mixtral 8x22B are strong choices for legal text generation. The key constraint is that contract text contains privileged information that cannot be sent to third-party APIs (OpenAI, Anthropic) without potentially waiving privilege. Self-hosting using vLLM or TGI on on-premise GPU servers (4× A100 80GB or 8× H100) ensures all data stays within the firm's infrastructure.

| Deployment | Pros | Cons |
|------------|------|------|
| **Self-hosted Llama 3 70B** | Full data control, no third-party risk | Higher ops burden, needs GPU hardware |
| **API (GPT-4, Claude)** | Best capability | Privilege risk, compliance issues |

**(b) Grounding with Approved Clause Library (RAG):**

Implement a RAG system using the firm's approved clause templates as the knowledge base:

```python
class ContractClauseRAG:
    def __init__(self):
        self.vector_db = load_clause_library()  # Approved templates only

    def generate_clause(self, description, jurisdiction):
        # Retrieve similar approved clauses
        templates = self.vector_db.search(
            query=description,
            filter={"jurisdiction": jurisdiction, "approved": True},
            top_k=5
        )

        # Generate based ONLY on retrieved templates
        prompt = f"""Based on the following approved clause templates,
        draft a clause for: {description}

        APPROVED TEMPLATES:
        {format_templates(templates)}

        Generate a clause using ONLY language from these approved templates.
        Include [TEMPLATE_ID] citations for each phrase used."""

        return llm.generate(prompt)
```

Add post-generation verification that checks if generated text matches patterns in the approved library, flagging novel language for attorney review.

**(c) Style and Jurisdiction Control:**

Provide control through:
1. **System prompts:** Define formality level (formal/standard/plain English) as part of the prompt
2. **Jurisdiction filtering:** Metadata tags on clause templates (e.g., "jurisdiction": "Delaware", "California") filter retrieval to jurisdiction-appropriate language
3. **Style LoRA:** Optionally train small LoRA adapters on examples of different formality levels that can be swapped at inference
4. **Few-shot examples:** Include 1-2 example clauses in the prompt matching desired style

**(d) On-Premise Deployment Architecture:**

```
┌─────────────────────────────────────────────────────────┐
│                   Firm's Data Center                     │
│                                                          │
│  ┌──────────────┐     ┌──────────────┐                  │
│  │  Attorney    │────►│   Gateway    │                  │
│  │  Interface   │     │   + Auth     │                  │
│  └──────────────┘     └──────┬───────┘                  │
│                              │                           │
│         ┌────────────────────┴────────────────┐         │
│         ▼                                     ▼          │
│  ┌──────────────┐                    ┌──────────────┐   │
│  │ Vector DB    │                    │ vLLM Server  │   │
│  │ (Clause Lib) │                    │ Llama 3 70B  │   │
│  │ Encrypted    │                    │ 4× A100 GPU  │   │
│  └──────────────┘                    └──────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │              Audit Log (Encrypted)                │   │
│  │  - All queries and responses logged               │   │
│  │  - 7-year retention for legal compliance          │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

All components on-premise, encrypted at rest and in transit, no external API calls.

**(e) Legal-Specific Safety Guardrails:**

| Guardrail | Purpose | Implementation |
|-----------|---------|----------------|
| **No legal advice** | Model summarizes/drafts, doesn't advise | System prompt: "You draft clauses; attorneys make legal judgments" |
| **Clause verification** | Ensure output matches approved patterns | Post-processing NER for legal phrases; flag novel language |
| **Jurisdiction validation** | Don't mix laws from different jurisdictions | Check generated text against jurisdiction-specific requirements |
| **Confidentiality markers** | Preserve privilege | Add "PRIVILEGED AND CONFIDENTIAL" headers; no external logging |
| **Human-in-loop** | Attorney reviews all generated text | UI requires attorney approval before clause is finalized |
| **Audit trail** | Document generation provenance | Log: who requested, what templates used, what was generated |

Critical: The system should never represent itself as providing legal advice. All outputs should be clearly marked as drafts requiring attorney review.

**Evaluation Rubric:**

| Criterion | Excellent (4) | Proficient (3) | Developing (2) | Beginning (1) |
|-----------|---------------|----------------|----------------|---------------|
| Model Selection | Self-hosted with privilege justification | Acknowledges privacy need | Mentions privacy vaguely | Uses external API |
| RAG Design | Approved-only retrieval with verification | RAG with approval filter | Basic RAG | No grounding |
| Control Mechanisms | Multiple (prompts, filtering, LoRA) | 2 control methods | 1 control method | No style control |
| Deployment | On-premise with encryption, audit | On-premise mentioned | Vague "private" hosting | Cloud deployment |
| Safety | 4+ legal-specific guardrails | 2-3 guardrails | Generic safety | None mentioned |

---

## Diagnostic Feedback & Review Recommendations

### Performance-Based Review Guide

| If Incorrect On... | Gap Identified | Review Section | Priority |
|--------------------|----------------|----------------|----------|
| Question 1 | Diffusion mechanics | Core Concepts 3 + Flashcard 3 | High |
| Question 2 | RAG limitations | Core Concepts 8, 9 + Practice P5 | High |
| Question 3 | CFG mathematics | Core Concepts 5 + Flashcard 3 | Medium |
| Question 4 | Fine-tuning selection | Core Concepts 7 + Practice P4 | Medium |
| Question 5 | System integration | All sections | Low |

### Targeted Review Recommendations

#### For Multiple Choice Errors (Questions 1-2)
**Indicates:** Foundational concept gaps in generation mechanisms
**Action:** Review:
- Study Notes: Core Concepts 3 (Diffusion) and 8 (Prompting/RAG)
- Flashcards: Cards 2 and 3
- Practice Problems: P2 (RAG) and P5 (Debug hallucination)
**Focus On:** Understanding WHY techniques work, not just what they do

#### For Short Answer Errors (Questions 3-4)
**Indicates:** Application difficulties with control and adaptation
**Action:** Practice:
- Practice Problems: P3 (diffusion control) and P4 (fine-tuning pipeline)
- Concept Map: Adaptation & Control cluster
**Focus On:** Connecting parameter choices to outcomes

#### For Essay Weakness (Question 5)
**Indicates:** Integration challenges across generative AI concepts
**Action:** Review interconnections:
- Concept Map: Full pathway traversal (all 4 pathways)
- Practice Problem P1 (model selection) and P2 (RAG implementation)
- Flashcard 5 (production system)
**Focus On:** Building complete systems with privacy and safety tradeoffs

### Mastery Indicators

- **5/5 Correct:** Strong mastery; ready for production generative AI development
- **4/5 Correct:** Good understanding; review indicated gap
- **3/5 Correct:** Moderate understanding; systematic review needed
- **2/5 or below:** Foundational gaps; restart from Concept Map Critical Path

---

## Skill Chain Traceability

```
Study Notes (Source) ──────────────────────────────────────────────────────┐
    │                                                                      │
    │  10 Core Concepts, 12 Key Terms, 4 Applications                      │
    │                                                                      │
    ├────────────┬────────────┬────────────┬────────────┐                  │
    │            │            │            │            │                  │
    ▼            ▼            ▼            ▼            ▼                  │
Concept Map  Flashcards   Practice    Quiz                                 │
    │            │        Problems      │                                  │
    │ 38 concepts│ 5 cards   │ 5 problems│ 5 questions                     │
    │ 55 rels    │ 2E/2M/1H  │ 1W/2S/1C/1D│ 2MC/2SA/1E                     │
    │ 4 pathways │           │           │                                 │
    │            │           │           │                                 │
    └─────┬──────┴─────┬─────┴─────┬─────┘                                 │
          │            │           │                                       │
          │ Centrality │ Practice  │                                       │
          │ → Card     │ → Quiz    │                                       │
          │ difficulty │ distractors│                                      │
          │            │           │                                       │
          └────────────┴───────────┴───────────────────────────────────────┘
                                   │
                          Quiz integrates ALL
                          upstream materials
```

---

## Complete 5-Skill Chain Summary

| Skill | Output | Key Contribution to Chain |
|-------|--------|---------------------------|
| study-notes-creator | 10 concepts, theory, applications | Foundation content |
| concept-map | 38 nodes, 55 relationships, 4 pathways | Structure + learning paths |
| flashcards | 5 cards (2E/2M/1H) | Memorization + critical concepts |
| practice-problems | 5 problems (1W/2S/1C/1D) | Procedural fluency + debugging |
| quiz | 5 questions (2MC/2SA/1E) | Assessment + diagnostic feedback |
