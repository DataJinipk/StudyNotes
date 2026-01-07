# Assessment Quiz: Lesson 7 - Generative AI

**Source:** Lessons/Lesson_7.md
**Subject Area:** AI Learning - Generative AI: Foundations, Architectures, and Applications
**Date Generated:** 2026-01-08
**Total Questions:** 5
**Estimated Time:** 35-45 minutes

---

## Instructions

This assessment evaluates your understanding of generative AI fundamentals, including model architectures, generation control, fine-tuning strategies, and deployment considerations. Answer all questions completely, showing your reasoning where applicable.

**Question Distribution:**
- Multiple Choice (2): Conceptual understanding (Remember/Understand)
- Short Answer (2): Application and analysis (Apply/Analyze)
- Essay (1): Synthesis and evaluation (Evaluate/Synthesize)

---

## Part A: Multiple Choice (10 points each)

### Question 1: Generative Model Architectures

**A team needs to generate high-quality images from text descriptions with the following requirements: (1) stable training without mode collapse, (2) ability to control generation through text prompts, and (3) diverse outputs from the same prompt. Which model architecture best satisfies these requirements?**

A) Generative Adversarial Network (GAN) with CLIP conditioning

B) Variational Autoencoder (VAE) with text encoder

C) Latent Diffusion Model with Classifier-Free Guidance

D) Autoregressive Transformer predicting image tokens

---

### Question 2: Fine-Tuning Strategy Selection

**A startup with limited compute (single 24GB GPU) wants to fine-tune a 13B parameter LLM on 10,000 customer service conversations. They need the fine-tuned model to maintain general language capabilities while specializing in their domain. Which approach is most appropriate?**

A) Full fine-tuning with gradient checkpointing and mixed precision

B) QLoRA with rank 16, targeting attention and MLP layers

C) Prompt tuning with 100 soft tokens prepended to each input

D) Knowledge distillation from the 13B model to a 1B model

---

## Part B: Short Answer (15 points each)

### Question 3: Sampling and Generation Control

**Context:** You are configuring a customer-facing chatbot that handles both factual queries (account balances, policy information) and creative tasks (writing personalized birthday messages).

**Tasks:**

a) Design a routing strategy that determines when to use factual vs. creative sampling parameters. What signals would you use to classify queries? (5 points)

b) Specify the temperature, top-p, and any other relevant parameters for each mode (factual and creative). Justify each choice. (5 points)

c) A user asks: "Write a haiku about my account balance of $1,234.56." This query combines both modes. How should the system handle it? (5 points)

---

### Question 4: Diffusion Model Analysis

**Context:** You are debugging a text-to-image pipeline using Stable Diffusion. Given the following configuration and outputs:

```
Prompt: "A photorealistic portrait of a scientist in a laboratory"
Negative prompt: ""
CFG scale: 12.0
Steps: 25
Sampler: Euler
```

**Results:** Images show correct subject matter but have oversaturated colors, slightly artificial skin tones, and occasional artifacts around edges.

**Tasks:**

a) Identify the most likely cause of the quality issues and explain the underlying mechanism. (5 points)

b) Propose a revised configuration that would address these issues while maintaining good prompt adherence. (5 points)

c) The client requests the ability to generate consistent images of the same scientist across multiple generations. What techniques or tools would enable this? (5 points)

---

## Part C: Essay (30 points)

### Question 5: RAG System Design for Enterprise Knowledge

**Prompt:** A multinational corporation wants to build an internal knowledge assistant that can answer employee questions by retrieving information from:
- 50,000+ policy documents (PDFs, Word files)
- Internal wiki pages (100,000+ articles)
- Recorded meeting transcripts (10,000+ hours)
- Slack message history (millions of messages)

The system must handle questions like:
- "What is our parental leave policy in Germany?"
- "Who was responsible for the Q3 marketing budget decision?"
- "Summarize the key points from last week's engineering all-hands"

**Your essay must address:**

1. **Architecture Design** (8 points)
   - Document processing pipeline for each source type
   - Embedding and indexing strategy
   - Retrieval approach for diverse content types
   - Generation component selection

2. **Chunking and Retrieval Strategy** (7 points)
   - How to chunk different document types effectively
   - Multi-stage retrieval for precision
   - Handling cross-document queries
   - Metadata filtering and access control

3. **Quality and Safety** (8 points)
   - Citation and source verification
   - Handling conflicting information across sources
   - Access control (who can see what)
   - Hallucination mitigation strategies

4. **Evaluation and Monitoring** (7 points)
   - Metrics for retrieval quality
   - Metrics for generation quality
   - Production monitoring approach
   - Continuous improvement strategy

**Evaluation Criteria:**
- Technical accuracy of architectural decisions
- Practical considerations for enterprise deployment
- Awareness of quality and safety challenges
- Coherent integration of components

**Word Limit:** 600-800 words

---

## Answer Key

### Question 1: Generative Model Architectures

**Correct Answer: C**

**Explanation:**

**Why Latent Diffusion with CFG (Option C) is best:**

| Requirement | How LDM+CFG Satisfies |
|-------------|----------------------|
| Stable training | Diffusion has no adversarial dynamics, converges reliably |
| Text control | CFG strengthens conditioning signal; proven text-to-image |
| Diverse outputs | Inherently stochastic; different seeds → different images |

**Why other options are inferior:**

- **Option A (GAN + CLIP):** GANs suffer from mode collapse and training instability. While CLIP can provide conditioning, GAN training remains challenging for high-quality diverse outputs.

- **Option B (VAE):** VAEs produce blurrier outputs due to the reconstruction loss encouraging averaging. Not state-of-the-art for image quality.

- **Option D (Autoregressive):** While capable (e.g., DALL-E 1), autoregressive image generation is slower and generally produces lower quality than diffusion for photorealistic images.

**Understanding Gap:** If you selected A, review the training dynamics of GANs vs. diffusion models and why diffusion is now preferred for image generation.

---

### Question 2: Fine-Tuning Strategy Selection

**Correct Answer: B**

**Explanation:**

**Why QLoRA (Option B) is most appropriate:**

| Constraint | How QLoRA Addresses |
|------------|---------------------|
| Single 24GB GPU | 13B model quantized to 4-bit ≈ 6.5GB, fits with LoRA |
| Maintain general capabilities | LoRA preserves frozen base weights, prevents forgetting |
| Domain specialization | Trainable adapters learn customer service patterns |
| 10K samples | Sufficient data for LoRA to learn domain-specific adaptations |

**Why other options are inferior:**

- **Option A (Full fine-tuning):** 13B model requires ~171GB for full fine-tuning (weights + gradients + optimizer). Even with checkpointing and mixed precision, doesn't fit on 24GB. Also risks catastrophic forgetting.

- **Option C (Prompt tuning):** 100 soft tokens provide limited capacity. Works for simple task adaptation but insufficient for learning complex customer service patterns from 10K examples.

- **Option D (Distillation):** Creates a smaller model, but loses significant capability. 1B model won't match 13B quality on complex queries. Also doesn't address the fine-tuning need.

**Understanding Gap:** If you selected A, review memory requirements for different fine-tuning approaches. Full fine-tuning memory ≈ 4-6× model size.

---

### Question 3: Sampling and Generation Control

**Model Answer:**

**a) Query Routing Strategy (5 points)**

**Classification signals:**

| Signal | Factual Indicator | Creative Indicator |
|--------|-------------------|-------------------|
| Keywords | "balance", "policy", "status", "when" | "write", "create", "fun", "imagine" |
| Query structure | Information-seeking question | Imperative/request form |
| Named entities | Account numbers, policy names | None or artistic subjects |
| Sentiment | Neutral, informational | Expressive, casual |

**Implementation:**
```python
def classify_query(query):
    # Lightweight classifier or rule-based
    creative_signals = ["write", "create", "compose", "fun", "poem", "story"]
    factual_signals = ["balance", "policy", "status", "what is", "how much"]

    creative_score = sum(1 for s in creative_signals if s in query.lower())
    factual_score = sum(1 for s in factual_signals if s in query.lower())

    # Or use a fine-tuned classifier for better accuracy
    return "creative" if creative_score > factual_score else "factual"
```

**b) Parameter Configuration (5 points)**

**Factual Mode:**
```yaml
temperature: 0.1-0.3
top_p: 0.9
max_tokens: 500
system_prompt: "Provide accurate information. If uncertain, say so."
```
**Justification:** Low temperature ensures consistent, accurate responses. Slight variation (0.1-0.3) allows natural phrasing without inventing facts.

**Creative Mode:**
```yaml
temperature: 0.8-1.0
top_p: 0.9
max_tokens: 300
presence_penalty: 0.3
system_prompt: "Be creative and personalized. Have fun with the response."
```
**Justification:** High temperature enables unexpected word choices. Presence penalty reduces repetition in creative writing.

**c) Hybrid Query Handling (5 points)**

For "Write a haiku about my account balance of $1,234.56":

**Strategy: Sequential pipeline**
```
1. EXTRACT (Factual, low temp):
   - Identify factual element: balance = $1,234.56
   - Verify from account data (if available)

2. GENERATE (Creative, high temp):
   - Input: "Write a haiku incorporating the number $1,234.56"
   - Use creative parameters

3. VALIDATE:
   - Ensure the exact dollar amount appears in output
   - No hallucinated account details
```

**Example output:**
```
Twelve thirty-four, change
Dollars rest in digital
Balance of my dreams
```

The factual element ($1,234.56) is preserved exactly, while the creative framing uses high-temperature generation.

---

### Question 4: Diffusion Model Analysis

**Model Answer:**

**a) Quality Issue Diagnosis (5 points)**

**Primary cause: CFG scale = 12.0 is too high**

**Mechanism:**
```
CFG formula: ε̂ = ε_uncond + w × (ε_cond - ε_uncond)

With w = 12.0:
- Conditional signal is amplified 12×
- Denoising overcorrects toward prompt
- Results in:
  1. Oversaturated colors (pushed to extremes)
  2. Artificial skin tones (natural variation suppressed)
  3. Edge artifacts (over-sharpening effects)
```

**Secondary issues:**
- No negative prompt to steer away from common artifacts
- Euler sampler can be less stable at higher CFG scales
- 25 steps is acceptable but more would help at high CFG

**b) Revised Configuration (5 points)**

```yaml
# Improved configuration
prompt: "A photorealistic portrait of a scientist in a laboratory"
negative_prompt: "oversaturated, artificial, plastic skin, cartoon,
                  painting, illustration, artifacts, blurry"

cfg_scale: 7.0-8.0      # Reduced from 12.0
steps: 35-40            # Increased for better refinement
sampler: "DPM++ 2M Karras"  # More stable than Euler

# Optional for photorealism
model: Use a photorealistic fine-tune (e.g., "Realistic Vision")
```

**Rationale:**
- CFG 7-8 provides good prompt adherence without over-amplification
- Negative prompt explicitly excludes unwanted characteristics
- More steps allow finer detail refinement
- DPM++ samplers handle CFG more gracefully

**c) Consistent Character Generation (5 points)**

**Techniques for generating consistent scientist images:**

| Technique | How It Works | Ease of Use |
|-----------|--------------|-------------|
| Fixed seed | Same random starting point → same face | Easy, but limited |
| IP-Adapter | Reference image guides facial features | Medium |
| LoRA fine-tuning | Train on 10-20 images of specific person | Complex |
| Textual Inversion | Learn embedding for person's appearance | Medium |
| ControlNet Face | Condition on facial landmarks | Medium |

**Recommended approach:**

```yaml
# Option 1: IP-Adapter (best balance of quality and ease)
ip_adapter_enabled: true
ip_adapter_scale: 0.6
reference_image: "scientist_reference.jpg"  # Initial generation or photo
seed: 42  # Fixed for reproducibility

# Option 2: LoRA (highest consistency, requires training)
# Train LoRA on 15-20 images of the character
# Use trigger word in prompts: "a portrait of sks_scientist..."
```

**Implementation note:** For production use, generate one "canonical" image, save the seed and parameters, then use IP-Adapter with that reference for variations.

---

### Question 5: RAG System Design for Enterprise Knowledge

**Rubric (30 points total):**

| Component | Excellent (Full) | Adequate (Half) | Insufficient (Minimal) |
|-----------|------------------|-----------------|------------------------|
| Architecture (8) | Complete pipeline for all sources, justified model choices | Covers main components but missing details | Incomplete or incorrect architecture |
| Chunking/Retrieval (7) | Source-specific chunking, multi-stage retrieval, metadata handling | Basic approach without source differentiation | Generic approach without consideration of sources |
| Quality/Safety (8) | Citations, conflict handling, access control, hallucination mitigation | Addresses some concerns | Missing major safety considerations |
| Evaluation (7) | Comprehensive metrics, monitoring, improvement strategy | Basic evaluation | Vague or missing evaluation |

**Model Answer:**

**1. Architecture Design**

The enterprise knowledge assistant requires a multi-source ingestion pipeline with unified retrieval and generation.

**Document Processing Pipeline:**

| Source | Processing Approach |
|--------|---------------------|
| PDFs/Word | Unstructured.io for text extraction; preserve tables as markdown; extract metadata (author, date, version) |
| Wiki pages | HTML parsing with section hierarchy preservation; link extraction for graph relationships |
| Meeting transcripts | Whisper for audio → text; speaker diarization; timestamp alignment |
| Slack messages | API extraction; thread grouping; author attribution |

**Embedding Strategy:**
- Primary: text-embedding-3-large (1536 dim) for semantic search
- Secondary: BM25 index for keyword coverage (legal terms, project names)
- Store embeddings in Pinecone (managed) or Milvus (self-hosted)

**Generation Component:**
- GPT-4 or Claude for complex reasoning with citations
- Fallback to Llama-2-70B for cost-sensitive queries
- Always include source attribution in system prompt

**2. Chunking and Retrieval Strategy**

**Source-Specific Chunking:**

| Source | Chunk Strategy | Size |
|--------|---------------|------|
| Policy docs | Section-based (preserve headings) | 512 tokens, 64 overlap |
| Wiki | Article summaries (L1) + paragraphs (L2) | Hierarchical |
| Transcripts | Speaker turns + 2-minute windows | 400 tokens |
| Slack | Thread-level (complete conversations) | Variable |

**Multi-Stage Retrieval:**
```
Query → Query expansion (HyDE) → Initial retrieval (50 candidates)
     → Reranking (cross-encoder) → Top-10 → Generation
```

**Cross-Document Handling:**
For queries spanning multiple sources (e.g., "Who decided on parental leave policy?"):
1. Retrieve policy document (policy text)
2. Retrieve meeting transcripts (decision context)
3. Retrieve Slack (informal discussions)
4. Synthesize with explicit source attribution

**Access Control:**
- Tag documents with access groups (HR, Engineering, All)
- Filter retrieval results based on user's group membership
- Never surface documents user cannot access, even if relevant

**3. Quality and Safety**

**Citation and Verification:**
- Every claim must cite specific document + section
- Format: "According to [HR Policy v3.2, Section 4.1]..."
- Post-generation: verify cited sections contain supporting text

**Handling Conflicting Information:**
```
When sources conflict (e.g., policy changed):
1. Prioritize by recency (newer > older)
2. Prioritize by authority (policy > Slack)
3. Present both with timestamps: "The current policy (2024) states X,
   though the previous version (2023) stated Y"
```

**Hallucination Mitigation:**
- System prompt: "Only use information from the provided sources"
- Require citations for every factual claim
- Confidence scoring: If retrieval confidence < threshold, respond "I couldn't find relevant information"
- Never extrapolate or assume beyond retrieved content

**4. Evaluation and Monitoring**

**Retrieval Metrics:**
- Recall@10 on human-labeled query-document pairs
- Mean Reciprocal Rank (MRR) for ranking quality
- Coverage: % of queries with at least one relevant result

**Generation Metrics:**
- Citation accuracy: % of citations that correctly support claims
- Faithfulness: NLI-based verification against sources
- Human evaluation: Accuracy, helpfulness (sampled)

**Production Monitoring:**
```
Dashboard metrics:
- Query volume and latency (P50, P95)
- Citation verification pass rate
- User feedback (thumbs up/down)
- Escalation rate to human support
```

**Continuous Improvement:**
- Log queries with low confidence for review
- Feedback loop: incorrect answers → labeled training data
- Weekly review of failure cases
- Quarterly re-evaluation on expanding test set
- Regular re-indexing as documents update

This architecture balances accuracy, safety, and scalability for enterprise deployment while maintaining appropriate access controls and continuous quality monitoring.

---

## Performance Interpretation Guide

| Score | Mastery Level | Recommended Action |
|-------|---------------|-------------------|
| 90-100% | **Mastery** | Ready for production generative AI development |
| 75-89% | **Proficient** | Review specific gaps, practice system design |
| 60-74% | **Developing** | Re-study core architectures and fine-tuning |
| Below 60% | **Foundational** | Complete re-review of Lesson 7, focus on fundamentals |

---

## Review Recommendations by Question

| If You Struggled With | Review These Sections |
|----------------------|----------------------|
| Question 1 | Lesson 7: Diffusion Models, Model Families Comparison |
| Question 2 | Lesson 7: Fine-Tuning, LoRA, QLoRA |
| Question 3 | Lesson 7: Sampling Strategies, Prompting |
| Question 4 | Lesson 7: CFG, Text-to-Image, ControlNet |
| Question 5 | Lesson 7: RAG, Deployment, Safety |

---

*Generated from Lesson 7: Generative AI | Quiz Skill*
