# Flashcards: Lesson 7 - Generative AI

**Source:** Lessons/Lesson_7.md
**Subject Area:** AI Learning - Generative AI: Foundations, Architectures, and Applications
**Date Generated:** 2026-01-08
**Total Cards:** 5 (2 Easy, 2 Medium, 1 Hard)

---

## Card Distribution

| Difficulty | Count | Bloom's Level | Focus Area |
|------------|-------|---------------|------------|
| Easy | 2 | Remember/Understand | Core definitions, model comparisons |
| Medium | 2 | Apply/Analyze | Technique selection, parameter effects |
| Hard | 1 | Evaluate/Synthesize | System design, tradeoff analysis |

---

## Easy Cards

### Card 1: Generative Model Families

**[FRONT]**
Compare the four major generative model families (Autoregressive, Diffusion, GAN, VAE) in terms of generation quality, speed, and training stability.

**[BACK]**
**Generative Model Family Comparison:**

| Family | Quality | Speed | Training | Mechanism |
|--------|---------|-------|----------|-----------|
| **Autoregressive** | High | Slow | Stable | P(x) = ∏P(xᵢ\|x<ᵢ) |
| **Diffusion** | High | Slow | Stable | Iterative denoising |
| **GAN** | High | Fast | Unstable | Adversarial game |
| **VAE** | Medium | Fast | Stable | Latent encoding/decoding |

**Key Distinctions:**

- **Autoregressive (LLMs):** Sequential token generation, excellent for text, slow inference
- **Diffusion (Stable Diffusion):** Many denoising steps, excellent for images, slow but parallelizable
- **GAN:** Generator vs. discriminator, fast inference but mode collapse risk
- **VAE:** Encode to latent space, decode back, fast but often blurry

**Modern Usage:**
- Text: Autoregressive (GPT, Claude)
- Images: Diffusion (DALL-E, Midjourney)
- GANs: Largely superseded by diffusion for images
- VAEs: Used as components (latent diffusion encoders)

**Difficulty:** Easy | **Bloom's Level:** Remember

---

### Card 2: Classifier-Free Guidance

**[FRONT]**
Explain Classifier-Free Guidance (CFG) in diffusion models. What is the formula, and how does the guidance scale affect generation?

**[BACK]**
**Classifier-Free Guidance Formula:**

```
ε̂ = ε_uncond + w × (ε_cond - ε_uncond)
```

| Term | Meaning |
|------|---------|
| ε̂ | Final noise prediction used for denoising |
| ε_uncond | Noise prediction without conditioning (empty prompt) |
| ε_cond | Noise prediction with text conditioning |
| w | Guidance scale (weight) |

**Guidance Scale Effects:**

| Scale (w) | Effect | Use Case |
|-----------|--------|----------|
| w = 1.0 | No guidance boost | Standard conditioning |
| w = 5-7 | Moderate adherence | Typical quality sweet spot |
| w = 7.5 | Strong adherence | Default for Stable Diffusion |
| w = 10-15 | Very strong | Maximum prompt following |
| w > 15 | Oversaturated | May degrade quality |

**Intuition:** CFG amplifies the "direction" from unconditional toward conditional generation. Higher w means stronger push toward the prompt, at the cost of diversity and potentially quality.

**Training Requirement:** Model must be trained with random prompt dropout (e.g., 10% of time) to learn unconditional generation.

**Difficulty:** Easy | **Bloom's Level:** Understand

---

## Medium Cards

### Card 3: Sampling Strategy Selection

**[FRONT]**
You are deploying an LLM for three different applications:
1. Legal contract generation (must be precise)
2. Creative story writing (needs variety)
3. Code completion (balance accuracy and alternatives)

For each, recommend temperature, top-p, and explain your reasoning.

**[BACK]**
**Application-Specific Sampling Configurations:**

**1. Legal Contract Generation**
```
Temperature: 0.0-0.2
Top-p: 0.9-1.0 (or not needed with low temp)
Reasoning: Legal documents require precision and consistency.
           Low temperature ensures deterministic, predictable output.
           Creativity is undesirable; accuracy is paramount.
```

**2. Creative Story Writing**
```
Temperature: 0.8-1.2
Top-p: 0.9-0.95
Reasoning: Stories benefit from unexpected word choices and variety.
           Higher temperature introduces controlled randomness.
           Top-p prevents completely nonsensical tokens while
           allowing creative exploration of the probability space.
```

**3. Code Completion**
```
Temperature: 0.4-0.6
Top-p: 0.9
Reasoning: Code needs to be syntactically correct (favor precision)
           but multiple valid solutions exist (allow some variety).
           Moderate temperature balances correctness with
           ability to suggest alternative implementations.
```

**General Principles:**

| Need | Temperature | Top-p |
|------|-------------|-------|
| Determinism | 0.0 | 1.0 |
| Slight variety | 0.3-0.5 | 0.9 |
| Creative | 0.7-1.0 | 0.85-0.95 |
| Experimental | 1.0+ | 0.8-0.9 |

**Difficulty:** Medium | **Bloom's Level:** Apply

---

### Card 4: Fine-Tuning Method Analysis

**[FRONT]**
Compare Full Fine-tuning, LoRA, and QLoRA for adapting a 70B parameter LLM to a medical question-answering task. Consider memory requirements, training time, and expected quality.

**[BACK]**
**Fine-Tuning Comparison for 70B Model:**

| Aspect | Full Fine-tune | LoRA | QLoRA |
|--------|----------------|------|-------|
| **Parameters Updated** | 70B (100%) | ~70M (0.1%) | ~70M (0.1%) |
| **GPU Memory** | ~560GB (8×A100) | ~160GB (2×A100) | ~48GB (1×A100) |
| **Training Time** | Days-Weeks | Hours-Days | Hours-Days |
| **Quality** | Best | Very Good | Good |
| **Risk** | Catastrophic forgetting | Low | Low |

**LoRA Mechanics:**
```
Original: W ∈ R^(d×d) → 70B params total
LoRA: W + BA where B ∈ R^(d×r), A ∈ R^(r×d)

With r = 16, d = 8192:
- Per-layer: 2 × 8192 × 16 = 262K params
- Much smaller than original matrix
```

**QLoRA Innovation:**
```
Base model: 4-bit quantized (70B → ~35GB)
LoRA adapters: FP16/BF16 (trainable)
Optimizer states: Paged to CPU when needed
→ Enables consumer GPU fine-tuning
```

**Medical QA Recommendation:**

For medical domain with accuracy requirements:
- **Best quality:** Full fine-tune (if resources available)
- **Practical choice:** LoRA with r=32-64, target attention layers
- **Limited resources:** QLoRA, may need careful validation

**Key Consideration:** Medical domain requires careful validation regardless of method—fine-tuning can introduce hallucinations in specialized domains.

**Difficulty:** Medium | **Bloom's Level:** Analyze

---

## Hard Cards

### Card 5: Multimodal RAG System Design

**[FRONT]**
Design a production-ready Retrieval-Augmented Generation system for a financial services company that needs to:
- Process quarterly reports (PDFs with text, tables, charts)
- Answer complex analytical questions
- Cite sources for compliance
- Handle 1000+ queries per day

Address: architecture components, model selection, chunking strategy, safety considerations, and evaluation approach.

**[BACK]**
**Financial RAG System Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│  PDF → Document AI → │ Text Chunks    │ → Embeddings →     │
│                      │ Table Extraction│                    │
│                      │ Chart Analysis  │    Vector DB       │
│                      │ (GPT-4V)        │    (Pinecone)      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    QUERY PIPELINE                            │
├─────────────────────────────────────────────────────────────┤
│  Query → Safety    → Query     → Hybrid    → Rerank        │
│          Filter     Expansion   Retrieval   (Cohere)        │
│                     (HyDE)     (Dense+BM25)                 │
└───────────────────────────┬─────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    GENERATION PIPELINE                       │
├─────────────────────────────────────────────────────────────┤
│  Context + Query → Claude/GPT-4 → Citation    → Response   │
│                    (low temp)     Verification   + Sources  │
└─────────────────────────────────────────────────────────────┘
```

**Component Details:**

**1. Document Processing:**
```
Chunking Strategy:
- Semantic chunking by section headers
- Chunk size: 512 tokens, overlap: 64 tokens
- Preserve table structure as markdown
- Charts: GPT-4V description + extracted data points

Metadata: document_id, page, section, date, report_type
```

**2. Model Selection:**

| Component | Model | Rationale |
|-----------|-------|-----------|
| Embeddings | text-embedding-3-large | Best retrieval quality |
| Reranker | Cohere Rerank | Improves precision |
| Generation | Claude 3 Opus / GPT-4 | Strong reasoning, citations |
| Vision | GPT-4V | Chart/table understanding |

**3. Retrieval Strategy:**
```
Hybrid Search:
- Dense: Cosine similarity on embeddings (weight: 0.7)
- Sparse: BM25 for keyword matching (weight: 0.3)
- Top-k: 20 → Rerank → Top-5 for context
```

**4. Safety & Compliance:**

| Concern | Mitigation |
|---------|------------|
| Hallucination | Require citations, confidence scores |
| Data leakage | Input/output filtering for PII |
| Compliance | All responses logged, auditable |
| Accuracy | Never claim certainty without source |

**5. Evaluation Approach:**

| Metric | Method |
|--------|--------|
| Retrieval | Recall@k, MRR on labeled queries |
| Generation | Human eval (accuracy, citation quality) |
| Faithfulness | NLI-based verification of claims |
| Production | User feedback, escalation rate |

**6. Scale Considerations:**
```
1000 queries/day ≈ 42/hour
- Cache frequent queries (30% hit rate typical)
- Batch embed during ingestion
- Async processing for complex queries
- Cost: ~$500-1000/month API costs
```

**Difficulty:** Hard | **Bloom's Level:** Synthesize

---

## Critical Knowledge Flags

The following concepts appear across multiple cards and represent essential knowledge:

| Concept | Cards | Significance |
|---------|-------|--------------|
| Diffusion models | 1, 2 | Primary image generation method |
| Temperature/sampling | 1, 3 | Controls generation behavior |
| Fine-tuning methods | 4, 5 | Model adaptation strategies |
| Retrieval augmentation | 5 | Grounding and accuracy |
| Safety considerations | 3, 5 | Production deployment |

---

## Study Recommendations

### Before These Cards
- Review Lesson 4 (Transformers) for architecture foundation
- Understand Lesson 3 (LLMs) for autoregressive generation
- Review Lesson 5 (Deep Learning) for training concepts

### After Mastering These Cards
- Implement a simple diffusion model
- Build a RAG system with open-source tools
- Experiment with LoRA fine-tuning on smaller models

### Spaced Repetition Schedule
| Session | Focus |
|---------|-------|
| Day 1 | Cards 1-2 (foundations) |
| Day 3 | Cards 3-4 (applications) |
| Day 7 | Card 5 (synthesis), review 1-4 |
| Day 14 | Full review, identify weak areas |

---

## Export Formats

### Anki-Compatible (Tab-Separated)
```
Compare autoregressive, diffusion, GAN, VAE models	AR: high quality, slow, stable; Diffusion: high, slow, stable; GAN: high, fast, unstable; VAE: medium, fast, stable
What is Classifier-Free Guidance formula?	ε̂ = ε_uncond + w(ε_cond - ε_uncond); higher w = stronger prompt adherence
Sampling params for legal vs creative vs code	Legal: T=0.2; Creative: T=0.9, top-p=0.9; Code: T=0.5
Compare Full Fine-tune, LoRA, QLoRA for 70B	Full: 560GB, best quality; LoRA: 160GB, very good; QLoRA: 48GB, good
Design financial RAG system	Hybrid retrieval + reranking + citation verification + safety filtering
```

---

*Generated from Lesson 7: Generative AI | Flashcard Skill*
