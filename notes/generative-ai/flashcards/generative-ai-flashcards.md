# Flashcards: Generative AI

**Source:** notes/generative-ai/generative-ai-study-notes.md
**Concept Map:** notes/generative-ai/concept-maps/generative-ai-concept-map.md
**Date Generated:** 2026-01-07
**Total Cards:** 5
**Distribution:** 2 Easy | 2 Medium | 1 Hard

---

## Card Distribution Strategy

| Difficulty | Count | Concepts Covered | Selection Rationale |
|------------|-------|------------------|---------------------|
| Easy | 2 | Foundation Models, LLMs & Prompting | Critical centrality (10-12 connections); foundational |
| Medium | 2 | Diffusion Models, Fine-tuning & Alignment | High centrality; practical techniques |
| Hard | 1 | Production Generative AI System | Integration across all concepts |

---

## Easy Cards

### Card 1 of 5 | Easy
**Concept:** Foundation Models and Multimodal AI
**Centrality:** Critical (12 connections)
**Related Concepts:** LLMs, Diffusion, Vision Encoders, Unified Architectures

#### Front
What are foundation models in the context of generative AI? Name three key characteristics that distinguish them from traditional AI models, and give examples of current foundation models.

#### Back
**Foundation Models** are large-scale AI models trained on broad data that can be adapted to a wide range of downstream tasks through fine-tuning or prompting.

**Three Key Characteristics:**

| Characteristic | Description | Example |
|----------------|-------------|---------|
| **Scale** | Billions of parameters; trained on massive datasets | GPT-4: ~1.7T params (estimated); trained on internet-scale text |
| **Emergence** | Capabilities that appear at scale but not in smaller models | In-context learning, chain-of-thought reasoning |
| **Adaptability** | Single model adapts to many tasks via prompting or fine-tuning | Same model does translation, summarization, coding, Q&A |

**Current Foundation Models:**

| Model | Developer | Modalities | Key Capability |
|-------|-----------|------------|----------------|
| GPT-4 / GPT-4o | OpenAI | Text, Image, Audio | Strongest reasoning, multimodal |
| Claude 3 | Anthropic | Text, Image | Long context (200K), safety |
| Gemini | Google | Text, Image, Video, Audio | Multimodal native |
| Llama 3 | Meta | Text | Open weights, fine-tunable |
| Stable Diffusion | Stability AI | Image | Open, customizable image gen |

**Key Distinction from Traditional Models:**
- Traditional: One model = one task (e.g., ResNet for ImageNet classification)
- Foundation: One model = many tasks, even tasks not seen during training

#### Mnemonic
**"Foundation = Floor for Everything"** — Foundation models provide the base capability that all applications build upon.

#### Common Misconceptions
- ❌ Foundation models understand like humans (they pattern match at scale)
- ❌ Bigger always means better (efficiency and architecture matter)
- ❌ Closed models are always better than open (Llama 3 competitive with GPT-3.5)

---

### Card 2 of 5 | Easy
**Concept:** LLMs, Prompting, and RAG
**Centrality:** Critical (10 connections for LLM, 7 for Prompting)
**Related Concepts:** Autoregressive, In-context Learning, Hallucination

#### Front
Explain the relationship between Large Language Models, prompting techniques, and Retrieval-Augmented Generation (RAG). Why is RAG particularly important for production LLM applications?

#### Back
**Large Language Models (LLMs):**
- Autoregressive Transformers predicting next token
- Learn patterns from massive text corpora
- Generate by sampling token-by-token

**Prompting Techniques:**

| Technique | Description | Use Case |
|-----------|-------------|----------|
| **Zero-shot** | Task description only | Simple, well-understood tasks |
| **Few-shot** | Include examples in prompt | Complex tasks, specific formats |
| **Chain-of-Thought** | "Think step by step" | Reasoning, math, logic |
| **System prompts** | Set persona/constraints | Consistent behavior |

**RAG (Retrieval-Augmented Generation):**
```
User Query → Retrieve relevant docs → Inject into prompt → LLM generates → Response
```

**Why RAG is Critical for Production:**

| Problem | How RAG Solves It |
|---------|-------------------|
| **Hallucination** | Grounds responses in retrieved facts |
| **Knowledge cutoff** | Access real-time/updated information |
| **Domain specificity** | Inject company/domain knowledge |
| **Citation** | Can point to source documents |
| **Privacy** | Keep sensitive data out of model training |

**RAG Architecture:**
```
┌──────────────┐     ┌───────────────┐     ┌─────────────┐
│ User Query   │────►│ Vector Search │────►│ Top-K Docs  │
└──────────────┘     │ (Embedding)   │     └──────┬──────┘
                     └───────────────┘            │
                                                  ▼
┌──────────────┐     ┌───────────────┐     ┌─────────────┐
│  Response    │◄────│     LLM       │◄────│ Augmented   │
└──────────────┘     └───────────────┘     │   Prompt    │
                                           └─────────────┘
```

#### Mnemonic
**"LLM + RAG = Less Lies, More Grounding"**

#### Common Misconceptions
- ❌ RAG eliminates all hallucination (LLM can still misuse retrieved content)
- ❌ More retrieval always helps (too much context can confuse the model)
- ❌ RAG replaces fine-tuning (they serve different purposes; often combined)

---

## Medium Cards

### Card 3 of 5 | Medium
**Concept:** Diffusion Models and Text-to-Image
**Centrality:** Critical (8 connections)
**Related Concepts:** Denoising, Latent Space, Classifier-free Guidance, CLIP

#### Front
Explain how diffusion models generate images. What is classifier-free guidance (CFG), and how does the guidance scale affect generation?

#### Back
**Diffusion Model Process:**

**Forward Process (Training):**
```
Clean Image x₀ → Add noise → x₁ → ... → xₜ → ... → Pure Noise xₜ
                  q(xₜ|xₜ₋₁)         ~N(0,I)
```

**Reverse Process (Generation):**
```
Pure Noise xₜ → Denoise → xₜ₋₁ → ... → x₁ → Clean Image x₀
   ~N(0,I)      pθ(xₜ₋₁|xₜ)              Generated!
```

**Model Training:**
- Model εθ learns to predict the noise added at each step
- Loss: ||ε - εθ(xₜ, t)||² (predict noise from noisy image + timestep)

**Latent Diffusion (Stable Diffusion):**
```
Image → VAE Encoder → Latent z → Diffusion → Latent ẑ → VAE Decoder → Image
        (compress)     (64×64×4)   (denoise)              (decompress)
```
- 64× less computation than pixel-space diffusion
- Same quality, much faster

**Classifier-free Guidance (CFG):**

Without CFG, model generates valid images but may weakly follow the prompt.

**CFG Formula:**
```
ε̃ = εuncond + w × (εcond - εuncond)
    ↑           ↑
    Base     Amplified conditioning direction
```

**Guidance Scale (w) Effects:**

| Scale | Effect | Quality |
|-------|--------|---------|
| w = 1 | No guidance | Diverse but weak prompt following |
| w = 3-5 | Light guidance | Balance of diversity and adherence |
| w = 7-8 | Standard | Good prompt following (typical default) |
| w = 12-15 | Strong guidance | Very literal, less creative |
| w > 20 | Extreme | Over-saturated, artifacts |

**Trade-off:** Higher CFG = more prompt adherence, less diversity, potential artifacts

#### Mnemonic
**"Diffusion: Noise → Denoise → Nice Image"**
**"CFG: Crank For Guidance"**

#### Common Misconceptions
- ❌ More steps always better (diminishing returns past ~30-50 steps)
- ❌ CFG = 1 means unconditional (it means equal weight to both)
- ❌ CLIP text encoder understands grammar (it understands concepts, not syntax)

#### Critical Flag
⚠️ **Negative prompts** work by setting them as the unconditional direction, effectively pushing *away* from those concepts.

---

### Card 4 of 5 | Medium
**Concept:** Fine-tuning, LoRA, and RLHF
**Centrality:** High (7 connections each)
**Related Concepts:** Adaptation, Alignment, DPO, Parameter Efficiency

#### Front
Compare full fine-tuning, LoRA, and RLHF. When would you use each approach to adapt a foundation model?

#### Back
**Comparison Table:**

| Method | What It Does | Parameters Updated | Use Case |
|--------|--------------|-------------------|----------|
| **Full Fine-tuning** | Update all model weights | 100% | Maximum customization; need lots of data |
| **LoRA** | Add small trainable matrices | 0.1-1% | Efficient customization; limited compute |
| **RLHF** | Align with human preferences | Varies | Safety, helpfulness, style alignment |

**Full Fine-tuning:**
```python
# All parameters trainable
model = AutoModelForCausalLM.from_pretrained("llama-3-8b")
for param in model.parameters():
    param.requires_grad = True  # Update everything
```
- **Pros:** Maximum adaptation capability
- **Cons:** Expensive (needs full-model compute), catastrophic forgetting risk
- **Use when:** Large domain shift, lots of data, compute available

**LoRA (Low-Rank Adaptation):**
```
Original: y = Wx
LoRA:     y = Wx + BAx   where B ∈ ℝᵐˣʳ, A ∈ ℝʳˣⁿ, r << min(m,n)

W stays frozen; only A and B are trained
Rank r = 8-64 typical; parameters = 2 × r × d per layer
```

- **Pros:** 10-100× less memory; can merge back into base model
- **Cons:** Limited expressivity for large distribution shifts
- **Use when:** Limited compute, want to preserve base capabilities

**QLoRA:** Quantized base model (4-bit) + LoRA → Fine-tune 70B model on single GPU!

**RLHF (Reinforcement Learning from Human Feedback):**
```
1. Collect preference data: (prompt, response_A, response_B, human_preference)
2. Train reward model: R(prompt, response) → scalar score
3. Optimize policy (LLM) with PPO to maximize reward while staying close to base model

Loss = E[R(response)] - β × KL(policy || base_model)
```

- **Pros:** Aligns with nuanced human preferences; improves safety
- **Cons:** Complex pipeline, reward hacking risks, expensive
- **Use when:** Need safety alignment, style/tone control, quality improvement

**DPO (Direct Preference Optimization):**
- Simplified alternative to RLHF
- Directly optimize on preference pairs without separate reward model
- Same results, simpler pipeline

**Decision Tree:**
```
Need to adapt foundation model?
        │
        ├── Major domain shift, lots of data?
        │       └── Full Fine-tuning
        │
        ├── Moderate adaptation, limited compute?
        │       └── LoRA / QLoRA
        │
        ├── Align with preferences, improve safety?
        │       └── RLHF or DPO
        │
        └── Task-specific behavior, maintain generality?
                └── Instruction Fine-tuning + LoRA
```

#### Mnemonic
**"Full = Flexible but Fat; LoRA = Lean; RLHF = Refined behavior"**

#### Common Misconceptions
- ❌ LoRA always worse than full fine-tuning (often comparable for moderate shifts)
- ❌ RLHF makes models smarter (it changes behavior/style, not raw capability)
- ❌ You need RLHF for every deployment (instruction tuning often sufficient)

---

## Hard Cards

### Card 5 of 5 | Hard
**Concept:** Production Generative AI System
**Centrality:** Integration (spans all concepts)
**Related Concepts:** All core concepts

#### Front
Design a production generative AI system for a healthcare company that needs to: (1) answer patient questions about medications using company-approved information, (2) summarize clinical notes for doctors, (3) ensure no hallucinated medical advice, and (4) maintain HIPAA compliance. Address model selection, RAG design, safety measures, and deployment architecture.

#### Back
**System Architecture:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Healthcare Generative AI Platform                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐     ┌─────────────────┐     ┌───────────────┐         │
│  │ Patient App  │────►│  API Gateway    │────►│  Auth/HIPAA   │         │
│  │ Doctor Portal│     │  (Rate Limit)   │     │  Compliance   │         │
│  └──────────────┘     └────────┬────────┘     └───────────────┘         │
│                                │                                         │
│                     ┌──────────┴──────────┐                             │
│                     ▼                     ▼                              │
│         ┌─────────────────┐    ┌─────────────────┐                      │
│         │ Patient Q&A     │    │ Clinical Notes  │                      │
│         │    Service      │    │   Summarizer    │                      │
│         └────────┬────────┘    └────────┬────────┘                      │
│                  │                      │                                │
│         ┌────────┴────────────┬─────────┴───────┐                       │
│         ▼                     ▼                 ▼                        │
│  ┌─────────────┐    ┌─────────────────┐   ┌──────────────┐              │
│  │  RAG Layer  │    │   LLM Service   │   │ Safety Layer │              │
│  │             │    │                 │   │              │              │
│  │ • Medication│    │ • Claude 3 API  │   │ • Input filter│             │
│  │   Database  │    │ • Self-hosted   │   │ • Output check│             │
│  │ • Approved  │    │   Llama 3 70B   │   │ • Hallucination│            │
│  │   Content   │    │ • Fallback      │   │   detection  │              │
│  │ • Clinical  │    │                 │   │ • Audit log  │              │
│  │   Guidelines│    └─────────────────┘   └──────────────┘              │
│  └─────────────┘                                                         │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    Vector Database (Pinecone/Weaviate)            │   │
│  │  • Medication info embeddings (ada-002 or open embedding model)   │   │
│  │  • Clinical guidelines (chunked, metadata-tagged)                 │   │
│  │  • Encrypted, access-controlled by role                           │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

**1. Model Selection:**

| Use Case | Model Choice | Justification |
|----------|--------------|---------------|
| Patient Q&A | Claude 3 Sonnet API | Strong safety training, good at declining inappropriate requests |
| Clinical summarization | Self-hosted Llama 3 70B | PHI stays on-premise; HIPAA compliance |
| Embeddings | OpenAI ada-002 or self-hosted | Balance of quality and compliance |
| Fallback | Smaller model with strict guardrails | Cost/availability redundancy |

**Why not GPT-4 for clinical notes?** PHI would transit to OpenAI servers—HIPAA risk.

**2. RAG Design for Medical Safety:**

```python
class MedicalRAG:
    def __init__(self):
        self.approved_sources_only = True  # CRITICAL: No web search
        self.citation_required = True

    def retrieve(self, query, user_role):
        # Role-based access
        if user_role == "patient":
            collections = ["patient_approved_medications", "patient_faq"]
        elif user_role == "doctor":
            collections = ["clinical_guidelines", "drug_interactions", "research"]

        # Retrieve with metadata filtering
        results = vector_db.search(
            query=query,
            collections=collections,
            filters={"approved": True, "version": "current"},
            top_k=5
        )

        # Return with source citations
        return [(doc.text, doc.source_url, doc.last_reviewed) for doc in results]

    def generate(self, query, retrieved_docs):
        prompt = f"""You are a healthcare information assistant.

CRITICAL RULES:
1. ONLY use information from the provided sources
2. If information is not in sources, say "I don't have approved information about this"
3. NEVER provide dosage recommendations without citing the source
4. For emergencies, always direct to "Call 911 or go to nearest ER"

Sources:
{format_sources(retrieved_docs)}

Patient Question: {query}

Provide a helpful response with citations [1], [2], etc."""

        return llm.generate(prompt)
```

**3. Safety Measures:**

| Layer | Implementation | Purpose |
|-------|----------------|---------|
| **Input Filter** | PII detection, prompt injection detection | Block malicious inputs |
| **Retrieval Safety** | Only approved sources, role-based access | Ensure accurate info |
| **Output Validation** | Medical NER to check claims against sources | Catch hallucinations |
| **Confidence Scoring** | Flag low-confidence for human review | Uncertainty handling |
| **Audit Logging** | Every query/response logged with PHI handling | Compliance, debugging |

**Hallucination Detection:**
```python
def check_hallucination(response, retrieved_sources):
    # Extract medical claims from response
    claims = extract_medical_claims(response)  # NER for drugs, dosages, conditions

    for claim in claims:
        # Verify each claim against retrieved sources
        if not verify_claim_in_sources(claim, retrieved_sources):
            return {
                "hallucination_detected": True,
                "unverified_claim": claim,
                "action": "flag_for_review"
            }

    return {"hallucination_detected": False}
```

**4. HIPAA Compliance Architecture:**

| Requirement | Implementation |
|-------------|----------------|
| PHI Encryption | AES-256 at rest, TLS 1.3 in transit |
| Access Control | Role-based; MFA required |
| Audit Trail | Immutable logs; 6-year retention |
| Data Minimization | Don't send PHI to external APIs |
| BAA | Business Associate Agreements with all vendors |

**Self-Hosted Model for PHI:**
```yaml
# Kubernetes deployment for Llama 3 70B
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama3-clinical
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
          - "--model=meta-llama/Llama-3-70B-Instruct"
          - "--tensor-parallel-size=4"  # 4 GPUs
        resources:
          limits:
            nvidia.com/gpu: 4
      nodeSelector:
        hipaa-zone: "true"  # Dedicated HIPAA-compliant nodes
```

**5. Monitoring & Quality:**

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Hallucination rate | <1% | >2% triggers review |
| Citation coverage | >95% | <90% triggers investigation |
| Response latency p99 | <3s | >5s pages on-call |
| User satisfaction | >4.5/5 | <4.0 triggers UX review |
| Safety filter triggers | Track trend | Sudden increase = possible attack |

#### Common Misconceptions
- ❌ RAG completely prevents hallucination (LLM can still misinterpret retrieved content)
- ❌ Self-hosted = automatically HIPAA compliant (infrastructure must also be compliant)
- ❌ More safety filters = safer (over-filtering reduces usefulness)

---

## Anki Export Format

```
# Card 1 - Easy - Foundation Models
What are foundation models? Name characteristics and examples.	Foundation models: large-scale models trained on broad data, adapted to many tasks. Characteristics: Scale (billions of params), Emergence (capabilities at scale), Adaptability (one model, many tasks). Examples: GPT-4, Claude 3, Gemini, Llama 3, Stable Diffusion.	generative-ai foundation

# Card 2 - Easy - LLMs and RAG
Explain LLMs, prompting, and RAG relationship. Why is RAG important?	LLMs: autoregressive text generation. Prompting: zero-shot, few-shot, CoT control. RAG: retrieve docs → inject in prompt → generate. RAG importance: reduces hallucination, enables real-time knowledge, domain specificity, citation capability.	generative-ai llm rag

# Card 3 - Medium - Diffusion Models
How do diffusion models generate images? What is classifier-free guidance?	Diffusion: gradually denoise random noise into image over T steps. Model learns to predict noise ε at each step. CFG: ε̃ = εuncond + w×(εcond - εuncond). Scale w=7-8 typical; higher = more prompt adherence, less diversity, potential artifacts.	generative-ai diffusion

# Card 4 - Medium - Fine-tuning Methods
Compare full fine-tuning, LoRA, and RLHF.	Full FT: update all params; maximum adaptation; expensive. LoRA: add small matrices (0.1-1% params); efficient; good for moderate shifts. RLHF: align with preferences via reward model + PPO; for safety/style. Use LoRA for compute-limited, RLHF for alignment.	generative-ai finetuning

# Card 5 - Hard - Production Healthcare System
Design healthcare generative AI with RAG, safety, HIPAA compliance.	Model: Claude API for patient Q&A (safety), self-hosted Llama for PHI (HIPAA). RAG: approved sources only, role-based access, mandatory citations. Safety: input filter, hallucination detection via NER claim verification, audit logs. HIPAA: encryption, access control, BAA, PHI stays on-premise.	generative-ai production healthcare
```

---

## Review Schedule

| Card | First Review | Second Review | Third Review | Mastery Review |
|------|--------------|---------------|--------------|----------------|
| Card 1 (Easy) | Day 1 | Day 3 | Day 7 | Day 14 |
| Card 2 (Easy) | Day 1 | Day 3 | Day 7 | Day 14 |
| Card 3 (Medium) | Day 1 | Day 4 | Day 10 | Day 21 |
| Card 4 (Medium) | Day 2 | Day 5 | Day 12 | Day 25 |
| Card 5 (Hard) | Day 3 | Day 7 | Day 14 | Day 30 |

---

## Cross-References

| Card | Study Notes Section | Concept Map Node | Practice Problem |
|------|---------------------|------------------|------------------|
| Card 1 | Concept 6: Multimodal | Foundation Models (12) | Problem 1 |
| Card 2 | Concepts 4, 8: LLMs, Prompting | LLMs (10), RAG (6) | Problem 2 |
| Card 3 | Concepts 3, 5: Diffusion, T2I | Diffusion (8) | Problem 3 |
| Card 4 | Concepts 7, 9: Fine-tuning, Safety | Fine-tuning (7), RLHF (7) | Problem 4 |
| Card 5 | All Concepts | Full integration | Problem 5 |
