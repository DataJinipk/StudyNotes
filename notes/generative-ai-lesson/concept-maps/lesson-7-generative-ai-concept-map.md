# Concept Map: Lesson 7 - Generative AI

**Source:** Lessons/Lesson_7.md
**Subject Area:** AI Learning - Generative AI: Foundations, Architectures, and Applications
**Date Generated:** 2026-01-08
**Total Concepts:** 38
**Total Relationships:** 55

---

## Visual Concept Map (Mermaid)

```mermaid
graph TD
    subgraph Foundations["Generative Foundations"]
        GENAI[Generative AI]
        PDATA[P(x) Distribution]
        SAMPLING[Sampling]
        LIKELIHOOD[Likelihood-Based]
        IMPLICIT[Implicit Density]
    end

    subgraph Autoregressive["Autoregressive Models"]
        AR[Autoregressive]
        CHAIN[Chain Rule]
        LLM[Large Language Models]
        TEMP[Temperature]
        TOPK[Top-k/Top-p]
        BEAM[Beam Search]
    end

    subgraph Diffusion["Diffusion Models"]
        DIFF[Diffusion]
        FORWARD[Forward Process]
        REVERSE[Reverse Process]
        NOISE[Noise Prediction]
        CFG[Classifier-Free Guidance]
        DDIM[DDIM/DPM-Solver]
    end

    subgraph TextToImage["Text-to-Image"]
        T2I[Text-to-Image]
        LATENT[Latent Diffusion]
        TEXTENC[Text Encoder]
        CONTROLNET[ControlNet]
        NEGPROMPT[Negative Prompts]
    end

    subgraph Multimodal["Multimodal"]
        MULTI[Multimodal Models]
        VISIONENC[Vision Encoder]
        CROSSATTN[Cross-Modal Attention]
        T2V[Text-to-Video]
        T2A[Text-to-Audio]
    end

    subgraph FineTuning["Fine-Tuning"]
        FT[Fine-Tuning]
        FULLFT[Full Fine-Tune]
        LORA[LoRA]
        QLORA[QLoRA]
        INSTRUCT[Instruction Tuning]
    end

    subgraph Prompting["Prompting"]
        PROMPT[Prompting]
        ZEROSHOT[Zero-Shot]
        FEWSHOT[Few-Shot]
        COT[Chain-of-Thought]
        RAG[RAG]
    end

    subgraph Safety["Safety & Alignment"]
        SAFETY[Safety]
        RLHF[RLHF]
        DPO[DPO]
        HALLUC[Hallucination]
        JAILBREAK[Jailbreaking]
    end

    subgraph Deployment["Deployment"]
        DEPLOY[Deployment]
        API[API Services]
        SELFHOST[Self-Hosted]
        QUANT[Quantization]
        EVAL[Evaluation]
    end

    %% Foundation relationships
    GENAI -->|"learns"| PDATA
    PDATA -->|"enables"| SAMPLING
    GENAI -->|"approach"| LIKELIHOOD
    GENAI -->|"approach"| IMPLICIT

    %% Autoregressive relationships
    LIKELIHOOD -->|"includes"| AR
    AR -->|"uses"| CHAIN
    AR -->|"scaled to"| LLM
    LLM -->|"controlled by"| TEMP
    LLM -->|"uses"| TOPK
    LLM -->|"uses"| BEAM

    %% Diffusion relationships
    LIKELIHOOD -->|"includes"| DIFF
    DIFF -->|"has"| FORWARD
    DIFF -->|"has"| REVERSE
    REVERSE -->|"learns"| NOISE
    DIFF -->|"controlled by"| CFG
    DIFF -->|"accelerated by"| DDIM

    %% Text-to-Image relationships
    DIFF -->|"enables"| T2I
    T2I -->|"uses"| LATENT
    T2I -->|"uses"| TEXTENC
    T2I -->|"extended by"| CONTROLNET
    T2I -->|"uses"| NEGPROMPT
    CFG -->|"critical for"| T2I

    %% Multimodal relationships
    MULTI -->|"uses"| VISIONENC
    MULTI -->|"uses"| CROSSATTN
    MULTI -->|"enables"| T2V
    MULTI -->|"enables"| T2A
    LLM -->|"combined with"| MULTI

    %% Fine-tuning relationships
    FT -->|"method"| FULLFT
    FT -->|"method"| LORA
    LORA -->|"variant"| QLORA
    FT -->|"method"| INSTRUCT
    LLM -->|"adapted via"| FT

    %% Prompting relationships
    PROMPT -->|"technique"| ZEROSHOT
    PROMPT -->|"technique"| FEWSHOT
    PROMPT -->|"technique"| COT
    PROMPT -->|"technique"| RAG
    LLM -->|"steered by"| PROMPT
    RAG -->|"reduces"| HALLUC

    %% Safety relationships
    SAFETY -->|"method"| RLHF
    SAFETY -->|"method"| DPO
    SAFETY -->|"addresses"| HALLUC
    SAFETY -->|"addresses"| JAILBREAK
    LLM -->|"aligned via"| SAFETY

    %% Deployment relationships
    DEPLOY -->|"option"| API
    DEPLOY -->|"option"| SELFHOST
    DEPLOY -->|"optimized by"| QUANT
    DEPLOY -->|"measured by"| EVAL
    GENAI -->|"requires"| DEPLOY
```

---

## Concept Hierarchy

```
GENERATIVE AI
├── FOUNDATIONS
│   ├── Generative vs Discriminative
│   │   ├── P(x) vs P(y|x)
│   │   └── Create vs Classify
│   ├── Likelihood-Based
│   │   ├── Explicit density modeling
│   │   └── Tractable likelihood
│   └── Implicit Density
│       └── Learn to sample without P(x)
│
├── AUTOREGRESSIVE MODELS
│   ├── Chain Rule Factorization
│   │   └── P(x) = ∏ P(xᵢ|x<ᵢ)
│   ├── Large Language Models
│   │   ├── GPT, Claude, Llama
│   │   └── Emergent capabilities
│   └── Sampling Strategies
│       ├── Temperature
│       ├── Top-k / Top-p
│       └── Beam Search
│
├── DIFFUSION MODELS
│   ├── Forward Process
│   │   └── Gradual noising to N(0,I)
│   ├── Reverse Process
│   │   └── Learned denoising
│   ├── Noise Prediction
│   │   └── ε_θ(xₜ, t) ≈ ε
│   ├── Classifier-Free Guidance
│   │   └── Amplify conditioning
│   └── Fast Sampling
│       ├── DDIM
│       └── DPM-Solver
│
├── TEXT-TO-IMAGE
│   ├── Latent Diffusion
│   │   └── Compress to latent space
│   ├── Text Encoders
│   │   ├── CLIP
│   │   └── T5
│   ├── ControlNet
│   │   └── Spatial conditioning
│   └── Negative Prompts
│       └── What to avoid
│
├── MULTIMODAL MODELS
│   ├── Vision Encoders
│   │   └── ViT, CLIP vision
│   ├── Cross-Modal Attention
│   │   └── Text-image interaction
│   ├── Text-to-Video
│   │   └── Sora, Runway
│   └── Text-to-Audio
│       └── AudioLM, MusicGen
│
├── FINE-TUNING
│   ├── Full Fine-Tuning
│   │   └── All parameters updated
│   ├── LoRA
│   │   └── Low-rank adapters
│   ├── QLoRA
│   │   └── Quantized + LoRA
│   └── Instruction Tuning
│       └── Diverse task training
│
├── PROMPTING
│   ├── Zero-Shot
│   │   └── Task description only
│   ├── Few-Shot
│   │   └── Include examples
│   ├── Chain-of-Thought
│   │   └── Step-by-step reasoning
│   └── RAG
│       └── Retrieval augmentation
│
├── SAFETY & ALIGNMENT
│   ├── RLHF
│   │   └── Human feedback optimization
│   ├── DPO
│   │   └── Direct preference optimization
│   ├── Hallucination
│   │   └── False confident statements
│   └── Jailbreaking
│       └── Adversarial attacks
│
└── DEPLOYMENT
    ├── API Services
    │   └── OpenAI, Anthropic
    ├── Self-Hosted
    │   └── Llama, Mistral
    ├── Optimization
    │   └── Quantization, caching
    └── Evaluation
        └── Human, automatic, benchmarks
```

---

## Relationship Matrix

| From Concept | To Concept | Relationship Type | Strength |
|--------------|------------|-------------------|----------|
| Generative AI | P(x) Distribution | learns | Strong |
| P(x) Distribution | Sampling | enables | Strong |
| Autoregressive | Chain Rule | uses | Strong |
| Autoregressive | LLM | scaled-to | Strong |
| LLM | Temperature | controlled-by | Strong |
| LLM | Prompting | steered-by | Strong |
| LLM | Fine-Tuning | adapted-via | Strong |
| LLM | Safety | aligned-via | Strong |
| Diffusion | Forward Process | has | Strong |
| Diffusion | Reverse Process | has | Strong |
| Reverse Process | Noise Prediction | learns | Strong |
| Diffusion | CFG | controlled-by | Strong |
| Diffusion | Text-to-Image | enables | Strong |
| Text-to-Image | Latent Diffusion | uses | Strong |
| Text-to-Image | Text Encoder | uses | Strong |
| CFG | Text-to-Image | critical-for | Strong |
| Multimodal | Vision Encoder | uses | Strong |
| Multimodal | Cross-Modal Attention | uses | Strong |
| LLM | Multimodal | combined-with | Strong |
| Fine-Tuning | LoRA | method | Strong |
| LoRA | QLoRA | variant | Strong |
| Prompting | Chain-of-Thought | technique | Strong |
| Prompting | RAG | technique | Strong |
| RAG | Hallucination | reduces | Strong |
| Safety | RLHF | method | Strong |
| Safety | DPO | method | Strong |
| Safety | Hallucination | addresses | Strong |
| Deployment | Quantization | optimized-by | Strong |
| Deployment | Evaluation | measured-by | Strong |
| Diffusion | DDIM | accelerated-by | Medium |
| Text-to-Image | ControlNet | extended-by | Medium |

---

## Centrality Index

**High Centrality (6+ connections):**

| Concept | Incoming | Outgoing | Total | Role |
|---------|----------|----------|-------|------|
| Large Language Models | 2 | 6 | 8 | **Central Entity** - Text generation |
| Diffusion | 1 | 6 | 7 | **Key Architecture** - Image generation |
| Text-to-Image | 2 | 5 | 7 | **Application** - Major use case |
| Fine-Tuning | 1 | 5 | 6 | **Adaptation** - Model customization |
| Prompting | 1 | 5 | 6 | **Control** - Steering generation |

**Medium Centrality (3-5 connections):**

| Concept | Incoming | Outgoing | Total | Role |
|---------|----------|----------|-------|------|
| Safety | 1 | 4 | 5 | Alignment and guardrails |
| Multimodal | 1 | 4 | 5 | Cross-modal capability |
| CFG | 1 | 3 | 4 | Generation control |
| LoRA | 2 | 2 | 4 | Efficient fine-tuning |
| RAG | 1 | 2 | 3 | Grounding technique |
| Deployment | 1 | 3 | 4 | Production systems |

**Low Centrality (1-2 connections):**
- Temperature, Top-k/Top-p, Beam Search, Forward/Reverse Process, DDIM, ControlNet, Vision Encoder, Zero-Shot, Few-Shot, Chain-of-Thought, RLHF, DPO, Quantization

---

## Learning Pathways

### Pathway 1: Text Generation Mastery
**Goal:** Understand autoregressive LLM generation
**Sequence:** P(x) → Chain Rule → Autoregressive → LLM → Sampling (Temperature, Top-p) → Prompting (Zero-shot, Few-shot, CoT)
**Prerequisites:** Lesson 4 (Transformers)
**Assessment:** Can configure sampling parameters and design effective prompts

### Pathway 2: Image Generation
**Goal:** Master diffusion-based image generation
**Sequence:** Diffusion → Forward/Reverse Process → Noise Prediction → CFG → Latent Diffusion → Text-to-Image → ControlNet → Negative Prompts
**Prerequisites:** Lesson 5 (Deep Learning)
**Assessment:** Can explain diffusion process and configure text-to-image systems

### Pathway 3: Model Adaptation
**Goal:** Customize foundation models
**Sequence:** Fine-Tuning → Full Fine-tune vs PEFT → LoRA → QLoRA → Instruction Tuning → Domain Adaptation
**Prerequisites:** Pathways 1-2
**Assessment:** Can select and implement appropriate fine-tuning strategy

### Pathway 4: Safe Deployment
**Goal:** Deploy generative AI responsibly
**Sequence:** Safety → RLHF/DPO → Hallucination → Jailbreaking → RAG → Deployment Options → Evaluation
**Prerequisites:** Pathways 1-3
**Assessment:** Can design safe, evaluated production systems

---

## Critical Path Analysis

**Minimum Viable Understanding (MVU):**
```
P(x) → Autoregressive → LLM → Diffusion → Text-to-Image → Prompting → Safety
```

**Rationale:** These seven concepts provide essential generative AI literacy:
1. **P(x):** What generative models learn
2. **Autoregressive:** How LLMs generate
3. **LLM:** The text foundation
4. **Diffusion:** How image models work
5. **Text-to-Image:** Major application area
6. **Prompting:** How to use these systems
7. **Safety:** Responsible deployment

**Expanded Path for Practitioners:**
```
P(x) → Autoregressive → LLM → Sampling Strategies → Diffusion → CFG →
Latent Diffusion → Text-to-Image → Multimodal → Fine-Tuning → LoRA →
Prompting → RAG → Chain-of-Thought → RLHF → Safety → Deployment → Evaluation
```

---

## Cross-Lesson Connections

### To Lesson 3 (LLMs)
| Generative AI Concept | LLM Connection | Implication |
|-----------------------|----------------|-------------|
| Autoregressive | Core LLM architecture | Foundation of text generation |
| Scaling laws | Emergent capabilities | Why bigger models work better |
| Prompting | In-context learning | LLM steering mechanism |

### To Lesson 4 (Transformers)
| Generative AI Concept | Transformer Connection | Implication |
|-----------------------|----------------------|-------------|
| LLM | Decoder-only Transformer | Architecture basis |
| Diffusion | Cross-attention in U-Net | Text conditioning mechanism |
| Multimodal | Vision Transformer | Image encoding |

### To Lesson 5 (Deep Learning)
| Generative AI Concept | Deep Learning Connection | Implication |
|-----------------------|-------------------------|-------------|
| Training | Optimization, regularization | How models learn |
| Fine-tuning | Transfer learning | Efficient adaptation |
| LoRA | Low-rank approximation | Parameter efficiency |

### To Lesson 6 (Reinforcement Learning)
| Generative AI Concept | RL Connection | Implication |
|-----------------------|---------------|-------------|
| RLHF | PPO optimization | Alignment mechanism |
| DPO | Preference learning | Direct optimization |
| Reward modeling | RL reward function | Human feedback encoding |

---

## Concept Definitions (Quick Reference)

| Concept | One-Line Definition |
|---------|---------------------|
| Generative AI | AI systems that create novel content |
| P(x) | Probability distribution over data |
| Autoregressive | Generate sequentially, each conditioned on previous |
| LLM | Large-scale autoregressive text model |
| Temperature | Sampling randomness control |
| Top-p | Nucleus sampling threshold |
| Diffusion | Iterative denoising from noise to data |
| Forward Process | Gradual noise addition |
| Reverse Process | Learned denoising |
| CFG | Classifier-free guidance; amplify conditioning |
| Latent Diffusion | Diffusion in compressed space |
| Text-to-Image | Generate images from text descriptions |
| ControlNet | Spatial conditioning for diffusion |
| Multimodal | Processing multiple modalities |
| Fine-tuning | Adapt pre-trained model to task |
| LoRA | Low-rank adaptation; efficient fine-tuning |
| QLoRA | Quantized model + LoRA |
| Prompting | Steering via input design |
| Zero-shot | Task description only |
| Few-shot | Include examples in prompt |
| Chain-of-Thought | Elicit step-by-step reasoning |
| RAG | Retrieval-Augmented Generation |
| RLHF | RL from Human Feedback |
| DPO | Direct Preference Optimization |
| Hallucination | Confident false generation |
| Jailbreaking | Adversarial prompt attacks |

---

## Study Recommendations

### Foundation First
1. Understand discriminative vs. generative distinction
2. Master autoregressive factorization
3. Trace diffusion forward/reverse processes

### Architecture Focus
1. Compare model families (AR, Diffusion, GAN, VAE)
2. Understand latent diffusion efficiency gains
3. Study cross-modal attention mechanisms

### Practical Skills
1. Configure sampling parameters for different tasks
2. Design effective prompts (zero-shot, few-shot, CoT)
3. Implement LoRA fine-tuning on small models

### Safety Awareness
1. Understand RLHF/DPO mechanisms
2. Recognize hallucination patterns
3. Design evaluation pipelines

---

*Generated from Lesson 7: Generative AI | Concept Map Skill*
