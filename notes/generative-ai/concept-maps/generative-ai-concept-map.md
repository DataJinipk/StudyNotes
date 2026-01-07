# Concept Map: Generative AI

**Source:** notes/generative-ai/generative-ai-study-notes.md
**Date Generated:** 2026-01-07
**Total Concepts:** 38
**Total Relationships:** 55
**Central Concept:** Foundation Models (12 connections)

---

## Visual Diagram (Mermaid)

```mermaid
flowchart TD
    subgraph Foundations["Generative Foundations"]
        GENMOD[Generative Modeling]
        PDATA[Data Distribution P(x)]
        SAMPLING[Sampling]
    end

    subgraph TextGen["Text Generation"]
        AUTOREG[Autoregressive]
        LLM[Large Language Models]
        TOKENIZE[Tokenization]
        DECODE[Decoding Strategies]
    end

    subgraph ImageGen["Image Generation"]
        DIFFUSION[Diffusion Models]
        DENOISE[Denoising]
        LATENT[Latent Space]
        T2I[Text-to-Image]
        CFG[Classifier-free Guidance]
    end

    subgraph Multimodal["Multimodal AI"]
        FOUNDATION[Foundation Models]
        VISION[Vision Encoder]
        UNIFIED[Unified Architecture]
        CROSSMODAL[Cross-modal]
    end

    subgraph Adaptation["Adaptation & Control"]
        FINETUNE[Fine-tuning]
        LORA[LoRA/PEFT]
        RLHF[RLHF]
        PROMPT[Prompting]
        ICL[In-context Learning]
        RAG[RAG]
    end

    subgraph Safety["Safety & Deployment"]
        ALIGN[Alignment]
        HALLUC[Hallucination]
        GUARD[Guardrails]
        DEPLOY[Deployment]
    end

    %% Foundation relationships
    GENMOD ==>|learns| PDATA
    PDATA ==>|enables| SAMPLING

    %% Text generation
    GENMOD ==>|approach| AUTOREG
    AUTOREG ==>|powers| LLM
    LLM --o TOKENIZE
    LLM --o DECODE

    %% Image generation
    GENMOD ==>|approach| DIFFUSION
    DIFFUSION ==>|uses| DENOISE
    DIFFUSION -.->|operates in| LATENT
    DIFFUSION ==>|enables| T2I
    T2I --o CFG

    %% Multimodal
    LLM ==>|scales to| FOUNDATION
    DIFFUSION ==>|component of| FOUNDATION
    FOUNDATION --o VISION
    FOUNDATION ==>|enables| UNIFIED
    UNIFIED -.->|achieves| CROSSMODAL

    %% Adaptation
    FOUNDATION ==>|adapted via| FINETUNE
    FINETUNE -->|efficient with| LORA
    FOUNDATION ==>|aligned via| RLHF
    LLM ==>|leverages| PROMPT
    PROMPT --o ICL
    PROMPT -.->|enhanced by| RAG

    %% Safety
    RLHF ==>|achieves| ALIGN
    LLM -.->|risks| HALLUC
    HALLUC -.->|mitigated by| RAG
    ALIGN ==>|implements| GUARD
    FOUNDATION ==>|requires| DEPLOY

    %% Styling
    style FOUNDATION fill:#e3f2fd,stroke:#1976d2,stroke-width:3px
    style LLM fill:#e8f5e9,stroke:#388e3c,stroke-width:3px
    style DIFFUSION fill:#fff3e0,stroke:#f57c00,stroke-width:3px
    style RLHF fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style PROMPT fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
```

**Diagram Key:**
- **Blue (FOUNDATION):** Central unifying concept
- **Green (LLM, PROMPT):** Text generation ecosystem
- **Orange (DIFFUSION):** Image generation pathway
- **Purple (RLHF):** Alignment and safety
- **Solid arrows (`==>`):** Primary/enabling relationships
- **Dashed arrows (`-.->`):** Secondary/uses relationships
- **Diamond arrows (`--o`):** Has-part/contains

---

## Concept Hierarchy

```
Generative AI
├── Foundations [CORE]
│   ├── Generative Modeling
│   │   ├── Probability Distribution P(x)
│   │   ├── Likelihood-based Models
│   │   ├── Implicit Density Models
│   │   └── Score-based Models
│   │
│   ├── Training Objectives
│   │   ├── Maximum Likelihood
│   │   ├── Variational Bounds
│   │   ├── Adversarial Loss
│   │   └── Score Matching
│   │
│   └── Sampling
│       ├── Ancestral Sampling
│       ├── MCMC Methods
│       └── Iterative Refinement
│
├── Text Generation [CORE]
│   ├── Autoregressive Models [HIGH CENTRALITY]
│   │   ├── Chain Rule Factorization
│   │   ├── Causal Attention
│   │   ├── Teacher Forcing
│   │   └── Sequential Generation
│   │
│   ├── Large Language Models [CRITICAL]
│   │   ├── Transformer Architecture
│   │   ├── Pre-training (Next Token)
│   │   ├── Scale & Emergence
│   │   ├── GPT Family
│   │   ├── Claude Family
│   │   └── Llama/Open Models
│   │
│   ├── Tokenization
│   │   ├── BPE
│   │   ├── WordPiece
│   │   └── SentencePiece
│   │
│   └── Decoding Strategies
│       ├── Greedy
│       ├── Beam Search
│       ├── Temperature
│       ├── Top-k
│       └── Top-p (Nucleus)
│
├── Image Generation [CORE]
│   ├── Diffusion Models [CRITICAL]
│   │   ├── Forward Process (Noising)
│   │   ├── Reverse Process (Denoising)
│   │   ├── DDPM
│   │   ├── DDIM
│   │   └── Score Matching
│   │
│   ├── Latent Diffusion [HIGH CENTRALITY]
│   │   ├── VAE Encoder/Decoder
│   │   ├── Compressed Latent Space
│   │   └── Computational Efficiency
│   │
│   ├── Text-to-Image [HIGH CENTRALITY]
│   │   ├── CLIP Text Encoder
│   │   ├── Cross-Attention Conditioning
│   │   ├── Stable Diffusion
│   │   ├── DALL-E
│   │   └── Midjourney
│   │
│   └── Control Mechanisms
│       ├── Classifier-free Guidance
│       ├── Negative Prompts
│       ├── ControlNet
│       └── Image-to-Image
│
├── Foundation Models [CORE - CENTRAL]
│   ├── Multimodal Models [HIGH CENTRALITY]
│   │   ├── GPT-4V / GPT-4o
│   │   ├── Claude 3 Vision
│   │   ├── Gemini
│   │   └── LLaVA
│   │
│   ├── Vision Encoders
│   │   ├── ViT
│   │   ├── CLIP
│   │   └── SigLIP
│   │
│   ├── Unified Architectures
│   │   ├── Shared Token Space
│   │   ├── Cross-modal Attention
│   │   └── Interleaved Generation
│   │
│   └── Video Generation
│       ├── Sora
│       ├── Temporal Consistency
│       └── Long-form Generation
│
├── Adaptation & Control [CORE]
│   ├── Fine-tuning [HIGH CENTRALITY]
│   │   ├── Full Fine-tuning
│   │   ├── LoRA [HIGH CENTRALITY]
│   │   ├── QLoRA
│   │   ├── Adapters
│   │   └── Prefix Tuning
│   │
│   ├── Alignment [HIGH CENTRALITY]
│   │   ├── RLHF [CRITICAL]
│   │   ├── DPO
│   │   ├── Constitutional AI
│   │   └── RLAIF
│   │
│   ├── Prompting [HIGH CENTRALITY]
│   │   ├── Zero-shot
│   │   ├── Few-shot
│   │   ├── Chain-of-Thought
│   │   ├── System Prompts
│   │   └── Prompt Engineering
│   │
│   └── Retrieval-Augmented
│       ├── RAG [HIGH CENTRALITY]
│       ├── Vector Databases
│       └── Chunking Strategies
│
└── Safety & Deployment [CORE]
    ├── Safety Concerns
    │   ├── Hallucination [HIGH CENTRALITY]
    │   ├── Bias
    │   ├── Harmful Content
    │   └── Jailbreaking
    │
    ├── Mitigation
    │   ├── Guardrails
    │   ├── Content Filtering
    │   ├── Red-teaming
    │   └── Watermarking
    │
    └── Deployment
        ├── API Services
        ├── Self-hosting
        ├── Quantization
        └── Cost Optimization
```

---

## Relationship Matrix

| From | Relationship | To | Strength | Notes |
|------|--------------|-----|----------|-------|
| Generative Modeling | learns | Data Distribution | Strong | Core objective |
| Data Distribution | enables | Sampling | Strong | Generation capability |
| Generative Modeling | approach | Autoregressive | Strong | Text paradigm |
| Generative Modeling | approach | Diffusion | Strong | Image paradigm |
| Autoregressive | powers | LLMs | Strong | Architecture basis |
| LLMs | has-part | Tokenization | Strong | Input processing |
| LLMs | has-part | Decoding | Strong | Output generation |
| Diffusion | uses | Denoising | Strong | Core mechanism |
| Diffusion | operates-in | Latent Space | Strong | Efficiency |
| Diffusion | enables | Text-to-Image | Strong | Key application |
| Text-to-Image | has-part | CFG | Strong | Control mechanism |
| LLMs | scales-to | Foundation Models | Strong | Evolution |
| Diffusion | component-of | Foundation Models | Strong | Multimodal |
| Foundation Models | has-part | Vision Encoder | Strong | Image understanding |
| Foundation Models | enables | Unified Architecture | Strong | Cross-modal |
| Foundation Models | adapted-via | Fine-tuning | Strong | Customization |
| Fine-tuning | efficient-with | LoRA | Strong | PEFT |
| Foundation Models | aligned-via | RLHF | Strong | Safety |
| LLMs | leverages | Prompting | Strong | Control |
| Prompting | has-part | ICL | Strong | Core capability |
| Prompting | enhanced-by | RAG | Strong | Grounding |
| RLHF | achieves | Alignment | Strong | Safety |
| LLMs | risks | Hallucination | Strong | Key challenge |
| Hallucination | mitigated-by | RAG | Strong | Solution |
| Alignment | implements | Guardrails | Strong | Deployment |
| Foundation Models | requires | Deployment | Strong | Production |

### Relationship Statistics
- **Total relationships:** 55
- **Most connected:** Foundation Models (12), LLMs (10), Diffusion (8), Prompting (7)
- **High-centrality:** Fine-tuning (7), RLHF (7), RAG (6), Text-to-Image (6)
- **Strongest cluster:** {Foundation Models, LLMs, Diffusion, Multimodal}
- **Bridge concepts:** Foundation Models (unifies text and image), RAG (connects prompting and safety)

---

## Concept Index

| Concept | Definition | Connections | Centrality | Card/Problem Rec |
|---------|------------|-------------|------------|------------------|
| Foundation Models | Large pretrained multimodal models | 12 | **Critical** | Card 1, Problem 1 |
| Large Language Models | Autoregressive text generation at scale | 10 | **Critical** | Card 2, Problem 2 |
| Diffusion Models | Iterative denoising generation | 8 | **Critical** | Card 3, Problem 3 |
| Prompting | Input design for model control | 7 | **High** | Card 2, Problem 2 |
| Fine-tuning | Adapting pretrained models | 7 | **High** | Card 4, Problem 4 |
| RLHF | Human feedback alignment | 7 | **High** | Card 4 |
| RAG | Retrieval-augmented generation | 6 | **High** | Card 2, Problem 2 |
| Text-to-Image | Language-conditioned image generation | 6 | **High** | Card 3, Problem 3 |
| LoRA | Parameter-efficient fine-tuning | 5 | High | Card 4, Problem 4 |
| Hallucination | False confident generation | 5 | High | Problem 5 |
| Alignment | Safety and value alignment | 5 | High | Card 4 |
| Autoregressive | Sequential generation paradigm | 5 | High | Card 2 |
| Latent Diffusion | Compressed space diffusion | 4 | High | Card 3 |
| Multimodal | Cross-modality understanding | 4 | Medium | Card 1 |
| In-context Learning | Prompt-based adaptation | 4 | Medium | Card 2 |
| Classifier-free Guidance | Conditional generation control | 4 | Medium | Card 3 |
| Tokenization | Text to tokens conversion | 3 | Medium | - |
| Decoding Strategies | Sampling methods | 3 | Medium | - |
| Vision Encoder | Image to embeddings | 3 | Medium | - |
| Guardrails | Safety mechanisms | 3 | Medium | Problem 5 |
| DPO | Direct Preference Optimization | 3 | Medium | - |
| Chain-of-Thought | Reasoning elicitation | 2 | Low | - |
| ControlNet | Spatial conditioning | 2 | Low | - |
| Quantization | Model compression | 2 | Low | - |

---

## Learning Pathways

### Pathway 1: Text Generation Focus
**Best for:** Building LLM applications

```
1. Autoregressive Modeling     How LLMs generate text
        ↓
2. Large Language Models       Architecture and scale
        ↓
3. Tokenization & Decoding     Input/output processing
        ↓
4. Prompting Techniques        Zero-shot, few-shot, CoT
        ↓
5. RAG                         Grounding with retrieval
        ↓
6. Fine-tuning & LoRA          Customization
        ↓
7. RLHF & Alignment            Safety and quality
```

**Estimated sessions:** 7-9

---

### Pathway 2: Image Generation Focus
**Best for:** Building visual AI applications

```
1. Diffusion Fundamentals      Noising and denoising
        ↓
2. DDPM / DDIM                 Sampling algorithms
        ↓
3. Latent Diffusion            Efficient generation
        ↓
4. Text-to-Image               CLIP, cross-attention
        ↓
5. Control Mechanisms          CFG, ControlNet
        ↓
6. Image Editing               Inpainting, outpainting
        ↓
7. Video Generation            Temporal extension
```

**Estimated sessions:** 6-8

---

### Pathway 3: Practitioner's Path
**Best for:** Building production applications

```
1. Foundation Model Landscape   What's available (GPT, Claude, Llama)
        ↓
2. Prompting Best Practices     System prompts, examples
        ↓
3. RAG Implementation           Vector DBs, chunking
        ↓
4. Fine-tuning Decisions        When and how
        ↓
5. LoRA / QLoRA                 Efficient adaptation
        ↓
6. Safety & Guardrails          Content filtering
        ↓
7. Deployment & Cost            API vs self-host
```

**Estimated sessions:** 6-7

---

### Pathway 4: Multimodal AI Path
**Best for:** Understanding unified AI systems

```
1. Vision Encoders              ViT, CLIP
        ↓
2. LLMs as Foundation           Text capabilities
        ↓
3. Vision-Language Models       GPT-4V, Claude 3
        ↓
4. Unified Architectures        Token space unification
        ↓
5. Cross-modal Generation       Image ↔ Text
        ↓
6. Video Generation             Sora, temporal modeling
```

**Estimated sessions:** 5-6

---

### Critical Path (Minimum Viable Understanding)

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    LLMs     │ ─► │  Prompting  │ ─► │    RAG      │ ─► │  Diffusion  │ ─► │ Foundation  │
│             │    │             │    │             │    │             │    │   Models    │
│  "Text"     │    │  "Control"  │    │  "Ground"   │    │  "Images"   │    │  "Unified"  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘

Minimum sessions: 5
Coverage: ~75% of generative AI practice
```

---

## Cross-Reference to Downstream Skills

### Flashcard Mapping
| Centrality | Recommended Card |
|------------|-----------------|
| Critical (Foundation Models, Multimodal) | Easy Card 1 - overview |
| Critical (LLMs, Prompting, RAG) | Easy Card 2 - text generation |
| Critical (Diffusion, Text-to-Image) | Medium Card 3 - image generation |
| High (Fine-tuning, LoRA, RLHF) | Medium Card 4 - adaptation |
| Integration (Full System) | Hard Card 5 - production system |

### Practice Problem Mapping
| Concept Cluster | Problem Type |
|-----------------|--------------|
| Model selection, capabilities | Warm-Up: Choose right model |
| Prompting, RAG implementation | Skill-Builder: Build RAG system |
| Diffusion, image generation | Skill-Builder: Control image gen |
| Fine-tuning pipeline | Challenge: Custom model training |
| Safety, hallucination | Debug/Fix: Safety issues |

### Quiz Question Mapping
| Relationship | Question Type |
|--------------|---------------|
| Diffusion mechanics | MC - Understanding |
| LLM vs Diffusion tradeoffs | MC - Comparison |
| Prompting strategy | SA - Application |
| Fine-tuning decisions | SA - Analysis |
| Complete generative system | Essay - Synthesis |
