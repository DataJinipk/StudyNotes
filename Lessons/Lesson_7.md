# Lesson 7: Generative AI

**Topic:** Generative AI: Foundations, Architectures, and Modern Applications
**Prerequisites:** Lesson 3 (LLMs), Lesson 4 (Transformers), Lesson 5 (Deep Learning)
**Estimated Study Time:** 3-4 hours
**Difficulty:** Advanced

---

## Learning Objectives

Upon completion of this lesson, learners will be able to:

1. **Analyze** the fundamental principles underlying different generative model families (autoregressive, diffusion, VAE, GAN)
2. **Evaluate** tradeoffs between generation quality, speed, controllability, and compute requirements
3. **Apply** prompting techniques and fine-tuning strategies to adapt foundation models
4. **Design** generative AI systems combining multiple modalities
5. **Critique** deployments considering safety, bias, hallucination, and societal implications

---

## Introduction

Generative AI represents a paradigm shift in artificial intelligence—from systems that classify and predict to systems that create. Unlike discriminative models that learn P(y|x), generative models learn the underlying data distribution P(x), enabling them to synthesize novel instances: text, images, audio, video, code, and more.

The field has advanced dramatically since 2020. Large Language Models demonstrate remarkable text generation, reasoning, and instruction-following. Diffusion models produce photorealistic images from text descriptions. Multimodal models bridge modalities, enabling unified vision-language understanding. These systems are transforming creative industries, software development, scientific research, and human-computer interaction.

This lesson provides a unified framework for understanding generative AI across modalities, connecting the theoretical foundations to practical deployment considerations.

---

## Core Concepts

### Concept 1: Foundations of Generative Modeling

Generative modeling aims to learn the probability distribution P(x) of training data, enabling sampling of new instances that could plausibly belong to that distribution.

**Generative vs. Discriminative:**

| Aspect | Discriminative | Generative |
|--------|----------------|------------|
| Learns | P(y\|x) | P(x) or P(x\|condition) |
| Goal | Classify/Predict | Sample/Create |
| Example | Image classifier | Image generator |
| Difficulty | Easier (decision boundary) | Harder (full distribution) |

**Generative Model Families:**

```
GENERATIVE MODELS
├── Likelihood-Based (Explicit Density)
│   ├── Autoregressive: P(x) = ∏ P(xᵢ|x<ᵢ)
│   ├── VAE: Learn latent space + decoder
│   └── Normalizing Flows: Invertible transformations
│
├── Implicit Density
│   └── GANs: Generator vs Discriminator game
│
└── Score-Based
    └── Diffusion: Learn ∇log P(x), denoise iteratively
```

**Key Insight:** Different model families make different tradeoffs:
- **Autoregressive:** High quality, slow sequential generation
- **VAE:** Fast generation, often blurry outputs
- **GAN:** High quality images, training instability
- **Diffusion:** High quality, slow (many denoising steps)

---

### Concept 2: Autoregressive Generation

Autoregressive models generate sequences by predicting one element at a time, conditioning each prediction on all previously generated elements.

**The Chain Rule Factorization:**

```
P(x₁, x₂, ..., xₙ) = P(x₁) · P(x₂|x₁) · P(x₃|x₁,x₂) · ... · P(xₙ|x₁...xₙ₋₁)
                   = ∏ᵢ P(xᵢ|x<ᵢ)
```

Each step predicts the next token given all previous tokens. This is the foundation of Large Language Models.

**Training vs. Inference:**

| Phase | Input | Target | Efficiency |
|-------|-------|--------|------------|
| Training (Teacher Forcing) | Ground truth prefix | Next token | Parallel across positions |
| Inference | Generated prefix | Next token | Sequential (one token at a time) |

**Sampling Strategies:**

| Strategy | Description | Use Case |
|----------|-------------|----------|
| Greedy | Always pick highest probability | Deterministic, repetitive |
| Beam Search | Keep top-k candidates | Translation, structured output |
| Temperature | Scale logits by 1/T | T<1: focused, T>1: diverse |
| Top-k | Sample from top k tokens | Balance quality/diversity |
| Top-p (Nucleus) | Sample from smallest set summing to p | Adaptive diversity |

**Temperature Effect:**

```
P(token) = softmax(logits / T)

T = 0.0: Deterministic (argmax)
T = 0.7: Focused but varied (typical use)
T = 1.0: As trained
T = 1.5+: Creative but potentially incoherent
```

---

### Concept 3: Diffusion Models

Diffusion models generate data by learning to reverse a gradual noising process, iteratively denoising random noise into coherent samples.

**The Two Processes:**

```
FORWARD PROCESS (Fixed, adds noise):
x₀ → x₁ → x₂ → ... → xₜ → ... → x_T ≈ N(0, I)
     +ε₁   +ε₂         +εₜ

REVERSE PROCESS (Learned, removes noise):
x_T → x_{T-1} → ... → xₜ₋₁ → ... → x₀
      -ε̂_{T}          -ε̂ₜ
```

**Forward Process (Adding Noise):**

```
q(xₜ|xₜ₋₁) = N(xₜ; √(1-βₜ)xₜ₋₁, βₜI)

After T steps: q(x_T|x₀) ≈ N(0, I)  (pure noise)
```

**Reverse Process (Denoising):**

The model learns to predict the noise ε added at each step:

```
ε_θ(xₜ, t) ≈ ε  (the actual noise added)

Training Loss: L = E[||ε - ε_θ(xₜ, t)||²]
```

**Why Diffusion Works:**

| Property | Benefit |
|----------|---------|
| Stable training | No adversarial dynamics (unlike GANs) |
| High quality | Many small denoising steps |
| Diversity | Inherently stochastic process |
| Controllable | Easy to add conditioning |

**Sampling Algorithms:**

| Algorithm | Steps | Quality | Speed |
|-----------|-------|---------|-------|
| DDPM | 1000 | High | Slow |
| DDIM | 50-100 | High | Moderate |
| DPM-Solver | 20-50 | High | Fast |
| Consistency Models | 1-4 | Good | Very Fast |

---

### Concept 4: Large Language Models as Generators

LLMs combine autoregressive generation with Transformer architecture and massive scale, enabling remarkable language capabilities.

**The LLM Recipe:**

```
1. Architecture: Decoder-only Transformer
2. Objective: Next token prediction (cross-entropy loss)
3. Scale: Billions of parameters, trillions of tokens
4. Alignment: RLHF/RLAIF for instruction following
```

**Emergent Capabilities:**

| Capability | Description | Emergence Scale |
|------------|-------------|-----------------|
| In-context Learning | Learn from prompt examples | ~1B parameters |
| Chain-of-Thought | Step-by-step reasoning | ~10B parameters |
| Instruction Following | Execute diverse commands | ~10B + fine-tuning |
| Code Generation | Write and debug code | ~10B parameters |

**Generation Control in LLMs:**

```
System Prompt: Set behavior, persona, constraints
   ↓
User Prompt: Task specification, examples
   ↓
Model Generation: Constrained by attention to prompts
   ↓
Post-processing: Safety filters, format validation
```

**Key Parameters:**

| Parameter | Effect | Typical Range |
|-----------|--------|---------------|
| Temperature | Creativity vs. focus | 0.0-1.5 |
| Top-p | Nucleus sampling threshold | 0.9-0.95 |
| Max tokens | Output length limit | Task-dependent |
| Stop sequences | Termination triggers | Task-specific |

---

### Concept 5: Text-to-Image Generation

Text-to-image models generate images from natural language descriptions by conditioning diffusion (or autoregressive) models on text embeddings.

**Architecture Overview:**

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Text Prompt  │ ──► │ Text Encoder │ ──► │  Conditioning │
│ "A cat on    │     │ (CLIP/T5)    │     │   Embedding   │
│  the moon"   │     └──────────────┘     └───────┬───────┘
└──────────────┘                                  │
                                                  ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Output     │ ◄── │   Decoder    │ ◄── │   Diffusion  │
│   Image      │     │   (VAE)      │     │   U-Net      │
└──────────────┘     └──────────────┘     └──────────────┘
```

**Latent Diffusion (Stable Diffusion):**

Instead of diffusing in pixel space, operate in compressed latent space:

```
Image (512×512×3) → VAE Encoder → Latent (64×64×4) → Diffusion → VAE Decoder → Image
                    8× compression
```

Benefits: 64× fewer pixels to denoise, much faster training and inference.

**Classifier-Free Guidance (CFG):**

Strengthen the conditioning signal by interpolating between conditional and unconditional:

```
ε̂ = ε_uncond + w × (ε_cond - ε_uncond)

w = 1.0: Standard conditioning
w = 7.5: Typical for good prompt adherence
w = 15+: Very strong adherence, may reduce quality
```

**Additional Controls:**

| Technique | Function |
|-----------|----------|
| Negative Prompts | Specify what to avoid |
| ControlNet | Add spatial conditioning (edges, pose, depth) |
| Inpainting | Regenerate specific regions |
| Img2Img | Start from existing image, vary strength |

---

### Concept 6: Multimodal Foundation Models

Multimodal models process and generate across multiple modalities within unified architectures.

**Architectures for Multimodality:**

```
EARLY FUSION                      LATE FUSION
┌─────────────────┐              ┌──────┐   ┌──────┐
│ Image + Text    │              │Image │   │ Text │
│   Tokens        │              │Encoder   │Encoder
│      ↓          │              └──┬───┘   └──┬───┘
│  Unified        │                 │          │
│  Transformer    │                 └────┬─────┘
└─────────────────┘                      │
                                   ┌─────▼─────┐
                                   │  Fusion   │
                                   │  Layer    │
                                   └───────────┘
```

**Vision-Language Models:**

| Model | Architecture | Capabilities |
|-------|--------------|--------------|
| GPT-4V | Early fusion | Image understanding, analysis |
| Claude 3 | Early fusion | Document analysis, visual QA |
| LLaVA | Late fusion | Image understanding, instruction |
| Gemini | Early fusion | Native multimodal generation |

**Cross-Modal Understanding:**

```
Input: [Image of a chart] + "What trend does this show?"
       ↓
Vision Encoder: Extract visual features → image tokens
       ↓
Unified Model: Attend across text and image tokens
       ↓
Output: "The chart shows exponential growth in Q3..."
```

**Emerging Modalities:**

| Modality | Examples | Key Models |
|----------|----------|------------|
| Text → Image | DALL-E 3, Midjourney, SD | Diffusion + text conditioning |
| Text → Video | Sora, Runway | Extended diffusion, temporal |
| Text → Audio | AudioLM, MusicGen | Autoregressive tokens |
| Speech → Text | Whisper | Encoder-decoder Transformer |
| Any → Any | Gemini, unified models | Native multimodal |

---

### Concept 7: Fine-Tuning and Adaptation

Fine-tuning adapts pre-trained foundation models to specific tasks or domains through continued training on targeted data.

**Fine-Tuning Spectrum:**

```
                    Parameters Updated
    ◄─────────────────────────────────────────────►
    Few                                         All

    Prompting  │  PEFT/LoRA  │  Full Fine-tune
    (0%)       │  (0.1-1%)   │  (100%)

    Fast,Cheap │  Balanced   │  Best Quality
    Limited    │  Good       │  Expensive
```

**LoRA (Low-Rank Adaptation):**

Instead of updating full weight matrices, add trainable low-rank decomposition:

```
Original: W (d×d parameters)
LoRA: W + BA where B (d×r), A (r×d), r << d

Example: d=4096, r=8
- Original: 16.7M parameters
- LoRA: 65K parameters (0.4%)
```

**QLoRA:**

Combine LoRA with quantization for memory efficiency:

```
Base model: 4-bit quantized (frozen)
LoRA adapters: Full precision (trainable)
→ Fine-tune 70B model on single 48GB GPU
```

**Fine-Tuning Methods Comparison:**

| Method | Memory | Compute | Quality | Risk |
|--------|--------|---------|---------|------|
| Full Fine-tune | High | High | Best | Forgetting |
| LoRA | Low | Medium | Very Good | Low |
| QLoRA | Very Low | Medium | Good | Low |
| Prompt Tuning | Minimal | Low | Limited | None |

**Instruction Tuning:**

Transform base models into instruction-following assistants:

```
Dataset: Diverse (instruction, response) pairs
         ├── "Summarize this article: ..." → summary
         ├── "Translate to French: ..." → translation
         ├── "Write code that: ..." → code
         └── "Explain why: ..." → explanation

Result: Model learns to follow arbitrary instructions
```

---

### Concept 8: Prompting and In-Context Learning

Prompting leverages the model's ability to learn from context without parameter updates.

**In-Context Learning (ICL):**

```
Prompt: Translate English to French:
        sea otter → loutre de mer
        peppermint → menthe poivrée
        cheese →

Model Output: fromage
```

The model learns the pattern from examples without any weight updates.

**Prompting Strategies:**

| Strategy | Description | Example |
|----------|-------------|---------|
| Zero-shot | Task description only | "Translate: hello → " |
| Few-shot | Include examples | Show 3 translations first |
| Chain-of-Thought | Request reasoning | "Let's think step by step" |
| Self-consistency | Sample multiple, vote | Generate 5, take majority |
| Tree-of-Thought | Explore reasoning branches | Evaluate multiple paths |

**Chain-of-Thought Prompting:**

```
Without CoT:
Q: If John has 5 apples and gives 2 to Mary, how many does he have?
A: 3

With CoT:
Q: If John has 5 apples and gives 2 to Mary, how many does he have?
A: Let's think step by step.
   John starts with 5 apples.
   He gives away 2 apples.
   5 - 2 = 3
   Therefore, John has 3 apples.
```

CoT significantly improves performance on reasoning tasks.

**Retrieval-Augmented Generation (RAG):**

```
User Query → Retriever → Relevant Documents → Augmented Prompt → LLM → Response
                ↓
        Vector Database
        (Embeddings of
         knowledge base)
```

RAG grounds generation in retrieved facts, reducing hallucination.

---

### Concept 9: Safety and Alignment

Generative AI safety encompasses preventing harmful outputs, ensuring factual accuracy, and aligning model behavior with human values.

**The Alignment Pipeline:**

```
Pre-training → Supervised Fine-tuning → RLHF/DPO → Deployment Guardrails
(Capabilities)  (Instruction Following)  (Preference Alignment)  (Runtime Safety)
```

**RLHF (Reinforcement Learning from Human Feedback):**

```
1. Collect human preference data: (prompt, response_A, response_B, preference)
2. Train reward model: r_φ(prompt, response) → scalar
3. Optimize policy via PPO: max E[r_φ(x,y) - β·KL(π||π_ref)]
```

**DPO (Direct Preference Optimization):**

Skip the reward model, directly optimize from preferences:

```
L_DPO = -E[log σ(β(log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]
```

Simpler than RLHF, no separate reward model needed.

**Key Safety Challenges:**

| Challenge | Description | Mitigation |
|-----------|-------------|------------|
| Hallucination | Confident false statements | RAG, citations, uncertainty |
| Jailbreaking | Adversarial prompt attacks | Red-teaming, robust training |
| Bias | Training data biases in output | Diverse data, bias auditing |
| Harmful Content | Unsafe, unethical outputs | RLHF, content filtering |
| Privacy | Leaking training data | Differential privacy, filtering |

**Classifier-Free Guidance for Safety:**

```
Safe generation = Amplify "helpful" direction
                  Suppress "harmful" direction
```

---

### Concept 10: Deployment and Applications

Deploying generative AI requires balancing quality, cost, latency, and safety across diverse application domains.

**Deployment Options:**

| Option | Pros | Cons |
|--------|------|------|
| API (OpenAI, Anthropic) | No infrastructure, latest models | Cost, data privacy, vendor lock-in |
| Self-hosted Open | Control, privacy, customization | Ops burden, compute costs |
| Hybrid | Balance control and capability | Complexity |

**Inference Optimization:**

| Technique | Speedup | Quality Impact |
|-----------|---------|----------------|
| Quantization (INT8/INT4) | 2-4× | Minimal |
| Speculative Decoding | 2-3× | None |
| KV Cache | Baseline | None |
| Batch Processing | Linear with batch | None |
| Model Distillation | 3-10× | Some degradation |

**Application Domains:**

```
CREATIVE                          PRODUCTIVITY
├── Writing assistance            ├── Code completion
├── Image generation              ├── Document drafting
├── Music composition             ├── Email automation
└── Video creation                └── Meeting summaries

SCIENTIFIC                        ENTERPRISE
├── Drug discovery                ├── Customer service
├── Protein design                ├── Knowledge search
├── Literature synthesis          ├── Data analysis
└── Hypothesis generation         └── Process automation
```

**Evaluation Approaches:**

| Approach | What It Measures | Limitations |
|----------|------------------|-------------|
| Perplexity | Language modeling quality | Doesn't measure usefulness |
| Human Eval | Real-world quality | Expensive, not scalable |
| Benchmarks (MMLU, HumanEval) | Specific capabilities | May not reflect deployment |
| LLM-as-Judge | Automated quality scoring | Bias toward similar models |
| A/B Testing | Production impact | Requires deployment |

---

## Practical Considerations

### Model Selection Guide

| Need | Recommended Approach |
|------|---------------------|
| General text tasks | GPT-4, Claude, Llama 70B |
| Code generation | GPT-4, Claude, CodeLlama |
| Image generation | DALL-E 3, Midjourney, SD XL |
| Low latency | Smaller models, speculative decoding |
| Privacy-sensitive | Self-hosted open models |
| Domain-specific | Fine-tuned models |

### Cost Optimization

```
Strategies:
1. Cache common queries
2. Use smaller models for simple tasks (routing)
3. Batch requests when latency allows
4. Quantize self-hosted models
5. Prompt optimization (fewer tokens)
```

### Quality Assurance

```
Pipeline:
Input Validation → Generation → Output Filtering → Human Review (sample)
       ↓                              ↓
  Reject invalid             Block harmful content
```

---

## Connections to Other Lessons

| Lesson | Connection |
|--------|------------|
| Lesson 3: LLMs | Core technology for text generation |
| Lesson 4: Transformers | Architecture underlying all modern generative models |
| Lesson 5: Deep Learning | Training techniques, optimization, regularization |
| Lesson 6: RL | RLHF for alignment, reward modeling |
| Lesson 1: Agent Skills | Generative AI as capability for agents |
| Lesson 2: Prompting | Core technique for steering generation |

---

## Case Study: Building a Document Analysis System

**Task:** Create a system that analyzes uploaded documents (PDFs with text and images) and answers questions about them.

**Architecture:**

```
1. Document Processing
   - PDF → Text extraction + Image extraction
   - Chunk text into segments (500 tokens overlap 50)
   - Generate embeddings for retrieval

2. Multimodal Understanding
   - Text chunks → Text embeddings (ada-002)
   - Images → Vision encoder → Image descriptions
   - Store in vector database

3. Query Processing
   - User question → Retrieve relevant chunks + images
   - Construct prompt with retrieved context
   - Generate response with citations

4. Safety Layer
   - Filter sensitive document content
   - Verify response doesn't hallucinate beyond sources
   - Confidence scoring on answers
```

**Implementation Choices:**

| Component | Choice | Rationale |
|-----------|--------|-----------|
| LLM | Claude 3 / GPT-4V | Native document understanding |
| Embeddings | text-embedding-3-large | High quality retrieval |
| Vector DB | Pinecone / Chroma | Managed vs. self-hosted |
| Chunking | Recursive character | Preserves structure |

---

## Summary

Generative AI encompasses systems that create novel content by learning underlying data distributions. Autoregressive models generate sequences token-by-token, achieving remarkable language capabilities through simple next-token prediction at scale. Diffusion models generate images through iterative denoising, offering stable training and high-quality outputs. Text-to-image systems condition diffusion on text embeddings, enabling natural language control of visual generation. Multimodal foundation models unify understanding and generation across text, images, audio, and video.

Fine-tuning adapts foundation models to specific tasks, with parameter-efficient methods (LoRA, QLoRA) enabling customization on limited compute. Prompting techniques leverage in-context learning for task adaptation without training. Safety and alignment—through RLHF, DPO, and guardrails—address hallucination, bias, and harmful content. Deployment requires balancing quality, cost, latency, and safety considerations.

The rapid advancement of generative AI is transforming creative work, software development, scientific research, and enterprise productivity. Understanding the tradeoffs between different architectures, generation strategies, and deployment options is essential for building effective applications.

---

## Quick Reference

### Model Family Comparison

| Family | Quality | Speed | Training | Best For |
|--------|---------|-------|----------|----------|
| Autoregressive | High | Slow | Stable | Text, code |
| Diffusion | High | Slow | Stable | Images, video |
| GAN | High | Fast | Unstable | Images (legacy) |
| VAE | Medium | Fast | Stable | Latent space |

### Key Equations

| Concept | Equation |
|---------|----------|
| Autoregressive | P(x) = ∏ P(xᵢ\|x<ᵢ) |
| Diffusion Loss | L = E[\|\|ε - ε_θ(xₜ, t)\|\|²] |
| CFG | ε̂ = ε_uncond + w(ε_cond - ε_uncond) |
| LoRA | W' = W + BA (r << d) |
| DPO | L = -E[log σ(β·Δlog π)] |

### Deployment Checklist

- [ ] Model selection (capability vs. cost)
- [ ] Safety guardrails (input/output filtering)
- [ ] Evaluation pipeline (automated + human)
- [ ] Cost monitoring and optimization
- [ ] Latency requirements met
- [ ] Privacy compliance verified

---

*Next Lesson: Lesson 8 - Neural Network Architectures*
