# StudyNotes

Comprehensive AI/ML study materials featuring structured learning chains for mastering artificial intelligence and machine learning concepts.

## Overview

This repository contains professional-grade educational content covering core AI/ML topics. Each topic follows a **5-Skill Learning Chain** methodology:

```
Study Notes → Concept Map → Flashcards → Practice Problems → Quiz
```

This progressive approach ensures deep understanding through multiple reinforcement strategies.

---

## Recommended Learning Path

The topics are organized in a logical progression, building from foundational concepts to advanced applications:

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │              RECOMMENDED LEARNING SEQUENCE                   │
                    └─────────────────────────────────────────────────────────────┘

    ┌─────────────────┐     ┌─────────────────┐
    │  1. NEURAL      │────►│  2. DEEP        │
    │    NETWORKS     │     │    LEARNING     │
    │  (Foundation)   │     │  (Architectures)│
    └─────────────────┘     └────────┬────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
          ┌─────────────────┐              ┌─────────────────┐
          │  3. COMPUTER    │              │  4. NATURAL     │
          │     VISION      │              │    LANGUAGE     │
          │  (Visual AI)    │              │   PROCESSING    │
          └────────┬────────┘              └────────┬────────┘
                   │                                │
                   └────────────────┬───────────────┘
                                    ▼
                          ┌─────────────────┐
                          │ 5. TRANSFORMERS │
                          │ (Modern Arch)   │
                          └────────┬────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                              ▼
          ┌─────────────────┐            ┌─────────────────┐
          │ 6. GENERATIVE   │            │ 7. REINFORTIC   │
          │      AI         │            │    LEARNING     │
          │ (Creation)      │            │  (Agents)       │
          └─────────────────┘            └─────────────────┘
```

---

## Topics Covered (Complete 5-Skill Chains)

### Tier 1: Foundations
*Start here to build core understanding*

| # | Topic | Description | Key Concepts | Difficulty |
|---|-------|-------------|--------------|------------|
| 1 | **[Neural Networks](notes/neural-networks/)** | Building blocks of deep learning | Perceptrons, activation functions, backpropagation, gradient descent | Intermediate |
| 2 | **[Deep Learning](notes/deep-learning/)** | Advanced neural architectures | CNNs, optimization (Adam, SGD), regularization, batch normalization, skip connections | Intermediate |

### Tier 2: Core Architecture
*The architecture powering modern AI*

| # | Topic | Description | Key Concepts | Difficulty |
|---|-------|-------------|--------------|------------|
| 3 | **[Transformers](notes/transformers/)** | Revolutionary sequence modeling | Self-attention, multi-head attention, positional encoding, BERT, GPT, encoder-decoder | Advanced |

### Tier 3: Application Domains
*Applying deep learning to real-world problems*

| # | Topic | Description | Key Concepts | Difficulty |
|---|-------|-------------|--------------|------------|
| 4 | **[Computer Vision](notes/computer-vision/)** | Visual understanding with AI | CNNs, object detection, segmentation, ResNet, YOLO, Vision Transformers | Advanced |
| 5 | **[Natural Language Processing](notes/natural-language-processing/)** | Text understanding and generation | Tokenization, embeddings, attention, seq2seq, language models | Advanced |

### Tier 4: Advanced Topics
*Cutting-edge AI capabilities*

| # | Topic | Description | Key Concepts | Difficulty |
|---|-------|-------------|--------------|------------|
| 6 | **[Generative AI](notes/generative-ai/)** | Creating content with AI | Diffusion models, LLMs, fine-tuning, RLHF, RAG, prompting | Advanced |
| 7 | **[Reinforcement Learning](notes/reinforcement-learning/)** | Learning through interaction | MDPs, Q-learning, policy gradients, actor-critic, DQN, PPO | Advanced |

### Additional Topics
*Supplementary materials*

| Topic | Description | Status |
|-------|-------------|--------|
| [Machine Learning](notes/machine-learning/) | Classical ML algorithms | Partial |
| [Large Language Models](notes/large-language-models/) | LLM-specific deep dive | Partial |
| [Prompt Engineering](notes/prompt-engineering/) | Effective LLM interaction | Partial |
| [Agentic AI](notes/agentic-ai/) | Autonomous AI agents | Partial |

---

## Repository Structure

```
StudyNotes/
├── notes/
│   ├── neural-networks/           # Tier 1: Foundation
│   │   ├── neural-networks-study-notes.md
│   │   ├── concept-maps/
│   │   ├── flashcards/
│   │   ├── practice/
│   │   └── quizzes/
│   │
│   ├── deep-learning/             # Tier 1: Foundation
│   │   ├── deep-learning-study-notes.md
│   │   ├── concept-maps/
│   │   ├── flashcards/
│   │   ├── practice/
│   │   └── quizzes/
│   │
│   ├── transformers/              # Tier 2: Core Architecture
│   │   ├── transformers-study-notes.md
│   │   ├── concept-maps/
│   │   ├── flashcards/
│   │   ├── practice/
│   │   └── quizzes/
│   │
│   ├── computer-vision/           # Tier 3: Applications
│   │   ├── computer-vision-study-notes.md
│   │   ├── concept-maps/
│   │   ├── flashcards/
│   │   ├── practice/
│   │   └── quizzes/
│   │
│   ├── natural-language-processing/  # Tier 3: Applications
│   │   ├── natural-language-processing-study-notes.md
│   │   ├── concept-maps/
│   │   ├── flashcards/
│   │   ├── practice/
│   │   └── quizzes/
│   │
│   ├── generative-ai/             # Tier 4: Advanced
│   │   ├── generative-ai-study-notes.md
│   │   ├── concept-maps/
│   │   ├── flashcards/
│   │   ├── practice/
│   │   └── quizzes/
│   │
│   ├── reinforcement-learning/    # Tier 4: Advanced
│   │   ├── reinforcement-learning-study-notes.md
│   │   ├── concept-maps/
│   │   ├── flashcards/
│   │   ├── practice/
│   │   └── quizzes/
│   │
│   └── [additional topics]/       # Supplementary materials
│
├── .claude/skills/                # Skill definitions
└── README.md
```

---

## 5-Skill Learning Chain

Each topic includes five interconnected learning materials:

### 1. Study Notes
Comprehensive academic-style notes including:
- Learning objectives aligned with Bloom's Taxonomy
- 10 core concepts with detailed explanations
- Theoretical frameworks
- Practical applications
- Critical analysis and current debates
- Key terminology tables
- Review questions

### 2. Concept Maps
Visual knowledge representations featuring:
- Mermaid diagrams for concept relationships
- Hierarchical concept organization
- Relationship matrices with connection strengths
- Multiple learning pathways (beginner to advanced)
- Central concept identification

### 3. Flashcards
Spaced-repetition compatible cards with:
- Difficulty distribution (2 Easy / 2 Medium / 1 Hard)
- Front/back format with detailed explanations
- Common misconceptions highlighted
- Mnemonics for retention
- Anki-compatible export format

### 4. Practice Problems
Hands-on exercises including:
- 1 Warm-up problem (foundational)
- 2 Skill-builder problems (application)
- 1 Challenge problem (synthesis)
- 1 Debug/fix scenario (troubleshooting)
- Complete solutions with explanations

### 5. Quizzes
Assessment materials featuring:
- 2 Multiple choice questions
- 2 Short answer questions
- 1 Essay question
- Complete answer keys with explanations
- Diagnostic feedback and review recommendations

---

## How to Use

### Recommended Study Path

**For Each Topic:**
```
Day 1: Study Notes (2-3 hours)
       └── Read through all core concepts
       └── Take notes on key terminology

Day 2: Concept Map (1 hour)
       └── Follow the visual diagram
       └── Trace learning pathways

Day 3: Flashcards (30 min/day ongoing)
       └── Import to Anki or use directly
       └── Review using spaced repetition

Day 4: Practice Problems (2-3 hours)
       └── Attempt before viewing solutions
       └── Work through all difficulty levels

Day 5: Quiz (1 hour)
       └── Complete under timed conditions
       └── Review diagnostic feedback
```

### Quick Reference
- **Concept review:** Flashcards + Concept Maps
- **Interview prep:** Practice Problems + Quiz
- **Deep understanding:** Study Notes + Essay questions

### Topic Dependencies

Before starting a topic, ensure you've completed prerequisites:

| Topic | Prerequisites |
|-------|---------------|
| Neural Networks | Basic math, Python |
| Deep Learning | Neural Networks |
| Transformers | Deep Learning |
| Computer Vision | Deep Learning |
| NLP | Deep Learning, (Transformers helpful) |
| Generative AI | Transformers, (NLP helpful) |
| Reinforcement Learning | Neural Networks |

---

## Content Statistics

| Metric | Count |
|--------|-------|
| Complete 5-Skill Chains | 7 topics |
| Study Notes | 7 comprehensive documents |
| Concept Maps | 7 visual diagrams |
| Flashcards | 35 cards (5 per topic) |
| Practice Problems | 35 problems (5 per topic) |
| Quiz Questions | 35 questions (5 per topic) |
| Total Concepts | 230+ |
| Total Relationships | 350+ |

---

## Concept Map Visualization

The concept maps use Mermaid syntax. To view them:

1. **GitHub**: Renders automatically in markdown preview
2. **VS Code**: Install "Markdown Preview Mermaid Support" extension
3. **Online**: Use [Mermaid Live Editor](https://mermaid.live/)

---

## Prerequisites

Recommended background knowledge:
- Basic programming (Python preferred)
- Linear algebra fundamentals (vectors, matrices)
- Calculus basics (derivatives, chain rule)
- Probability and statistics

---

## Contributing

Contributions are welcome! Please ensure any additions follow the established 5-Skill Chain format and maintain academic rigor.

---

## License

This educational content is provided for learning purposes.

---

*Generated with [Claude Code](https://claude.ai/claude-code)*
