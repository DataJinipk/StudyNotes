# Flashcards: Lesson 6 - Reinforcement Learning

**Source:** Lessons/Lesson_6.md
**Subject Area:** AI Learning - Reinforcement Learning: Foundations, Algorithms, and Decision-Making
**Date Generated:** 2026-01-08
**Total Cards:** 5 (2 Easy, 2 Medium, 1 Hard)

---

## Card Distribution

| Difficulty | Count | Bloom's Level | Focus Area |
|------------|-------|---------------|------------|
| Easy | 2 | Remember/Understand | Core definitions, key distinctions |
| Medium | 2 | Apply/Analyze | Algorithm selection, update rules |
| Hard | 1 | Evaluate/Synthesize | System design, tradeoff analysis |

---

## Easy Cards

### Card 1: Bellman Equation Foundation

**[FRONT]**
What is the Bellman optimality equation for the action-value function Q*(s,a), and what does each component represent?

**[BACK]**
**Bellman Optimality Equation for Q*:**

```
Q*(s,a) = R(s,a) + γ Σ_{s'} P(s'|s,a) max_{a'} Q*(s',a')
```

**Components:**

| Component | Meaning |
|-----------|---------|
| Q*(s,a) | Optimal value of taking action a in state s |
| R(s,a) | Immediate reward for action a in state s |
| γ | Discount factor (0 ≤ γ ≤ 1) |
| P(s'\|s,a) | Transition probability to s' given s, a |
| max_{a'} Q*(s',a') | Value of best action in next state |

**Key Insight:** The optimal value equals immediate reward plus the discounted value of acting optimally thereafter. This recursive structure enables iterative computation of optimal values.

**Difficulty:** Easy | **Bloom's Level:** Remember

---

### Card 2: On-Policy vs Off-Policy Learning

**[FRONT]**
Explain the difference between on-policy and off-policy learning in reinforcement learning. Classify SARSA and Q-learning accordingly.

**[BACK]**
**On-Policy Learning:**
- Learns the value of the policy currently being followed
- The behavior policy (for exploration) = target policy (being learned)
- Update uses the action actually taken

**Off-Policy Learning:**
- Learns the value of a different (target) policy than the one being followed
- Behavior policy ≠ target policy
- Can learn optimal policy while following exploratory policy

**Algorithm Classification:**

| Algorithm | Type | Update Target |
|-----------|------|---------------|
| SARSA | On-policy | r + γQ(s', a') where a' is the action actually taken |
| Q-learning | Off-policy | r + γ max_{a'} Q(s', a') regardless of action taken |

**Practical Implication:** Q-learning can learn the optimal policy while exploring (e.g., ε-greedy), but SARSA learns the value of the exploratory policy itself.

**Difficulty:** Easy | **Bloom's Level:** Understand

---

## Medium Cards

### Card 3: Policy Gradient vs Value-Based Selection

**[FRONT]**
You are designing an RL system for a robotic arm with 7 degrees of freedom (continuous joint angles as actions). Compare policy gradient methods versus value-based methods (like DQN) for this problem. Which approach is more suitable and why?

**[BACK]**
**Policy Gradient Methods are More Suitable**

**Why DQN is Problematic:**

| Challenge | Explanation |
|-----------|-------------|
| Action discretization | 7D continuous space → exponential discrete actions |
| Curse of dimensionality | Even 10 bins per joint = 10^7 actions |
| max operation | Computing max over continuous actions requires optimization |

**Why Policy Gradient Works:**

| Advantage | Explanation |
|-----------|-------------|
| Natural continuous output | Policy network outputs μ, σ for Gaussian over each joint |
| No argmax needed | Sample actions directly from π_θ(a\|s) |
| Smooth optimization | Gradients flow through continuous action space |

**Recommended Approach:**
- **PPO** or **SAC** with Gaussian policy
- Network outputs: mean μ_θ(s) and log std σ_θ(s) for each joint
- Actions sampled: a_i ~ N(μ_i, σ_i²)

**Alternative:** DDPG (Deep Deterministic Policy Gradient) for deterministic continuous control, though SAC's stochastic policy often performs better.

**Difficulty:** Medium | **Bloom's Level:** Apply

---

### Card 4: TD Error and Value Update Analysis

**[FRONT]**
Given the following scenario, compute the TD error and new Q-value:

- Current Q(s, a) = 5.0
- Reward received r = 2.0
- Next state value max Q(s', ·) = 6.0
- Discount factor γ = 0.9
- Learning rate α = 0.1

Show your work using the Q-learning update rule.

**[BACK]**
**Q-Learning Update Rule:**
```
Q(s,a) ← Q(s,a) + α [r + γ max Q(s',·) - Q(s,a)]
```

**Step 1: Compute TD Target**
```
Target = r + γ max Q(s',·)
       = 2.0 + 0.9 × 6.0
       = 2.0 + 5.4
       = 7.4
```

**Step 2: Compute TD Error (δ)**
```
δ = Target - Q(s,a)
  = 7.4 - 5.0
  = 2.4
```

**Step 3: Compute New Q-Value**
```
Q_new(s,a) = Q(s,a) + α × δ
           = 5.0 + 0.1 × 2.4
           = 5.0 + 0.24
           = 5.24
```

**Interpretation:**
- Positive TD error (2.4) indicates the outcome was better than expected
- Q-value increases from 5.0 to 5.24, moving toward the target
- With α = 0.1, we take a 10% step toward the new estimate

**Difficulty:** Medium | **Bloom's Level:** Apply

---

## Hard Cards

### Card 5: Deep RL System Design

**[FRONT]**
You are tasked with training an RL agent to play a complex strategy game from pixel observations. The game has:
- High-dimensional visual input (84×84×3 images)
- 18 discrete actions
- Sparse rewards (only at game end)
- Episodes lasting 10,000+ steps

Design a complete deep RL solution addressing: (1) algorithm selection with justification, (2) architecture design, (3) exploration strategy, and (4) training stability measures.

**[BACK]**
**Complete Deep RL System Design:**

**1. Algorithm Selection: Rainbow DQN or PPO**

| Option | Rationale |
|--------|-----------|
| Rainbow DQN | Discrete actions, off-policy (sample efficient), combines multiple improvements |
| PPO | Stable training, handles long episodes well, simpler to tune |

**Recommendation:** Start with PPO for stability; consider Rainbow if sample efficiency is critical.

**2. Architecture Design:**

```
Input: 84×84×3 frame stack (4 frames for temporal info)
       ↓
CNN Encoder:
  - Conv2d(4, 32, 8×8, stride 4) + ReLU
  - Conv2d(32, 64, 4×4, stride 2) + ReLU
  - Conv2d(64, 64, 3×3, stride 1) + ReLU
  - Flatten → 3136 features
       ↓
MLP Head:
  - Linear(3136, 512) + ReLU
  - Linear(512, 18) [for Q] or (18, 1) [for π, V]
```

**Frame stacking:** 4 consecutive frames capture motion/velocity information.

**3. Exploration Strategy (Multi-Pronged):**

| Strategy | Purpose |
|----------|---------|
| ε-greedy decay | Baseline exploration (1.0 → 0.01 over 1M frames) |
| Intrinsic curiosity (ICM) | Address sparse rewards in early training |
| Noisy networks | Parameter noise for consistent exploration |

**Critical for sparse rewards:** Add curiosity bonus r_intrinsic based on prediction error of a learned dynamics model.

**4. Training Stability Measures:**

| Measure | Implementation |
|---------|----------------|
| Target network | Update every 10,000 steps (DQN) |
| Experience replay | 1M buffer, prioritized sampling |
| Reward clipping | Clip to [-1, 1] range |
| Gradient clipping | Global norm ≤ 10 |
| Frame skip | Repeat actions for 4 frames |
| Observation normalization | Running mean/std normalization |

**For PPO specifically:**
- Clip ratio ε = 0.2
- GAE λ = 0.95 for advantage estimation
- Entropy bonus coefficient = 0.01

**Training Recommendations:**
- ~50-100M environment frames
- Distributed training with multiple parallel environments
- Log gradient norms, Q-value distributions, episode returns
- Checkpoints every 1M frames

**Difficulty:** Hard | **Bloom's Level:** Synthesize

---

## Critical Knowledge Flags

The following concepts appear across multiple cards and represent essential knowledge:

| Concept | Cards | Significance |
|---------|-------|--------------|
| Bellman equations | 1, 4 | Foundation for all value-based RL |
| TD error | 4, 5 | Core learning signal in TD methods |
| On/off-policy | 2, 3, 5 | Determines algorithm properties |
| Exploration | 2, 5 | Critical for learning success |
| Continuous actions | 3 | Policy gradient domain |

---

## Study Recommendations

### Before These Cards
- Review MDP definition and components
- Understand expected value and probability basics

### After Mastering These Cards
- Study specific algorithms (DQN variants, PPO internals)
- Practice implementing Q-learning from scratch
- Explore RLHF applications to LLMs

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
What is the Bellman optimality equation for Q*(s,a)?	Q*(s,a) = R(s,a) + γ Σ P(s'|s,a) max Q*(s',a')
On-policy vs off-policy: classify SARSA and Q-learning	SARSA: on-policy (uses actual action); Q-learning: off-policy (uses max)
7-DoF robot arm: policy gradient or DQN?	Policy gradient (PPO/SAC) for continuous action spaces
TD error calculation with Q=5, r=2, maxQ'=6, γ=0.9, α=0.1	δ=2.4, Q_new=5.24
Design deep RL for sparse-reward game from pixels	Rainbow/PPO + CNN encoder + intrinsic motivation + stability measures
```

---

*Generated from Lesson 6: Reinforcement Learning | Flashcard Skill*
