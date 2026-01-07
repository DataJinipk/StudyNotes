# Flashcard Set: Reinforcement Learning

**Source:** notes/reinforcement-learning/reinforcement-learning-study-notes.md
**Concept Map Reference:** notes/reinforcement-learning/concept-maps/reinforcement-learning-concept-map.md
**Date Generated:** 2026-01-07
**Total Cards:** 5
**Distribution:** 2 Easy | 2 Medium | 1 Hard

---

## Critical Knowledge Summary

Concepts appearing in multiple cards (prioritize for review):
- **Value Functions (V, Q)**: Appears in Cards 1, 2, 4, 5 (central to all RL)
- **Bellman Equations**: Appears in Cards 1, 2 (theoretical foundation)
- **Policy**: Appears in Cards 1, 3, 4, 5 (learning objective)
- **TD Learning**: Appears in Cards 2, 5 (core algorithm family)
- **Exploration**: Appears in Cards 2, 3, 5 (fundamental challenge)

---

## Flashcards

---
### Card 1 | Easy
**Cognitive Level:** Remember/Understand
**Concept:** MDP Framework and Value Functions
**Source Section:** Core Concepts 1, 2, 3
**Concept Map Centrality:** Value Functions (9), MDP (4)

**FRONT (Question):**
What are the five components of a Markov Decision Process (MDP), and what do the state-value function V(s) and action-value function Q(s,a) represent?

**BACK (Answer):**
**Five MDP Components:**

| Component | Symbol | Definition |
|-----------|--------|------------|
| **States** | S | All possible situations |
| **Actions** | A | All possible choices |
| **Transitions** | P(s'\|s,a) | Probability of next state given current state and action |
| **Rewards** | R(s,a,s') | Immediate feedback for transitions |
| **Discount** | γ ∈ [0,1] | How much to value future vs. immediate rewards |

**Value Functions:**

```
V^π(s) = E[Σ γ^t r_t | s_0 = s, π]
       = Expected cumulative discounted reward
         starting from state s, following policy π

Q^π(s,a) = E[Σ γ^t r_t | s_0 = s, a_0 = a, π]
         = Expected cumulative discounted reward
           starting from state s, taking action a,
           then following policy π
```

**Key Relationships:**
- V tells you "how good is this state?"
- Q tells you "how good is this action in this state?"
- V^π(s) = Σ_a π(a|s) Q^π(s,a) (value = expected Q over policy)
- π*(s) = argmax_a Q*(s,a) (optimal policy from optimal Q)

**Markov Property:**
The future depends only on the current state, not the history:
P(s_{t+1}|s_t, a_t) = P(s_{t+1}|s_0, a_0, ..., s_t, a_t)

**Critical Knowledge Flag:** Yes - Foundation for all RL algorithms

---

---
### Card 2 | Easy
**Cognitive Level:** Understand
**Concept:** Bellman Equations and Q-Learning
**Source Section:** Core Concepts 4, 5, 6
**Concept Map Centrality:** Bellman (6), Q-Learning (7), TD (5)

**FRONT (Question):**
Write the Bellman optimality equation for Q*, explain its meaning, and show how Q-learning uses it as an update rule. Why is Q-learning called "off-policy"?

**BACK (Answer):**
**Bellman Optimality Equation for Q*:**
```
Q*(s,a) = Σ_{s'} P(s'|s,a) [R(s,a,s') + γ max_{a'} Q*(s',a')]
```

**Meaning:**
- The optimal value of taking action a in state s equals:
- Expected immediate reward R(s,a,s')
- Plus discounted value of the best action in the next state
- Averaged over all possible next states s'

**Q-Learning Update Rule:**
```
Q(s,a) ← Q(s,a) + α [r + γ max_{a'} Q(s',a') - Q(s,a)]
                    └────────────────────────────────┘
                              TD Error (δ)
```

| Component | Meaning |
|-----------|---------|
| α | Learning rate (step size) |
| r | Observed reward |
| γ max Q(s',a') | Estimate of future value (greedy) |
| Q(s,a) | Current estimate |
| TD Error | Difference between new and old estimate |

**Why "Off-Policy":**
```
Behavior Policy:    The policy used to SELECT actions (e.g., ε-greedy)
Target Policy:      The policy being LEARNED (greedy: max_a Q)

Q-learning uses max_{a'} Q(s',a') in updates regardless of which
action was actually taken. This means it learns about the optimal
policy while following a different exploratory policy.

Compare to SARSA (on-policy):
Q(s,a) ← Q(s,a) + α [r + γ Q(s',a') - Q(s,a)]
                           ↑
                   Uses actual next action a', not max
```

**Critical Knowledge Flag:** Yes - Core algorithm for value-based RL

---

---
### Card 3 | Medium
**Cognitive Level:** Apply
**Concept:** Policy Gradient Methods
**Source Section:** Core Concepts 7
**Concept Map Centrality:** Policy Gradient (5), Policy (6)

**FRONT (Question):**
Explain the policy gradient theorem, show the REINFORCE algorithm update, and describe why policy gradients have high variance. What is a baseline, and how does it help?

**BACK (Answer):**
**Policy Gradient Theorem:**
```
∇_θ J(θ) = E_π [∇_θ log π_θ(a|s) · Q^π(s,a)]
```

**Meaning:**
- J(θ) is the objective: expected cumulative reward
- Gradient points in direction that increases probability of high-value actions
- ∇ log π(a|s) is the "score function" - direction to increase action probability
- Weighted by Q(s,a) - how good the action is

**REINFORCE Algorithm:**
```python
# For each episode:
1. Generate trajectory τ = (s_0, a_0, r_1, s_1, a_1, r_2, ...)
2. For each timestep t in trajectory:
   - Compute return G_t = Σ_{k=0}^{T-t} γ^k r_{t+k+1}
   - Update: θ ← θ + α · G_t · ∇_θ log π_θ(a_t|s_t)
```

**Mathematical Update:**
```
θ ← θ + α · G_t · ∇_θ log π_θ(a_t|s_t)
          ↑         ↑
       Return    Score function
    (how good)  (increase probability)
```

**High Variance Problem:**
- G_t varies enormously between trajectories
- Same action might get G=100 in one episode, G=5 in another
- Gradient estimates are noisy → slow, unstable learning
- Requires many samples for reliable gradients

**Baseline Solution:**
```
∇_θ J(θ) = E [∇_θ log π_θ(a|s) · (Q^π(s,a) - b(s))]
                                  └─────┬─────┘
                               Advantage-like term
```

**Why It Helps:**
- b(s) is a baseline (often V(s))
- Subtracting baseline doesn't change expected gradient (unbiased)
- But reduces variance: (Q - V) fluctuates less than Q alone
- Actions better than average get positive updates
- Actions worse than average get negative updates

**Common Baselines:**
| Baseline | Formula | Notes |
|----------|---------|-------|
| Value function | V(s) | Most common; gives advantage |
| Average return | E[G] | Simple but less effective |
| Moving average | Running mean of G | Online estimation |

**Critical Knowledge Flag:** Yes - Foundation for modern policy optimization

---

---
### Card 4 | Medium
**Cognitive Level:** Analyze
**Concept:** Actor-Critic Methods
**Source Section:** Core Concepts 8
**Concept Map Centrality:** Actor-Critic (6)

**FRONT (Question):**
Design an Actor-Critic architecture, explaining the role of each component. Compare A2C and PPO: what problem does PPO's clipping mechanism solve, and how?

**BACK (Answer):**
**Actor-Critic Architecture:**

```
            ┌──────────────────────────────────────────────┐
            │                    State s                    │
            └─────────────────────┬────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
        ┌───────────────────┐       ┌───────────────────┐
        │      ACTOR        │       │      CRITIC       │
        │    π_θ(a|s)       │       │      V_w(s)       │
        │                   │       │                   │
        │  Policy Network   │       │  Value Network    │
        │  Outputs: action  │       │  Outputs: scalar  │
        │  probabilities    │       │  state value      │
        └─────────┬─────────┘       └─────────┬─────────┘
                  │                           │
                  ▼                           ▼
              Select action a            Compute advantage
                  │                     A = r + γV(s') - V(s)
                  │                           │
                  └───────────┬───────────────┘
                              │
                              ▼
                    Update both networks:
                    - Actor: ∇θ ∝ A · ∇log π(a|s)
                    - Critic: minimize (V(s) - G)²
```

**Component Roles:**

| Component | Role | Updates Based On |
|-----------|------|------------------|
| **Actor** | Selects actions, represents policy | Policy gradient with advantage |
| **Critic** | Evaluates states, reduces variance | TD error or Monte Carlo returns |
| **Advantage** | How much better than average | A(s,a) = Q(s,a) - V(s) ≈ r + γV(s') - V(s) |

**A2C vs. PPO Comparison:**

| Aspect | A2C | PPO |
|--------|-----|-----|
| **Update** | Standard policy gradient | Clipped policy gradient |
| **Stability** | Can have large, destructive updates | Constrains update size |
| **Implementation** | Simpler | Slightly more complex |
| **Performance** | Good baseline | Often better, more robust |

**The Problem PPO Solves:**
```
Standard Policy Gradient Problem:
- Large gradient steps can dramatically change policy
- π_new might be very different from π_old
- Performance can collapse after one bad update
- Trust region methods (TRPO) constrain KL divergence but are complex
```

**PPO's Clipping Mechanism:**
```
L^CLIP(θ) = E[min(r(θ)·A, clip(r(θ), 1-ε, 1+ε)·A)]

Where:
- r(θ) = π_θ(a|s) / π_θ_old(a|s)  (probability ratio)
- ε ≈ 0.2 (clipping parameter)
- A = advantage estimate
```

**How It Works:**
```
If A > 0 (good action):
  - Want to increase π(a|s)
  - But clip prevents r(θ) > 1+ε
  - Limits how much we increase probability

If A < 0 (bad action):
  - Want to decrease π(a|s)
  - But clip prevents r(θ) < 1-ε
  - Limits how much we decrease probability

Result: Policy changes bounded; stable training
```

**Critical Knowledge Flag:** Yes - Dominant approach in modern deep RL

---

---
### Card 5 | Hard
**Cognitive Level:** Evaluate/Synthesize
**Concept:** Deep RL: DQN and Stability
**Source Section:** Core Concepts 9, 10
**Concept Map Centrality:** DQN (5), integrates TD, Q-Learning, Deep RL

**FRONT (Question):**
Synthesize the complete DQN algorithm, explaining: (1) why naive deep Q-learning is unstable, (2) how experience replay addresses sample correlation, (3) how target networks address moving targets, and (4) the "deadly triad" and when it can cause divergence.

**BACK (Answer):**
**1. Why Naive Deep Q-Learning is Unstable:**

```
Naive Approach:
Q_θ(s,a) ← Q_θ(s,a) + α[r + γ max Q_θ(s',a') - Q_θ(s,a)]
                              └──────┬──────┘
                           Uses same network!
```

**Problems:**
| Issue | Cause | Effect |
|-------|-------|--------|
| **Correlated samples** | Sequential experience (s_t, s_{t+1}, ...) | Gradients are correlated; unstable |
| **Non-stationary targets** | Target depends on Q_θ which is changing | Chasing moving target |
| **Overestimation** | max operator + noise = upward bias | Q-values explode |

**2. Experience Replay Solution:**

```
┌────────────────────────────────────────────────────────┐
│                   Replay Buffer D                       │
│  ┌──────────────────────────────────────────────────┐  │
│  │ (s_1, a_1, r_1, s'_1) │ (s_2, a_2, r_2, s'_2) │...│  │
│  └──────────────────────────────────────────────────┘  │
│         ↑ Store                    ↓ Sample randomly   │
└─────────┴──────────────────────────┴───────────────────┘
```

**How It Helps:**
- Store transitions in buffer, sample random mini-batches
- Breaks temporal correlation between samples
- Each transition can be used multiple times (sample efficiency)
- Smooths out learning by mixing old and new experiences

**3. Target Network Solution:**

```
Two Networks:
- Online Network Q_θ: Updated every step
- Target Network Q_θ^-: Updated slowly (copy every N steps)

Update Rule:
Q_θ(s,a) ← Q_θ(s,a) + α[r + γ max Q_θ^-(s',a') - Q_θ(s,a)]
                              └───────┬───────┘
                         Uses frozen target network!

Periodic Update:
Every N steps: θ^- ← θ
(Or soft update: θ^- ← τθ + (1-τ)θ^-, τ ≈ 0.001)
```

**How It Helps:**
- Target values don't change during learning period
- Provides stable target to fit towards
- Reduces oscillation and divergence

**Complete DQN Algorithm:**

```python
Initialize:
  Q_θ (online network), Q_θ^- (target network, copy of Q_θ)
  Replay buffer D (capacity N)

For each episode:
  s = initial state
  For each step:
    # Select action (ε-greedy)
    a = argmax Q_θ(s,a) with prob 1-ε, else random

    # Execute and observe
    s', r, done = env.step(a)

    # Store transition
    D.store(s, a, r, s', done)

    # Sample mini-batch
    batch = D.sample(batch_size=32)

    # Compute targets
    for (s_i, a_i, r_i, s'_i, done_i) in batch:
      if done_i:
        y_i = r_i
      else:
        y_i = r_i + γ max_{a'} Q_θ^-(s'_i, a')

    # Gradient descent on (Q_θ(s_i, a_i) - y_i)²
    θ ← θ - α ∇_θ Σ (Q_θ(s_i, a_i) - y_i)²

    # Periodic target update
    if step % target_update_freq == 0:
      θ^- ← θ

    s = s'
```

**4. The Deadly Triad:**

```
The Deadly Triad (causes potential divergence):

    ┌─────────────────┐
    │   Function      │     Using neural networks
    │ Approximation   │     instead of tables
    └────────┬────────┘
             │
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
┌──────────┐    ┌──────────┐
│  Boot-   │    │   Off-   │
│ strapping│    │  Policy  │
│          │    │          │
│TD targets│    │Q-learning│
│use Q est.│    │behavior≠ │
│          │    │ target   │
└──────────┘    └──────────┘
```

**When All Three Combine:**
- Function approximation generalizes (intended)
- But can generalize errors from bootstrapping
- Off-policy makes distribution mismatch worse
- Updates in one part of state space affect others
- Can create feedback loops → divergence

**Mitigation Strategies:**
| Strategy | How It Helps |
|----------|--------------|
| Target networks | Reduce bootstrap instability |
| Experience replay | Better sample distribution |
| Clipping gradients | Prevent explosion |
| Double DQN | Reduce overestimation |
| Careful architecture | Residual connections, normalization |

**Critical Knowledge Flag:** Yes - Essential for understanding deep RL stability

---

---

## Export Formats

### Anki-Compatible (Tab-Separated)
```
What are the 5 MDP components and what do V(s) and Q(s,a) represent?	MDP: States, Actions, Transitions P(s'|s,a), Rewards R, Discount γ. V(s) = expected return from state s. Q(s,a) = expected return from taking a in s. Policy derived from Q: π*(s) = argmax Q*(s,a).	easy::mdp::rl
Write Bellman equation for Q* and explain Q-learning's off-policy nature	Q*(s,a) = E[R + γ max Q*(s',a')]. Q-learning: Q ← Q + α[r + γ max Q(s',a') - Q(s,a)]. Off-policy: uses max (greedy) in update regardless of actual action taken.	easy::qlearning::rl
Explain policy gradient theorem, REINFORCE, and variance reduction	∇J = E[∇log π(a|s) · Q(s,a)]. REINFORCE: θ ← θ + α·G·∇log π. High variance from G variability. Baseline b(s) reduces variance: use (Q-V) instead of Q.	medium::policygradient::rl
Design Actor-Critic and explain PPO's clipping	Actor (policy π_θ) + Critic (value V_w). Advantage A = r + γV(s') - V(s). PPO clips ratio r(θ) = π_new/π_old to [1-ε, 1+ε] preventing destructive updates.	medium::actorcritic::rl
Synthesize DQN: instability causes and solutions, deadly triad	Instability: correlated samples, moving targets. Solutions: Experience replay (break correlation), Target networks (stable targets). Deadly triad: function approx + bootstrapping + off-policy → potential divergence.	hard::dqn::rl
```

### CSV Format
```csv
"Front","Back","Difficulty","Concept","Centrality"
"MDP components and V/Q functions?","5 components: S,A,P,R,γ. V(s)=expected return from s. Q(s,a)=expected return from (s,a).","Easy","Framework","Critical"
"Bellman equation and Q-learning?","Q*=E[R+γ max Q*]. Update uses max regardless of action taken (off-policy).","Easy","Algorithms","Critical"
"Policy gradient and variance?","∇J=E[∇log π · Q]. High variance; reduce with baseline (Q-V).","Medium","Policy Methods","High"
"Actor-Critic and PPO?","Actor (policy) + Critic (value). PPO clips probability ratio for stability.","Medium","Hybrid Methods","High"
"DQN stability and deadly triad?","Replay + target networks. Triad: approx + bootstrap + off-policy = instability.","Hard","Deep RL","High"
```

---

## Source Mapping

| Card | Source Sections | Concept Map Nodes | Key Terms |
|------|-----------------|-------------------|-----------|
| 1 | Concepts 1, 2, 3 | MDP, Value Functions | State, action, reward, V(s), Q(s,a) |
| 2 | Concepts 4, 5, 6 | Bellman, Q-Learning, TD | Bellman equation, off-policy, TD error |
| 3 | Concept 7 | Policy Gradient, REINFORCE | Score function, baseline, variance |
| 4 | Concept 8 | Actor-Critic, PPO | Advantage, clipping, actor, critic |
| 5 | Concepts 9, 10 | DQN, Experience Replay, Target Networks | Deadly triad, stability, deep RL |
