# Lesson 6: Reinforcement Learning

**Topic:** Reinforcement Learning: Foundations, Algorithms, and Decision-Making Under Uncertainty
**Prerequisite:** Lesson 5 (Deep Learning fundamentals for Deep RL section)
**Estimated Study Time:** 3-4 hours
**Difficulty:** Advanced

---

## Learning Objectives

Upon completion of this lesson, learners will be able to:

1. **Analyze** the components of reinforcement learning problems including agents, environments, states, actions, and rewards
2. **Apply** Bellman equations and temporal difference methods to compute value functions
3. **Compare** value-based, policy-based, and actor-critic algorithms for different problem characteristics
4. **Evaluate** exploration strategies and their impact on learning efficiency
5. **Design** RL solutions considering sample efficiency, stability, and real-world deployment challenges

---

## Introduction

Reinforcement Learning (RL) represents a fundamentally different paradigm from supervised and unsupervised learning. Rather than learning from labeled examples or discovering patterns in data, an RL agent learns through interaction with an environment, receiving scalar reward signals that indicate the quality of its decisions. This trial-and-error learning process mirrors how humans and animals acquire skills—through experience, feedback, and adaptation.

The power of RL lies in its generality: any sequential decision-making problem can be formulated within the RL framework. From mastering board games to controlling robots, from optimizing data centers to personalizing recommendations, RL provides a principled approach to learning optimal behavior over time.

---

## Core Concepts

### Concept 1: The Reinforcement Learning Framework

The RL framework formalizes the interaction between an **agent** (the learner) and an **environment** (everything external to the agent). At each timestep t:

1. Agent observes state s_t
2. Agent selects action a_t based on its policy π
3. Environment transitions to state s_{t+1}
4. Agent receives reward r_{t+1}

```
┌─────────────────────────────────────────────┐
│                                             │
│    ┌───────┐         action a_t             │
│    │       │ ─────────────────────────────► │
│    │ Agent │                          Environment
│    │       │ ◄───────────────────────────── │
│    └───────┘    state s_t, reward r_t       │
│                                             │
└─────────────────────────────────────────────┘
```

**Key Components:**

| Component | Symbol | Description |
|-----------|--------|-------------|
| State | s ∈ S | Representation of the current situation |
| Action | a ∈ A | Choice available to the agent |
| Reward | r ∈ ℝ | Scalar feedback signal |
| Policy | π(a\|s) | Strategy mapping states to action probabilities |
| Trajectory | τ | Sequence (s_0, a_0, r_1, s_1, a_1, r_2, ...) |

**The Objective:** Find a policy π that maximizes expected cumulative reward:

```
J(π) = E_τ~π [Σ_{t=0}^{∞} γ^t r_{t+1}]
```

where γ ∈ [0, 1] is the discount factor balancing immediate versus future rewards.

---

### Concept 2: Markov Decision Processes (MDPs)

An MDP provides the mathematical formalization of RL problems, defined by the tuple (S, A, P, R, γ):

| Element | Definition |
|---------|------------|
| S | Set of states |
| A | Set of actions |
| P(s'\|s,a) | Transition probability: P(s_{t+1} = s' \| s_t = s, a_t = a) |
| R(s,a,s') | Reward function |
| γ | Discount factor |

**The Markov Property:**

The crucial assumption: the future depends only on the current state, not the history:

```
P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ..., s_0, a_0) = P(s_{t+1} | s_t, a_t)
```

This property enables tractable computation—we only need to track the current state, not the entire history.

**Problem Variants:**

| Variant | Characteristic | Example |
|---------|---------------|---------|
| Episodic | Terminates after finite steps | Game with win/lose |
| Continuing | Runs indefinitely | Process control |
| Model-based | Transition dynamics P known/learned | Planning |
| Model-free | Learn directly from experience | Most deep RL |

---

### Concept 3: Value Functions

Value functions quantify "how good" it is to be in a state or take an action, measuring expected cumulative discounted reward.

**State-Value Function V^π(s):**

Expected return starting from state s, following policy π:

```
V^π(s) = E_π [Σ_{t=0}^{∞} γ^t r_{t+1} | s_0 = s]
```

**Action-Value Function Q^π(s,a):**

Expected return starting from s, taking action a, then following π:

```
Q^π(s,a) = E_π [Σ_{t=0}^{∞} γ^t r_{t+1} | s_0 = s, a_0 = a]
```

**Relationship Between V and Q:**

```
V^π(s) = Σ_a π(a|s) Q^π(s,a)        (V as expectation over Q)
Q^π(s,a) = R(s,a) + γ Σ_{s'} P(s'|s,a) V^π(s')   (Q in terms of V)
```

**Optimal Value Functions:**

```
V*(s) = max_π V^π(s)    (best possible value)
Q*(s,a) = max_π Q^π(s,a)

π*(s) = argmax_a Q*(s,a)  (greedy policy from Q*)
```

---

### Concept 4: Bellman Equations

Bellman equations express the recursive structure of value functions—the value of a state depends on immediate reward plus discounted future value.

**Bellman Expectation Equation (for policy π):**

```
V^π(s) = Σ_a π(a|s) [R(s,a) + γ Σ_{s'} P(s'|s,a) V^π(s')]
```

In words: Value = Expected[Immediate Reward + Discounted Next State Value]

**Bellman Optimality Equation:**

```
V*(s) = max_a [R(s,a) + γ Σ_{s'} P(s'|s,a) V*(s')]

Q*(s,a) = R(s,a) + γ Σ_{s'} P(s'|s,a) max_{a'} Q*(s',a')
```

The optimal value uses max instead of expectation—always take the best action.

**Why Bellman Equations Matter:**

| Application | Method |
|-------------|--------|
| Known dynamics | Dynamic Programming (Policy/Value Iteration) |
| Unknown dynamics | Temporal Difference Learning |
| Large state spaces | Function approximation + bootstrapping |

---

### Concept 5: Temporal Difference Learning

TD learning combines ideas from Monte Carlo (learning from experience) and Dynamic Programming (bootstrapping from estimates).

**TD(0) Update Rule:**

```
V(s_t) ← V(s_t) + α [r_{t+1} + γV(s_{t+1}) - V(s_t)]
                    └──────────────────────────────┘
                              TD error (δ)
```

**Key Properties:**

| Property | Monte Carlo | TD Learning |
|----------|-------------|-------------|
| Updates when | End of episode | Every step |
| Uses estimates | No (actual returns) | Yes (bootstrapping) |
| Requires episodes | Yes | No (works with continuing) |
| Variance | High | Lower |
| Bias | Unbiased | Biased (but consistent) |

**SARSA (On-Policy TD Control):**

```
Q(s,a) ← Q(s,a) + α [r + γQ(s',a') - Q(s,a)]
```

Updates Q using the action a' actually taken in s' (follows current policy).

---

### Concept 6: Q-Learning

Q-Learning is the foundational off-policy algorithm that learns the optimal Q* directly.

**Q-Learning Update:**

```
Q(s,a) ← Q(s,a) + α [r + γ max_{a'} Q(s',a') - Q(s,a)]
                         └──────────────────┘
                         Uses max, not actual a'
```

**Off-Policy Nature:**

| Aspect | Implication |
|--------|-------------|
| Behavior policy | Can be exploratory (e.g., ε-greedy) |
| Target policy | Always greedy (max Q) |
| Advantage | Learn optimal policy while exploring |
| Convergence | Guaranteed under Robbins-Monro conditions |

**Comparison: SARSA vs Q-Learning:**

```
SARSA:      Q(s,a) ← Q(s,a) + α [r + γQ(s',a') - Q(s,a)]     ← on-policy
Q-Learning: Q(s,a) ← Q(s,a) + α [r + γ max Q(s',·) - Q(s,a)] ← off-policy
```

SARSA learns the value of the policy being followed (including exploration mistakes). Q-learning learns optimal values regardless of exploration policy.

---

### Concept 7: Policy Gradient Methods

Instead of learning value functions, policy gradient methods directly optimize the policy π_θ(a|s).

**The Policy Gradient Theorem:**

```
∇_θ J(θ) = E_π [∇_θ log π_θ(a|s) · Q^π(s,a)]
```

Interpretation: Increase probability of actions proportional to how good they are.

**REINFORCE Algorithm:**

```python
# Sample trajectory τ using π_θ
for each (s_t, a_t) in trajectory:
    G_t = Σ_{k=t}^{T} γ^{k-t} r_k    # Return from t
    θ ← θ + α · ∇_θ log π_θ(a_t|s_t) · G_t
```

**Advantages of Policy Gradients:**

| Advantage | Explanation |
|-----------|-------------|
| Continuous actions | Natural parameterization (e.g., Gaussian π) |
| Stochastic policies | Can learn optimal stochastic behavior |
| Convergence | Guaranteed local optimum |

**Challenge: High Variance**

Using full returns G_t introduces high variance. Solution: subtract a baseline b(s):

```
∇_θ J ∝ ∇_θ log π_θ(a|s) · [Q(s,a) - b(s)]
```

Common baseline: V(s), giving the advantage function A(s,a) = Q(s,a) - V(s).

---

### Concept 8: Actor-Critic Methods

Actor-Critic combines policy gradients (actor) with value function estimation (critic).

**Architecture:**

```
┌─────────────────────────────────────────────────┐
│                                                 │
│  Actor: π_θ(a|s)          Critic: V_w(s)        │
│  ─────────────────        ──────────────        │
│  Updates policy           Estimates value        │
│  using advantage          Reduces variance       │
│                                                 │
└─────────────────────────────────────────────────┘
```

**Advantage Function:**

```
A(s,a) = Q(s,a) - V(s)
       ≈ r + γV(s') - V(s)   (TD estimate)
```

Advantage measures how much better action a is compared to average behavior in state s.

**A2C Update (Advantage Actor-Critic):**

```
# Critic update (minimize TD error)
L_critic = (r + γV_w(s') - V_w(s))²

# Actor update (policy gradient with advantage)
∇_θ J = ∇_θ log π_θ(a|s) · A(s,a)
```

**PPO (Proximal Policy Optimization):**

Constrains policy updates to prevent destructive large changes:

```
L^{CLIP}(θ) = E [min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]

where r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)
```

PPO is currently the most widely used policy gradient algorithm due to its stability and simplicity.

---

### Concept 9: Deep Reinforcement Learning

Deep RL combines RL algorithms with neural network function approximators, enabling learning from high-dimensional inputs.

**DQN (Deep Q-Network) Innovations:**

| Innovation | Purpose |
|------------|---------|
| Experience Replay | Store transitions (s,a,r,s') in buffer, sample randomly to break correlation |
| Target Network | Separate network for TD target, updated periodically for stability |
| CNN Features | Learn state representations from raw pixels |

**DQN Algorithm:**

```
1. Store transition (s, a, r, s') in replay buffer
2. Sample minibatch from buffer
3. Compute target: y = r + γ max_{a'} Q_{target}(s', a')
4. Update Q_θ to minimize (Q_θ(s,a) - y)²
5. Periodically: θ_target ← θ
```

**The Deadly Triad:**

Combining these three elements can cause divergence:

1. **Function approximation** (neural networks)
2. **Bootstrapping** (TD learning)
3. **Off-policy learning**

DQN's innovations (replay, target networks) mitigate but don't fully solve this instability.

**Modern Deep RL Algorithms:**

| Algorithm | Type | Key Feature |
|-----------|------|-------------|
| DQN | Value-based | Experience replay + target networks |
| Double DQN | Value-based | Reduces overestimation bias |
| Dueling DQN | Value-based | Separate V and advantage streams |
| A3C/A2C | Actor-Critic | Parallel actors |
| PPO | Actor-Critic | Clipped objective for stability |
| SAC | Actor-Critic | Entropy regularization, off-policy |
| TD3 | Actor-Critic | Twin critics, delayed updates |

---

### Concept 10: Exploration vs. Exploitation

The exploration-exploitation tradeoff is fundamental: exploit current knowledge for immediate reward, or explore to discover potentially better strategies.

**Exploration Strategies:**

| Strategy | Mechanism | Properties |
|----------|-----------|------------|
| ε-greedy | Random action with prob ε | Simple, often effective |
| Boltzmann | Sample a ~ exp(Q(s,a)/τ) | Soft preference for high-Q |
| UCB | a = argmax Q(s,a) + c√(log t / N(s,a)) | Optimism under uncertainty |
| Intrinsic motivation | Bonus for novel states | Addresses sparse rewards |

**ε-greedy Details:**

```
With probability ε:  take random action
With probability 1-ε: take argmax_a Q(s,a)
```

Common schedule: ε decays from 1.0 to 0.1 over training.

**UCB (Upper Confidence Bound):**

```
a = argmax_a [Q(s,a) + c · √(log t / N(s,a))]
                       └──────────────────────┘
                         Exploration bonus
```

Actions tried less often (small N) get larger bonus—optimism in the face of uncertainty.

**Intrinsic Motivation:**

For sparse reward environments, add curiosity-driven bonuses:

```
r_total = r_extrinsic + β · r_intrinsic

r_intrinsic examples:
- Prediction error (ICM)
- State novelty (count-based)
- Information gain
```

---

## Practical Considerations

### Sample Efficiency

RL often requires millions of environment interactions. Strategies to improve efficiency:

| Approach | Description |
|----------|-------------|
| Model-based RL | Learn environment dynamics, plan |
| Offline RL | Learn from logged data without interaction |
| Transfer learning | Pre-train on related tasks |
| Reward shaping | Provide intermediate rewards |

### Reward Design

The reward function defines the objective. Common pitfalls:

| Problem | Example | Mitigation |
|---------|---------|------------|
| Reward hacking | Agent finds unintended shortcuts | Careful specification |
| Sparse rewards | Only terminal reward | Reward shaping, intrinsic motivation |
| Reward misspecification | Optimizes proxy, not true goal | RLHF, inverse RL |

### Training Stability

Deep RL training is notoriously unstable. Best practices:

- Use target networks (DQN) or clipped updates (PPO)
- Normalize observations and rewards
- Careful hyperparameter tuning (especially learning rate)
- Multiple random seeds
- Gradient clipping

---

## Connections to Other Lessons

| Lesson | Connection |
|--------|------------|
| Lesson 5: Deep Learning | Neural network architectures, optimization, gradient flow |
| Lesson 3: LLMs | RLHF uses RL to align language models |
| Lesson 4: Transformers | Decision Transformer frames RL as sequence modeling |
| Lesson 1: Agent Skills | RL enables learning complex agent behaviors |

---

## Case Study: Training an RLHF Reward Model

Consider fine-tuning an LLM using Reinforcement Learning from Human Feedback (RLHF):

**Setup:**
- State s: Prompt + partial response
- Action a: Next token
- Policy π_θ: The LLM
- Reward model r_φ: Learned from human preferences

**Algorithm (PPO for RLHF):**

```
1. Sample prompts from dataset
2. Generate responses using current π_θ
3. Score responses using reward model r_φ
4. Compute advantage estimates
5. Update π_θ using PPO objective with KL constraint:

   L = E[r_φ(response)] - β · KL(π_θ || π_ref)
```

The KL penalty prevents the policy from deviating too far from the reference model, maintaining language quality.

**Challenges:**
- Reward model may have blind spots
- Distribution shift as policy improves
- Balancing helpfulness vs. safety objectives

---

## Summary

Reinforcement learning provides a principled framework for sequential decision-making through environment interaction. MDPs formalize the problem; value functions quantify expected returns; Bellman equations express recursive value relationships. TD learning enables model-free learning from experience, with Q-learning providing off-policy optimization. Policy gradient methods directly optimize policies, while actor-critic architectures combine both approaches for stability and efficiency. Deep RL scales these algorithms to complex domains using neural networks, though stability challenges remain. The exploration-exploitation tradeoff is fundamental—effective exploration often determines learning success. Understanding RL is essential for building adaptive, learning agents, and increasingly relevant as RLHF becomes central to LLM alignment.

---

## Quick Reference

### Core Equations

| Equation | Name |
|----------|------|
| V^π(s) = E[Σ γ^t r_t \| s_0=s, π] | State-Value Definition |
| Q^π(s,a) = E[Σ γ^t r_t \| s_0=s, a_0=a, π] | Action-Value Definition |
| V*(s) = max_a [R(s,a) + γ Σ P(s'\|s,a)V*(s')] | Bellman Optimality (V) |
| Q*(s,a) = R(s,a) + γ Σ P(s'\|s,a) max Q*(s',a') | Bellman Optimality (Q) |
| Q(s,a) ← Q(s,a) + α[r + γ max Q(s',·) - Q(s,a)] | Q-Learning Update |
| ∇J = E[∇ log π(a\|s) · Q(s,a)] | Policy Gradient Theorem |

### Algorithm Selection Guide

| Scenario | Recommended Approach |
|----------|---------------------|
| Discrete actions, moderate states | DQN or Double DQN |
| Continuous actions | PPO, SAC, or TD3 |
| Sample efficiency critical | Model-based RL or SAC |
| Stability important | PPO |
| Offline data available | Conservative Q-Learning, Decision Transformer |

---

*Next Lesson: Lesson 7 - Generative AI*
