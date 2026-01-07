# Practice Problems: Lesson 6 - Reinforcement Learning

**Source:** Lessons/Lesson_6.md
**Subject Area:** AI Learning - Reinforcement Learning: Foundations, Algorithms, and Decision-Making
**Date Generated:** 2026-01-08
**Total Problems:** 5
**Estimated Completion Time:** 90-120 minutes

---

## Problem Distribution

| Type | Count | Difficulty | Focus |
|------|-------|------------|-------|
| Warm-Up | 1 | Foundation | Direct concept application |
| Skill-Builder | 2 | Intermediate | Multi-step procedures |
| Challenge | 1 | Advanced | Complex synthesis |
| Debug/Fix | 1 | Diagnostic | Error identification |

---

## Problem 1: Warm-Up - Bellman Equation Computation

**Difficulty:** Foundation
**Estimated Time:** 15 minutes
**Concepts:** Value functions, Bellman equations, expected value

### Problem Statement

Consider a simple 3-state MDP with the following characteristics:

**States:** S = {A, B, C} where C is terminal
**Actions:** {left, right}
**Discount factor:** γ = 0.9

**Transition dynamics:**
| State | Action | Next State | Probability | Reward |
|-------|--------|------------|-------------|--------|
| A | right | B | 1.0 | +5 |
| A | left | A | 1.0 | -1 |
| B | right | C | 0.8 | +10 |
| B | right | A | 0.2 | +0 |
| B | left | A | 1.0 | +2 |

**Task:**
Given the policy π that always chooses "right", compute V^π(A) and V^π(B) using the Bellman expectation equation.

### Hints

<details>
<summary>Hint 1 (Conceptual)</summary>
Start with the terminal state: V^π(C) = 0 (no future rewards from terminal).
Then use the Bellman equation to express V^π(B) in terms of V^π(A) and V^π(C).
</details>

<details>
<summary>Hint 2 (Procedural)</summary>
For state B under policy π (right):
V^π(B) = 0.8 × [R(B,right,C) + γV(C)] + 0.2 × [R(B,right,A) + γV(A)]
</details>

<details>
<summary>Hint 3 (Structural)</summary>
You'll get two equations with two unknowns (V(A) and V(B)).
V(A) = ... (involves V(B))
V(B) = ... (involves V(A))
Solve the system algebraically.
</details>

### Solution

**Step 1: Establish V^π(C)**
```
V^π(C) = 0  (terminal state, no future rewards)
```

**Step 2: Write Bellman equation for V^π(B)**

Under policy π = "right":
```
V^π(B) = P(C|B,right)[R + γV(C)] + P(A|B,right)[R + γV(A)]
       = 0.8 × [10 + 0.9 × 0] + 0.2 × [0 + 0.9 × V(A)]
       = 0.8 × 10 + 0.2 × 0.9 × V(A)
       = 8 + 0.18 × V(A)
```

**Step 3: Write Bellman equation for V^π(A)**

Under policy π = "right":
```
V^π(A) = P(B|A,right)[R + γV(B)]
       = 1.0 × [5 + 0.9 × V(B)]
       = 5 + 0.9 × V(B)
```

**Step 4: Solve the system**

Substitute V(B) into V(A) equation:
```
V(A) = 5 + 0.9 × [8 + 0.18 × V(A)]
V(A) = 5 + 7.2 + 0.162 × V(A)
V(A) - 0.162 × V(A) = 12.2
0.838 × V(A) = 12.2
V(A) = 12.2 / 0.838 ≈ 14.56
```

Then:
```
V(B) = 8 + 0.18 × 14.56
     = 8 + 2.62
     ≈ 10.62
```

**Final Answer:**
- V^π(A) ≈ **14.56**
- V^π(B) ≈ **10.62**

**Verification:** V(A) should be higher because it leads to B which often leads to the +10 terminal reward, while also collecting +5 along the way.

---

## Problem 2: Skill-Builder - Q-Learning Implementation

**Difficulty:** Intermediate
**Estimated Time:** 25 minutes
**Concepts:** Q-learning update, ε-greedy exploration, off-policy learning

### Problem Statement

You are implementing Q-learning for a grid world. The agent starts with all Q-values initialized to 0.

**Parameters:**
- Learning rate α = 0.1
- Discount factor γ = 0.95
- Exploration rate ε = 0.1

**Episode experience (sequence of transitions):**
```
Step 1: s=(0,0), a=RIGHT, r=0, s'=(0,1)
Step 2: s=(0,1), a=RIGHT, r=0, s'=(0,2)
Step 3: s=(0,2), a=DOWN,  r=+10, s'=TERMINAL
```

**Task:**
1. Show the Q-table updates after each step
2. After training, which action would a greedy policy select in state (0,0)?
3. If ε = 0.1, what is the probability of taking action RIGHT in state (0,0) after training?

### Hints

<details>
<summary>Hint 1 (Update Rule)</summary>
Q-learning update: Q(s,a) ← Q(s,a) + α[r + γ max_{a'} Q(s',a') - Q(s,a)]
</details>

<details>
<summary>Hint 2 (Backward Propagation)</summary>
Process steps in order. Step 3's update affects Q(0,2,DOWN) first.
Then Step 2's update uses the new Q(0,2) values.
</details>

<details>
<summary>Hint 3 (ε-greedy)</summary>
With probability ε: random action (uniform over all actions)
With probability 1-ε: greedy action (argmax Q)
P(RIGHT) = ε × (1/4) + (1-ε) × 1 if RIGHT is greedy
</details>

### Solution

**Initial Q-table:** All Q(s,a) = 0

**Step 1: Update Q(s=(0,0), a=RIGHT)**
```
Target = r + γ max Q(s',·) = 0 + 0.95 × max Q((0,1),·) = 0 + 0.95 × 0 = 0
Q((0,0), RIGHT) ← 0 + 0.1 × [0 - 0] = 0
```
No change (bootstrap from zero).

**Step 2: Update Q(s=(0,1), a=RIGHT)**
```
Target = r + γ max Q(s',·) = 0 + 0.95 × max Q((0,2),·) = 0 + 0.95 × 0 = 0
Q((0,1), RIGHT) ← 0 + 0.1 × [0 - 0] = 0
```
No change yet.

**Step 3: Update Q(s=(0,2), a=DOWN)**
```
s' = TERMINAL, so max Q(s',·) = 0
Target = r + γ × 0 = 10 + 0 = 10
Q((0,2), DOWN) ← 0 + 0.1 × [10 - 0] = 1.0
```

**Q-table after Episode 1:**
| State | UP | DOWN | LEFT | RIGHT |
|-------|-----|------|------|-------|
| (0,0) | 0 | 0 | 0 | 0 |
| (0,1) | 0 | 0 | 0 | 0 |
| (0,2) | 0 | **1.0** | 0 | 0 |

**After more episodes (showing convergence pattern):**

If we repeat this episode:
- Step 3: Q((0,2),DOWN) → 1.0 + 0.1×[10-1.0] = 1.9
- Step 2: Q((0,1),RIGHT) → 0 + 0.1×[0+0.95×1.9] = 0.18
- Step 1: Q((0,0),RIGHT) → 0 + 0.1×[0+0.95×0.18] = 0.017

**After convergence:**
The optimal Q-values would be:
- Q*((0,2), DOWN) = 10
- Q*((0,1), RIGHT) = 0.95 × 10 = 9.5
- Q*((0,0), RIGHT) = 0.95 × 9.5 = 9.025

**Question 2 Answer:** Greedy policy in (0,0) selects **RIGHT** (highest Q-value).

**Question 3 Answer:**
Assuming 4 actions (UP, DOWN, LEFT, RIGHT) and RIGHT is greedy:
```
P(RIGHT) = ε × (1/4) + (1-ε) × 1
         = 0.1 × 0.25 + 0.9 × 1
         = 0.025 + 0.9
         = 0.925 (92.5%)
```

---

## Problem 3: Skill-Builder - Policy Gradient Analysis

**Difficulty:** Intermediate
**Estimated Time:** 25 minutes
**Concepts:** Policy gradient theorem, REINFORCE, variance reduction

### Problem Statement

You are training a policy π_θ(a|s) using REINFORCE on a simple bandit problem (single state, 3 actions).

**Current policy parameters produce:**
- π(a₁|s) = 0.5
- π(a₂|s) = 0.3
- π(a₃|s) = 0.2

**Sampled episode:**
- Action taken: a₂
- Return received: G = +8

**Baseline value (estimated):** b(s) = +5

**Task:**
1. Compute the policy gradient for this sample WITHOUT the baseline
2. Compute the policy gradient WITH the baseline (using advantage)
3. Explain why the baseline reduces variance while keeping the gradient unbiased

### Hints

<details>
<summary>Hint 1 (Gradient Formula)</summary>
REINFORCE gradient: ∇_θ J ∝ ∇_θ log π_θ(a|s) × G
With baseline: ∇_θ J ∝ ∇_θ log π_θ(a|s) × (G - b(s))
</details>

<details>
<summary>Hint 2 (Log Derivative)</summary>
For softmax policy: ∇_θ log π(a_i) = e_i - π
where e_i is one-hot vector for action i
</details>

<details>
<summary>Hint 3 (Why Unbiased)</summary>
E[∇log π × b] = b × E[∇log π] = b × 0 (gradient of constant sums to zero)
</details>

### Solution

**Part 1: Gradient WITHOUT baseline**

The REINFORCE update direction is proportional to:
```
∇_θ log π_θ(a₂|s) × G
```

For a softmax policy, ∇log π(a₂) pushes probability mass toward a₂:
```
Gradient direction ∝ (e₂ - π) × G
                   = ([0, 1, 0] - [0.5, 0.3, 0.2]) × 8
                   = [-0.5, 0.7, -0.2] × 8
                   = [-4.0, +5.6, -1.6]
```

Interpretation: Increase π(a₂) by +5.6 units, decrease π(a₁) by 4.0, decrease π(a₃) by 1.6 (scaled by learning rate).

**Part 2: Gradient WITH baseline**

Using advantage A = G - b = 8 - 5 = 3:
```
Gradient direction ∝ (e₂ - π) × A
                   = [-0.5, 0.7, -0.2] × 3
                   = [-1.5, +2.1, -0.6]
```

**Comparison:**
| Component | No Baseline | With Baseline | Ratio |
|-----------|-------------|---------------|-------|
| Δπ(a₁) | -4.0 | -1.5 | 2.67× |
| Δπ(a₂) | +5.6 | +2.1 | 2.67× |
| Δπ(a₃) | -1.6 | -0.6 | 2.67× |

The baseline scales down the gradient magnitude by G/A = 8/3 ≈ 2.67×.

**Part 3: Why baseline reduces variance while staying unbiased**

**Unbiasedness proof:**
```
E_a~π [∇log π(a|s) × b(s)]
= b(s) × E_a~π [∇log π(a|s)]
= b(s) × Σ_a π(a|s) × ∇log π(a|s)
= b(s) × Σ_a π(a|s) × [∇π(a|s) / π(a|s)]
= b(s) × Σ_a ∇π(a|s)
= b(s) × ∇Σ_a π(a|s)
= b(s) × ∇(1)
= b(s) × 0
= 0
```

Since adding baseline contributes zero in expectation, the gradient estimate remains unbiased.

**Variance reduction intuition:**

| Factor | Without Baseline | With Baseline |
|--------|------------------|---------------|
| Multiplier | G (varies: 0, 5, 10, 15...) | A = G - b (varies: -5, 0, +5, +10...) |
| Mean | E[G] = μ | E[A] = μ - b ≈ 0 |
| Variance source | Scales with |G| | Scales with |G - b| |

When b ≈ E[G], the advantage A centers around zero. Positive advantages increase action probability; negative advantages decrease it. This is more informative than always-positive returns that always push in the same direction with varying magnitude.

---

## Problem 4: Challenge - Complete RL System Design

**Difficulty:** Advanced
**Estimated Time:** 30 minutes
**Concepts:** Algorithm selection, architecture design, exploration, stability

### Problem Statement

You are tasked with designing an RL system for an autonomous delivery drone with the following characteristics:

**Environment:**
- State: 12D vector (position, velocity, orientation, battery level, package status)
- Actions: Continuous 4D (throttle, pitch, roll, yaw rates)
- Episode length: ~500 steps average
- Reward: +100 for successful delivery, -50 for crash, -0.1 per step (encourage efficiency)

**Constraints:**
- Training budget: 10 million environment steps
- Safety: Crashes must be minimized during training
- Deployment: Must work on real drone after sim training

**Task:**
Design a complete RL solution addressing:
1. Algorithm selection with justification
2. Network architecture
3. Exploration strategy that respects safety
4. Sim-to-real transfer considerations
5. Training stability measures

### Hints

<details>
<summary>Hint 1 (Algorithm)</summary>
Continuous actions + sample efficiency need → consider SAC or PPO.
Safety concern → consider constrained RL or careful exploration.
</details>

<details>
<summary>Hint 2 (Safety)</summary>
Safe exploration techniques: action smoothing, learned constraints, conservative exploration near boundaries, reward shaping to penalize risky states.
</details>

<details>
<summary>Hint 3 (Sim-to-Real)</summary>
Domain randomization (vary physics parameters), system identification, conservative deployment policy, real-world fine-tuning with safety constraints.
</details>

### Solution

**1. Algorithm Selection: SAC (Soft Actor-Critic)**

| Requirement | SAC Advantage |
|-------------|---------------|
| Continuous actions | Natural Gaussian policy output |
| Sample efficiency | Off-policy + replay buffer |
| Exploration | Entropy maximization built-in |
| Stability | Twin Q-networks reduce overestimation |

**Alternative:** PPO if stability is paramount, but SAC's sample efficiency is valuable with 10M step budget.

**Hyperparameters:**
```
- Learning rate: 3e-4 (actor and critic)
- Batch size: 256
- Replay buffer: 1M transitions
- Target smoothing τ: 0.005
- Entropy coefficient α: auto-tuned
- Discount γ: 0.99
```

**2. Network Architecture**

```
State Encoder:
  Input: 12D state vector
  Hidden: [256, 256] with ReLU

Actor (Policy):
  Input: 256D encoded state
  Hidden: [256, 256] with ReLU
  Output: 4D mean μ, 4D log_std
  Action: a ~ tanh(N(μ, σ²))  # bounded actions

Critic (Twin Q-Networks):
  Input: 256D state + 4D action
  Hidden: [256, 256] with ReLU
  Output: 1D Q-value
  Two separate critics for min Q estimation
```

**3. Safe Exploration Strategy**

| Technique | Implementation |
|-----------|----------------|
| Action smoothing | Low-pass filter on actions: a_t = 0.8×a_{t-1} + 0.2×π(s_t) |
| Boundary avoidance | Reward shaping: r -= 10 × max(0, altitude_threshold - altitude) |
| Conservative initialization | Pre-train policy on expert demonstrations (behavior cloning) |
| Safety critic | Additional critic predicting crash probability, reject high-risk actions |

**Exploration schedule:**
```
# Entropy coefficient schedule
α_init = 0.2 (high exploration early)
α_final = 0.05 (reduced at deployment)
Decay: linear over 5M steps
```

**4. Sim-to-Real Transfer**

| Strategy | Implementation |
|----------|----------------|
| Domain randomization | Vary: mass ±20%, drag ±30%, motor latency ±50ms, wind gusts |
| Observation noise | Add Gaussian noise to state: σ = 0.05 × state_range |
| Action delay | Random delay 0-3 steps to simulate real latency |
| Dynamics randomization | Randomize physics parameters each episode |

**Deployment strategy:**
```
1. Train in randomized sim (8M steps)
2. Fine-tune on high-fidelity sim (1M steps)
3. Deploy with conservative policy (lower entropy)
4. Optional: online adaptation with safety constraints
```

**5. Training Stability Measures**

| Measure | Purpose |
|---------|---------|
| Gradient clipping | Global norm ≤ 1.0 |
| Target networks | Soft update τ = 0.005 |
| Learning rate warmup | Linear warmup over 10K steps |
| Reward normalization | Running mean/std normalization |
| Early termination | End episode on unsafe state detection |
| Checkpointing | Save every 100K steps, keep best 5 |

**Monitoring metrics:**
```
- Episode return (smoothed)
- Crash rate per 1000 episodes
- Average episode length
- Q-value statistics (mean, max to detect overestimation)
- Policy entropy
- Gradient norms
```

**Training timeline:**
```
Steps 0-1M: High exploration, expect crashes, learning basic control
Steps 1-5M: Improving delivery rate, fewer crashes
Steps 5-8M: Domain randomization emphasis, robustness
Steps 8-10M: Fine-tuning, conservative policy for deployment
```

---

## Problem 5: Debug/Fix - Training Failure Diagnosis

**Difficulty:** Diagnostic
**Estimated Time:** 20 minutes
**Concepts:** Training instability, reward design, exploration failures

### Problem Statement

A colleague is training DQN on a navigation task and reports the following issues. For each scenario, identify the problem and propose a fix.

**Scenario A: Flat Q-values**
```
Training log:
Episode 100: avg_return = -50, max_Q = 0.12, min_Q = 0.08
Episode 500: avg_return = -48, max_Q = 0.15, min_Q = 0.11
Episode 1000: avg_return = -47, max_Q = 0.14, min_Q = 0.10
```
Q-values barely change and returns don't improve.

**Scenario B: Q-value explosion**
```
Training log:
Episode 100: avg_return = 5, max_Q = 50
Episode 200: avg_return = 6, max_Q = 500
Episode 300: avg_return = 4, max_Q = 50000
Episode 400: avg_return = NaN, max_Q = inf
```

**Scenario C: Oscillating performance**
```
Training log:
Episode 100: avg_return = 20
Episode 200: avg_return = 45
Episode 300: avg_return = 15
Episode 400: avg_return = 50
Episode 500: avg_return = 10
Episode 600: avg_return = 55
```
Performance swings wildly despite training progress.

### Hints

<details>
<summary>Hint A</summary>
Flat Q-values suggest the learning signal isn't propagating. Check: replay buffer size, exploration, reward scale, network capacity.
</details>

<details>
<summary>Hint B</summary>
Q-value explosion indicates overestimation and instability. Check: target network update frequency, learning rate, reward clipping.
</details>

<details>
<summary>Hint C</summary>
Oscillating performance often relates to policy churn or evaluation variance. Check: ε schedule, target network updates, evaluation methodology.
</details>

### Solution

**Scenario A: Flat Q-values - Diagnosis & Fix**

**Likely Problems:**
1. **Insufficient exploration:** Agent stuck in local behavior, never finds rewards
2. **Sparse rewards:** No learning signal reaching early states
3. **Replay buffer too large:** Old, irrelevant experiences dominate

**Diagnostic checks:**
```python
# Check exploration
print(f"ε at episode 1000: {epsilon}")  # If ε < 0.1, too low too early

# Check reward distribution in buffer
rewards = [t.reward for t in replay_buffer.sample(10000)]
print(f"Reward stats: mean={np.mean(rewards)}, std={np.std(rewards)}")
# If all rewards ≈ -0.1 (step penalty), agent never reaches goal
```

**Fixes:**

| Problem | Solution |
|---------|----------|
| Low exploration | Increase ε, slower decay schedule |
| Sparse rewards | Add reward shaping (distance to goal) |
| Large buffer | Start with smaller buffer (100K), prioritized replay |
| Dead network | Check for dying ReLUs, use LeakyReLU |

**Recommended fix:**
```python
# Slower ε decay
epsilon = max(0.1, 1.0 - episode/5000)  # Was decaying too fast

# Add reward shaping
shaped_reward = reward + 0.1 * (prev_dist - curr_dist)  # Reward progress
```

---

**Scenario B: Q-value explosion - Diagnosis & Fix**

**Likely Problems:**
1. **Target network not used or updated too frequently**
2. **Learning rate too high**
3. **Rewards not clipped/normalized**

**Root cause:** Q-value overestimation feedback loop
```
High Q → high TD target → even higher Q → ...
```

**Fixes:**

| Problem | Solution |
|---------|----------|
| No target network | Add target network, update every 10K steps |
| Fast target update | Increase target update interval (1K → 10K) |
| High learning rate | Reduce to 1e-4 or 5e-5 |
| Large rewards | Clip rewards to [-1, 1] |

**Recommended fix:**
```python
# Add target network with slow updates
if step % 10000 == 0:
    target_network.load_state_dict(q_network.state_dict())

# Use target for TD computation
with torch.no_grad():
    target = reward + gamma * target_network(next_state).max()

# Clip rewards
clipped_reward = np.clip(reward, -1, 1)

# Consider Double DQN to reduce overestimation
# a* = argmax Q_online(s', a')
# target = r + γ Q_target(s', a*)
```

---

**Scenario C: Oscillating performance - Diagnosis & Fix**

**Likely Problems:**
1. **Evaluation variance:** Too few evaluation episodes
2. **Policy churn:** Q-function changes cause policy instability
3. **ε decay during evaluation:** Evaluation includes exploration

**Fixes:**

| Problem | Solution |
|---------|----------|
| Evaluation variance | Use 10+ evaluation episodes, separate eval env |
| Policy churn | Larger replay buffer, slower learning rate |
| ε in evaluation | Use ε=0 for evaluation, ε>0 only for training |

**Recommended fix:**
```python
# Separate training and evaluation
def evaluate(env, policy, n_episodes=10):
    returns = []
    for _ in range(n_episodes):
        state = env.reset()
        episode_return = 0
        done = False
        while not done:
            action = policy.greedy_action(state)  # ε = 0
            state, reward, done, _ = env.step(action)
            episode_return += reward
        returns.append(episode_return)
    return np.mean(returns), np.std(returns)

# Report mean ± std for meaningful comparison
mean_return, std_return = evaluate(eval_env, policy)
print(f"Episode {ep}: {mean_return:.1f} ± {std_return:.1f}")
```

**Additional stability:**
```python
# Use larger replay buffer
replay_buffer = ReplayBuffer(size=500000)  # Was 50000

# Slower learning rate
optimizer = Adam(lr=1e-4)  # Was 1e-3

# Gradient clipping
torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=10)
```

---

## Self-Assessment Guide

After completing these problems, evaluate your understanding:

| Level | Criteria |
|-------|----------|
| **Mastery** | Completed all problems correctly, identified multiple valid approaches |
| **Proficient** | Solved 4/5 problems, minor errors in complex scenarios |
| **Developing** | Solved warm-up and skill-builders, struggled with challenge/debug |
| **Foundational** | Need review of core concepts before attempting problems |

### Common Mistakes to Avoid

1. **Forgetting terminal state handling:** V(terminal) = 0, no bootstrapping
2. **Confusing on/off-policy:** SARSA uses actual a', Q-learning uses max
3. **Ignoring exploration in deployment:** ε should be 0 for evaluation
4. **Overlooking reward scale:** Large rewards cause instability
5. **Not using target networks:** Essential for DQN stability

---

*Generated from Lesson 6: Reinforcement Learning | Practice Problems Skill*
