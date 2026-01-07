# Practice Problems: Reinforcement Learning

**Source:** notes/reinforcement-learning/reinforcement-learning-study-notes.md
**Concept Map Reference:** notes/reinforcement-learning/concept-maps/reinforcement-learning-concept-map.md
**Date Generated:** 2026-01-07
**Total Problems:** 5
**Distribution:** 1 Warm-Up | 2 Skill-Builder | 1 Challenge | 1 Debug/Fix

---

## Problem Distribution Strategy

| Problem | Type | Concepts Tested | Difficulty | Time Est. |
|---------|------|-----------------|------------|-----------|
| P1 | Warm-Up | Q-Learning Updates | Low | 10-15 min |
| P2 | Skill-Builder | Bellman Equations, Value Iteration | Medium | 20-25 min |
| P3 | Skill-Builder | DQN Architecture Design | Medium | 20-25 min |
| P4 | Challenge | Actor-Critic Implementation | High | 35-45 min |
| P5 | Debug/Fix | Exploration Failures | Medium | 25-30 min |

---

## Problems

---

### Problem 1 | Warm-Up
**Concept:** Q-Learning Update Rule
**Source Section:** Core Concepts 6
**Concept Map Node:** Q-Learning (7 connections)
**Related Flashcard:** Card 2
**Estimated Time:** 10-15 minutes

#### Problem Statement

Consider a simple gridworld with 4 states {S1, S2, S3, S4} and 2 actions {Left, Right}. The agent starts with all Q-values initialized to 0.

**Environment Dynamics:**
```
S1 ←→ S2 ←→ S3 ←→ S4 (Terminal, Reward +10)

- Moving Right from S3 to S4 gives reward +10 (terminal)
- All other transitions give reward 0
- Moving Left from S1 stays at S1 (reward 0)
- Moving Right from S4 is not possible (terminal state)
```

**Parameters:**
- Learning rate α = 0.5
- Discount factor γ = 0.9
- All Q(s,a) initialized to 0

**Experience sequence:**
```
Step 1: S2, Right → S3, reward 0
Step 2: S3, Right → S4, reward 10 (terminal)
Step 3: S1, Right → S2, reward 0
Step 4: S2, Right → S3, reward 0
```

**Tasks:**
1. Apply the Q-learning update after each step
2. Show the complete Q-table after all 4 updates
3. What action would a greedy policy select in state S2 after these updates?

---

#### Hints

<details>
<summary>Hint 1 (Conceptual)</summary>

Q-learning update formula:
Q(s,a) ← Q(s,a) + α[r + γ max_{a'} Q(s',a') - Q(s,a)]

For terminal states, there is no future value, so the target is just r.
</details>

<details>
<summary>Hint 2 (Approach)</summary>

After Step 2, Q(S3, Right) will become non-zero (you get the +10 reward).
This value will then propagate backwards in subsequent updates.

Remember: max_{a'} Q(s',a') means looking at both actions in s' and taking the maximum.
</details>

<details>
<summary>Hint 3 (Solution Direction)</summary>

Step 2 update: Q(S3, R) = 0 + 0.5[10 + 0 - 0] = 5
Step 4 update: Q(S2, R) = 0 + 0.5[0 + 0.9×5 - 0] = 2.25
</details>

---

#### Solution

**Initial Q-table:**
```
| State | Left | Right |
|-------|------|-------|
| S1    | 0    | 0     |
| S2    | 0    | 0     |
| S3    | 0    | 0     |
```

**Step 1: S2, Right → S3, r=0**
```
Q(S2, Right) ← Q(S2, Right) + α[r + γ max Q(S3,·) - Q(S2, Right)]
Q(S2, Right) ← 0 + 0.5[0 + 0.9 × max(0, 0) - 0]
Q(S2, Right) ← 0 + 0.5[0 + 0 - 0]
Q(S2, Right) ← 0
```

**Step 2: S3, Right → S4 (terminal), r=10**
```
Q(S3, Right) ← Q(S3, Right) + α[r + γ × 0 - Q(S3, Right)]
              (terminal state: no future value)
Q(S3, Right) ← 0 + 0.5[10 + 0 - 0]
Q(S3, Right) ← 5
```

**Q-table after Step 2:**
```
| State | Left | Right |
|-------|------|-------|
| S1    | 0    | 0     |
| S2    | 0    | 0     |
| S3    | 0    | 5     |
```

**Step 3: S1, Right → S2, r=0**
```
Q(S1, Right) ← Q(S1, Right) + α[r + γ max Q(S2,·) - Q(S1, Right)]
Q(S1, Right) ← 0 + 0.5[0 + 0.9 × max(0, 0) - 0]
Q(S1, Right) ← 0
```

**Step 4: S2, Right → S3, r=0**
```
Q(S2, Right) ← Q(S2, Right) + α[r + γ max Q(S3,·) - Q(S2, Right)]
Q(S2, Right) ← 0 + 0.5[0 + 0.9 × max(0, 5) - 0]
Q(S2, Right) ← 0 + 0.5[0 + 0.9 × 5 - 0]
Q(S2, Right) ← 0 + 0.5[4.5]
Q(S2, Right) ← 2.25
```

**Final Q-table after all 4 updates:**
```
| State | Left | Right |
|-------|------|-------|
| S1    | 0    | 0     |
| S2    | 0    | 2.25  |
| S3    | 0    | 5     |
```

**Task 3: Greedy Policy in S2**
- Q(S2, Left) = 0
- Q(S2, Right) = 2.25
- Greedy: argmax → **Right**

**Note:** With more experience, values will continue to propagate:
- Q(S1, Right) will eventually become ~0.9 × 2.25 = 2.025
- Values converge to: Q(S3,R)=10, Q(S2,R)=9, Q(S1,R)=8.1 (with γ=0.9)

---

#### Common Mistakes

| Mistake | Why It's Wrong | Correct Approach |
|---------|----------------|------------------|
| Using γ for terminal transitions | Terminal has no future | Set future value to 0 for terminal states |
| Not using max for next state | Q-learning is off-policy | Always use max_{a'} Q(s',a'), not actual next action |
| Forgetting the subtraction | Missing old Q in TD error | Formula is α[target - Q(s,a)], not just α × target |
| Updating wrong state-action | Updating next state instead | Update Q(s,a) for the (state, action) that was TAKEN |

---

#### Extension Challenge

Repeat the same exercise using SARSA instead of Q-learning. Assume the same experience sequence with actions: Step 3 next action would be Right, Step 4 next action would be Right. How do the final Q-values differ?

---

---

### Problem 2 | Skill-Builder
**Concept:** Bellman Equations and Value Iteration
**Source Section:** Core Concepts 3, 4
**Concept Map Node:** Bellman Equations (6), Value Functions (9)
**Related Flashcard:** Card 1, Card 2
**Estimated Time:** 20-25 minutes

#### Problem Statement

Consider a robot navigation MDP with 3 states {A, B, C} where C is a terminal goal state.

**MDP Definition:**
```
States: {A, B, C}
Actions: {go} (deterministic transitions)
Transitions:
  - From A: go → B (probability 1.0)
  - From B: go → C (probability 0.8), stay at B (probability 0.2)
  - C is terminal

Rewards:
  - Reaching C: +100
  - Staying at B: -1
  - All other transitions: 0

Discount factor: γ = 0.9
```

**Tasks:**
1. Write the Bellman optimality equation for V*(A) and V*(B)
2. Perform 3 iterations of value iteration starting from V(A) = V(B) = 0
3. After iteration 3, what is the optimal value of starting in state A?
4. Why does the value of B include both possible outcomes?

---

#### Hints

<details>
<summary>Hint 1 (Conceptual)</summary>

Bellman optimality equation:
V*(s) = max_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γV*(s')]

Since there's only one action, there's no max needed—just the expectation over next states.
</details>

<details>
<summary>Hint 2 (Approach)</summary>

For state B:
- With prob 0.8: go to C, get +100
- With prob 0.2: stay at B, get -1

V(B) = 0.8 × [100 + γ×0] + 0.2 × [-1 + γ×V(B)]

This creates a recursive equation you can solve.
</details>

<details>
<summary>Hint 3 (Solution Direction)</summary>

For iteration, compute V(B) first (closer to terminal), then V(A).

Iteration 1: V(B) = 0.8×100 + 0.2×(-1 + 0.9×0) = 79.8
Iteration 2: V(B) = 0.8×100 + 0.2×(-1 + 0.9×79.8) = ...
</details>

---

#### Solution

**Task 1: Bellman Optimality Equations**

For state A (deterministic transition to B):
```
V*(A) = P(B|A,go) × [R(A,go,B) + γV*(B)]
V*(A) = 1.0 × [0 + 0.9 × V*(B)]
V*(A) = 0.9 × V*(B)
```

For state B (stochastic transition):
```
V*(B) = P(C|B,go) × [R(B,go,C) + γV*(C)] + P(B|B,go) × [R(B,go,B) + γV*(B)]
V*(B) = 0.8 × [100 + 0.9 × 0] + 0.2 × [-1 + 0.9 × V*(B)]
V*(B) = 80 + 0.2 × (-1 + 0.9 × V*(B))
V*(B) = 80 - 0.2 + 0.18 × V*(B)
V*(B) = 79.8 + 0.18 × V*(B)
```

Note: V*(C) = 0 because C is terminal (no future rewards).

**Task 2: Value Iteration**

**Initialization:** V₀(A) = 0, V₀(B) = 0

**Iteration 1:**
```
V₁(B) = 0.8 × [100 + 0.9 × V₀(C)] + 0.2 × [-1 + 0.9 × V₀(B)]
V₁(B) = 0.8 × [100 + 0] + 0.2 × [-1 + 0]
V₁(B) = 80 + (-0.2)
V₁(B) = 79.8

V₁(A) = 1.0 × [0 + 0.9 × V₀(B)]
V₁(A) = 0.9 × 0
V₁(A) = 0
```

Wait—we should use the updated V₁(B) or V₀(B)?
In standard value iteration, we use values from the *previous* iteration.

Let me redo with synchronous updates:

**Iteration 1 (using V₀ values):**
```
V₁(B) = 0.8 × [100 + 0] + 0.2 × [-1 + 0.9 × 0]
V₁(B) = 80 - 0.2 = 79.8

V₁(A) = 0 + 0.9 × 0 = 0
```

Actually, for efficiency, let's use Gauss-Seidel (asynchronous) updates where we use the latest values:

**Iteration 1:**
```
V₁(B) = 0.8 × [100] + 0.2 × [-1 + 0.9 × 0]
V₁(B) = 80 - 0.2 = 79.8

V₁(A) = 0.9 × V₁(B) = 0.9 × 79.8 = 71.82
```

**Iteration 2:**
```
V₂(B) = 0.8 × [100] + 0.2 × [-1 + 0.9 × 79.8]
V₂(B) = 80 + 0.2 × [-1 + 71.82]
V₂(B) = 80 + 0.2 × 70.82
V₂(B) = 80 + 14.164 = 94.164

V₂(A) = 0.9 × V₂(B) = 0.9 × 94.164 = 84.748
```

**Iteration 3:**
```
V₃(B) = 0.8 × [100] + 0.2 × [-1 + 0.9 × 94.164]
V₃(B) = 80 + 0.2 × [-1 + 84.748]
V₃(B) = 80 + 0.2 × 83.748
V₃(B) = 80 + 16.750 = 96.750

V₃(A) = 0.9 × V₃(B) = 0.9 × 96.750 = 87.075
```

**Summary Table:**
```
| Iteration | V(A)   | V(B)   |
|-----------|--------|--------|
| 0         | 0      | 0      |
| 1         | 71.82  | 79.8   |
| 2         | 84.75  | 94.16  |
| 3         | 87.08  | 96.75  |
| ∞ (conv.) | ~87.8  | ~97.6  |
```

**Task 3: Value after iteration 3**
V₃(A) = **87.075**

**Task 4: Why B includes both outcomes**

The value of B must account for uncertainty in the environment:
- 80% of the time, the agent successfully reaches the goal (+100)
- 20% of the time, the agent fails and must try again (-1 penalty, then continue from B)

The Bellman equation computes the **expected value**—a weighted average over all possible outcomes. This is essential because:
1. The agent cannot control which outcome occurs
2. The policy must be evaluated under the actual environment dynamics
3. Both outcomes contribute to the long-run expected reward

The self-referential term (0.9 × V(B)) captures the value of "trying again" when the transition fails.

---

#### Common Mistakes

| Mistake | Why It's Wrong | Correct Approach |
|---------|----------------|------------------|
| Forgetting probability weights | Not all outcomes equally likely | Multiply each outcome by its probability |
| Terminal state has value | Terminal means no more rewards | V(terminal) = 0 for discounted problems |
| Using wrong iteration values | Mixing iteration k and k+1 | Use consistent values (sync or async) |
| Missing the self-loop value | B can transition to itself | Include P(B|B) × [R + γV(B)] term |

---

#### Extension Challenge

Solve for the exact converged values V*(A) and V*(B) algebraically. (Hint: Solve the system of equations from Task 1.)

---

---

### Problem 3 | Skill-Builder
**Concept:** DQN Architecture Design
**Source Section:** Core Concepts 9
**Concept Map Node:** DQN (5), Experience Replay (3), Target Networks (3)
**Related Flashcard:** Card 5
**Estimated Time:** 20-25 min

#### Problem Statement

You're designing a DQN system to train an agent to play a simplified racing game with the following characteristics:

**Environment:**
- Observation: 84×84 grayscale image (current game frame)
- Actions: 5 discrete actions {accelerate, brake, left, right, no-op}
- Episode length: ~1000 steps
- Rewards: +1 for distance traveled, -10 for collision (terminal)

**Requirements:**
- The agent must learn from raw pixel input
- Training budget: 1 million environment steps
- Hardware: Single GPU with 8GB memory

**Tasks:**
1. Design the neural network architecture (CNN + fully connected layers) with specific layer dimensions
2. Specify the experience replay buffer size and justify your choice
3. Describe the target network update strategy (hard vs. soft update) with specific parameters
4. Calculate the memory requirements for your replay buffer
5. Propose an exploration schedule (ε decay) appropriate for 1M steps

---

#### Hints

<details>
<summary>Hint 1 (Conceptual)</summary>

DQN architecture typically uses:
- 3-4 convolutional layers for feature extraction
- 1-2 fully connected layers for Q-value prediction
- Output layer size = number of actions

For 84×84 input, the original DQN paper used: Conv(8×8, stride 4) → Conv(4×4, stride 2) → Conv(3×3, stride 1) → FC → FC
</details>

<details>
<summary>Hint 2 (Approach)</summary>

Memory calculation for replay buffer:
- Each transition: (s, a, r, s', done)
- State is 84×84 = 7,056 pixels
- With uint8: 7,056 bytes per state
- Need to store s and s': ~14KB per transition
</details>

<details>
<summary>Hint 3 (Solution Direction)</summary>

For 1M steps with single GPU:
- Replay buffer: 100K-500K transitions (memory constraint)
- ε decay: Start at 1.0, anneal to 0.1 over first 100K-500K steps
- Target update: Every 1K-10K steps (hard) or τ=0.001 (soft)
</details>

---

#### Solution

**Task 1: Neural Network Architecture**

```
Input: 84×84×1 grayscale image

CONVOLUTIONAL LAYERS:
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: Conv2D                                             │
│   - Filters: 32                                             │
│   - Kernel: 8×8                                             │
│   - Stride: 4                                               │
│   - Activation: ReLU                                        │
│   - Output: 20×20×32                                        │
├─────────────────────────────────────────────────────────────┤
│ Layer 2: Conv2D                                             │
│   - Filters: 64                                             │
│   - Kernel: 4×4                                             │
│   - Stride: 2                                               │
│   - Activation: ReLU                                        │
│   - Output: 9×9×64                                          │
├─────────────────────────────────────────────────────────────┤
│ Layer 3: Conv2D                                             │
│   - Filters: 64                                             │
│   - Kernel: 3×3                                             │
│   - Stride: 1                                               │
│   - Activation: ReLU                                        │
│   - Output: 7×7×64 = 3,136 features                        │
└─────────────────────────────────────────────────────────────┘

FULLY CONNECTED LAYERS:
┌─────────────────────────────────────────────────────────────┐
│ Flatten: 3,136 → 3,136                                      │
├─────────────────────────────────────────────────────────────┤
│ Layer 4: Dense(512)                                         │
│   - Activation: ReLU                                        │
│   - Output: 512                                             │
├─────────────────────────────────────────────────────────────┤
│ Layer 5: Dense(5)                                           │
│   - Activation: None (linear Q-values)                      │
│   - Output: 5 (one Q-value per action)                      │
└─────────────────────────────────────────────────────────────┘

Total Parameters: ~1.7M
- Conv1: 8×8×1×32 + 32 = 2,080
- Conv2: 4×4×32×64 + 64 = 32,832
- Conv3: 3×3×64×64 + 64 = 36,928
- FC1: 3,136×512 + 512 = 1,606,144
- FC2: 512×5 + 5 = 2,565
```

**Task 2: Experience Replay Buffer Size**

```
Buffer Size: 200,000 transitions

Justification:
1. Memory constraint: ~2.8GB for buffer (see calculation below)
2. Sufficient diversity: 200K covers ~200 episodes (1K steps each)
3. Not too large: Old experiences become stale
4. Industry standard: Original DQN used 1M, but we're memory-limited
5. 200K = 20% of training budget → recent experience well-represented
```

**Task 3: Target Network Update Strategy**

```
Strategy: Hard update every 10,000 steps

Parameters:
- Update frequency: Every 10,000 environment steps
- At update: θ_target ← θ_online (full copy)

Justification:
1. 1M steps ÷ 10K = 100 target updates during training
2. Frequent enough for value propagation
3. Not too frequent to destabilize learning
4. Simple implementation compared to soft updates

Alternative (soft update):
- Update every step: θ_target ← 0.001×θ_online + 0.999×θ_target
- More stable but slower value propagation
- Better for continuous control; hard update fine for discrete actions
```

**Task 4: Memory Requirements Calculation**

```
Per Transition Storage:
┌─────────────────────────────────────────────────────────────┐
│ Component        │ Type    │ Size                           │
├──────────────────┼─────────┼────────────────────────────────┤
│ State (s)        │ uint8   │ 84×84×1 = 7,056 bytes          │
│ Action (a)       │ int32   │ 4 bytes                        │
│ Reward (r)       │ float32 │ 4 bytes                        │
│ Next State (s')  │ uint8   │ 84×84×1 = 7,056 bytes          │
│ Done flag        │ bool    │ 1 byte                         │
├──────────────────┼─────────┼────────────────────────────────┤
│ Total per trans. │         │ 14,121 bytes ≈ 14 KB           │
└─────────────────────────────────────────────────────────────┘

Total Buffer Memory:
- 200,000 transitions × 14,121 bytes = 2.82 GB
- Within 8GB GPU memory budget (leaves room for networks)

Optimization options:
- Store only s, compute s' = s[t+1] from buffer → saves 50%
- Compress frames with difference encoding
- Use uint8 for rewards if discrete
```

**Task 5: Exploration Schedule**

```
ε-Greedy Annealing Schedule:

┌──────────────────────────────────────────────────────────────┐
│ ε                                                            │
│ 1.0 ─────┐                                                   │
│          │                                                   │
│          │                                                   │
│ 0.5      └─────────┐                                        │
│                    │                                         │
│                    └─────────────────────────────────────── │
│ 0.1                          Final ε                        │
│──────────────────────────────────────────────────────────── │
│ 0      100K    250K    500K    750K    1M    Steps          │
└──────────────────────────────────────────────────────────────┘

Schedule:
- Start: ε = 1.0 (100% random)
- Linear decay over first 250,000 steps
- Final: ε = 0.1 (10% random exploration)
- After 250K: Fixed at 0.1

Formula:
ε(t) = max(0.1, 1.0 - (t / 250,000) × 0.9)

Justification:
1. First 25% of training: Aggressive exploration to discover strategies
2. 10% final exploration: Maintain some exploration for robustness
3. Not too slow: Need to exploit learned behavior
4. Not too fast: Risk missing good strategies early
```

**Complete DQN Configuration Summary:**

```python
config = {
    # Network
    'input_shape': (84, 84, 1),
    'n_actions': 5,
    'conv_layers': [(32, 8, 4), (64, 4, 2), (64, 3, 1)],
    'fc_layers': [512],

    # Replay
    'buffer_size': 200_000,
    'batch_size': 32,
    'learning_starts': 10_000,  # Fill buffer before learning

    # Target network
    'target_update_freq': 10_000,

    # Exploration
    'epsilon_start': 1.0,
    'epsilon_final': 0.1,
    'epsilon_decay_steps': 250_000,

    # Optimization
    'learning_rate': 2.5e-4,
    'gamma': 0.99,
    'optimizer': 'Adam',

    # Training
    'total_steps': 1_000_000,
    'train_freq': 4,  # Update every 4 env steps
}
```

---

#### Common Mistakes

| Mistake | Why It's Wrong | Correct Approach |
|---------|----------------|------------------|
| Output size ≠ n_actions | Must have one Q-value per action | Output layer = Dense(n_actions) |
| Buffer too small | Not enough diversity | Minimum ~50K for stable learning |
| Never updating target | Values never propagate | Update every 1K-10K steps |
| ε decays to 0 | Stops exploring entirely | Keep small ε (0.01-0.1) forever |
| Float32 for states | Wastes memory | Use uint8, convert in training loop |

---

#### Extension Challenge

Modify the architecture to use frame stacking (4 frames as channels) for temporal information. How does this affect memory requirements and network architecture?

---

---

### Problem 4 | Challenge
**Concept:** Actor-Critic Implementation
**Source Section:** Core Concepts 7, 8
**Concept Map Node:** Actor-Critic (6), Policy Gradient (5)
**Related Flashcard:** Card 3, Card 4
**Estimated Time:** 35-45 minutes

#### Problem Statement

You're implementing an Advantage Actor-Critic (A2C) algorithm for a continuous control task: balancing a pole on a cart (CartPole).

**Environment:**
- State: 4D vector [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
- Actions: Discrete {push_left, push_right}
- Reward: +1 for each timestep the pole is balanced
- Episode terminates when pole angle > 15° or cart moves too far
- Goal: Maximize episode length (up to 500 steps)

**Tasks:**
1. Design the actor and critic network architectures
2. Write the advantage estimation formula using TD(0) error
3. Implement the loss functions for both actor and critic in pseudocode
4. Explain why we subtract the baseline (value) from returns in the actor loss
5. Propose a modification to handle the entropy bonus for exploration

---

#### Hints

<details>
<summary>Hint 1 (Conceptual)</summary>

Actor-Critic components:
- Actor: π_θ(a|s) - outputs action probabilities (or distribution parameters)
- Critic: V_φ(s) - outputs scalar state value

Both can share lower layers for efficiency.
</details>

<details>
<summary>Hint 2 (Approach)</summary>

Advantage estimation with TD(0):
A(s,a) = r + γV(s') - V(s)

This replaces the high-variance Monte Carlo return G with a bootstrapped estimate.
</details>

<details>
<summary>Hint 3 (Solution Direction)</summary>

Actor loss: -log π(a|s) × A (maximize probability of good actions)
Critic loss: (V(s) - (r + γV(s')))² (minimize TD error)

Entropy bonus: -β × Σ π(a|s) log π(a|s) (encourage exploration)
</details>

---

#### Solution

**Task 1: Network Architectures**

```
SHARED ARCHITECTURE (Feature Extraction):
┌─────────────────────────────────────────────────────────────┐
│ Input: State s ∈ R⁴                                         │
│        [cart_pos, cart_vel, pole_angle, pole_angvel]        │
├─────────────────────────────────────────────────────────────┤
│ Hidden Layer 1: Dense(64)                                   │
│   - Activation: ReLU                                        │
├─────────────────────────────────────────────────────────────┤
│ Hidden Layer 2: Dense(64)                                   │
│   - Activation: ReLU                                        │
│   - Output: 64-dimensional feature vector                   │
└─────────────────────────────────────────────────────────────┘
                         │
           ┌─────────────┴─────────────┐
           │                           │
           ▼                           ▼
┌───────────────────────┐   ┌───────────────────────┐
│       ACTOR           │   │       CRITIC          │
│  Policy Head π_θ(a|s) │   │  Value Head V_φ(s)    │
├───────────────────────┤   ├───────────────────────┤
│ Dense(2)              │   │ Dense(1)              │
│ Activation: Softmax   │   │ Activation: None      │
│                       │   │                       │
│ Output: [P(left),     │   │ Output: V(s) ∈ R      │
│          P(right)]    │   │ (state value)         │
└───────────────────────┘   └───────────────────────┘

Parameter Sharing Benefits:
- Fewer total parameters
- Shared features for action and value
- Faster training

Parameter Counts:
- Shared: 4×64 + 64×64 = 256 + 4,096 = 4,352
- Actor head: 64×2 = 128
- Critic head: 64×1 = 64
- Total: ~4,544 parameters
```

**Task 2: Advantage Estimation with TD(0)**

```
Advantage Formula:

A(sₜ, aₜ) = δₜ = rₜ + γV(sₜ₊₁) - V(sₜ)
                 └────┬────┘   └──┬──┘
              TD Target     Current estimate

Where:
- rₜ = immediate reward at step t
- γ = discount factor (e.g., 0.99)
- V(sₜ₊₁) = critic's estimate of next state value
- V(sₜ) = critic's estimate of current state value
- δₜ = TD error = advantage estimate

For terminal states:
A(sₜ, aₜ) = rₜ + γ×0 - V(sₜ) = rₜ - V(sₜ)
                  │
         (no future value)

Interpretation:
- A > 0: Action was better than expected → increase probability
- A < 0: Action was worse than expected → decrease probability
- A ≈ 0: Action matched expectation → small update
```

**Task 3: Loss Functions (Pseudocode)**

```python
def compute_losses(states, actions, rewards, next_states, dones, gamma=0.99):
    """
    Compute actor and critic losses for A2C.

    Args:
        states: Batch of states [B, 4]
        actions: Batch of actions taken [B]
        rewards: Batch of rewards [B]
        next_states: Batch of next states [B, 4]
        dones: Batch of terminal flags [B]
    """

    # ============ CRITIC LOSS ============
    # Current value estimates
    values = critic(states)                    # [B, 1] → [B]

    # TD targets (no gradient through next state value)
    with no_gradient():
        next_values = critic(next_states)      # [B, 1] → [B]
        td_targets = rewards + gamma * next_values * (1 - dones)

    # Critic loss: MSE between predicted and TD target
    critic_loss = mean((values - td_targets) ** 2)


    # ============ ADVANTAGE ============
    # TD error as advantage estimate (stop gradient through values)
    with no_gradient():
        advantages = td_targets - values       # [B]
        # Optional: normalize advantages for stability
        advantages = (advantages - mean(advantages)) / (std(advantages) + 1e-8)


    # ============ ACTOR LOSS ============
    # Action probabilities from policy
    action_probs = actor(states)               # [B, 2] softmax output

    # Log probability of taken actions
    log_probs = log(action_probs[range(B), actions])  # [B]

    # Policy gradient loss (negative because we maximize)
    actor_loss = -mean(log_probs * advantages)


    # ============ ENTROPY BONUS ============
    # Encourage exploration by maximizing entropy
    entropy = -sum(action_probs * log(action_probs + 1e-8), dim=1)  # [B]
    entropy_bonus = mean(entropy)

    # Final actor loss with entropy regularization
    actor_loss = actor_loss - entropy_coef * entropy_bonus


    # ============ COMBINED LOSS ============
    # Can train jointly or separately
    total_loss = actor_loss + value_coef * critic_loss

    return total_loss, actor_loss, critic_loss, entropy_bonus
```

**Task 4: Why Subtract Baseline (Value)?**

```
Without baseline (REINFORCE):
∇J = E[∇log π(a|s) × Gₜ]

Problem: Gₜ (return) has high variance
- Same action might give G=200 in one episode, G=50 in another
- Gradients are noisy → slow, unstable learning

With baseline (Actor-Critic):
∇J = E[∇log π(a|s) × (Gₜ - V(s))]
                       └───┬───┘
                      Advantage

Why this works mathematically:
- E[∇log π(a|s) × V(s)] = 0 for any baseline b(s)
- Subtracting V(s) doesn't change expected gradient (unbiased)
- But reduces variance because (G - V) fluctuates less than G

Intuition:
┌─────────────────────────────────────────────────────────────┐
│ Without baseline:                                           │
│   "Got return 100 → increase action probability"            │
│   But is 100 good or bad? Depends on the state!             │
│                                                             │
│ With baseline:                                              │
│   "Got return 100, expected 90 → action was +10 better"     │
│   "Got return 100, expected 110 → action was -10 worse"     │
│                                                             │
│ Baseline provides context: "better or worse than average?"  │
└─────────────────────────────────────────────────────────────┘

In A2C:
- V(s) estimates "how good is this state on average?"
- A(s,a) = Q(s,a) - V(s) estimates "how much better is this action?"
- Only actions that are better than baseline get increased
```

**Task 5: Entropy Bonus for Exploration**

```
Entropy Definition:
H(π(·|s)) = -Σₐ π(a|s) log π(a|s)

- High entropy: Uniform distribution → maximum randomness
- Low entropy: Peaked distribution → deterministic
- For 2 actions: max entropy = log(2) ≈ 0.693

Modified Loss:
L_actor = -E[log π(a|s) × A] - β × H(π(·|s))
                               └──────┬──────┘
                            Entropy bonus (negative sign
                            because we minimize loss but
                            want to maximize entropy)

Implementation:
```python
# Entropy for discrete actions
def compute_entropy(action_probs):
    """
    action_probs: [B, n_actions] softmax probabilities
    returns: [B] entropy for each state
    """
    # Add small epsilon to prevent log(0)
    log_probs = torch.log(action_probs + 1e-8)
    entropy = -torch.sum(action_probs * log_probs, dim=1)
    return entropy

# In loss computation
entropy = compute_entropy(action_probs)      # [B]
entropy_loss = -entropy_coef * entropy.mean()  # Scalar

# Typical values for entropy coefficient
entropy_coef = 0.01  # Small regularization
# Can anneal: start high (0.1), decay to low (0.001)
```

Benefits of Entropy Bonus:
1. Prevents premature convergence to deterministic policy
2. Encourages exploration throughout training
3. Makes policy more robust to environment variations
4. Helps escape local optima
```

---

#### Common Mistakes

| Mistake | Why It's Wrong | Correct Approach |
|---------|----------------|------------------|
| Backprop through TD target | Destabilizes critic training | Stop gradient on V(s') in TD target |
| Advantage with gradient | Couples actor and critic updates | Detach advantage computation |
| Forgetting done mask | Values leak across episodes | Multiply next_value by (1-done) |
| Wrong sign on actor loss | Gradient descent but want to maximize | Use negative log prob × advantage |
| Entropy with wrong sign | Minimizing entropy reduces exploration | Subtract entropy from loss (maximize it) |

---

#### Extension Challenge

Extend the A2C implementation to support Generalized Advantage Estimation (GAE) with λ parameter. Show the formula and explain how λ trades off bias and variance.

---

---

### Problem 5 | Debug/Fix
**Concept:** Exploration Failures
**Source Section:** Core Concepts 10
**Concept Map Node:** Exploration (4), Exploitation (linked)
**Related Flashcard:** Card 2, Card 5
**Estimated Time:** 25-30 minutes

#### Problem Statement

A colleague is training a DQN agent to navigate a maze with the following characteristics:

**Environment:**
- 20×20 grid maze with sparse rewards
- Reward: +100 for reaching goal (terminal)
- Reward: -1 for each step (encourages efficiency)
- Reward: 0 otherwise
- Starting position: Random
- Goal position: Fixed in corner

**Observed Issues:**
```
After 500,000 training steps:

Issue 1: The agent learns to spin in circles near the start
- High Q-values for action "turn left" in many states
- Never reaches the goal during training

Issue 2: Replay buffer analysis shows:
- 98% of transitions have reward = -1
- 1.9% have reward = -1 (wall collision penalty)
- 0.1% have reward = +100 (goal reached)

Issue 3: Training curves:
- Average episode reward: -200 (agent dies after 200 steps timeout)
- No improvement over 500K steps
- Q-values are uniformly high for all actions (~50)

Issue 4: Configuration used:
- ε-greedy with ε = 0.1 (fixed throughout training)
- Replay buffer size: 10,000
- Learning rate: 0.001
- Discount γ = 0.99
```

**Tasks:**
1. Diagnose the root causes of each issue
2. Explain why the current exploration strategy fails
3. Propose 3 specific fixes with justifications
4. Suggest additional debugging steps to verify your hypotheses
5. Recommend an exploration strategy better suited to sparse rewards

---

#### Hints

<details>
<summary>Hint 1 (Conceptual)</summary>

Sparse reward problems suffer from:
- Random exploration rarely reaches the goal
- No learning signal until goal is found
- Need structured exploration or reward shaping

With ε = 0.1, random walk probability of reaching a corner in 20×20 grid is extremely low.
</details>

<details>
<summary>Hint 2 (Approach)</summary>

Issues to consider:
- ε = 0.1 fixed from start means only 10% exploration from the beginning
- Small replay buffer (10K) may lose the rare goal transitions
- High discount (0.99) needs many steps to propagate reward signal

The "spinning" behavior suggests the agent found a local optimum that avoids wall penalties.
</details>

<details>
<summary>Hint 3 (Solution Direction)</summary>

Fixes to consider:
1. ε annealing starting from 1.0
2. Larger replay buffer with prioritized replay
3. Intrinsic motivation / curiosity-driven exploration
4. Reward shaping (distance to goal)
5. Curriculum learning (start near goal)
</details>

---

#### Solution

**Task 1: Root Cause Diagnosis**

| Issue | Symptom | Root Cause |
|-------|---------|------------|
| **Issue 1: Spinning behavior** | Agent repeats same action | Insufficient exploration; found local optimum that avoids -1 wall penalty |
| **Issue 2: Rare goal transitions** | 0.1% goal rate | Random exploration almost never reaches goal in 20×20 maze; goal too far |
| **Issue 3: No improvement** | Flat training curve | No learning signal; Q-values based on noise; overestimation |
| **Issue 4: Fixed ε = 0.1** | 90% exploitation from start | Never explores enough to find goal even once |

**Detailed Analysis:**

```
EXPLORATION FAILURE CHAIN:

1. Fixed ε = 0.1 from start
   ↓
2. Agent exploits random initial Q-values (all ~0)
   ↓
3. Random walk with 10% exploration in 20×20 maze
   → Probability of reaching corner goal ≈ 0.001%
   ↓
4. No positive reward signal ever received
   ↓
5. Q-values trained only on r = -1 transitions
   ↓
6. Agent learns to avoid walls (local optimum)
   ↓
7. "Spinning" = avoiding walls while never progressing to goal
```

**Task 2: Why Current Exploration Fails**

```
MATHEMATICAL ANALYSIS:

Random walk in 20×20 grid to reach corner:
- Expected steps: O(n²) for n×n grid ≈ 400 steps
- But episode timeout at 200 steps!
- With ε = 0.1, only 10% of steps are random
- Effectively: 20 random steps per episode → nowhere near goal

REPLAY BUFFER PROBLEM:
- Buffer size: 10,000 transitions
- Agent generates ~200 transitions/episode
- Buffer refreshes every 50 episodes
- If goal reached once in 1000 episodes → that transition is overwritten!

Q-VALUE OVERESTIMATION:
- All Q-values around 50 for all actions
- With no positive signal, this is pure noise + overestimation
- Agent can't distinguish good vs. bad states
```

**Task 3: Three Specific Fixes**

**Fix 1: Aggressive ε Annealing with Warmup**
```python
# Current (broken):
epsilon = 0.1  # Fixed

# Fixed:
def get_epsilon(step, warmup_steps=50000, decay_steps=400000):
    if step < warmup_steps:
        return 1.0  # 100% random exploration initially
    else:
        progress = (step - warmup_steps) / decay_steps
        return max(0.05, 1.0 - progress)

# Results in:
# Steps 0-50K: ε = 1.0 (pure exploration)
# Steps 50K-450K: ε decays from 1.0 to 0.05
# Steps 450K+: ε = 0.05

Justification:
- Warmup period ensures many goal discoveries before exploitation
- Slow decay maintains exploration throughout training
- Final ε = 0.05 still allows occasional exploration
```

**Fix 2: Larger Prioritized Replay Buffer**
```python
# Current (broken):
buffer_size = 10000  # Too small, uniform sampling

# Fixed:
buffer_size = 500000  # Much larger

# Plus: Prioritized Experience Replay (PER)
class PrioritizedReplayBuffer:
    def sample(self, batch_size):
        # Sample proportional to TD error
        # High TD error = surprising = important
        priorities = self.td_errors ** alpha  # alpha = 0.6
        probs = priorities / sum(priorities)
        indices = random.choice(len(buffer), batch_size, p=probs)
        return self.buffer[indices]

Justification:
- Large buffer retains rare goal transitions
- Prioritized sampling focuses on surprising/important transitions
- Goal transitions (high TD error) sampled more frequently
```

**Fix 3: Intrinsic Motivation / Curiosity**
```python
# Add curiosity-driven exploration bonus
class ICM:  # Intrinsic Curiosity Module
    def __init__(self):
        self.forward_model = ForwardModel()  # Predicts next state
        self.inverse_model = InverseModel()  # Predicts action from states

    def intrinsic_reward(self, s, a, s_next):
        # Reward = prediction error
        s_next_pred = self.forward_model(s, a)
        error = ||s_next_pred - s_next||²
        return beta * error  # beta = 0.01

# Modified reward:
total_reward = extrinsic_reward + intrinsic_reward

Justification:
- Agent rewarded for visiting unpredictable states
- Encourages systematic exploration of unseen areas
- Works without any extrinsic reward initially
```

**Additional Fixes to Consider:**
```
4. Reward Shaping:
   reward += -0.1 * distance_to_goal
   (Gives gradient toward goal without changing optimal policy)

5. Curriculum Learning:
   Start episodes near goal, gradually increase distance
   (Ensures learning signal from start)

6. Double DQN:
   Reduce overestimation that makes all actions look equally good
```

**Task 4: Debugging Steps to Verify Hypotheses**

```python
# 1. Verify exploration coverage
def track_state_visits(agent, episodes=1000):
    visit_count = np.zeros((20, 20))
    for ep in range(episodes):
        s = env.reset()
        while not done:
            visit_count[s[0], s[1]] += 1
            a = agent.act(s)
            s, r, done, _ = env.step(a)
    return visit_count

# Expect: With ε=0.1, most squares never visited
# Should see: Heavy concentration in starting area

# 2. Check goal discovery frequency
def count_goal_reaches(agent, episodes=1000):
    goals = 0
    for ep in range(episodes):
        # Run episode
        if reached_goal:
            goals += 1
    return goals / episodes

# Current: ~0.1% (essentially never)
# After fix: Should see increase to 5-20%

# 3. Analyze Q-value distribution
def analyze_q_values(agent):
    q_values = []
    for s in all_states:
        q_values.append(agent.get_q_values(s))
    print(f"Q mean: {np.mean(q_values)}")
    print(f"Q std: {np.std(q_values)}")
    print(f"Q max - min: {np.max(q_values) - np.min(q_values)}")

# Problem: All ~50, low variance → no discrimination
# After fix: Should see variance, higher near goal

# 4. Visualize policy
def visualize_policy(agent):
    policy_grid = np.zeros((20, 20, 4))  # 4 directions
    for x in range(20):
        for y in range(20):
            s = [x, y]
            q = agent.get_q_values(s)
            policy_grid[x, y, np.argmax(q)] = 1
    plot_arrows(policy_grid)

# Should show: Clear direction toward goal after training
```

**Task 5: Better Exploration Strategy for Sparse Rewards**

```
RECOMMENDED: Count-Based Exploration + Prioritized Replay

┌─────────────────────────────────────────────────────────────┐
│                Count-Based Exploration                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Bonus = β / √(N(s) + 1)                                   │
│                                                             │
│   Where N(s) = number of times state s has been visited     │
│                                                             │
│   Effect: Higher bonus for less-visited states              │
│           Naturally decays as states become familiar        │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Implementation:
```python
class CountBasedExploration:
    def __init__(self, beta=0.1):
        self.visit_counts = defaultdict(int)
        self.beta = beta

    def get_bonus(self, state):
        state_hash = tuple(state)
        count = self.visit_counts[state_hash]
        return self.beta / np.sqrt(count + 1)

    def update(self, state):
        self.visit_counts[tuple(state)] += 1

# Modified training
def train_step():
    # ... get transition (s, a, r, s', done)

    # Add exploration bonus
    intrinsic = exploration.get_bonus(s_prime)
    augmented_reward = r + intrinsic

    # Update visit count
    exploration.update(s_prime)

    # Store with augmented reward
    buffer.add(s, a, augmented_reward, s_prime, done)
```

Why this works for sparse rewards:
1. Provides continuous learning signal even without extrinsic reward
2. Naturally prioritizes unexplored regions
3. Self-annealing: bonus decreases as exploration completes
4. Compatible with any base algorithm (DQN, A2C, etc.)
5. Guaranteed to eventually visit all reachable states
```

---

#### Common Mistakes

| Mistake | Why It's Wrong | Correct Approach |
|---------|----------------|------------------|
| Only increasing ε | Doesn't address sparse reward signal | Combine with intrinsic motivation |
| Reward shaping that changes optimal policy | Can lead to suboptimal behavior | Use potential-based shaping |
| Very small final ε | May not recover from early mistakes | Keep ε ≥ 0.01 for robustness |
| Ignoring replay buffer | Rare experiences get overwritten | Use large buffer + prioritization |

---

#### Extension Challenge

Design an experiment to compare ε-greedy, Boltzmann exploration, and count-based exploration on this sparse-reward maze. What metrics would you track, and what results would you expect?

---

---

## Skills Integration Summary

This practice problem set integrates with the full skill chain:

```
Study Notes (10 Concepts)
        ↓
Concept Map (26 concepts, 42 relationships)
        ↓
Flashcards (5 cards: 2E/2M/1H)
        ↓
Practice Problems ← YOU ARE HERE
        ↓
Quiz (5 questions: 2MC/2SA/1E)
```

| Problem | Concepts Practiced | Prepares For |
|---------|-------------------|--------------|
| P1 | Q-Learning updates | Quiz Q1, Q2 |
| P2 | Bellman equations, Value iteration | Quiz Q1 |
| P3 | DQN architecture | Quiz Q3 |
| P4 | Actor-Critic, Policy gradient | Quiz Q4, Q5 |
| P5 | Exploration strategies | Quiz Q4 |
