# Assessment Quiz: Reinforcement Learning

**Source Material:** notes/reinforcement-learning/flashcards/reinforcement-learning-flashcards.md
**Practice Problems:** notes/reinforcement-learning/practice/reinforcement-learning-practice-problems.md
**Concept Map:** notes/reinforcement-learning/concept-maps/reinforcement-learning-concept-map.md
**Original Study Notes:** notes/reinforcement-learning/reinforcement-learning-study-notes.md
**Date Generated:** 2026-01-07
**Total Questions:** 5
**Estimated Completion Time:** 30-40 minutes
**Distribution:** 2 Multiple Choice | 2 Short Answer | 1 Essay

---

## Instructions

- **Multiple Choice:** Select the single best answer
- **Short Answer:** Respond in 2-4 sentences
- **Essay:** Provide a comprehensive response (1-2 paragraphs)

---

## Questions

### Question 1 | Multiple Choice
**Cognitive Level:** Remember/Understand
**Concept Tested:** Q-Learning vs. SARSA
**Source Section:** Core Concepts 5, 6
**Concept Map Node:** Q-Learning (7 connections), SARSA (3 connections)
**Related Flashcard:** Card 2
**Related Practice Problem:** P1

What is the fundamental difference between Q-learning and SARSA, and how does this affect their behavior in environments where exploration might lead to dangerous states?

A) Q-learning learns faster because it samples more transitions; SARSA learns the safe policy because it uses bootstrapping

B) Q-learning is on-policy and learns the policy it follows; SARSA is off-policy and learns the optimal policy regardless of actions taken

C) Q-learning uses max_{a'} Q(s',a') in its update (off-policy, learns optimal Q*); SARSA uses Q(s',a') where a' is the actual next action (on-policy, learns Q for the policy being followed)

D) Q-learning requires a model of the environment; SARSA is model-free and only needs experience

---

### Question 2 | Multiple Choice
**Cognitive Level:** Remember/Understand
**Concept Tested:** Policy Gradient Variance
**Source Section:** Core Concepts 7
**Concept Map Node:** Policy Gradient (5), Value Functions (9)
**Related Flashcard:** Card 3
**Related Practice Problem:** P4

Why do policy gradient methods typically have high variance, and how does the advantage function A(s,a) = Q(s,a) - V(s) help address this problem?

A) Variance comes from function approximation errors; the advantage function uses exact values which eliminates approximation

B) Variance comes from stochastic transitions; the advantage normalizes by the expected value, making updates independent of environment randomness

C) Variance comes from using Monte Carlo returns G_t which vary across trajectories; subtracting the baseline V(s) centers updates around zero without changing the expected gradient

D) Variance comes from action discretization; the advantage function works in continuous spaces which have lower variance

---

### Question 3 | Short Answer
**Cognitive Level:** Apply/Analyze
**Concept Tested:** DQN Stability Techniques
**Source Section:** Core Concepts 9
**Concept Map Node:** DQN (5), Experience Replay (3), Target Networks (3)
**Related Flashcard:** Card 5
**Related Practice Problem:** P3
**Expected Response Length:** 3-4 sentences

Deep Q-learning with neural networks is notoriously unstable compared to tabular Q-learning. Explain the two main sources of this instability (sample correlation and non-stationary targets) and describe how experience replay and target networks address each problem respectively.

---

### Question 4 | Short Answer
**Cognitive Level:** Apply/Analyze
**Concept Tested:** Exploration-Exploitation Tradeoff
**Source Section:** Core Concepts 10
**Concept Map Node:** Exploration (4), Exploitation (linked)
**Related Flashcard:** Card 5
**Related Practice Problem:** P5
**Expected Response Length:** 3-4 sentences

A robotics company is training an RL agent to control a robotic arm for assembly tasks. During training, exploration could cause the arm to collide with objects and damage equipment. Analyze the exploration-exploitation tradeoff in this safety-critical context and propose two strategies that would allow the agent to learn effectively while minimizing dangerous exploration.

---

### Question 5 | Essay
**Cognitive Level:** Evaluate/Synthesize
**Concepts Tested:** Complete RL System Design
**Source Sections:** All Core Concepts, Practical Applications
**Concept Map:** Full pathway traversal
**Related Flashcard:** Cards 2, 4, 5
**Related Practice Problem:** P3, P4, P5
**Expected Response Length:** 1-2 paragraphs

You are the lead ML engineer designing an RL system to optimize inventory management for a retail chain. The system must: (1) decide daily reorder quantities for 1000 products across 50 stores, (2) minimize stockouts while avoiding excess inventory, (3) adapt to seasonal demand patterns, and (4) operate within a budget constraint for total inventory value.

Design a complete solution addressing: (a) state and action space formulation—how would you represent the problem as an MDP?; (b) algorithm selection—would you use value-based (DQN), policy-based (PPO), or actor-critic methods, and why?; (c) reward design—what reward function would capture the business objectives, and what challenges might arise?; (d) handling the large action space (1000 products × quantity decisions); and (e) how you would safely deploy and update the system in production.

**Evaluation Criteria:**
- [ ] Formulates MDP with appropriate state/action representation
- [ ] Justifies algorithm choice with domain-specific reasoning
- [ ] Designs reward function addressing multiple business objectives
- [ ] Addresses large action space challenge
- [ ] Proposes safe deployment strategy

---

## Answer Key

### Question 1 | Multiple Choice
**Correct Answer:** C

**Explanation:**
The key difference lies in the update target:

| Algorithm | Update Target | Policy Learned |
|-----------|--------------|----------------|
| **Q-learning** | r + γ max_{a'} Q(s',a') | Optimal Q* (greedy) |
| **SARSA** | r + γ Q(s',a') where a' is actual next action | Q for current behavior policy |

Q-learning is **off-policy**: it learns about the optimal policy (always taking the best action) regardless of which action the agent actually took for exploration. SARSA is **on-policy**: it learns the value of the policy actually being followed, including exploratory actions.

**In dangerous environments:**
- Q-learning learns the optimal policy assuming it will always be greedy—it may underestimate danger if exploration leads to bad states
- SARSA learns the value of the policy including exploration—if ε-greedy exploration sometimes leads to dangerous states, SARSA will learn to avoid nearby states even with optimal actions

**Why Other Options Are Incorrect:**
- A) Inverts the on/off-policy distinction; learning speed depends on many factors
- B) Swaps the definitions—Q-learning is off-policy, SARSA is on-policy
- D) Both are model-free; neither requires environment model

**Understanding Gap Indicator:**
If answered incorrectly, review the Q-learning and SARSA update rules in Card 2 and Practice Problem 1.

---

### Question 2 | Multiple Choice
**Correct Answer:** C

**Explanation:**
Policy gradient methods estimate gradients from sampled trajectories:
```
∇J ≈ (1/N) Σ ∇log π(a|s) × G_t
```

**Source of Variance:**
- The return G_t = Σ γ^k r_{t+k} varies enormously between trajectories
- Same action in same state might yield G=100 in one episode, G=50 in another
- This variance propagates to gradient estimates, making learning slow and unstable

**How Advantage Helps:**
```
∇J = E[∇log π(a|s) × (G_t - V(s))]
                      └────┬────┘
                      Advantage
```

- Subtracting V(s) doesn't change the expected gradient (mathematically proven)
- But (G - V) has lower variance than G alone
- Advantage centers updates: positive for better-than-average actions, negative for worse
- Actions better than baseline get increased probability; worse get decreased

**Why Other Options Are Incorrect:**
- A) Advantage uses approximations too; the benefit is variance reduction, not exactness
- B) Transitions affect both G and V equally; baseline doesn't cancel environment randomness
- D) Continuous vs. discrete has nothing to do with policy gradient variance

**Understanding Gap Indicator:**
If answered incorrectly, review the baseline derivation in Card 3 and the variance reduction explanation in Practice Problem 4.

---

### Question 3 | Short Answer
**Model Answer:**

Deep Q-learning faces two main instabilities absent in tabular methods. First, **sample correlation**: sequential experience (s_t, s_{t+1}, s_{t+2}...) creates correlated training samples, violating the i.i.d. assumption of SGD and causing the network to oscillate or diverge. **Experience replay** addresses this by storing transitions in a buffer and sampling random mini-batches, breaking temporal correlations and allowing each transition to be reused multiple times.

Second, **non-stationary targets**: the TD target r + γ max Q_θ(s',a') depends on the same network being trained, creating a moving target problem—as Q changes, so does the target. **Target networks** solve this by maintaining a separate, slowly-updated copy of Q for computing targets; the target network is updated only periodically (every N steps) or with soft updates (θ_target ← τθ + (1-τ)θ_target), providing stable targets during learning.

**Key Components Required:**
- [ ] Identifies sample correlation as instability source
- [ ] Explains how experience replay breaks correlation
- [ ] Identifies non-stationary targets as instability source
- [ ] Explains how target networks provide stable targets

**Partial Credit Guidance:**
- Full credit: Both problems identified with clear mechanism explanation for each solution
- Partial credit: Identifies problems but vague on how solutions address them
- No credit: Confuses the two solutions or misidentifies the problems

**Understanding Gap Indicator:**
If answered poorly, review the DQN stability section in Card 5 and Practice Problem 3.

---

### Question 4 | Short Answer
**Model Answer:**

In safety-critical robotics, the exploration-exploitation tradeoff presents a fundamental tension: the agent needs exploration to discover effective control strategies, but random exploration risks damaging equipment. Two strategies to address this:

First, **sim-to-real transfer with domain randomization**: train the agent extensively in a physics simulator where exploration is safe, randomizing physical parameters (friction, mass, sensor noise) to ensure robustness. Transfer the learned policy to the real robot with reduced or no exploration (ε ≈ 0). This provides safe exploration in simulation while limiting real-world risk.

Second, **constrained RL with safety layers**: define hard constraints (e.g., maximum force, forbidden workspace regions) and implement a safety filter that overrides unsafe actions before execution. Techniques like Control Barrier Functions or learned safety critics can predict dangerous actions and substitute safe alternatives, allowing the agent to explore within a safe operating envelope.

**Key Components Required:**
- [ ] Acknowledges the exploration-danger tradeoff
- [ ] Proposes at least two viable strategies
- [ ] Explains how each strategy reduces risk while enabling learning

**Partial Credit Guidance:**
- Full credit: Clear tradeoff analysis + two concrete strategies with rationale
- Partial credit: Mentions strategies but lacks detail on how they reduce risk
- No credit: Suggests eliminating exploration entirely (no learning) or ignores safety

**Understanding Gap Indicator:**
If answered poorly, review exploration concepts in the Critical Analysis section and Practice Problem 5.

---

### Question 5 | Essay
**Model Answer:**

**MDP Formulation:**
The state space would include: current inventory levels for each product-store combination (1000 × 50 = 50,000 dimensions), day of week/season encodings for demand patterns, recent sales velocity, pending orders in transit, and current budget utilization. The action space represents reorder quantities—for tractability, I would discretize quantities into bins (0, small, medium, large reorder) giving 4^1000 possible joint actions, which is intractable. To address this, I would factorize the action space by treating each product independently or using action embeddings, or employ a hierarchical approach where a high-level policy allocates budget across categories and low-level policies handle individual products.

**Algorithm Selection:**
I would use **PPO (Proximal Policy Optimization)** for several reasons: (1) it handles the continuous/large discrete action space more naturally than value-based methods through policy parameterization; (2) PPO's clipping mechanism provides stable training crucial for business applications where policy collapse would be costly; (3) the actor-critic architecture allows credit assignment across the multi-step inventory cycle (order today, receive in 3 days, sell over weeks); (4) PPO works well with recurrent architectures needed to capture seasonal patterns. DQN would struggle with the combinatorial action space, and pure policy gradient methods would have prohibitive variance given the long reward horizons.

**Reward Design:**
The reward function would balance competing objectives: `R = α × revenue - β × stockout_cost - γ × holding_cost - δ × max(0, total_inventory - budget)`. Stockout cost might be proportional to lost sales plus customer satisfaction penalty; holding cost reflects capital tied up in inventory. Challenges include: (1) reward shaping could create local optima (e.g., always ordering zero to avoid holding costs); (2) sparse rewards from seasonal products; (3) multi-objective tradeoffs requiring careful coefficient tuning; (4) delayed consequences—ordering decisions today affect availability weeks later.

**Large Action Space Solutions:**
Given 1000 products, I would not treat actions as a single 1000-dimensional decision. Instead: (1) factored action space with shared policy networks that take product features as input and output per-product decisions; (2) attention mechanisms to model product interactions (substitutes, complements); (3) action constraints ensuring total orders don't exceed budget; (4) sequential decision-making within each timestep, ordering products one-by-one conditioned on previous decisions.

**Safe Deployment:**
Production deployment would follow a staged approach: (1) train on historical data using offline RL/batch RL methods to initialize a reasonable policy without live experimentation; (2) deploy with a safety layer that constrains actions to within ±20% of current heuristic system; (3) gradual relaxation of constraints as the system proves reliable; (4) A/B testing comparing RL suggestions to baseline system before full handover; (5) human oversight dashboard with alerts for unusual recommendations; (6) fallback to rule-based system if anomalies detected. Continuous monitoring of KPIs (stockout rate, inventory turnover, profit margin) would trigger retraining or rollback if degradation occurs.

**Evaluation Rubric:**

| Criterion | Excellent (4) | Proficient (3) | Developing (2) | Beginning (1) |
|-----------|---------------|----------------|----------------|---------------|
| MDP Formulation | Complete state/action space with tractability considerations | Reasonable formulation with minor gaps | Partial formulation | Incorrect or missing formulation |
| Algorithm Choice | Justified selection with domain reasoning | Appropriate algorithm with some justification | Names algorithm without justification | Inappropriate choice |
| Reward Design | Multi-objective reward with challenge awareness | Reasonable reward function | Incomplete reward | Missing or incorrect reward |
| Action Space | Concrete solution for scale | Acknowledges problem with partial solution | Mentions scale issue | Ignores action space challenge |
| Deployment | Staged, safe deployment with monitoring | Mentions safety considerations | Partial deployment plan | No deployment consideration |

**Understanding Gap Indicator:**
If response lacks depth, this may indicate:
- Difficulty formulating real-world problems as MDPs
- Weak understanding of algorithm tradeoffs for different problem types
- Limited awareness of deployment challenges in production RL

---

## Diagnostic Feedback & Review Recommendations

### Performance-Based Review Guide

| If Incorrect On... | Gap Identified | Review Section | Priority |
|--------------------|----------------|----------------|----------|
| Question 1 | On-policy vs. off-policy | Core Concepts 5-6 + Flashcard 2 | High |
| Question 2 | Policy gradient variance | Core Concept 7 + Flashcard 3 | High |
| Question 3 | DQN stability mechanisms | Core Concept 9 + Flashcard 5 | Medium |
| Question 4 | Exploration in practice | Core Concept 10 + Practice P5 | Medium |
| Question 5 | Full system integration | All sections | Low |

### Targeted Review Recommendations

#### For Multiple Choice Errors (Questions 1-2)
**Indicates:** Foundational concept gaps
**Action:** Review:
- Study Notes: Core Concepts 5-7 (Q-Learning, TD, Policy Gradient)
- Flashcards: Cards 2 and 3
- Practice Problems: P1 (Q-learning computation) and P4 (Actor-Critic)
**Focus On:** Understanding WHY different update rules lead to different learned policies

#### For Short Answer Errors (Questions 3-4)
**Indicates:** Application difficulties
**Action:** Practice:
- Practice Problems: P3 (DQN architecture) and P5 (Exploration debugging)
- Concept Map: DQN cluster and Exploration-Exploitation relationship
**Focus On:** Connecting theoretical concepts to practical implementation decisions

#### For Essay Weakness (Question 5)
**Indicates:** Integration challenges
**Action:** Review interconnections:
- Concept Map: Full pathway from MDP to Algorithms to Deployment
- All Practice Problems for procedural fluency
- Study Notes: Practical Applications section
**Focus On:** Building mental models that connect problem formulation → algorithm selection → implementation → deployment

### Mastery Indicators

- **5/5 Correct:** Strong mastery; ready for RL implementation projects
- **4/5 Correct:** Good understanding; review indicated gap
- **3/5 Correct:** Moderate understanding; systematic review + more practice
- **2/5 or below:** Foundational gaps; restart from Concept Map Critical Path

---

## Skill Chain Traceability

```
Study Notes (Source) ──────────────────────────────────────────────────────┐
    │                                                                      │
    │  10 Core Concepts, 14 Key Terms, 4 Applications                      │
    │                                                                      │
    ├────────────┬────────────┬────────────┬────────────┐                  │
    │            │            │            │            │                  │
    ▼            ▼            ▼            ▼            ▼                  │
Concept Map  Flashcards   Practice    Quiz                                 │
    │            │        Problems      │                                  │
    │ 26 concepts│ 5 cards   │ 5 problems│ 5 questions                     │
    │ 42 rels    │ 2E/2M/1H  │ 1W/2S/1C/1D│ 2MC/2SA/1E                     │
    │ 4 pathways │           │           │                                 │
    │            │           │           │                                 │
    └─────┬──────┴─────┬─────┴─────┬─────┘                                 │
          │            │           │                                       │
          │ Centrality │ Practice  │                                       │
          │ → Card     │ → Quiz    │                                       │
          │ difficulty │ distractors│                                      │
          │            │           │                                       │
          └────────────┴───────────┴───────────────────────────────────────┘
                                   │
                          Quiz integrates ALL
                          upstream materials
```

---

## Complete 5-Skill Chain Summary

| Skill | Output | Key Contribution to Chain |
|-------|--------|---------------------------|
| study-notes-creator | 10 concepts, theory, applications | Foundation content |
| concept-map | 26 nodes, 42 relationships, 4 pathways | Structure + learning paths |
| flashcards | 5 cards (2E/2M/1H) | Memorization + critical flags |
| practice-problems | 5 problems (1W/2S/1C/1D) | Procedural fluency + common mistakes |
| quiz | 5 questions (2MC/2SA/1E) | Assessment + diagnostic feedback |
