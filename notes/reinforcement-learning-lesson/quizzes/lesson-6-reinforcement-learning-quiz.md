# Assessment Quiz: Lesson 6 - Reinforcement Learning

**Source:** Lessons/Lesson_6.md
**Subject Area:** AI Learning - Reinforcement Learning: Foundations, Algorithms, and Decision-Making
**Date Generated:** 2026-01-08
**Total Questions:** 5
**Estimated Time:** 35-45 minutes

---

## Instructions

This assessment evaluates your understanding of reinforcement learning fundamentals, including MDPs, value functions, temporal difference learning, policy gradients, and practical RL system design. Answer all questions completely, showing your reasoning where applicable.

**Question Distribution:**
- Multiple Choice (2): Conceptual understanding (Remember/Understand)
- Short Answer (2): Application and analysis (Apply/Analyze)
- Essay (1): Synthesis and evaluation (Evaluate/Synthesize)

---

## Part A: Multiple Choice (10 points each)

### Question 1: Temporal Difference Learning

**Which statement correctly characterizes the relationship between TD learning, Monte Carlo methods, and Dynamic Programming?**

A) TD learning requires a model of the environment like Dynamic Programming, but updates incrementally like Monte Carlo methods

B) TD learning combines bootstrapping from Dynamic Programming (updating from estimates) with learning from experience like Monte Carlo methods

C) TD learning waits until the end of an episode like Monte Carlo, but uses the Bellman equation like Dynamic Programming

D) TD learning is equivalent to Monte Carlo methods when the discount factor γ = 1 and all episodes terminate

---

### Question 2: Algorithm Classification

**A robotics researcher needs to train a policy for a 6-DoF robotic arm (continuous joint torques) to stack blocks. The environment provides sparse rewards (+1 for successful stack, 0 otherwise). Which algorithm family and specific algorithm would be most appropriate?**

A) Value-based: DQN with prioritized experience replay for sample efficiency

B) Policy gradient: REINFORCE with baseline for direct continuous action optimization

C) Actor-Critic: SAC for continuous actions, sample efficiency, and built-in exploration

D) Model-based: World model + planning for maximum sample efficiency in sparse reward settings

---

## Part B: Short Answer (15 points each)

### Question 3: Value Function Analysis

**Context:** Consider an RL agent learning to navigate a 4x4 grid world. The agent starts at position (0,0) and must reach goal at (3,3). Each step costs -1 reward, and reaching the goal gives +10. The discount factor is γ = 0.9.

**Tasks:**

a) After training, would you expect V*(0,0) to be positive or negative? Justify your answer with a rough calculation assuming optimal path length of 6 steps. (5 points)

b) Explain why Q*(s, a) might be negative for the optimal action while V*(s) is positive, or vice versa. Provide a concrete example. (5 points)

c) The agent uses ε-greedy exploration with ε = 0.2. After observing that the trained policy sometimes takes suboptimal paths, a colleague suggests setting ε = 0 for deployment. Is this correct? What considerations apply? (5 points)

---

### Question 4: Policy Gradient Mechanics

**Context:** You are implementing a policy gradient algorithm for a continuous control task.

**Tasks:**

a) Write the policy gradient theorem equation and explain what each term represents. Why is the log probability gradient used instead of the probability directly? (5 points)

b) Your training shows high variance in gradient estimates, causing unstable learning. Describe two techniques to reduce variance, and explain the tradeoff for each. (5 points)

c) After adding a baseline b(s), training becomes more stable but you notice the policy converges to a local optimum. Your colleague suggests the baseline might be causing bias. Is this concern valid? Prove or disprove mathematically. (5 points)

---

## Part C: Essay (30 points)

### Question 5: RLHF System Design

**Prompt:** You are designing the reinforcement learning component of an RLHF (Reinforcement Learning from Human Feedback) system to fine-tune a language model for helpful, harmless, and honest responses.

**Your essay must address:**

1. **RL Formulation** (7 points)
   - Define the state, action, and reward in the RLHF context
   - Explain how the reward model is trained from human preferences
   - Discuss the role of the reference policy and KL penalty

2. **Algorithm Choice** (7 points)
   - Justify using PPO over alternatives (DQN, REINFORCE, SAC)
   - Explain the clipping mechanism and its purpose
   - Discuss batch size and update frequency considerations

3. **Training Challenges** (8 points)
   - Reward hacking: how might the model exploit the reward function?
   - Distribution shift: how does the policy distribution change during training?
   - Mode collapse: how can diversity be maintained?

4. **Evaluation and Safety** (8 points)
   - How to evaluate RLHF success beyond reward model scores?
   - What safety considerations are unique to LLM RLHF?
   - How to detect and mitigate harmful behaviors learned during RLHF?

**Evaluation Criteria:**
- Technical accuracy of RL concepts applied to RLHF
- Understanding of LLM-specific considerations
- Awareness of failure modes and mitigations
- Coherent integration of concepts

**Word Limit:** 600-800 words

---

## Answer Key

### Question 1: Temporal Difference Learning

**Correct Answer: B**

**Explanation:**

| Method | Uses Model? | Bootstraps? | Updates When? |
|--------|-------------|-------------|---------------|
| Dynamic Programming | Yes | Yes | Any time (sweep) |
| Monte Carlo | No | No | End of episode |
| TD Learning | No | Yes | Every step |

- **Option B is correct:** TD learning bootstraps (updates estimates from estimates) like DP, but learns from direct experience without a model like MC.

- **Option A is incorrect:** TD learning is model-free, unlike DP.

- **Option C is incorrect:** TD updates after every step, not end of episode.

- **Option D is incorrect:** TD(0) is not equivalent to MC even with γ=1; TD bootstraps from current estimates while MC uses actual returns.

**Understanding Gap:** If you selected C or D, review the distinction between bootstrapping (using estimates) and Monte Carlo returns (using actual experienced rewards).

---

### Question 2: Algorithm Classification

**Correct Answer: C**

**Explanation:**

**Why SAC (Option C) is best:**

| Requirement | SAC Capability |
|-------------|----------------|
| Continuous actions | Gaussian policy with reparameterization |
| Sparse rewards | Entropy maximization encourages exploration |
| Sample efficiency | Off-policy with large replay buffer |
| Stability | Twin critics reduce overestimation |

**Why other options are inferior:**

- **Option A (DQN):** Cannot handle continuous actions without discretization, which is impractical for 6-DoF control

- **Option B (REINFORCE):** High variance, poor sample efficiency, struggles with sparse rewards

- **Option D (Model-based):** While sample efficient, learning accurate dynamics for contact-rich manipulation is extremely difficult

**Understanding Gap:** If you selected A, review the distinction between discrete and continuous action spaces and their algorithm requirements.

---

### Question 3: Value Function Analysis

**Model Answer:**

**a) V*(0,0) Sign Analysis (5 points)**

Optimal path: 6 steps from (0,0) to (3,3)
```
V*(0,0) = -1 + γ(-1) + γ²(-1) + γ³(-1) + γ⁴(-1) + γ⁵(-1) + γ⁶(+10)
        = Σ_{t=0}^{5} -γ^t + γ⁶ × 10
```

With γ = 0.9:
```
Step costs: -1 - 0.9 - 0.81 - 0.729 - 0.656 - 0.590 = -4.685
Goal reward: 0.9⁶ × 10 = 0.531 × 10 = 5.31

V*(0,0) ≈ -4.69 + 5.31 = +0.62
```

**V*(0,0) is positive** because the discounted goal reward exceeds the cumulative step penalties.

**b) Q* vs V* Sign Relationship (5 points)**

Q*(s,a) and V*(s) can have opposite signs when:

**Example:** Agent is one step from goal (state s_near), with two actions:
- Optimal action a* (toward goal): Q*(s_near, a*) = -1 + 0.9×10 = +8
- Bad action a_bad (away from goal): Q*(s_near, a_bad) = -1 + 0.9×V(farther) ≈ -1 + 0.9×(-2) = -2.8

V*(s_near) = max Q* = +8 (positive)
Q*(s_near, a_bad) = -2.8 (negative)

The value function represents the best case, while Q for suboptimal actions can be much worse.

**c) ε = 0 for Deployment (5 points)**

**Yes, setting ε = 0 for deployment is correct.**

Considerations:

| Factor | Analysis |
|--------|----------|
| Evaluation accuracy | ε > 0 introduces random actions, making evaluation noisy |
| Optimal behavior | Trained Q-values should guide best actions without exploration |
| Stochasticity | If environment is stochastic, greedy policy is still appropriate |
| Distributional shift | Deployment environment should match training; if not, some exploration might help adapt |

**Caveat:** If the deployment environment differs from training (distribution shift), small ε might help discover better actions. But for evaluation of learned policy, ε = 0 is standard.

---

### Question 4: Policy Gradient Mechanics

**Model Answer:**

**a) Policy Gradient Theorem (5 points)**

```
∇_θ J(θ) = E_{s~d^π, a~π} [∇_θ log π_θ(a|s) · Q^π(s,a)]
```

| Term | Meaning |
|------|---------|
| ∇_θ J(θ) | Gradient of expected return w.r.t. policy parameters |
| E_{s~d^π} | Expectation over state distribution under policy |
| ∇_θ log π_θ(a\|s) | Score function: direction to increase action probability |
| Q^π(s,a) | Action-value: how good the action was |

**Why log probability?**
```
∇_θ π(a|s) = π(a|s) · ∇_θ log π(a|s)
```
The log derivative allows importance weighting—samples are drawn from π, and log gradient automatically accounts for this. Also provides numerical stability for small probabilities.

**b) Variance Reduction Techniques (5 points)**

**Technique 1: Baseline Subtraction**
```
∇J ∝ ∇log π(a|s) · [Q(s,a) - b(s)]
```
- **Benefit:** Centers returns around zero, reducing magnitude
- **Tradeoff:** Requires estimating b(s), typically another network

**Technique 2: Actor-Critic (Use TD estimate)**
```
A(s,a) ≈ r + γV(s') - V(s)   instead of full return G
```
- **Benefit:** Lower variance than Monte Carlo returns
- **Tradeoff:** Introduces bias from value function approximation error

**Other options:** Reward normalization, larger batch sizes, entropy regularization.

**c) Baseline Bias Proof (5 points)**

**The concern is NOT valid.** Baseline does not introduce bias.

**Proof:**
```
E_a~π [∇log π(a|s) · b(s)]
= b(s) · E_a~π [∇log π(a|s)]
= b(s) · Σ_a π(a|s) · ∇_θ log π(a|s)
= b(s) · Σ_a π(a|s) · [∇_θ π(a|s) / π(a|s)]
= b(s) · Σ_a ∇_θ π(a|s)
= b(s) · ∇_θ [Σ_a π(a|s)]
= b(s) · ∇_θ [1]
= b(s) · 0
= 0
```

Since the baseline term has zero expected contribution, the gradient remains unbiased.

**Local optimum cause:** The local optimum is due to policy gradient's local optimization nature, not the baseline. Solutions: entropy regularization, better exploration, multiple random seeds.

---

### Question 5: RLHF System Design

**Rubric (30 points total):**

| Component | Excellent (Full) | Adequate (Half) | Insufficient (Minimal) |
|-----------|------------------|-----------------|------------------------|
| RL Formulation (7) | Complete MDP definition, reward model training, KL explanation | Partial formulation, missing key elements | Incorrect or missing formulation |
| Algorithm Choice (7) | PPO justification with clipping explanation, practical considerations | Mentions PPO without full justification | Wrong algorithm or no justification |
| Training Challenges (8) | All three challenges with specific mitigations | Identifies challenges without solutions | Missing major challenges |
| Evaluation & Safety (8) | Beyond-reward evaluation, specific safety measures | General safety discussion | Vague or missing safety |

**Model Answer:**

**1. RL Formulation**

In RLHF for language models, the MDP is defined as:

**State:** The prompt plus any tokens generated so far. Each state is a sequence of tokens representing the conversation context.

**Action:** The next token to generate. The action space is the vocabulary (typically 32K-100K tokens).

**Reward:** Provided by a reward model r_φ trained on human preference data. The reward model is trained by:
1. Collecting pairs of responses (y_w, y_l) to the same prompt, where w is preferred
2. Training r_φ to satisfy: r_φ(x, y_w) > r_φ(x, y_l)
3. Using Bradley-Terry model: P(y_w ≻ y_l) = σ(r_φ(y_w) - r_φ(y_l))

**Reference Policy and KL Penalty:**
The optimization objective includes a KL divergence term:
```
J(θ) = E[r_φ(x,y)] - β · KL(π_θ || π_ref)
```
The reference policy π_ref is typically the supervised fine-tuned (SFT) model. The KL penalty prevents the RL policy from deviating too far, maintaining language fluency and preventing reward hacking.

**2. Algorithm Choice: PPO**

PPO is preferred for RLHF because:

**Over DQN:** Language generation requires modeling a distribution over tokens, not just selecting the argmax. DQN's discrete max operation doesn't naturally produce diverse, fluent text.

**Over REINFORCE:** REINFORCE has high variance due to using full trajectory returns. For long sequences (100+ tokens), this variance makes training impractical.

**Over SAC:** While SAC handles continuous actions well, token generation is discrete. SAC's entropy term is also different from PPO's clipping, which provides more stable updates.

**PPO's Clipping Mechanism:**
```
L^CLIP = E[min(r_t A_t, clip(r_t, 1-ε, 1+ε) A_t)]
where r_t = π_θ(a|s) / π_old(a|s)
```
This prevents destructive large policy updates. If the policy changes too much (r_t far from 1), the objective is clipped, stopping gradient flow for that sample.

**Practical Considerations:**
- Batch size: Large batches (512-2048 prompts) for stable gradient estimates
- Updates: Multiple PPO epochs per batch, but not too many to avoid overfitting to current reward model

**3. Training Challenges**

**Reward Hacking:**
The model may learn to exploit patterns in the reward model rather than genuinely being helpful. Examples:
- Excessive verbosity (reward model trained on longer = better)
- Sycophantic responses (agreeing with user gets higher reward)
- Mitigation: Diverse reward model training data, multiple reward models, human evaluation checkpoints

**Distribution Shift:**
As π_θ improves, it generates responses different from the reward model's training distribution. The reward model may give unreliable scores for out-of-distribution outputs.
- Mitigation: KL constraint keeps policy near reference, periodic reward model retraining, ensemble of reward models

**Mode Collapse:**
The model may converge to a narrow set of "safe" responses that score well, losing diversity.
- Mitigation: Entropy bonus in objective, temperature sampling during training, best-of-n sampling evaluation

**4. Evaluation and Safety**

**Beyond Reward Model Evaluation:**
- Human evaluation studies (helpfulness, harmlessness, honesty ratings)
- Benchmark performance (TruthfulQA, HHH eval)
- Red-teaming: adversarial prompting to find failures
- Downstream task performance (does helpfulness transfer?)

**LLM-Specific Safety Considerations:**
- Jailbreak resistance: model shouldn't be manipulated into harmful outputs
- Hallucination: RLHF may increase confident but false statements
- Dual-use: helpful capability can be misused
- Bias amplification: RLHF may amplify biases in preference data

**Detection and Mitigation:**
- Constitutional AI: add rule-based constraints
- Safety classifiers: filter outputs post-hoc
- Iterative red-teaming and retraining
- Transparency: document training data and known limitations
- Staged deployment with monitoring

RLHF represents a powerful but delicate process requiring careful attention to reward model quality, training stability, and safety evaluation throughout development.

---

## Performance Interpretation Guide

| Score | Mastery Level | Recommended Action |
|-------|---------------|-------------------|
| 90-100% | **Mastery** | Ready for advanced RL research topics |
| 75-89% | **Proficient** | Review specific gaps, practice implementation |
| 60-74% | **Developing** | Re-study core algorithms, more practice problems |
| Below 60% | **Foundational** | Complete re-review of Lesson 6, start with value functions |

---

## Review Recommendations by Question

| If You Struggled With | Review These Sections |
|----------------------|----------------------|
| Question 1 | Lesson 6: TD Learning, comparison with MC and DP |
| Question 2 | Lesson 6: Algorithm families, continuous action handling |
| Question 3 | Lesson 6: Value functions, Bellman equations, exploration |
| Question 4 | Lesson 6: Policy gradients, variance reduction |
| Question 5 | Lesson 6: Actor-Critic, PPO; Lesson 3: RLHF |

---

*Generated from Lesson 6: Reinforcement Learning | Quiz Skill*
