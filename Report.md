# Project Report — Collaboration and Competition

## Learning Algorithm

This project implements **self-play DDPG (Deep Deterministic Policy Gradients)** for the Unity Tennis environment, where two agents learn to cooperatively rally a ball over a net.

### DDPG Overview

DDPG is an actor-critic algorithm for continuous action spaces that combines ideas from DPG (Deterministic Policy Gradient) and DQN:

- **Actor network** μ(s|θ^μ): Learns a deterministic policy mapping states to continuous actions.
- **Critic network** Q(s,a|θ^Q): Evaluates state-action pairs, providing gradient signal for the actor.

The actor is updated via the deterministic policy gradient:

```
∇_θμ J ≈ E[ ∇_a Q(s,a|θ^Q)|a=μ(s) · ∇_θμ μ(s|θ^μ) ]
```

### Self-Play Architecture

Since Tennis is a symmetric environment — an optimal policy for one side is optimal for the other — both agents share:

- **One actor network** (and its target)
- **One critic network** (and its target)
- **One replay buffer**

At each timestep, both agents observe their own states, take actions through the shared actor, and contribute their individual transitions to the shared buffer. This doubles the effective experience rate compared to a single agent and ensures both sides of the court are equally represented in training.

### Target Networks

Like DQN, DDPG uses target networks for stability, updated via **soft updates**:

```
θ_target = τ · θ_local + (1 - τ) · θ_target
```

with τ = 0.001, blending 0.1% of the local weights per learning step.

### Exploration Noise with Decay

For exploration, DDPG adds Ornstein-Uhlenbeck noise to the actor's output:

```
dx = θ(μ - x)dt + σdW
```

**Noise decay** is the critical adaptation for this environment. In the [Continuous Control project](https://github.com/trwilcoxson/ddpg-continuous-control), constant exploration noise worked because the reward signal was dense — the agent received feedback every timestep. Tennis has **sparse rewards**: +0.1 only when the ball crosses the net, -0.01 when it hits the ground. This means:

1. **Early training**: The agent needs aggressive exploration (σ = 0.2) to discover that hitting the ball is rewarded at all.
2. **Mid training**: As the agent learns to rally, noise must decrease so it doesn't disrupt the fragile cooperative equilibrium.
3. **Late training**: A small noise floor (σ = 0.01) maintains minimal exploration without destabilizing the policy.

The noise sigma decays by a factor of 0.9995 per episode, dropping from 0.2 to ~0.09 by episode 1500. Without this decay, the agent would learn to rally briefly but then lose the policy as noise disrupts the delicate coordination between the two sides.

## Network Architecture

### Actor

```
Input (24) → FC1 (128, ReLU) → FC2 (128, ReLU) → FC3 (2, Tanh)
```

| Layer | Parameters |
|---|---|
| FC1: 24 → 128 | 3,200 |
| FC2: 128 → 128 | 16,512 |
| FC3: 128 → 2 | 258 |
| **Total** | **19,970** |

The 24-dimensional state (3 stacked frames × 8 variables for ball/racket position and velocity) is processed through two hidden layers. The final Tanh constrains actions to [-1, 1].

### Critic

```
Input (24) → FC1 (128, ReLU) → [concat action (2)] → FC2 (130→128, ReLU) → FC3 (128→64, ReLU) → FC4 (64→1)
```

| Layer | Parameters |
|---|---|
| FC1: 24 → 128 | 3,200 |
| FC2: 130 → 128 | 16,768 |
| FC3: 128 → 64 | 8,256 |
| FC4: 64 → 1 | 65 |
| **Total** | **28,289** |

Actions are injected after the first hidden layer (as recommended in the DDPG paper). The narrowing architecture (128 → 64 → 1) forces the critic to form compressed representations, which helps with generalization in sparse-reward settings.

**No BatchNorm**: The [Continuous Control ablation study](https://github.com/trwilcoxson/ddpg-continuous-control) showed BatchNorm provides only a modest convergence advantage (~7% fewer episodes) for DDPG. Tennis has a simpler state space (24 dims, all in similar ranges) compared to Reacher (33 dims with mixed physical units), making BatchNorm unnecessary.

**Weight initialization**: Hidden layers use fan-in uniform initialization (±1/√fan_in). Output layers use uniform(-3e-3, 3e-3) to ensure initial outputs are near zero.

## Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Replay buffer size | 100,000 | Smaller than Reacher (2 agents vs. 20; less experience per episode) |
| Batch size | 128 | Standard size; smaller than Reacher's 256 due to less data |
| Discount factor (γ) | 0.99 | Standard; values future rewards highly |
| Soft update rate (τ) | 0.001 | Slow blending for target network stability |
| Actor learning rate | 1e-4 | Standard DDPG, Adam optimizer |
| Critic learning rate | 1e-3 | 10x actor LR (standard DDPG practice) |
| Learn every | 1 step | Only 2 agents producing transitions (vs. 20 in Reacher) |
| Num updates | 1 | Single update per step (vs. 10 in Reacher) |
| OU θ | 0.15 | Standard mean-reversion rate |
| OU σ (initial) | 0.2 | Standard noise scale |
| **Noise decay** | **0.9995/episode** | Reduces σ from 0.2 → ~0.09 by episode 1500 |
| **Noise floor** | **0.01** | Prevents noise from vanishing completely |
| Gradient clipping | 1.0 | Clips critic gradients to stabilize training |
| Max episodes | 5,000 | Patience for sparse rewards (Reacher needed only 300) |

### Key Differences from Continuous Control (Reacher)

| Parameter | Reacher | Tennis | Why |
|---|---|---|---|
| Buffer size | 1,000,000 | 100,000 | 2 agents vs. 20; much less data |
| Batch size | 256 | 128 | Proportional to data availability |
| Hidden units | 256 | 128 | Simpler environment (24 vs. 33 state dims) |
| Learn every | 20 steps | 1 step | Few transitions per step; learn from all of them |
| Num updates | 10 | 1 | Balanced with learn_every=1 |
| BatchNorm | Yes | No | Simpler state space; ablation showed it's optional |
| Noise decay | None | 0.9995/ep | Critical for sparse rewards |
| Max episodes | 300 | 5,000 | Sparse rewards need patience |

## Plot of Rewards

![Training Scores](scores_plot.png)

The plot shows the max score across both agents per episode (light) and the 100-episode rolling average (dark). The environment was **solved at episode 1336**, when the 100-episode rolling average first exceeded 0.5. Training continued for 200 additional episodes, during which the rolling average peaked at approximately 1.2.

**Training dynamics**: Scores remain near zero for the first ~1200 episodes as the agents explore randomly — this extended "desert" is characteristic of sparse-reward environments where the agent must discover the rewarding behavior (hitting the ball over the net) through random exploration before it can begin learning. Around episode 1200, individual scores begin spiking (0.1–0.6), and the rolling average rises rapidly from 0.01 to 0.5 over ~130 episodes. This sudden onset reflects the cooperative nature of Tennis: once one agent learns to return the ball, the other agent starts receiving positive experiences too, creating a positive feedback loop.

**Greedy evaluation**: The saved checkpoint was tested over 100 episodes with no exploration noise. The agent achieved an average score of **2.042** (σ = 0.906, min = 0.000, max = 2.700), confirming robust performance well above the 0.5 solve threshold. The occasional zero-score episodes likely reflect the inherent stochasticity of the initial ball placement.

## Ideas for Future Work

1. **MADDPG (Multi-Agent DDPG)** (Lowe et al., 2017): The full MADDPG algorithm gives each agent its own actor and critic, where the critic receives the observations and actions of all agents. This "centralized training, decentralized execution" approach could handle asymmetric environments where the two sides have different optimal strategies.

2. **Prioritized Experience Replay** (Schaul et al., 2016): In sparse-reward environments like Tennis, the vast majority of transitions have zero reward. Prioritizing high-reward transitions would focus learning on the rare informative experiences, potentially accelerating convergence through the initial "desert" phase.

3. **TD3 (Twin Delayed DDPG)** (Fujimoto et al., 2018): Addresses DDPG's overestimation bias with twin critics, delayed policy updates, and target policy smoothing. These stabilization techniques could reduce the variance in the learned policy and produce more consistent rally behavior.

4. **PPO with Self-Play**: PPO's clipped objective is more robust to hyperparameter choices than DDPG. Combined with self-play, it could provide more stable training without the need for careful noise decay tuning.

5. **Curriculum Learning**: Start with shorter rallies (e.g., reward for just making contact) and gradually increase the difficulty. This would bridge the sparse-reward gap by providing denser early feedback, potentially eliminating the 1200-episode desert observed in training.
