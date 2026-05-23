# Herbivore Intelligence — RL Substrate v2

Design document for the next-generation AEONS RL substrate. Replaces the
current 1-step REINFORCE per-organism MLP (`intelligence_level_herbivore_1.rs`)
with an architecture that can scale to the long-term simulation vision:
emergent pursuit, mating, play, and limb-based locomotion — none of
which an oracle can supply.

> **Companion reading:** [`/RL_Discussion.md`](../RL_Discussion.md) for
> why the current substrate fails. This document picks up where that
> one left off: not "why doesn't pursuit work", but "what does the
> substrate need to *be* before any of the long-term behaviours can
> emerge".

---

## 1. Goals & non-goals

### Goals

1. **Solve sparse-reward credit assignment.** Reward earned at tick T must
   propagate back through the trajectory that caused it. Pursuit reward
   should reinforce the *approach*, not just the final tick of contact.
2. **Solve the exploration bootstrap.** A fresh agent in an empty world
   must have an intrinsic reason to act, otherwise REINFORCE has nothing
   to start from.
3. **Sample efficiency**. Pool experience across the herbivore population
   so 100 organisms produce ~100× the gradient signal of 1, not 1× repeated.
4. **Continuous-action support**. Direction + speed (and later, per-limb
   torque) are continuous. The network output cannot saturate itself
   into a corner (the failure mode the histogram analysis confirmed).
5. **Backward compatibility with the AEONS substrate**: per-organism
   identity preserved, selection pressure operates via reproduction,
   inheritance carries over learned features.

### Non-goals (for v2)

- Multi-agent self-play protocols (mating logic). Reserved for v3.
- Hierarchical task decomposition. Reserved for v3 once limbs land.
- Curriculum learning infrastructure. Manual reward shaping is
  sufficient for the first iteration.

---

## 2. Architecture overview

```
                     ┌──────────────────────────────────────────┐
                     │              SHARED BACKBONE             │
   state s ────────► │  obs ⊕ gene_vec  →  Linear → ReLU →      │ ─── features
   gene vec g  ────► │  Linear → ReLU (32 → 64 → 64 hidden)     │
                     └──────────────────────────────────────────┘
                                        │
                       ┌────────────────┴────────────────┐
                       │                                 │
                ┌──────▼──────┐                  ┌───────▼────────┐
                │ ACTOR HEAD  │                  │  CRITIC HEAD   │
                │  64 → 16 → 3│ ─────► action a  │  64 → 16 → 1   │ ─► V(s)
                │  (μ_speed,  │      (sampled)   │  (scalar value)│
                │   μ_θ, σ_a) │                  └────────────────┘
                └─────────────┘
                       │
                       ▼
        action = (speed_a, θ) sampled from N(μ, σ²)
        movement_direction = (sin(πθ), 0, cos(πθ))    ← intrinsically bounded
        movement_speed     = ((speed_a+1)/2) · MAX_SPEED

                       ▼
              ┌────────────────────────┐
              │  CURIOSITY MODULE (RND)│
              │  predictor f_φ tracks  │ ────►  r_intrinsic = ||f_φ(s) − f_θ̄(s)||²
              │  fixed random target   │
              │  f_θ̄ — error ∝ novelty │
              └────────────────────────┘

                       ▼
                       ▼              ┌─────────────────────┐
        (s, a, r_ext + λ·r_int, s′) ─►│  REPLAY BUFFER      │
                                      │  capacity ~64k      │
                                      │  per-agent strands  │
                                      └─────────────────────┘
                                              │
                            ┌─────────────────┴─────────────────┐
                            │     TRAINING WORKER (5 Hz)        │
                            │  sample minibatch, compute n-step │
                            │  advantage, A2C loss, Adam step   │
                            └───────────────────────────────────┘
```

Three key departures from the current substrate:

1. **Shared trunk + per-organism gene conditioning.** The backbone is *one*
   network, not 4096. Per-organism behavioural variation is injected via
   a small per-organism gene vector (8-16 floats) that's concatenated to
   the observation. Selection pressure operates by reproduction
   propagating the gene vector; the trunk learns from the pooled
   experience of every agent. ~100× the sample efficiency of the
   current per-organism-row approach.

2. **Actor-Critic instead of REINFORCE.** A critic head predicts `V(s)`.
   The advantage `A(s, a) = R_n(s) − V(s)` is the gradient target,
   where `R_n` is an n-step bootstrapped return (n ≈ 16-32 ticks).
   Solves the credit-assignment failure that doomed pursuit.

3. **Intrinsic motivation via RND** (Random Network Distillation). A
   small "predictor" network is trained to match the output of a fixed,
   random "target" network on the same input. Prediction error is the
   novelty signal; novel states get an intrinsic reward bonus. This is
   the missing ingredient for play and for bootstrapping any rare-reward
   behaviour. Cheap (one extra forward + one tiny backward per tick),
   no manual reward shaping required.

---

## 3. Core components

### 3.1 State representation

Inputs to the backbone (32 dims after concatenation with the 16-dim gene vec):

| Idx | Field | Source |
|---|---|---|
| 0 | hunger | `Organism::hunger` |
| 1 | dopamine | `Organism::dopamine` |
| 2 | target_distance / SENSORY_RADIUS | `Organism::target_distance` |
| 3 | has_photo | nearest_prey present? |
| 4 | rel_x / WORLD_MODEL_RADIUS | `nearest_prey` |
| 5 | rel_z / WORLD_MODEL_RADIUS | `nearest_prey` |
| 6 | rel_vx / VELOCITY_NORM_SCALE | prey's velocity x |
| 7 | rel_vz / VELOCITY_NORM_SCALE | prey's velocity z |
| 8 | self_speed / MAX_SPEED | `Organism::movement_speed` |
| 9-12 | second-nearest photo block (rel_x, rel_z, has, dist) | world_model |
| 13-14 | nearest carnivore (rel_x, rel_z) — flee signal | world_model |
| 15 | energy / max_energy | derived |
| 16-31 | per-organism gene vector | inherited / mutated |

The gene vector is *not just brain hyperparameters* — it carries any
slowly-evolving features the agent should adapt over generations:
risk preferences, hunger thresholds, conspecific tolerance, etc.
Genes mutate with Gaussian noise on reproduction.

### 3.2 Action space

Two output dimensions (speed + angle), one of which is intrinsically
bounded by trigonometric mapping rather than tanh-clipping:

```
μ_speed ∈ ℝ  (linear)        → speed_a = clamp(tanh(μ_speed), -1, 1)
μ_angle ∈ ℝ  (linear)        → θ      = clamp(tanh(μ_angle), -1, 1)
σ_a     ∈ (0, ∞)  (softplus) → noise std (learned, per-output)

a_speed  = μ_speed + σ_a · ε_speed
a_angle  = μ_angle + σ_a · ε_angle

speed     = ((a_speed + 1)/2) · MAX_SPEED
direction = (sin(π·a_angle), 0, cos(π·a_angle))    ← always unit-length
```

Two crucial differences from the current `(dir_x, dir_z)` setup:

- **Angle-based direction can't saturate uselessly.** `tanh(μ_angle) = +1`
  and `tanh(μ_angle) = -1` correspond to *different bearings*, not the
  same one. The policy never has to reach into the corner of a hypercube
  to express a sharp turn.
- **σ is learned per-output**, not a fixed hyperparameter. At high
  uncertainty the policy explores broadly; once it's confident σ shrinks
  organically. The current fixed σ = 0.25 forces permanent exploration
  noise even after the policy has converged.

### 3.3 Network details

```
Backbone:  Linear(32 → 64) ─► ReLU ─► Linear(64 → 64) ─► ReLU
Actor:     Linear(64 → 16) ─► ReLU ─► Linear(16 → 5)
           outputs = [μ_speed, μ_angle, log σ_speed, log σ_angle, log σ_unused]
Critic:    Linear(64 → 16) ─► ReLU ─► Linear(16 → 1)
RND tgt:   Linear(32 → 64) ─► ReLU ─► Linear(64 → 64)  (frozen)
RND pred:  Linear(32 → 64) ─► ReLU ─► Linear(64 → 64)  (trained)
```

Total parameter count: ~12k weights. Tiny by modern DRL standards but
adequate for the herbivore action space. GPU memory: ~50 KB.

### 3.4 Advantage estimation — n-step bootstrap

Per agent, accumulate a rolling window of `n = 16` ticks. At each
training step:

```
A_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... + γ^{n-1}·r_{t+n-1}
      + γ^n · V(s_{t+n})
      − V(s_t)
```

`γ = 0.98` (decay over ~50 ticks = 7.5s of virtual time).

Actor loss: `−log π(a_t | s_t) · stop_grad(A_t)`
Critic loss: `(R_t − V(s_t))²` where `R_t` is the same n-step return
                                  *without* the V(s_t) subtraction.

Total loss: `L = L_actor + c_v · L_critic − c_h · H[π]`

`c_v = 0.5` (critic weight), `c_h = 0.01` (entropy bonus to prevent
premature commitment).

### 3.5 Experience replay

Replay is what amortises rare events. A single eat (~once per minute
under current conditions) gets reused 60× before being evicted.

- **Capacity**: 64k transitions, FIFO.
- **Per-transition storage**: state (32 f32) + action (2 f32) + reward
  (f32) + next-state (32 f32) + done (bool) ≈ 268 bytes. Buffer total:
  ~17 MB. Fits comfortably on GPU.
- **Sampling**: minibatch of 256 transitions per training step.
- **On-policy vs off-policy**: A2C is on-policy in textbook form, but
  short-horizon stale data (≤ 5 s old) is fine in practice and yields
  much better sample efficiency. We use **V-trace clipping** (the
  importance-ratio bound from IMPALA) to safely train on slightly
  off-policy samples.
- **Per-agent strands**: when an organism dies/respawns, its transitions
  stay in the buffer (they're attributable to states, not entities).
  Recycled slots don't pollute the gradient.

### 3.6 Intrinsic motivation (RND)

Random Network Distillation. Cheap, well-studied, robust to
hyperparameter choice. Replaces the per-agent dopamine-decay-as-only-
exploration-driver mess.

```
target  = f_θ̄(s)            (random weights, frozen at init)
pred    = f_φ(s)            (trainable)
loss    = ||pred − target||²
reward  = stop_grad(loss)   # the same number is the intrinsic reward
```

A state visited many times has low prediction error → low intrinsic
reward. A novel state has high error → high intrinsic reward. The
agent is incentivised to seek novel observations.

This is *the* mechanism for play. Without an extrinsic reward, the
agent still wants to do new things.

Combined reward:
```
r_total = r_ext + λ · r_intrinsic
```
`λ = 0.5` initially; could be annealed if extrinsic reward density
increases.

### 3.7 Genes & inheritance

Genes are a 16-dim float vector inherited verbatim from parent on
reproduction, with Gaussian mutation (`σ_mutate = 0.05`). This vector
is concatenated with the state observation at every forward pass —
so the *same* shared policy produces different behaviour for
different gene values.

Selection pressure:
- Successful agents (more energy → more reproduction) propagate their
  genes.
- The gene vector is *also stored in the existing DNA slots* so the
  speciation system and lineage tree continue to work.
- Mutation rate and gene-vector size are themselves tunable, but the
  16-dim default is chosen so the brain can use roughly half for
  behavioural traits and half as identity tags.

---

## 4. Pool layout & systems

### 4.1 Pool resource

```rust
pub struct BrainPoolL4 {
    // Shared parameters — one set across all herbivores.
    backbone:     SharedBackbone<MyBackend>,
    actor:        ActorHead<MyBackend>,
    critic:       CriticHead<MyBackend>,
    rnd_target:   RndTarget<MyBackend>,   // frozen
    rnd_pred:     RndPredictor<MyBackend>,
    optimizer:    Box<dyn ActorCriticOpt>,
    rnd_opt:      Box<dyn RndOpt>,

    // Per-organism state.
    free:         Vec<u32>,
    map:          HashMap<Entity, u32>,
    n:            usize,
    pub device:   CudaDevice,

    // Gene vector — [N, GENE_DIM]. Inherited at reproduction.
    genes:        Vec<f32>,
    // Rolling rollout buffer per slot — [N, ROLLOUT_LEN, …].
    rollouts:     Vec<RolloutRing>,
    // Shared global replay buffer.
    replay:       ReplayBuffer,

    train_timer:  Timer,                  // training every 200ms
}
```

### 4.2 Schedule

```
PreUpdate:
    assign_brains_l4         // entity → slot, copy parent genes if any
    free_brains_l4           // recycle slots on Heterotroph removal

Update (gated by HETERO_BRAIN_TICK_INTERVAL = 150ms):
    rebuild_world_model_grid
    update_target_distance
    apply_brain_l4           // observe → forward → sample → drive movement
                             //         → push transition to rollout

Update (gated by TRAINING_TICK_INTERVAL = 200ms, decoupled from observe tick):
    consolidate_rollouts     // flush completed n-step windows to replay
    train_brain_l4           // sample minibatch, A2C + RND step
```

`apply_brain_l4` is fast (forward only). The training step is
decoupled: it samples from replay, can use larger batches, and runs
on a slower timer so the simulation isn't bottlenecked on gradient
updates.

### 4.3 Integration with existing systems

| System | Change |
|---|---|
| `predation.rs` | unchanged — still bumps `Organism::dopamine` and `predations`. The brain consumes these as observations + reward sources. |
| `reproduction.rs` | hands off the **gene vector** to the offspring slot (currently only weight rows). Adds mutation. |
| `sensory.rs` | unchanged — populates `target_distance` which is one of the input dims. |
| `energy.rs` | unchanged — dopamine depletion + hunger update. Note: dopamine is now *only* a state observation, not the reward signal. The brain reads reward from Δdopamine directly *plus* the curiosity bonus *plus* progress. |
| `lineages/speciation.rs` | reads the gene vector (now in DNA slots) like today. No code change. |
| `colony.rs` save/load | brain pool snapshot needs to dump shared parameters (~50 KB) + per-slot genes (~2 KB × N). Format bump to AEONS004. |

---

## 5. Reward design

Three additive channels, each with its own weight:

| Channel | Source | Weight |
|---|---|---|
| **Extrinsic — outcome** | Δ(`predations` + `reproductions`) per tick. Big positive spikes. | 1.0 |
| **Shaping** | Δ`target_distance` / SENSORY_RADIUS (negative when distance shrinks → reward) | 0.3 |
| **Intrinsic — curiosity** | RND prediction error on current state | 0.5 |

Dopamine and hunger are *no longer reward channels themselves*. They
go in the state observation only. This decouples the reward function
from the agent's internal energetics — the agent learns by maximising
reward, not by reading a number off its own homeostat. Removes the
"dopamine decay = constant negative reward" pathology (hypothesis A1
in RL_Discussion.md).

---

## 6. Hyperparameters

| Symbol | Value | Notes |
|---|---|---|
| `STATE_DIM` | 32 (16 obs + 16 gene) | |
| `HIDDEN` | 64 | backbone, single layer width |
| `GENE_DIM` | 16 | per-organism identity vector |
| `ROLLOUT_LEN` (n) | 16 | n-step return horizon |
| `GAMMA` (γ) | 0.98 | reward discount per tick |
| `LAMBDA_INTRINSIC` (λ) | 0.5 | curiosity vs extrinsic balance |
| `BATCH_SIZE` | 256 | per training step |
| `REPLAY_CAPACITY` | 64 000 | transitions |
| `LR_POLICY` | 3e-4 | Adam |
| `LR_RND` | 1e-3 | Adam (predictor only) |
| `MUTATION_STD` | 0.05 | per-gene Gaussian noise on reproduction |
| `BRAIN_TICK` | 150 ms | observe + act cadence |
| `TRAIN_TICK` | 200 ms | training cadence (decoupled) |
| `IS_CLIP` (V-trace) | 1.0 | importance-ratio cap |
| `ENTROPY_COEF` (c_h) | 0.01 | bonus to prevent collapse |
| `VALUE_COEF` (c_v) | 0.5 | critic loss weight |

---

## 7. Migration plan

The new pool is added as a **peer** of `BrainPoolHerbivore1`, not a
replacement. Both run simultaneously, gated on `IntelligenceLevel`:

- `Level1` → existing supervised brain (after the
  3D-vs-2D-mismatch revert). Default for new herbivores during the
  transition period.
- **`Level4` (new)** → `BrainPoolL4`. Opt-in by `.species` files or
  by a runtime resource toggle.

Phase plan:

1. **Build the substrate** (target: 2-3 weeks).
   - Skeleton pool, backbone + actor + critic, basic A2C loop.
   - Replay buffer + V-trace.
   - RND.
   - Save/load.
2. **Validate on pursuit** (target: 1 week).
   - Co-run with the supervised baseline. Use the existing dataset
     export to measure Δpredations per herbivore over a 2-hour run.
   - Pass criterion: A2C herbivores match or exceed supervised baseline
     **without an oracle**. If they tie, the substrate is real.
3. **Validate on play / exploration** (target: 2 weeks).
   - Add a sparse "novelty bonus" landscape (e.g., un-mapped terrain
     tiles). Measure: do A2C herbivores explore more of the map than
     supervised ones over the same wall time?
4. **Deprecate supervised pool** (when A2C is dominant on multiple
   tasks). Mark `Level1` as legacy. Default new spawns to `Level4`.

---

## 8. Risks & open questions

### Architectural

- **Per-organism gene head vs fully shared**: simpler version uses no
  per-organism gene vector at all. Lets us see if the *substrate*
  works before adding the diversity layer. Could be a Phase-0.5 step.
- **V-trace correctness with the per-agent rollout structure**: needs
  care so importance ratios are computed against the *actual* policy
  in use at sample time, not the current one. Standard issue, well
  understood in IMPALA / R2D2 literature.
- **GPU memory for replay**: 17 MB is fine for one species. If we
  eventually have separate pools per intelligence level × per
  classification (herbivore / carnivore / …), it multiplies. Cap is
  ~1 GB before we'd need to think harder.

### Behavioural

- **Reward channel weights are guesses.** The 1.0 / 0.3 / 0.5 starting
  values should be sweep-tuned once the substrate runs.
- **Curiosity can hijack the policy.** A pathological agent might
  pursue novelty over food and starve. Mitigation: weight λ down over
  time, or cap intrinsic reward magnitude.
- **Shared trunk could homogenise the population.** All herbivores
  inherit the same learned behaviour through the shared trunk; the
  gene vector is the only individual variation. If this turns out to
  suppress diversity, escalate to per-species heads (one shared trunk,
  one head per species).

### Forward compatibility

- **Limb control will need a hierarchical structure.** v2 doesn't
  build it, but the actor head can be expanded from 2 outputs to N
  outputs with no other change — leaving room for per-limb torque
  channels later.
- **Mating will need multi-agent value estimation.** v2 doesn't
  address this. The right time to design it is when reproductive
  partner-selection becomes a per-agent decision rather than a
  population-wide rule.
- **Communication / signals between agents.** A natural extension is
  to add a "signal" output that other nearby agents can read as part
  of their observation. Trivial architectural change; massive
  behavioural implications. Defer to v3.

---

## 9. Success criteria

The new substrate has succeeded when, on a clean 2-hour run with the
same world / cohort settings that produced the failing baseline:

1. **Pursuit emerges from scratch** — median `cos(movement, toward-nearest-photo)`
   for herbivores with a target rises to > 0.5 (vs current −0.115).
2. **Δpredations distribution shifts right** — fraction of organisms
   eating during the run rises from 79.6% to > 95%.
3. **Dopamine isn't pinned at zero** — at least 30% of herbivores have
   `dopamine > 0.1` at any snapshot (vs current 0.6%).
4. **RND novelty exploration** — measured by unique map cells visited
   by the population, A2C cohort covers > 1.5× the area of supervised
   baseline.

Failure on any of those by week 4 of substrate work triggers a design
review: either the choice of A2C/RND was wrong, or there's an
implementation bug worth pausing to find.
