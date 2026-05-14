// ============================================================================
//   new_intelligence_L1.rs — DQN heterotroph brain pool   (PSEUDOCODE ONLY)
// ============================================================================
//
// This file is a DESIGN DOCUMENT, not compilable Rust. It sketches a drop-in
// replacement for `intelligence_level_1_hetero.rs` based on Q-learning with
// experience replay and a target network — exactly the recipe from Mnih et al.
// 2015 ("Human-level control through deep reinforcement learning", Nature).
//
// Why DQN and not REINFORCE / PPO / SAC:
//   • DQN is the simplest deep-RL algorithm that reliably learns from sparse,
//     delayed rewards in a discrete action space — which is exactly what prey
//     pursuit is once we discretise heading.
//   • It sidesteps every Tier-1 algorithm bug we found in the REINFORCE
//     implementation (see intelligence_analysis.txt §EXECUTIVE SUMMARY).
//   • Per-organism networks fit the existing batched-pool architecture
//     unchanged — the pool is still N independent rows of a [N, ...] tensor.
//
// Out of scope (deliberately):
//   • Prioritised experience replay      — uniform sampling is simpler.
//   • Double DQN / dueling DQN           — incremental; not needed first.
//   • Distributional / noisy nets / NAF  — experimental.
//   • Recurrent state / LSTM             — body-frame state + replay is
//                                          enough for the Markov assumption.
//   • Continuous-action methods (SAC,    — discrete actions are sufficient
//     DDPG)                                for pursuit.
//
// All constants and shapes are placeholders; numbers chosen as "known-good
// DQN defaults" and to be tuned empirically.
//
// ============================================================================
//   1. CONSTANTS
// ============================================================================

const STATE_DIM:    usize = 18           // see §3 for layout
const HIDDEN:       usize = 64
const N_ACTIONS:    usize = 9            // 8 compass dirs + 1 "stop"

const GAMMA:        f32 = 0.95           // discount; short horizon for pursuit
const LR:           f32 = 5e-4           // Adam learning rate
const HUBER_DELTA:  f32 = 1.0            // Huber loss transition

const REPLAY_CAPACITY: usize = 1024      // per-organism ring buffer size
const BATCH_PER_SLOT:  usize = 32        // transitions sampled per training step per organism
const MIN_REPLAY:      usize = 64        // do not train a slot until buffer has this many

const EPSILON_START:       f32 = 1.00    // start fully random
const EPSILON_END:         f32 = 0.05    // permanent floor for exploration
const EPSILON_DECAY_STEPS: u32 = 100_000 // global decision count over which ε ramps

const TARGET_SYNC_STEPS: u32 = 1_000     // hard-copy online → target every K training steps
const GRAD_CLIP_NORM:    f32 = 10.0      // ||grad||₂ clamp

const MAX_SPEED:    f32 = 20.0           // matches existing apply_movement


// ============================================================================
//   2. DISCRETE ACTION SPACE
// ============================================================================
//
// Nine actions. Eight are unit XZ-plane direction vectors at MAX_SPEED.
// The ninth ("stop") halts movement without changing the current heading.
// Eight directions is fine-grained enough for pursuit in a 2D world — finer
// quantisation only matters near the prey, where speed control matters more.
//
//   index   direction (x, z)            label
//   ─────   ─────────────────           ─────
//     0     ( 0.000,  1.000)            N
//     1     ( 0.707,  0.707)            NE
//     2     ( 1.000,  0.000)            E
//     3     ( 0.707, -0.707)            SE
//     4     ( 0.000, -1.000)            S
//     5     (-0.707, -0.707)            SW
//     6     (-1.000,  0.000)            W
//     7     (-0.707,  0.707)            NW
//     8     ( 0.000,  0.000)            STOP (speed = 0, dir unchanged)

const ACTION_DIRS: [(f32, f32); N_ACTIONS] = [
    ( 0.000,  1.000),  ( 0.707,  0.707),  ( 1.000,  0.000),  ( 0.707, -0.707),
    ( 0.000, -1.000),  (-0.707, -0.707),  (-1.000,  0.000),  (-0.707,  0.707),
    ( 0.000,  0.000),
]


// ============================================================================
//   3. STATE VECTOR  (BODY FRAME)
// ============================================================================
//
// State is encoded in the organism's OWN frame, so heading does not need
// to appear explicitly — the same prey configuration produces the same
// state regardless of which way the organism is facing. This is the single
// most important fix: §S1 of intelligence_analysis.txt is eliminated by
// construction.
//
//   index   field                                  source
//   ─────   ──────────────────────────────────     ─────────────────────────
//   [0]     own_energy / max_energy                Organism
//   [1]     own_speed / MAX_SPEED                  Organism.movement_speed
//   [2..6]  nearest prey #1: rel_x_body / R,       WorldModelGrid +
//             rel_z_body / R,                      Transform.yaw rotation
//             distance / R,
//             is_visible (0 or 1)
//   [6..10] nearest prey #2: same 4 fields
//   [10..14] nearest prey #3: same 4 fields
//   [14..18] nearest prey #4: same 4 fields
//
//   R = WORLD_MODEL_RADIUS (currently 60.0 — keep the same).
//
// If fewer than 4 prey are visible the trailing slots get all-zeros and
// is_visible = 0.  The network learns "is_visible = 0 ⇒ ignore the other
// three fields of this slot" from data; no special masking needed.

fn encode_state(
    organism: &Organism,
    transform: &Transform,
    grid: &WorldModelGrid,
) -> [f32; STATE_DIM]:

    let mut s = [0.0; STATE_DIM]

    // Channel 0: own energy.
    s[0] = (organism.energy / organism.max_energy).clamp(0.0, 1.0)

    // Channel 1: own speed.
    s[1] = (organism.movement_speed / MAX_SPEED).clamp(0.0, 1.0)

    // Channels 2..18: top-4 nearest prey, in body frame.
    //
    // Inverse-yaw rotation: world coords → body coords.
    //   body_x =  cos(yaw) · world_x + sin(yaw) · world_z
    //   body_z = -sin(yaw) · world_x + cos(yaw) · world_z
    //
    let yaw   = transform.rotation.to_euler(YXZ).1
    let cos_y = cos(yaw)
    let sin_y = sin(yaw)

    let neighbours = grid.k_nearest(transform.translation.xz(), k = 4)
    for i in 0..4:
        let base = 2 + i * 4
        if let Some(n) = neighbours.get(i):
            let rel = n.position.xz() - transform.translation.xz()
            let body_x =  cos_y * rel.x + sin_y * rel.z
            let body_z = -sin_y * rel.x + cos_y * rel.z
            s[base + 0] = (body_x / WORLD_MODEL_RADIUS).clamp(-1.0, 1.0)
            s[base + 1] = (body_z / WORLD_MODEL_RADIUS).clamp(-1.0, 1.0)
            s[base + 2] = (rel.length() / WORLD_MODEL_RADIUS).clamp(0.0, 1.0)
            s[base + 3] = 1.0
        // else: leave zeros; is_visible = 0 already.

    return s


// ============================================================================
//   4. NETWORK
// ============================================================================
//
// Two hidden layers, ReLU, linear output (Q-values are unbounded).
// He-normal init for the hidden layers (standard for ReLU).
// Tiny uniform init for the output layer so initial Q-values are near zero —
// keeps the first few updates from drifting wildly.
//
// Per-row batched matmul exactly as the existing pool:
//   forward shape:  [BATCH, STATE_DIM] → [BATCH, N_ACTIONS]
//   weights shape:  w1 [N, STATE_DIM, HIDDEN], etc.
//
// `BATCH` is BATCH_PER_SLOT during training, 1 during action selection.

struct QNetMlp:
    w1: Param<Tensor[N, STATE_DIM, HIDDEN]>
    b1: Param<Tensor[N, HIDDEN]>
    w2: Param<Tensor[N, HIDDEN, HIDDEN]>
    b2: Param<Tensor[N, HIDDEN]>
    w3: Param<Tensor[N, HIDDEN, N_ACTIONS]>
    b3: Param<Tensor[N, N_ACTIONS]>

    fn new(device, n):
        // He-normal: std = sqrt(2 / fan_in)
        w1 = NormalInit(std = sqrt(2.0 / STATE_DIM)).init([n, STATE_DIM, HIDDEN], device)
        w2 = NormalInit(std = sqrt(2.0 / HIDDEN   )).init([n, HIDDEN, HIDDEN], device)
        // Small uniform init on the output → Q-values start near zero.
        w3 = UniformInit(-1e-3, +1e-3).init([n, HIDDEN, N_ACTIONS], device)
        b1, b2, b3 = ZeroInit.init(...)
        return Self { w1, b1, w2, b2, w3, b3 }

    // x: [n, BATCH, STATE_DIM]  → out: [n, BATCH, N_ACTIONS]
    // (BATCH = 1 for action selection, BATCH = BATCH_PER_SLOT for training.)
    fn forward(x):
        h = relu(batched_matmul(x, w1) + b1.unsqueeze(BATCH dim))
        h = relu(batched_matmul(h, w2) + b2.unsqueeze(BATCH dim))
        q = batched_matmul(h, w3) + b3.unsqueeze(BATCH dim)
        return q

    // Hard copy of every Param's underlying tensor — for target sync.
    // No autograd link between source and copy.
    fn clone_weights_into(other: &mut QNetMlp):
        other.w1 = self.w1.clone_no_grad()
        ... same for b1, w2, b2, w3, b3.


// ============================================================================
//   5. POOL RESOURCE
// ============================================================================

struct BrainPoolL1Hetero:
    n: usize                                  // = OrganismPoolSize
    online: QNetMlp                           // the network we train
    target: QNetMlp                           // periodically synced copy
    optimizer: AdamOptimizer
    device: CudaDevice

    // Slot bookkeeping (identical to existing pool):
    free: Vec<u32>                            // free-list of slot indices
    map:  HashMap<Entity, u32>                // entity → slot

    // Per-slot "last decision" cache, for building (s, a, r, s') tuples:
    prev_state:  Vec<f32>                     // flat [n * STATE_DIM]
    prev_action: Vec<u32>                     // [n]
    prev_energy: Vec<f32>                     // [n]
    has_prev:    Vec<bool>                    // [n]

    // Per-slot replay buffer (ring), flat for cache friendliness:
    replay_state:      Vec<f32>               // [n * REPLAY_CAPACITY * STATE_DIM]
    replay_next_state: Vec<f32>               // [n * REPLAY_CAPACITY * STATE_DIM]
    replay_action:     Vec<u32>               // [n * REPLAY_CAPACITY]
    replay_reward:     Vec<f32>               // [n * REPLAY_CAPACITY]
    replay_done:       Vec<u8>                // [n * REPLAY_CAPACITY] — 0 or 1
    replay_write_idx:  Vec<u32>               // [n] — next slot index to overwrite
    replay_fill:       Vec<u32>               // [n] — entries valid (≤ REPLAY_CAPACITY)

    // Global counters:
    decisions_made: u32                       // total ε-greedy decisions; drives ε decay
    train_steps:    u32                       // total gradient updates; drives target sync

    rng: SmallRng                             // for ε-greedy + replay sampling


// ============================================================================
//   6. CONSTRUCTION  (matches the existing FromWorld pattern)
// ============================================================================

impl FromWorld for BrainPoolL1Hetero:
    fn from_world(world: &mut World) -> Self:
        let n = world
            .get_resource::<OrganismPoolSize>()
            .map(|r| r.0.max(1))
            .unwrap_or(1)
        let device = CudaDevice::default()
        warmup(&device, n)                    // compile CubeCL kernels at max shape
        let online = QNetMlp::new(&device, n)
        let target = QNetMlp::new(&device, n)
        online.clone_weights_into(&mut target) // target starts == online
        return Self {
            n, online, target,
            optimizer: AdamConfig::new().init(),
            device,
            free:        (0..n as u32).rev().collect(),
            map:         HashMap::new(),
            prev_state:  vec![0.0; n * STATE_DIM],
            prev_action: vec![0;   n],
            prev_energy: vec![0.0; n],
            has_prev:    vec![false; n],
            replay_state:      vec![0.0; n * REPLAY_CAPACITY * STATE_DIM],
            replay_next_state: vec![0.0; n * REPLAY_CAPACITY * STATE_DIM],
            replay_action:     vec![0;   n * REPLAY_CAPACITY],
            replay_reward:     vec![0.0; n * REPLAY_CAPACITY],
            replay_done:       vec![0;   n * REPLAY_CAPACITY],
            replay_write_idx:  vec![0;   n],
            replay_fill:       vec![0;   n],
            decisions_made:    0,
            train_steps:       0,
            rng: SmallRng::seed_from_u64(0xAEONS),
        }


// ============================================================================
//   7. SLOT LIFECYCLE  (PreUpdate systems, chained)
// ============================================================================

fn assign_brains_l1_hetero(
    mut pool: NonSendMut<BrainPoolL1Hetero>,
    added:    Query<Entity, Added<Heterotroph>>,
    mut commands: Commands,
):
    for entity in added:
        if let Some(slot) = pool.free.pop():
            let s = slot as usize
            pool.map.insert(entity, slot)
            // Reset per-slot scratch; weights stay (inherited from previous
            // tenant — gives "instinct" via prior training).
            pool.has_prev[s] = false
            pool.replay_fill[s] = 0
            pool.replay_write_idx[s] = 0
            pool.prev_energy[s] = 0.0
            commands.entity(entity).try_insert(BrainSlotL1Hetero(slot))

fn free_brains_l1_hetero(
    mut pool: NonSendMut<BrainPoolL1Hetero>,
    mut removed: RemovedComponents<Heterotroph>,
):
    for entity in removed.read():
        if let Some(slot) = pool.map.remove(&entity):
            pool.free.push(slot)


// ============================================================================
//   8. BRAIN TICK  (Update, on_timer = BRAIN_TICK_INTERVAL)
// ============================================================================
//
// Each tick, for every live heterotroph:
//   (a) encode current state                                   — §3
//   (b) if we have a prev (s, a), build (s, a, r, s', done) and
//       push to that slot's replay buffer                      — §8
//   (c) ε-greedy action selection                              — §8
//   (d) apply action: write movement_direction + movement_speed
//   (e) update prev_* cache
//
// After the per-organism loop:
//   (f) one mini-batch gradient step                           — §9
//   (g) target-network sync if it's time                       — §9
//
// Net work per tick: N forward passes (action selection) + 1 mini-batch
// forward+backward+step. With burn-cuda batched matmul this is one GPU
// call for selection (shape [n, 1, STATE_DIM]) and one for training
// (shape [n, BATCH_PER_SLOT, STATE_DIM]).

fn apply_intelligence_level_1_hetero(
    mut pool: NonSendMut<BrainPoolL1Hetero>,
    mut q:    Query<(&mut Organism, &Transform, &BrainSlotL1Hetero)>,
    grid:     Res<WorldModelGrid>,
):
    let n = pool.n
    // ── (a, b, c, d, e): per-organism loop. ────────────────────────────
    //
    // Build a [n, 1, STATE_DIM] input tensor for action selection in ONE
    // GPU call rather than n separate forwards. We assemble it on the
    // CPU as a flat vec and upload once.
    //
    let mut act_input = vec![0.0; n * STATE_DIM]
    let mut active    = vec![false; n]          // slots with a live organism this tick

    for (organism, transform, slot_comp) in q.iter():
        let s = slot_comp.0 as usize
        if s >= n: continue
        active[s] = true

        let cur_state = encode_state(&organism, &transform, &grid)

        // (b) Build transition for the PREVIOUS decision and push to replay.
        if pool.has_prev[s]:
            let r = (organism.energy - pool.prev_energy[s]).clamp(-1.0, 1.0)
            let done = (organism.energy <= 0.0) as u8
            push_transition(
                &mut pool, s,
                &pool.prev_state[s * STATE_DIM .. (s+1) * STATE_DIM],
                pool.prev_action[s],
                r,
                &cur_state,
                done,
            )

        // Stage current state for the action-selection forward pass.
        act_input[s * STATE_DIM .. (s+1) * STATE_DIM].copy_from_slice(&cur_state)

        // Cache for the NEXT tick's transition (we still need to fill
        // prev_action below once we've chosen one).
        pool.prev_state[s * STATE_DIM .. (s+1) * STATE_DIM].copy_from_slice(&cur_state)
        pool.prev_energy[s] = organism.energy

    // ── Single batched forward for action selection. ───────────────────
    //
    // act_input: [n, STATE_DIM] → reshape to [n, 1, STATE_DIM]
    // q_values:  [n, 1, N_ACTIONS] → squeeze to [n, N_ACTIONS]
    //
    let input_t  = Tensor::from(act_input, [n, 1, STATE_DIM], pool.device)
    let q_values = pool.online.forward(input_t).squeeze(dim=1) // [n, N_ACTIONS]
    let q_cpu    = q_values.into_data().to_vec::<f32>()         // CPU mirror

    // Current ε (linear decay, capped at floor).
    let progress = (pool.decisions_made as f32 / EPSILON_DECAY_STEPS as f32).min(1.0)
    let epsilon  = EPSILON_START + progress * (EPSILON_END - EPSILON_START)

    // ── (c, d, e): assign action to each live organism. ────────────────
    for (mut organism, _, slot_comp) in q.iter_mut():
        let s = slot_comp.0 as usize
        if s >= n || !active[s]: continue

        let action: u32 = if pool.rng.gen::<f32>() < epsilon:
            pool.rng.gen_range(0..N_ACTIONS as u32)
        else:
            let row = &q_cpu[s * N_ACTIONS .. (s+1) * N_ACTIONS]
            argmax(row) as u32

        // (d) Apply.
        if action == 8:
            organism.movement_speed = 0.0
            // movement_direction unchanged
        else:
            let (dx, dz) = ACTION_DIRS[action as usize]
            organism.movement_direction = Vec3::new(dx, 0.0, dz)
            organism.movement_speed     = MAX_SPEED

        // (e) Cache action for next tick's transition.
        pool.prev_action[s] = action
        pool.has_prev[s]    = true
        pool.decisions_made = pool.decisions_made.saturating_add(1)

    // ── (f) Training step. ──────────────────────────────────────────────
    train_dqn(&mut pool)

    // ── (g) Target sync. ────────────────────────────────────────────────
    if pool.train_steps >= TARGET_SYNC_STEPS:
        pool.online.clone_weights_into(&mut pool.target)
        pool.train_steps = 0


// ============================================================================
//   9. REPLAY + TRAINING
// ============================================================================

fn push_transition(
    pool:       &mut BrainPoolL1Hetero,
    slot:       usize,
    state:      &[f32; STATE_DIM],
    action:     u32,
    reward:     f32,
    next_state: &[f32; STATE_DIM],
    done:       u8,
):
    let i = pool.replay_write_idx[slot] as usize
    let s_base = slot * REPLAY_CAPACITY * STATE_DIM + i * STATE_DIM
    pool.replay_state    [s_base .. s_base + STATE_DIM].copy_from_slice(state)
    pool.replay_next_state[s_base .. s_base + STATE_DIM].copy_from_slice(next_state)
    let scalar_idx = slot * REPLAY_CAPACITY + i
    pool.replay_action[scalar_idx] = action
    pool.replay_reward[scalar_idx] = reward
    pool.replay_done  [scalar_idx] = done

    pool.replay_write_idx[slot] = ((i + 1) % REPLAY_CAPACITY) as u32
    if (pool.replay_fill[slot] as usize) < REPLAY_CAPACITY:
        pool.replay_fill[slot] += 1


fn train_dqn(pool: &mut BrainPoolL1Hetero):
    // ── Build mini-batch on CPU. ────────────────────────────────────────
    //
    // For each slot, sample BATCH_PER_SLOT indices uniformly from
    // [0, replay_fill[slot]). Slots without MIN_REPLAY samples contribute
    // zeros and are masked out of the loss.
    //
    // Final shapes (all flat):
    //   s_batch   : [n * BATCH_PER_SLOT * STATE_DIM]
    //   ns_batch  : [n * BATCH_PER_SLOT * STATE_DIM]
    //   a_batch   : [n * BATCH_PER_SLOT]
    //   r_batch   : [n * BATCH_PER_SLOT]
    //   d_batch   : [n * BATCH_PER_SLOT]
    //   mask_batch: [n * BATCH_PER_SLOT]   (1.0 if slot trained this tick, else 0.0)

    let n = pool.n
    let s_batch:  Vec<f32> = zeros(n * BATCH_PER_SLOT * STATE_DIM)
    let ns_batch: Vec<f32> = zeros(n * BATCH_PER_SLOT * STATE_DIM)
    let a_batch:  Vec<u32> = zeros(n * BATCH_PER_SLOT)
    let r_batch:  Vec<f32> = zeros(n * BATCH_PER_SLOT)
    let d_batch:  Vec<f32> = zeros(n * BATCH_PER_SLOT)
    let mask:     Vec<f32> = zeros(n * BATCH_PER_SLOT)

    let mut any_active = false
    for s in 0..n:
        if pool.replay_fill[s] < MIN_REPLAY as u32: continue
        any_active = true
        let fill = pool.replay_fill[s] as usize
        for b in 0..BATCH_PER_SLOT:
            let i = pool.rng.gen_range(0..fill)
            // Copy state[s][i] into s_batch[s][b], etc.
            let src_s = s * REPLAY_CAPACITY * STATE_DIM + i * STATE_DIM
            let dst_s = s * BATCH_PER_SLOT  * STATE_DIM + b * STATE_DIM
            s_batch [dst_s .. dst_s + STATE_DIM].copy_from(&pool.replay_state    [src_s .. src_s + STATE_DIM])
            ns_batch[dst_s .. dst_s + STATE_DIM].copy_from(&pool.replay_next_state[src_s .. src_s + STATE_DIM])
            let src_scalar = s * REPLAY_CAPACITY + i
            let dst_scalar = s * BATCH_PER_SLOT  + b
            a_batch[dst_scalar] = pool.replay_action[src_scalar]
            r_batch[dst_scalar] = pool.replay_reward[src_scalar]
            d_batch[dst_scalar] = pool.replay_done  [src_scalar] as f32
            mask  [dst_scalar] = 1.0

    if !any_active: return                  // not enough data anywhere yet

    // ── Upload to GPU as [n, BATCH_PER_SLOT, ...]. ─────────────────────
    let s_t  = Tensor::from(s_batch,  [n, BATCH_PER_SLOT, STATE_DIM], pool.device)
    let ns_t = Tensor::from(ns_batch, [n, BATCH_PER_SLOT, STATE_DIM], pool.device)
    let a_t  = Tensor::from(a_batch,  [n, BATCH_PER_SLOT],            pool.device)  // int64 indices
    let r_t  = Tensor::from(r_batch,  [n, BATCH_PER_SLOT],            pool.device)
    let d_t  = Tensor::from(d_batch,  [n, BATCH_PER_SLOT],            pool.device)
    let m_t  = Tensor::from(mask,     [n, BATCH_PER_SLOT],            pool.device)

    // ── Q(s, a) from the ONLINE network (carries gradient). ────────────
    //
    // q_pred:    [n, BATCH_PER_SLOT, N_ACTIONS]
    // q_pred_a:  [n, BATCH_PER_SLOT]               — gathered at the action index
    //
    let q_pred   = pool.online.forward(s_t)
    let q_pred_a = gather(q_pred, dim = -1, index = a_t.unsqueeze(-1)).squeeze(-1)

    // ── max_a' Q_target(s', a') from the TARGET network. NO GRADIENT. ──
    //
    // Computed inside a no-grad scope; the result is detached before use.
    //
    let q_next_max = no_grad:
        let q_next = pool.target.forward(ns_t)                     // [n, B, N_ACTIONS]
        q_next.max(dim = -1).detach()                              // [n, B]

    // ── TD target. (1 − done) zeros the bootstrap on terminal transitions.
    let td_target = r_t + GAMMA * q_next_max * (1.0 - d_t)
    let td_target = td_target.detach()                              // belt and braces

    // ── Huber loss per sample (robust to predation reward spikes). ─────
    //
    //   L(e) = ½·e²            if |e| < δ
    //        = δ·(|e| − ½·δ)   otherwise
    //
    let td_err   = q_pred_a - td_target                             // [n, B]
    let abs_err  = td_err.abs()
    let quad     = 0.5 * td_err.powi(2)
    let linear   = HUBER_DELTA * (abs_err - 0.5 * HUBER_DELTA)
    let per_sample = where(abs_err < HUBER_DELTA, quad, linear)     // [n, B]

    // ── Mask + reduce. Each row independent; loss is the mean over
    //    active (slot, batch-idx) pairs. Dividing by m_t.sum() (not by
    //    n × BATCH_PER_SLOT) keeps the effective gradient magnitude
    //    independent of how many slots are currently trainable. ──────
    let masked = per_sample * m_t
    let denom  = m_t.sum().clamp(1.0, infinity)                     // avoid div-by-0
    let loss   = masked.sum() / denom

    // ── Gradient step (with norm clipping). ────────────────────────────
    let grads = loss.backward()
    let grads = grads.clip_norm(GRAD_CLIP_NORM)                     // standard DQN trick
    pool.online = pool.optimizer.step(LR, pool.online, grads)
    pool.train_steps = pool.train_steps.saturating_add(1)


// ============================================================================
//  10. WARMUP  (CubeCL kernel pre-compile, identical pattern to current pool)
// ============================================================================

fn warmup(device: &CudaDevice, n: usize):
    // Construct dummy online + target nets and one mini-batch at the
    // maximum shape, run one forward + Huber loss + backward + step.
    // CubeCL caches every kernel the hot path will invoke.
    let online = QNetMlp::new(device, n)
    let mut optimizer = AdamConfig::new().init()
    let s_t  = Tensor::zeros([n, BATCH_PER_SLOT, STATE_DIM], device)
    let ns_t = Tensor::zeros([n, BATCH_PER_SLOT, STATE_DIM], device)
    let a_t  = Tensor::zeros([n, BATCH_PER_SLOT], device)
    let r_t  = Tensor::zeros([n, BATCH_PER_SLOT], device)
    let d_t  = Tensor::zeros([n, BATCH_PER_SLOT], device)
    let m_t  = Tensor::ones ([n, BATCH_PER_SLOT], device)
    // ... same loss block as train_dqn ...
    let _ = optimizer.step(LR, online, loss.backward())


// ============================================================================
//  11. SAVE / LOAD
// ============================================================================
//
// The on-disk snapshot needs three weight matrices per slot:
//     w1 [STATE_DIM × HIDDEN], b1 [HIDDEN],
//     w2 [HIDDEN × HIDDEN],    b2 [HIDDEN],
//     w3 [HIDDEN × N_ACTIONS], b3 [N_ACTIONS]
//
// Drop the old PoolSnapshot's policy / value-head fields; replace with the
// three Q-net layers. The replay buffer is NOT serialised — a restored
// organism starts with empty replay and re-fills from new experience.
// The target network is also not serialised — initialised to == online on
// load.


// ============================================================================
//  12. INTEGRATION CHECKLIST
// ============================================================================
//
// To turn this design into compilable Rust:
//
//   1. Translate this file into `intelligence_level_1_hetero.rs`, deleting
//      the existing REINFORCE/A2C body. Keep the file path and the
//      `BrainPoolL1Hetero`, `BrainSlotL1Hetero` type names so other modules
//      (`colony.rs` save/load, `behaviour.rs` schedule, `rl_helpers.rs`)
//      don't need renaming.
//
//   2. `behaviour.rs::BehaviourPlugin::build` already calls
//      `init_non_send_resource::<BrainPoolL1Hetero>()` and registers the
//      `assign_brains_l1_hetero` / `free_brains_l1_hetero` /
//      `apply_intelligence_level_1_hetero` systems with the right schedule
//      placement. No changes needed.
//
//   3. `apply_movement` already consumes `movement_direction` and
//      `movement_speed`. No changes needed.
//
//   4. Delete reward shaping in this file. The single reward channel is
//      `r = clamp(Δenergy, ±1)`. Predation events produce positive spikes;
//      everything else is ~0. Q-learning + replay handles the sparsity.
//
//   5. Delete `BrainOptL1Hetero` and the value head. DQN has no policy
//      head and no value head — just Q.
//
//   6. Update `PoolSnapshot` / `BrainRestore` in `rl_helpers.rs` to the
//      new field set (§11 above).
//
//   7. Update CLAUDE.md to replace the REINFORCE-with-baseline narrative
//      under "Behaviour: two intelligence levels" with a DQN description.
//
//   8. Photoautotroph pool (`intelligence_level_1_photo.rs`) is OUT OF
//      SCOPE for this rewrite. It can keep its current oracle-MSE
//      approach (the audit didn't cover it because the user's question
//      was about prey pursuit specifically).
//
//
// Expected behaviour after rewrite, with the new cohort settings
// (500 photo + 80 hetero, AI_TRAINING_MODE = false):
//
//   • First ~1-2 minutes: random walks (ε ≈ 1.0). Heterotrophs starve
//     and reproduce; selection pressure drives early policies.
//   • 2-30 minutes: replay buffers fill (≥ MIN_REPLAY per slot). ε
//     decays linearly toward 0.05 over ~100k decisions. First Bellman
//     updates start refining Q-values around predation events.
//   • 30+ minutes: visible pursuit behaviour emerges. Heterotrophs that
//     happen to inherit good Q-rows from culled ancestors converge faster.
//
//   • If pursuit does NOT emerge after ~2 hours, suspect (in order):
//       a) prey too sparse in the world → reduce WORLD_MODEL_RADIUS or
//          increase INITIAL_PHOTOAUTOTROPHS.
//       b) γ too high — try γ = 0.9 for shorter credit-assignment horizon.
//       c) ε floor too high — reduce EPSILON_END to 0.01.
//       d) BATCH_PER_SLOT too small — try 64.
//
// ============================================================================
//                              END OF DESIGN
// ============================================================================
