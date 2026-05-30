use crate::cell::*;
use crate::frontend::ShowGizmo;
use crate::volumetric_growth::{build_mesh_from_ocg, build_smoothed_mesh_from_ocg};
use crate::world_geometry::{HeightmapSampler, MapSize};
use crate::movement::DirectionTimer;
use bevy::prelude::*;
use rand::prelude::*;

// Re-export the Organism module so existing `use crate::colony::*;` imports
// keep finding Organism, OrganismRoot, OrganismKind, Photoautotroph, etc.
pub use crate::organism::*;

// Initial cohort sizes are derived at spawn-time from three
// resources (all launcher-set, defaulting to the matching
// `DEFAULT_*` constants):
//   * herbivores        = StartHeterotrophs    (launcher: "Start Heterotroph Number")
//   * photoautotrophs   = MaxOrganisms − MaxHerbivores
// The running-population cap is `MaxOrganisms` (and herbivore-only
// cap is `MaxHerbivores`), both editable at runtime via the
// statistics panel. Keeping the START count decoupled from the
// CAP lets the user seed a small starter cohort and let reproduction
// fill the room up to the cap. See `spawn_colony`.

/// Initial Krishi cohort size. `pub` so `krishi.rs` reads it directly —
/// keeps every "how many of X spawn at startup" knob in one place.
pub const INITIAL_KRISHI: u32 = 1;

/// Angular damping for the BASE body of a limb-based organism. High,
/// because the base has no PD actuator of its own — joint-constraint
/// reaction torques from the limbs would otherwise integrate unbounded
/// and spin the body up.
const BASE_ANGULAR_DAMPING: f32 = 3.0;

/// Angular damping for LIMB bodies. Much lower than the base so the
/// PD controller can produce dynamic swings — the policy can't lift
/// the legs off the ground if every torque it commands gets drained
/// to friction within a frame.
const LIMB_ANGULAR_DAMPING: f32 = 1.0;

/// Linear damping for every limb-based body part (base + limbs).
/// Light — enough to bleed drift between actuator pulses, not enough
/// to lock the organism in place.
const LIMB_LINEAR_DAMPING:  f32 = 0.2;

/// Material density used when deriving each limb-based body part's
/// mass from its compound collider. Density × volume → mass; lower
/// density → lower mass → lower normal force at ground contacts →
/// less friction force resisting limb rotation. Set to 0.2 (down
/// from the natural 1.0) because at full density a 25-cell organism
/// sitting on the heightfield was friction-pinned at every ground
/// contact, dwarfing the brain's PD torques.
const LIMB_BODY_DENSITY: f32 = 0.2;

/// Friction coefficient applied to every limb-based body part.
///
/// Locomotion needs GRIP, not slip: to "press a foot down and back to
/// drive the body forward" the planted foot must not slide. An earlier
/// pass set this to 0.05 with a `Min` combine rule to stop limbs
/// sticking — but that starved the organism of traction (feet slid,
/// body never propelled), and a test run confirmed near-zero motion.
/// Raised to 1.0 with an `Average` combine rule, so a limb↔terrain
/// (default μ = 0.5) contact resolves to μ = 0.75 — firm grip when
/// planted. This does NOT prevent lifting: a foot lifted off the
/// ground has zero normal force and therefore zero friction
/// regardless of μ, so the lift→reposition→plant→press gait cycle
/// works (the swing phase is frictionless for free).
const LIMB_FRICTION_COEFFICIENT: f32 = 1.0;


/// Compute the (parent_anchor, limb_anchor) pair for a limb's
/// `SphericalJoint`, placing the pivot at the FACE MIDPOINT between
/// the parent cell adjacent to the attachment and the limb's first
/// cell. With the limb rebased so its first cell sits at the limb's
/// local origin, this gives:
///
///   * `anchor1` = `(parent_cell_pos + pivot) / 2`
///       — point on the parent body, on the parent-cell ↔ limb-first-
///       cell shared rhombic face.
///   * `anchor2` = `(parent_cell_pos − pivot) / 2`
///       — same point expressed in the limb's local frame; the limb's
///       first cell sits at `(0, 0, 0)` and the anchor lies between
///       the limb origin and the (offset) direction of the parent
///       cell.
///
/// The two anchors coincide in world space so the cells sit on
/// opposite sides of the joint — like a shoulder joint at the body
/// surface rather than a rotation around the limb's own centroid.
///
/// "Parent cell adjacent to the attachment" is the parent cell
/// closest to `pivot`; by construction of the species-editor's
/// candidate-driven placement, that's exactly one RD lattice step
/// away. Falls back to `Vec3::ZERO` when the parent body has no
/// cells (defensive — should never happen for a valid attachment).
fn limb_joint_anchors(
    parent_cells: &[Cell],
    pivot:        Vec3,
) -> (Vec3, Vec3) {
    let parent_cell_pos = parent_cells.iter()
        .map(|c| c.local_pos)
        .min_by(|a, b| {
            a.distance_squared(pivot)
                .partial_cmp(&b.distance_squared(pivot))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(Vec3::ZERO);
    let anchor1 = (parent_cell_pos + pivot) * 0.5;
    let anchor2 = (parent_cell_pos - pivot) * 0.5;
    (anchor1, anchor2)
}


pub struct ColonyPlugin;

impl Plugin for ColonyPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SaveRequested>();
        // `init_resource` is idempotent — it only inserts when no
        // resource of this type already exists. main.rs sets the load
        // path BEFORE add_plugins runs, so this is just the no-load
        // fallback for callers that wire ColonyPlugin without setting
        // a path. Using `insert_resource` here would unconditionally
        // overwrite main.rs's value.
        app.init_resource::<ColonyLoadPath>();
        app.init_resource::<crate::simulation_settings::AutoSpawnHeteros>();
        app.init_resource::<crate::simulation_settings::StartHeterotrophs>();
        app.init_resource::<crate::simulation_settings::StartPhotoautotrophs>();
        app.init_resource::<crate::simulation_settings::MinHeteroCount>();
        app.init_resource::<crate::simulation_settings::MinHeteroCountEditState>();
        app.init_resource::<AutosaveTimer>();
        app.init_resource::<crate::dataset_export::ExportDatasetRequested>();
        app.init_resource::<crate::dataset_export::AutoExportSchedule>();
        // Rotate prior-run dataset artefacts ONCE at process start,
        // before any auto-export fires (the earliest milestone is
        // 5 virtual minutes, so the Startup schedule has all the
        // time it needs).
        app.add_systems(Startup, |_w: &mut World| {
            crate::dataset_export::rotate_existing_datasets();
        });
        app.init_resource::<crate::time_series_log::TimeSeriesLogger>();
        app.add_systems(Update, spawn_colony.run_if(resource_exists::<HeightmapSampler>));
        app.add_systems(Update, animate_limbs);
        app.add_systems(Update, save_colony_system);
        app.add_systems(Update, autosave_system);
        app.add_systems(Update, auto_spawn_heteros);
        app.add_systems(Update, crate::dataset_export::tick_auto_export_schedule);
        app.add_systems(Update, crate::dataset_export::export_dataset_system);
        app.add_systems(Update, crate::time_series_log::tick_time_series_logger);
        // Default lineage record attachment. Fires once per organism
        // — on the tick AFTER spawn (Added<OrganismRoot> filter).
        // Offspring of reproduction already get a LineageRecord
        // explicitly inserted by `reproduction.rs` (with `parent_id`
        // populated), so the `Without<LineageRecord>` filter skips
        // them here.
        app.add_systems(Update, assign_lineage_records);
    }
}


/// Default attach: every OrganismRoot that doesn't already have a
/// `LineageRecord` gets one minted as `new_initial` (parent_id =
/// None, spawn_time = current virtual time). Covers initial cohort,
/// auto-spawn, editor placement, and loaded organisms. Reproduction
/// offspring already carry a `new_offspring` record by the time
/// this runs.
fn assign_lineage_records(
    mut commands:  Commands,
    virtual_time:  Res<Time<Virtual>>,
    new_q:         Query<Entity, (Added<OrganismRoot>, Without<LineageRecord>)>,
) {
    let t = virtual_time.elapsed_secs();
    for e in &new_q {
        commands.entity(e).try_insert(LineageRecord::new_initial(t));
    }
}


/// Optional path to a `.colony` file. When `Some(path)`, the very first
/// `spawn_colony` invocation reads that file and spawns the saved
/// organisms instead of generating a fresh colony. When `None`, the
/// fresh-colony path runs as before. Set by `main.rs` from the second
/// positional CLI argument; defaults to `None` (`Option::default()`)
/// when no caller provides a value.
#[derive(Resource, Default)]
pub struct ColonyLoadPath(pub Option<String>);


// ── Colony save ──────────────────────────────────────────────────────────────
//
// The binary file written by `save_colony_system` uses the layout below.
// All multi-byte values are little-endian; booleans and `u8`-sized enums
// take exactly 1 byte. The format starts with an 8-byte magic so a future
// loader can refuse to read garbage:
//
//   "AEONS001"                                  (8 bytes magic + version)
//   u32  organism_count
//   for each organism:
//       u8   kind                                (0 = Photo, 1 = Hetero)
//       3×f32 position                           (Transform.translation)
//       4×f32 rotation                           (Transform.rotation, xyzw)
//       f32  energy
//       u8   in_sunlight
//       u8   reproduced
//       u8   reproductions
//       f32  movement_speed
//       3×f32 movement_direction
//       3×f32 velocity
//       u8   is_climbing
//       f32  climb_energy_debt
//       i32  photo_cell_count
//       i32  non_photo_cell_count
//       u8   symmetry                            (0 = NoSymmetry, 1 = Bilateral)
//       u8   is_sessile
//       u8   has_variable_form
//       u32  body_part_count
//       for each body_part:
//           u8   kind                            (0 = Body, 1 = Limb, 2 = Organ)
//           3×f32 local_offset
//           u8   consumed
//           u8   debug_blue
//           u8   regrowable
//           u8   attachment_present              (0 or 1)
//           if attachment_present:
//               u32  parent_idx
//               3×f32 origin_local
//               4×f32 attachment_rotation
//           u32  cell_count
//           for each cell:
//               3×f32 local_pos
//               u8    cell_type                  (0 = Photo, 1 = NonPhoto)
//               f32   cell_energy
//               u8    neighbour_count
//           u32  ocg_count
//           for each ocg entry:
//               u32   idx
//               3×f32 pos
//               u8    cell_type
//
// Cached `PhotosyntheticCell` data is intentionally NOT serialised — it is
// fully derivable from `cell_type` + `neighbour_count`, so a future loader
// just calls `physiology::recompute_body_parts` after rehydrating.

/// Current save format magic.
///
/// Version history:
///   * v003 — appends a per-organism brain section (weights + REINFORCE
///     prev_* state) so loaded colonies resume in *exactly* the
///     state they were saved in, neural-network state included.
///   * v002 — added the 1-byte `intelligence_level` field after
///     `has_variable_form`. No brain weights.
///   * v001 — original. No `intelligence_level`, no brain weights.
///
/// All older versions are still loadable. The loader synthesises any
/// missing fields from the deterministic spawn-time rules; missing
/// brain sections leave each organism's slot at the recycled / default
/// initial weights, so behaviour starts un-trained but the colony
/// composition (positions, body parts, energy) is restored correctly.
/// Save format magic. v004 (current) carries the new per-organism
/// herbivore_1 A2C brain — backbone + actor + critic, 12 weight
/// tensors per slot, plus the REINFORCE prev-state / prev-action /
/// prev-dopamine / prev-target-distance scalars. v003 carried the
/// old single-MLP REINFORCE brain (4 weight tensors); the new code
/// can still read v003 organism structure (positions, body parts,
/// energy) but drops the v003 brain block — the saved weights are
/// for a different architecture and aren't restorable.
const SAVE_MAGIC:             &[u8;8] = b"AEONS007";
const SAVE_MAGIC_LEGACY_V006: &[u8;8] = b"AEONS006";
const SAVE_MAGIC_LEGACY_V005: &[u8;8] = b"AEONS005";
const SAVE_MAGIC_LEGACY_V004: &[u8;8] = b"AEONS004";
const SAVE_MAGIC_LEGACY_V003: &[u8;8] = b"AEONS003";
const SAVE_MAGIC_LEGACY_V002: &[u8;8] = b"AEONS002";
const SAVE_MAGIC_LEGACY_V001: &[u8;8] = b"AEONS001";

/// One-shot resource: when set to `Some(path)`, `save_colony_system`
/// writes the current world to that path on the next Update tick and
/// resets the resource to `None`.
///
/// Two producers set this:
///   * The Save button in the statistics panel opens an rfd
///     "Save As" dialog and stores the user's chosen path.
///   * `autosave_system` (also in `colony.rs`) fires on a real-time
///     timer and stores an `autosaves/autosave_<timestamp>.colony`
///     path.
///
/// If both fire on the same tick the later one wins. Acceptable —
/// autosaves are 5 minutes apart by default, collisions are vanishingly
/// rare and harmless either way.
#[derive(Resource, Default)]
pub struct SaveRequested(pub Option<std::path::PathBuf>);


fn save_colony_system(
    mut save_requested: ResMut<SaveRequested>,
    organisms: Query<
        (Entity, &Transform, &Organism,
         Has<Photoautotroph>, Has<Heterotroph>, Has<Carnivore>),
        With<OrganismRoot>,
    >,
    // The herbivore_1 pool is the only one that contributes a brain
    // block in v004. The L1-photo / L1-hetero / L2 / L3 pool stubs
    // are kept registered as `NonSendResource` for legacy reasons
    // but are not consulted during save — their slots have no
    // routable organisms.
    pool: NonSend<crate::intelligence_level_herbivore_1_sliding::BrainPoolHerbivore1>,
    // v007 addition — limb-based brain pools. One snapshot per pool,
    // taken once before iterating organisms (each does a small batch
    // of GPU→CPU syncs). Each organism with a `BrainSlotX_Limb`
    // contributes one limb-brain block to its record.
    pool_limb_h:  NonSend<crate::intelligence_level_herbivore_1_limb::BrainPoolHerbivore1Limb>,
    pool_limb_l2: NonSend<crate::intelligence_level_2_limb::BrainPoolL2Limb>,
    pool_limb_l3: NonSend<crate::intelligence_level_3_limb::BrainPoolL3Limb>,
) {
    let Some(target_path) = save_requested.0.take() else { return };

    let mut buf: Vec<u8> = Vec::with_capacity(64 * 1024);
    buf.extend_from_slice(SAVE_MAGIC);

    // v007: one limb-pool snapshot per pool, reused across all organisms.
    let snap_h  = pool_limb_h.0.snapshot();
    let snap_l2 = pool_limb_l2.0.snapshot();
    let snap_l3 = pool_limb_l3.0.snapshot();

    // Two-pass to write the count up front. iter() over Query is cheap.
    let count = organisms.iter()
        .filter(|(_, _, _, is_photo, is_hetero, _)| *is_photo || *is_hetero)
        .count() as u32;
    put_u32(&mut buf, count);

    for (entity, transform, org, is_photo, is_hetero, is_carn) in organisms.iter() {
        let kind: u8 = if is_photo { 0 } else if is_hetero { 1 } else { continue };

        put_u8(&mut buf, kind);
        put_vec3(&mut buf, transform.translation);
        put_quat(&mut buf, transform.rotation);
        put_f32(&mut buf, org.energy);
        put_u8(&mut buf, org.in_sunlight as u8);
        put_u8(&mut buf, org.reproduced as u8);
        put_u8(&mut buf, org.reproductions);
        put_f32(&mut buf, org.movement_speed);
        put_vec3(&mut buf, org.movement_direction);
        put_vec3(&mut buf, org.velocity);
        put_u8(&mut buf, org.is_climbing as u8);
        put_f32(&mut buf, org.climb_energy_debt);
        put_i32(&mut buf, org.photo_cell_count);
        put_i32(&mut buf, org.non_photo_cell_count);
        put_u8(&mut buf, match org.symmetry {
            Symmetry::NoSymmetry => 0,
            Symmetry::Bilateral  => 1,
        });
        put_u8(&mut buf, org.is_sessile as u8);
        put_u8(&mut buf, org.has_variable_form as u8);
        // v006 addition — movement paradigm.
        put_u8(&mut buf, org.sliding_movement as u8);
        // v002 addition — saved so loaded organisms keep their
        // assigned intelligence level instead of being re-rolled.
        put_u8(&mut buf, match org.intelligence_level {
            IntelligenceLevel::Level0 => 0,
            IntelligenceLevel::Level1 => 1,
            IntelligenceLevel::Level2 => 2,
            IntelligenceLevel::Level3 => 3,
        });

        put_u32(&mut buf, org.body_parts.len() as u32);
        for bp in &org.body_parts {
            put_u8(&mut buf, match bp.kind {
                BodyPartKind::Body  => 0,
                BodyPartKind::Limb  => 1,
                BodyPartKind::Organ => 2,
            });
            put_vec3(&mut buf, bp.local_offset);
            put_u8(&mut buf, bp.consumed   as u8);
            put_u8(&mut buf, bp.debug_blue as u8);
            put_u8(&mut buf, bp.regrowable as u8);

            match &bp.attachment {
                Some(att) => {
                    put_u8(&mut buf, 1);
                    put_u32(&mut buf, att.parent_idx as u32);
                    put_vec3(&mut buf, att.origin_local);
                    put_quat(&mut buf, att.rotation);
                }
                None => put_u8(&mut buf, 0),
            }

            put_u32(&mut buf, bp.cells.len() as u32);
            for cell in &bp.cells {
                put_vec3(&mut buf, cell.local_pos);
                put_u8(&mut buf, cell_type_byte(cell.cell_type));
                put_f32(&mut buf, cell.cell_energy);
                put_u8(&mut buf, cell.neighbour_count);
            }

            put_u32(&mut buf, bp.ocg.len() as u32);
            for (idx, pos, ct) in &bp.ocg {
                put_u32(&mut buf, *idx as u32);
                put_vec3(&mut buf, *pos);
                put_u8(&mut buf, cell_type_byte(*ct));
            }
        }

        // ── v004 brain section ───────────────────────────────────
        // Only `Level1 + Heterotroph + !Carnivore` organisms have a
        // brain in the new pool. Everyone else (photoautotrophs,
        // carnivores, Krishi, etc.) emits `brain_present = 0`. The
        // 12 weight tensors + REINFORCE state are pulled per-slot
        // from the GPU. The loader's hard-error policy means saved
        // tensor lengths must match the current architecture
        // exactly (see HEADER_VERSION + restore_slot checks).
        let is_l1_herbivore = matches!(org.intelligence_level, IntelligenceLevel::Level1)
                              && is_hetero
                              && !is_carn;
        let brain_data = if is_l1_herbivore {
            pool.map.get(&entity).map(|&slot| pool.extract_slot(slot))
        } else {
            None
        };
        match brain_data {
            None => put_u8(&mut buf, 0),
            Some(b) => {
                put_u8(&mut buf, 1);
                // Format shared with .species v3 — see
                // `intelligence_level_herbivore_1_sliding::encode_brain_restore`.
                crate::intelligence_level_herbivore_1_sliding::encode_brain_restore(&mut buf, &b);
            }
        }

        // ── v007 limb-brain block. ────────────────────────────────────
        // For limb-based heterotrophs, find which limb pool this
        // organism is enrolled in (by intelligence_level + carnivore
        // marker), extract its weights, write a kind tag + payload.
        // Non-limb organisms write a single zero byte.
        //
        // Kind tags: 0 = none, 1 = herbivore_1_limb, 2 = l2_limb,
        // 3 = l3_limb. Older readers (≤ v006) stop before this byte
        // and never see it.
        let limb_brain = if !org.sliding_movement && is_hetero {
            match org.intelligence_level {
                IntelligenceLevel::Level1 if !is_carn =>
                    snap_h.extract(entity).map(|b| (1u8, b)),
                IntelligenceLevel::Level2 =>
                    snap_l2.extract(entity).map(|b| (2u8, b)),
                IntelligenceLevel::Level3 =>
                    snap_l3.extract(entity).map(|b| (3u8, b)),
                _ => None,
            }
        } else {
            None
        };
        match limb_brain {
            None => put_u8(&mut buf, 0),
            Some((kind, b)) => {
                put_u8(&mut buf, kind);
                crate::limb_ppo::encode_brain_restore_limb(&mut buf, &b);
            }
        }
        let _ = is_carn;   // referenced only above; silence unused
    }

    // Ensure the parent directory exists (autosave path is
    // `autosaves/...`, which may not have been created yet).
    if let Some(parent) = target_path.parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                error!("failed to create save directory {}: {}", parent.display(), e);
                return;
            }
        }
    }

    match std::fs::write(&target_path, &buf) {
        Ok(())  => info!("colony saved to {} — {} organisms, {} bytes",
                        target_path.display(), count, buf.len()),
        Err(e)  => error!("failed to save colony to {}: {}",
                          target_path.display(), e),
    }
}


// ── Autosave ────────────────────────────────────────────────────────────────
//
// Real-time timer-driven autosave. Every `AUTOSAVE_INTERVAL_SECS` of
// wall-clock time (not virtual time — backups should keep firing at a
// predictable cadence even at 10× sim speed or while paused-but-running
// brains) the system populates `SaveRequested(Some(...))` with a
// timestamped path under `autosaves/`. The actual write happens next
// tick in `save_colony_system`.
//
// Cross-platform: `std::fs::create_dir_all` works on Linux, Windows,
// and macOS. The folder is `autosaves` relative to the current working
// directory (i.e. the repo root when launching via `cargo run`).

/// Per-app timer state for the autosave loop. Stores the count-down
/// to the next autosave fire. Resets to `AUTOSAVE_INTERVAL_SECS`
/// every time we trigger.
#[derive(Resource)]
pub struct AutosaveTimer {
    pub remaining_secs: f32,
}

impl Default for AutosaveTimer {
    fn default() -> Self {
        Self { remaining_secs: (crate::simulation_settings::AUTOSAVE_INTERVAL_MINUTES * 60.0) }
    }
}

pub fn autosave_system(
    real_time:          Res<Time<bevy::time::Real>>,
    mut timer:          ResMut<AutosaveTimer>,
    mut save_requested: ResMut<SaveRequested>,
    sim_running:        Res<crate::simulation_settings::SimulationRunning>,
) {
    // Freeze the autosave countdown while the simulation is paused.
    // Otherwise the real-time clock keeps ticking and an autosave
    // fires mid-pause — clobbering the state the user expects to
    // stay frozen on disk too. Resuming continues the countdown from
    // wherever it was, so a long pause doesn't immediately trigger
    // a save on resume either.
    if !sim_running.0 { return; }

    timer.remaining_secs -= real_time.delta_secs();
    if timer.remaining_secs > 0.0 { return; }
    timer.remaining_secs = (crate::simulation_settings::AUTOSAVE_INTERVAL_MINUTES * 60.0);

    // Skip if a save is already pending — avoids clobbering the
    // user's just-chosen Save-As path before the writer has had a
    // chance to consume it.
    if save_requested.0.is_some() { return; }

    let now = chrono::Local::now();
    let filename = format!("autosave_{}.colony", now.format("%d-%m-%Y-%H-%M-%S"));
    let path = std::path::Path::new("autosaves").join(filename);
    save_requested.0 = Some(path);
}


// ── Colony load ──────────────────────────────────────────────────────────────

/// Decoded representation of one organism from a `.colony` file. Mirrors
/// the on-disk record one-to-one — `spawn_loaded_organism` consumes one
/// of these to materialise the entity hierarchy with the saved state
/// preserved verbatim.
struct LoadedRecord {
    pos:       Vec3,
    rotation:  Quat,
    kind:      OrganismKind,
    organism:  Organism,
    /// Saved sliding-pool herbivore_1 brain weights + REINFORCE state.
    /// `None` for Level 0 organisms, for v001/v002 saves (predate
    /// brain serialisation), and for any organism that isn't enrolled
    /// in the sliding herbivore_1 pool.
    brain:     Option<crate::intelligence_level_herbivore_1_sliding::BrainRestoreHerbivore1>,
    /// Saved limb-pool brain payload (v007+). `None` for sliding
    /// organisms, organisms without a limb pool, or older save
    /// formats. `spawn_loaded_organism` attaches a `BrainRestoreLimb`
    /// component which the matching limb pool's `assign_brains_*`
    /// system consumes next PreUpdate.
    brain_limb: Option<crate::limb_ppo::BrainRestoreLimb>,
}


fn load_colony_from_file(path: &str) -> std::io::Result<Vec<LoadedRecord>> {
    let bytes = std::fs::read(path)?;
    let mut c = 0usize;

    // Magic header check. v004 is current (per-organism A2C brain);
    // v003 is the previous REINFORCE single-MLP brain; v002 adds
    // the intelligence_level byte; v001 has none. v001-v003 still
    // load — their brain blocks just get dropped (v002/v003) or
    // never existed (v001), and every herbivore comes up with
    // fresh-init weights.
    if bytes.len() < SAVE_MAGIC.len() {
        return Err(std::io::Error::other("file too short — missing magic"));
    }
    let magic = &bytes[..SAVE_MAGIC.len()];
    let format_v007 = magic == SAVE_MAGIC;
    let format_v006 = magic == SAVE_MAGIC_LEGACY_V006;
    let format_v005 = magic == SAVE_MAGIC_LEGACY_V005;
    let format_v004 = magic == SAVE_MAGIC_LEGACY_V004;
    let format_v003 = magic == SAVE_MAGIC_LEGACY_V003;
    let format_v002 = magic == SAVE_MAGIC_LEGACY_V002;
    let format_v001 = magic == SAVE_MAGIC_LEGACY_V001;
    if !format_v007 && !format_v006 && !format_v005 && !format_v004 && !format_v003 && !format_v002 && !format_v001 {
        return Err(std::io::Error::other(
            "magic mismatch — not an AEONS colony save (or unsupported version)",
        ));
    }
    // v002+ all share the intelligence_level byte after has_variable_form.
    let has_intelligence_byte = format_v007 || format_v006 || format_v005 || format_v004 || format_v003 || format_v002;
    // v006+ adds the sliding_movement byte after has_variable_form / before
    // intelligence_level. Older saves all default to `true`.
    let has_sliding_byte = format_v007 || format_v006;
    // v007+ appends a limb-brain block (kind byte + optional payload)
    // after the existing sliding-brain block per organism.
    let has_limb_brain_section = format_v007;
    c += SAVE_MAGIC.len();

    let count = read_u32(&bytes, &mut c)?;
    let mut out: Vec<LoadedRecord> = Vec::with_capacity(count as usize);

    for _ in 0..count {
        let kind_byte = read_u8(&bytes, &mut c)?;
        let kind = match kind_byte {
            0 => OrganismKind::Photoautotroph,
            1 => OrganismKind::Heterotroph,
            other => return Err(std::io::Error::other(
                format!("unknown organism kind tag: {other}"),
            )),
        };
        let pos      = read_vec3(&bytes, &mut c)?;
        let rotation = read_quat(&bytes, &mut c)?;
        let energy             = read_f32(&bytes, &mut c)?;
        let in_sunlight        = read_u8 (&bytes, &mut c)? != 0;
        let reproduced         = read_u8 (&bytes, &mut c)? != 0;
        let reproductions      = read_u8 (&bytes, &mut c)?;
        let movement_speed     = read_f32(&bytes, &mut c)?;
        let movement_direction = read_vec3(&bytes, &mut c)?;
        let velocity           = read_vec3(&bytes, &mut c)?;
        let is_climbing        = read_u8 (&bytes, &mut c)? != 0;
        let climb_energy_debt  = read_f32(&bytes, &mut c)?;
        let photo_cell_count     = read_i32(&bytes, &mut c)?;
        let non_photo_cell_count = read_i32(&bytes, &mut c)?;
        let symmetry = match read_u8(&bytes, &mut c)? {
            0 => Symmetry::NoSymmetry,
            1 => Symmetry::Bilateral,
            other => return Err(std::io::Error::other(
                format!("unknown symmetry tag: {other}"),
            )),
        };
        let is_sessile        = read_u8(&bytes, &mut c)? != 0;
        let has_variable_form = read_u8(&bytes, &mut c)? != 0;
        // v006 inserted the sliding_movement byte here. Pre-v006 files
        // pre-date physics-based movement; default to `true`.
        let mut sliding_movement = if has_sliding_byte {
            read_u8(&bytes, &mut c)? != 0
        } else {
            true
        };
        // Same invariant `spawn_organism` enforces: sessile ⇒ sliding.
        // Catches older `.colony` files that pre-date the editor's UI
        // grey-out and saved a sessile+limb organism by mistake.
        if is_sessile { sliding_movement = true; }

        // v002+ saves the intelligence level explicitly; v001 didn't,
        // so we synthesise it via the same deterministic rule that
        // was in force when those saves were written.
        let intelligence_level = if has_intelligence_byte {
            match read_u8(&bytes, &mut c)? {
                0 => IntelligenceLevel::Level0,
                1 => IntelligenceLevel::Level1,
                2 => IntelligenceLevel::Level2,
                3 => IntelligenceLevel::Level3,
                other => return Err(std::io::Error::other(
                    format!("unknown intelligence-level tag: {other}"),
                )),
            }
        } else if is_sessile {
            IntelligenceLevel::Level0
        } else {
            match kind {
                OrganismKind::Photoautotroph => IntelligenceLevel::Level1,
                OrganismKind::Heterotroph    => IntelligenceLevel::Level3,
            }
        };

        let bp_count = read_u32(&bytes, &mut c)?;
        let mut body_parts: Vec<BodyPart> = Vec::with_capacity(bp_count as usize);
        for _ in 0..bp_count {
            let bp_kind = match read_u8(&bytes, &mut c)? {
                0 => BodyPartKind::Body,
                1 => BodyPartKind::Limb,
                2 => BodyPartKind::Organ,
                other => return Err(std::io::Error::other(
                    format!("unknown body-part kind tag: {other}"),
                )),
            };
            let local_offset = read_vec3(&bytes, &mut c)?;
            let consumed     = read_u8(&bytes, &mut c)? != 0;
            let debug_blue   = read_u8(&bytes, &mut c)? != 0;
            let regrowable   = read_u8(&bytes, &mut c)? != 0;

            let attachment_present = read_u8(&bytes, &mut c)? != 0;
            let attachment = if attachment_present {
                let parent_idx   = read_u32(&bytes, &mut c)? as usize;
                let origin_local = read_vec3(&bytes, &mut c)?;
                let rotation     = read_quat(&bytes, &mut c)?;
                Some(crate::body_part::Attachment { parent_idx, origin_local, rotation })
            } else {
                None
            };

            let cell_count = read_u32(&bytes, &mut c)?;
            let mut cells: Vec<Cell> = Vec::with_capacity(cell_count as usize);
            for _ in 0..cell_count {
                let local_pos       = read_vec3(&bytes, &mut c)?;
                let cell_type       = read_cell_type(&bytes, &mut c)?;
                let cell_energy     = read_f32 (&bytes, &mut c)?;
                let neighbour_count = read_u8  (&bytes, &mut c)?;
                // Reconstruct the cached PhotosyntheticCell from the saved
                // neighbour count rather than serialising it — fully
                // derivable, smaller files.
                let photo = match cell_type {
                    CellType::Photo    => Some(PhotosyntheticCell::new(
                        neighbour_count,
                        crate::energy::PHOTO_PRODUCTION_PER_CELL,
                    )),
                    CellType::NonPhoto | CellType::Placeholder => None,
                };
                cells.push(Cell { local_pos, cell_type, cell_energy, neighbour_count, photo });
            }

            let ocg_count = read_u32(&bytes, &mut c)?;
            let mut ocg: Vec<(usize, Vec3, CellType)> = Vec::with_capacity(ocg_count as usize);
            for _ in 0..ocg_count {
                let idx = read_u32(&bytes, &mut c)? as usize;
                let p   = read_vec3(&bytes, &mut c)?;
                let ct  = read_cell_type(&bytes, &mut c)?;
                ocg.push((idx, p, ct));
            }

            body_parts.push(BodyPart {
                kind: bp_kind,
                local_offset,
                cells,
                ocg,
                attachment,
                consumed,
                debug_blue,
                regrowable,
            });
        }

        // ── Brain section. ───────────────────────────────────────
        // Format depends on the magic:
        //   * v004 — the new per-organism herbivore_1 A2C payload
        //            (12 weight tensors + REINFORCE prev_*).
        //            Tensor lengths are validated against the
        //            current architecture; mismatch is a hard
        //            error (user-selected behaviour).
        //   * v003 — the old single-MLP REINFORCE payload (4
        //            tensors + prev_state/action/energy/baseline).
        //            Architectures are incompatible; we still
        //            consume the bytes so the file parses, but
        //            drop the data — the loaded organism comes
        //            up with fresh-init weights.
        //   * v001/v002 — no brain section at all.
        let brain: Option<crate::intelligence_level_herbivore_1_sliding::BrainRestoreHerbivore1>
            = if format_v005 || format_v006 || format_v007 {
                // v005+: shared `BrainRestore` (4-tensor MLP) +
                // the herbivore_1 8-byte magic prefix. (Earlier the
                // gate was `format_v005` only — v006 saves left the
                // brain block unconsumed, mis-aligning every record
                // that followed.)
                let brain_present = read_u8(&bytes, &mut c)?;
                if brain_present == 1 {
                    Some(crate::intelligence_level_herbivore_1_sliding::decode_brain_restore(
                        &bytes, &mut c,
                    )?)
                } else {
                    None
                }
            } else if format_v004 {
                // Retired 12-tensor (backbone+actor+critic) brain
                // block from the pre-L3-port A2C architecture. Consume
                // + drop — fresh weights for these organisms on load.
                // Layout: brain_present byte, then 12 length-prefixed
                // f32 vectors + 2 length-prefixed f32 vectors
                // (prev_state, prev_action) + 2 f32 scalars.
                let brain_present = read_u8(&bytes, &mut c)?;
                if brain_present == 1 {
                    for _ in 0..14 {
                        let n = read_u32(&bytes, &mut c)? as usize;
                        for _ in 0..n { let _ = read_f32(&bytes, &mut c)?; }
                    }
                    let _ = read_f32(&bytes, &mut c)?; // prev_dopamine
                    let _ = read_f32(&bytes, &mut c)?; // prev_target_distance
                }
                None
            } else if format_v003 {
                // Old single-MLP REINFORCE brain block. Consume + drop
                // (architecture differs from the current pool).
                let brain_present = read_u8(&bytes, &mut c)?;
                if brain_present == 1 {
                    for _ in 0..6 {
                        let n = read_u32(&bytes, &mut c)? as usize;
                        for _ in 0..n { let _ = read_f32(&bytes, &mut c)?; }
                    }
                    let _ = read_f32(&bytes, &mut c)?;  // prev_energy
                    let _ = read_f32(&bytes, &mut c)?;  // baseline
                    let _ = read_u8 (&bytes, &mut c)?;  // has_prev
                }
                None
            } else {
                None
            };

        // v007 limb-brain block. Kind tag 0..3 (0=none, 1=herbivore_1_limb,
        // 2=l2_limb, 3=l3_limb). When non-zero we deserialise the
        // `BrainRestoreLimb` payload — `spawn_loaded_organism` then
        // attaches it as a component for the matching limb pool's
        // `assign_brains_*_limb` to consume.
        let brain_limb: Option<crate::limb_ppo::BrainRestoreLimb> = if has_limb_brain_section {
            let kind = read_u8(&bytes, &mut c)?;
            if kind == 0 {
                None
            } else if kind > 3 {
                return Err(std::io::Error::other(
                    format!("unknown limb-brain kind tag: {kind}"),
                ));
            } else {
                Some(crate::limb_ppo::decode_brain_restore_limb(&bytes, &mut c)?)
            }
        } else {
            None
        };

        // `adult` is not in the save format; it's derivable from
        // (`has_variable_form`, total grown cell count): non-variable
        // form organisms are always adult, variable-form become adult
        // when they reach `MAX_CELLS`.
        let total_cells = (photo_cell_count + non_photo_cell_count).max(0) as usize;
        let adult = !has_variable_form
            || total_cells >= crate::volumetric_growth::MAX_CELLS;

        let mut organism = Organism {
            body_parts,
            symmetry,
            intelligence_level,
            is_sessile,
            has_variable_form,
            sliding_movement,
            limb_targets: [0.0; 6],
            adult,
            photo_cell_count,
            non_photo_cell_count,
            energy,
            in_sunlight,
            reproduced,
            reproductions,
            // predations is brain-side bookkeeping only — not part of
            // the .colony save format. Always zero on load.
            predations: 0,
            hunger: 0.0,
            // Not serialised in the .colony format — defaults to 0
            // on load. The first depletion tick + first reward event
            // will set it from there.
            dopamine: 0.0,
            // Sentinel = SENSORY_RADIUS ("out of range"). The first
            // sensory tick after load overwrites this with a real
            // distance if a photo is nearby.
            target_distance: crate::sensory::SENSORY_RADIUS,
            movement_speed,
            movement_direction,
            velocity,
            is_climbing,
            climb_energy_debt,
            // Will be derived from the loaded body_parts on the next
            // line — saving the value would have been redundant.
            cached_bounding_radius: 0.0,
            // Structural slots filled in here; brain genes will be
            // populated later by `sync_dna_from_brain_pool` once the
            // brain pool's `assign_brains_l1_hetero` claims a slot.
            dna: crate::lineages::dna::structural_dna(
                kind,
                symmetry,
                has_variable_form,
                is_sessile,
                intelligence_level,
            ),
            // Will be classified on the first speciation tick after
            // load — species ids aren't persisted to `.colony` files
            // (they're a runtime classification, not a property of
            // the saved organism).
            species_id: None,
        };
        organism.recompute_bounding_radius();

        out.push(LoadedRecord { pos, rotation, kind, organism, brain, brain_limb });
    }

    Ok(out)
}


/// Materialise a loaded organism record. Mirrors `spawn_organism`'s entity
/// hierarchy construction (root + body-part children, branches parented
/// to their attachment-target body part) but uses the saved Transform
/// and the fully-populated Organism component AS-IS — no fresh randomised
/// movement direction, no rebuilt cell-count caches, no `recompute_body_parts`
/// call (the saved cells already carry their neighbour counts and the
/// photo cache was reconstructed during decoding).
fn spawn_loaded_organism(
    record:    LoadedRecord,
    smoothing: bool,
    commands:  &mut Commands,
    meshes:    &mut ResMut<Assets<Mesh>>,
    materials: &OrganismMaterials,
    rng:       &mut impl rand::Rng,
) -> Entity {
    let LoadedRecord { pos, rotation, kind, organism, brain, brain_limb } = record;
    if organism.body_parts.is_empty() {
        // Defensive — skip organisms with no body parts (should never
        // happen on a valid save).
        return commands.spawn_empty().id();
    }

    let body_parts_snapshot = organism.body_parts.clone();
    // Capture intelligence level + adult flag + movement paradigm
    // before `organism` is moved into the spawn bundle. The Avian
    // physics block below needs `sliding_movement`.
    let intelligence_level = organism.intelligence_level;
    let adult              = organism.adult;
    let sliding_movement   = organism.sliding_movement;
    let direction_interval = 1.0 + rng.random::<f32>() * 9.0;

    let mut root_cmd = commands.spawn((
        Transform { translation: pos, rotation, ..default() },
        Visibility::Visible,
        OrganismRoot,
        organism,
        DirectionTimer::new(direction_interval),
    ));
    match kind {
        OrganismKind::Photoautotroph => { root_cmd.insert(Photoautotroph); }
        OrganismKind::Heterotroph    => { root_cmd.insert(Heterotroph); }
    }
    match intelligence_level {
        IntelligenceLevel::Level0 => {
            root_cmd.insert(crate::intelligence_level_0::BrainLevel0);
        }
        IntelligenceLevel::Level1
        | IntelligenceLevel::Level2
        | IntelligenceLevel::Level3 => {}
    }
    // Attach the saved brain weights (if any) so the matching pool's
    // `assign_brains_*` writes them into the slot it picks. Skipped
    // for Level 0 (no pool) and for v001/v002 saves (no brain bytes).
    if let Some(b) = brain {
        if !matches!(intelligence_level, IntelligenceLevel::Level0) {
            root_cmd.insert(b);
        }
    }
    // v007 limb-brain payload: attached as a component so the matching
    // limb pool's `assign_brains_*_limb` writes the weights into the
    // freshly-allocated row. Routing is implicit — each pool's assign
    // filter (`intelligence_level + !sliding_movement`) picks exactly
    // one consumer.
    if let Some(b) = brain_limb {
        root_cmd.insert(b);
    }
    let root = root_cmd.id();

    let mut bp_entities: Vec<Entity> = Vec::with_capacity(body_parts_snapshot.len());
    for (idx, bp) in body_parts_snapshot.iter().enumerate() {
        // Adult-at-load → use the smoothed mesh, but only when
        // `smoothing` is on; otherwise the faceted build that
        // `continuous_growth` will eventually re-smooth (or leave
        // faceted, depending on the flag at that future tick).
        let mesh_handle = if bp.regrowable && !bp.ocg.is_empty() {
            let mesh = if adult && smoothing {
                build_smoothed_mesh_from_ocg(&bp.ocg)
            } else {
                build_mesh_from_ocg(&bp.ocg)
            };
            Some(meshes.add(mesh))
        } else {
            None
        };

        let transform = match &bp.attachment {
            Some(a) => Transform {
                translation: a.origin_local,
                rotation:    a.rotation,
                ..Default::default()
            },
            None => Transform::from_translation(Vec3::ZERO),
        };

        let mut child_cmd = commands.spawn((
            transform,
            Visibility::Visible,
            BodyPartIndex(idx),
            ShowGizmo,
        ));
        if let Some(mh) = mesh_handle {
            let mat = materials.handle_for(kind, bp);
            child_cmd.insert((
                Mesh3d(mh),
                MeshMaterial3d(mat),
                OrganismMesh,
                // Per-organism meshes don't cast shadows: the shadow
                // pass would otherwise re-extract every body-part
                // entity per cascade per frame, dominating render cost.
                // Terrain still casts shadows; organism shadows are
                // negligible at our cell scale.
                bevy::light::NotShadowCaster,
            ));
        }
        let child = child_cmd.id();

        // Flat hierarchy — see the matching block in `spawn_organism`
        // for the rationale (phantom-organism fix on body-part despawn
        // cascade).
        let _ = &bp.attachment;
        commands.entity(root).add_child(child);

        bp_entities.push(child);
    }

    // ── Avian3d physics + LimbAnimation (same code path as
    //   `spawn_organism`). Without this, loaded limb-based organisms
    //   would have body-part entities but no rigid bodies, no joints,
    //   no contact flags, no animation — the brain runs but the PD
    //   torque step has nothing to torque, so the organism freezes
    //   at its saved pose and looks "limbless / unmoving" compared
    //   to its fresh-spawned counterpart. Sliding organisms also
    //   need their kinematic compound collider so other physics
    //   bodies can collide with them.
    if sliding_movement {
        // Single compound on the root.
        let mut shapes: Vec<(avian3d::prelude::Position, avian3d::prelude::Rotation, avian3d::prelude::Collider)> = Vec::new();
        for bp in &body_parts_snapshot {
            if !bp.is_alive() { continue; }
            let part_offset = bp.attachment.as_ref()
                .map_or(Vec3::ZERO, |a| a.origin_local);
            for cell in &bp.cells {
                let pos = part_offset + cell.local_pos;
                shapes.push((
                    avian3d::prelude::Position(pos),
                    avian3d::prelude::Rotation::default(),
                    avian3d::prelude::Collider::sphere(
                        crate::cell::CELL_COLLISION_RADIUS,
                    ),
                ));
            }
        }
        if !shapes.is_empty() {
            commands.entity(root).insert((
                avian3d::prelude::RigidBody::Kinematic,
                avian3d::prelude::Collider::compound(shapes),
            ));
        }
        // LimbAnimation on procedural-style limb body parts (kind =
        // Limb). Mirrors the in-loop insertion in spawn_organism.
        for (idx, bp) in body_parts_snapshot.iter().enumerate() {
            if !bp.is_alive() { continue; }
            if !matches!(bp.kind, crate::cell::BodyPartKind::Limb) { continue; }
            use rand::RngExt;
            let mirror = bp.attachment.as_ref()
                .is_some_and(|a| a.origin_local.x < 0.0);
            let two_pi = std::f32::consts::TAU;
            let la = LimbAnimation {
                freqs:  [
                    rng.random_range(1.5..5.0),
                    rng.random_range(1.5..5.0),
                    rng.random_range(1.5..5.0),
                ],
                phases: [
                    rng.random::<f32>() * two_pi,
                    rng.random::<f32>() * two_pi,
                    rng.random::<f32>() * two_pi,
                ],
                amps:   [
                    rng.random_range(0.3..0.7),
                    rng.random_range(0.3..0.7),
                    rng.random_range(0.3..0.7),
                ],
                mirror,
            };
            commands.entity(bp_entities[idx]).insert(la);
        }
    } else {
        // Limb-based: per-part Dynamic + joints + contact flags.
        for (idx, bp) in body_parts_snapshot.iter().enumerate() {
            if !bp.is_alive() { continue; }
            let mut shapes: Vec<(avian3d::prelude::Position, avian3d::prelude::Rotation, avian3d::prelude::Collider)> = Vec::new();
            for cell in &bp.cells {
                shapes.push((
                    avian3d::prelude::Position(cell.local_pos),
                    avian3d::prelude::Rotation::default(),
                    avian3d::prelude::Collider::sphere(
                        crate::cell::CELL_COLLISION_RADIUS,
                    ),
                ));
            }
            if !shapes.is_empty() {
                let ang_damping = if idx == 0 { BASE_ANGULAR_DAMPING } else { LIMB_ANGULAR_DAMPING };
                // Build the compound collider once so we can both
                // attach it to the body AND derive mass properties
                // from its shape (see comment below).
                let collider = avian3d::prelude::Collider::compound(shapes);
                commands.entity(bp_entities[idx]).insert((
                    avian3d::prelude::RigidBody::Dynamic,
                    collider.clone(),
                    // Avian's auto-mass-compute pipeline silently fails
                    // to populate `ComputedMass` / `ComputedAngularInertia`
                    // for these compound colliders. Defaults of those
                    // types store INVERSE values that init to zero
                    // (representing infinite mass/inertia), so every
                    // `Forces::apply_force` / `apply_torque` call
                    // computes `acceleration = inverse_mass * force = 0`
                    // and the body never moves. Explicitly deriving the
                    // mass properties from the compound's shape ×
                    // `LIMB_BODY_DENSITY` bypasses the broken auto-
                    // compute path. Diagnosed May 2026 via the
                    // constant-torque experiment in `avian_setup`.
                    avian3d::prelude::MassPropertiesBundle::from_shape(&collider, LIMB_BODY_DENSITY),
                    // Grip friction (see `LIMB_FRICTION_COEFFICIENT`):
                    // `Average` combine with the terrain's default 0.5
                    // gives a contact μ = 0.75 so planted feet propel
                    // the body instead of sliding.
                    avian3d::prelude::Friction::new(LIMB_FRICTION_COEFFICIENT)
                        .with_combine_rule(avian3d::prelude::CoefficientCombine::Average),
                    avian3d::prelude::CollisionEventsEnabled,
                    crate::avian_setup::LimbContact::default(),
                    // Base body gets heavy damping to absorb joint-
                    // reaction torques; limbs get lighter damping so
                    // the PD controller can produce dynamic swings.
                    avian3d::prelude::AngularDamping(ang_damping),
                    avian3d::prelude::LinearDamping(LIMB_LINEAR_DAMPING),
                    // Avian's default sleep plugin freezes low-velocity
                    // bodies, and the brain's `Forces::apply_torque`
                    // writes don't wake them — every limb organism
                    // settled, slept, and never moved again. Marker
                    // keeps these bodies permanently integrated so the
                    // PD controller's torques actually take effect.
                    avian3d::prelude::SleepingDisabled,
                    // Telemetry: the PD controller writes the last
                    // commanded torque here every frame; the dataset
                    // exporter reads it to log torque_norm per organism.
                    crate::avian_setup::LastAppliedTorque::default(),
                ));
            }
        }
        // Spherical joints — one per appendage. The pivot is placed on
        // the FACE between the parent cell adjacent to the attachment
        // and the limb's first cell (see `limb_joint_anchors`), so the
        // limb rotates like an anatomical shoulder joint rather than
        // around its own centroid. Compliance is set to a tiny non-
        // zero value (`LIMB_JOINT_COMPLIANCE`) — strictly-rigid joints
        // (compliance = 0) are numerically fragile under the high PD
        // torques the brain commands, and a microscopic give lets the
        // solver converge cleanly.
        for (idx, bp) in body_parts_snapshot.iter().enumerate() {
            if !bp.is_alive() { continue; }
            let Some(attach) = bp.attachment.as_ref() else { continue; };
            let parent_e = bp_entities[attach.parent_idx];
            let child_e  = bp_entities[idx];
            let parent_cells = &body_parts_snapshot[attach.parent_idx].cells;
            let (anchor1, anchor2) = limb_joint_anchors(parent_cells, attach.origin_local);
            commands.spawn(
                avian3d::prelude::SphericalJoint::new(parent_e, child_e)
                    .with_local_anchor1(anchor1)
                    .with_local_anchor2(anchor2)
                    .with_point_compliance(crate::avian_setup::LIMB_JOINT_COMPLIANCE),
            );
        }
    }

    root
}


// ── Little-endian reader helpers ─────────────────────────────────────────────

#[inline]
fn read_u8(buf: &[u8], c: &mut usize) -> std::io::Result<u8> {
    if *c + 1 > buf.len() { return Err(std::io::Error::other("EOF reading u8")); }
    let v = buf[*c]; *c += 1; Ok(v)
}
#[inline]
fn read_u32(buf: &[u8], c: &mut usize) -> std::io::Result<u32> {
    if *c + 4 > buf.len() { return Err(std::io::Error::other("EOF reading u32")); }
    let v = u32::from_le_bytes(buf[*c..*c+4].try_into().unwrap());
    *c += 4; Ok(v)
}
#[inline]
fn read_i32(buf: &[u8], c: &mut usize) -> std::io::Result<i32> {
    if *c + 4 > buf.len() { return Err(std::io::Error::other("EOF reading i32")); }
    let v = i32::from_le_bytes(buf[*c..*c+4].try_into().unwrap());
    *c += 4; Ok(v)
}
#[inline]
fn read_f32(buf: &[u8], c: &mut usize) -> std::io::Result<f32> {
    if *c + 4 > buf.len() { return Err(std::io::Error::other("EOF reading f32")); }
    let v = f32::from_le_bytes(buf[*c..*c+4].try_into().unwrap());
    *c += 4; Ok(v)
}
#[inline]
fn read_vec3(buf: &[u8], c: &mut usize) -> std::io::Result<Vec3> {
    let x = read_f32(buf, c)?;
    let y = read_f32(buf, c)?;
    let z = read_f32(buf, c)?;
    Ok(Vec3::new(x, y, z))
}
#[inline]
fn read_quat(buf: &[u8], c: &mut usize) -> std::io::Result<Quat> {
    let x = read_f32(buf, c)?;
    let y = read_f32(buf, c)?;
    let z = read_f32(buf, c)?;
    let w = read_f32(buf, c)?;
    Ok(Quat::from_xyzw(x, y, z, w))
}
#[inline]
fn read_cell_type(buf: &[u8], c: &mut usize) -> std::io::Result<CellType> {
    match read_u8(buf, c)? {
        0 => Ok(CellType::Photo),
        1 => Ok(CellType::NonPhoto),
        2 => Ok(CellType::Placeholder),
        other => Err(std::io::Error::other(format!("unknown cell-type tag: {other}"))),
    }
}

/// Canonical byte tag for a `CellType` in the `.colony` binary format.
/// Must stay in sync with `read_cell_type`.
#[inline]
fn cell_type_byte(ct: CellType) -> u8 {
    match ct {
        CellType::Photo       => 0,
        CellType::NonPhoto    => 1,
        CellType::Placeholder => 2,
    }
}


// ── Little-endian writer helpers ─────────────────────────────────────────────

#[inline] fn put_u8 (b: &mut Vec<u8>, v: u8)  { b.push(v); }
#[inline] fn put_u32(b: &mut Vec<u8>, v: u32) { b.extend_from_slice(&v.to_le_bytes()); }
#[inline] fn put_i32(b: &mut Vec<u8>, v: i32) { b.extend_from_slice(&v.to_le_bytes()); }
#[inline] fn put_f32(b: &mut Vec<u8>, v: f32) { b.extend_from_slice(&v.to_le_bytes()); }
#[inline] fn put_vec3(b: &mut Vec<u8>, v: Vec3) {
    put_f32(b, v.x); put_f32(b, v.y); put_f32(b, v.z);
}
#[inline] fn put_quat(b: &mut Vec<u8>, q: Quat) {
    put_f32(b, q.x); put_f32(b, q.y); put_f32(b, q.z); put_f32(b, q.w);
}


// Organism + markers + OrganismKind moved to `organism.rs`. They're
// re-exported above via `pub use crate::organism::*;` so existing imports
// continue to find them through `crate::colony::*`.

// ── Spawning ─────────────────────────────────────────────────────────────────

fn spawn_colony(
    mut commands:    Commands,
    mut meshes:      ResMut<Assets<Mesh>>,
    mut materials:   ResMut<Assets<StandardMaterial>>,
    heightmap:       Res<HeightmapSampler>,
    load_path:       Res<ColonyLoadPath>,
    smoothing:       Res<crate::simulation_settings::Smoothing>,
    map_size:        Res<MapSize>,
    max_photoautotrophs: Res<crate::simulation_settings::MaxPhotoautotrophs>,
    max_herbivores:      Res<crate::simulation_settings::MaxHerbivores>,
    start_heteros:       Res<crate::simulation_settings::StartHeterotrophs>,
    start_photos:        Res<crate::simulation_settings::StartPhotoautotrophs>,
    mut spawned:     Local<bool>,
) {
    if *spawned { return; }
    *spawned = true;

    let materials = OrganismMaterials::new(&mut materials);
    // Mirror the OrganismMaterials into a Resource so runtime spawners
    // (`auto_spawn_heteros`) can reuse the same handles without
    // rebuilding the StandardMaterials. Clone is cheap — each field is
    // a Handle<StandardMaterial>.
    commands.insert_resource(OrganismMaterials {
        photo:      materials.photo.clone(),
        hetero:     materials.hetero.clone(),
        debug_blue: materials.debug_blue.clone(),
    });
    let mut rng = rand::rng();

    // If a colony save file was supplied on the command line, try to
    // restore it. On any failure (missing file, malformed bytes, version
    // mismatch) we log the reason and fall through to fresh generation
    // so the run still produces something visible.
    if let Some(path) = &load_path.0 {
        match load_colony_from_file(path) {
            Ok(records) => {
                let n = records.len();
                for record in records {
                    spawn_loaded_organism(record, smoothing.0, &mut commands, &mut meshes, &materials, &mut rng);
                }
                info!("loaded colony from {} — {} organisms restored", path, n);
                return;
            }
            Err(e) => {
                error!("failed to load colony from {}: {} — falling back to fresh generation", path, e);
            }
        }
    }

    // Derive initial cohort sizes from the launcher-set spawn-count
    // resources (defaulting to the matching `DEFAULT_*` constants).
    // Both spawn counts are independent of the running-population
    // caps (`MaxPhotoautotrophs` / `MaxHerbivores`) so the user can
    // seed a small starter cohort and let reproduction backfill the
    // population up to each class's cap.
    let n_herbivores      = start_heteros.0;
    let n_photoautotrophs = start_photos.0;
    info!(
        "spawn_colony: target cohort = {} photoautotrophs + {} herbivores \
         (StartPhotoautotrophs={}, StartHeterotrophs={}, \
          MaxPhotoautotrophs={}, MaxHerbivores={})",
        n_photoautotrophs, n_herbivores,
        start_photos.0, start_heteros.0,
        max_photoautotrophs.0, max_herbivores.0,
    );

    for _ in 0..n_photoautotrophs {
        // Spawn strictly inside the WORLD_SAFETY_MARGIN inset so the
        // organism is born inside the same XZ band that
        // `apply_world_bounds` keeps it within at runtime.
        let x = rng.random_range(
            crate::world_geometry::WORLD_SAFETY_MARGIN
                ..(map_size.x - crate::world_geometry::WORLD_SAFETY_MARGIN),
        );
        let z = rng.random_range(
            crate::world_geometry::WORLD_SAFETY_MARGIN
                ..(map_size.z - crate::world_geometry::WORLD_SAFETY_MARGIN),
        );
        let y = heightmap.height_at(x, z) + 1.0;

        // 80% of photoautotrophs are variable-form: NoSymmetry,
        // sessile, plant-like. The remaining 20% are Bilateral and
        // mobile (animal-like).
        let has_variable_form = rng.random::<f32>() < 0.8;
        let (symmetry, body_parts, max_e) = if has_variable_form {
            // Single root cell, asymmetric growth, branching enabled.
            let ocg = vec![(0usize, Vec3::ZERO, CellType::Photo)];
            let parts = vec![root_body_part_from_ocg(&ocg)];
            let max_e = (ocg.len() as f32) * crate::energy::MAX_ENERGY_PER_CELL;
            (Symmetry::NoSymmetry, parts, max_e)
        } else {
            // Bilateral seed: cells at (±MIN_X_BILATERAL, 0, 0) — exact
            // +X axis-aligned RD neighbours sharing a rhombic face on
            // the YZ-plane. Both go into a SINGLE body part whose OCG
            // contains both halves; `build_mesh_from_ocg`'s weld+dedup
            // fuses them at the seam (see `bilateral_body_part_from_right_ocg`).
            let right_seed = vec![(
                0usize,
                Vec3::new(crate::body_part::MIN_X_BILATERAL, 0.0, 0.0),
                CellType::Photo,
            )];
            let parts = vec![
                crate::body_part::bilateral_body_part_from_right_ocg(&right_seed)
            ];
            let max_e = 2.0 * crate::energy::MAX_ENERGY_PER_CELL;
            (Symmetry::Bilateral, parts, max_e)
        };

        // Initial cohort intelligence level — rolled once per organism.
        // For photoautotrophs the 80%-Level0 target falls out of the
        // sessile branch (sessile == has_variable_form for photos), so
        // mobile photos always go to Level1. Inherited verbatim by
        // offspring after this point.
        let intel = IntelligenceLevel::for_initial_spawn(
            OrganismKind::Photoautotroph,
            has_variable_form,
            &mut rng,
        );
        spawn_organism(
            Vec3::new(x, y, z),
            body_parts,
            OrganismKind::Photoautotroph,
            symmetry,
            has_variable_form,
            // is_sessile is forced true inside spawn_organism whenever
            // has_variable_form is true; we pass false here to keep the
            // signal explicit.
            false,
            intel,
            smoothing.0,
            max_e * 0.5,
            // Initial cohort: sliding (legacy) movement until a species
            // with `sliding_movement = false` is imported / spawned.
            true,
            &mut commands,
            &mut meshes,
            &materials,
            &mut rng,
        );
    }

    for _ in 0..n_herbivores {
        // Spawn strictly inside the WORLD_SAFETY_MARGIN inset so the
        // organism is born inside the same XZ band that
        // `apply_world_bounds` keeps it within at runtime.
        let x = rng.random_range(
            crate::world_geometry::WORLD_SAFETY_MARGIN
                ..(map_size.x - crate::world_geometry::WORLD_SAFETY_MARGIN),
        );
        let z = rng.random_range(
            crate::world_geometry::WORLD_SAFETY_MARGIN
                ..(map_size.z - crate::world_geometry::WORLD_SAFETY_MARGIN),
        );
        let y = heightmap.height_at(x, z) + 1.0;
        // Single bilateral body part — see comment for the
        // photoautotroph cohort above.
        let right_seed = vec![(
            0usize,
            Vec3::new(crate::body_part::MIN_X_BILATERAL, 0.0, 0.0),
            CellType::NonPhoto,
        )];
        let body_parts = vec![
            crate::body_part::bilateral_body_part_from_right_ocg(&right_seed)
        ];
        let max_e = 2.0 * crate::energy::MAX_ENERGY_PER_CELL;
        // Heterotroph intelligence: 50/40/10 across L1/L2/L3 — see
        // `IntelligenceLevel::for_initial_spawn`.
        let intel = IntelligenceLevel::for_initial_spawn(
            OrganismKind::Heterotroph,
            false,
            &mut rng,
        );
        spawn_organism(
            Vec3::new(x, y, z),
            body_parts,
            OrganismKind::Heterotroph,
            Symmetry::Bilateral,
            false,  // has_variable_form — heterotrophs are always mobile
            false,  // is_sessile
            intel,
            smoothing.0,
            max_e * 0.5,
            true,   // sliding_movement — initial cohort uses legacy sliding
            &mut commands,
            &mut meshes,
            &materials,
            &mut rng,
        );
    }
}


/// Build the canonical root body part from a flat OCG. Cells mirror the OCG
/// positions; the part is `regrowable` (mutation can extend it) and not in
/// debug-blue mode.
pub fn root_body_part_from_ocg(ocg: &[(usize, Vec3, CellType)]) -> BodyPart {
    let cells = ocg.iter()
        .map(|(_, pos, ct)| Cell::new(*pos, *ct))
        .collect();
    BodyPart {
        kind:         BodyPartKind::Body,
        local_offset: Vec3::ZERO,
        cells,
        ocg:          ocg.to_vec(),
        attachment:   None,
        consumed:     false,
        debug_blue:   false,
        regrowable:   true,
    }
}


/// Set of shared StandardMaterial handles passed to `spawn_organism`. Sharing
/// one handle per (kind × debug-flag) pair keeps GPU bind-group churn minimal.
///
/// Stored as a `Resource` so runtime spawners (`auto_spawn_heteros`,
/// editor-mode placement) can reuse the same handles without rebuilding
/// the `StandardMaterial`s. `spawn_colony` populates it on first run.
#[derive(Resource)]
pub struct OrganismMaterials {
    pub photo:       Handle<StandardMaterial>,
    pub hetero:      Handle<StandardMaterial>,
    pub debug_blue:  Handle<StandardMaterial>,
}

impl OrganismMaterials {
    pub fn new(materials: &mut Assets<StandardMaterial>) -> Self {
        // The trophic material is now WHITE so the mesh's per-vertex
        // colours (assigned per cell by `build_mesh_from_ocg`) come
        // through unmultiplied — every cell shows its own colour
        // regardless of the body part's overall trophic kind. The
        // `photo` and `hetero` handles share one underlying material.
        let body = materials.add(StandardMaterial {
            base_color: Color::WHITE,
            ..default()
        });
        Self {
            photo:      body.clone(),
            hetero:     body,
            // `debug_blue` is kept as a full-part override for
            // procedural reproduction appendages (which seed from a
            // non-Placeholder cell and would otherwise look identical
            // to the base body). Set on the BodyPart's `debug_blue`
            // flag at creation time.
            debug_blue: materials.add(StandardMaterial {
                base_color: Color::srgb(0.2, 0.4, 0.95),
                ..default()
            }),
        }
    }

    /// Pick the material handle for one body part, given the trophic kind of
    /// its owning organism. Per-cell colouring lives on the mesh itself
    /// (`Mesh::ATTRIBUTE_COLOR`), so this only needs to choose between the
    /// shared white body material and the debug-blue full-part override.
    pub fn handle_for(&self, _kind: OrganismKind, bp: &BodyPart) -> Handle<StandardMaterial> {
        if bp.debug_blue {
            self.debug_blue.clone()
        } else {
            // Same handle as `self.hetero`; `kind` is unused now that
            // colour comes from per-vertex data.
            self.photo.clone()
        }
    }
}


/// Per-limb erratic-rotation parameters. Attached at spawn to every
/// body-part entity whose `BodyPart::kind == BodyPartKind::Limb`. Each
/// axis (X / Y / Z) oscillates independently with its own random
/// frequency, phase and amplitude — incommensurate frequencies make the
/// compound rotation read as chaotic rather than periodic. `mirror`
/// inverts all three angles so the left half of a bilateral limb pair
/// swings opposite to its right twin.
#[derive(Component, Clone, Debug)]
pub struct LimbAnimation {
    pub freqs:  [f32; 3],
    pub phases: [f32; 3],
    pub amps:   [f32; 3],
    pub mirror: bool,
}

/// Apply the per-limb erratic rotation each frame. Runs in `Update`;
/// `TransformSystems::Propagate` (PostUpdate) picks it up the same
/// frame. Reads the default `Time` (virtual clock), so animation freezes
/// in lock-step with the rest of the simulation when paused.
pub fn animate_limbs(
    time:  Res<Time>,
    mut q: Query<(&LimbAnimation, &mut Transform)>,
) {
    let t = time.elapsed_secs();
    for (la, mut tr) in &mut q {
        let sign = if la.mirror { -1.0 } else { 1.0 };
        let x = sign * la.amps[0] * (t * la.freqs[0] + la.phases[0]).sin();
        let y = sign * la.amps[1] * (t * la.freqs[1] + la.phases[1]).sin();
        let z = sign * la.amps[2] * (t * la.freqs[2] + la.phases[2]).sin();
        tr.rotation = Quat::from_euler(EulerRot::XYZ, x, y, z);
    }
}


/// Construct + register an organism from a list of body parts at world
/// position `pos`. Each body part owns its own OCG; the mesh for each
/// regrowable part is built by replaying the part's OCG through
/// `build_mesh_from_ocg`. Used by both initial colony spawn and
/// reproduction.
///
/// Hierarchy produced:
///   OrganismRoot (transform = pos, has Organism + trophic marker)
///   ├── body-part-0 child (Mesh3d, parent of any branches that attach to it)
///   │   └── body-part-1 child (if attached to part 0; rotates around its
///   │                          attachment.origin_local in part-0's frame)
///   └── ...
pub fn spawn_organism(
    pos:                Vec3,
    mut body_parts:     Vec<BodyPart>,
    kind:               OrganismKind,
    symmetry:           Symmetry,
    has_variable_form:  bool,
    is_sessile:         bool,
    intelligence_level: IntelligenceLevel,
    smoothing:          bool,
    initial_energy:     f32,
    // Movement paradigm — see `Organism::sliding_movement`. Pass `true`
    // for legacy / sliding organisms (current behaviour); `false` to
    // spawn the organism into Avian's physics world for limb-based
    // locomotion. Defaults to `true` at every current call site.
    sliding_movement:   bool,
    commands:           &mut Commands,
    meshes:             &mut ResMut<Assets<Mesh>>,
    materials:          &OrganismMaterials,
    rng:                &mut impl rand::Rng,
) -> Entity {
    // Enforce the invariant: variable-form organisms are always sessile and
    // always NoSymmetry. Caller bugs that violate this are silently fixed
    // up here so downstream systems can trust the fields.
    let symmetry  = if has_variable_form { Symmetry::NoSymmetry } else { symmetry };
    let is_sessile = is_sessile || has_variable_form;
    // Sessile organisms MUST be on the sliding-movement path: their root
    // is a `RigidBody::Kinematic` (immovable to physics; `apply_movement`
    // skips it). Spawning a sessile organism into the limb-based Dynamic
    // body path would let gravity, joint reactions, and collisions push
    // it around — which presents to the player as a "sliding" sessile
    // organism, the exact thing the sessile flag is supposed to prevent.
    let sliding_movement = sliding_movement || is_sessile;
    if body_parts.is_empty() {
        // Defensive — callers should always provide at least the root part.
        // Returning a bogus entity would silently corrupt downstream queries.
        panic!("spawn_organism called with empty body_parts");
    }

    // Bring per-cell physiology caches in sync with the assembled cell list
    // before the Organism component is built. This populates each cell's
    // `neighbour_count` and (for Photo cells) `PhotosyntheticCell::*`. The
    // photosynthesis tick reads those caches per-frame; if we skipped this
    // step every cell would produce zero energy.
    crate::physiology::recompute_body_parts(&mut body_parts);

    let angle     = rng.random::<f32>() * std::f32::consts::TAU;
    let direction = Vec3::new(angle.cos(), 0.0, angle.sin());
    let speed     = match kind {
        OrganismKind::Photoautotroph => 0.0,
        OrganismKind::Heterotroph    => 15.0 + rng.random::<f32>() * 10.0,
    };

    // Adult at spawn for any organism that won't grow during its own
    // lifetime. Variable-form organisms grow via `continuous_growth`
    // and only become adult once they reach `MAX_CELLS`; everything
    // else is born "fully grown" (its body plan is whatever it was
    // born with — reproduction grows the OFFSPRING by one cell, not
    // the parent).
    let adult = !has_variable_form;

    let mut organism = Organism {
        body_parts: body_parts.clone(),
        symmetry,
        intelligence_level,
        is_sessile,
        has_variable_form,
        sliding_movement,
        limb_targets: [0.0; 6],
        adult,
        photo_cell_count:     0,
        non_photo_cell_count: 0,
        energy: initial_energy.max(0.0),
        in_sunlight: false,
        reproduced: false,
        reproductions: 0,
        predations: 0,
        hunger: 0.0,
        dopamine: 0.0,
        target_distance: crate::sensory::SENSORY_RADIUS,
        movement_speed: speed,
        movement_direction: direction,
        velocity: Vec3::ZERO,
        is_climbing: false,
        climb_energy_debt: 0.0,
        // Populated by recompute_cell_counts below.
        cached_bounding_radius: 0.0,
        // Structural DNA slots — filled here so newly-reproduced /
        // user-placed organisms enter the speciation system with a
        // valid vector from frame 1. Brain-gene slots stay 0 until
        // `sync_dna_from_brain_pool` runs.
        dna: crate::lineages::dna::structural_dna(
            kind,
            symmetry,
            has_variable_form,
            is_sessile,
            intelligence_level,
        ),
        // Inherited from parent (when reproducing) in a separate
        // post-spawn step by `reproduction.rs`; left `None` for
        // initial-cohort spawns + editor placements so the
        // classification tick assigns them.
        species_id: None,
    };
    organism.recompute_cell_counts();

    let direction_interval = 1.0 + rng.random::<f32>() * 9.0;
    let spawn_rotation     = Quat::from_rotation_y(angle);

    let mut root_cmd = commands.spawn((
        Transform::from_translation(pos).with_rotation(spawn_rotation),
        Visibility::Visible,
        OrganismRoot,
        organism,
        DirectionTimer::new(direction_interval),
    ));
    match kind {
        OrganismKind::Photoautotroph => { root_cmd.insert(Photoautotroph); }
        OrganismKind::Heterotroph    => { root_cmd.insert(Heterotroph); }
    }
    // Wire intelligence-pool markers off the stored `intelligence_level`
    // field. Level 0 → `BrainLevel0` keeps the entity out of the L1/L3
    // assign queries. Levels 1 and 3 don't need a marker at spawn —
    // their `assign_*` systems pick the entity up via the trophic
    // marker (`Photoautotroph` / `Heterotroph`) and assign a slot.
    match intelligence_level {
        IntelligenceLevel::Level0 => {
            root_cmd.insert(crate::intelligence_level_0::BrainLevel0);
        }
        IntelligenceLevel::Level1
        | IntelligenceLevel::Level2
        | IntelligenceLevel::Level3 => {}
    }
    let root = root_cmd.id();

    // Spawn one mesh child per body part. Branches are parented to their
    // attachment-target body part's entity so Bevy's transform propagation
    // gives rotation-around-origin for free. We assume body parts are
    // ordered such that any branch's parent_idx < its own index, which is
    // the order callers naturally produce (parts cloned then appended).
    let mut bp_entities: Vec<Entity> = Vec::with_capacity(body_parts.len());
    for (idx, bp) in body_parts.iter().enumerate() {
        // Skip mesh creation for non-regrowable / empty parts (e.g. Krishi).
        // Adult organisms get the smoothed mesh straight away (when
        // `smoothing` is on); growing (variable-form) organisms get the
        // faceted mesh that `continuous_growth` will replace once they
        // reach MAX_CELLS. With `smoothing` off the faceted mesh is
        // used unconditionally.
        let mesh_handle = if bp.regrowable && !bp.ocg.is_empty() {
            let mesh = if adult && smoothing {
                build_smoothed_mesh_from_ocg(&bp.ocg)
            } else {
                build_mesh_from_ocg(&bp.ocg)
            };
            Some(meshes.add(mesh))
        } else {
            None
        };

        // Branch transform: translate by origin_local in parent's frame,
        // apply attachment.rotation. Root: identity.
        let transform = match &bp.attachment {
            Some(a) => Transform {
                translation: a.origin_local,
                rotation:    a.rotation,
                ..Default::default()
            },
            None => Transform::from_translation(Vec3::ZERO),
        };

        let mut child_cmd = commands.spawn((
            transform,
            Visibility::Visible,
            BodyPartIndex(idx),
            ShowGizmo,
        ));
        if let Some(mh) = mesh_handle {
            let mat = materials.handle_for(kind, bp);
            child_cmd.insert((
                Mesh3d(mh),
                MeshMaterial3d(mat),
                OrganismMesh,
                bevy::light::NotShadowCaster,
            ));
        }
        let child = child_cmd.id();

        // Limb animation: kind=Limb body parts get the `LimbAnimation`
        // marker with randomized per-axis oscillator parameters. Mirror
        // twins (entity origin on the −X side) get the inverted-sign
        // animation so the pair swings opposite, like a real animal's
        // limbs. The actual rotation update lives in `animate_limbs`.
        //
        // Limb-based organisms drive their joints via the physics
        // engine (Phase 4 hooks brain PD targets into Avian), so the
        // kinematic rotation marker is only inserted for sliding
        // organisms — physics-driven limbs would fight the animation.
        if sliding_movement
            && matches!(bp.kind, crate::cell::BodyPartKind::Limb) {
            use rand::RngExt;
            let mirror = bp.attachment.as_ref()
                .is_some_and(|a| a.origin_local.x < 0.0);
            let two_pi = std::f32::consts::TAU;
            let la = LimbAnimation {
                freqs:  [
                    rng.random_range(1.5..5.0),
                    rng.random_range(1.5..5.0),
                    rng.random_range(1.5..5.0),
                ],
                phases: [
                    rng.random::<f32>() * two_pi,
                    rng.random::<f32>() * two_pi,
                    rng.random::<f32>() * two_pi,
                ],
                amps:   [
                    rng.random_range(0.3..0.7),
                    rng.random_range(0.3..0.7),
                    rng.random_range(0.3..0.7),
                ],
                mirror,
            };
            commands.entity(child).insert(la);
        }

        // Parent EVERY body-part entity directly under the OrganismRoot
        // — branches included. Earlier this nested branches under their
        // parent body-part entity for "rotation-around-pivot via
        // transform propagation", but body-part transforms are identity
        // in practice (Quat::IDENTITY rotation, Vec3::ZERO translation
        // on the root part) so siblings-under-root gives the same
        // world transform. The flat layout fixes a phantom-organism bug:
        // when predation despawned body_parts[0]'s entity, Bevy's
        // recursive try_despawn cascaded to every branch nested under
        // it, even though `prey_dead == false` because other body
        // parts were still alive in the Organism data — the root
        // entity then survived with zero children and the data
        // continued to claim alive body parts, becoming an invisible
        // ghost that still photosynthesised and reproduced.
        let _ = &bp.attachment; // attachment kept on the data side for
                                // continuous_growth's branch geometry;
                                // the entity hierarchy is flat.
        commands.entity(root).add_child(child);

        bp_entities.push(child);
    }

    // ── Avian3d physics components ─────────────────────────────────────
    //
    // Sliding organisms (current behaviour for everything):
    //   * `RigidBody::Kinematic` on the OrganismRoot — its transform is
    //     written by `apply_movement`; Avian's kinematic sync follows.
    //   * One compound collider on the root built from every cell
    //     across every body part, with cell positions offset by the
    //     body part's attachment origin (so the collider matches the
    //     world positions the rest of the codebase already uses).
    //
    // Limb-based organisms (none until Phase 5's species-editor toggle):
    //   * Each body part entity gets `RigidBody::Dynamic` + its own
    //     compound collider built from THAT part's cells.
    //   * For each appendage (`attachment.is_some()`) we spawn a
    //     `SphericalJoint` entity connecting the parent body-part
    //     entity to the appendage entity, anchored at
    //     `attachment.origin_local` on the parent and `Vec3::ZERO` on
    //     the child (limb body parts are rebased so their local origin
    //     sits at the first-cell pivot).
    if sliding_movement {
        let mut shapes: Vec<(avian3d::prelude::Position, avian3d::prelude::Rotation, avian3d::prelude::Collider)> = Vec::new();
        for bp in &body_parts {
            if !bp.is_alive() { continue; }
            let part_offset = bp.attachment.as_ref()
                .map_or(Vec3::ZERO, |a| a.origin_local);
            for cell in &bp.cells {
                let pos = part_offset + cell.local_pos;
                shapes.push((
                    avian3d::prelude::Position(pos),
                    avian3d::prelude::Rotation::default(),
                    avian3d::prelude::Collider::sphere(
                        crate::cell::CELL_COLLISION_RADIUS,
                    ),
                ));
            }
        }
        if !shapes.is_empty() {
            commands.entity(root).insert((
                avian3d::prelude::RigidBody::Kinematic,
                avian3d::prelude::Collider::compound(shapes),
            ));
        }
    } else {
        // Limb-based: per-part dynamic bodies.
        for (idx, bp) in body_parts.iter().enumerate() {
            if !bp.is_alive() { continue; }
            let mut shapes: Vec<(avian3d::prelude::Position, avian3d::prelude::Rotation, avian3d::prelude::Collider)> = Vec::new();
            for cell in &bp.cells {
                shapes.push((
                    avian3d::prelude::Position(cell.local_pos),
                    avian3d::prelude::Rotation::default(),
                    avian3d::prelude::Collider::sphere(
                        crate::cell::CELL_COLLISION_RADIUS,
                    ),
                ));
            }
            if !shapes.is_empty() {
                let ang_damping = if idx == 0 { BASE_ANGULAR_DAMPING } else { LIMB_ANGULAR_DAMPING };
                // See parallel spawn_organism path: explicit mass
                // properties because Avian's auto-compute leaves
                // `ComputedAngularInertia` at its INFINITY default
                // (inverse = 0) for our compound colliders.
                let collider = avian3d::prelude::Collider::compound(shapes);
                commands.entity(bp_entities[idx]).insert((
                    avian3d::prelude::RigidBody::Dynamic,
                    collider.clone(),
                    // Density lowered from 1.0 so total organism mass
                    // stays light enough that ground-contact friction
                    // can be overcome by the brain's PD torques. See
                    // `LIMB_BODY_DENSITY` for rationale.
                    avian3d::prelude::MassPropertiesBundle::from_shape(&collider, LIMB_BODY_DENSITY),
                    // Grip friction (see `LIMB_FRICTION_COEFFICIENT`):
                    // `Average` combine with the terrain's default 0.5
                    // gives a contact μ = 0.75 so planted feet propel
                    // the body instead of sliding.
                    avian3d::prelude::Friction::new(LIMB_FRICTION_COEFFICIENT)
                        .with_combine_rule(avian3d::prelude::CoefficientCombine::Average),
                    // Per-entity event toggle so Avian's narrow phase
                    // emits `CollisionStart` / `CollisionEnd` messages
                    // for this body part — consumed by
                    // `update_limb_contacts` to populate `LimbContact`.
                    avian3d::prelude::CollisionEventsEnabled,
                    crate::avian_setup::LimbContact::default(),
                    // Base body gets heavy damping to absorb joint-
                    // reaction torques; limbs get lighter damping so
                    // the PD controller can produce dynamic swings.
                    avian3d::prelude::AngularDamping(ang_damping),
                    avian3d::prelude::LinearDamping(LIMB_LINEAR_DAMPING),
                    // Avian's default sleep plugin freezes low-velocity
                    // bodies, and the brain's `Forces::apply_torque`
                    // writes don't wake them — every limb organism
                    // settled, slept, and never moved again. Marker
                    // keeps these bodies permanently integrated so the
                    // PD controller's torques actually take effect.
                    avian3d::prelude::SleepingDisabled,
                    // Telemetry: the PD controller writes the last
                    // commanded torque here every frame; the dataset
                    // exporter reads it to log torque_norm per organism.
                    crate::avian_setup::LastAppliedTorque::default(),
                ));
            }
        }
        // Joints — one spherical joint per appendage. Skips the base
        // body, which has no attachment. See `limb_joint_anchors` for
        // pivot-placement details (joint sits on the cell-to-cell
        // face) and `LIMB_JOINT_COMPLIANCE` for the small non-zero
        // compliance that keeps the XPBD solver well-conditioned.
        for (idx, bp) in body_parts.iter().enumerate() {
            if !bp.is_alive() { continue; }
            let Some(attach) = bp.attachment.as_ref() else { continue; };
            let parent_e = bp_entities[attach.parent_idx];
            let child_e  = bp_entities[idx];
            let parent_cells = &body_parts[attach.parent_idx].cells;
            let (anchor1, anchor2) = limb_joint_anchors(parent_cells, attach.origin_local);
            commands.spawn(
                avian3d::prelude::SphericalJoint::new(parent_e, child_e)
                    .with_local_anchor1(anchor1)
                    .with_local_anchor2(anchor2)
                    .with_point_compliance(crate::avian_setup::LIMB_JOINT_COMPLIANCE),
            );
        }
    }

    root
}


// ── Auto-spawn heterotrophs ─────────────────────────────────────────────────
//
// Tops the heterotroph population up to `MinHeteroCount` whenever a
// heterotroph death event fires AND `AutoSpawnHeteros(true)`.
//
// Zero-overhead in the steady state: `RemovedComponents<Heterotroph>`
// is empty between actual death events, so the early-return after the
// flag check costs nothing. When the flag is off we drain the reader
// once to avoid event backlog growth.
//
// Newly spawned organisms have no `BrainInheritance` component, so the
// brain pool's slot-assign path samples fresh random hyperparameter
// genes (see `intelligence_level_1_hetero::assign_brains_l1_hetero`)
// rather than inheriting from any parent.
pub fn auto_spawn_heteros(
    auto:           Res<crate::simulation_settings::AutoSpawnHeteros>,
    min_count:      Res<crate::simulation_settings::MinHeteroCount>,
    mut removed:    RemovedComponents<Heterotroph>,
    heteros:        Query<(), With<Heterotroph>>,
    org_mats:       Option<Res<OrganismMaterials>>,
    heightmap:      Option<Res<HeightmapSampler>>,
    map_size:       Option<Res<MapSize>>,
    smoothing:      Option<Res<crate::simulation_settings::Smoothing>>,
    mut commands:   Commands,
    mut meshes:     ResMut<Assets<Mesh>>,
) {
    if !auto.0 {
        for _ in removed.read() {}
        return;
    }

    let dead = removed.read().count();
    if dead == 0 { return; }

    let (Some(org_mats), Some(heightmap), Some(map_size), Some(smoothing)) =
        (org_mats, heightmap, map_size, smoothing) else { return };

    let current = heteros.iter().count();
    if current >= min_count.0 { return; }
    let to_spawn = min_count.0 - current;

    let mut rng = rand::rng();
    for _ in 0..to_spawn {
        // Spawn strictly inside the WORLD_SAFETY_MARGIN inset so the
        // organism is born inside the same XZ band that
        // `apply_world_bounds` keeps it within at runtime.
        let x = rng.random_range(
            crate::world_geometry::WORLD_SAFETY_MARGIN
                ..(map_size.x - crate::world_geometry::WORLD_SAFETY_MARGIN),
        );
        let z = rng.random_range(
            crate::world_geometry::WORLD_SAFETY_MARGIN
                ..(map_size.z - crate::world_geometry::WORLD_SAFETY_MARGIN),
        );
        let y = heightmap.height_at(x, z) + 1.0;

        let right_seed = vec![(
            0usize,
            Vec3::new(crate::body_part::MIN_X_BILATERAL, 0.0, 0.0),
            CellType::NonPhoto,
        )];
        let body_parts = vec![
            crate::body_part::bilateral_body_part_from_right_ocg(&right_seed)
        ];
        let max_e = 2.0 * crate::energy::MAX_ENERGY_PER_CELL;
        let intel = IntelligenceLevel::for_initial_spawn(
            OrganismKind::Heterotroph,
            false,
            &mut rng,
        );
        spawn_organism(
            Vec3::new(x, y, z),
            body_parts,
            OrganismKind::Heterotroph,
            Symmetry::Bilateral,
            false, // has_variable_form
            false, // is_sessile
            intel,
            smoothing.0,
            max_e * 0.5,
            true,  // sliding_movement — auto-spawned heteros use legacy sliding
            &mut commands,
            &mut meshes,
            &org_mats,
            &mut rng,
        );
    }
}
