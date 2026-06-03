// Binary `.colony` save / load — the AEONS format reader & writer.
//
// Split out of `colony.rs`: this module owns the on-disk byte format only —
// serialising live components to bytes (`save_colony_system`,
// `autosave_system`) and parsing bytes back into `LoadedRecord`s
// (`load_colony_from_file`), plus the little-endian primitive helpers and
// `TimeNotation`. Materialising a `LoadedRecord` into an entity hierarchy
// stays in `colony.rs` (`spawn_loaded_organism`), next to `spawn_organism`,
// since the two share the entity-construction / Avian-setup logic.
//
// `crate::colony` re-exports everything here via `pub use`, so external paths
// like `crate::colony::SaveRequested` keep resolving.

use bevy::prelude::*;
use crate::cell::*;
use crate::organism::*;


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
const SAVE_MAGIC:             &[u8;8] = b"AEONS008";
const SAVE_MAGIC_LEGACY_V007: &[u8;8] = b"AEONS007";
const SAVE_MAGIC_LEGACY_V006: &[u8;8] = b"AEONS006";
const SAVE_MAGIC_LEGACY_V005: &[u8;8] = b"AEONS005";
const SAVE_MAGIC_LEGACY_V004: &[u8;8] = b"AEONS004";
const SAVE_MAGIC_LEGACY_V003: &[u8;8] = b"AEONS003";
const SAVE_MAGIC_LEGACY_V002: &[u8;8] = b"AEONS002";
const SAVE_MAGIC_LEGACY_V001: &[u8;8] = b"AEONS001";

/// Elapsed virtual time written into the v008 colony file as a handful
/// of cheap integers (the "time notation"), right after the magic.
/// On load the simulation's virtual clock is advanced to this point so
/// a resumed colony continues from the exact virtual time it was saved
/// at. No separate time-tracking machinery — the clock itself carries
/// absolute time, so every virtual-time consumer (sim timer, auto-
/// export milestones, logs) resumes for free.
#[derive(Clone, Copy, Default)]
pub(crate) struct TimeNotation {
    pub(crate) hours:   u32,
    pub(crate) minutes: u32,
    pub(crate) seconds: u32,
}

impl TimeNotation {
    fn from_secs(total: u32) -> Self {
        Self {
            hours:   total / 3600,
            minutes: (total % 3600) / 60,
            seconds: total % 60,
        }
    }
    pub(crate) fn total_secs(&self) -> u32 {
        self.hours * 3600 + self.minutes * 60 + self.seconds
    }
}

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


pub(crate) fn save_colony_system(
    mut save_requested: ResMut<SaveRequested>,
    // Source of the v008 time-notation — the current virtual elapsed
    // time, written so a reload resumes the clock at this exact point.
    virtual_time: Res<Time<Virtual>>,
    organisms: Query<
        (Entity, &Transform, &Organism,
         Has<Photoautotroph>, Has<Heterotroph>, Has<Carnivore>),
        With<OrganismRoot>,
    >,
    // All three SLIDING brain pools. A colony save is a full image of
    // the simulation, so every brain's current weights are persisted:
    // a sliding organism's `BrainRestore` is pulled from whichever pool
    // its `intelligence_level` routes it to (herbivore_1 / L2 / L3).
    pool:    NonSend<crate::intelligence_level_herbivore_1_sliding::BrainPoolHerbivore1>,
    pool_l2: NonSend<crate::intelligence_level_2_sliding::BrainPoolL2>,
    pool_l3: NonSend<crate::intelligence_level_3_sliding::BrainPoolL3>,
    // Limb-based brain pools. One snapshot per pool, taken once before
    // iterating organisms (each does a small batch of GPU→CPU syncs).
    // Each limb organism contributes one limb-brain block to its record.
    pool_limb_h:  NonSend<crate::intelligence_level_herbivore_1_limb::BrainPoolHerbivore1Limb>,
    pool_limb_l2: NonSend<crate::intelligence_level_2_limb::BrainPoolL2Limb>,
    pool_limb_l3: NonSend<crate::intelligence_level_3_limb::BrainPoolL3Limb>,
) {
    let Some(target_path) = save_requested.0.take() else { return };

    let mut buf: Vec<u8> = Vec::with_capacity(64 * 1024);
    buf.extend_from_slice(SAVE_MAGIC);

    // v008 time notation — current virtual elapsed time as cheap
    // integers, immediately after the magic. The clock already carries
    // absolute time (it's set on load), so this round-trips across
    // save → load → save chains without any separate bookkeeping.
    let notation = TimeNotation::from_secs(virtual_time.elapsed_secs() as u32);
    put_u32(&mut buf, notation.hours);
    put_u32(&mut buf, notation.minutes);
    put_u32(&mut buf, notation.seconds);

    // One snapshot per pool (sliding + limb), reused across all
    // organisms — a single GPU→CPU transfer each, not one per organism.
    let snap_sl_h  = pool.snapshot();
    let snap_sl_l2 = pool_l2.snapshot();
    let snap_sl_l3 = pool_l3.snapshot();
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

        // ── Sliding-brain section ─────────────────────────────────
        // `present byte + shared BrainRestore payload`. For a SLIDING
        // heterotroph we pull its weights from whichever sliding pool
        // its `intelligence_level` routes it to:
        //   * Level1 + !Carnivore → herbivore_1
        //   * Level2              → L2
        //   * Level3              → L3
        // The `BrainRestore` payload is self-describing (length-prefixed
        // tensors), and on load the matching pool's `assign_brains_*`
        // restores it — selected by the same `intelligence_level`, which
        // is also serialised — so no pool tag is needed in the file.
        // Limb organisms (and photoautotrophs / Krishi) emit 0 here and
        // carry their weights in the limb-brain block below instead.
        let sliding_brain = if org.sliding_movement && is_hetero {
            match org.intelligence_level {
                IntelligenceLevel::Level1 if !is_carn => snap_sl_h.extract(entity),
                IntelligenceLevel::Level2             => snap_sl_l2.extract(entity),
                IntelligenceLevel::Level3             => snap_sl_l3.extract(entity),
                _ => None,
            }
        } else {
            None
        };
        match sliding_brain {
            None => put_u8(&mut buf, 0),
            Some(b) => {
                put_u8(&mut buf, 1);
                // Shared `BrainRestore` encoding used by all three
                // sliding pools (they differ only in HIDDEN, which the
                // length-prefixed payload captures).
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
pub(crate) struct LoadedRecord {
    pub(crate) pos:       Vec3,
    pub(crate) rotation:  Quat,
    pub(crate) kind:      OrganismKind,
    pub(crate) organism:  Organism,
    /// Saved sliding-pool herbivore_1 brain weights + REINFORCE state.
    /// `None` for Level 0 organisms, for v001/v002 saves (predate
    /// brain serialisation), and for any organism that isn't enrolled
    /// in the sliding herbivore_1 pool.
    pub(crate) brain:     Option<crate::intelligence_level_herbivore_1_sliding::BrainRestoreHerbivore1>,
    /// Saved limb-pool brain payload (v007+). `None` for sliding
    /// organisms, organisms without a limb pool, or older save
    /// formats. `spawn_loaded_organism` attaches a `BrainRestoreLimb`
    /// component which the matching limb pool's `assign_brains_*`
    /// system consumes next PreUpdate.
    pub(crate) brain_limb: Option<crate::limb_ppo::BrainRestoreLimb>,
}


pub(crate) fn load_colony_from_file(path: &str) -> std::io::Result<(TimeNotation, Vec<LoadedRecord>)> {
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
    let format_v008 = magic == SAVE_MAGIC;
    let format_v007 = magic == SAVE_MAGIC_LEGACY_V007;
    let format_v006 = magic == SAVE_MAGIC_LEGACY_V006;
    let format_v005 = magic == SAVE_MAGIC_LEGACY_V005;
    let format_v004 = magic == SAVE_MAGIC_LEGACY_V004;
    let format_v003 = magic == SAVE_MAGIC_LEGACY_V003;
    let format_v002 = magic == SAVE_MAGIC_LEGACY_V002;
    let format_v001 = magic == SAVE_MAGIC_LEGACY_V001;
    if !format_v008 && !format_v007 && !format_v006 && !format_v005 && !format_v004 && !format_v003 && !format_v002 && !format_v001 {
        return Err(std::io::Error::other(
            "magic mismatch — not an AEONS colony save (or unsupported version)",
        ));
    }
    // v008 has the same per-organism record layout as v007 (it only
    // adds the time notation after the magic), so every v007 feature
    // flag also holds for v008.
    let format_v007_layout = format_v008 || format_v007;
    // v002+ all share the intelligence_level byte after has_variable_form.
    let has_intelligence_byte = format_v007_layout || format_v006 || format_v005 || format_v004 || format_v003 || format_v002;
    // v006+ adds the sliding_movement byte after has_variable_form / before
    // intelligence_level. Older saves all default to `true`.
    let has_sliding_byte = format_v007_layout || format_v006;
    // v007+ appends a limb-brain block (kind byte + optional payload)
    // after the existing sliding-brain block per organism.
    let has_limb_brain_section = format_v007_layout;
    c += SAVE_MAGIC.len();

    // v008 time notation (hours, minutes, seconds) immediately after the
    // magic. Older formats have no notation → resume at t=0.
    let notation = if format_v008 {
        let hours   = read_u32(&bytes, &mut c)?;
        let minutes = read_u32(&bytes, &mut c)?;
        let seconds = read_u32(&bytes, &mut c)?;
        TimeNotation { hours, minutes, seconds }
    } else {
        TimeNotation::default()
    };

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
                    CellType::NonPhoto | CellType::Placeholder | CellType::SubLimb => None,
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
            = if format_v005 || format_v006 || format_v007_layout {
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
            limb_targets: [0.0; 8],
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

    Ok((notation, out))
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
        3 => Ok(CellType::SubLimb),
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
        CellType::SubLimb     => 3,
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
