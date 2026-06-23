// Binary `.colony` save / load: the on-disk byte format reader & writer
// (serialise live components → bytes, parse bytes → `LoadedRecord`s) plus
// little-endian helpers and `TimeNotation`. Materialising a `LoadedRecord`
// into an entity hierarchy lives in `colony.rs` (`spawn_loaded_organism`).
// Re-exported via `crate::colony` so `crate::colony::SaveRequested` resolves.

use bevy::prelude::*;
use crate::cell::*;
use crate::organism::*;
use crate::environment::{WaterLevel, DEFAULT_WATER_LEVEL};


// ── Colony save ──────────────────────────────────────────────────────────────
//
// Layout below. All multi-byte values little-endian; bools / u8-enums = 1
// byte. Leading 8-byte magic lets the loader reject foreign files:
//
//   "AEONS001"                                  (8 bytes magic + version)
//   (v008+) 3×u32 elapsed time                   (hours, minutes, seconds)
//   (v009+) f32  water_level                      (global water surface Y)
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
//       u8   movement_mode                        (v009+ tag: 0=Sliding, 1=LimbBasedWalking,
//                                                   2=Swimming, 3=Flying;
//                                                   v006..v008: old bool 1=Sliding/0=Limb)
//       u8   ground_based                         (v010+; older saves derive it
//                                                   from movement_mode)
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
//               u8    cell_type                  (CellType::tag(); 0 = Photo, 1 = Absorption, …)
//               f32   cell_energy
//               u8    neighbour_count
//           u32  ocg_count
//           for each ocg entry:
//               u32   idx
//               3×f32 pos
//               u8    cell_type
//
// Cached `PhotosyntheticCell` data is NOT serialised — fully derivable
// from `cell_type` + `neighbour_count` on load.

/// Current save format magic (v011). The loader accepts v001-v011;
/// missing fields are synthesised from deterministic spawn-time rules,
/// and brain blocks for an obsolete architecture are parsed-and-dropped
/// (those organisms come up with fresh-init weights, but their structure
/// — positions, body parts, energy — is restored correctly).
/// v011 adds a per-organism SPECIES NAME string (after the intelligence byte)
/// so species identity — and thus the per-species shared brain — survives a
/// reload: on load the named species is resolved/created in the registry and
/// `species_id` is pinned at spawn (older saves carry no name → classified fresh).
const SAVE_MAGIC:             &[u8;8] = b"AEONS012";
const SAVE_MAGIC_LEGACY_V011: &[u8;8] = b"AEONS011";
const SAVE_MAGIC_LEGACY_V010: &[u8;8] = b"AEONS010";
const SAVE_MAGIC_LEGACY_V009: &[u8;8] = b"AEONS009";
const SAVE_MAGIC_LEGACY_V008: &[u8;8] = b"AEONS008";
const SAVE_MAGIC_LEGACY_V007: &[u8;8] = b"AEONS007";
const SAVE_MAGIC_LEGACY_V006: &[u8;8] = b"AEONS006";
const SAVE_MAGIC_LEGACY_V005: &[u8;8] = b"AEONS005";
const SAVE_MAGIC_LEGACY_V004: &[u8;8] = b"AEONS004";
const SAVE_MAGIC_LEGACY_V003: &[u8;8] = b"AEONS003";
const SAVE_MAGIC_LEGACY_V002: &[u8;8] = b"AEONS002";
const SAVE_MAGIC_LEGACY_V001: &[u8;8] = b"AEONS001";

/// Elapsed virtual time stored in the v008 file (right after the magic)
/// as h/m/s integers. On load the virtual clock is advanced to this
/// point, so every virtual-time consumer resumes from the saved instant.
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

/// One-shot: when `Some(path)`, `save_colony_system` writes the world
/// there next tick and resets to `None`. Producers: the Save button and
/// `autosave_system`. If both fire the same tick the later wins (rare,
/// harmless).
#[derive(Resource, Default)]
pub struct SaveRequested(pub Option<std::path::PathBuf>);


pub(crate) fn save_colony_system(
    mut save_requested: ResMut<SaveRequested>,
    // Source of the v008 time-notation: WALL-clock run age (uncapped real
    // time accumulated while running), not the capped virtual clock.
    run_elapsed: Res<crate::simulation_settings::RunElapsed>,
    // v009 global water level, written into the header after the time block.
    water:        Res<WaterLevel>,
    organisms: Query<
        (Entity, &Transform, &Organism,
         Has<Photoautotroph>, Has<Heterotroph>, Has<Carnivore>),
        With<OrganismRoot>,
    >,
    // All three sliding pools — every brain's weights are persisted,
    // pulled from whichever pool `intelligence_level` routes the organism to.
    pool:    NonSend<crate::intelligence_level_herbivore_1_sliding::BrainPoolHerbivore1>,
    pool_l2: NonSend<crate::intelligence_level_2_sliding::BrainPoolL2>,
    pool_l3: NonSend<crate::intelligence_level_3_sliding::BrainPoolL3>,
    // Limb pools. Snapshotted once before iterating (each = a GPU→CPU sync).
    pool_limb_h:  NonSend<crate::intelligence_level_herbivore_1_limb::BrainPoolHerbivore1Limb>,
    pool_limb_l2: NonSend<crate::intelligence_level_2_limb::BrainPoolL2Limb>,
    pool_limb_l3: NonSend<crate::intelligence_level_3_limb::BrainPoolL3Limb>,
    // Swim pool — persisted in the limb-brain block as kind 4 (shares the
    // `BrainRestoreLimb` payload struct; routed back to the swim pool on load
    // by the organism's movement mode).
    pool_swim:    NonSend<crate::intelligence_level_1_swimming::BrainPoolSwim1>,
    // v011 species identity: maps each organism's `species_id` to its display
    // name so the name (not the volatile id) is what gets persisted.
    registry: Res<crate::lineages::species::SpeciesRegistry>,
) {
    let Some(target_path) = save_requested.0.take() else { return };

    let mut buf: Vec<u8> = Vec::with_capacity(64 * 1024);
    buf.extend_from_slice(SAVE_MAGIC);

    // v008 time notation, immediately after the magic.
    let notation = TimeNotation::from_secs(run_elapsed.0 as u32);
    put_u32(&mut buf, notation.hours);
    put_u32(&mut buf, notation.minutes);
    put_u32(&mut buf, notation.seconds);

    // v009 global water level — after the time block, before the count.
    put_f32(&mut buf, water.0);

    // One snapshot per pool, reused across all organisms (single GPU→CPU
    // transfer each, not one per organism).
    let snap_sl_h  = pool.snapshot();
    let snap_sl_l2 = pool_l2.snapshot();
    let snap_sl_l3 = pool_l3.snapshot();
    let snap_h  = pool_limb_h.0.snapshot();
    let snap_l2 = pool_limb_l2.0.snapshot();
    let snap_l3 = pool_limb_l3.0.snapshot();
    let snap_swim = pool_swim.0.snapshot();

    // Two-pass to write the count up front.
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
        // v006 addition — movement paradigm. v009: now a 4-way tag
        // (0=Sliding, 1=LimbBasedWalking, 2=Swimming, 3=Flying) in place
        // of the old sliding/limb bool.
        put_u8(&mut buf, match org.movement_mode {
            MovementMode::Sliding          => 0,
            MovementMode::LimbBasedWalking => 1,
            MovementMode::Swimming         => 2,
            MovementMode::Flying           => 3,
        });
        // v010 addition — ground- vs water-based (floating phototrophs).
        put_u8(&mut buf, org.ground_based as u8);
        // v002 addition — saved so loaded organisms keep their
        // assigned intelligence level instead of being re-rolled.
        put_u8(&mut buf, match org.intelligence_level {
            IntelligenceLevel::Level0 => 0,
            IntelligenceLevel::Level1 => 1,
            IntelligenceLevel::Level2 => 2,
            IntelligenceLevel::Level3 => 3,
        });
        // v011 species name (u32 len + UTF-8). Empty when the organism is
        // unclassified or its species is gone — load treats empty as "no name"
        // and classifies it fresh.
        let species_name = org.species_id
            .and_then(|id| registry.get(id).map(|s| s.name.clone()))
            .unwrap_or_default();
        put_u32(&mut buf, species_name.len() as u32);
        buf.extend_from_slice(species_name.as_bytes());

        put_u32(&mut buf, org.body_parts.len() as u32);
        for bp in &org.body_parts {
            put_u8(&mut buf, match bp.kind {
                BodyPartKind::Body    => 0,
                BodyPartKind::Limb    => 1,
                BodyPartKind::Organ   => 2,
                BodyPartKind::Segment => 3,
                BodyPartKind::Static  => 4,
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

        // Sliding-brain section: present-byte + self-describing
        // `BrainRestore` payload. Pool routed by `intelligence_level`
        // (also serialised), so no pool tag in the file. Limb organisms
        // and photoautotrophs emit 0 here.
        let sliding_brain = if org.movement_mode.is_sliding() && is_hetero {
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
                // Shared encoding for all three sliding pools (they differ
                // only in HIDDEN, captured by the length-prefixed payload).
                crate::intelligence_level_herbivore_1_sliding::encode_brain_restore(&mut buf, &b);
            }
        }

        // v007 limb-brain block: kind tag + payload. Kind tags 0=none,
        // 1=herbivore_1_limb, 2=l2_limb, 3=l3_limb (routed by intelligence_level
        // + carnivore marker), 4=swim (v011 — same `BrainRestoreLimb` payload,
        // swim dims). Non-PPO organisms write 0.
        let limb_brain = if !org.movement_mode.is_sliding() && is_hetero {
            if org.movement_mode.is_swimming() {
                snap_swim.extract(entity).map(|b| (4u8, b))
            } else {
                match org.intelligence_level {
                    IntelligenceLevel::Level1 if !is_carn =>
                        snap_h.extract(entity).map(|b| (1u8, b)),
                    IntelligenceLevel::Level2 =>
                        snap_l2.extract(entity).map(|b| (2u8, b)),
                    IntelligenceLevel::Level3 =>
                        snap_l3.extract(entity).map(|b| (3u8, b)),
                    _ => None,
                }
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
        let _ = is_carn;
    }

    // Ensure the parent directory exists (autosave path may not yet).
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
// Real-time (NOT virtual) timer so backups keep a predictable cadence
// regardless of sim speed or pause. Populates `SaveRequested` with a
// timestamped `autosaves/` path; the write happens next tick in
// `save_colony_system`. Folder is relative to the cwd (repo root).

/// Autosave countdown to the next fire; resets each trigger.
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
    // Freeze the countdown while paused, else the real-time clock keeps
    // ticking and an autosave fires mid-pause (clobbering frozen state).
    if !sim_running.0 { return; }

    timer.remaining_secs -= real_time.delta_secs();
    if timer.remaining_secs > 0.0 { return; }
    timer.remaining_secs = (crate::simulation_settings::AUTOSAVE_INTERVAL_MINUTES * 60.0);

    // Skip if a save is already pending — don't clobber a user's
    // just-chosen Save-As path before the writer consumes it.
    if save_requested.0.is_some() { return; }

    let now = chrono::Local::now();
    let filename = format!("autosave_{}.colony", now.format("%d-%m-%Y-%H-%M-%S"));
    let path = std::path::Path::new("autosaves").join(filename);
    save_requested.0 = Some(path);
}


// ── Colony load ──────────────────────────────────────────────────────────────

/// Decoded organism record from a `.colony` file; `spawn_loaded_organism`
/// consumes one to materialise the entity hierarchy verbatim.
pub(crate) struct LoadedRecord {
    pub(crate) pos:       Vec3,
    pub(crate) rotation:  Quat,
    pub(crate) kind:      OrganismKind,
    pub(crate) organism:  Organism,
    /// Saved sliding-pool brain weights + state. `None` for Level 0,
    /// pre-brain saves, or non-sliding-pool organisms.
    pub(crate) brain:     Option<crate::intelligence_level_herbivore_1_sliding::BrainRestoreHerbivore1>,
    /// Saved limb-pool brain payload (v007+). `None` for sliding
    /// organisms / older formats. Attached as a `BrainRestoreLimb`
    /// for the matching limb pool to consume next PreUpdate.
    pub(crate) brain_limb: Option<crate::limb_ppo::BrainRestoreLimb>,
    /// Saved species name (v011+). `None` for unclassified organisms / older
    /// formats. `spawn_loaded_organism` resolves it to a registry species and
    /// pins `species_id` at spawn so the per-species brain restore lands right.
    pub(crate) species_name: Option<String>,
}


/// Validate an untrusted element count read from the file BEFORE allocating.
/// A corrupt or version-incompatible file can decode a garbage length whose
/// `Vec::with_capacity` would attempt a multi-GB allocation and ABORT the whole
/// process (an alloc failure can't be caught like a normal error). A file
/// claiming `count` elements that each occupy at least `min_elem_bytes` is
/// impossible if that exceeds the bytes actually remaining — reject it as a
/// clean `Err` the caller can surface instead.
fn checked_count(
    count:          u32,
    min_elem_bytes: usize,
    cursor:         usize,
    total:          usize,
    what:           &str,
) -> std::io::Result<usize> {
    let remaining    = total.saturating_sub(cursor);
    let max_possible = remaining / min_elem_bytes.max(1);
    if count as usize > max_possible {
        return Err(std::io::Error::other(format!(
            "{what}: implausible count {count} ({remaining} byte(s) left, \u{2264} {max_possible} possible) \
             \u{2014} file is corrupt or from an unsupported version"
        )));
    }
    Ok(count as usize)
}

pub(crate) fn load_colony_from_file(
    path: &str,
) -> std::io::Result<(TimeNotation, f32, Vec<LoadedRecord>)> {
    let bytes = std::fs::read(path)?;
    let mut c = 0usize;

    // Magic header check. All of v001-v010 load; obsolete brain blocks
    // are dropped (those organisms come up fresh-init).
    if bytes.len() < SAVE_MAGIC.len() {
        return Err(std::io::Error::other("file too short — missing magic"));
    }
    let magic = &bytes[..SAVE_MAGIC.len()];
    let format_v012 = magic == SAVE_MAGIC;
    // v012 == v011 LAYOUT + geometry-rescale marker. Treat format_v011 as "v011
    // layout" so every v011 gate (incl. has_water_level) also covers v012;
    // `format_v012` alone marks colonies already stored at the new 0.5 scale.
    let format_v011 = format_v012 || magic == SAVE_MAGIC_LEGACY_V011;
    let format_v010 = magic == SAVE_MAGIC_LEGACY_V010;
    let format_v009 = magic == SAVE_MAGIC_LEGACY_V009;
    let format_v008 = magic == SAVE_MAGIC_LEGACY_V008;
    let format_v007 = magic == SAVE_MAGIC_LEGACY_V007;
    let format_v006 = magic == SAVE_MAGIC_LEGACY_V006;
    let format_v005 = magic == SAVE_MAGIC_LEGACY_V005;
    let format_v004 = magic == SAVE_MAGIC_LEGACY_V004;
    let format_v003 = magic == SAVE_MAGIC_LEGACY_V003;
    let format_v002 = magic == SAVE_MAGIC_LEGACY_V002;
    let format_v001 = magic == SAVE_MAGIC_LEGACY_V001;
    if !format_v011 && !format_v010 && !format_v009 && !format_v008 && !format_v007 && !format_v006 && !format_v005 && !format_v004 && !format_v003 && !format_v002 && !format_v001 {
        return Err(std::io::Error::other(
            "magic mismatch — not an AEONS colony save (or unsupported version)",
        ));
    }
    // v008..v011 share the v007 per-organism record layout (apart from the
    // movement byte's meaning, the v010 ground_based byte, and the v011 species
    // name — all handled below), so every v007 feature flag also holds for them.
    let format_v007_layout = format_v011 || format_v010 || format_v009 || format_v008 || format_v007;
    // v002+ have the intelligence_level byte after has_variable_form.
    let has_intelligence_byte = format_v007_layout || format_v006 || format_v005 || format_v004 || format_v003 || format_v002;
    // v006+ have the movement byte (before intelligence_level). In v006..v008
    // it is the old sliding/limb bool; in v009+ it is a 4-way movement_mode
    // tag. Older saves default to Sliding.
    let has_movement_byte = format_v007_layout || format_v006;
    // v009+: the movement byte is the new 4-way tag (else the old bool).
    let movement_byte_is_tag = format_v011 || format_v010 || format_v009;
    // v010+: ground_based byte right after the movement byte.
    let has_ground_byte = format_v011 || format_v010;
    // v008+ carry the elapsed-time block right after the magic.
    let has_time_block = format_v011 || format_v010 || format_v009 || format_v008;
    // v009+ carry the global water level after the time block. (v011 was added
    // to this set late — omitting it here mis-read the water f32 AS the organism
    // count, yielding a ~1.1-billion-element allocation that aborted the process.)
    let has_water_level = format_v011 || format_v010 || format_v009;
    // v007+ append a limb-brain block after the sliding-brain block.
    let has_limb_brain_section = format_v007_layout;
    // v011: a per-organism species-name string right after the intelligence byte.
    let has_species_name = format_v011;
    c += SAVE_MAGIC.len();

    // v008+ time notation right after the magic; older formats → t=0.
    let notation = if has_time_block {
        let hours   = read_u32(&bytes, &mut c)?;
        let minutes = read_u32(&bytes, &mut c)?;
        let seconds = read_u32(&bytes, &mut c)?;
        TimeNotation { hours, minutes, seconds }
    } else {
        TimeNotation::default()
    };

    // v009+ global water level, after the time block; older formats default.
    let water_level = if has_water_level {
        read_f32(&bytes, &mut c)?
    } else {
        DEFAULT_WATER_LEVEL
    };

    let count = read_u32(&bytes, &mut c)?;
    let count = checked_count(count, 32, c, bytes.len(), "organism count")?;
    let mut out: Vec<LoadedRecord> = Vec::with_capacity(count);

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
        // Movement byte (v006+); pre-v006 default to Sliding.
        //   * v009: 4-way tag (0=Sliding, 1=LimbBasedWalking, 2=Swimming, 3=Flying).
        //   * v006..v008: old sliding/limb bool (1 => Sliding, 0 => LimbBasedWalking).
        let mut movement_mode = if has_movement_byte {
            let b = read_u8(&bytes, &mut c)?;
            if movement_byte_is_tag {
                match b {
                    0 => MovementMode::Sliding,
                    1 => MovementMode::LimbBasedWalking,
                    2 => MovementMode::Swimming,
                    3 => MovementMode::Flying,
                    other => return Err(std::io::Error::other(
                        format!("unknown movement-mode tag: {other}"),
                    )),
                }
            } else if b != 0 {
                MovementMode::Sliding
            } else {
                MovementMode::LimbBasedWalking
            }
        } else {
            MovementMode::Sliding
        };
        // Invariant (same as spawn_organism): sessile ⇒ Sliding.
        if is_sessile { movement_mode = MovementMode::Sliding; }

        // v010: ground- vs water-based byte; older saves derive it from the
        // movement mode (the only pre-v010 water dwellers were swimmers).
        let ground_based = if has_ground_byte {
            read_u8(&bytes, &mut c)? != 0
        } else {
            movement_mode.default_ground_based()
        };
        // Invariant (same as spawn_organism): a fluid movement mode can never
        // be ground-based.
        let ground_based = ground_based && movement_mode.default_ground_based();

        // Intelligence level: explicit in v002+; synthesised for v001
        // via the deterministic spawn rule of the era.
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

        // v011 species name (u32 len + UTF-8). Empty / absent ⇒ `None` (the
        // organism is classified fresh on the first speciation tick, as before).
        let species_name: Option<String> = if has_species_name {
            let len = read_u32(&bytes, &mut c)? as usize;
            if c + len > bytes.len() {
                return Err(std::io::Error::other("EOF reading species name"));
            }
            let s = String::from_utf8_lossy(&bytes[c..c + len]).into_owned();
            c += len;
            if s.is_empty() { None } else { Some(s) }
        } else {
            None
        };

        let bp_count = read_u32(&bytes, &mut c)?;
        let bp_count = checked_count(bp_count, 16, c, bytes.len(), "body-part count")?;
        let mut body_parts: Vec<BodyPart> = Vec::with_capacity(bp_count);
        for _ in 0..bp_count {
            let bp_kind = match read_u8(&bytes, &mut c)? {
                0 => BodyPartKind::Body,
                1 => BodyPartKind::Limb,
                2 => BodyPartKind::Organ,
                3 => BodyPartKind::Segment,
                4 => BodyPartKind::Static,
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
            let cell_count = checked_count(cell_count, 18, c, bytes.len(), "cell count")?;
            let mut cells: Vec<Cell> = Vec::with_capacity(cell_count);
            for _ in 0..cell_count {
                let local_pos       = read_vec3(&bytes, &mut c)?;
                let cell_type       = read_cell_type(&bytes, &mut c)?;
                let cell_energy     = read_f32 (&bytes, &mut c)?;
                let neighbour_count = read_u8  (&bytes, &mut c)?;
                // Reconstruct the cached PhotosyntheticCell (not serialised).
                let photo = cell_type.is_photo().then(|| {
                    PhotosyntheticCell::new(
                        neighbour_count,
                        crate::energy::PHOTO_PRODUCTION_PER_CELL,
                    )
                });
                cells.push(Cell { local_pos, cell_type, cell_energy, neighbour_count, photo });
            }

            let ocg_count = read_u32(&bytes, &mut c)?;
            let ocg_count = checked_count(ocg_count, 17, c, bytes.len(), "OCG count")?;
            let mut ocg: Vec<(usize, Vec3, CellType)> = Vec::with_capacity(ocg_count);
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

        // Brain section, format depends on magic. v005+ is the current
        // shared `BrainRestore` (restored); v004 / v003 are obsolete
        // architectures whose bytes are consumed-and-dropped (those
        // organisms come up fresh-init); v001/v002 have no section.
        let brain: Option<crate::intelligence_level_herbivore_1_sliding::BrainRestoreHerbivore1>
            = if format_v005 || format_v006 || format_v007_layout {
                // v005+: present-byte + shared `BrainRestore`. (Gate must
                // include v006/v007 or their blocks go unconsumed and
                // mis-align every following record.)
                let brain_present = read_u8(&bytes, &mut c)?;
                if brain_present == 1 {
                    Some(crate::intelligence_level_herbivore_1_sliding::decode_brain_restore(
                        &bytes, &mut c,
                    )?)
                } else {
                    None
                }
            } else if format_v004 {
                // Retired 12-tensor A2C block: present-byte, 14 length-
                // prefixed f32 vectors, then 2 f32 scalars. Consume + drop.
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
                // Retired single-MLP REINFORCE block: present-byte, 6
                // f32 vectors, 2 f32 scalars, 1 u8. Consume + drop.
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

        // v007 limb-brain block: kind tag 0..4 (0=none, 1/2/3 = the limb pools,
        // 4 = swim (v011)). All non-zero kinds carry a `BrainRestoreLimb` payload
        // (swim reuses the struct, swim dims); `spawn_loaded_organism` attaches
        // it and the organism's movement mode routes it to the limb or swim pool.
        let brain_limb: Option<crate::limb_ppo::BrainRestoreLimb> = if has_limb_brain_section {
            let kind = read_u8(&bytes, &mut c)?;
            if kind == 0 {
                None
            } else if kind > 4 {
                return Err(std::io::Error::other(
                    format!("unknown limb-brain kind tag: {kind}"),
                ));
            } else {
                Some(crate::limb_ppo::decode_brain_restore_limb(&bytes, &mut c)?)
            }
        } else {
            None
        };

        // `adult` not serialised — derived: non-variable-form always
        // adult, variable-form adult once at `MAX_CELLS`.
        let total_cells = (photo_cell_count + non_photo_cell_count).max(0) as usize;
        let adult = !has_variable_form
            || total_cells >= crate::volumetric_growth::MAX_CELLS;

        // Geometry migration: pre-v012 colonies store positions at the canonical
        // 1.0 scale; rescale internal geometry ×GEOMETRY_SCALE (NOT the world
        // transform) to match the current organism-geometry scale.
        // recompute_bounding_radius below then derives the correct (scaled) radius.
        if !format_v012 {
            let g = crate::simulation_settings::GEOMETRY_SCALE;
            for bp in body_parts.iter_mut() {
                bp.local_offset *= g;
                if let Some(att) = bp.attachment.as_mut() { att.origin_local *= g; }
                for cell in bp.cells.iter_mut() { cell.local_pos *= g; }
                for e in bp.ocg.iter_mut() { e.1 *= g; }
            }
        }

        let mut organism = Organism {
            body_parts,
            symmetry,
            intelligence_level,
            is_sessile,
            has_variable_form,
            movement_mode,
            ground_based,
            limb_targets: [0.0; 10],
            adult,
            photo_cell_count,
            non_photo_cell_count,
            energy,
            in_sunlight,
            reproduced,
            reproductions,
            // Not serialised — zero on load.
            predations: 0,
            hunger: 0.0,
            dopamine: 0.0,
            // Sentinel = out-of-range; first sensory tick overwrites it.
            target_distance: crate::sensory::SENSORY_RADIUS,
            movement_speed,
            movement_direction,
            velocity,
            is_climbing,
            climb_energy_debt,
            // Derived from body_parts below.
            cached_bounding_radius: 0.0,
            // Structural slots now; brain genes filled by
            // `sync_dna_from_brain_pool` once a slot is claimed.
            dna: crate::lineages::dna::structural_dna(
                kind,
                symmetry,
                has_variable_form,
                is_sessile,
                intelligence_level,
            ),
            // Not persisted (runtime classification); set on the first
            // speciation tick after load.
            species_id: None,
        };
        organism.recompute_bounding_radius();

        out.push(LoadedRecord { pos, rotation, kind, organism, brain, brain_limb, species_name });
    }

    Ok((notation, water_level, out))
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
    let tag = read_u8(buf, c)?;
    CellType::from_tag(tag)
        .ok_or_else(|| std::io::Error::other(format!("unknown cell-type tag: {tag}")))
}

/// Canonical byte tag for a `CellType` in the `.colony` binary format —
/// the single source of truth is `CellType::tag()`.
#[inline]
fn cell_type_byte(ct: CellType) -> u8 {
    ct.tag()
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
