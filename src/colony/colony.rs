use crate::cell::*;
use crate::frontend::ShowGizmo;
use crate::volumetric_growth::{build_mesh_from_ocg, build_smoothed_mesh_from_ocg};
use crate::world_geometry::{HeightmapSampler, MAP_MAX_X, MAP_MAX_Z};
use crate::movement::DirectionTimer;
use bevy::prelude::*;
use rand::prelude::*;

// Re-export the Organism module so existing `use crate::colony::*;` imports
// keep finding Organism, OrganismRoot, OrganismKind, Photoautotroph, etc.
pub use crate::organism::*;

/// Hard cap on simulation population. Both brain pools size their tensors to
/// this constant — exceeding it would silently miss organisms in the batched
/// MLP forward pass.
pub const MAXIMUM_ORGANISMS: usize = 2000;

/// Initial population of each trophic strategy. Photoautotrophs dominate so
/// the food web has plenty of prey for the smaller heterotroph predator pool.
const INITIAL_PHOTOAUTOTROPHS: u32 = 1000;
const INITIAL_HETEROTROPHS:    u32 = 200;

/// Initial Krishi cohort size. `pub` so `krishi.rs` reads it directly —
/// keeps every "how many of X spawn at startup" knob in one place.
pub const INITIAL_KRISHI: u32 = 1;


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
        app.add_systems(Update, spawn_colony.run_if(resource_exists::<HeightmapSampler>));
        app.add_systems(Update, save_colony_system);
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

const SAVE_PATH:  &str    = "src/test.colony";
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
const SAVE_MAGIC:             &[u8;8] = b"AEONS003";
const SAVE_MAGIC_LEGACY_V002: &[u8;8] = b"AEONS002";
const SAVE_MAGIC_LEGACY_V001: &[u8;8] = b"AEONS001";

/// One-shot resource: when set to `true`, `save_colony_system` writes the
/// current world to `SAVE_PATH` on the next Update tick and resets the
/// flag. The Save button in the statistics panel is the only place that
/// flips it on; the user-facing pause it requests is handled by the same
/// click handler.
#[derive(Resource, Default)]
pub struct SaveRequested(pub bool);


fn save_colony_system(
    mut save_requested: ResMut<SaveRequested>,
    organisms: Query<
        (Entity, &Transform, &Organism, Has<Photoautotroph>, Has<Heterotroph>),
        With<OrganismRoot>,
    >,
    // NonSend access to all four brain pools so we can read every
    // assigned slot's GPU weights into CPU vectors once each
    // (`pool.snapshot()` does the four GPU→CPU transfers); after
    // that the per-organism brain extraction is pure CPU work.
    pool_l1p: NonSend<crate::intelligence_level_1_photo::BrainPoolL1Photo>,
    pool_l1h: NonSend<crate::intelligence_level_1_hetero::BrainPoolL1Hetero>,
    pool_l2:  NonSend<crate::intelligence_level_2::BrainPoolL2>,
    pool_l3:  NonSend<crate::intelligence_level_3::BrainPoolL3>,
) {
    if !save_requested.0 { return; }
    save_requested.0 = false;

    // ── Pool snapshots — one set of GPU→CPU transfers per pool. ──
    // Doing this up front (vs per-organism) drops the cost from
    // O(organisms × 4) GPU syncs to O(pools × 4) = 16 total.
    let snap_l1p = pool_l1p.snapshot();
    let snap_l1h = pool_l1h.snapshot();
    let snap_l2  = pool_l2 .snapshot();
    let snap_l3  = pool_l3 .snapshot();

    let mut buf: Vec<u8> = Vec::with_capacity(64 * 1024);
    buf.extend_from_slice(SAVE_MAGIC);

    // Two-pass to write the count up front. iter() over Query is cheap.
    let count = organisms.iter()
        .filter(|(_, _, _, is_photo, is_hetero)| *is_photo || *is_hetero)
        .count() as u32;
    put_u32(&mut buf, count);

    for (entity, transform, org, is_photo, is_hetero) in organisms.iter() {
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
                put_u8(&mut buf, match cell.cell_type {
                    CellType::Photo    => 0,
                    CellType::NonPhoto => 1,
                });
                put_f32(&mut buf, cell.cell_energy);
                put_u8(&mut buf, cell.neighbour_count);
            }

            put_u32(&mut buf, bp.ocg.len() as u32);
            for (idx, pos, ct) in &bp.ocg {
                put_u32(&mut buf, *idx as u32);
                put_vec3(&mut buf, *pos);
                put_u8(&mut buf, match ct {
                    CellType::Photo    => 0,
                    CellType::NonPhoto => 1,
                });
            }
        }

        // ── v003 brain section ───────────────────────────────────
        // Look up this organism in the pool that matches its
        // intelligence level. If the slot exists (it usually does
        // — slot assignment runs in PreUpdate), serialise the
        // weights + REINFORCE state. Otherwise emit `brain_present
        // = 0` and the loader will recreate the slot with the
        // pool's default / recycled weights.
        let brain_data = match (org.intelligence_level, is_photo, is_hetero) {
            (IntelligenceLevel::Level0, _, _) => None,
            (IntelligenceLevel::Level1, true,  _   ) => snap_l1p.extract(entity),
            (IntelligenceLevel::Level1, false, true) => snap_l1h.extract(entity),
            (IntelligenceLevel::Level2, false, true) => snap_l2 .extract(entity),
            (IntelligenceLevel::Level3, false, true) => snap_l3 .extract(entity),
            _ => None,  // shouldn't happen — defensive
        };
        match brain_data {
            None => put_u8(&mut buf, 0),
            Some(b) => {
                put_u8(&mut buf, 1);
                put_u32(&mut buf, b.w1.len() as u32);
                for &x in &b.w1 { put_f32(&mut buf, x); }
                put_u32(&mut buf, b.b1.len() as u32);
                for &x in &b.b1 { put_f32(&mut buf, x); }
                put_u32(&mut buf, b.w2.len() as u32);
                for &x in &b.w2 { put_f32(&mut buf, x); }
                put_u32(&mut buf, b.b2.len() as u32);
                for &x in &b.b2 { put_f32(&mut buf, x); }
                put_u32(&mut buf, b.prev_state.len() as u32);
                for &x in &b.prev_state { put_f32(&mut buf, x); }
                put_u32(&mut buf, b.prev_action.len() as u32);
                for &x in &b.prev_action { put_f32(&mut buf, x); }
                put_f32(&mut buf, b.prev_energy);
                put_f32(&mut buf, b.baseline);
                put_u8 (&mut buf, b.has_prev as u8);
            }
        }
    }

    match std::fs::write(SAVE_PATH, &buf) {
        Ok(())  => info!("colony saved to {} — {} organisms, {} bytes", SAVE_PATH, count, buf.len()),
        Err(e)  => error!("failed to save colony to {}: {}", SAVE_PATH, e),
    }
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
    /// Saved brain weights + REINFORCE state. `None` for Level 0
    /// organisms (no pool to restore into) and for v001/v002 saves
    /// (which predate brain serialisation). When `Some`,
    /// `spawn_loaded_organism` attaches a `BrainRestore` component
    /// and the pool's `assign_brains_*` will install the weights
    /// next PreUpdate.
    brain:     Option<crate::rl_helpers::BrainRestore>,
}


fn load_colony_from_file(path: &str) -> std::io::Result<Vec<LoadedRecord>> {
    let bytes = std::fs::read(path)?;
    let mut c = 0usize;

    // Magic header check. We accept v003 (current — adds brain
    // weights + REINFORCE prev_*), v002 (adds intelligence_level
    // byte), and v001 (oldest — none of those). For v001/v002 the
    // loader synthesises any missing fields from the deterministic
    // spawn-time rules.
    if bytes.len() < SAVE_MAGIC.len() {
        return Err(std::io::Error::other("file too short — missing magic"));
    }
    let magic = &bytes[..SAVE_MAGIC.len()];
    let format_v003 = magic == SAVE_MAGIC;
    let format_v002 = magic == SAVE_MAGIC_LEGACY_V002;
    let format_v001 = magic == SAVE_MAGIC_LEGACY_V001;
    if !format_v003 && !format_v002 && !format_v001 {
        return Err(std::io::Error::other(
            "magic mismatch — not an AEONS colony save (or unsupported version)",
        ));
    }
    // v002+ all share the intelligence_level byte after has_variable_form.
    let has_intelligence_byte = format_v003 || format_v002;
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
                    CellType::NonPhoto => None,
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

        // ── v003 brain section (only if format_v003). ────────────
        // Parsed BEFORE building the Organism so any read errors
        // surface here rather than propagating into the spawn
        // pipeline.
        let brain: Option<crate::rl_helpers::BrainRestore> = if format_v003 {
            let brain_present = read_u8(&bytes, &mut c)?;
            if brain_present == 1 {
                let n_w1 = read_u32(&bytes, &mut c)? as usize;
                let mut w1 = Vec::with_capacity(n_w1);
                for _ in 0..n_w1 { w1.push(read_f32(&bytes, &mut c)?); }
                let n_b1 = read_u32(&bytes, &mut c)? as usize;
                let mut b1 = Vec::with_capacity(n_b1);
                for _ in 0..n_b1 { b1.push(read_f32(&bytes, &mut c)?); }
                let n_w2 = read_u32(&bytes, &mut c)? as usize;
                let mut w2 = Vec::with_capacity(n_w2);
                for _ in 0..n_w2 { w2.push(read_f32(&bytes, &mut c)?); }
                let n_b2 = read_u32(&bytes, &mut c)? as usize;
                let mut b2 = Vec::with_capacity(n_b2);
                for _ in 0..n_b2 { b2.push(read_f32(&bytes, &mut c)?); }
                let n_ps = read_u32(&bytes, &mut c)? as usize;
                let mut prev_state = Vec::with_capacity(n_ps);
                for _ in 0..n_ps { prev_state.push(read_f32(&bytes, &mut c)?); }
                let n_pa = read_u32(&bytes, &mut c)? as usize;
                let mut prev_action = Vec::with_capacity(n_pa);
                for _ in 0..n_pa { prev_action.push(read_f32(&bytes, &mut c)?); }
                let prev_energy = read_f32(&bytes, &mut c)?;
                let baseline    = read_f32(&bytes, &mut c)?;
                let has_prev    = read_u8 (&bytes, &mut c)? != 0;
                Some(crate::rl_helpers::BrainRestore {
                    w1, b1, w2, b2, prev_state, prev_action,
                    prev_energy, baseline, has_prev,
                })
            } else {
                None
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
            adult,
            photo_cell_count,
            non_photo_cell_count,
            energy,
            in_sunlight,
            reproduced,
            reproductions,
            movement_speed,
            movement_direction,
            velocity,
            is_climbing,
            climb_energy_debt,
            // Will be derived from the loaded body_parts on the next
            // line — saving the value would have been redundant.
            cached_bounding_radius: 0.0,
        };
        organism.recompute_bounding_radius();

        out.push(LoadedRecord { pos, rotation, kind, organism, brain });
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
    let LoadedRecord { pos, rotation, kind, organism, brain } = record;
    if organism.body_parts.is_empty() {
        // Defensive — skip organisms with no body parts (should never
        // happen on a valid save).
        return commands.spawn_empty().id();
    }

    let body_parts_snapshot = organism.body_parts.clone();
    // Capture intelligence level + adult flag before `organism` is
    // moved into the spawn bundle.
    let intelligence_level = organism.intelligence_level;
    let adult              = organism.adult;
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

        let parent_entity = match &bp.attachment {
            Some(a) => bp_entities[a.parent_idx],
            None    => root,
        };
        commands.entity(parent_entity).add_child(child);

        bp_entities.push(child);
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
        other => Err(std::io::Error::other(format!("unknown cell-type tag: {other}"))),
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
    mut spawned:     Local<bool>,
) {
    if *spawned { return; }
    *spawned = true;

    let materials = OrganismMaterials::new(&mut materials);
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

    for _ in 0..INITIAL_PHOTOAUTOTROPHS {
        let x = rng.random_range(0.0_f32..MAP_MAX_X);
        let z = rng.random_range(0.0_f32..MAP_MAX_Z);
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
            &mut commands,
            &mut meshes,
            &materials,
            &mut rng,
        );
    }

    for _ in 0..INITIAL_HETEROTROPHS {
        let x = rng.random_range(0.0_f32..MAP_MAX_X);
        let z = rng.random_range(0.0_f32..MAP_MAX_Z);
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
pub struct OrganismMaterials {
    pub photo:       Handle<StandardMaterial>,
    pub hetero:      Handle<StandardMaterial>,
    pub debug_blue:  Handle<StandardMaterial>,
}

impl OrganismMaterials {
    pub fn new(materials: &mut Assets<StandardMaterial>) -> Self {
        Self {
            photo:      materials.add(StandardMaterial {
                base_color: Color::srgb(0.2, 0.8, 0.2),
                ..default()
            }),
            hetero:     materials.add(StandardMaterial {
                base_color: Color::srgb(0.8, 0.2, 0.2),
                ..default()
            }),
            debug_blue: materials.add(StandardMaterial {
                base_color: Color::srgb(0.2, 0.4, 0.95),
                ..default()
            }),
        }
    }

    /// Pick the material handle for one body part, given the trophic kind of
    /// its owning organism.
    pub fn handle_for(&self, kind: OrganismKind, bp: &BodyPart) -> Handle<StandardMaterial> {
        if bp.debug_blue {
            self.debug_blue.clone()
        } else {
            match kind {
                OrganismKind::Photoautotroph => self.photo.clone(),
                OrganismKind::Heterotroph    => self.hetero.clone(),
            }
        }
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
        adult,
        photo_cell_count:     0,
        non_photo_cell_count: 0,
        energy: initial_energy.max(0.0),
        in_sunlight: false,
        reproduced: false,
        reproductions: 0,
        movement_speed: speed,
        movement_direction: direction,
        velocity: Vec3::ZERO,
        is_climbing: false,
        climb_energy_debt: 0.0,
        // Populated by recompute_cell_counts below.
        cached_bounding_radius: 0.0,
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

        // Parent the entity: branches under their parent body-part entity,
        // root under the OrganismRoot.
        let parent_entity = match &bp.attachment {
            Some(a) => bp_entities[a.parent_idx],
            None    => root,
        };
        commands.entity(parent_entity).add_child(child);

        bp_entities.push(child);
    }

    root
}
