// Species editor — `.species` binary writer.
//
// File layout (little-endian throughout):
//
//   "AEONSS01"                                 8 bytes — magic + version
//   u8  kind                                   0 = Photo, 1 = Hetero
//   u8  symmetry                               0 = NoSymmetry, 1 = Bilateral
//   u8  intelligence_level                     0..=3 (Level0..Level3)
//   u8  has_variable_form                      0/1
//   u8  is_sessile                             0/1
//   u32 ocg_count                              number of OCG entries
//   for each ocg entry:
//       u32 idx
//       3×f32 local position
//       u8 cell_type                           CellType::tag(); 0 = Photo, 1 = Absorption, …
//
// For Bilateral organisms, the OCG written is the RIGHT-half only —
// the import path will mirror to produce the full body, identical to
// the simulation's reproduction pipeline.
//
// The on-disk size for a maxed-out 30-cell species is ~512 bytes.

use bevy::prelude::*;
use std::path::Path;

use crate::cell::{BodyPartKind, CellType};
use crate::colony::{IntelligenceLevel, Symmetry};
use crate::organism::MovementMode;

use super::session::{Classification, Grounding, Metabolism, SpeciesSession};

/// Version deltas (loader accepts v1–v8; older versions default the
/// missing fields as noted):
///   v2: `classification` byte after `is_sessile` (v1 → Herbivore).
///   v3: optional brain block after the OCG: `u8 brain_present`, then if 1
///       the `intelligence_level_herbivore_1_sliding::{en,de}code_brain_restore`
///       payload. Editor saves write 0; the Individuum Navigator's "export
///       trained" workflow writes 1 with a live pool snapshot.
///   v4: MULTIPLE body parts — `u32 body_part_count`, then per part
///       `u32 name_len`/name/`u32 ocg_count`/OCG. Part 0 = base body.
///       v1–v3 (single OCG) load as one "Base Body" part.
///   v5: per-part `u8 is_limb` after the name (limbs rotate about their
///       first cell at runtime). v1–v4 → false.
///   v6: `u8 sliding` after `classification` (1 = sliding, 0 = limb).
///       Pre-v6 → true (the pre-PPO default).
///   v7: per-part `u32 parent` after `is_limb` (0 = base; another limb = sub-limb).
///   v8: the movement byte widens to the 4-way `MovementMode` mapping —
///       {0=>LimbBasedWalking, 1=>Sliding, 2=>Swimming, 3=>Flying}. v6/v7 keep
///       the 2-way {0=>LimbBasedWalking, 1=>Sliding}; the 0/1 encoding is
///       preserved so v6/v7 sliding & limb files load identically.
///   v9: `u8 ground_based` after the movement byte (1 = ground-based, 0 =
///       water-based — floating phototrophs). Pre-v9 derives it from the
///       movement mode (`default_ground_based()`).
///   v10: the brain block's leading byte becomes a brain-KIND tag
///        {0 = none, 1 = sliding (`BrainRestore`), 2 = PPO walker/swim
///        (`BrainRestoreLimb`-shaped payload; the spawned organism's movement
///        mode routes it to the limb or swim pool)}. Written ONLY by the
///        trained-export path (`encode_species_with_brain`); editor saves stay
///        v9 with the old `brain_present` byte. The two are byte-compatible for
///        kind 0/1, so a v10 reader still loads v9 files.
///   v11: the per-part `u8 is_limb` (v5) becomes a `u8 kind` BodyPartKind tag
///        {0=Body, 1=Limb, 2=Organ, 3=Segment, 4=Static}. Drives mirroring +
///        joint wiring at spawn (Limb splits into a Bilateral pair; Segment/Static
///        fuse to one midline part; Static gets a rigid fixed joint, no brain
///        link). Pre-v11 `is_limb`: true→Limb, false→Organ (both moved+mirrored,
///        so this preserves runtime behaviour). Editor saves AND trained exports
///        both write v11; the brain block keeps the v10 KIND-tag scheme.
///   v12: OCG positions stored at the new 0.5 geometry scale (no load-time
///        rescale); layout otherwise identical to v11.
///   v13: the v9 grounding byte widens from a BOOL to a tri-state tag
///        {0 = water, 1 = ground, 2 = ocean-floor (benthic, seafloor-spawned)}.
///        The 0/1 values are unchanged, so v9–v12 files load identically.
const MAGIC_V13:    &[u8; 8] = b"AEONSS13";
const MAGIC_V12:    &[u8; 8] = b"AEONSS12";
const MAGIC_V11:    &[u8; 8] = b"AEONSS11";
const MAGIC_V10:    &[u8; 8] = b"AEONSS10";
const MAGIC_V9:     &[u8; 8] = b"AEONSS09";
const MAGIC_V8:     &[u8; 8] = b"AEONSS08";
const MAGIC_V7:     &[u8; 8] = b"AEONSS07";
const MAGIC_V6:     &[u8; 8] = b"AEONSS06";
const MAGIC_V5:     &[u8; 8] = b"AEONSS05";
const MAGIC_V4:     &[u8; 8] = b"AEONSS04";
const MAGIC_V3:     &[u8; 8] = b"AEONSS03";
const MAGIC_V2:     &[u8; 8] = b"AEONSS02";
const MAGIC_V1:     &[u8; 8] = b"AEONSS01";
const MAGIC: &[u8; 8] = MAGIC_V13;


pub fn dispatch_save_requests(mut session: ResMut<SpeciesSession>) {
    let Some(path) = session.save_requested.take() else { return };

    // Ensure the parent directory exists (cross-platform).
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                error!("species save: failed to create dir {}: {}", parent.display(), e);
                return;
            }
        }
    }

    let bytes = encode_species(&session);
    match std::fs::write(&path, &bytes) {
        Ok(()) => {
            let cells: usize = session.body_parts.iter().map(|p| p.ocg.len()).sum();
            info!(
                "species saved to {} — {} body parts, {} cells, {} bytes",
                path.display(), session.body_parts.len(), cells, bytes.len(),
            );
            session.dirty = false;
        }
        Err(e) => error!("species save: write to {} failed: {}", path.display(), e),
    }
}


fn encode_species(session: &SpeciesSession) -> Vec<u8> {
    let total_cells: usize = session.body_parts.iter().map(|p| p.ocg.len()).sum();
    let mut buf: Vec<u8> = Vec::with_capacity(64 + total_cells * 17);
    buf.extend_from_slice(MAGIC);

    let kind_byte: u8 = match session.draft.metabolism {
        Metabolism::Photoautotroph => 0,
        Metabolism::Heterotroph    => 1,
    };
    let sym_byte: u8 = match session.draft.symmetry {
        Symmetry::NoSymmetry => 0,
        Symmetry::Bilateral  => 1,
    };
    let intel_byte: u8 = match session.draft.intelligence {
        IntelligenceLevel::Level0 => 0,
        IntelligenceLevel::Level1 => 1,
        IntelligenceLevel::Level2 => 2,
        IntelligenceLevel::Level3 => 3,
    };
    let var_byte: u8     = if session.draft.form.as_bool() { 1 } else { 0 };
    // is_sessile comes from the Mobility cycler; the editor lets form and
    // sessility be picked independently — spawn enforces the invariant.
    let sessile_byte: u8 = if session.draft.mobility.is_sessile() { 1 } else { 0 };

    let classification_byte: u8 = match session.draft.classification {
        Classification::Herbivore => 0,
        Classification::Carnivore => 1,
    };

    buf.push(kind_byte);
    buf.push(sym_byte);
    buf.push(intel_byte);
    buf.push(var_byte);
    buf.push(sessile_byte);
    buf.push(classification_byte);
    // v8: movement paradigm byte. Preserves the v6/v7 0/1 encoding
    // (LimbBasedWalking=0, Sliding=1) and extends it (Swimming=2, Flying=3).
    // NOTE: this is DELIBERATELY the inverse of the .colony format's tag
    // (Sliding=0, LimbBasedWalking=1 — see colony_save_load.rs), forced by
    // .species v6/v7 back-compat. Do NOT "align" the two — it breaks round-trips.
    buf.push(movement_byte(session.draft.movement));
    // v9/v13: grounding byte. The EFFECTIVE tri-state tag (phototroph cycler
    // choice, heterotroph movement-derived, movement-clamped) so the byte
    // equals the spawned organism's placement + behaviour. 0/1 match the
    // pre-v13 bool (water/ground); 2 = ocean-floor (benthic).
    buf.push(session.draft.effective_grounding().to_species_tag());

    // Skip empty parts (appendage begun but never given a cell). Filtering
    // shifts indices, so each part's `parent` (index into the FULL list) must
    // be REMAPPED to its index in the filtered list.
    let kept: Vec<usize> = session.body_parts.iter().enumerate()
        .filter(|(_, p)| !p.ocg.is_empty()).map(|(i, _)| i).collect();
    let mut remap = vec![0usize; session.body_parts.len()];
    for (filtered_i, &full_i) in kept.iter().enumerate() { remap[full_i] = filtered_i; }
    buf.extend_from_slice(&(kept.len() as u32).to_le_bytes());
    for &full_i in &kept {
        let part = &session.body_parts[full_i];
        // A parent always has cells (contact-derived) → kept; defaults to 0 = base.
        let parent_filtered = remap.get(part.parent).copied().unwrap_or(0);
        write_body_part(&mut buf, &part.name, part.kind, parent_filtered, &part.ocg);
    }

    // Brain block — editor saves write 0 (weights regenerated at spawn).
    // Only `encode_species_with_brain` writes a non-zero block.
    buf.push(0);
    buf
}

/// Movement-byte encoding for the v8 `.species` format. Preserves the v6/v7
/// 0/1 mapping (LimbBasedWalking=0, Sliding=1) and extends it (Swimming=2,
/// Flying=3) so old sliding/limb files round-trip unchanged.
fn movement_byte(m: MovementMode) -> u8 {
    match m {
        MovementMode::LimbBasedWalking => 0,
        MovementMode::Sliding          => 1,
        MovementMode::Swimming         => 2,
        MovementMode::Flying           => 3,
    }
}

/// v11 per-part kind tag. Same mapping as the `.colony` format
/// (`colony_save_load::cell_type_byte`'s sibling) for cross-format consistency.
fn body_part_kind_byte(k: BodyPartKind) -> u8 {
    match k {
        BodyPartKind::Body    => 0,
        BodyPartKind::Limb    => 1,
        BodyPartKind::Organ   => 2,
        BodyPartKind::Segment => 3,
        BodyPartKind::Static  => 4,
    }
}

/// Write one body part: `u32 name_len`, name bytes, `u8 kind` (v11 BodyPartKind
/// tag — replaces the v5 `is_limb` bool), `u32 parent` (v7+ — index of the parent
/// body part; 0 = main body), `u32 ocg_count`, then the OCG entries (`u32 idx`,
/// 3×`f32` pos, `u8` cell_type).
fn write_body_part(buf: &mut Vec<u8>, name: &str, kind: BodyPartKind, parent: usize, ocg: &[(usize, Vec3, CellType)]) {
    let name_bytes = name.as_bytes();
    buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(name_bytes);
    buf.push(body_part_kind_byte(kind));                     // v11
    buf.extend_from_slice(&(parent as u32).to_le_bytes());   // v7
    buf.extend_from_slice(&(ocg.len() as u32).to_le_bytes());
    for &(idx, pos, ct) in ocg {
        buf.extend_from_slice(&(idx as u32).to_le_bytes());
        buf.extend_from_slice(&pos.x.to_le_bytes());
        buf.extend_from_slice(&pos.y.to_le_bytes());
        buf.extend_from_slice(&pos.z.to_le_bytes());
        buf.push(ct.tag());
    }
}


/// A trained-organism export: the full body plan + a brain payload, written as
/// a v10 `.species` (brain block = KIND tag + payload). The Individuum-Navigator
/// "export" worker builds `parts` and picks the brain kind:
///   * Sliding herbivore — a single bilateral right-half "Base Body" part (as
///     the old export did), `symmetry = Bilateral`, `movement = Sliding`.
///   * Walker / swimmer — the LITERAL multi-part body plan (each runtime part
///     verbatim: `is_limb` from `BodyPartKind::Limb`, `parent` from the
///     attachment, full OCG), `symmetry = NoSymmetry`, the real movement mode.
///     No bilateral re-mirroring, so respawn rebuilds the exact structure the
///     policy was trained on.
/// `parts`: `(name, kind, parent_index, ocg)`, index 0 = base body.
pub fn encode_species_with_brain(
    metabolism:        Metabolism,
    symmetry:          Symmetry,
    intelligence:      IntelligenceLevel,
    has_variable_form: bool,
    is_sessile:        bool,
    classification:    Classification,
    movement:          MovementMode,
    ground_based:      bool,
    parts:             &[(String, BodyPartKind, usize, Vec<(usize, Vec3, CellType)>)],
    brain:             &LoadedBrain,
) -> Vec<u8> {
    let total_cells: usize = parts.iter().map(|p| p.3.len()).sum();
    let mut buf: Vec<u8> = Vec::with_capacity(64 + total_cells * 17 + 64 * 1024);
    buf.extend_from_slice(MAGIC_V13);

    let kind_byte: u8 = match metabolism {
        Metabolism::Photoautotroph => 0,
        Metabolism::Heterotroph    => 1,
    };
    let sym_byte: u8 = match symmetry {
        Symmetry::NoSymmetry => 0,
        Symmetry::Bilateral  => 1,
    };
    let intel_byte: u8 = match intelligence {
        IntelligenceLevel::Level0 => 0,
        IntelligenceLevel::Level1 => 1,
        IntelligenceLevel::Level2 => 2,
        IntelligenceLevel::Level3 => 3,
    };
    buf.push(kind_byte);
    buf.push(sym_byte);
    buf.push(intel_byte);
    buf.push(if has_variable_form { 1 } else { 0 });
    buf.push(if is_sessile { 1 } else { 0 });
    buf.push(match classification {
        Classification::Herbivore => 0,
        Classification::Carnivore => 1,
    });
    // Movement byte — 4-way `MovementMode` (v8+ mapping).
    buf.push(match movement {
        MovementMode::LimbBasedWalking => 0,
        MovementMode::Sliding          => 1,
        MovementMode::Swimming         => 2,
        MovementMode::Flying           => 3,
    });
    // v9/v13: grounding byte (tri-state tag — 0 water, 1 ground, 2 ocean-floor).
    // A live organism carries only the `ground_based` bool, so an export can only
    // ever be water (0) or ground (1) — never ocean-floor (re-mark it in the editor).
    buf.push(if ground_based { 1 } else { 0 });

    buf.extend_from_slice(&(parts.len() as u32).to_le_bytes());
    for (name, kind, parent, ocg) in parts {
        write_body_part(&mut buf, name, *kind, *parent, ocg);
    }

    // Brain block: KIND tag {1 = sliding, 2 = PPO walker/swim} + payload.
    match brain {
        LoadedBrain::Sliding(b) => {
            buf.push(1);
            crate::intelligence_level_herbivore_1_sliding::encode_brain_restore(&mut buf, b);
        }
        LoadedBrain::Ppo(b) => {
            buf.push(2);
            crate::limb_ppo::encode_brain_restore_limb(&mut buf, b);
        }
    }
    buf
}


/// Parse a `.species` file (v1–v8). Public for the (later) import path.
/// Errors on magic mismatch / truncated record / unknown enum tag.
#[allow(dead_code)]
pub fn load_species(path: &Path) -> std::io::Result<LoadedSpecies> {
    let bytes = std::fs::read(path)?;
    decode_species(&bytes)
}


/// One decoded body part: its name and right-half (Bilateral) / full
/// (NoSymmetry) OCG, exactly as stored. Index 0 is the base body.
#[derive(Clone, Debug)]
pub struct LoadedBodyPart {
    pub name: String,
    /// Part kind (v11 `BodyPartKind` tag). Pre-v11 files map `is_limb`:
    /// true→`Limb`, false→`Organ`; v1–v4 default to `Limb`. The base body
    /// (index 0) ignores this (always spawns as `Body`).
    pub kind: BodyPartKind,
    /// Index of the parent body part (0 = main body). A value pointing at
    /// another limb makes this a sub-limb. v1–v6 files default to `0`.
    pub parent: usize,
    pub ocg:  Vec<(usize, Vec3, CellType)>,
}

/// A brain payload decoded from a `.species` file. The variant selects which
/// restore COMPONENT the spawner attaches; for `Ppo` the spawned organism's
/// movement mode then routes it to the limb (walker) or swim pool — both PPO
/// pools share the `BrainRestoreLimb` payload struct (different dims, validated
/// at restore).
#[derive(Clone, Debug)]
pub enum LoadedBrain {
    Sliding(crate::intelligence_level_herbivore_1_sliding::BrainRestoreHerbivore1),
    Ppo(crate::limb_ppo::BrainRestoreLimb),
}

/// Decoded species record. Mirrors `SpeciesSession`'s payload but with
/// all enums resolved to their canonical types.
#[allow(dead_code)]
pub struct LoadedSpecies {
    pub metabolism:        Metabolism,
    pub symmetry:          Symmetry,
    pub intelligence:      IntelligenceLevel,
    pub has_variable_form: bool,
    pub is_sessile:        bool,
    /// v1 files default this to `Herbivore`; v2 reads it from disk.
    pub classification:    Classification,
    /// Movement paradigm. v1–v5 files default to `Sliding`; v6/v7 read the
    /// 2-way byte; v8+ reads the 4-way `MovementMode` byte. Mapped directly
    /// into `Organism::movement_mode` at spawn.
    pub movement:          MovementMode,
    /// Ground- vs water-based (v9+; older files derive it from `movement`).
    /// `false` for floating phototrophs and all fluid movement modes. Mapped
    /// directly into `Organism::ground_based` at spawn.
    pub ground_based:      bool,
    /// Benthic (ocean-floor) phototroph (v13+): a ground-based alga that SPAWNS
    /// on submerged terrain. Only consulted by `spawn_species_instance` for
    /// placement — the runtime organism is plain ground-based. `false` for all
    /// older files and every non-ocean-floor species.
    pub ocean_floor:       bool,
    /// All body parts (index 0 = base body). v1–v3 files yield a single
    /// "Base Body" part.
    pub body_parts:        Vec<LoadedBodyPart>,
    /// `Some` when the file carries trained brain weights (v3+ sliding, or v10
    /// PPO walker/swim). Spawning from such a species attaches the matching
    /// restore component to every copy so all spawned instances boot with the
    /// saved weights instead of fresh random init.
    pub brain: Option<LoadedBrain>,
}


#[allow(dead_code)]
fn decode_species(bytes: &[u8]) -> std::io::Result<LoadedSpecies> {
    use std::io::{Error, ErrorKind};
    fn err(msg: &str) -> Error { Error::new(ErrorKind::InvalidData, msg.to_string()) }

    if bytes.len() < 8 + 5 + 4 { return Err(err("species file too short for header")); }
    let is_v13 = &bytes[..8] == MAGIC_V13;
    // v13 only widens the grounding byte to a tri-state (adds ocean-floor); its
    // layout + geometry scale are identical to v12. Folding it into `is_v12`
    // makes every v12/v11 gate (scaled OCG, kind tag, parent, brain block) cover
    // v13 too — only the grounding read interprets the tag (always tag-decoded).
    let is_v12 = is_v13 || &bytes[..8] == MAGIC_V12;
    // v12 == v11 LAYOUT + a geometry-rescale marker (OCG stored at the new 0.5
    // scale). Treating `is_v11` as "v11 layout" makes every v11 gate (kind tag,
    // parent, brain block) also cover v12/v13; `is_v12` marks scaled files.
    let is_v11 = is_v12 || &bytes[..8] == MAGIC_V11;
    let is_v10 = &bytes[..8] == MAGIC_V10;
    let is_v9 = &bytes[..8] == MAGIC_V9;
    let is_v8 = &bytes[..8] == MAGIC_V8;
    let is_v7 = &bytes[..8] == MAGIC_V7;
    let is_v6 = &bytes[..8] == MAGIC_V6;
    let is_v5 = &bytes[..8] == MAGIC_V5;
    let is_v4 = &bytes[..8] == MAGIC_V4;
    let is_v3 = &bytes[..8] == MAGIC_V3;
    let is_v2 = &bytes[..8] == MAGIC_V2;
    let is_v1 = &bytes[..8] == MAGIC_V1;
    if !is_v11 && !is_v10 && !is_v9 && !is_v8 && !is_v7 && !is_v6 && !is_v5 && !is_v4 && !is_v3 && !is_v2 && !is_v1 {
        return Err(err("species magic mismatch (expected AEONSS01..AEONSS13)"));
    }

    let mut c = 8usize;
    let kind_byte    = bytes[c]; c += 1;
    let sym_byte     = bytes[c]; c += 1;
    let intel_byte   = bytes[c]; c += 1;
    let var_byte     = bytes[c]; c += 1;
    let sessile_byte = bytes[c]; c += 1;
    let classification = if is_v2 || is_v3 || is_v4 || is_v5 || is_v6 || is_v7 || is_v8 || is_v9 || is_v10 || is_v11 {
        let b = bytes[c]; c += 1;
        match b {
            0 => Classification::Herbivore,
            1 => Classification::Carnivore,
            _ => return Err(err("unknown classification tag")),
        }
    } else {
        Classification::Herbivore   // v1 default
    };
    // Movement byte after classification. v8+ uses the 4-way MovementMode
    // mapping {0=>LimbBasedWalking,1=>Sliding,2=>Swimming,3=>Flying}; v6/v7 keep
    // the 2-way {0=>LimbBasedWalking,1=>Sliding}; older files default Sliding.
    let movement = if is_v8 || is_v9 || is_v10 || is_v11 {
        let b = bytes[c]; c += 1;
        match b {
            0 => MovementMode::LimbBasedWalking,
            1 => MovementMode::Sliding,
            2 => MovementMode::Swimming,
            3 => MovementMode::Flying,
            _ => return Err(err("unknown movement tag")),
        }
    } else if is_v6 || is_v7 {
        let b = bytes[c]; c += 1;
        match b {
            0 => MovementMode::LimbBasedWalking,
            1 => MovementMode::Sliding,
            _ => return Err(err("unknown movement tag")),
        }
    } else {
        MovementMode::Sliding
    };

    // v9: grounding byte after movement; older files derive it from the
    // movement mode. v9–v12 stored a BOOL (0 = water, 1 = ground); v13 widened
    // it to a tri-state tag (0 = water, 1 = ground, 2 = ocean-floor benthic) —
    // the 0/1 values are unchanged, so old files decode identically via the same
    // tag map. Clamped so a fluid movement mode can never be ground-anchored
    // (the `spawn_organism` invariant); an ocean-floor tag on a fluid mode
    // degrades to water-based. (`is_v11` already covers v12/v13.)
    let grounding = if is_v9 || is_v10 || is_v11 {
        let b = bytes[c]; c += 1;
        let raw = Grounding::from_species_tag(b);
        if raw.is_ground_based() && !movement.default_ground_based() {
            Grounding::WaterBased
        } else {
            raw
        }
    } else if movement.default_ground_based() {
        Grounding::GroundBased
    } else {
        Grounding::WaterBased
    };
    let ground_based = grounding.is_ground_based();
    let ocean_floor  = grounding.is_ocean_floor();

    let metabolism = match kind_byte {
        0 => Metabolism::Photoautotroph,
        1 => Metabolism::Heterotroph,
        _ => return Err(err("unknown metabolism tag")),
    };
    let symmetry = match sym_byte {
        0 => Symmetry::NoSymmetry,
        1 => Symmetry::Bilateral,
        _ => return Err(err("unknown symmetry tag")),
    };
    let intelligence = match intel_byte {
        0 => IntelligenceLevel::Level0,
        1 => IntelligenceLevel::Level1,
        2 => IntelligenceLevel::Level2,
        3 => IntelligenceLevel::Level3,
        _ => return Err(err("unknown intelligence_level tag")),
    };
    let has_variable_form = var_byte != 0;
    let is_sessile        = sessile_byte != 0;

    // Body parts. v4/v5/v6 store a count + named parts; v1–v3 store a single
    // OCG which we wrap as one "Base Body" part.
    let mut body_parts: Vec<LoadedBodyPart> = if is_v4 || is_v5 || is_v6 || is_v7 || is_v8 || is_v9 || is_v10 || is_v11 {
        if bytes.len() < c + 4 { return Err(err("missing body_part_count")); }
        let count = u32::from_le_bytes(bytes[c..c+4].try_into().unwrap()) as usize; c += 4;
        // v5+ carries a per-part attribute byte; v11 makes it a `kind` tag (else `is_limb`).
        let has_attr_byte = is_v5 || is_v6 || is_v7 || is_v8 || is_v9 || is_v10 || is_v11;
        let attr_is_kind  = is_v11;
        let has_parent    = is_v7 || is_v8 || is_v9 || is_v10 || is_v11;
        let mut parts = Vec::with_capacity(count);
        for _ in 0..count {
            parts.push(read_species_body_part(bytes, &mut c, has_attr_byte, attr_is_kind, has_parent)?);
        }
        parts
    } else {
        if bytes.len() < c + 4 { return Err(err("missing ocg_count")); }
        let ocg_count = u32::from_le_bytes(bytes[c..c+4].try_into().unwrap()) as usize; c += 4;
        let ocg = read_species_ocg(bytes, &mut c, ocg_count)?;
        vec![LoadedBodyPart { name: "Base Body".to_string(), kind: BodyPartKind::Body, parent: 0, ocg }]
    };

    // Brain block. v10: a KIND tag {0 none, 1 sliding, 2 PPO walker/swim}.
    // v3–v9: the old `u8 brain_present` (0/1, sliding only). v1/v2 carry none.
    let brain: Option<LoadedBrain> = if is_v10 || is_v11 {
        if c >= bytes.len() {
            None
        } else {
            let kind = bytes[c]; c += 1;
            match kind {
                0 => None,
                1 => Some(LoadedBrain::Sliding(
                    crate::intelligence_level_herbivore_1_sliding::decode_brain_restore(bytes, &mut c)?)),
                2 => Some(LoadedBrain::Ppo(
                    crate::limb_ppo::decode_brain_restore_limb(bytes, &mut c)?)),
                _ => return Err(err("unknown brain kind tag")),
            }
        }
    } else if is_v3 || is_v4 || is_v5 || is_v6 || is_v7 || is_v8 || is_v9 {
        if c >= bytes.len() {
            None
        } else {
            let brain_present = bytes[c]; c += 1;
            if brain_present == 1 {
                Some(LoadedBrain::Sliding(
                    crate::intelligence_level_herbivore_1_sliding::decode_brain_restore(bytes, &mut c)?))
            } else {
                None
            }
        }
    } else {
        None
    };

    // Geometry migration: pre-v12 files store OCG positions at the canonical 1.0
    // scale; rescale ×GEOMETRY_SCALE to match the current organism-geometry scale.
    // Done BEFORE spawn so bilateral expansion uses the (scaled) MIN_X_BILATERAL.
    if !is_v12 {
        let g = crate::simulation_settings::GEOMETRY_SCALE;
        for part in body_parts.iter_mut() {
            for e in part.ocg.iter_mut() { e.1 *= g; }
        }
    }

    Ok(LoadedSpecies {
        metabolism, symmetry, intelligence, has_variable_form, is_sessile,
        classification, movement, ground_based, ocean_floor, body_parts, brain,
    })
}

/// Read one body part (name [+ attr byte if v5+] [+ parent u32 if v7+] + OCG),
/// advancing `c`. `has_attr_byte` = v5+; when `attr_is_kind` (v11+) the byte is a
/// `BodyPartKind` tag, else the legacy `is_limb` bool (true→Limb, false→Organ —
/// both moved+mirrored pre-v11, so this preserves runtime behaviour).
fn read_species_body_part(bytes: &[u8], c: &mut usize, has_attr_byte: bool, attr_is_kind: bool, has_parent: bool) -> std::io::Result<LoadedBodyPart> {
    let err = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, m.to_string());
    if bytes.len() < *c + 4 { return Err(err("body part name length truncated")); }
    let name_len = u32::from_le_bytes(bytes[*c..*c+4].try_into().unwrap()) as usize; *c += 4;
    if bytes.len() < *c + name_len { return Err(err("body part name truncated")); }
    let name = String::from_utf8_lossy(&bytes[*c..*c+name_len]).into_owned(); *c += name_len;
    let kind = if has_attr_byte {
        if bytes.len() < *c + 1 { return Err(err("body part kind/is_limb truncated")); }
        let b = bytes[*c]; *c += 1;
        if attr_is_kind {
            match b {
                0 => BodyPartKind::Body,
                1 => BodyPartKind::Limb,
                2 => BodyPartKind::Organ,
                3 => BodyPartKind::Segment,
                4 => BodyPartKind::Static,
                _ => return Err(err("unknown body-part kind tag")),
            }
        } else if b != 0 {
            BodyPartKind::Limb
        } else {
            BodyPartKind::Organ
        }
    } else {
        BodyPartKind::Limb   // v1–v4: no attr byte; appendages were motorized
    };
    let parent = if has_parent {
        if bytes.len() < *c + 4 { return Err(err("body part parent truncated")); }
        let p = u32::from_le_bytes(bytes[*c..*c+4].try_into().unwrap()) as usize; *c += 4;
        p
    } else {
        0
    };
    if bytes.len() < *c + 4 { return Err(err("body part ocg_count truncated")); }
    let ocg_count = u32::from_le_bytes(bytes[*c..*c+4].try_into().unwrap()) as usize; *c += 4;
    let ocg = read_species_ocg(bytes, c, ocg_count)?;
    Ok(LoadedBodyPart { name, kind, parent, ocg })
}

/// Read `count` OCG entries advancing `c`.
fn read_species_ocg(bytes: &[u8], c: &mut usize, count: usize) -> std::io::Result<Vec<(usize, Vec3, CellType)>> {
    let err = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, m.to_string());
    let mut ocg = Vec::with_capacity(count);
    for _ in 0..count {
        if bytes.len() < *c + 4 + 12 + 1 { return Err(err("ocg entry truncated")); }
        let idx = u32::from_le_bytes(bytes[*c..*c+4].try_into().unwrap()) as usize; *c += 4;
        let x = f32::from_le_bytes(bytes[*c..*c+4].try_into().unwrap()); *c += 4;
        let y = f32::from_le_bytes(bytes[*c..*c+4].try_into().unwrap()); *c += 4;
        let z = f32::from_le_bytes(bytes[*c..*c+4].try_into().unwrap()); *c += 4;
        let ct_byte = bytes[*c]; *c += 1;
        let ct = CellType::from_tag(ct_byte).ok_or_else(|| err("unknown cell_type tag"))?;
        ocg.push((idx, Vec3::new(x, y, z), ct));
    }
    Ok(ocg)
}
