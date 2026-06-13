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
//       u8 cell_type                           0 = Photo, 1 = NonPhoto
//
// For Bilateral organisms, the OCG written is the RIGHT-half only —
// the import path will mirror to produce the full body, identical to
// the simulation's reproduction pipeline.
//
// The on-disk size for a maxed-out 30-cell species is ~512 bytes.

use bevy::prelude::*;
use std::path::Path;

use crate::cell::CellType;
use crate::colony::{IntelligenceLevel, Symmetry};
use crate::organism::MovementMode;

use super::session::{Classification, Metabolism, SpeciesSession};

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
const MAGIC_V9:     &[u8; 8] = b"AEONSS09";
const MAGIC_V8:     &[u8; 8] = b"AEONSS08";
const MAGIC_V7:     &[u8; 8] = b"AEONSS07";
const MAGIC_V6:     &[u8; 8] = b"AEONSS06";
const MAGIC_V5:     &[u8; 8] = b"AEONSS05";
const MAGIC_V4:     &[u8; 8] = b"AEONSS04";
const MAGIC_V3:     &[u8; 8] = b"AEONSS03";
const MAGIC_V2:     &[u8; 8] = b"AEONSS02";
const MAGIC_V1:     &[u8; 8] = b"AEONSS01";
const MAGIC: &[u8; 8] = MAGIC_V9;


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
    // v9: ground- vs water-based. The EFFECTIVE value (phototroph cycler
    // choice, heterotroph movement-derived) so the byte equals the spawned
    // organism's `Organism::ground_based`.
    buf.push(session.draft.effective_ground_based() as u8);

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
        write_body_part(&mut buf, &part.name, part.is_limb, parent_filtered, &part.ocg);
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

/// Write one body part: `u32 name_len`, name bytes, `u8 is_limb`,
/// `u32 parent` (v7+ — index of the parent body part; 0 = main body),
/// `u32 ocg_count`, then the OCG entries (`u32 idx`, 3×`f32` pos, `u8`
/// cell_type).
fn write_body_part(buf: &mut Vec<u8>, name: &str, is_limb: bool, parent: usize, ocg: &[(usize, Vec3, CellType)]) {
    let name_bytes = name.as_bytes();
    buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(name_bytes);
    buf.push(if is_limb { 1 } else { 0 });
    buf.extend_from_slice(&(parent as u32).to_le_bytes());   // v7
    buf.extend_from_slice(&(ocg.len() as u32).to_le_bytes());
    for &(idx, pos, ct) in ocg {
        buf.extend_from_slice(&(idx as u32).to_le_bytes());
        buf.extend_from_slice(&pos.x.to_le_bytes());
        buf.extend_from_slice(&pos.y.to_le_bytes());
        buf.extend_from_slice(&pos.z.to_le_bytes());
        buf.push(match ct {
            CellType::Photo       => 0,
            CellType::NonPhoto    => 1,
            CellType::Placeholder => 2,
            CellType::SubLimb     => 3,
            CellType::YellowCell  => 4,
            CellType::OrangeCell  => 5,
            CellType::BrownCell   => 6,
        });
    }
}


/// `encode_species` plus a saved brain payload — the Individuum Navigator's
/// "export trained organism" button, snapshotting the live pool brain. Same
/// header but `brain_present = 1` + the `BrainRestoreHerbivore1` payload.
pub fn encode_species_with_brain(
    metabolism:        Metabolism,
    symmetry:          Symmetry,
    intelligence:      IntelligenceLevel,
    has_variable_form: bool,
    is_sessile:        bool,
    classification:    Classification,
    ocg:               &[(usize, Vec3, CellType)],
    brain:             &crate::intelligence_level_herbivore_1_sliding::BrainRestoreHerbivore1,
) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::with_capacity(64 + ocg.len() * 17 + 32 * 1024);
    buf.extend_from_slice(MAGIC);

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
    let var_byte:     u8 = if has_variable_form { 1 } else { 0 };
    let sessile_byte: u8 = if is_sessile        { 1 } else { 0 };
    let classification_byte: u8 = match classification {
        Classification::Herbivore => 0,
        Classification::Carnivore => 1,
    };
    buf.push(kind_byte);
    buf.push(sym_byte);
    buf.push(intel_byte);
    buf.push(var_byte);
    buf.push(sessile_byte);
    buf.push(classification_byte);
    // Trained exports come from the sliding herbivore_1 brain, so the
    // movement byte is Sliding (1 in both the v6/v7 and v8 mappings).
    buf.push(1);
    // v9: Sliding ⇒ ground-based.
    buf.push(1);
    // v5: a trained export is a single "Base Body" part (never a limb).
    buf.extend_from_slice(&1u32.to_le_bytes());
    write_body_part(&mut buf, "Base Body", false, 0, ocg);
    buf.push(1);   // brain_present
    crate::intelligence_level_herbivore_1_sliding::encode_brain_restore(&mut buf, brain);
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
    /// `true` when the part is flagged as a limb (animated at runtime).
    /// v1–v4 files default this to `false` on load.
    pub is_limb: bool,
    /// Index of the parent body part (0 = main body). A value pointing at
    /// another limb makes this a sub-limb. v1–v6 files default to `0`.
    pub parent: usize,
    pub ocg:  Vec<(usize, Vec3, CellType)>,
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
    /// All body parts (index 0 = base body). v1–v3 files yield a single
    /// "Base Body" part.
    pub body_parts:        Vec<LoadedBodyPart>,
    /// `Some` when the file carries trained brain weights (v3/v4 with
    /// `brain_present = 1`). Spawning from such a species attaches
    /// the restore payload to every copy so all spawned instances
    /// boot with the saved weights instead of fresh random init.
    pub brain: Option<crate::intelligence_level_herbivore_1_sliding::BrainRestoreHerbivore1>,
}


#[allow(dead_code)]
fn decode_species(bytes: &[u8]) -> std::io::Result<LoadedSpecies> {
    use std::io::{Error, ErrorKind};
    fn err(msg: &str) -> Error { Error::new(ErrorKind::InvalidData, msg.to_string()) }

    if bytes.len() < 8 + 5 + 4 { return Err(err("species file too short for header")); }
    let is_v9 = &bytes[..8] == MAGIC_V9;
    let is_v8 = &bytes[..8] == MAGIC_V8;
    let is_v7 = &bytes[..8] == MAGIC_V7;
    let is_v6 = &bytes[..8] == MAGIC_V6;
    let is_v5 = &bytes[..8] == MAGIC_V5;
    let is_v4 = &bytes[..8] == MAGIC_V4;
    let is_v3 = &bytes[..8] == MAGIC_V3;
    let is_v2 = &bytes[..8] == MAGIC_V2;
    let is_v1 = &bytes[..8] == MAGIC_V1;
    if !is_v9 && !is_v8 && !is_v7 && !is_v6 && !is_v5 && !is_v4 && !is_v3 && !is_v2 && !is_v1 {
        return Err(err("species magic mismatch (expected AEONSS01..AEONSS09)"));
    }

    let mut c = 8usize;
    let kind_byte    = bytes[c]; c += 1;
    let sym_byte     = bytes[c]; c += 1;
    let intel_byte   = bytes[c]; c += 1;
    let var_byte     = bytes[c]; c += 1;
    let sessile_byte = bytes[c]; c += 1;
    let classification = if is_v2 || is_v3 || is_v4 || is_v5 || is_v6 || is_v7 || is_v8 || is_v9 {
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
    let movement = if is_v8 || is_v9 {
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

    // v9: ground- vs water-based byte after movement; older files derive it
    // from the movement mode. Clamped so a fluid movement mode can never
    // claim ground-based (same invariant as `spawn_organism`).
    let ground_based = if is_v9 {
        let b = bytes[c]; c += 1;
        (b != 0) && movement.default_ground_based()
    } else {
        movement.default_ground_based()
    };

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
    let body_parts: Vec<LoadedBodyPart> = if is_v4 || is_v5 || is_v6 || is_v7 || is_v8 || is_v9 {
        if bytes.len() < c + 4 { return Err(err("missing body_part_count")); }
        let count = u32::from_le_bytes(bytes[c..c+4].try_into().unwrap()) as usize; c += 4;
        let mut parts = Vec::with_capacity(count);
        for _ in 0..count {
            parts.push(read_species_body_part(bytes, &mut c, is_v5 || is_v6 || is_v7 || is_v8 || is_v9, is_v7 || is_v8 || is_v9)?);
        }
        parts
    } else {
        if bytes.len() < c + 4 { return Err(err("missing ocg_count")); }
        let ocg_count = u32::from_le_bytes(bytes[c..c+4].try_into().unwrap()) as usize; c += 4;
        let ocg = read_species_ocg(bytes, &mut c, ocg_count)?;
        vec![LoadedBodyPart { name: "Base Body".to_string(), is_limb: false, parent: 0, ocg }]
    };

    // Brain block (v3+). Layout: u8 brain_present, and if 1 the shared
    // BrainRestoreHerbivore1 payload. v1/v2 carry none.
    let brain = if is_v3 || is_v4 || is_v5 || is_v6 || is_v7 || is_v8 || is_v9 {
        if c >= bytes.len() {
            None
        } else {
            let brain_present = bytes[c]; c += 1;
            if brain_present == 1 {
                Some(crate::intelligence_level_herbivore_1_sliding::decode_brain_restore(
                    bytes, &mut c,
                )?)
            } else {
                None
            }
        }
    } else {
        None
    };

    Ok(LoadedSpecies {
        metabolism, symmetry, intelligence, has_variable_form, is_sessile,
        classification, movement, ground_based, body_parts, brain,
    })
}

/// Read one body part (name [+ is_limb if v5+] [+ parent u32 if v7+] + OCG),
/// advancing `c`. `has_is_limb` = v5+, `has_parent` = v7+.
fn read_species_body_part(bytes: &[u8], c: &mut usize, has_is_limb: bool, has_parent: bool) -> std::io::Result<LoadedBodyPart> {
    let err = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, m.to_string());
    if bytes.len() < *c + 4 { return Err(err("body part name length truncated")); }
    let name_len = u32::from_le_bytes(bytes[*c..*c+4].try_into().unwrap()) as usize; *c += 4;
    if bytes.len() < *c + name_len { return Err(err("body part name truncated")); }
    let name = String::from_utf8_lossy(&bytes[*c..*c+name_len]).into_owned(); *c += name_len;
    let is_limb = if has_is_limb {
        if bytes.len() < *c + 1 { return Err(err("body part is_limb truncated")); }
        let b = bytes[*c]; *c += 1;
        b != 0
    } else {
        false
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
    Ok(LoadedBodyPart { name, is_limb, parent, ocg })
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
        let ct = match ct_byte {
            0 => CellType::Photo,
            1 => CellType::NonPhoto,
            2 => CellType::Placeholder,
            3 => CellType::SubLimb,
            4 => CellType::YellowCell,
            5 => CellType::OrangeCell,
            6 => CellType::BrownCell,
            _ => return Err(err("unknown cell_type tag")),
        };
        ocg.push((idx, Vec3::new(x, y, z), ct));
    }
    Ok(ocg)
}
