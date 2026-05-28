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

use super::session::{Classification, Metabolism, SpeciesSession};

/// v3 appends an optional brain block immediately after the OCG
/// entries: one byte `brain_present`, and if 1 the same payload
/// `colony.rs` writes for `AEONS004` colony files (the shared
/// helpers in `intelligence_level_herbivore_1::{encode,decode}_brain_restore`).
/// The species editor itself always writes `brain_present = 0`
/// (random init is regenerated at spawn time, not baked into the
/// file); the "export trained organism as a species" workflow in
/// the Individuum Navigator writes `brain_present = 1` with the
/// organism's live brain snapshotted from the pool.
///
/// v2 introduced the `classification` byte (Herbivore / Carnivore)
/// after `is_sessile`. v1 files are still accepted by
/// `load_species` and default to `Classification::Herbivore`.
/// v4 stores MULTIPLE body parts, each with a UTF-8 name. After the 6
/// metadata bytes it writes `u32 body_part_count`, then per part:
/// `u32 name_len`, name bytes, `u32 ocg_count`, then the OCG entries.
/// The base body is part 0; later parts are appendages. The optional
/// brain block follows the last part. v1–v3 stored a single OCG; the
/// loader wraps those as a single "Base Body" part.
const MAGIC_V4:     &[u8; 8] = b"AEONSS04";
const MAGIC_V3:     &[u8; 8] = b"AEONSS03";
const MAGIC_V2:     &[u8; 8] = b"AEONSS02";
const MAGIC_V1:     &[u8; 8] = b"AEONSS01";
const MAGIC: &[u8; 8] = MAGIC_V4;


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
    // is_sessile now comes from the explicit Mobility cycler, not
    // derived from `form`. Variable-form organisms in the simulation
    // are also sessile by invariant, but the editor lets the user
    // pick them independently — the simulation will enforce its
    // invariant when the species is later imported and spawned.
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

    // v4: multi-part body. Skip empty parts (e.g. an appendage that was
    // begun but never given a cell). body_part_count, then per part.
    let parts: Vec<&super::session::EditorBodyPart> =
        session.body_parts.iter().filter(|p| !p.ocg.is_empty()).collect();
    buf.extend_from_slice(&(parts.len() as u32).to_le_bytes());
    for part in parts {
        write_body_part(&mut buf, &part.name, &part.ocg);
    }

    // Brain block — editor saves write `brain_present = 0`. Weights are
    // regenerated as fresh random init at spawn. Only the "export
    // trained" workflow writes a non-zero brain block, via
    // `encode_species_with_brain`.
    buf.push(0);
    buf
}

/// Write one body part: `u32 name_len`, name bytes, `u32 ocg_count`,
/// then the OCG entries (`u32 idx`, 3×`f32` pos, `u8` cell_type).
fn write_body_part(buf: &mut Vec<u8>, name: &str, ocg: &[(usize, Vec3, CellType)]) {
    let name_bytes = name.as_bytes();
    buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(name_bytes);
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
        });
    }
}


/// Variant of `encode_species` that ALSO carries a saved brain
/// payload. Used by the per-row "export trained organism as a
/// species" button in the Individuum Navigator: the snapshot is
/// taken from the running pool at click time, and the .species
/// file becomes a genuine trained template that any later import
/// will spawn copies of.
///
/// Builds the same metadata header as `encode_species` but writes
/// `brain_present = 1` and serialises the supplied
/// `BrainRestoreHerbivore1` via the shared helper.
pub fn encode_species_with_brain(
    metabolism:        Metabolism,
    symmetry:          Symmetry,
    intelligence:      IntelligenceLevel,
    has_variable_form: bool,
    is_sessile:        bool,
    classification:    Classification,
    ocg:               &[(usize, Vec3, CellType)],
    brain:             &crate::intelligence_level_herbivore_1::BrainRestoreHerbivore1,
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
    // v4: a trained export is a single "Base Body" part.
    buf.extend_from_slice(&1u32.to_le_bytes());
    write_body_part(&mut buf, "Base Body", ocg);
    buf.push(1);   // brain_present
    crate::intelligence_level_herbivore_1::encode_brain_restore(&mut buf, brain);
    buf
}


/// Parse a `.species` file. Public for the (later) import path. Errors
/// on magic mismatch / truncated record / unknown enum tag.
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
    /// All body parts (index 0 = base body). v1–v3 files yield a single
    /// "Base Body" part.
    pub body_parts:        Vec<LoadedBodyPart>,
    /// `Some` when the file carries trained brain weights (v3/v4 with
    /// `brain_present = 1`). Spawning from such a species attaches
    /// the restore payload to every copy so all spawned instances
    /// boot with the saved weights instead of fresh random init.
    pub brain: Option<crate::intelligence_level_herbivore_1::BrainRestoreHerbivore1>,
}


#[allow(dead_code)]
fn decode_species(bytes: &[u8]) -> std::io::Result<LoadedSpecies> {
    use std::io::{Error, ErrorKind};
    fn err(msg: &str) -> Error { Error::new(ErrorKind::InvalidData, msg.to_string()) }

    if bytes.len() < 8 + 5 + 4 { return Err(err("species file too short for header")); }
    let is_v4 = &bytes[..8] == MAGIC_V4;
    let is_v3 = &bytes[..8] == MAGIC_V3;
    let is_v2 = &bytes[..8] == MAGIC_V2;
    let is_v1 = &bytes[..8] == MAGIC_V1;
    if !is_v4 && !is_v3 && !is_v2 && !is_v1 {
        return Err(err("species magic mismatch (expected AEONSS01..AEONSS04)"));
    }

    let mut c = 8usize;
    let kind_byte    = bytes[c]; c += 1;
    let sym_byte     = bytes[c]; c += 1;
    let intel_byte   = bytes[c]; c += 1;
    let var_byte     = bytes[c]; c += 1;
    let sessile_byte = bytes[c]; c += 1;
    let classification = if is_v2 || is_v3 || is_v4 {
        let b = bytes[c]; c += 1;
        match b {
            0 => Classification::Herbivore,
            1 => Classification::Carnivore,
            _ => return Err(err("unknown classification tag")),
        }
    } else {
        // v1 didn't store a classification — default to Herbivore so
        // old saves load as the more common case.
        Classification::Herbivore
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

    // Body parts. v4 stores a count + named parts; v1–v3 store a single
    // OCG which we wrap as one "Base Body" part.
    let body_parts: Vec<LoadedBodyPart> = if is_v4 {
        if bytes.len() < c + 4 { return Err(err("missing body_part_count")); }
        let count = u32::from_le_bytes(bytes[c..c+4].try_into().unwrap()) as usize; c += 4;
        let mut parts = Vec::with_capacity(count);
        for _ in 0..count {
            parts.push(read_species_body_part(bytes, &mut c)?);
        }
        parts
    } else {
        if bytes.len() < c + 4 { return Err(err("missing ocg_count")); }
        let ocg_count = u32::from_le_bytes(bytes[c..c+4].try_into().unwrap()) as usize; c += 4;
        let ocg = read_species_ocg(bytes, &mut c, ocg_count)?;
        vec![LoadedBodyPart { name: "Base Body".to_string(), ocg }]
    };

    // Brain block (v3 + v4). Layout: u8 brain_present, and if 1 the
    // shared BrainRestoreHerbivore1 payload. v1/v2 carry none.
    let brain = if is_v3 || is_v4 {
        if c >= bytes.len() {
            None
        } else {
            let brain_present = bytes[c]; c += 1;
            if brain_present == 1 {
                Some(crate::intelligence_level_herbivore_1::decode_brain_restore(
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
        classification, body_parts, brain,
    })
}

/// Read one v4 body part (name + OCG) advancing `c`.
fn read_species_body_part(bytes: &[u8], c: &mut usize) -> std::io::Result<LoadedBodyPart> {
    let err = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, m.to_string());
    if bytes.len() < *c + 4 { return Err(err("body part name length truncated")); }
    let name_len = u32::from_le_bytes(bytes[*c..*c+4].try_into().unwrap()) as usize; *c += 4;
    if bytes.len() < *c + name_len { return Err(err("body part name truncated")); }
    let name = String::from_utf8_lossy(&bytes[*c..*c+name_len]).into_owned(); *c += name_len;
    if bytes.len() < *c + 4 { return Err(err("body part ocg_count truncated")); }
    let ocg_count = u32::from_le_bytes(bytes[*c..*c+4].try_into().unwrap()) as usize; *c += 4;
    let ocg = read_species_ocg(bytes, c, ocg_count)?;
    Ok(LoadedBodyPart { name, ocg })
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
            _ => return Err(err("unknown cell_type tag")),
        };
        ocg.push((idx, Vec3::new(x, y, z), ct));
    }
    Ok(ocg)
}
