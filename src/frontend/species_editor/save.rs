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

/// v2 introduces the `classification` byte (Herbivore / Carnivore)
/// after `is_sessile`. v1 files are still accepted by `load_species`
/// — they default to `Classification::Herbivore`.
const MAGIC_V2:     &[u8; 8] = b"AEONSS02";
const MAGIC_V1:     &[u8; 8] = b"AEONSS01";
const MAGIC: &[u8; 8] = MAGIC_V2;


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
            info!(
                "species saved to {} — {} cells, {} bytes",
                path.display(), session.ocg.len(), bytes.len(),
            );
            session.dirty = false;
        }
        Err(e) => error!("species save: write to {} failed: {}", path.display(), e),
    }
}


fn encode_species(session: &SpeciesSession) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::with_capacity(64 + session.ocg.len() * 17);
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
    buf.extend_from_slice(&(session.ocg.len() as u32).to_le_bytes());

    for &(idx, pos, ct) in &session.ocg {
        buf.extend_from_slice(&(idx as u32).to_le_bytes());
        buf.extend_from_slice(&pos.x.to_le_bytes());
        buf.extend_from_slice(&pos.y.to_le_bytes());
        buf.extend_from_slice(&pos.z.to_le_bytes());
        let ct_byte: u8 = match ct {
            CellType::Photo    => 0,
            CellType::NonPhoto => 1,
        };
        buf.push(ct_byte);
    }
    buf
}


/// Parse a `.species` file. Public for the (later) import path. Errors
/// on magic mismatch / truncated record / unknown enum tag.
#[allow(dead_code)]
pub fn load_species(path: &Path) -> std::io::Result<LoadedSpecies> {
    let bytes = std::fs::read(path)?;
    decode_species(&bytes)
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
    pub ocg:               Vec<(usize, Vec3, CellType)>,
}


#[allow(dead_code)]
fn decode_species(bytes: &[u8]) -> std::io::Result<LoadedSpecies> {
    use std::io::{Error, ErrorKind};
    fn err(msg: &str) -> Error { Error::new(ErrorKind::InvalidData, msg.to_string()) }

    if bytes.len() < 8 + 5 + 4 { return Err(err("species file too short for header")); }
    let is_v2 = &bytes[..8] == MAGIC_V2;
    let is_v1 = &bytes[..8] == MAGIC_V1;
    if !is_v2 && !is_v1 { return Err(err("species magic mismatch (expected AEONSS01 or AEONSS02)")); }

    let mut c = 8usize;
    let kind_byte    = bytes[c]; c += 1;
    let sym_byte     = bytes[c]; c += 1;
    let intel_byte   = bytes[c]; c += 1;
    let var_byte     = bytes[c]; c += 1;
    let sessile_byte = bytes[c]; c += 1;
    let classification = if is_v2 {
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
    let ocg_count = u32::from_le_bytes(bytes[c..c+4].try_into().unwrap()) as usize; c += 4;

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

    let mut ocg = Vec::with_capacity(ocg_count);
    for _ in 0..ocg_count {
        if bytes.len() < c + 4 + 12 + 1 { return Err(err("ocg entry truncated")); }
        let idx = u32::from_le_bytes(bytes[c..c+4].try_into().unwrap()) as usize; c += 4;
        let x = f32::from_le_bytes(bytes[c..c+4].try_into().unwrap()); c += 4;
        let y = f32::from_le_bytes(bytes[c..c+4].try_into().unwrap()); c += 4;
        let z = f32::from_le_bytes(bytes[c..c+4].try_into().unwrap()); c += 4;
        let ct_byte = bytes[c]; c += 1;
        let ct = match ct_byte {
            0 => CellType::Photo,
            1 => CellType::NonPhoto,
            _ => return Err(err("unknown cell_type tag")),
        };
        ocg.push((idx, Vec3::new(x, y, z), ct));
    }

    Ok(LoadedSpecies {
        metabolism, symmetry, intelligence, has_variable_form, is_sessile, classification, ocg,
    })
}
