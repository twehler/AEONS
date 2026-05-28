// Editor → .colony writer.
//
// We re-implement the v003 layout (described in `colony.rs`'s save
// header comment) here so the editor stays self-contained — we
// can't call into `save_colony_system` because that wants live
// `Organism` components and brain pools, neither of which exist in
// the editor. brain_present is always 0; the loader will assign
// default brain weights to every organism.

use std::io::Write;

use bevy::prelude::*;

use crate::cell::{BodyPartKind, CellType};
use crate::organism::{IntelligenceLevel, Symmetry};
use crate::colony_editor::template::{Metabolism, OrganismTemplate};
use crate::energy::MAX_ENERGY_PER_CELL;


const SAVE_MAGIC: &[u8; 8] = b"AEONS003";


/// Write every organism in `templates` to `path` in the v003 .colony
/// format. The file is overwritten on each call.
pub fn write_colony(path: &str, templates: &[OrganismTemplate]) -> std::io::Result<()> {
    let mut buf: Vec<u8> = Vec::with_capacity(4096);
    buf.extend_from_slice(SAVE_MAGIC);
    put_u32(&mut buf, templates.len() as u32);

    for tpl in templates {
        write_organism(&mut buf, tpl);
    }

    let mut f = std::fs::File::create(path)?;
    f.write_all(&buf)?;
    f.sync_all()?;
    Ok(())
}


fn write_organism(buf: &mut Vec<u8>, tpl: &OrganismTemplate) {
    // ── kind / transform ────────────────────────────────────────────
    let kind_byte: u8 = match tpl.metabolism {
        Metabolism::Photoautotroph => 0,
        Metabolism::Heterotroph    => 1,
    };
    put_u8(buf, kind_byte);
    put_vec3(buf, tpl.position);
    put_quat(buf, Quat::IDENTITY);

    // Body / cell counts: for species-loaded templates the OCG can be
    // mixed-type and any length; iterate to count exactly. The
    // cycler-default fallback path (custom_ocg == None) produces
    // homogenous 1 or 2-cell OCGs, so the same counting still
    // yields the correct numbers in that case.
    let ocg_for_counts = tpl.build_ocg();
    let mut photo_cells: i32     = 0;
    let mut non_photo_cells: i32 = 0;
    for (_, _, ct) in &ocg_for_counts {
        match ct {
            CellType::Photo                            => photo_cells     += 1,
            CellType::NonPhoto | CellType::Placeholder => non_photo_cells += 1,
        }
    }
    let cells = photo_cells + non_photo_cells;

    // ── per-organism scalar state ───────────────────────────────────
    let max_energy = cells as f32 * MAX_ENERGY_PER_CELL;
    put_f32(buf, max_energy * 0.5);   // energy: half-tank, same as fresh-spawn
    put_u8 (buf, 0);                  // in_sunlight
    put_u8 (buf, 0);                  // reproduced
    put_u8 (buf, 0);                  // reproductions
    put_f32(buf, 0.0);                // movement_speed
    put_vec3(buf, Vec3::Z);           // movement_direction
    put_vec3(buf, Vec3::ZERO);        // velocity
    put_u8 (buf, 0);                  // is_climbing
    put_f32(buf, 0.0);                // climb_energy_debt
    put_i32(buf, photo_cells);
    put_i32(buf, non_photo_cells);
    put_u8 (buf, match tpl.symmetry {
        Symmetry::NoSymmetry => 0,
        Symmetry::Bilateral  => 1,
    });
    put_u8 (buf, tpl.is_sessile() as u8);
    put_u8 (buf, tpl.has_variable_form() as u8);
    put_u8 (buf, match tpl.intelligence {
        IntelligenceLevel::Level0 => 0,
        IntelligenceLevel::Level1 => 1,
        IntelligenceLevel::Level2 => 2,
        IntelligenceLevel::Level3 => 3,
    });

    // ── body parts ──────────────────────────────────────────────────
    // Single body part holding every cell of the organism.
    put_u32(buf, 1);
    let ocg = tpl.build_ocg();

    // Per-body-part header.
    put_u8 (buf, match BodyPartKind::Body {
        BodyPartKind::Body  => 0,
        BodyPartKind::Limb  => 1,
        BodyPartKind::Organ => 2,
    });
    put_vec3(buf, Vec3::ZERO);   // local_offset
    put_u8 (buf, 0);             // consumed
    put_u8 (buf, 0);             // debug_blue
    put_u8 (buf, 1);             // regrowable

    // No attachment (root body part).
    put_u8 (buf, 0);

    // ── cells ───────────────────────────────────────────────────────
    // Per-cell type written from each OCG entry rather than the
    // organism's metabolism, so species-loaded templates with mixed
    // Photo + NonPhoto cells round-trip through the save format
    // accurately.  `neighbour_count = 0` is a placeholder; the
    // loader calls `physiology::recompute_body_parts` after
    // rehydration, which rebuilds the correct count from cell
    // positions.
    put_u32(buf, ocg.len() as u32);
    for (_, pos, ct) in &ocg {
        put_vec3(buf, *pos);
        put_u8 (buf, match ct {
            CellType::Photo       => 0,
            CellType::NonPhoto    => 1,
            CellType::Placeholder => 2,
        });
        put_f32(buf, 1.0);       // cell_energy — DEFAULT_CELL_ENERGY
        put_u8 (buf, 0);         // neighbour_count — recomputed at load
    }

    // ── ocg ─────────────────────────────────────────────────────────
    put_u32(buf, ocg.len() as u32);
    for (idx, pos, ct) in &ocg {
        put_u32(buf, *idx as u32);
        put_vec3(buf, *pos);
        put_u8 (buf, match ct {
            CellType::Photo       => 0,
            CellType::NonPhoto    => 1,
            CellType::Placeholder => 2,
        });
    }

    // ── v003 brain section ─────────────────────────────────────────
    // No brain payload — the loader will install default / recycled
    // weights for the organism's slot when the brain pool's
    // assign_brains_* system runs the next PreUpdate.
    put_u8(buf, 0);
}


// ── Little-endian writer helpers (mirror the simulation's helpers) ──────────

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
