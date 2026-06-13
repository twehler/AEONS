// Editor → .colony writer.
//
// Re-implements the layout from `colony.rs`'s save header (must stay
// byte-compatible) so the editor stays self-contained — it can't call
// `save_colony_system`, which needs live `Organism` + brain pools. Both
// the sliding-brain and limb-brain blocks are emitted as "absent"; the
// loader's `assign_brains_*` systems install fresh weights on load.

use std::io::Write;

use bevy::prelude::*;

use crate::cell::{BodyPartKind, CellType};
use crate::organism::{IntelligenceLevel, MovementMode, Symmetry};
use crate::colony_editor::template::{Metabolism, OrganismTemplate};
use crate::energy::MAX_ENERGY_PER_CELL;


const SAVE_MAGIC: &[u8; 8] = b"AEONS010";


/// Write every organism in `templates` to `path` in the v010 .colony
/// format. The file is overwritten on each call. Must stay byte-identical
/// to `colony_save_load::save_colony_system`'s header + per-organism layout.
/// `water_level` is the global water surface Y stored in the v009+ header.
pub fn write_colony(
    path: &str,
    templates: &[OrganismTemplate],
    water_level: f32,
) -> std::io::Result<()> {
    let mut buf: Vec<u8> = Vec::with_capacity(4096);
    buf.extend_from_slice(SAVE_MAGIC);
    // Elapsed time (h, m, s) — editor-authored colonies resume at t=0.
    put_u32(&mut buf, 0);
    put_u32(&mut buf, 0);
    put_u32(&mut buf, 0);
    // v009 global water level — after the time block, before the count.
    put_f32(&mut buf, water_level);
    put_u32(&mut buf, templates.len() as u32);

    for tpl in templates {
        write_organism(&mut buf, tpl);
    }

    let mut f = std::fs::File::create(path)?;
    f.write_all(&buf)?;
    f.sync_all()?;
    Ok(())
}


/// One body part as written to disk, without dragging in `Cell`/`BodyPart`
/// allocation paths.
struct WireBodyPart {
    kind:         BodyPartKind,
    /// `None` for root; `Some((parent_idx, origin_local))` for appendages.
    /// Rotation is always serialised as identity.
    attachment:   Option<(u32, Vec3)>,
    /// Cell `local_pos` (rebased to the first-cell pivot for Limb parts).
    cells:        Vec<(Vec3, CellType)>,
    /// OCG list (same rebasing as `cells` for Limb parts).
    ocg:          Vec<(usize, Vec3, CellType)>,
}

fn root_part(ocg: &[(usize, Vec3, CellType)]) -> WireBodyPart {
    WireBodyPart {
        kind:       BodyPartKind::Body,
        attachment: None,
        cells:      ocg.iter().map(|(_, p, ct)| (*p, *ct)).collect(),
        ocg:        ocg.to_vec(),
    }
}

fn appendage_part(ocg: &[(usize, Vec3, CellType)], is_limb: bool, parent_idx: u32) -> WireBodyPart {
    if is_limb {
        // Rebase to first-cell pivot — mirrors `placement::limb_body_part`
        // so saves round-trip pixel-for-pixel.
        let pivot = ocg.first().map(|(_, p, _)| *p).unwrap_or(Vec3::ZERO);
        let shifted: Vec<(usize, Vec3, CellType)> = ocg.iter()
            .map(|(i, p, ct)| (*i, *p - pivot, *ct))
            .collect();
        WireBodyPart {
            kind:       BodyPartKind::Limb,
            attachment: Some((parent_idx, pivot)),
            cells:      shifted.iter().map(|(_, p, ct)| (*p, *ct)).collect(),
            ocg:        shifted,
        }
    } else {
        WireBodyPart {
            kind:       BodyPartKind::Organ,
            attachment: Some((parent_idx, Vec3::ZERO)),
            cells:      ocg.iter().map(|(_, p, ct)| (*p, *ct)).collect(),
            ocg:        ocg.to_vec(),
        }
    }
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

    // Build root + appendages (with bilateral mirroring) and the same
    // editor→runtime parent mapping as `placement::spawn_real_organism`,
    // so saved attachment indices match what spawn builds.
    let root_ocg = tpl.build_ocg();
    let mut parts: Vec<WireBodyPart> = vec![root_part(&root_ocg)];
    match tpl.symmetry {
        Symmetry::NoSymmetry => {
            for (app_raw, is_limb, parent) in &tpl.custom_appendages {
                parts.push(appendage_part(app_raw, *is_limb, *parent as u32));
            }
        }
        Symmetry::Bilateral => {
            let mut right_of: Vec<u32> = vec![0];
            let mut left_of:  Vec<u32> = vec![0];
            for (app_raw, is_limb, parent) in &tpl.custom_appendages {
                let p_right = right_of[*parent];
                let p_left  = left_of[*parent];
                let r_idx = parts.len() as u32;
                parts.push(appendage_part(app_raw, *is_limb, p_right));
                let l_idx = parts.len() as u32;
                let mirrored = crate::body_part::mirror_right_to_left(app_raw);
                parts.push(appendage_part(&mirrored, *is_limb, p_left));
                right_of.push(r_idx);
                left_of.push(l_idx);
            }
        }
    }

    // Cell tallies summed across every part.
    let mut photo_cells: i32     = 0;
    let mut non_photo_cells: i32 = 0;
    for p in &parts {
        for (_, ct) in &p.cells {
            match ct {
                CellType::Photo                            => photo_cells     += 1,
                CellType::NonPhoto | CellType::Placeholder | CellType::SubLimb
                | CellType::YellowCell | CellType::OrangeCell | CellType::BrownCell => non_photo_cells += 1,
            }
        }
    }
    let total_cells = photo_cells + non_photo_cells;

    // ── per-organism scalar state ───────────────────────────────────
    let max_energy = total_cells as f32 * MAX_ENERGY_PER_CELL;
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
    // movement paradigm: v009 4-way tag (0=Sliding, 1=LimbBasedWalking,
    // 2=Swimming, 3=Flying) — written directly from the template's MovementMode.
    put_u8 (buf, match tpl.movement_mode {
        MovementMode::Sliding          => 0,
        MovementMode::LimbBasedWalking => 1,
        MovementMode::Swimming         => 2,
        MovementMode::Flying           => 3,
    });
    // v010: ground- vs water-based (floating phototrophs).
    put_u8 (buf, tpl.ground_based as u8);
    // intelligence level — saved so loaders don't re-roll it.
    put_u8 (buf, match tpl.intelligence {
        IntelligenceLevel::Level0 => 0,
        IntelligenceLevel::Level1 => 1,
        IntelligenceLevel::Level2 => 2,
        IntelligenceLevel::Level3 => 3,
    });

    // ── body parts ──────────────────────────────────────────────────
    put_u32(buf, parts.len() as u32);
    for part in &parts {
        put_u8 (buf, match part.kind {
            BodyPartKind::Body  => 0,
            BodyPartKind::Limb  => 1,
            BodyPartKind::Organ => 2,
        });
        put_vec3(buf, Vec3::ZERO);   // local_offset
        put_u8 (buf, 0);             // consumed
        put_u8 (buf, 0);             // debug_blue
        put_u8 (buf, 1);             // regrowable

        match &part.attachment {
            Some((parent_idx, origin_local)) => {
                put_u8 (buf, 1);
                put_u32(buf, *parent_idx);
                put_vec3(buf, *origin_local);
                put_quat(buf, Quat::IDENTITY);
            }
            None => put_u8(buf, 0),
        }

        // ── cells ───────────────────────────────────────────────────
        // `neighbour_count = 0` placeholder; the loader recomputes it
        // via `physiology::recompute_body_parts` after rehydration.
        put_u32(buf, part.cells.len() as u32);
        for (pos, ct) in &part.cells {
            put_vec3(buf, *pos);
            put_u8 (buf, match ct {
                CellType::Photo       => 0,
                CellType::NonPhoto    => 1,
                CellType::Placeholder => 2,
                CellType::SubLimb     => 3,
                CellType::YellowCell  => 4,
                CellType::OrangeCell  => 5,
                CellType::BrownCell   => 6,
            });
            put_f32(buf, 1.0);       // cell_energy — DEFAULT_CELL_ENERGY
            put_u8 (buf, 0);         // neighbour_count — recomputed at load
        }

        // ── ocg ─────────────────────────────────────────────────────
        put_u32(buf, part.ocg.len() as u32);
        for (idx, pos, ct) in &part.ocg {
            put_u32(buf, *idx as u32);
            put_vec3(buf, *pos);
            put_u8 (buf, match ct {
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

    // sliding-brain section: 0 = no payload (loader installs fresh weights).
    put_u8(buf, 0);

    // limb-brain section: kind tag 0 = none (loader installs fresh weights).
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
