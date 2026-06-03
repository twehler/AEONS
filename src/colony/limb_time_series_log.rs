// Continuous per-organism time-series logger for LIMB-BASED organisms.
//
// The milestone dataset snapshots (`dataset_export.rs`) capture one
// instant per organism, which can't reveal whether a limb organism's
// gait actually OSCILLATES, whether its body TRAVELS, or whether it just
// spins in place. This logger fills that gap: every `LOG_INTERVAL_SECS`
// virtual seconds it appends one CSV row per limb-based organism
// (`sliding_movement == false`) to
// `datasets/limb_time_series_<timestamp>.csv`, capturing the full
// locomotion picture over time so `data-analysis/limb_brains.R` can trace
// it.
//
// Per row we record:
//   * the 6 brain outputs (`limb_target_*`) — now the CPG gait params
//     (per pair: amplitude, posture offset, steering bias);
//   * `motor_target_norm` — max |LastAppliedTorque| across the organism's
//     body parts. `drive_limb_motors` stores the commanded hinge
//     angle (radians) there, so this is the "is the CPG driving the
//     motors, and is the setpoint non-zero / changing?" signal;
//   * the BASE body part's world position + planar speed + angular-speed
//     magnitude (from Avian) — "is the body travelling, or spinning in
//     place?";
//   * `target_distance` (XZ distance to the nearest photo) — "is it
//     closing on prey?";
//   * energy + predations — outcome.
//
// Sampling at 1 s (limb organisms are few) aliases the ~`GAIT_FREQUENCY_HZ`
// gait, so across many samples `motor_target_norm` and the base velocity
// SCATTER if the gait is live and stay flat if it is not — enough to
// distinguish "CPG oscillating" from "CPG dead" without a high-rate trace.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use bevy::prelude::*;
use avian3d::prelude::{LinearVelocity, AngularVelocity};

use crate::colony::{Organism, OrganismRoot};
use crate::cell::BodyPartIndex;
use crate::avian_setup::LastAppliedTorque;


/// Cadence for limb time-series rows (virtual seconds). 1 s gives a dense
/// trajectory; with only a handful of limb organisms the file stays tiny.
const LOG_INTERVAL_SECS: f32 = 1.0;


#[derive(Resource, Default)]
pub struct LimbTimeSeriesLogger {
    writer:         Option<BufWriter<File>>,
    last_log_secs:  f32,
    init_attempted: bool,
}


/// Per-organism Avian state for the base body part + max commanded torque.
#[derive(Default, Clone, Copy)]
struct BaseState {
    pos:             Vec3,
    lin_vel:         Vec3,
    ang_vel:         Vec3,
    motor_target:    f32, // max |LastAppliedTorque| across this organism's parts
    base_contact:    bool, // is the base body part touching anything?
    contact_count:   u32,  // total contacts across all of this organism's parts
    limbs_total:     u32,  // number of LIMB parts (idx > 0) on this organism
    limbs_planted:   u32,  // number of those limb parts currently touching the ground
    seen:            bool,
}


pub fn tick_limb_time_series_logger(
    mut logger:   ResMut<LimbTimeSeriesLogger>,
    sim_running:  Res<crate::simulation_settings::SimulationRunning>,
    virtual_time: Res<Time<Virtual>>,
    diagnostics:  Res<bevy::diagnostic::DiagnosticsStore>,
    heightmap:    Option<Res<crate::world_geometry::HeightmapSampler>>,
    photos:       Query<&GlobalTransform, (With<crate::colony::Photoautotroph>, With<OrganismRoot>)>,
    org_q:        Query<(Entity, &Organism), With<OrganismRoot>>,
    bp_q:         Query<(
        &bevy::prelude::ChildOf,
        &BodyPartIndex,
        &GlobalTransform,
        &LinearVelocity,
        &AngularVelocity,
        &LastAppliedTorque,
        &crate::avian_setup::LimbContact,
    )>,
) {
    if !sim_running.0 { return; }

    let now = virtual_time.elapsed_secs();
    if now - logger.last_log_secs < LOG_INTERVAL_SECS { return; }
    logger.last_log_secs = now;

    // Lazy-init the writer on the first log tick.
    if !logger.init_attempted {
        logger.init_attempted = true;
        let path = PathBuf::from(format!(
            "datasets/limb_time_series_{}.csv",
            chrono::Local::now().format("%d-%m-%Y-%H-%M-%S"),
        ));
        if let Some(parent) = path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                error!("limb-time-series: failed to create dir {}: {}", parent.display(), e);
                return;
            }
        }
        match File::create(&path) {
            Ok(f) => {
                let mut w = BufWriter::new(f);
                if let Err(e) = w.write_all(
                    b"virtual_time_secs;entity_id;intelligence;sliding;\
                      limb_target_0;limb_target_1;limb_target_2;limb_target_3;\
                      limb_target_4;limb_target_5;limb_target_6;limb_target_7;\
                      motor_target_norm;base_x;base_y;base_z;\
                      base_speed_xz;base_ang_vel_mag;\
                      base_contact;contact_count;nearest_org_dist;max_part_dist;min_clearance;\
                      min_limb_clearance;base_clearance;limb_planted_frac;approach_align;\
                      target_distance;energy;predations;fps\n",
                ) {
                    error!("limb-time-series: header write failed: {e}");
                    return;
                }
                info!("limb time-series logging to {}", path.display());
                logger.writer = Some(w);
            }
            Err(e) => {
                error!("limb-time-series: failed to create {}: {}", path.display(), e);
            }
        }
    }

    // Global render/step rate — low FPS at high TimeSpeed means large
    // effective physics dt, which itself can destabilise the solver, so we
    // log it alongside the per-organism state.
    let fps = diagnostics
        .get(&bevy::diagnostic::FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|d| d.smoothed())
        .unwrap_or(0.0) as f32;

    // Photo positions, for the approach-alignment metric (is the body's
    // velocity actually pointing at the nearest prey? — the clean
    // directedness signal, unconfounded by predation).
    let photo_pos: Vec<Vec3> = photos.iter().map(|gt| gt.translation()).collect();

    let Some(writer) = logger.writer.as_mut() else { return; };

    // Build per-organism base-body state from the Avian body-part query.
    let mut base: std::collections::HashMap<Entity, BaseState> =
        std::collections::HashMap::new();
    // All body-part world positions per organism, to measure how far limbs /
    // sub-limbs have SEPARATED from the base (joints should hold them within
    // a few units; large values = the joint failed and the part flew off).
    let mut all_parts: std::collections::HashMap<Entity, Vec<Vec3>> =
        std::collections::HashMap::new();
    // LIMB-only part positions (idx > 0), to measure the FEET / sub-limbs'
    // ground clearance separately from the belly — that's the part the user
    // saw sinking, and what the non-penetration floor must keep on the surface.
    let mut limb_parts: std::collections::HashMap<Entity, Vec<Vec3>> =
        std::collections::HashMap::new();
    for (child_of, idx, gt, lin, ang, torque, contact) in &bp_q {
        let root = child_of.parent();
        all_parts.entry(root).or_default().push(gt.translation());
        if idx.0 != 0 { limb_parts.entry(root).or_default().push(gt.translation()); }
        let entry = base.entry(root).or_default();
        let t_norm = torque.0.length();
        if t_norm > entry.motor_target { entry.motor_target = t_norm; }
        entry.contact_count += contact.count;
        if idx.0 == 0 {
            entry.pos          = gt.translation();
            entry.lin_vel      = lin.0;
            entry.ang_vel      = ang.0;
            entry.base_contact = contact.in_contact;
            entry.seen         = true;
        } else {
            // Limb part: count it, and whether it's planted on the ground.
            // A natural posture has the feet PLANTED (high planted fraction)
            // bearing the body, not dangling in the air.
            entry.limbs_total += 1;
            if contact.in_contact { entry.limbs_planted += 1; }
        }
    }

    // Nearest OTHER limb-organism distance (XZ), to test whether explosions
    // coincide with inter-organism proximity/collision. O(n²) over the few
    // dozen limb organisms, once per log tick — cheap.
    let positions: Vec<(Entity, Vec3)> =
        base.iter().filter(|(_, b)| b.seen).map(|(&e, b)| (e, b.pos)).collect();
    let nearest_dist = |e: Entity, p: Vec3| -> f32 {
        let mut best = f32::MAX;
        for &(oe, op) in &positions {
            if oe == e { continue; }
            let dx = op.x - p.x; let dz = op.z - p.z;
            let d = (dx * dx + dz * dz).sqrt();
            if d < best { best = d; }
        }
        if best == f32::MAX { 999.0 } else { best }
    };

    let mut row = String::with_capacity(256);
    for (entity, org) in org_q.iter() {
        // Limb-based organisms only (the sliding pools have their own
        // time-series logger).
        if org.sliding_movement { continue; }

        let b = base.get(&entity).copied().unwrap_or_default();
        let speed_xz = (b.lin_vel.x * b.lin_vel.x + b.lin_vel.z * b.lin_vel.z).sqrt();
        let ang_mag  = b.ang_vel.length();
        let il = match org.intelligence_level {
            crate::organism::IntelligenceLevel::Level0 => 0,
            crate::organism::IntelligenceLevel::Level1 => 1,
            crate::organism::IntelligenceLevel::Level2 => 2,
            crate::organism::IntelligenceLevel::Level3 => 3,
        };

        let near = if b.seen { nearest_dist(entity, b.pos) } else { 999.0 };
        // Max distance of any body part from the base — the limb/sub-limb
        // SEPARATION metric. Should stay small (creature's own size); large
        // = a joint failed and the part separated.
        let max_part_dist = all_parts.get(&entity).map_or(0.0, |ps| {
            ps.iter().map(|p| (*p - b.pos).length()).fold(0.0_f32, f32::max)
        });
        // Penetration metric: the SMALLEST clearance of any body part above
        // the terrain (part world-Y minus heightmap height at that XZ). A
        // negative value means a part's origin is BELOW the ground surface —
        // i.e. it has sunk into the "concrete" world mesh. 0 ≈ resting on the
        // surface; positive = held above it. This is the signal for the
        // "nothing penetrates the world mesh" goal.
        let min_clearance = match (&heightmap, all_parts.get(&entity)) {
            (Some(hm), Some(ps)) => ps.iter()
                .map(|p| p.y - hm.height_at(p.x, p.z))
                .fold(f32::MAX, f32::min),
            _ => 999.0,
        };
        // Posture metrics for the "natural posture" goal:
        //  * base_clearance — how high the BASE body sits above the terrain.
        //    A natural standing posture holds the body UP off the ground
        //    (positive), borne by the legs; ~0 = belly dragging on the floor.
        //  * limb_planted_frac — fraction of the limb parts (feet) currently
        //    touching the ground. ~1 = all feet planted (natural, weight-
        //    bearing); low = legs dangling in the air "defying gravity".
        let base_clearance = match &heightmap {
            Some(hm) if b.seen => b.pos.y - hm.height_at(b.pos.x, b.pos.z),
            _ => 999.0,
        };
        let limb_planted_frac = if b.limbs_total > 0 {
            b.limbs_planted as f32 / b.limbs_total as f32
        } else { 0.0 };
        // FEET/sub-limb clearance: the minimum clearance over the LIMB parts
        // only (excludes the belly). This is the direct measure of the user's
        // complaint — "the sub-limbs are always sunken". Should stay >= ~0
        // (the non-penetration floor keeps the feet on the surface), even when
        // the belly is allowed to graze/slide on the ground.
        let min_limb_clearance = match (&heightmap, limb_parts.get(&entity)) {
            (Some(hm), Some(ps)) if !ps.is_empty() => ps.iter()
                .map(|p| p.y - hm.height_at(p.x, p.z))
                .fold(f32::MAX, f32::min),
            _ => 999.0,
        };
        // Approach alignment: dot(base velocity dir, dir to nearest photo),
        // in XZ. +1 = moving straight at nearest prey, −1 = straight away,
        // 0 = sideways/still. The clean directed-pursuit metric.
        let approach_align = {
            let v = Vec2::new(b.lin_vel.x, b.lin_vel.z);
            let mut best = f32::MAX; let mut pdir = Vec2::ZERO;
            for pp in &photo_pos {
                let d = Vec2::new(pp.x - b.pos.x, pp.z - b.pos.z);
                let dist = d.length();
                if dist > 0.01 && dist < best { best = dist; pdir = d / dist; }
            }
            if v.length() > 0.05 && best < f32::MAX { v.normalize().dot(pdir) } else { 0.0 }
        };

        row.clear();
        use std::fmt::Write as _;
        let _ = write!(
            row,
            "{:.3};{};{};{};\
             {:.4};{:.4};{:.4};{:.4};{:.4};{:.4};{:.4};{:.4};\
             {:.4};{:.3};{:.3};{:.3};\
             {:.4};{:.4};\
             {};{};{:.2};{:.2};{:.3};{:.3};{:.3};{:.3};{:.3};\
             {:.3};{:.3};{};{:.1}\n",
            now, entity.index(), il, bool01(org.sliding_movement),
            org.limb_targets[0], org.limb_targets[1], org.limb_targets[2], org.limb_targets[3],
            org.limb_targets[4], org.limb_targets[5], org.limb_targets[6], org.limb_targets[7],
            b.motor_target, b.pos.x, b.pos.y, b.pos.z,
            speed_xz, ang_mag,
            bool01(b.base_contact), b.contact_count, near, max_part_dist, min_clearance,
            min_limb_clearance, base_clearance, limb_planted_frac, approach_align,
            org.target_distance, org.energy, org.predations, fps,
        );
        let _ = writer.write_all(row.as_bytes());
    }
    let _ = writer.flush();
}

#[inline]
fn bool01(b: bool) -> u8 { if b { 1 } else { 0 } }
