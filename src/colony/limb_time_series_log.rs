// Continuous per-organism time-series logger for LIMB-BASED organisms.
//
// Milestone snapshots (`dataset_export.rs`) capture one instant and can't show
// whether a gait oscillates, the body travels, or it just spins in place.
// Every `LOG_INTERVAL_SECS` virtual seconds this appends one CSV row per
// limb-based organism (`!movement_mode.is_sliding()`) to
// `datasets/limb_time_series_<timestamp>.csv` for `data-analysis/limb_brains.R`.
//
// Per row: the brain outputs (`limb_target_*`); `motor_target_norm` (max
// |LastAppliedTorque|, the commanded hinge angle in radians); the BASE body's
// world pos + planar/angular speed; `target_distance` to nearest photo;
// energy + predations.
//
// 1 s sampling aliases the ~`GAIT_FREQUENCY_HZ` gait, so across samples the
// torque/velocity SCATTER if the gait is live and stay flat if dead.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use bevy::prelude::*;
use bevy_rapier3d::prelude::Velocity;

use crate::colony::{Organism, OrganismRoot};
use crate::cell::BodyPartIndex;
use crate::rapier_setup::LastAppliedTorque;


/// Cadence for limb time-series rows (virtual seconds).
const LOG_INTERVAL_SECS: f32 = 1.0;


#[derive(Resource, Default)]
pub struct LimbTimeSeriesLogger {
    writer:         Option<BufWriter<File>>,
    last_log_secs:  f32,
    init_attempted: bool,
    /// Running MAX joint separation per organism SINCE the last emitted row. The
    /// logger runs every frame but only writes every `LOG_INTERVAL_SECS`; without
    /// this, a sub-second separation transient (the kind seen on screen between
    /// 1 Hz samples) is computed and discarded. Folded each frame, emitted + cleared
    /// each row, so `joint_sep_max` is a TRUE interval max, not an instantaneous sample.
    acc_joint_sep:  std::collections::HashMap<Entity, f32>,
    /// Virtual time each organism was first seen — used to skip the first ~1 s, where
    /// the freshly-spawned child `GlobalTransform`s aren't yet propagated and the
    /// joint-separation calc reads a spurious ~5-8 (the artifact that masqueraded as
    /// real separation and triggered a wrong multibody detour).
    first_seen:     std::collections::HashMap<Entity, f32>,
    /// Base (idx 0) world position last frame — a large jump means the fall-reset
    /// teleported the organism, whose 1-frame anchor mismatch is also not real
    /// separation; that frame is skipped from the accumulator.
    last_base_pos:  std::collections::HashMap<Entity, Vec3>,
}


/// Per-organism Avian state for the base body part + max commanded torque.
#[derive(Default, Clone, Copy)]
struct BaseState {
    pos:             Vec3,
    up_y:            f32,  // world-Y component of the base body's local +Y (uprightness)
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
        &Velocity,
        &LastAppliedTorque,
        &crate::rapier_setup::LimbContact,
    )>,
    joint_q:      Query<&crate::rapier_setup::LimbJointDrive>,
    gt_q:         Query<&GlobalTransform>,
) {
    if !sim_running.0 { return; }
    let now = virtual_time.elapsed_secs();

    // Per-organism MAX joint separation THIS FRAME: world distance between each
    // limb's rotation point (its anchor) and the parent attachment point.
    // Guiderail: must stay ~0 (a rigid joint never separates). >0 ⇒ the joint is
    // being pulled apart (what the user observed as "limbs separating").
    let mut joint_sep_frame: std::collections::HashMap<Entity, f32> = std::collections::HashMap::new();
    for drive in &joint_q {
        if let (Ok(limb_gt), Ok(par_gt)) = (gt_q.get(drive.limb_entity), gt_q.get(drive.parent)) {
            let limb_pt = limb_gt.transform_point(drive.anchor2);
            let par_pt  = par_gt.transform_point(drive.anchor1);
            let sep = (limb_pt - par_pt).length();
            let m = joint_sep_frame.entry(drive.organism).or_insert(0.0);
            if sep > *m { *m = sep; }
        }
    }

    // Base (idx 0) world position THIS frame, for teleport detection.
    let mut base_now: std::collections::HashMap<Entity, Vec3> = std::collections::HashMap::new();
    for (child_of, idx, gt, _, _, _) in &bp_q {
        if idx.0 == 0 { base_now.insert(child_of.parent(), gt.translation()); }
    }

    // Fold this frame's separation into the running interval-max, skipping
    // GLITCH frames so the metric is honest AND catches sub-second transients:
    //   * first SETTLE seconds of an organism's life → spawn-propagation artifact;
    //   * a base jump > TELEPORT_JUMP → the fall-reset teleported it this frame.
    const SETTLE_SECS: f32 = 1.0;
    const TELEPORT_JUMP: f32 = 1.0;
    for (&org, &sep) in &joint_sep_frame {
        let first = *logger.first_seen.entry(org).or_insert(now);
        let cur = base_now.get(&org).copied();
        let teleported = matches!((logger.last_base_pos.get(&org).copied(), cur),
            (Some(prev), Some(c)) if (c - prev).length() > TELEPORT_JUMP);
        if let Some(c) = cur { logger.last_base_pos.insert(org, c); }
        if now - first < SETTLE_SECS || teleported { continue; }
        let m = logger.acc_joint_sep.entry(org).or_insert(0.0);
        if sep > *m { *m = sep; }
    }

    if now - logger.last_log_secs < LOG_INTERVAL_SECS { return; }
    logger.last_log_secs = now;
    // Emit the interval-max separation and reset the accumulator for the next window.
    let joint_sep = std::mem::take(&mut logger.acc_joint_sep);

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
                      target_distance;energy;predations;fps;twist_cmd_norm;joint_sep_max;\
                      base_upright;base_tilt_deg;standing;stand_score;\
                      is_swimming;base_speed3d\n",
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

    // Render/step rate: low FPS at high TimeSpeed → large physics dt that can
    // destabilise the solver, so log it alongside per-organism state.
    let fps = diagnostics
        .get(&bevy::diagnostic::FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|d| d.smoothed())
        .unwrap_or(0.0) as f32;

    // Photo positions for the approach-alignment metric.
    let photo_pos: Vec<Vec3> = photos.iter().map(|gt| gt.translation()).collect();

    let Some(writer) = logger.writer.as_mut() else { return; };

    // Per-organism base-body state from the Avian body-part query.
    let mut base: std::collections::HashMap<Entity, BaseState> =
        std::collections::HashMap::new();
    // All body-part world positions per organism, for the limb-separation
    // metric (large values = a joint failed and the part flew off).
    let mut all_parts: std::collections::HashMap<Entity, Vec<Vec3>> =
        std::collections::HashMap::new();
    // LIMB-only positions (idx > 0), for feet/sub-limb ground clearance
    // measured separately from the belly.
    let mut limb_parts: std::collections::HashMap<Entity, Vec<Vec3>> =
        std::collections::HashMap::new();
    for (child_of, idx, gt, vel, torque, contact) in &bp_q {
        let root = child_of.parent();
        all_parts.entry(root).or_default().push(gt.translation());
        if idx.0 != 0 { limb_parts.entry(root).or_default().push(gt.translation()); }
        let entry = base.entry(root).or_default();
        let t_norm = torque.0.length();
        if t_norm > entry.motor_target { entry.motor_target = t_norm; }
        entry.contact_count += contact.count;
        if idx.0 == 0 {
            entry.pos          = gt.translation();
            entry.up_y         = (gt.rotation() * Vec3::Y).y;
            entry.lin_vel      = vel.linear;
            entry.ang_vel      = vel.angular;
            entry.base_contact = contact.in_contact;
            entry.seen         = true;
        } else {
            // Limb part: count it and whether it's planted (natural posture
            // bears the body on planted feet, not dangling).
            entry.limbs_total += 1;
            if contact.in_contact { entry.limbs_planted += 1; }
        }
    }

    // Nearest OTHER limb-organism distance (XZ): does instability coincide
    // with inter-organism proximity? O(n²) over a few dozen organisms.
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
        // Limb-based organisms only (sliding pools log elsewhere).
        if org.movement_mode.is_sliding() { continue; }

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
        // Limb-separation metric: max part distance from the base (large =
        // a joint failed).
        let max_part_dist = all_parts.get(&entity).map_or(0.0, |ps| {
            ps.iter().map(|p| (*p - b.pos).length()).fold(0.0_f32, f32::max)
        });
        // Penetration metric: smallest clearance of any part above terrain
        // (part Y − heightmap height). Negative = sunk into the world mesh.
        let min_clearance = match (&heightmap, all_parts.get(&entity)) {
            (Some(hm), Some(ps)) => ps.iter()
                .map(|p| p.y - hm.height_at(p.x, p.z))
                .fold(f32::MAX, f32::min),
            _ => 999.0,
        };
        // Posture metrics: base_clearance = how high the base sits above
        // terrain (~0 = belly dragging); limb_planted_frac = fraction of feet
        // touching the ground (~1 = natural weight-bearing).
        let base_clearance = match &heightmap {
            Some(hm) if b.seen => b.pos.y - hm.height_at(b.pos.x, b.pos.z),
            _ => 999.0,
        };
        let limb_planted_frac = if b.limbs_total > 0 {
            b.limbs_planted as f32 / b.limbs_total as f32
        } else { 0.0 };
        // Feet/sub-limb clearance: min clearance over LIMB parts only
        // (excludes the belly). Should stay >= ~0 via the non-penetration floor.
        let min_limb_clearance = match (&heightmap, limb_parts.get(&entity)) {
            (Some(hm), Some(ps)) if !ps.is_empty() => ps.iter()
                .map(|p| p.y - hm.height_at(p.x, p.z))
                .fold(f32::MAX, f32::min),
            _ => 999.0,
        };
        // Approach alignment (XZ): dot(velocity dir, dir to nearest photo).
        // +1 = straight at prey, −1 = away, 0 = sideways/still.
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

        // ── STANDING diagnostics (the goal's success signals) ──
        // base_upright: world-Y of the base's local +Y (1=upright, 0=on side,
        // −1=inverted). base_tilt_deg: angle off vertical. standing: the success
        // flag (upright AND belly off the floor AND enough planted legs incl.
        // sub-limbs). stand_score: the reward's primary tall×level×support term.
        let up_y       = b.up_y;
        let upright01  = (0.5 * (1.0 + up_y)).clamp(0.0, 1.0);
        let tilt_deg   = up_y.clamp(-1.0, 1.0).acos().to_degrees();
        let tall = if base_clearance < 900.0 {
            (base_clearance / crate::simulation_settings::STAND_HEIGHT_TARGET).clamp(0.0, 1.0)
        } else { 0.0 };
        let foot_support = (b.limbs_planted as f32
            / crate::simulation_settings::STAND_MIN_FEET).clamp(0.0, 1.0);
        let stand_score = tall * upright01 * foot_support;
        let standing = (upright01 > crate::simulation_settings::STAND_UPRIGHT_MIN
            && !b.base_contact
            && (b.limbs_planted as f32) >= crate::simulation_settings::STAND_MIN_FEET) as u8;

        row.clear();
        use std::fmt::Write as _;
        let _ = write!(
            row,
            "{:.3};{};{};{};\
             {:.4};{:.4};{:.4};{:.4};{:.4};{:.4};{:.4};{:.4};\
             {:.4};{:.3};{:.3};{:.3};\
             {:.4};{:.4};\
             {};{};{:.2};{:.2};{:.3};{:.3};{:.3};{:.3};{:.3};\
             {:.3};{:.3};{};{:.1};{:.4};{:.4};\
             {:.4};{:.2};{};{:.4};\
             {};{:.4}\n",
            now, entity.index(), il, bool01(org.movement_mode.is_sliding()),
            org.limb_targets[0], org.limb_targets[1], org.limb_targets[2], org.limb_targets[3],
            org.limb_targets[4], org.limb_targets[5], org.limb_targets[6], org.limb_targets[7],
            b.motor_target, b.pos.x, b.pos.y, b.pos.z,
            speed_xz, ang_mag,
            bool01(b.base_contact), b.contact_count, near, max_part_dist, min_clearance,
            min_limb_clearance, base_clearance, limb_planted_frac, approach_align,
            org.target_distance, org.energy, org.predations, fps,
            // TWIST usage: mean |twist command| over the grouped twist outputs
            // (limb_targets[MAX_LIMB_JOINTS..]). >0 ⇒ the brain is actively twisting.
            {
                let tw = &org.limb_targets[crate::simulation_settings::MAX_LIMB_JOINTS..];
                tw.iter().map(|t| t.abs()).sum::<f32>() / tw.len().max(1) as f32
            },
            // JOINT SEPARATION: max world distance over this organism's limb joints
            // between the limb's rotation point and its parent attachment point.
            joint_sep.get(&entity).copied().unwrap_or(0.0),
            // STANDING success signals.
            up_y, tilt_deg, standing, stand_score,
            // SWIMMING: trophic-mode flag + the base body's full 3D linear
            // speed (vs. base_speed_xz which drops the vertical component) so
            // the R suite can correlate stroke outputs with 3D propulsion.
            bool01(org.movement_mode.is_swimming()),
            (b.lin_vel.x * b.lin_vel.x + b.lin_vel.y * b.lin_vel.y + b.lin_vel.z * b.lin_vel.z).sqrt(),
        );
        let _ = writer.write_all(row.as_bytes());
    }
    let _ = writer.flush();
}

#[inline]
fn bool01(b: bool) -> u8 { if b { 1 } else { 0 } }
