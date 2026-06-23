// HIGH-FREQUENCY spawn-force probe for LIMB-BASED organisms.
//
// The 1 Hz `limb_time_series_log` cannot see an IMMEDIATE spawn instability
// (a body flung within the first few frames of life, before the first 1 s
// sample, and inside that logger's 1 s settle-skip). This probe logs EVERY
// frame for the first `PROBE_WINDOW_SECS` of each limb organism's life, then
// throttles, capturing the physical forces themselves so the cause of a fling
// is visible in the data rather than inferred:
//
//   * `max_force_mag` / `max_force_y` — largest `ExternalForce.force` over the
//     organism's parts (the accumulated per-step drag + twist + confinement).
//   * `max_torque_mag` — largest `ExternalForce.torque`.
//   * `max_linvel` / `max_linvel_y` / `max_angvel` — largest Rapier velocities
//     (the symptom: a fling shows here first).
//   * `worst_part_idx` / `worst_part_kind` — which body part has that max linear
//     velocity (0 = base/Body, else the limb/appendage index; kind 0=Body,
//     1=Limb, 2=Organ).
//   * `max_lever` — largest distance from a part's body ORIGIN to its cell
//     centroid (≈ COM): a big lever arm means a small joint/drag force makes a
//     large torque. The appendage-not-rebased bug shows as a large lever here.
//   * `max_anchor_sep` — largest joint anchor separation (parent anchor1 vs
//     child anchor2 in world space): >0 ⇒ a joint constraint is being violated.
//   * `base_grav_scale` — the water-gated gravity scale on the base (1 above
//     water, 0 submerged).
//   * `base_y` / `min_part_y` / `max_part_y` — vertical spread (a fling spikes
//     `max_part_y`; a sink drops `min_part_y`).
//
// Output: `datasets/limb_force_probe_<timestamp>.csv`, analysed by
// `data-analysis/limb_force_probe.R`.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use bevy::prelude::*;
use bevy_rapier3d::prelude::{ExternalForce, GravityScale, Velocity, RapierRigidBodyHandle, RapierRigidBodySet};
use bevy_rapier3d::utils::iso_to_transform;

use crate::colony::{Organism, OrganismRoot};
use crate::cell::BodyPartIndex;
use crate::rapier_setup::LimbJointDrive;


/// Log every frame for this many virtual seconds after an organism is first
/// seen (the window where an immediate spawn fling happens), then throttle.
const PROBE_WINDOW_SECS: f32 = 6.0;
/// Post-window sampling cadence (virtual seconds) — keeps the file bounded
/// while still tracking longer-term drift.
const PROBE_THROTTLE_SECS: f32 = 1.0;


#[derive(Resource, Default)]
pub struct LimbForceProbe {
    writer:         Option<BufWriter<File>>,
    init_attempted: bool,
    /// Virtual time each organism was first seen (start of its per-frame window).
    first_seen:     std::collections::HashMap<Entity, f32>,
    /// Last virtual time a row was emitted per organism (for post-window throttle).
    last_log:       std::collections::HashMap<Entity, f32>,
}


/// Per-organism accumulator over its parts this frame.
#[derive(Default, Clone, Copy)]
struct OrgForce {
    seen:           bool,
    base_y:         f32,
    base_grav:      f32,
    min_part_y:     f32,
    max_part_y:     f32,
    max_force_mag:  f32,
    max_force_y:    f32,
    max_torque_mag: f32,
    max_linvel:     f32,
    max_linvel_y:   f32,
    max_angvel:     f32,
    worst_part_idx: i32,
    worst_part_kind: u8,
    max_lever:      f32,
}


pub fn tick_limb_force_probe(
    mut probe:    ResMut<LimbForceProbe>,
    sim_running:  Res<crate::simulation_settings::SimulationRunning>,
    virtual_time: Res<Time<Virtual>>,
    org_q:        Query<(Entity, &Organism), With<OrganismRoot>>,
    bp_q:         Query<(
        &bevy::prelude::ChildOf,
        &BodyPartIndex,
        &GlobalTransform,
        &Velocity,
        Option<&ExternalForce>,
        Option<&GravityScale>,
    )>,
    joint_q:      Query<&LimbJointDrive>,
    gt_q:         Query<&GlobalTransform>,
    // PHYSICS-TRUTH separation: read each body's actual Rapier isometry, immune
    // to any Bevy GlobalTransform writeback lag (the Transform-diff `max_anchor_sep`
    // can read high for multibody links even when the joint is rigidly coupled).
    rb_set_q:     Query<&RapierRigidBodySet>,
    handle_q:     Query<&RapierRigidBodyHandle>,
) {
    if !sim_running.0 { return; }
    let now = virtual_time.elapsed_secs();
    let rb_set = rb_set_q.single().ok();

    // ── Joint anchor separation per organism (constraint-violation signal). ──
    // Two measures: `anchor_sep` from Bevy GlobalTransforms (legacy, writeback-lag
    // prone) and `rapier_sep` from the live Rapier body isometries (the truth).
    let mut anchor_sep: std::collections::HashMap<Entity, f32> = std::collections::HashMap::new();
    let mut rapier_sep: std::collections::HashMap<Entity, f32> = std::collections::HashMap::new();
    for drive in &joint_q {
        if let (Ok(limb_gt), Ok(par_gt)) = (gt_q.get(drive.limb_entity), gt_q.get(drive.parent)) {
            let sep = (limb_gt.transform_point(drive.anchor2)
                     - par_gt.transform_point(drive.anchor1)).length();
            let m = anchor_sep.entry(drive.organism).or_insert(0.0);
            if sep > *m { *m = sep; }
        }
        // Rapier-truth: world anchor points from the actual body poses.
        if let Some(set) = rb_set {
            if let (Ok(lh), Ok(ph)) = (handle_q.get(drive.limb_entity), handle_q.get(drive.parent)) {
                if let (Some(lb), Some(pb)) = (set.bodies.get(lh.0), set.bodies.get(ph.0)) {
                    let lt = iso_to_transform(lb.position());
                    let pt = iso_to_transform(pb.position());
                    let sep = (lt.transform_point(drive.anchor2)
                             - pt.transform_point(drive.anchor1)).length();
                    let m = rapier_sep.entry(drive.organism).or_insert(0.0);
                    if sep > *m { *m = sep; }
                }
            }
        }
    }

    // ── Per-organism force/velocity accumulation over its parts. ──
    // `kind_of` is read from the Organism's body_parts (the entity carries only
    // BodyPartIndex), so the worst part's kind (Body/Limb/Organ) is known.
    let mut acc: std::collections::HashMap<Entity, OrgForce> = std::collections::HashMap::new();
    for (child_of, idx, gt, vel, ext, grav) in &bp_q {
        let root = child_of.parent();
        let Ok((_, org)) = org_q.get(root) else { continue };
        if org.movement_mode.is_sliding() { continue; }

        let e = acc.entry(root).or_insert_with(|| OrgForce {
            min_part_y: f32::MAX, max_part_y: f32::MIN, worst_part_idx: -1, ..Default::default()
        });
        e.seen = true;
        let py = gt.translation().y;
        if py < e.min_part_y { e.min_part_y = py; }
        if py > e.max_part_y { e.max_part_y = py; }

        if let Some(f) = ext {
            let fl = f.force.length();
            if fl > e.max_force_mag { e.max_force_mag = fl; }
            if f.force.y.abs() > e.max_force_y.abs() { e.max_force_y = f.force.y; }
            let tl = f.torque.length();
            if tl > e.max_torque_mag { e.max_torque_mag = tl; }
        }
        let lv = vel.linear.length();
        if lv > e.max_linvel {
            e.max_linvel = lv;
            e.worst_part_idx = idx.0 as i32;
            e.worst_part_kind = org.body_parts.get(idx.0)
                .map(|bp| match bp.kind {
                    crate::cell::BodyPartKind::Body    => 0u8,
                    crate::cell::BodyPartKind::Limb    => 1,
                    crate::cell::BodyPartKind::Organ   => 2,
                    crate::cell::BodyPartKind::Segment => 3,
                    crate::cell::BodyPartKind::Static  => 4,
                }).unwrap_or(255);
        }
        if vel.linear.y.abs() > e.max_linvel_y.abs() { e.max_linvel_y = vel.linear.y; }
        let av = vel.angular.length();
        if av > e.max_angvel { e.max_angvel = av; }

        // Lever arm: distance from this part's body origin to its cell centroid
        // (≈ COM). A large value means joint/drag forces produce a large torque
        // — the not-rebased-appendage signature (origin at root, COM far away).
        if let Some(bp) = org.body_parts.get(idx.0) {
            if !bp.cells.is_empty() {
                let c = bp.cells.iter().map(|c| c.local_pos).sum::<Vec3>() / bp.cells.len() as f32;
                let lever = c.length();
                if lever > e.max_lever { e.max_lever = lever; }
            }
        }

        if idx.0 == 0 {
            e.base_y = py;
            e.base_grav = grav.map(|g| g.0).unwrap_or(1.0);
        }
    }

    // ── Lazy-init writer. ──
    if !probe.init_attempted {
        probe.init_attempted = true;
        let path = PathBuf::from(format!(
            "datasets/limb_force_probe_{}.csv",
            chrono::Local::now().format("%d-%m-%Y-%H-%M-%S"),
        ));
        if let Some(parent) = path.parent() { let _ = std::fs::create_dir_all(parent); }
        match File::create(&path) {
            Ok(f) => {
                let mut w = BufWriter::new(f);
                if w.write_all(
                    b"virtual_time_secs;entity_id;age_secs;is_swimming;\
                      base_y;min_part_y;max_part_y;base_grav_scale;\
                      max_force_mag;max_force_y;max_torque_mag;\
                      max_linvel;max_linvel_y;max_angvel;\
                      worst_part_idx;worst_part_kind;max_lever;max_anchor_sep;max_rapier_sep\n",
                ).is_ok() {
                    info!("limb force probe logging to {}", path.display());
                    probe.writer = Some(w);
                }
            }
            Err(e) => error!("limb-force-probe: create failed: {e}"),
        }
    }
    // ── Phase 1: decide which organisms emit this frame (mutates the
    // first_seen / last_log maps), collecting the formatted rows. Done BEFORE
    // borrowing `probe.writer` so the map mutations and the writer borrow
    // don't alias `*probe`. ──
    use std::fmt::Write as _;
    let mut rows: Vec<String> = Vec::new();
    for (entity, org) in org_q.iter() {
        if org.movement_mode.is_sliding() { continue; }
        let Some(a) = acc.get(&entity).copied() else { continue };
        if !a.seen { continue; }

        let first = *probe.first_seen.entry(entity).or_insert(now);
        let age = now - first;
        // Per-frame inside the window; throttled afterwards.
        if age > PROBE_WINDOW_SECS {
            let last = probe.last_log.get(&entity).copied().unwrap_or(f32::MIN);
            if now - last < PROBE_THROTTLE_SECS { continue; }
        }
        probe.last_log.insert(entity, now);

        let mut row = String::with_capacity(256);
        let _ = write!(
            row,
            "{:.3};{};{:.3};{};\
             {:.3};{:.3};{:.3};{:.1};\
             {:.4};{:.4};{:.4};\
             {:.4};{:.4};{:.4};\
             {};{};{:.4};{:.4};{:.4}\n",
            now, entity.index(), age, bool01(org.movement_mode.is_swimming()),
            a.base_y, a.min_part_y, a.max_part_y, a.base_grav,
            a.max_force_mag, a.max_force_y, a.max_torque_mag,
            a.max_linvel, a.max_linvel_y, a.max_angvel,
            a.worst_part_idx, a.worst_part_kind, a.max_lever,
            anchor_sep.get(&entity).copied().unwrap_or(0.0),
            rapier_sep.get(&entity).copied().unwrap_or(0.0),
        );
        rows.push(row);
    }

    // ── Phase 2: write. ──
    let Some(writer) = probe.writer.as_mut() else { return; };
    for row in &rows { let _ = writer.write_all(row.as_bytes()); }
    let _ = writer.flush();
}

#[inline]
fn bool01(b: bool) -> u8 { if b { 1 } else { 0 } }
