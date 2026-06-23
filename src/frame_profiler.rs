// Frame profiler — a coarse, behaviour-neutral wall-time breakdown to locate
// the FPS bottleneck (companion to behaviour.rs's `BRAINTICK` line).
//
// Pass 1 found the frame is CPU/main-world bound (≈47 of 48ms) and that the
// rapier PHYSICS solve is ≈44ms of it (91% of the frame) on the multibody
// Snakes. Pass 2 (this file) splits physics into rapier's three phases so we
// know WHICH part to attack — the fix differs entirely per phase:
//
//   * p_sync  (PhysicsSet::SyncBackend)   — Bevy→rapier: (re)building/updating
//                                           colliders & bodies. Large here ⇒
//                                           per-frame collider rebuild.
//   * p_step  (PhysicsSet::StepSimulation)— broad/narrow collision + the
//                                           constraint/multibody solve. Large
//                                           here ⇒ contact-pair explosion or
//                                           solver iteration cost.
//   * p_wb    (PhysicsSet::Writeback)     — rapier→Bevy transforms.
//
// Also logs live rigid-body / collider counts (scale of the physics world) and
// the population. Reported once per real second as `FRAMEPROF`, in PER-FRAME ms.

use bevy::prelude::*;
use bevy_rapier3d::prelude::{PhysicsSet, RigidBody, Collider};
use std::collections::HashMap;
use std::time::Instant;

#[derive(Resource, Default)]
pub struct FrameProfiler {
    /// Open span start stamps (name → when `begin` was called).
    open:   HashMap<&'static str, Instant>,
    /// Accumulated ms per span over the current ~1s reporting window.
    acc:    HashMap<&'static str, f32>,
    /// Frames completed this window (incremented at `frame_main` end).
    frames: u32,
    /// Fixed-update physics steps this window (incremented per rapier step).
    /// Disambiguates "each step is expensive" from "many steps per frame"
    /// (the FixedUpdate catch-up feedback loop).
    steps:  u32,
    /// Window start; `None` until the first report tick seeds it.
    window: Option<Instant>,
}

impl FrameProfiler {
    fn begin(&mut self, name: &'static str) {
        self.open.insert(name, Instant::now());
    }
    fn end(&mut self, name: &'static str) {
        if let Some(t) = self.open.remove(name) {
            *self.acc.entry(name).or_insert(0.0) += t.elapsed().as_secs_f32() * 1000.0;
        }
    }
    fn pf(&self, name: &str, frames: u32) -> f32 {
        self.acc.get(name).copied().unwrap_or(0.0) / frames.max(1) as f32
    }
}

pub struct FrameProfilerPlugin;

impl Plugin for FrameProfilerPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FrameProfiler>();
        // Whole main-world frame: First (runs before everything) → Last.
        app.add_systems(First, frame_main_begin);
        app.add_systems(Last, (frame_main_end, report_frame_profile).chain());
        // Physics runs `in_fixed_schedule`; split it at the three PhysicsSet
        // boundaries. Each helper closes the previous phase and opens the next,
        // so the three spans partition the rapier step with no gaps/overlap.
        app.add_systems(FixedUpdate, phys_sync_begin .before(PhysicsSet::SyncBackend));
        app.add_systems(FixedUpdate, phys_sync_to_step.after(PhysicsSet::SyncBackend)   .before(PhysicsSet::StepSimulation));
        app.add_systems(FixedUpdate, phys_step_to_wb  .after(PhysicsSet::StepSimulation).before(PhysicsSet::Writeback));
        app.add_systems(FixedUpdate, phys_wb_end      .after(PhysicsSet::Writeback));
    }
}

fn frame_main_begin(mut p: ResMut<FrameProfiler>) { p.begin("frame_main"); }
fn frame_main_end(mut p: ResMut<FrameProfiler>)   { p.end("frame_main"); p.frames += 1; }

fn phys_sync_begin(mut p: ResMut<FrameProfiler>)   { p.steps += 1; p.begin("p_sync"); }
fn phys_sync_to_step(mut p: ResMut<FrameProfiler>) { p.end("p_sync"); p.begin("p_step"); }
fn phys_step_to_wb(mut p: ResMut<FrameProfiler>)   { p.end("p_step"); p.begin("p_wb"); }
fn phys_wb_end(mut p: ResMut<FrameProfiler>)       { p.end("p_wb"); }

/// Log a `FRAMEPROF` line once per real second, in PER-FRAME milliseconds, with
/// the physics phase split, physics-world scale (body/collider counts), and the
/// live population.
fn report_frame_profile(
    mut p:      ResMut<FrameProfiler>,
    bodies:     Query<(), With<RigidBody>>,
    colliders:  Query<(), With<Collider>>,
    heteros:    Query<(), With<crate::colony::Heterotroph>>,
    carnivores: Query<(), With<crate::colony::Carnivore>>,
    photos:     Query<(), With<crate::colony::Photoautotroph>>,
) {
    let now = Instant::now();
    let start = *p.window.get_or_insert(now);
    let secs = (now - start).as_secs_f32();
    if secs < 1.0 { return; }

    let frames = p.frames;
    if frames == 0 {
        // Sim paused / no frames — reset the window, don't divide by zero.
        p.acc.clear();
        p.window = Some(now);
        return;
    }

    let fps      = frames as f32 / secs;
    let frame_ms = 1000.0 / fps;
    let main_pf  = p.pf("frame_main", frames);
    let sync_pf  = p.pf("p_sync", frames);
    let step_pf  = p.pf("p_step", frames);
    let wb_pf    = p.pf("p_wb",   frames);
    let phys_pf  = sync_pf + step_pf + wb_pf;
    let other_pf = (main_pf - phys_pf).max(0.0);
    let gpu_gap  = (frame_ms - main_pf).max(0.0);

    // Disambiguate per-step cost vs steps-per-frame (the catch-up loop).
    let steps         = p.steps;
    let steps_per_fr  = steps as f32 / frames as f32;
    let phys_per_step = if steps > 0 { phys_pf * frames as f32 / steps as f32 } else { 0.0 };

    let n_body  = bodies.iter().count();
    let n_coll  = colliders.iter().count();
    let n_het   = heteros.iter().count();
    let n_carn  = carnivores.iter().count();
    let n_photo = photos.iter().count();

    info!(
        "FRAMEPROF fps={:.1} frame={:.1}ms | main={:.1} (physics={:.1}: sync={:.1} step={:.1} wb={:.1} | other={:.1}) | gpu={:.1} | steps/fr={:.1} phys/step={:.1}ms | bodies={} colliders={} | pop het={} carn={} photo={}",
        fps, frame_ms, main_pf, phys_pf, sync_pf, step_pf, wb_pf, other_pf, gpu_gap,
        steps_per_fr, phys_per_step, n_body, n_coll, n_het, n_carn, n_photo,
    );

    p.acc.clear();
    p.frames = 0;
    p.steps = 0;
    p.window = Some(now);
}
