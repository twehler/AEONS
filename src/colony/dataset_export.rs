// CSV dataset export. One-shot worker driven by `ExportDatasetRequested`.
//
// Row order: Carnivores → Herbivores → Photoautotrophs.
// Columns: identity + transform + every scalar/enum field of `Organism`
// (body_parts summarised by two derived counts), followed by all 19 DNA
// slots as individual `dna_<name>` columns.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use bevy::prelude::*;

use crate::colony::{Carnivore, Heterotroph, Organism, OrganismRoot, Photoautotroph};
use crate::lineages::dna::{DNA_DIM, DNA_FIELD_NAMES};
use crate::organism::{IntelligenceLevel, LineageRecord, Symmetry};
use std::collections::HashMap;


/// Filled by the statistics-panel handler on a click; consumed by
/// `export_dataset_system`. `None` means nothing to do.
#[derive(Resource, Default)]
pub struct ExportDatasetRequested(pub Option<PathBuf>);


/// Schedule of automatic dataset exports keyed on virtual elapsed time.
/// Each entry fires once when `elapsed_secs()` first crosses its threshold,
/// then is popped. Keyed on virtual time, so sim pausing just delays it.
/// Writes `datasets/<label>.csv` plus side-cars; dir created on demand.
#[derive(Resource)]
pub struct AutoExportSchedule {
    pub pending: Vec<(f32, &'static str)>,
}

impl Default for AutoExportSchedule {
    fn default() -> Self {
        // (virtual seconds, label/filename stem), declared in time-order
        // so the drain processes them in sequence. Fixed filenames so
        // re-runs overwrite cleanly after `rotate_existing_datasets`.
        Self {
            pending: vec![
                // Baseline. Fires at t=15 s, not 0: world load +
                // spawn_colony are gated on the async glb load and don't
                // finish until ~7 s, so a t=0 snapshot would be empty.
                (     15.0, "simulation_dataset_0_SECONDS"),
                (     60.0, "simulation_dataset_1_MINUTE"),
                (    180.0, "simulation_dataset_3_MINUTES"),
                (    300.0, "simulation_dataset_5_MINUTES"),
                (    420.0, "simulation_dataset_7_MINUTES"),
                (    600.0, "simulation_dataset_10_MINUTES"),
                (   1200.0, "simulation_dataset_20_MINUTES"),
                (   1800.0, "simulation_dataset_30_MINUTES"),
                (   3600.0, "simulation_dataset_60_MINUTES"),
                (  10800.0, "simulation_dataset_3_HOURS"),
                (  21600.0, "simulation_dataset_6_HOURS"),
                (  43200.0, "simulation_dataset_12_HOURS"),
                (  86400.0, "simulation_dataset_24_HOURS"),
                ( 172800.0, "simulation_dataset_48_HOURS"),
                ( 345600.0, "simulation_dataset_96_HOURS"),
                ( 561600.0, "simulation_dataset_156_HOURS"),
            ],
        }
    }
}


/// Rotate prior-run dataset artefacts so each new run gets the bare names
/// free. Bumps every `#N` to `#N+1` (descending, so we never overwrite an
/// existing file), then renames suffix-free `simulation_dataset_*.csv` to
/// `…#1.csv`. Operates on `datasets/`; no-op if absent. Side-cars rotate the
/// same way (they share the prefix). Rename failures are logged and swallowed.
pub fn rotate_existing_datasets() {
    let dir = std::path::Path::new("datasets");
    if !dir.exists() { return; }

    let entries = match std::fs::read_dir(dir) {
        Ok(it) => it,
        Err(e) => {
            warn!("rotate_existing_datasets: read_dir({}) failed: {}",
                  dir.display(), e);
            return;
        }
    };

    // (path, stem, level); level == 0 means no `#N` suffix.
    let mut found: Vec<(std::path::PathBuf, String, u32)> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() { continue; }
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n,
            None    => continue,
        };
        if let Some((stem, level)) = parse_rotation_filename(name) {
            found.push((path, stem, level));
        }
    }

    // Highest level first so renames never collide.
    found.sort_by(|a, b| b.2.cmp(&a.2));

    for (old_path, stem, level) in found {
        let new_name = format!("{stem}#{}.csv", level + 1);
        let new_path = dir.join(&new_name);
        if let Err(e) = std::fs::rename(&old_path, &new_path) {
            warn!("rotate_existing_datasets: rename {} → {} failed: {}",
                  old_path.display(), new_path.display(), e);
        }
    }
}

/// Parse `simulation_dataset_<stem>[#<level>].csv` → (stem, level).
/// `None` if the name lacks the dataset prefix or `.csv` extension.
fn parse_rotation_filename(name: &str) -> Option<(String, u32)> {
    let trimmed = name.strip_suffix(".csv")?;
    if !trimmed.starts_with("simulation_dataset_") { return None; }
    if let Some(hash_idx) = trimmed.rfind('#') {
        let level_str = &trimmed[hash_idx + 1..];
        if let Ok(level) = level_str.parse::<u32>() {
            return Some((trimmed[..hash_idx].to_string(), level));
        }
    }
    Some((trimmed.to_string(), 0))
}


/// Drains `AutoExportSchedule` as virtual time crosses each threshold,
/// filling `ExportDatasetRequested` with a `datasets/` path. Skips while a
/// prior request is still pending (the resource is single-slot).
pub fn tick_auto_export_schedule(
    virtual_time: Res<Time<Virtual>>,
    mut schedule: ResMut<AutoExportSchedule>,
    mut request:  ResMut<ExportDatasetRequested>,
) {
    if schedule.pending.is_empty() { return; }
    if request.0.is_some() { return; }

    let now = virtual_time.elapsed_secs();
    // First entry whose threshold has been crossed (declaration order).
    let idx = schedule.pending.iter().position(|(t, _)| now >= *t);
    if let Some(i) = idx {
        let (_t, label) = schedule.pending.remove(i);
        let dir = std::path::Path::new("datasets");
        if let Err(e) = std::fs::create_dir_all(dir) {
            error!("auto-export: failed to create dir {}: {}", dir.display(), e);
            return;
        }
        let path = dir.join(format!("{label}.csv"));
        info!("auto-export firing at virtual t={:.1}s → {}", now, path.display());
        request.0 = Some(path);
    }
}


/// Snapshot every live organism into a `;`-separated CSV at the requested
/// path. The path is `.take()`-d so one click → one write. Errors are
/// logged and swallowed so a filesystem failure doesn't disrupt the sim.
pub fn export_dataset_system(
    mut req: ResMut<ExportDatasetRequested>,
    query:   Query<
        (
            Entity, &Organism, &Transform,
            Has<Photoautotroph>, Has<Heterotroph>, Has<Carnivore>,
            Option<&LineageRecord>,
        ),
        With<OrganismRoot>,
    >,
    // Per body-part Avian state + last-applied-torque, joined to the
    // organism root via `ChildOf::parent()`, for the limb-pool CSV writer.
    limb_bp_q: Query<(
        &bevy::prelude::ChildOf,
        &crate::cell::BodyPartIndex,
        &bevy_rapier3d::prelude::Velocity,
        &crate::rapier_setup::LastAppliedTorque,
    )>,
    virtual_time: Res<Time<Virtual>>,
    pool:         NonSend<crate::intelligence_level_herbivore_1_sliding::BrainPoolHerbivore1>,
    pool_limb_h:  NonSend<crate::intelligence_level_herbivore_1_limb::BrainPoolHerbivore1Limb>,
    pool_limb_l2: NonSend<crate::intelligence_level_2_limb::BrainPoolL2Limb>,
    pool_limb_l3: NonSend<crate::intelligence_level_3_limb::BrainPoolL3Limb>,
) {
    let Some(path) = req.0.take() else { return };

    let file = match File::create(&path) {
        Ok(f) => f,
        Err(e) => {
            error!("export-dataset: failed to create {}: {}", path.display(), e);
            return;
        }
    };
    let mut out = BufWriter::new(file);

    // Single GPU forward pass; per-organism lookups are then pure CPU.
    let telemetry = pool.snapshot_telemetry();

    if let Err(e) = write_csv(&mut out, &query, virtual_time.elapsed_secs(), &pool, &telemetry) {
        error!("export-dataset: write failure on {}: {}", path.display(), e);
        return;
    }

    if let Err(e) = out.flush() {
        error!("export-dataset: flush failure on {}: {}", path.display(), e);
        return;
    }

    info!("exported simulation dataset to {}", path.display());

    // ── Side-car: `<stem>_training_stats.csv` — last
    // ≤ TRAINING_HISTORY_CAP completed training steps, one row each.
    let stats_path = match path.file_stem().and_then(|s| s.to_str()) {
        Some(stem) => path.with_file_name(format!("{stem}_training_stats.csv")),
        None       => path.with_extension("training_stats.csv"),
    };
    match File::create(&stats_path) {
        Ok(f) => {
            let mut sw = BufWriter::new(f);
            if let Err(e) = write_training_stats_csv(&mut sw, pool.training_history()) {
                error!("training-stats: write failure on {}: {}", stats_path.display(), e);
            } else {
                let _ = sw.flush();
                info!("training stats written to {}", stats_path.display());
            }
        }
        Err(e) => error!("training-stats: failed to create {}: {}", stats_path.display(), e),
    }

    // ── Side-car: `<stem>_brain_probe.csv` — 10 top + 10 bottom + 10
    // random-middle eaters, one row per weight tensor with summary stats.
    let probe_path = match path.file_stem().and_then(|s| s.to_str()) {
        Some(stem) => path.with_file_name(format!("{stem}_brain_probe.csv")),
        None       => path.with_extension("brain_probe.csv"),
    };
    match File::create(&probe_path) {
        Ok(f) => {
            let mut pw = BufWriter::new(f);
            if let Err(e) = write_brain_probe_csv(&mut pw, &query, &pool) {
                error!("brain-probe: write failure on {}: {}", probe_path.display(), e);
            } else {
                let _ = pw.flush();
                info!("brain probe written to {}", probe_path.display());
            }
        }
        Err(e) => error!("brain-probe: failed to create {}: {}", probe_path.display(), e),
    }

    // ── Per-pool limb-brain side-cars `<stem>_limb_<level>.csv`: one row
    // per enrolled organism with state + per-organism log_std stats.
    let snap_h  = pool_limb_h.0.snapshot();
    let snap_l2 = pool_limb_l2.0.snapshot();
    let snap_l3 = pool_limb_l3.0.snapshot();

    // Per-organism Avian telemetry: BASE body's lin/ang velocity (idx 0)
    // + max |torque| across parts, keyed by the OrganismRoot entity.
    let mut limb_telemetry: std::collections::HashMap<Entity, LimbBodyTelemetry> =
        std::collections::HashMap::new();
    for (child_of, idx, vel, torque) in &limb_bp_q {
        let root = child_of.parent();
        let entry = limb_telemetry.entry(root).or_default();
        let t_norm = torque.0.length();
        if t_norm > entry.max_torque_norm { entry.max_torque_norm = t_norm; }
        if idx.0 == 0 {
            entry.base_lin_vel = vel.linear;
            entry.base_ang_vel = vel.angular;
            entry.base_seen = true;
        }
    }

    write_limb_pool_csv(
        &path, "limb_herbivore_1", &snap_h, &query, &limb_telemetry,
        virtual_time.elapsed_secs(),
    );
    write_limb_pool_csv(
        &path, "limb_l2", &snap_l2, &query, &limb_telemetry,
        virtual_time.elapsed_secs(),
    );
    write_limb_pool_csv(
        &path, "limb_l3", &snap_l3, &query, &limb_telemetry,
        virtual_time.elapsed_secs(),
    );

    // ── Side-car: limb-pool training-stats CSVs (one per pool with
    // history). Schema matches the sliding pool's so one R reader fits both.
    write_limb_training_stats(&path, "limb_herbivore_1", pool_limb_h.0.training_history());
    write_limb_training_stats(&path, "limb_l2",          pool_limb_l2.0.training_history());
    write_limb_training_stats(&path, "limb_l3",          pool_limb_l3.0.training_history());
}

/// Per-organism Avian telemetry for `write_limb_pool_csv`. All-zero default
/// so organisms whose body parts aren't yet in Avian's query still get a row.
#[derive(Default, Clone, Copy)]
struct LimbBodyTelemetry {
    base_lin_vel:    Vec3,
    base_ang_vel:    Vec3,
    max_torque_norm: f32,
    /// True iff the base body part appeared in the query. False usually
    /// means a sliding organism (no Avian dynamic bodies).
    base_seen:       bool,
}

/// Write a limb-pool's training-step history to `<stem>_<pool>_training_stats.csv`.
/// No-op when the history is empty.
fn write_limb_training_stats(
    main_path:  &std::path::Path,
    pool_label: &str,
    history:    &std::collections::VecDeque<crate::limb_ppo::LimbTrainingStep>,
) {
    if history.is_empty() { return; }
    let stem = main_path.file_stem().and_then(|s| s.to_str()).unwrap_or("dataset");
    let out_path = main_path.with_file_name(format!("{stem}_{pool_label}_training_stats.csv"));
    let file = match File::create(&out_path) {
        Ok(f) => f,
        Err(e) => {
            error!("{pool_label}: failed to create {}: {}", out_path.display(), e);
            return;
        }
    };
    let mut w = BufWriter::new(file);
    let _ = writeln!(
        &mut w,
        "step;virtual_time_secs;n_active;actor_loss;critic_loss;entropy;\
         total_loss;mean_return;return_var;supervised_loss"
    );
    for s in history.iter() {
        let _ = writeln!(
            &mut w,
            "{};{:.3};{};{:.6};{:.6};{:.6};{:.6};{:.6};{:.6};{:.6}",
            s.step, s.virtual_time_secs, s.n_active,
            s.actor_loss, s.critic_loss, s.entropy,
            s.total_loss, s.mean_return, s.return_var,
            s.supervised_loss,
        );
    }
    let _ = w.flush();
    info!("limb-pool training stats written to {}", out_path.display());
}

/// Write a limb-pool snapshot to `<stem>_<pool>.csv`: one row per enrolled
/// organism with `Organism` state + per-dim actor `log_std`. No-op if empty.
fn write_limb_pool_csv(
    main_path:    &std::path::Path,
    pool_label:   &str,
    snap:         &crate::limb_ppo::LimbPoolSnapshot,
    query:        &Query<
        (
            Entity, &Organism, &Transform,
            Has<Photoautotroph>, Has<Heterotroph>, Has<Carnivore>,
            Option<&LineageRecord>,
        ),
        With<OrganismRoot>,
    >,
    telemetry: &std::collections::HashMap<Entity, LimbBodyTelemetry>,
    t_virtual: f32,
) {
    if snap.map.is_empty() { return; }

    let stem = main_path.file_stem().and_then(|s| s.to_str()).unwrap_or("dataset");
    let out_path = main_path.with_file_name(format!("{stem}_{pool_label}.csv"));
    let file = match File::create(&out_path) {
        Ok(f)  => f,
        Err(e) => {
            error!("{pool_label}: failed to create {}: {}", out_path.display(), e);
            return;
        }
    };
    let mut w = BufWriter::new(file);

    // Header. Includes the BASE body's lin/ang velocity + max commanded
    // torque magnitude across all body parts this tick.
    let _ = writeln!(
        &mut w,
        "entity;t_virtual;slot;x;y;z;energy;energy_norm;predations;reproductions;\
         is_carnivore;intelligence_level;\
         limb_target_0;limb_target_1;limb_target_2;limb_target_3;\
         limb_target_4;limb_target_5;limb_target_6;limb_target_7;\
         log_std_0;log_std_1;log_std_2;log_std_3;\
         log_std_4;log_std_5;log_std_6;log_std_7;\
         base_lin_vel_x;base_lin_vel_y;base_lin_vel_z;\
         base_ang_vel_x;base_ang_vel_y;base_ang_vel_z;\
         base_speed_xz;max_torque_norm"
    );

    for (e, org, transform, _is_photo, _is_hetero, is_carn, _lineage) in query.iter() {
        let Some(&slot) = snap.map.get(&e) else { continue };
        let s = slot as usize;

        // Per-dim log_std from the flat [N * OUT] buffer.
        let ls_base = s * crate::limb_ppo::OUT;
        let log_std = &snap.actor_log_std[ls_base..ls_base + crate::limb_ppo::OUT];

        let max_e = (org.grown_cell_count() as f32) * crate::energy::MAX_ENERGY_PER_CELL;
        let energy_norm = if max_e > 0.0 { (org.energy / max_e).clamp(0.0, 1.0) } else { 0.0 };

        let tel = telemetry.get(&e).copied().unwrap_or_default();
        let base_speed_xz = (tel.base_lin_vel.x * tel.base_lin_vel.x
                             + tel.base_lin_vel.z * tel.base_lin_vel.z).sqrt();

        let _ = writeln!(
            &mut w,
            "{};{};{};{};{};{};{};{};{};{};{};{};\
             {};{};{};{};{};{};{};{};\
             {};{};{};{};{};{};{};{};\
             {};{};{};{};{};{};{};{}",
            e.index(), t_virtual, slot,
            transform.translation.x, transform.translation.y, transform.translation.z,
            org.energy, energy_norm,
            org.predations, org.reproductions,
            bool01(is_carn),
            intelligence_label(org.intelligence_level),
            org.limb_targets[0], org.limb_targets[1], org.limb_targets[2], org.limb_targets[3],
            org.limb_targets[4], org.limb_targets[5], org.limb_targets[6], org.limb_targets[7],
            log_std[0], log_std[1], log_std[2], log_std[3],
            log_std[4], log_std[5], log_std[6], log_std[7],
            tel.base_lin_vel.x, tel.base_lin_vel.y, tel.base_lin_vel.z,
            tel.base_ang_vel.x, tel.base_ang_vel.y, tel.base_ang_vel.z,
            base_speed_xz, tel.max_torque_norm,
        );
    }

    let _ = w.flush();
    info!("limb-pool dataset written to {}", out_path.display());
}


/// Compute summary statistics on a flat weight vector.
/// Returns `(n, mean, std, min, max, l2_norm)`.
fn tensor_stats(v: &[f32]) -> (usize, f32, f32, f32, f32, f32) {
    let n = v.len();
    if n == 0 { return (0, 0.0, 0.0, 0.0, 0.0, 0.0); }
    let sum: f32 = v.iter().sum();
    let mean = sum / n as f32;
    let mut ss = 0.0_f32;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sq_sum = 0.0_f32;
    for &x in v {
        let d = x - mean; ss += d * d;
        if x < min { min = x; }
        if x > max { max = x; }
        sq_sum += x * x;
    }
    let std = (ss / n as f32).sqrt();
    let l2  = sq_sum.sqrt();
    (n, mean, std, min, max, l2)
}


fn write_brain_probe_csv<W: Write>(
    out:   &mut W,
    query: &Query<
        (
            Entity, &Organism, &Transform,
            Has<Photoautotroph>, Has<Heterotroph>, Has<Carnivore>,
            Option<&LineageRecord>,
        ),
        With<OrganismRoot>,
    >,
    pool:  &crate::intelligence_level_herbivore_1_sliding::BrainPoolHerbivore1,
) -> std::io::Result<()> {
    use rand::seq::SliceRandom;

    out.write_all(
        b"entity_id;predations;group;tensor;n;mean;std;min;max;l2_norm\n",
    )?;

    // Level1 + Heterotroph + !Carnivore organisms with a herbivore_1 slot.
    let mut candidates: Vec<(Entity, u32, u32)> = Vec::new();  // (entity, slot, predations)
    for (e, org, _tf, _is_photo, is_hetero, is_carn, _lr) in query.iter() {
        if !is_hetero || is_carn { continue; }
        if !matches!(org.intelligence_level, IntelligenceLevel::Level1) { continue; }
        if let Some(&slot) = pool.map.get(&e) {
            candidates.push((e, slot, org.predations as u32));
        }
    }
    if candidates.is_empty() { return Ok(()); }

    // Sort descending by predations so [0..10] = top, last 10 = bottom.
    candidates.sort_by(|a, b| b.2.cmp(&a.2));
    let n = candidates.len();
    let top_k    = candidates.iter().take(10).copied().collect::<Vec<_>>();
    let bottom_k = candidates.iter().rev().take(10).copied().collect::<Vec<_>>();
    // Random middle: indices not in top or bottom.
    let mut middle: Vec<(Entity, u32, u32)> = if n > 20 {
        candidates[10..n.saturating_sub(10)].to_vec()
    } else {
        Vec::new()
    };
    let mut rng = rand::rng();
    middle.shuffle(&mut rng);
    let random_k: Vec<_> = middle.into_iter().take(10).collect();

    let groups: [(&str, &[(Entity, u32, u32)]); 3] = [
        ("top",    &top_k),
        ("bottom", &bottom_k),
        ("random", &random_k),
    ];

    // Single-MLP tensor layout: w1, b1, w2, b2.
    let tensor_names = ["w1", "b1", "w2", "b2"];

    let mut row = String::with_capacity(192);
    for (group_name, items) in groups.iter() {
        for &(entity, slot, predations) in *items {
            let brain = pool.extract_slot(slot);
            let vecs: [&Vec<f32>; 4] = [
                &brain.w1, &brain.b1, &brain.w2, &brain.b2,
            ];
            for (name, v) in tensor_names.iter().zip(vecs.iter()) {
                let (n, mean, std, min, max, l2) = tensor_stats(v);
                row.clear();
                use std::fmt::Write as _;
                let _ = write!(
                    row,
                    "{};{};{};{};{};{:.6};{:.6};{:.6};{:.6};{:.6}\n",
                    entity.index(), predations, group_name, name,
                    n, mean, std, min, max, l2,
                );
                out.write_all(row.as_bytes())?;
            }
        }
    }

    Ok(())
}


/// Write the training-stats side-car CSV (one row per completed
/// train_step in the pool's history ring).
fn write_training_stats_csv<W: Write>(
    out:     &mut W,
    history: &std::collections::VecDeque<crate::intelligence_level_herbivore_1_sliding::TrainingStep>,
) -> std::io::Result<()> {
    out.write_all(
        b"step;virtual_time_secs;n_active;actor_loss;critic_loss;entropy;\
          total_loss;mean_return;return_var;supervised_loss\n",
    )?;
    let mut row = String::with_capacity(256);
    for s in history.iter() {
        row.clear();
        use std::fmt::Write as _;
        let _ = write!(
            row,
            "{};{:.3};{};{:.6};{:.6};{:.6};{:.6};{:.6};{:.6};{:.6}\n",
            s.step, s.virtual_time_secs, s.n_active,
            s.actor_loss, s.critic_loss, s.entropy,
            s.total_loss, s.mean_return, s.return_var,
            s.supervised_loss,
        );
        out.write_all(row.as_bytes())?;
    }
    Ok(())
}


/// Trophic class for row-ordering and the `trophic_class` column. Derived
/// from marker components, not `Organism` content (cell counts can drift
/// mid-consumption).
#[derive(Clone, Copy, PartialEq, Eq)]
enum TrophicClass {
    Carnivore,
    Herbivore,
    Photoautotroph,
}

impl TrophicClass {
    fn label(self) -> &'static str {
        match self {
            TrophicClass::Carnivore      => "Carnivore",
            TrophicClass::Herbivore      => "Herbivore",
            TrophicClass::Photoautotroph => "Photoautotroph",
        }
    }

    /// Sort key — lower = earlier row.
    fn order(self) -> u8 {
        match self {
            TrophicClass::Carnivore      => 0,
            TrophicClass::Herbivore      => 1,
            TrophicClass::Photoautotroph => 2,
        }
    }
}


fn classify(is_photo: bool, is_hetero: bool, is_carn: bool) -> Option<TrophicClass> {
    if is_carn  && is_hetero { return Some(TrophicClass::Carnivore); }
    if is_hetero             { return Some(TrophicClass::Herbivore); }
    if is_photo              { return Some(TrophicClass::Photoautotroph); }
    None
}


fn write_csv<W: Write>(
    out:       &mut W,
    query:     &Query<
        (
            Entity, &Organism, &Transform,
            Has<Photoautotroph>, Has<Heterotroph>, Has<Carnivore>,
            Option<&LineageRecord>,
        ),
        With<OrganismRoot>,
    >,
    now_secs:  f32,
    pool:      &crate::intelligence_level_herbivore_1_sliding::BrainPoolHerbivore1,
    telemetry: &[crate::intelligence_level_herbivore_1_sliding::BrainTelemetry],
) -> std::io::Result<()> {
    // ── Header ────────────────────────────────────────────────
    let mut header = String::new();
    let cols = [
        // Identity / context
        "entity_id", "trophic_class", "species_id",
        "pos_x", "pos_y", "pos_z",
        // Body plan
        "symmetry", "intelligence_level", "is_sessile",
        "has_variable_form", "adult",
        "photo_cell_count", "non_photo_cell_count",
        "grown_cell_count", "alive_body_part_count",
        // Energetics / RL signals
        "energy", "in_sunlight",
        "reproduced", "reproductions", "predations",
        "hunger", "dopamine", "target_distance",
        // Movement
        "movement_speed",
        "movement_dir_x", "movement_dir_y", "movement_dir_z",
        "velocity_x",     "velocity_y",     "velocity_z",
        "is_climbing", "climb_energy_debt", "cached_bounding_radius",
        // Lineage
        "parent_id", "age_secs",
        "times_reproduced_self", "child_alive_count",
        // Brain telemetry — only for herbivore_1 pool members
        // (Level1 + Heterotroph + !Carnivore); others emit empty cells.
        "brain_mu_speed", "brain_mu_angle",
        "brain_log_sigma_speed", "brain_log_sigma_angle",
        "brain_value_v",
        "brain_last_reward", "brain_mean_reward_64",
        "brain_last_eat_component", "brain_last_progress_component",
        "brain_last_oracle_component",
    ];
    for (i, c) in cols.iter().enumerate() {
        if i > 0 { header.push(';'); }
        header.push_str(c);
    }
    for name in DNA_FIELD_NAMES.iter() {
        header.push(';');
        header.push_str("dna_");
        header.push_str(name);
    }
    header.push('\n');
    out.write_all(header.as_bytes())?;

    // Parent→children count map: one pass incrementing each parent_id.
    let mut child_count: HashMap<Entity, u32> = HashMap::with_capacity(64);
    for (_e, _o, _t, _p, _h, _c, lr) in query.iter() {
        if let Some(LineageRecord { parent_id: Some(p), .. }) = lr {
            *child_count.entry(*p).or_insert(0) += 1;
        }
    }

    // Collect + sort by (TrophicClass, entity) to force the row order
    // carnivores → herbivores → photoautotrophs (queries are archetype-order).
    let mut rows: Vec<(TrophicClass, Entity, &Organism, &Transform, Option<&LineageRecord>)> = query.iter()
        .filter_map(|(e, org, tf, is_photo, is_hetero, is_carn, lr)| {
            classify(is_photo, is_hetero, is_carn).map(|tc| (tc, e, org, tf, lr))
        })
        .collect();
    rows.sort_by_key(|(tc, e, _, _, _)| (tc.order(), e.index()));

    for (tc, entity, organism, transform, lr) in rows {
        let kids = child_count.get(&entity).copied().unwrap_or(0);
        // None when the organism isn't in the herbivore_1 pool → empty cells.
        let telem = pool.map.get(&entity)
            .and_then(|&slot| telemetry.get(slot as usize));
        write_row(out, tc, entity, organism, transform, lr, kids, now_secs, telem)?;
    }

    Ok(())
}


fn write_row<W: Write>(
    out:        &mut W,
    tc:         TrophicClass,
    entity:     Entity,
    organism:   &Organism,
    transform:  &Transform,
    lineage:    Option<&LineageRecord>,
    child_count: u32,
    now_secs:   f32,
    telem:      Option<&crate::intelligence_level_herbivore_1_sliding::BrainTelemetry>,
) -> std::io::Result<()> {
    let pos = transform.translation;

    // Single owned String → one write syscall. .6 precision matches the
    // .colony save format so the file round-trips with negligible loss.
    let mut row = String::with_capacity(512);

    macro_rules! push {
        ($($t:tt)*) => {{ use std::fmt::Write as _; let _ = write!(row, $($t)*); }};
    }
    macro_rules! sep { () => { row.push(';'); } }

    // Identity
    push!("{}", entity.index());
    sep!(); push!("{}", tc.label());
    sep!(); match organism.species_id {
        Some(id) => push!("{}", id),
        None     => (),                  // empty cell on None
    };
    // Position
    sep!(); push!("{:.6}", pos.x);
    sep!(); push!("{:.6}", pos.y);
    sep!(); push!("{:.6}", pos.z);

    // Body plan
    sep!(); push!("{}", symmetry_label(organism.symmetry));
    sep!(); push!("{}", intelligence_label(organism.intelligence_level));
    sep!(); push!("{}", bool01(organism.is_sessile));
    sep!(); push!("{}", bool01(organism.has_variable_form));
    sep!(); push!("{}", bool01(organism.adult));
    sep!(); push!("{}", organism.photo_cell_count);
    sep!(); push!("{}", organism.non_photo_cell_count);
    sep!(); push!("{}", organism.grown_cell_count());
    sep!(); push!("{}", organism.alive_body_part_count());

    // Energetics / RL signals
    sep!(); push!("{:.6}", organism.energy);
    sep!(); push!("{}", bool01(organism.in_sunlight));
    sep!(); push!("{}", bool01(organism.reproduced));
    sep!(); push!("{}", organism.reproductions);
    sep!(); push!("{}", organism.predations);
    sep!(); push!("{:.6}", organism.hunger);
    sep!(); push!("{:.6}", organism.dopamine);
    sep!(); push!("{:.6}", organism.target_distance);

    // Movement
    sep!(); push!("{:.6}", organism.movement_speed);
    sep!(); push!("{:.6}", organism.movement_direction.x);
    sep!(); push!("{:.6}", organism.movement_direction.y);
    sep!(); push!("{:.6}", organism.movement_direction.z);
    sep!(); push!("{:.6}", organism.velocity.x);
    sep!(); push!("{:.6}", organism.velocity.y);
    sep!(); push!("{:.6}", organism.velocity.z);
    sep!(); push!("{}", bool01(organism.is_climbing));
    sep!(); push!("{:.6}", organism.climb_energy_debt);
    sep!(); push!("{:.6}", organism.cached_bounding_radius);

    // Lineage. `parent_id`: empty for initial-cohort, else parent's entity
    // index. `age_secs`: virtual-time delta since spawn, or 0 if no record.
    sep!();
    if let Some(lr) = lineage {
        if let Some(p) = lr.parent_id { push!("{}", p.index()); }
        // else: empty cell
    }
    sep!();
    let age_secs = lineage.map(|lr| (now_secs - lr.spawn_time_secs).max(0.0))
                          .unwrap_or(0.0);
    push!("{:.3}", age_secs);
    sep!();
    push!("{}", lineage.map(|lr| lr.times_reproduced_self).unwrap_or(0));
    sep!();
    push!("{}", child_count);

    // Brain telemetry — ".6" floats when available, blank otherwise.
    let push_telem = |row: &mut String, v: Option<f32>| {
        use std::fmt::Write as _;
        row.push(';');
        if let Some(x) = v { let _ = write!(row, "{:.6}", x); }
    };
    push_telem(&mut row, telem.map(|t| t.mu_speed));
    push_telem(&mut row, telem.map(|t| t.mu_angle));
    push_telem(&mut row, telem.map(|t| t.log_sigma_speed));
    push_telem(&mut row, telem.map(|t| t.log_sigma_angle));
    push_telem(&mut row, telem.map(|t| t.value_v));
    push_telem(&mut row, telem.map(|t| t.last_reward));
    push_telem(&mut row, telem.map(|t| t.mean_reward_64));
    push_telem(&mut row, telem.map(|t| t.last_eat_component));
    push_telem(&mut row, telem.map(|t| t.last_progress_component));
    push_telem(&mut row, telem.map(|t| t.last_oracle_component));

    // DNA — DNA_DIM slots in DNA_FIELD_NAMES order; empty cells past
    // the available length guard a short/corrupt dna vector.
    for i in 0..DNA_DIM {
        sep!();
        if let Some(v) = organism.dna.get(i) {
            push!("{:.6}", v);
        }
    }

    row.push('\n');
    out.write_all(row.as_bytes())
}


#[inline]
fn bool01(b: bool) -> u8 { if b { 1 } else { 0 } }

fn symmetry_label(s: Symmetry) -> &'static str {
    match s {
        Symmetry::NoSymmetry => "NoSymmetry",
        Symmetry::Bilateral  => "Bilateral",
    }
}

fn intelligence_label(il: IntelligenceLevel) -> &'static str {
    match il {
        IntelligenceLevel::Level0 => "Level0",
        IntelligenceLevel::Level1 => "Level1",
        IntelligenceLevel::Level2 => "Level2",
        IntelligenceLevel::Level3 => "Level3",
    }
}
