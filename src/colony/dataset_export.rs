// CSV dataset export.
//
// One-shot worker driven by the `ExportDatasetRequested` resource.
// When the user clicks "Export Simulation Dataset" in the statistics
// panel, that handler pauses the simulation, opens a native save
// dialog, and writes the chosen path into the resource. The system
// below picks the path up on the next Update tick, snapshots every
// living organism, and writes a `;`-separated CSV file.
//
// Row order: Carnivores → Herbivores → Photoautotrophs. Within each
// group, organisms are emitted in whatever order the Bevy query
// yields them (effectively archetype-then-spawn order).
//
// Columns: identity + transform + every scalar/enum field of
// `Organism` (the `body_parts` Vec is summarised by two derived
// counts; the raw nested structure would not fit in a flat CSV
// cleanly), followed by all 19 DNA slots emitted as individual
// `dna_<name>` columns — the DNA vector itself never appears as one
// opaque column.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use bevy::prelude::*;

use crate::colony::{Carnivore, Heterotroph, Organism, OrganismRoot, Photoautotroph};
use crate::lineages::dna::{DNA_DIM, DNA_FIELD_NAMES};
use crate::organism::{IntelligenceLevel, LineageRecord, Symmetry};
use std::collections::HashMap;


/// Filled by the statistics-panel handler on a click. Consumed by
/// `export_dataset_system` on the next Update tick. `None` means
/// "nothing to do".
#[derive(Resource, Default)]
pub struct ExportDatasetRequested(pub Option<PathBuf>);


/// Schedule of automatic dataset exports keyed on virtual elapsed
/// time. Each entry fires once when `Time<Virtual>::elapsed_secs()`
/// first crosses the configured threshold; the entry is consumed
/// (popped) so subsequent ticks don't re-fire. Survives sim
/// pausing — only virtual time advances the trigger, so pausing
/// the simulation just delays the export.
///
/// Entries are written to `datasets/<label>.csv` (and the matching
/// `_training_stats.csv` / `_brain_probe.csv` side-cars via the
/// normal export pipeline). The directory is created on demand.
#[derive(Resource)]
pub struct AutoExportSchedule {
    pub pending: Vec<(f32, &'static str)>,
}

impl Default for AutoExportSchedule {
    fn default() -> Self {
        // (virtual seconds, label/filename stem). One entry per
        // milestone — declared in time-order so the drain processes
        // them in sequence. Filenames are fixed (not timestamped) so
        // re-runs OVERWRITE prior exports cleanly (`File::create`
        // truncates) AFTER `rotate_existing_datasets` has bumped the
        // previous run's artefacts to `#1` at startup.
        Self {
            pending: vec![
                (    300.0, "simulation_dataset_5_MINUTES"),
                (    600.0, "simulation_dataset_10_MINUTES"),
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


/// Rotate prior-run dataset artefacts so each new simulation starts
/// with the bare names free. Bumps every `#N` to `#N+1` (descending,
/// so we never write to a file that exists), then renames every
/// suffix-free `simulation_dataset_*.csv` to `…#1.csv`. This produces
/// a clean history: the current run lives at the bare names, the
/// previous run at `#1`, the run before that at `#2`, and so on.
///
/// Operates on the `datasets/` directory. Skips silently if the
/// directory doesn't exist yet (first run). Side-cars
/// (`_training_stats.csv`, `_brain_probe.csv`) get rotated the same
/// way — the suffix pattern matcher treats them as independent files
/// that all happen to share the `simulation_dataset_` prefix.
///
/// Failures are logged at `warn!` and otherwise swallowed; one
/// stuck rename shouldn't crash the simulation.
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

    // Collect every dataset-shaped file as (path, stem, level)
    // where `level == 0` means no `#N` suffix.
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

    // Highest level first so we never collide. Within a level the
    // order doesn't matter — each rename's destination is unique.
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
/// Returns `None` if the name doesn't match the dataset prefix or
/// extension. Stems may contain underscores, digits, mixed case —
/// everything between the dataset prefix and `#`/`.csv`.
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


/// Drains `AutoExportSchedule` as virtual time crosses each
/// threshold, populating `ExportDatasetRequested` with a path under
/// `datasets/`. Idempotent — once an entry fires, it's removed.
/// Skips silently if a prior export request is still pending
/// (avoids racing on the single-slot resource).
pub fn tick_auto_export_schedule(
    virtual_time: Res<Time<Virtual>>,
    mut schedule: ResMut<AutoExportSchedule>,
    mut request:  ResMut<ExportDatasetRequested>,
) {
    if schedule.pending.is_empty() { return; }
    if request.0.is_some() { return; }

    let now = virtual_time.elapsed_secs();
    // Find the first entry whose threshold has been crossed. Stable
    // ordering means we drain in the order entries were declared.
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


/// Snapshot every live organism into a `;`-separated CSV at the
/// requested path. The path is `.take()`-d so a single click produces
/// a single write — subsequent ticks find the resource empty and
/// skip. Errors are logged at `error!` and silently swallowed
/// otherwise so a file-system failure doesn't disrupt the simulation.
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
    virtual_time: Res<Time<Virtual>>,
    pool:         NonSend<crate::intelligence_level_herbivore_1::BrainPoolHerbivore1>,
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

    // Snapshot every slot's brain output once — single forward
    // pass on the GPU, then per-organism row lookups are pure
    // CPU indexing.
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

    // ── Side-car: training-statistics CSV. Filename derived from
    // the dataset path: `<stem>_training_stats.csv` in the same
    // directory. Contains the last ≤ TRAINING_HISTORY_CAP completed
    // training steps, one row each. Errors are non-fatal — the
    // dataset CSV is the primary artifact.
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

    // ── Side-car: brain-weight probe CSV. Picks 10 top eaters +
    // 10 bottom eaters + 10 random middle for weight inspection.
    // For each selected organism, writes 12 rows (one per weight
    // tensor) with n/mean/std/min/max/l2_norm. Compact enough to
    // analyse in R without a binary reader.
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
    pool:  &crate::intelligence_level_herbivore_1::BrainPoolHerbivore1,
) -> std::io::Result<()> {
    use rand::seq::SliceRandom;

    out.write_all(
        b"entity_id;predations;group;tensor;n;mean;std;min;max;l2_norm\n",
    )?;

    // Collect every Level1 + Heterotroph + !Carnivore organism that
    // currently has a slot in the herbivore_1 pool. Others can't
    // contribute a brain payload.
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
    // Random middle: skip indices that are in top or bottom.
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

    // Post-L3-port tensor layout: a single MLP (w1, b1, w2, b2). The
    // old 12-tensor (backbone+actor+critic) layout was retired with
    // the architecture change. CSV column counts adjust automatically
    // — only the rows-per-organism count changes (12 → 4).
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
    history: &std::collections::VecDeque<crate::intelligence_level_herbivore_1::TrainingStep>,
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


/// Trophic class used for both row-ordering and the `trophic_class`
/// column. Determined from the entity's marker components, not from
/// `Organism` content (cell counts can drift mid-consumption).
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
    pool:      &crate::intelligence_level_herbivore_1::BrainPoolHerbivore1,
    telemetry: &[crate::intelligence_level_herbivore_1::BrainTelemetry],
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
        // Brain telemetry — only populated for Level1 + Heterotroph
        // + !Carnivore organisms (i.e. herbivore_1 pool members).
        // Other organisms emit empty cells.
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

    // ── Build parent→children count map ──────────────────────
    // One pass over the cohort: for every alive organism whose
    // LineageRecord has a parent_id, increment the parent's count.
    // O(n); n ~ 200, so the cost is negligible per export.
    let mut child_count: HashMap<Entity, u32> = HashMap::with_capacity(64);
    for (_e, _o, _t, _p, _h, _c, lr) in query.iter() {
        if let Some(LineageRecord { parent_id: Some(p), .. }) = lr {
            *child_count.entry(*p).or_insert(0) += 1;
        }
    }

    // ── Collect + sort rows ───────────────────────────────────
    // Bevy queries iterate archetype-order which mixes the
    // trophic classes; collect with the (TrophicClass, entity)
    // sort key to honour the user-requested ordering
    // (carnivores → herbivores → photoautotrophs).
    let mut rows: Vec<(TrophicClass, Entity, &Organism, &Transform, Option<&LineageRecord>)> = query.iter()
        .filter_map(|(e, org, tf, is_photo, is_hetero, is_carn, lr)| {
            classify(is_photo, is_hetero, is_carn).map(|tc| (tc, e, org, tf, lr))
        })
        .collect();
    rows.sort_by_key(|(tc, e, _, _, _)| (tc.order(), e.index()));

    // ── Per-row write ─────────────────────────────────────────
    for (tc, entity, organism, transform, lr) in rows {
        let kids = child_count.get(&entity).copied().unwrap_or(0);
        // Brain telemetry lookup: pool maps `entity` → slot, slot
        // indexes into the precomputed `telemetry` Vec. None means
        // the organism isn't in the herbivore_1 pool (photo,
        // carnivore, Level0, etc.) — writer emits empty cells.
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
    telem:      Option<&crate::intelligence_level_herbivore_1::BrainTelemetry>,
) -> std::io::Result<()> {
    let pos = transform.translation;

    // Build the row as a single owned String so we can write it in
    // one syscall.  Per-cell precision is .6 (matches the .colony
    // save format) so the file round-trips with negligible loss.
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

    // Lineage. `parent_id` writes an empty cell when the organism
    // is initial-cohort (no parent), else the parent's `Entity` as
    // an integer index for matching with the `entity_id` column.
    // `age_secs` is the virtual-time delta since this organism
    // spawned, or 0 when no LineageRecord is attached.
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

    // Brain telemetry. Nine columns; written as ".6" floats when
    // available, blank otherwise. Allows correlating each
    // organism's policy output with its observed behaviour at
    // snapshot time.
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

    // DNA — exactly DNA_DIM slots, matching DNA_FIELD_NAMES order.
    // Defend against a corrupt / partially-populated dna vector by
    // emitting empty cells past the available length.
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
