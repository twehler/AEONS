// Continuous per-organism time-series logger.
//
// Every `LOG_INTERVAL_SECS` virtual seconds, writes one CSV row per Level1
// herbivore to `datasets/time_series_<timestamp>.csv` (for
// `data-analysis/time_series.R`) to trace per-individual learning trajectories
// over virtual time. File opened lazily, kept open for the process lifetime.
// Gated on the sim running so paused intervals don't add redundant rows.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use bevy::prelude::*;

use crate::colony::{Carnivore, Heterotroph, Organism, OrganismRoot};
use crate::intelligence_level_herbivore_1_sliding::{
    BrainPoolHerbivore1, BrainSlotHerbivore1,
};


/// Cadence for time-series rows (virtual seconds).
const LOG_INTERVAL_SECS: f32 = 5.0;


/// Lazy-init logger state; `writer` is `None` until the first log tick.
#[derive(Resource, Default)]
pub struct TimeSeriesLogger {
    writer:         Option<BufWriter<File>>,
    last_log_secs:  f32,
    /// True once writer-open was attempted; prevents repeated error-spam.
    init_attempted: bool,
}


pub fn tick_time_series_logger(
    mut logger:   ResMut<TimeSeriesLogger>,
    sim_running:  Res<crate::simulation_settings::SimulationRunning>,
    virtual_time: Res<Time<Virtual>>,
    pool:         NonSend<BrainPoolHerbivore1>,
    query:        Query<
        (Entity, &Organism, &BrainSlotHerbivore1),
        (With<OrganismRoot>, With<Heterotroph>, Without<Carnivore>),
    >,
) {
    if !sim_running.0 { return; }

    let now = virtual_time.elapsed_secs();
    if now - logger.last_log_secs < LOG_INTERVAL_SECS { return; }
    logger.last_log_secs = now;

    // Lazy-init the writer on the first log tick.
    if !logger.init_attempted {
        logger.init_attempted = true;
        let path = PathBuf::from(format!(
            "datasets/time_series_{}.csv",
            chrono::Local::now().format("%d-%m-%Y-%H-%M-%S"),
        ));
        if let Some(parent) = path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                error!("time-series: failed to create dir {}: {}", parent.display(), e);
                return;
            }
        }
        // Timestamped filename → not touched by `rotate_existing_datasets`
        // (which only matches the `simulation_dataset_*` prefix).
        match File::create(&path) {
            Ok(f) => {
                let mut w = BufWriter::new(f);
                if let Err(e) = w.write_all(
                    b"virtual_time_secs;entity_id;predations;dopamine;energy;hunger;\
                      target_distance;movement_speed;\
                      brain_mu_speed;brain_mu_angle;brain_log_sigma_angle;brain_value_v;\
                      brain_last_reward;brain_mean_reward_64;\
                      brain_last_eat_component;brain_last_progress_component\n",
                ) {
                    error!("time-series: header write failed: {e}");
                    return;
                }
                info!("time-series logging to {}", path.display());
                logger.writer = Some(w);
            }
            Err(e) => {
                error!("time-series: failed to create {}: {}", path.display(), e);
            }
        }
    }

    let Some(writer) = logger.writer.as_mut() else { return; };

    // Single GPU forward pass; per-organism lookups are then pure CPU.
    let telemetry = pool.snapshot_telemetry();

    let mut row = String::with_capacity(256);
    for (entity, organism, slot) in query.iter() {
        let s = slot.0 as usize;
        if s >= telemetry.len() { continue; }
        let t = &telemetry[s];
        row.clear();
        use std::fmt::Write as _;
        let _ = write!(
            row,
            "{:.3};{};{};{:.4};{:.3};{:.4};{:.3};{:.4};\
             {:.6};{:.6};{:.6};{:.6};{:.6};{:.6};{:.6};{:.6}\n",
            now, entity.index(),
            organism.predations, organism.dopamine, organism.energy, organism.hunger,
            organism.target_distance, organism.movement_speed,
            t.mu_speed, t.mu_angle, t.log_sigma_angle, t.value_v,
            t.last_reward, t.mean_reward_64,
            t.last_eat_component, t.last_progress_component,
        );
        let _ = writer.write_all(row.as_bytes());
    }
    // Flush every tick so a crash doesn't lose the most recent window.
    let _ = writer.flush();
}
