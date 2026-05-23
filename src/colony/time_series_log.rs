// Continuous per-organism time-series logger.
//
// Every `LOG_INTERVAL_SECS` virtual seconds the system writes one
// CSV row per Level1 herbivore to `logs/time_series_<timestamp>.csv`.
// The file is opened lazily on the first log tick and stays open
// for the lifetime of the process. Lets us trace per-individual
// learning trajectories (mu_angle, value_v, mean_reward_64 over
// virtual time) instead of relying on two point-in-time snapshots
// at minute 1 and minute 45 to infer learning dynamics.
//
// Cadence is gated on the simulation actually running, so paused
// intervals don't pollute the log with redundant rows.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use bevy::prelude::*;

use crate::colony::{Carnivore, Heterotroph, Organism, OrganismRoot};
use crate::intelligence_level_herbivore_1::{
    BrainPoolHerbivore1, BrainSlotHerbivore1,
};


/// Default cadence for time-series rows. 5 virtual seconds gives
/// ~12 samples per minute per herbivore; with ~200 herbivores
/// that's ~24k rows / hour or ~3 MB / hour — comfortable.
const LOG_INTERVAL_SECS: f32 = 5.0;


/// Lazy-init logger state. The writer is `None` until the first
/// log tick, at which point a file is created at
/// `logs/time_series_<timestamp>.csv` and the header is written.
#[derive(Resource, Default)]
pub struct TimeSeriesLogger {
    writer:         Option<BufWriter<File>>,
    last_log_secs:  f32,
    /// Set true once we've TRIED to open the writer; prevents
    /// repeated error-spam if file creation failed once.
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
            "logs/time_series_{}.csv",
            chrono::Local::now().format("%d-%m-%Y-%H-%M-%S"),
        ));
        if let Some(parent) = path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                error!("time-series: failed to create dir {}: {}", parent.display(), e);
                return;
            }
        }
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

    // Single GPU forward pass to populate per-slot telemetry, then
    // pure CPU indexing per organism.
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
    // Flush every tick so a crash doesn't lose the most recent
    // window. Cost is one fsync ~= a few ms; we're already running
    // at 5 s cadence so amortised cost is negligible.
    let _ = writer.flush();
}
