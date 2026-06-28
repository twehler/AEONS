#[path = "world/world_geometry.rs"]    mod world_geometry;
#[path = "world/world_format.rs"]      mod world_format;
#[path = "world/environment.rs"]       mod environment;
#[path = "world/water.rs"]             mod water;

#[path = "colony/colony.rs"]           mod colony;
#[path = "colony/colony_save_load.rs"] mod colony_save_load;
#[path = "colony/organism.rs"]         mod organism;
#[path = "colony/cell.rs"]             mod cell;
#[path = "colony/body_part.rs"]        mod body_part;
#[path = "colony/energy.rs"]           mod energy;
#[path = "colony/reproduction.rs"]     mod reproduction;
#[path = "colony/dataset_export.rs"]   mod dataset_export;
#[path = "colony/time_series_log.rs"]  mod time_series_log;
#[path = "colony/limb_time_series_log.rs"]  mod limb_time_series_log;
#[path = "colony/limb_force_probe.rs"]      mod limb_force_probe;

#[path = "growth/volumetric_growth/mod.rs"] mod volumetric_growth;
#[path = "growth/mutation.rs"]              mod mutation;
#[path = "growth/continuous_growth.rs"]     mod continuous_growth;

#[path = "behaviour/behaviour.rs"]                  mod behaviour;
#[path = "behaviour/world_model.rs"]                mod world_model;
#[path = "behaviour/rl_helpers.rs"]                 mod rl_helpers;
#[path = "behaviour/intelligence_level_0.rs"]       mod intelligence_level_0;
#[path = "behaviour/sliding_movement/sliding_reinforce.rs"]                  mod sliding_reinforce;
#[path = "behaviour/sliding_movement/intelligence_level_2_sliding.rs"]       mod intelligence_level_2_sliding;
#[path = "behaviour/sliding_movement/intelligence_level_3_sliding.rs"]       mod intelligence_level_3_sliding;
#[path = "behaviour/sliding_movement/intelligence_level_herbivore_1_sliding.rs"] mod intelligence_level_herbivore_1_sliding;
#[path = "behaviour/limb_based_movement/limb_ppo.rs"]                                mod limb_ppo;
#[path = "behaviour/limb_based_movement/intelligence_level_herbivore_1_limb.rs"]     mod intelligence_level_herbivore_1_limb;
#[path = "behaviour/limb_based_movement/intelligence_level_2_limb.rs"]               mod intelligence_level_2_limb;
#[path = "behaviour/limb_based_movement/intelligence_level_3_limb.rs"]               mod intelligence_level_3_limb;
#[path = "behaviour/swimming_movement/swim_ppo.rs"]                                  mod swim_ppo;
#[path = "behaviour/swimming_movement/intelligence_level_1_swimming.rs"]             mod intelligence_level_1_swimming;
#[path = "behaviour/predation.rs"]                  mod predation;
#[path = "behaviour/photosynthesis.rs"]             mod photosynthesis;
#[path = "behaviour/sensory.rs"]                    mod sensory;

#[path = "movement_physics/movement.rs"]           mod movement;
#[path = "movement_physics/organism_collision.rs"] mod organism_collision;
#[path = "movement_physics/rapier_setup.rs"]        mod rapier_setup;

#[path = "frame_profiler.rs"]          mod frame_profiler;

#[path = "physiology/physiology.rs"]   mod physiology;

#[path = "lineages/mod.rs"]            mod lineages;

mod player_plugin;

#[path = "frontend/frontend.rs"]                mod frontend;
#[path = "frontend/launcher.rs"]                mod launcher;
#[path = "frontend/statistics_panel.rs"]        mod statistics_panel;
mod simulation_settings;
#[path = "frontend/individuum_navigator.rs"]    mod individuum_navigator;
#[path = "frontend/species_navigator.rs"]       mod species_navigator;
#[path = "frontend/tree_view.rs"]               mod tree_view;
#[path = "frontend/ui_modal.rs"]                mod ui_modal;

// Colony editor — alternate entry point: reuses `WorldPlugin`/`WaterPlugin`
// for terrain, skips every simulation plugin. Doubles as the in-engine
// Edit-Colony window mode.
#[path = "frontend/colony_editor/camera.rs"]    mod camera;
#[path = "frontend/colony_editor/mod.rs"]       mod colony_editor;

// Species editor — manual organism construction with `.species` save output.
// Lives at `SPECIES_EDITOR_ORIGIN` so its visuals don't overlap the sim.
#[path = "frontend/species_editor/mod.rs"]      mod species_editor;

// Map editor — top-down terrain vertex-colour painting (visual-only, runtime-only).
#[path = "frontend/map_editor/mod.rs"]          mod map_editor;

use bevy::{
    prelude::*,
    pbr::wireframe::{WireframePlugin, WireframeConfig},
    render::{RenderPlugin, settings::{WgpuSettings, WgpuFeatures}},
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin}
};
use bevy_inspector_egui::{bevy_egui::EguiPlugin, prelude::*};
use bevy_inspector_egui::quick::WorldInspectorPlugin;

use std::env;
use std::path::Path;
use std::process::Command;

use launcher::{LaunchMode, run_launcher};


// ── Entry point ──────────────────────────────────────────────────────────────
//
// One binary, two roles, decided by argv:
//   * NO positional path args → launcher mode (eframe window); on Start it
//     re-spawns this executable as a subprocess and waits for it to exit.
//   * POSITIONAL path arg(s)  → simulation mode (Bevy + Burn).
//
// Subprocess is required: winit's `EventLoop::new` is a singleton and hard-
// errors if called twice in one process, so an eframe loop can't be followed
// by a Bevy loop. The parent `cargo run` shell stays blocked until exit.

fn main() {
    let args: Vec<String> = env::args().collect();
    let show_wireframe = args.iter().any(|a| a == "--wireframe");
    let editor_flag    = args.iter().any(|a| a == "--editor");
    // `--trainingmode` — boot with AI-training mode active (heterotroph
    // despawn suppressed at 0 energy).
    let training_mode  = args.iter().any(|a| a == "--trainingmode");
    // `--reload-limb-brains` — honour saved limb-brain weights even under the
    // STANDING task (default: standing fresh-inits, since legacy colonies hold
    // locomotion brains). Set this when loading a colony that was SAVED after a
    // standing run, so the trained standing policy is re-loaded and re-demonstrated
    // (the durable-success artifact — see colonies/*_standing.colony).
    let reload_limb_brains = args.iter().any(|a| a == "--reload-limb-brains");
    // `--map-size X Z` — parse before collecting positionals so the two
    // numeric values don't end up in the positional list.
    let map_size = parse_map_size(&args).unwrap_or(world_geometry::MapSize::default());
    let max_phototrophs        = parse_flag::<usize>(&args, "--max-phototrophs");
    let max_herbivores         = parse_flag::<usize>(&args, "--max-herbivores");
    let start_heterotrophs     = parse_flag::<usize>(&args, "--start-heteros");
    let start_photoautotrophs  = parse_flag::<usize>(&args, "--start-photos");
    // Headless / batch run-control flags (used for autonomous data runs):
    //   --time-speed X        initial TimeSpeed multiplier (virtual time)
    //   --exit-after-secs N    auto-exit after N seconds of VIRTUAL time
    let time_speed       = parse_flag::<f32>(&args, "--time-speed");
    let exit_after_secs  = parse_flag::<f32>(&args, "--exit-after-secs");
    // `--water-level Y` — world-space Y of the global water surface (seeds the
    // `WaterLevel` resource). `--adjust-colony-dimensions` — when loading a
    // colony, override its saved dimensions with the launcher values.
    let water_level      = parse_flag::<f32>(&args, "--water-level");
    let adjust_colony_dimensions = args.iter().any(|a| a == "--adjust-colony-dimensions");
    let positional             = collect_positionals(&args);

    if positional.is_empty() && !editor_flag {
        // Launcher mode — chosen mode is forwarded to a freshly-spawned
        // child (winit EventLoop is a singleton, so re-spawn is required).
        let Some(mode) = run_launcher() else { return; };
        match mode {
            LaunchMode::RunSimulation {
                map_path, colony_path, wireframe, map_x, map_z,
                max_phototrophs, max_herbivores,
                start_heterotrophs, start_photoautotrophs,
                training_mode: launcher_training_mode,
                water_level, adjust_colony_dimensions,
            } => {
                respawn(&[
                    Some(map_path),
                    colony_path,
                    if wireframe { Some("--wireframe".into()) } else { None },
                    Some("--map-size".into()),
                    Some(map_x.to_string()),
                    Some(map_z.to_string()),
                    Some("--max-phototrophs".into()),
                    Some(max_phototrophs.to_string()),
                    Some("--max-herbivores".into()),
                    Some(max_herbivores.to_string()),
                    Some("--start-heteros".into()),
                    Some(start_heterotrophs.to_string()),
                    Some("--start-photos".into()),
                    Some(start_photoautotrophs.to_string()),
                    // World-space water-surface Y for the `WaterLevel` resource.
                    Some("--water-level".into()),
                    Some(water_level.to_string()),
                    // Override a loaded colony's saved dimensions with the
                    // launcher values (currently the water level).
                    if adjust_colony_dimensions { Some("--adjust-colony-dimensions".into()) } else { None },
                    // Launcher's checkbox is the source of truth for the
                    // child, overriding the parent's `--trainingmode`.
                    if launcher_training_mode { Some("--trainingmode".into()) } else { None },
                ]);
            }
            LaunchMode::RunEditor { map_path, map_x, map_z } => {
                respawn(&[
                    Some(map_path),
                    Some("--editor".into()),
                    Some("--map-size".into()),
                    Some(map_x.to_string()),
                    Some(map_z.to_string()),
                ]);
            }
        }
    } else if editor_flag {
        // Editor mode (re-spawned child or direct CLI invocation).
        let map_path = positional.first().cloned().unwrap_or_else(|| "assets/waterworld1.glb".into());
        run_editor(map_path, map_size);
    } else {
        // Simulation mode (re-spawned child or direct CLI invocation).
        let map_path    = positional[0].clone();
        let colony_path = positional.get(1).cloned();
        run_simulation(map_path, colony_path, show_wireframe, map_size,
                       max_phototrophs, max_herbivores,
                       start_heterotrophs, start_photoautotrophs,
                       training_mode, time_speed, exit_after_secs, reload_limb_brains,
                       water_level, adjust_colony_dimensions);
    }
}


/// Threshold (seconds of VIRTUAL time) at which `exit_after_virtual_secs`
/// quits the app. Present only when `--exit-after-secs` is passed.
#[derive(bevy::prelude::Resource)]
struct ExitAfterVirtualSecs(f32);

/// Auto-exit once the simulation's virtual clock passes the configured
/// threshold — lets unattended data-collection runs terminate cleanly
/// after capturing the milestone exports + time-series, without a manual
/// kill. Uses `Time<Virtual>` so it's measured in sim time, not wall time.
fn exit_after_virtual_secs(
    vtime: bevy::prelude::Res<bevy::prelude::Time<bevy::prelude::Virtual>>,
    limit: bevy::prelude::Res<ExitAfterVirtualSecs>,
    mut save_requested: bevy::prelude::ResMut<colony::SaveRequested>,
    mut exit: bevy::prelude::MessageWriter<bevy::app::AppExit>,
    mut save_stage: bevy::prelude::Local<u8>,
) {
    if vtime.elapsed_secs() < limit.0 { return; }
    // SAVE-ON-EXIT: persist the final state (incl. trained limb-brain weights) to
    // `autosaves/exit_<time>.colony` BEFORE quitting, so unattended training runs
    // leave a reloadable artifact (reload with `--reload-limb-brains`). Stage 0:
    // request the save. Stage 1+: wait until `save_colony_system` has consumed the
    // request (flushed to disk), then exit.
    match *save_stage {
        0 => {
            let now = chrono::Local::now();
            let path = std::path::Path::new("autosaves")
                .join(format!("exit_{}.colony", now.format("%d-%m-%Y-%H-%M-%S")));
            bevy::log::info!("exit-after-secs: virtual time {:.1}s ≥ {:.1}s — saving {} before quit",
                vtime.elapsed_secs(), limit.0, path.display());
            save_requested.0 = Some(path);
            *save_stage = 1;
        }
        _ => {
            // Give the writer at least one tick to flush, then exit.
            if save_requested.0.is_none() {
                bevy::log::info!("exit-after-secs: save flushed — quitting");
                exit.write(bevy::app::AppExit::Success);
            }
        }
    }
}

/// Parse a `--flag VALUE` f32 out of argv (generic). Used by the
/// autonomous-run controls `--time-speed` / `--exit-after-secs`.
/// Parse `--flag VALUE` out of argv: find `flag`, then parse the single
/// token that follows it as `T`. Returns `None` if the flag is absent,
/// has no following token, or that token doesn't parse as `T` — so every
/// caller's "missing or invalid ⇒ fall back to default" semantics holds.
fn parse_flag<T: std::str::FromStr>(args: &[String], flag: &str) -> Option<T> {
    let pos = args.iter().position(|a| a == flag)?;
    args.get(pos + 1)?.parse::<T>().ok()
}

/// Parse `--map-size X Z` out of argv. Returns `None` if the flag is
/// absent or either value doesn't parse as a positive f32. (Two-value
/// flag, so it can't use the single-value `parse_flag` helper.)
fn parse_map_size(args: &[String]) -> Option<world_geometry::MapSize> {
    let pos = args.iter().position(|a| a == "--map-size")?;
    let x = args.get(pos + 1)?.parse::<f32>().ok()?;
    let z = args.get(pos + 2)?.parse::<f32>().ok()?;
    if x <= 0.0 || z <= 0.0 { return None; }
    Some(world_geometry::MapSize { x, z })
}


/// Collect positional CLI arguments. Skips known `--flag` tokens AND
/// the two values that follow `--map-size` (which are numeric and
/// would otherwise be picked up as positionals).
fn collect_positionals(args: &[String]) -> Vec<String> {
    // Value-bearing flags → how many tokens *after* the flag are its values
    // (and so must be skipped along with the flag). Boolean flags
    // (--wireframe/--editor/--trainingmode/…) take 0 values and don't need an
    // entry — they're caught by the generic `--`-prefix skip below. Adding a
    // new value-bearing flag means adding one row here and nowhere else.
    const VALUE_FLAGS: &[(&str, usize)] = &[
        ("--map-size",        2),
        ("--max-phototrophs", 1),
        ("--max-herbivores",  1),
        ("--start-heteros",   1),
        ("--start-photos",    1),
        ("--water-level",     1),
    ];

    let mut out = Vec::new();
    let mut i = 1; // skip argv[0]
    while i < args.len() {
        let a = &args[i];
        if let Some((_, n_values)) = VALUE_FLAGS.iter().find(|(flag, _)| a == flag) {
            i += 1 + n_values; // skip flag + its value tokens
            continue;
        }
        if a.starts_with("--") {
            i += 1;
            continue;
        }
        out.push(a.clone());
        i += 1;
    }
    out
}


/// Re-spawn the current executable as a child process with the given
/// argv (after argv[0]). `None` entries are skipped so call sites can
/// emit conditional flags compactly. Blocks until the child exits so
/// the user's `cargo run` shell stays attached.
fn respawn(parts: &[Option<String>]) {
    let exe = match env::current_exe() {
        Ok(p)  => p,
        Err(e) => {
            eprintln!("Failed to locate own executable for re-launch: {e}");
            return;
        }
    };
    let args: Vec<String> = parts.iter().filter_map(|p| p.clone()).collect();
    match Command::new(&exe).args(&args).spawn() {
        Ok(mut child) => { let _ = child.wait(); }
        Err(e) => eprintln!("Failed to spawn child process: {e}"),
    }
}


// ── Simulation (Bevy + Burn) ─────────────────────────────────────────────────

fn run_simulation(
    map_path:               String,
    colony_path:            Option<String>,
    show_wireframe:         bool,
    map_size:               world_geometry::MapSize,
    max_phototrophs:        Option<usize>,
    max_herbivores:         Option<usize>,
    start_heterotrophs:     Option<usize>,
    start_photoautotrophs:  Option<usize>,
    training_mode:          bool,
    time_speed:             Option<f32>,
    exit_after_secs:        Option<f32>,
    reload_limb_brains:     bool,
    water_level:            Option<f32>,
    adjust_colony_dimensions: bool,
) {
    if let Ok(mut cache_path) = std::env::current_dir() {
        cache_path.push("caches");
        cache_path.push("cubecl");
        // Ensure the folders actually exist on the hard drive.
        std::fs::create_dir_all(&cache_path).ok();
        unsafe {
            std::env::set_var("CUBECL_CACHE_DIR", cache_path.to_string_lossy().as_ref());
        }
    }

    let mut app = App::new();

    let world_path_input = Path::new(&map_path);

    // `Time<Virtual>` keeps Bevy's default 250 ms `max_delta` — the anti-
    // spiral-of-death cap. The fixed-update accumulator grows by the (capped)
    // virtual delta each frame and FixedUpdate drains it at 64 Hz, so an
    // UNCAPPED delta lets a single slow frame (CUDA kernel compile, heavy
    // multibody spawn) inject many seconds of catch-up → hundreds–thousands of
    // physics+brain steps in one frame → a self-sustaining freeze at ~0 FPS
    // with the machine mostly idle (latency/single-thread-bound). The cap
    // bounds catch-up to ≤16 steps/frame; under genuine slowness the sim now
    // degrades gracefully (virtual time falls behind wall time) instead of
    // dying. WALL-clock elapsed for exports/saves comes from `RunElapsed`
    // (accumulated from `Time<Real>`, uncapped), NOT this capped clock.
    let virtual_time = Time::<Virtual>::default();
    app.insert_resource(virtual_time);

    // Inject the optional colony-load path so ColonyPlugin's spawn system
    // can pick the load-vs-generate branch.
    app.insert_resource(colony::ColonyLoadPath(colony_path));
    // World extent — drives `compute_normalisation` plus every spawn-bound
    // / pos-normalisation site that previously read the MAP_MAX_X/Z consts.
    app.insert_resource(map_size);

    // Global water surface (`--water-level Y`; absent ⇒ `DEFAULT_WATER_LEVEL`).
    // A loaded `.colony` overrides this with its saved level unless
    // `--adjust-colony-dimensions` was passed.
    app.insert_resource(environment::WaterLevel(
        water_level.unwrap_or(environment::DEFAULT_WATER_LEVEL),
    ));
    app.insert_resource(simulation_settings::AdjustColonyDimensions(adjust_colony_dimensions));

    // Photoautotroph running-population cap (`--max-phototrophs N`). Brain-
    // pool sizing is independent (below). Inserted BEFORE `BehaviourPlugin`
    // builds the pools; absent ⇒ `init_resource` default.
    if let Some(n) = max_phototrophs {
        let n = n.max(1);
        app.insert_resource(simulation_settings::MaxPhotoautotrophs(n));
    }
    // Herbivore reproduction cap (`--max-herbivores N`; absent ⇒ default).
    // The GPU brain pool (`OrganismPoolSize`) is sized 4× this — only
    // heterotrophs use brains, and 4× headroom lets the user raise the cap
    // at runtime before the pool runs dry (extra heteros spawn brain-less
    // until a slot frees).
    if let Some(n) = max_herbivores {
        let n = n.max(1);
        app.insert_resource(simulation_settings::MaxHerbivores(n));
        app.insert_resource(simulation_settings::OrganismPoolSize((n * 4).max(16)));
    }
    // Initial-cohort herbivore count (`--start-heteros N`; absent ⇒ default).
    if let Some(n) = start_heterotrophs {
        app.insert_resource(simulation_settings::StartHeterotrophs(n));
    }
    // Initial-cohort photoautotroph count (`--start-photos N`; absent ⇒ default).
    if let Some(n) = start_photoautotrophs {
        app.insert_resource(simulation_settings::StartPhotoautotrophs(n));
    }

    // AI-training mode — inserted BEFORE `FrontendPlugin`'s idempotent
    // `init_resource::<AiTrainingMode>()` so our value is preserved.
    app.insert_resource(simulation_settings::AiTrainingMode(training_mode));
    app.insert_resource(simulation_settings::ReloadLimbBrains(reload_limb_brains));

    // `--time-speed X` — initial virtual-time multiplier. Inserted before
    // `FrontendPlugin`'s idempotent `init_resource::<TimeSpeed>()` so it's preserved.
    if let Some(ts) = time_speed {
        app.insert_resource(simulation_settings::TimeSpeed(ts.max(0.01)));
    }
    // `--exit-after-secs N` — auto-quit after N seconds of VIRTUAL time
    // (for unattended data-collection runs). The system is only added when
    // the flag is present, so normal interactive runs never exit early.
    if let Some(secs) = exit_after_secs {
        app.insert_resource(ExitAfterVirtualSecs(secs));
        app.add_systems(Update, exit_after_virtual_secs);
    }

    app.add_plugins(DefaultPlugins.set(RenderPlugin {
        render_creation: WgpuSettings {
            features: WgpuFeatures::POLYGON_MODE_LINE,
            ..default()
            }.into(),
            ..default()
        }))
        .add_plugins(rapier_setup::RapierSetupPlugin)
        .add_plugins(frame_profiler::FrameProfilerPlugin)
        .add_plugins(world_geometry::WorldPlugin{
            // glb resolves relative to Bevy's asset root (`assets/`); a
            // leading `assets/` segment is stripped by the plugin.
            world_path: world_path_input.to_string_lossy().into_owned(),
        })
        .add_plugins(player_plugin::PlayerPlugin)
        .add_plugins(frontend::FrontendPlugin)
        .add_plugins(movement::MovementPlugin)
        .add_plugins(colony::ColonyPlugin)
        .add_plugins(energy::EnergyPlugin)
        .add_plugins(physiology::PhysiologyPlugin)
        .add_plugins(reproduction::ReproductionPlugin)
        .add_plugins(continuous_growth::ContinuousGrowthPlugin)
        .add_plugins(water::WaterPlugin)
        .add_plugins(predation::PredationPlugin)
        .add_plugins(behaviour::BehaviourPlugin)
        .add_plugins(lineages::LineagesPlugin)
        .add_plugins(species_editor::SpeciesEditorPlugin)
        .add_plugins(map_editor::MapEditorPlugin);

    //app.add_plugins(EguiPlugin::default());
    //app.add_plugins(WorldInspectorPlugin::new());
    app.add_systems(Startup, setup);

    if show_wireframe {
        app.add_plugins(WireframePlugin::default());
        app.insert_resource(WireframeConfig {
            global: true,
            default_color: Color::WHITE,
        });
        println!("----- RUNNING IN WIREFRAME MODE -----")
    }

    app.run();
}


// ── Colony editor (Bevy, no simulation) ──────────────────────────────────────

fn run_editor(map_path: String, map_size: world_geometry::MapSize) {
    let world_path_input = Path::new(&map_path);

    let mut app = App::new();
    app.insert_resource(map_size);
    // Editor's `water.rs` systems + the editor colony writer need these.
    app.insert_resource(environment::WaterLevel::default());
    app.insert_resource(simulation_settings::AdjustColonyDimensions(false));
    app.add_plugins(DefaultPlugins.set(RenderPlugin {
        render_creation: WgpuSettings {
            features: WgpuFeatures::POLYGON_MODE_LINE,
            ..default()
        }.into(),
        ..default()
    }))
    // World mesh + heightmap. Same plugin the simulation uses, so
    // any .glb that loads in AEONS loads in the editor.
    .add_plugins(world_geometry::WorldPlugin {
        world_path: world_path_input.to_string_lossy().into_owned(),
    })
    .add_plugins(water::WaterPlugin)
    .add_plugins(colony_editor::ColonyEditorPlugin);

    app.run();
}


fn setup(mut commands: Commands) {
    // Light. Bevy's default 4-cascade shadow config re-extracts the full
    // caster set per cascade per frame — the dominant render cost at our
    // entity counts. Single cascade + halved shadow-map size cuts that with
    // no visible difference on our flat-ish terrain.
    commands.insert_resource(bevy::light::DirectionalLightShadowMap { size: 1024 });
    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0,
            shadows_enabled: true,
            ..default()
        },
        // Render to layer 0 (simulation) AND layer 1 (species editor) so
        // species cells are lit when the camera renders only layer 1.
        bevy::camera::visibility::RenderLayers::from_layers(&[0, species_editor::SPECIES_EDITOR_LAYER]),
        bevy::light::CascadeShadowConfigBuilder {
            num_cascades:             1,
            minimum_distance:         0.1,
            maximum_distance:         200.0,
            first_cascade_far_bound:  200.0,
            overlap_proportion:       0.2,
        }.build(),
        Transform::from_rotation(Quat::from_euler(
            EulerRot::XYZ,
            -std::f32::consts::FRAC_PI_4,
            std::f32::consts::FRAC_PI_4,
            0.0,
        )),
    ));
}
