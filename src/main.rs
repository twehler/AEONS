#[path = "world/world_geometry.rs"]    mod world_geometry;
#[path = "world/environment.rs"]       mod environment;
#[path = "world/water.rs"]             mod water;

#[path = "colony/colony.rs"]           mod colony;
#[path = "colony/organism.rs"]         mod organism;
#[path = "colony/cell.rs"]             mod cell;
#[path = "colony/body_part.rs"]        mod body_part;
#[path = "colony/energy.rs"]           mod energy;
#[path = "colony/reproduction.rs"]     mod reproduction;
#[path = "colony/krishi.rs"]           mod krishi;
#[path = "colony/dataset_export.rs"]   mod dataset_export;
#[path = "colony/time_series_log.rs"]  mod time_series_log;

#[path = "growth/volumetric_growth/mod.rs"] mod volumetric_growth;
#[path = "growth/mutation.rs"]              mod mutation;
#[path = "growth/continuous_growth.rs"]     mod continuous_growth;

#[path = "behaviour/behaviour.rs"]                  mod behaviour;
#[path = "behaviour/world_model.rs"]                mod world_model;
#[path = "behaviour/rl_helpers.rs"]                 mod rl_helpers;
#[path = "behaviour/intelligence_level_0.rs"]       mod intelligence_level_0;
#[path = "behaviour/intelligence_level_1_photo.rs"] mod intelligence_level_1_photo;
#[path = "behaviour/intelligence_level_1_hetero.rs"] mod intelligence_level_1_hetero;
#[path = "behaviour/intelligence_level_2.rs"]       mod intelligence_level_2;
#[path = "behaviour/intelligence_level_3.rs"]       mod intelligence_level_3;
#[path = "behaviour/intelligence_level_herbivore_1.rs"] mod intelligence_level_herbivore_1;
#[path = "behaviour/predation.rs"]                  mod predation;
#[path = "behaviour/photosynthesis.rs"]             mod photosynthesis;
#[path = "behaviour/sensory.rs"]                    mod sensory;

#[path = "movement_physics/movement.rs"]           mod movement;
#[path = "movement_physics/organism_collision.rs"] mod organism_collision;

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

// Colony editor — alternate entry point, reuses `WorldPlugin` /
// `WaterPlugin` for terrain rendering but skips every simulation
// plugin. The flycam in `editor_camera` is distinct from
// `player_plugin`'s capture-and-hold camera. The editor's source
// now lives under `frontend/colony_editor/` since it doubles as the
// in-engine Edit-Colony window mode.
#[path = "frontend/colony_editor/camera.rs"]    mod camera;
#[path = "frontend/colony_editor/mod.rs"]       mod colony_editor;

// Species editor — manual organism construction with `.species` save
// output. Lives at world position `SPECIES_EDITOR_ORIGIN` so its
// visuals don't overlap the running simulation.
#[path = "frontend/species_editor/mod.rs"]      mod species_editor;

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
//
//   * NO positional path args   → launcher mode (eframe window). On Start
//     the launcher re-spawns this same executable as a subprocess with the
//     chosen paths and waits for it to exit.
//   * POSITIONAL path arg(s)    → simulation mode (Bevy + Burn).
//
// Why a subprocess? winit (which both eframe and bevy_winit use) hard-
// errors with `RecreationAttempt` if `EventLoop::new` is called twice in
// one process — there is no supported way to run an eframe event loop
// followed by a Bevy event loop sequentially. Forking the simulation
// into a fresh child avoids that singleton restriction; from the user's
// perspective they still launch one binary and the launcher still drives
// path selection. The original `cargo run` shell stays blocked on the
// parent until the simulation exits, mirroring normal Unix UX.

fn main() {
    let args: Vec<String> = env::args().collect();
    let show_wireframe = args.iter().any(|a| a == "--wireframe");
    let editor_flag    = args.iter().any(|a| a == "--editor");
    // `--trainingmode` — boots the simulation with AI-training mode
    // active (heterotroph despawn is suppressed at 0 energy). The
    // statistics-panel checkbox shows as checked from the first
    // frame because `update_ai_training_checkbox_mark` triggers on
    // the resource's just-inserted "changed" flag.
    let training_mode  = args.iter().any(|a| a == "--trainingmode");
    // `--map-size X Z` — parse before collecting positionals so the
    // two numeric values don't end up in the positional list.
    let map_size = parse_map_size(&args).unwrap_or(world_geometry::MapSize::default());
    let max_phototrophs        = parse_max_phototrophs(&args);
    let max_herbivores         = parse_max_herbivores(&args);
    let start_heterotrophs     = parse_start_heterotrophs(&args);
    let start_photoautotrophs  = parse_start_photoautotrophs(&args);
    let positional             = collect_positionals(&args);

    if positional.is_empty() && !editor_flag {
        // Launcher mode — show the eframe window with both action
        // buttons. The chosen mode is forwarded to a freshly-spawned
        // child process (subprocess re-spawn is necessary because
        // winit's EventLoop is a singleton).
        let Some(mode) = run_launcher() else { return; };
        match mode {
            LaunchMode::RunSimulation {
                map_path, colony_path, wireframe, map_x, map_z,
                max_phototrophs, max_herbivores,
                start_heterotrophs, start_photoautotrophs,
                training_mode: launcher_training_mode,
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
                    // The launcher's checkbox is the source of truth
                    // for the child: even if the parent was started
                    // with `--trainingmode`, an unchecked box here
                    // resets it to false (and vice-versa).
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
        let map_path = positional.first().cloned().unwrap_or_else(|| "assets/world.glb".into());
        run_editor(map_path, map_size);
    } else {
        // Simulation mode (re-spawned child or direct CLI invocation).
        let map_path    = positional[0].clone();
        let colony_path = positional.get(1).cloned();
        run_simulation(map_path, colony_path, show_wireframe, map_size,
                       max_phototrophs, max_herbivores,
                       start_heterotrophs, start_photoautotrophs,
                       training_mode);
    }
}


/// Parse `--map-size X Z` out of argv. Returns `None` if the flag is
/// absent or either value doesn't parse as a positive f32.
fn parse_map_size(args: &[String]) -> Option<world_geometry::MapSize> {
    let pos = args.iter().position(|a| a == "--map-size")?;
    let x = args.get(pos + 1)?.parse::<f32>().ok()?;
    let z = args.get(pos + 2)?.parse::<f32>().ok()?;
    if x <= 0.0 || z <= 0.0 { return None; }
    Some(world_geometry::MapSize { x, z })
}

/// Parse `--max-phototrophs N` out of argv. Returns `None` if the flag
/// is missing or unparseable; callers fall back to
/// `DEFAULT_MAX_PHOTOAUTOTROPHS` (via FrontendPlugin's `init_resource`).
fn parse_max_phototrophs(args: &[String]) -> Option<usize> {
    let pos = args.iter().position(|a| a == "--max-phototrophs")?;
    args.get(pos + 1)?.parse::<usize>().ok()
}

/// Parse `--max-herbivores N` out of argv. Returns `None` if the flag
/// is missing or unparseable; callers fall back to
/// `DEFAULT_MAX_HERBIVORES` (via `MaxHerbivores::default()`).
fn parse_max_herbivores(args: &[String]) -> Option<usize> {
    let pos = args.iter().position(|a| a == "--max-herbivores")?;
    args.get(pos + 1)?.parse::<usize>().ok()
}

/// Parse `--start-heteros N` out of argv. Returns `None` if the flag
/// is missing or unparseable; callers fall back to
/// `DEFAULT_START_HETEROTROPHS` via `StartHeterotrophs::default()`.
fn parse_start_heterotrophs(args: &[String]) -> Option<usize> {
    let pos = args.iter().position(|a| a == "--start-heteros")?;
    args.get(pos + 1)?.parse::<usize>().ok()
}

/// Parse `--start-photos N` out of argv. Returns `None` if the flag
/// is missing or unparseable; callers fall back to
/// `DEFAULT_START_PHOTOAUTOTROPHS` via `StartPhotoautotrophs::default()`.
fn parse_start_photoautotrophs(args: &[String]) -> Option<usize> {
    let pos = args.iter().position(|a| a == "--start-photos")?;
    args.get(pos + 1)?.parse::<usize>().ok()
}


/// Collect positional CLI arguments. Skips known `--flag` tokens AND
/// the two values that follow `--map-size` (which are numeric and
/// would otherwise be picked up as positionals).
fn collect_positionals(args: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let mut i = 1; // skip argv[0]
    while i < args.len() {
        let a = &args[i];
        if a == "--map-size" {
            i += 3; // skip flag + two values
            continue;
        }
        if a == "--max-phototrophs" {
            i += 2; // skip flag + value
            continue;
        }
        if a == "--max-herbivores" {
            i += 2; // skip flag + value
            continue;
        }
        if a == "--start-heteros" {
            i += 2; // skip flag + value
            continue;
        }
        if a == "--start-photos" {
            i += 2; // skip flag + value
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
    let movement_mode = movement::MovementMode::TwoD;

    let world_path_input = Path::new(&map_path);

    // `Time<Virtual>` defaults its `max_delta` to 250 ms — frames
    // that take longer have the excess silently discarded from
    // virtual time. That's Bevy's anti-spiral-of-death safety, but
    // it conflicts with AEONS's observational use of the simulation
    // clock: the A2C training step, CubeCL kernel compilation, and
    // dense predation passes all occasionally produce frames well
    // past 250 ms, and each such frame underspends virtual time by
    // `real_delta − 250 ms`. Over a 2-hour run this can swallow
    // tens of minutes from the displayed sim timer.
    //
    // Bump the cap to 60 s so realistic frame stutters don't clip
    // virtual-time accumulation. A genuine pathology that produces
    // a >60 s frame is also a genuine reason to want the clock
    // pinned anyway.
    let mut virtual_time = Time::<Virtual>::default();
    virtual_time.set_max_delta(std::time::Duration::from_secs(60));
    app.insert_resource(virtual_time);

    // Inject the optional colony-load path so ColonyPlugin's spawn system
    // can pick the load-vs-generate branch.
    app.insert_resource(colony::ColonyLoadPath(colony_path));
    // World extent — drives `compute_normalisation` plus every spawn-bound
    // / pos-normalisation site that previously read the MAP_MAX_X/Z consts.
    app.insert_resource(map_size);

    // Photoautotroph cap. The launcher's "Max Phototrophic Organisms"
    // field flows through here as `--max-phototrophs N`. Drives ONLY
    // the photo running-population cap; brain-pool sizing is
    // independent (see below). Falling through to `init_resource`
    // defaults (= DEFAULT_MAX_PHOTOAUTOTROPHS) when no flag is
    // supplied is fine — the resources are inserted BEFORE
    // `BehaviourPlugin` builds the brain pools.
    if let Some(n) = max_phototrophs {
        let n = n.max(1);
        app.insert_resource(simulation_settings::MaxPhotoautotrophs(n));
    }
    // Herbivore reproduction cap from the launcher's "Max Herbivores"
    // field (or `--max-herbivores N` argv). When absent, the default
    // from `MaxHerbivores::default()` (i.e. `DEFAULT_MAX_HERBIVORES`)
    // is used via the resource's `init_resource` registration.
    //
    // The GPU brain-pool size (`OrganismPoolSize`) is derived from
    // this value with generous headroom: heterotrophs are the only
    // brain-using organisms, so the pool only needs to cover them.
    // We multiply by 4× so the user can raise `MaxHerbivores` at
    // runtime through the statistics panel without running out of
    // brain slots — beyond 4× the brain pool will run dry and extra
    // heteros spawn brain-less until a slot frees up.
    if let Some(n) = max_herbivores {
        let n = n.max(1);
        app.insert_resource(simulation_settings::MaxHerbivores(n));
        app.insert_resource(simulation_settings::OrganismPoolSize((n * 4).max(16)));
    }
    // Initial-cohort herbivore count from the launcher's "Start
    // Heterotroph Number" field (or `--start-heteros N`). When
    // absent, defaults to `DEFAULT_START_HETEROTROPHS` via the
    // resource's `init_resource` registration.
    if let Some(n) = start_heterotrophs {
        app.insert_resource(simulation_settings::StartHeterotrophs(n));
    }
    // Initial-cohort photoautotroph count from the launcher's
    // "Spawn Phototrophic Organisms" field (or `--start-photos N`).
    // When absent, falls back to the resource's `init_resource`
    // default = `DEFAULT_START_PHOTOAUTOTROPHS`.
    if let Some(n) = start_photoautotrophs {
        app.insert_resource(simulation_settings::StartPhotoautotrophs(n));
    }

    // AI-training mode — inserted BEFORE `FrontendPlugin` adds its
    // `init_resource::<AiTrainingMode>()` so the idempotent init
    // sees our value and preserves it. With `training_mode = true`,
    // `energy::manage_energy` will suppress heterotroph despawns at
    // 0 energy, and `update_ai_training_checkbox_mark` will see the
    // resource as "changed" on the first frame and flip the
    // statistics-panel checkbox mark to visible.
    app.insert_resource(simulation_settings::AiTrainingMode(training_mode));

    app.add_plugins(DefaultPlugins.set(RenderPlugin {
        render_creation: WgpuSettings {
            features: WgpuFeatures::POLYGON_MODE_LINE,
            ..default()
            }.into(),
            ..default()
        }))
        .add_plugins(world_geometry::WorldPlugin{
            // std::path::Path doesn't have to_string() implemented because
            // input could be anything; use .to_string_lossy().into_owned()
            // instead. The glb is resolved relative to Bevy's asset root
            // (`assets/`); a leading `assets/` segment is stripped by the
            // plugin.
            world_path: world_path_input.to_string_lossy().into_owned(),
        })
        .add_plugins(player_plugin::PlayerPlugin)
        .add_plugins(frontend::FrontendPlugin)
        .add_plugins(movement::MovementPlugin::with_mode(movement_mode))
        .add_plugins(colony::ColonyPlugin)
        .add_plugins(energy::EnergyPlugin)
        .add_plugins(physiology::PhysiologyPlugin)
        .add_plugins(reproduction::ReproductionPlugin)
        .add_plugins(continuous_growth::ContinuousGrowthPlugin)
        .add_plugins(water::WaterPlugin)
        .add_plugins(predation::PredationPlugin)
        .add_plugins(behaviour::BehaviourPlugin)
        .add_plugins(krishi::KrishiPlugin)
        .add_plugins(lineages::LineagesPlugin)
        .add_plugins(species_editor::SpeciesEditorPlugin);

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
    // Light. Bevy's default `CascadeShadowConfig` uses 4 cascades and
    // re-extracts the full caster set per cascade per frame — at our
    // mesh-entity counts that's the dominant render-pipeline cost.
    // Override to a single cascade with the same far-plane as the camera
    // (300 units): one extract pass, one shadow draw, identical visual
    // for our flat-ish maps. `DirectionalLightShadowMap::size = 2048` is
    // also the Bevy default; halving it cheapens the shadow-pass GPU
    // workload without visible aliasing on our terrain scale.
    commands.insert_resource(bevy::light::DirectionalLightShadowMap { size: 1024 });
    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0,
            shadows_enabled: true,
            ..default()
        },
        // Render to both the default layer (0, simulation) AND the
        // species editor layer (1) so species cells get illuminated
        // when the camera is rendering only layer 1.
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
