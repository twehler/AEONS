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
#[path = "behaviour/predation.rs"]                  mod predation;
#[path = "behaviour/photosynthesis.rs"]             mod photosynthesis;

#[path = "movement_physics/movement.rs"]           mod movement;
#[path = "movement_physics/organism_collision.rs"] mod organism_collision;

#[path = "physiology/physiology.rs"]   mod physiology;

mod player_plugin;

#[path = "frontend/frontend.rs"]                mod frontend;
#[path = "frontend/launcher.rs"]                mod launcher;
#[path = "frontend/statistics_panel.rs"]        mod statistics_panel;
#[path = "frontend/simulation_settings.rs"]     mod simulation_settings;
#[path = "frontend/individuum_navigator.rs"]    mod individuum_navigator;

// Colony editor — alternate entry point, reuses `WorldPlugin` /
// `WaterPlugin` for terrain rendering but skips every simulation
// plugin. The flycam in `editor_camera` is distinct from
// `player_plugin`'s capture-and-hold camera.
#[path = "colony_editor/camera.rs"]             mod camera;
#[path = "colony_editor/mod.rs"]                mod colony_editor;

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
    let positional: Vec<String> = args.iter()
        .skip(1)
        .filter(|a| !a.starts_with("--"))
        .cloned()
        .collect();

    if positional.is_empty() && !editor_flag {
        // Launcher mode — show the eframe window with both action
        // buttons. The chosen mode is forwarded to a freshly-spawned
        // child process (subprocess re-spawn is necessary because
        // winit's EventLoop is a singleton).
        let Some(mode) = run_launcher() else { return; };
        match mode {
            LaunchMode::RunSimulation { map_path, colony_path, wireframe } => {
                respawn(&[
                    Some(map_path),
                    colony_path,
                    if wireframe { Some("--wireframe".into()) } else { None },
                ]);
            }
            LaunchMode::RunEditor { map_path } => {
                respawn(&[
                    Some(map_path),
                    Some("--editor".into()),
                ]);
            }
        }
    } else if editor_flag {
        // Editor mode (re-spawned child or direct CLI invocation).
        let map_path = positional.first().cloned().unwrap_or_else(|| "assets/world.glb".into());
        run_editor(map_path);
    } else {
        // Simulation mode (re-spawned child or direct CLI invocation).
        let map_path    = positional[0].clone();
        let colony_path = positional.get(1).cloned();
        run_simulation(map_path, colony_path, show_wireframe);
    }
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

fn run_simulation(map_path: String, colony_path: Option<String>, show_wireframe: bool) {
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

    // Inject the optional colony-load path so ColonyPlugin's spawn system
    // can pick the load-vs-generate branch.
    app.insert_resource(colony::ColonyLoadPath(colony_path));

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
        .add_plugins(behaviour::BehaviourPlugin);

    // Krishi is disabled in heterotroph-movement RL debug mode so the
    // training environment doesn't have a second predator class polluting
    // the experiment.
    if simulation_settings::HETEROTROPH_MOVEMENT_AI_DEBUGGING {
        app.add_plugins(krishi::KrishiPlugin);
    }

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

fn run_editor(map_path: String) {
    let world_path_input = Path::new(&map_path);

    let mut app = App::new();
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
