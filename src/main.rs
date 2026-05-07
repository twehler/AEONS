#[path = "world/world_geometry.rs"]    mod world_geometry;
#[path = "world/environment.rs"]       mod environment;
#[path = "world/water.rs"]             mod water;

#[path = "colony/colony.rs"]           mod colony;
#[path = "colony/cell.rs"]             mod cell;
#[path = "colony/energy.rs"]           mod energy;
#[path = "colony/reproduction.rs"]     mod reproduction;
#[path = "colony/krishi.rs"]           mod krishi;

#[path = "growth/volumetric_growth/mod.rs"] mod volumetric_growth;
#[path = "growth/mutation.rs"]          mod mutation;

#[path = "behaviour/behaviour.rs"]            mod behaviour;
#[path = "behaviour/intelligence_level_1.rs"] mod intelligence_level_1;
#[path = "behaviour/intelligence_level_3.rs"] mod intelligence_level_3;
#[path = "behaviour/predation.rs"]            mod predation;
#[path = "behaviour/photosynthesis.rs"]       mod photosynthesis;

#[path = "movement_physics/movement.rs"]           mod movement;
#[path = "movement_physics/organism_collision.rs"] mod organism_collision;

mod player_plugin;
mod viewport_settings;

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





fn main() {

    let args: Vec<String> = env::args().collect();
    let show_wireframe = args.iter().any(|arg| arg == "--wireframe");

    // args[0] is always the program name
    // args[1] = input path (.glb under assets/)
    if args.len() < 2 {
        println!("Usage: cargo run --bin AEONS <world.glb>");
        return;
    }

    let world_path_input = Path::new(&args[1]);


    if let Ok(mut cache_path) = std::env::current_dir() {
        cache_path.push("caches");
        cache_path.push("cubecl");
        
        // Ensure the folders actually exist on the hard drive
        std::fs::create_dir_all(&cache_path).ok();

        unsafe {
            std::env::set_var("CUBECL_CACHE_DIR", cache_path.to_string_lossy().as_ref());
        }
    }

    let mut app = App::new();

    let movement_mode = movement::MovementMode::TwoD;

    app.add_plugins(DefaultPlugins.set(RenderPlugin {
        render_creation: WgpuSettings {
            features: WgpuFeatures::POLYGON_MODE_LINE,
            ..default()
            }.into(),
            ..default()
        }))
        .add_plugins(world_geometry::WorldPlugin{
            // std::path::Path doesn't have to_string() implemented because input could be anything
            // solution: use .to_string_lossy().into_owned() instead.
            // The glb is resolved relative to Bevy's asset root (`assets/`);
            // a leading `assets/` segment is stripped by the plugin.
            world_path: world_path_input.to_string_lossy().into_owned(),
        })
        .add_plugins(player_plugin::PlayerPlugin)
        .add_plugins(viewport_settings::ViewportSettingsPlugin)
        .add_plugins(movement::MovementPlugin::with_mode(movement_mode))
        .add_plugins(colony::ColonyPlugin)
        .add_plugins(energy::EnergyPlugin)
        .add_plugins(reproduction::ReproductionPlugin)
        .add_plugins(water::WaterPlugin)
        .add_plugins(predation::PredationPlugin)
        .add_plugins(behaviour::BehaviourPlugin)
        .add_plugins(krishi::KrishiPlugin)

        //.add_plugins(EguiPlugin::default())
        //.add_plugins(WorldInspectorPlugin::new())
        .add_systems(Startup, setup);

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

fn setup (mut commands: Commands) {

    // Light
    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(
            EulerRot::XYZ,
            -std::f32::consts::FRAC_PI_4,
            std::f32::consts::FRAC_PI_4,
            0.0,
        )),
    ));
}


