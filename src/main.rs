mod world_geometry;
mod bevy_flycam;
mod viewport_settings;
mod colony;
mod cell;
mod movement;
mod organism_collision;
mod energy;
mod reproduction;
mod environment;
mod water;
mod predation;
mod growth;
mod behaviour;

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
    // args[1] = input path
    if args.len() < 2 {
        println!("Usage: cargo run --bin AEONS <file.terrain>");
        return;
    }

    let terrain_path_input = Path::new(&args[1]);


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
            // solution: use .to_string_lossy().into_owned() instead
            terrain_path: terrain_path_input.to_string_lossy().into_owned(),
            texture_path: "texture_atlas_8x8.png".to_string(),
        })
        .add_plugins(bevy_flycam::PlayerPlugin)
        .add_plugins(viewport_settings::ViewportSettingsPlugin)
        .add_plugins(movement::MovementPlugin::with_mode(movement_mode))
        .add_plugins(colony::ColonyPlugin)
        .add_plugins(energy::EnergyPlugin)
        //.add_plugins(reproduction::ReproductionPlugin)
        .add_plugins(water::WaterPlugin)
        .add_plugins(predation::PredationPlugin)
        .add_plugins(growth::GrowthPlugin)
        .add_plugins(behaviour::BehaviourPlugin)

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


