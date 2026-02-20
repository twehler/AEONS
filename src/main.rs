mod world_geometry;
mod bevy_flycam;
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
    // args[1] = input path
    if args.len() < 2 {
        println!("Usage: cargo run --bin voxel_es <world.bin>");
        return;
    }

    let map_path_input = Path::new(&args[1]);

    let mut app = App::new();

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
            map_path: map_path_input.to_string_lossy().into_owned(),
        })
        .add_plugins(bevy_flycam::PlayerPlugin)
        .add_plugins(viewport_settings::ViewportSettingsPlugin)

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


