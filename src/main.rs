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

#[path = "behaviour/behaviour.rs"]            mod behaviour;
#[path = "behaviour/intelligence_level_1.rs"] mod intelligence_level_1;
#[path = "behaviour/intelligence_level_3.rs"] mod intelligence_level_3;
#[path = "behaviour/predation.rs"]            mod predation;
#[path = "behaviour/photosynthesis.rs"]       mod photosynthesis;

#[path = "movement_physics/movement.rs"]           mod movement;
#[path = "movement_physics/organism_collision.rs"] mod organism_collision;

#[path = "physiology/physiology.rs"]   mod physiology;

mod player_plugin;

#[path = "frontend/frontend.rs"]                mod frontend;
#[path = "frontend/statistics_panel.rs"]        mod statistics_panel;
#[path = "frontend/simulation_settings.rs"]     mod simulation_settings;

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
use std::sync::mpsc;

use eframe::egui;


// ── Launcher layout ──────────────────────────────────────────────────────────
//
// Tweak these constants to reposition / resize the banner image, the path
// text fields, and the Start button without touching the rest of the file.

const LAUNCHER_WINDOW_SIZE:    egui::Vec2 = egui::vec2(800.0, 500.0);
const BANNER_PATH:             &str       = "logos/dna.png";
const BANNER_POS:              egui::Pos2 = egui::pos2(0.0, 0.0);
const BANNER_SIZE:             egui::Vec2 = egui::vec2(800.0, 200.0);
const TEXTFIELD_POS:           egui::Pos2 = egui::pos2(60.0, 230.0);
const TEXTFIELD_SIZE:          egui::Vec2 = egui::vec2(680.0, 32.0);
const TEXTFIELD_FONT_SIZE:     f32        = 16.0;
const COLONYFIELD_POS:         egui::Pos2 = egui::pos2(60.0, 295.0);
const COLONYFIELD_SIZE:        egui::Vec2 = egui::vec2(680.0, 32.0);
const BUTTON_POS:              egui::Pos2 = egui::pos2(220.0, 370.0);
const BUTTON_SIZE:             egui::Vec2 = egui::vec2(360.0, 90.0);
const BUTTON_FONT_SIZE:        f32        = 28.0;
const BUTTON_LABEL:            &str       = "START AEONS";
const DEFAULT_MAP_PATH:        &str       = "assets/world.glb";
const DEFAULT_COLONY_PATH:     &str       = "";


/// What the launcher hands back to `main` once the user clicks Start.
struct LauncherChoice {
    map_path:    String,
    colony_path: Option<String>,
    wireframe:   bool,
}


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
    let positional: Vec<String> = args.iter()
        .skip(1)
        .filter(|a| !a.starts_with("--"))
        .cloned()
        .collect();

    if positional.is_empty() {
        // Launcher mode.
        let Some(choice) = run_launcher() else {
            // User closed the launcher window without pressing Start.
            return;
        };
        respawn_as_simulation(choice);
    } else {
        // Simulation mode (either re-spawned by the launcher or invoked
        // directly from the command line for testing).
        let map_path    = positional[0].clone();
        let colony_path = positional.get(1).cloned();
        run_simulation(LauncherChoice {
            map_path,
            colony_path,
            wireframe: show_wireframe,
        });
    }
}


/// Re-launch the current executable in simulation mode, passing the
/// chosen paths as positional arguments. Blocks until the child exits.
fn respawn_as_simulation(choice: LauncherChoice) {
    let exe = match env::current_exe() {
        Ok(p)  => p,
        Err(e) => {
            eprintln!("Failed to locate own executable for re-launch: {e}");
            return;
        }
    };

    let mut args: Vec<String> = vec![choice.map_path];
    if let Some(c) = choice.colony_path { args.push(c); }
    if choice.wireframe { args.push("--wireframe".to_string()); }

    match Command::new(&exe).args(&args).spawn() {
        Ok(mut child) => { let _ = child.wait(); }
        Err(e) => eprintln!("Failed to spawn simulation child process: {e}"),
    }
}


// ── Launcher (egui) ──────────────────────────────────────────────────────────

fn run_launcher() -> Option<LauncherChoice> {
    let (tx, rx) = mpsc::channel::<LauncherChoice>();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("AEONS Launcher")
            .with_inner_size(LAUNCHER_WINDOW_SIZE)
            .with_resizable(false),
        ..Default::default()
    };

    let result = eframe::run_native(
        "AEONS Launcher",
        options,
        Box::new(move |cc| {
            Ok(Box::new(LauncherApp::new(cc, tx.clone())) as Box<dyn eframe::App>)
        }),
    );

    if result.is_err() { return None; }

    // The LauncherApp sends exactly once on Start. If the window was closed
    // without clicking Start the channel stays empty and we get None.
    rx.try_recv().ok()
}


struct LauncherApp {
    map_path:    String,
    colony_path: String,
    wireframe:   bool,
    banner:      Option<egui::TextureHandle>,
    status:      String,
    tx:          mpsc::Sender<LauncherChoice>,
}

impl LauncherApp {
    fn new(cc: &eframe::CreationContext<'_>, tx: mpsc::Sender<LauncherChoice>) -> Self {
        // Allow `--wireframe` on the command line to pre-tick the box.
        let wireframe = env::args().any(|a| a == "--wireframe");
        Self {
            map_path:    DEFAULT_MAP_PATH.to_string(),
            colony_path: DEFAULT_COLONY_PATH.to_string(),
            wireframe,
            banner:      load_banner(&cc.egui_ctx),
            status:      String::new(),
            tx,
        }
    }

    fn launch(&mut self, ctx: &egui::Context) {
        let map = self.map_path.trim();
        if map.is_empty() {
            self.status = "Please enter a path to a .glb world file.".into();
            return;
        }
        let colony = self.colony_path.trim();
        let choice = LauncherChoice {
            map_path:    map.to_string(),
            colony_path: if colony.is_empty() { None } else { Some(colony.to_string()) },
            wireframe:   self.wireframe,
        };
        let _ = self.tx.send(choice);
        // Closes the eframe viewport — `eframe::run_native` returns control
        // to `main`, which then proceeds to `run_simulation`.
        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
    }
}

fn load_banner(ctx: &egui::Context) -> Option<egui::TextureHandle> {
    let bytes = std::fs::read(BANNER_PATH).ok()?;
    let img = image::load_from_memory(&bytes).ok()?.to_rgba8();
    let size = [img.width() as usize, img.height() as usize];
    let pixels = img.into_raw();
    let color_image = egui::ColorImage::from_rgba_unmultiplied(size, &pixels);
    Some(ctx.load_texture("aeons_banner", color_image, egui::TextureOptions::LINEAR))
}

impl eframe::App for LauncherApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let panel_frame = egui::Frame::default()
            .inner_margin(egui::Margin::ZERO)
            .outer_margin(egui::Margin::ZERO);

        egui::CentralPanel::default()
            .frame(panel_frame)
            .show(ctx, |ui| {
                if let Some(tex) = &self.banner {
                    let rect = egui::Rect::from_min_size(BANNER_POS, BANNER_SIZE);
                    ui.painter().image(
                        tex.id(),
                        rect,
                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                        egui::Color32::WHITE,
                    );
                }

                let textfield_rect = egui::Rect::from_min_size(TEXTFIELD_POS, TEXTFIELD_SIZE);
                ui.put(
                    textfield_rect,
                    egui::TextEdit::singleline(&mut self.map_path)
                        .hint_text("Path to .glb world file")
                        .font(egui::FontId::proportional(TEXTFIELD_FONT_SIZE)),
                );

                let colonyfield_rect = egui::Rect::from_min_size(COLONYFIELD_POS, COLONYFIELD_SIZE);
                ui.put(
                    colonyfield_rect,
                    egui::TextEdit::singleline(&mut self.colony_path)
                        .hint_text("Optional: path to .colony save file (leave empty to start fresh)")
                        .font(egui::FontId::proportional(TEXTFIELD_FONT_SIZE)),
                );

                let button_rect = egui::Rect::from_min_size(BUTTON_POS, BUTTON_SIZE);
                let label = egui::RichText::new(BUTTON_LABEL).size(BUTTON_FONT_SIZE).strong();
                if ui.put(button_rect, egui::Button::new(label)).clicked() {
                    self.launch(ctx);
                }

                if !self.status.is_empty() {
                    let status_pos = egui::pos2(BUTTON_POS.x, BUTTON_POS.y + BUTTON_SIZE.y + 6.0);
                    let status_rect =
                        egui::Rect::from_min_size(status_pos, egui::vec2(BUTTON_SIZE.x, 24.0));
                    ui.put(status_rect, egui::Label::new(&self.status));
                }
            });
    }
}


// ── Simulation (Bevy + Burn) ─────────────────────────────────────────────────

fn run_simulation(choice: LauncherChoice) {
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

    let world_path_input = Path::new(&choice.map_path);

    // Inject the optional colony-load path so ColonyPlugin's spawn system
    // can pick the load-vs-generate branch.
    app.insert_resource(colony::ColonyLoadPath(choice.colony_path));

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

        //.add_plugins(EguiPlugin::default())
        //.add_plugins(WorldInspectorPlugin::new())
        .add_systems(Startup, setup);

    if choice.wireframe {
        app.add_plugins(WireframePlugin::default());
        app.insert_resource(WireframeConfig {
            global: true,
            default_color: Color::WHITE,
        });
        println!("----- RUNNING IN WIREFRAME MODE -----")
    }

    app.run();
}


fn setup(mut commands: Commands) {
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
