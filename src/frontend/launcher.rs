// Combined launcher (eframe / egui).
//
// Returns a `LaunchMode` that `main.rs` consumes to re-spawn the binary with
// the chosen argv. Subprocess re-spawn is required because winit's `EventLoop`
// is a singleton — eframe and Bevy windows can't coexist in one process.

use std::env;
use std::sync::mpsc;

use eframe::egui;

use crate::simulation_settings::{
    DEFAULT_MAX_PHOTOAUTOTROPHS, DEFAULT_MAX_HERBIVORES,
    DEFAULT_START_HETEROTROPHS, DEFAULT_START_PHOTOAUTOTROPHS,
    DEFAULT_MAP_X, DEFAULT_MAP_Z,
};
use crate::environment::DEFAULT_WATER_LEVEL;

/// Soft upper bound for the launcher's population-cap DragValue fields.
const LAUNCHER_POPULATION_CAP: usize = 100_000;


// ── Layout constants ─────────────────────────────────────────────────────────

const LAUNCHER_WINDOW_SIZE:    egui::Vec2 = egui::vec2(800.0, 720.0);
const BANNER_PATH:             &str       = "logos/dna.png";
const BANNER_POS:              egui::Pos2 = egui::pos2(0.0, 0.0);
const BANNER_SIZE:             egui::Vec2 = egui::vec2(800.0, 200.0);

const TEXTFIELD_POS:           egui::Pos2 = egui::pos2(60.0, 230.0);
const TEXTFIELD_SIZE:          egui::Vec2 = egui::vec2(580.0, 32.0);
const TEXTFIELD_FONT_SIZE:     f32        = 16.0;
const COLONYFIELD_POS:         egui::Pos2 = egui::pos2(60.0, 295.0);
const COLONYFIELD_SIZE:        egui::Vec2 = egui::vec2(580.0, 32.0);
const OPEN_BTN_SIZE:           egui::Vec2 = egui::vec2(90.0, 32.0);
const OPEN_MAP_BTN_POS:        egui::Pos2 = egui::pos2(650.0, 230.0);
const OPEN_COLONY_BTN_POS:     egui::Pos2 = egui::pos2(650.0, 295.0);

// AI-training-mode checkbox. NOT part of the colony-dimension gate — stays
// editable even with a colony loaded (orthogonal to spawn-count overrides).
// Seeded from `--trainingmode` argv (see `LauncherApp::new`).
const TRAININGMODE_ROW_TOP:    f32        = 332.0;
const TRAININGMODE_POS:        egui::Pos2 = egui::pos2(60.0, TRAININGMODE_ROW_TOP);
const TRAININGMODE_SIZE:       egui::Vec2 = egui::vec2(400.0, 24.0);

// Map-size row: two `DragValue` widgets (X / Z) plus a heading label.
const MAPSIZE_ROW_TOP:         f32        = 372.0;
const MAPSIZE_LABEL_POS:       egui::Pos2 = egui::pos2(60.0, MAPSIZE_ROW_TOP);
const MAPSIZE_LABEL_SIZE:      egui::Vec2 = egui::vec2(130.0, 32.0);
const MAPSIZE_X_LABEL_POS:     egui::Pos2 = egui::pos2(200.0, MAPSIZE_ROW_TOP);
const MAPSIZE_X_DRAG_POS:      egui::Pos2 = egui::pos2(225.0, MAPSIZE_ROW_TOP);
const MAPSIZE_Z_LABEL_POS:     egui::Pos2 = egui::pos2(345.0, MAPSIZE_ROW_TOP);
const MAPSIZE_Z_DRAG_POS:      egui::Pos2 = egui::pos2(370.0, MAPSIZE_ROW_TOP);
const MAPSIZE_FIELD_SIZE:      egui::Vec2 = egui::vec2(110.0, 32.0);
const MAPSIZE_MIN:             f32        = 50.0;
const MAPSIZE_MAX:             f32        = 8192.0;

// Water-level field — sits beside the map-size fields on the same row.
// Unlike the always-editable map-size fields, it follows the spawn-control
// enabled flag (greyed when a colony is loaded without "Adjust colony
// dimensions"). Forwarded via `--water-level Y` into the `WaterLevel` resource.
const WATERLEVEL_LABEL_POS:    egui::Pos2 = egui::pos2(500.0, MAPSIZE_ROW_TOP);
const WATERLEVEL_LABEL_SIZE:   egui::Vec2 = egui::vec2(110.0, 32.0);
const WATERLEVEL_DRAG_POS:     egui::Pos2 = egui::pos2(615.0, MAPSIZE_ROW_TOP);
const WATERLEVEL_FIELD_SIZE:   egui::Vec2 = egui::vec2(110.0, 32.0);
const WATERLEVEL_MIN:          f32        = -8192.0;
const WATERLEVEL_MAX:          f32        = 8192.0;

// Adjust-colony-dimensions checkbox. Gates the downstream spawn-control widgets
// when a colony file is loaded (contract documented on the `LauncherApp` field).
const ADJUSTCOL_ROW_TOP:       f32        = 407.0;
const ADJUSTCOL_POS:           egui::Pos2 = egui::pos2(60.0, ADJUSTCOL_ROW_TOP);
const ADJUSTCOL_SIZE:          egui::Vec2 = egui::vec2(400.0, 24.0);

// Max-Phototrophic-Organisms row. Wide label box to fit the long label.
const MAXPHOTO_ROW_TOP:          f32        = 442.0;
const MAXPHOTO_LABEL_POS:        egui::Pos2 = egui::pos2(60.0, MAXPHOTO_ROW_TOP);
const MAXPHOTO_LABEL_SIZE:       egui::Vec2 = egui::vec2(220.0, 32.0);
const MAXPHOTO_DRAG_POS:         egui::Pos2 = egui::pos2(285.0, MAXPHOTO_ROW_TOP);
const MAXPHOTO_FIELD_SIZE:       egui::Vec2 = egui::vec2(110.0, 32.0);

// Spawn-phototrophic-organisms row. Initial-cohort photo count at
// `spawn_colony`; independent from `MaxPhotoautotrophs` (the running cap).
const SPAWNPHOTO_ROW_TOP:        f32        = 477.0;
const SPAWNPHOTO_LABEL_POS:      egui::Pos2 = egui::pos2(60.0, SPAWNPHOTO_ROW_TOP);
const SPAWNPHOTO_LABEL_SIZE:     egui::Vec2 = egui::vec2(265.0, 32.0);
const SPAWNPHOTO_DRAG_POS:       egui::Pos2 = egui::pos2(330.0, SPAWNPHOTO_ROW_TOP);
const SPAWNPHOTO_FIELD_SIZE:     egui::Vec2 = egui::vec2(110.0, 32.0);

// Max-herbivores row. Seeds the `MaxHerbivores` reproduction cap.
const MAXHERB_ROW_TOP:         f32        = 512.0;
const MAXHERB_LABEL_POS:       egui::Pos2 = egui::pos2(60.0, MAXHERB_ROW_TOP);
const MAXHERB_LABEL_SIZE:      egui::Vec2 = egui::vec2(180.0, 32.0);
const MAXHERB_DRAG_POS:        egui::Pos2 = egui::pos2(245.0, MAXHERB_ROW_TOP);
const MAXHERB_FIELD_SIZE:      egui::Vec2 = egui::vec2(110.0, 32.0);

// Start-heterotrophs row. Initial-cohort herbivore count at `spawn_colony`;
// independent from `MaxHerbivores` (the running cap).
const STARTHET_ROW_TOP:        f32        = 547.0;
const STARTHET_LABEL_POS:      egui::Pos2 = egui::pos2(60.0, STARTHET_ROW_TOP);
const STARTHET_LABEL_SIZE:     egui::Vec2 = egui::vec2(220.0, 32.0);
const STARTHET_DRAG_POS:       egui::Pos2 = egui::pos2(285.0, STARTHET_ROW_TOP);
const STARTHET_FIELD_SIZE:     egui::Vec2 = egui::vec2(110.0, 32.0);

const ACTION_BTN_SIZE:         egui::Vec2 = egui::vec2(330.0, 90.0);
const ACTION_ROW_TOP:          f32        = 600.0;
const ACTION_BTN_FONT_SIZE:    f32        = 24.0;
const START_BTN_LABEL:         &str       = "START AEONS";
const DEFAULT_MAP_PATH:        &str       = "assets/waterworld1.glb";
// Empty ⇒ fresh-spawn mode (spawn fields drive the cohort). Selecting a
// `.colony` greys out the spawn widgets (gated by "Adjust colony dimensions").
const DEFAULT_COLONY_PATH:     &str       = "";


// ── Public API ───────────────────────────────────────────────────────────────

/// Outcome of the launcher: which workflow the user picked.
pub enum LaunchMode {
    /// Boot the full simulation (Bevy + Burn brain pools + every plugin).
    RunSimulation {
        map_path:       String,
        colony_path:    Option<String>,
        wireframe:      bool,
        map_x:          f32,
        map_z:          f32,
        max_phototrophs:  usize,
        /// Herbivore reproduction soft cap (`--max-herbivores N`).
        max_herbivores: usize,
        /// Startup heterotroph cohort (`--start-heteros N`); independent of `MaxHerbivores`.
        start_heterotrophs: usize,
        /// Startup photoautotroph cohort (`--start-photos N`); independent of `MaxPhotoautotrophs`.
        start_photoautotrophs: usize,
        /// AI-training-mode state at click-of-START (checkbox wins over the
        /// seeded `--trainingmode` argv).
        training_mode:  bool,
        /// World-space Y of the global water surface (`--water-level Y`).
        water_level:    f32,
        /// When a colony is loaded, whether to override its saved dimensions
        /// (currently the water level) with the launcher-chosen values
        /// (`--adjust-colony-dimensions`).
        adjust_colony_dimensions: bool,
    },
    /// Boot the colony editor (Bevy + minimal plugin set, no AI).
    RunEditor {
        map_path: String,
        map_x:    f32,
        map_z:    f32,
    },
}

/// Open the launcher. `Some(mode)` if an action button was clicked, `None` if closed.
pub fn run_launcher() -> Option<LaunchMode> {
    let (tx, rx) = mpsc::channel::<LaunchMode>();

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
    rx.try_recv().ok()
}


// ── Internals ────────────────────────────────────────────────────────────────

struct LauncherApp {
    map_path:       String,
    colony_path:    String,
    wireframe:      bool,
    map_x:          f32,
    map_z:          f32,
    max_phototrophs:  usize,
    /// Herbivore reproduction soft cap (`--max-herbivores N`).
    max_herbivores: usize,
    /// Initial-cohort herbivore count (`--start-heteros N`).
    start_heterotrophs: usize,
    /// Initial-cohort photoautotroph count (`--start-photos N`).
    start_photoautotrophs: usize,
    /// AI-training-mode checkbox; seeded from `--trainingmode`, mutable. The
    /// value held at click-of-START is what the child boots with.
    training_mode:  bool,
    /// World-space Y of the global water surface. Follows the spawn-control
    /// enabled flag (greyed with a colony loaded unless "Adjust colony
    /// dimensions" is ticked).
    water_level:    f32,
    /// When a colony is loaded, spawn-control widgets are disabled so it loads
    /// as recorded; checking this re-enables them to override the caps. No
    /// effect with no colony selected (widgets always editable then).
    adjust_colony_dimensions: bool,
    banner:        Option<egui::TextureHandle>,
    status:        String,
    tx:            mpsc::Sender<LaunchMode>,
}

impl LauncherApp {
    fn new(cc: &eframe::CreationContext<'_>, tx: mpsc::Sender<LaunchMode>) -> Self {
        let wireframe     = env::args().any(|a| a == "--wireframe");
        let training_mode = env::args().any(|a| a == "--trainingmode");
        Self {
            map_path:       DEFAULT_MAP_PATH.to_string(),
            colony_path:    DEFAULT_COLONY_PATH.to_string(),
            wireframe,
            map_x:          DEFAULT_MAP_X,
            map_z:          DEFAULT_MAP_Z,
            max_phototrophs:  DEFAULT_MAX_PHOTOAUTOTROPHS,
            max_herbivores: DEFAULT_MAX_HERBIVORES,
            start_heterotrophs:    DEFAULT_START_HETEROTROPHS,
            start_photoautotrophs: DEFAULT_START_PHOTOAUTOTROPHS,
            training_mode,
            water_level:    DEFAULT_WATER_LEVEL,
            adjust_colony_dimensions: false,
            banner:         load_banner(&cc.egui_ctx),
            status:         String::new(),
            tx,
        }
    }

    fn launch_simulation(&mut self, ctx: &egui::Context) {
        let map = self.map_path.trim();
        if map.is_empty() {
            self.status = "Please enter a path to a .glb or .world world file.".into();
            return;
        }
        let colony = self.colony_path.trim();
        let mode = LaunchMode::RunSimulation {
            map_path:       map.to_string(),
            colony_path:    if colony.is_empty() { None } else { Some(colony.to_string()) },
            wireframe:      self.wireframe,
            map_x:          self.map_x,
            map_z:          self.map_z,
            max_phototrophs:  self.max_phototrophs.max(1),
            max_herbivores: self.max_herbivores,
            start_heterotrophs:    self.start_heterotrophs,
            start_photoautotrophs: self.start_photoautotrophs,
            training_mode:  self.training_mode,
            water_level:    self.water_level,
            adjust_colony_dimensions: self.adjust_colony_dimensions,
        };
        let _ = self.tx.send(mode);
        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
    }

    fn launch_editor(&mut self, ctx: &egui::Context) {
        let map = self.map_path.trim();
        if map.is_empty() {
            self.status = "Please enter a path to a .glb or .world world file.".into();
            return;
        }
        // Colony field is silently ignored in editor mode (simulation-only).
        let _ = self.tx.send(LaunchMode::RunEditor {
            map_path: map.to_string(),
            map_x:    self.map_x,
            map_z:    self.map_z,
        });
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

fn dialog_start_dir(current: &str) -> std::path::PathBuf {
    let trimmed = current.trim();
    if !trimmed.is_empty() {
        let p = std::path::Path::new(trimmed);
        if let Some(parent) = p.parent() {
            if parent.is_dir() {
                return parent.to_path_buf();
            }
        }
    }
    env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."))
}

impl eframe::App for LauncherApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
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

                // ── Map field + Open dialog ─────────────────────────
                let textfield_rect = egui::Rect::from_min_size(TEXTFIELD_POS, TEXTFIELD_SIZE);
                ui.put(
                    textfield_rect,
                    egui::TextEdit::singleline(&mut self.map_path)
                        .hint_text("Path to .glb or .world world file")
                        .font(egui::FontId::proportional(TEXTFIELD_FONT_SIZE)),
                );
                let open_map_rect = egui::Rect::from_min_size(OPEN_MAP_BTN_POS, OPEN_BTN_SIZE);
                if ui.put(open_map_rect, egui::Button::new("Open…")).clicked() {
                    let start = dialog_start_dir(&self.map_path);
                    // `set_parent(frame)` ties the dialog to the window so the
                    // portal stacks it in front (Wayland places it behind otherwise).
                    if let Some(p) = rfd::FileDialog::new()
                        .add_filter("AEONS world / glTF", &["world", "glb"])
                        .add_filter("All files", &["*"])
                        .set_directory(&start)
                        .set_parent(frame)
                        .pick_file()
                    {
                        self.map_path = p.to_string_lossy().into_owned();
                    }
                }

                // ── Colony field (simulation-only) + Open dialog ────
                let colonyfield_rect = egui::Rect::from_min_size(COLONYFIELD_POS, COLONYFIELD_SIZE);
                ui.put(
                    colonyfield_rect,
                    egui::TextEdit::singleline(&mut self.colony_path)
                        .hint_text("Optional: .colony save file (simulation only)")
                        .font(egui::FontId::proportional(TEXTFIELD_FONT_SIZE)),
                );
                let open_colony_rect = egui::Rect::from_min_size(OPEN_COLONY_BTN_POS, OPEN_BTN_SIZE);
                if ui.put(open_colony_rect, egui::Button::new("Open…")).clicked() {
                    let start = dialog_start_dir(&self.colony_path);
                    if let Some(p) = rfd::FileDialog::new()
                        .add_filter("AEONS colony (.colony)", &["colony"])
                        .add_filter("All files", &["*"])
                        .set_directory(&start)
                        .set_parent(frame)
                        .pick_file()
                    {
                        self.colony_path = p.to_string_lossy().into_owned();
                    }
                }

                // ── AI-training-mode checkbox ───────────────────────
                // OUTSIDE the colony-dimension gate, so it stays editable with
                // or without a colony loaded (orthogonal to spawn overrides).
                let training_rect = egui::Rect::from_min_size(
                    TRAININGMODE_POS, TRAININGMODE_SIZE);
                ui.put(
                    training_rect,
                    egui::Checkbox::new(&mut self.training_mode, "AI-training mode"),
                );

                // ── Map-size row ────────────────────────────────────
                // X/Z dimensions the world is normalised to; forwarded via
                // `--map-size X Z` into the `MapSize` resource.
                let mapsize_label_rect = egui::Rect::from_min_size(
                    MAPSIZE_LABEL_POS, MAPSIZE_LABEL_SIZE);
                ui.put(
                    mapsize_label_rect,
                    egui::Label::new(
                        egui::RichText::new("Map size:").size(TEXTFIELD_FONT_SIZE),
                    ),
                );

                let x_label_rect = egui::Rect::from_min_size(
                    MAPSIZE_X_LABEL_POS, egui::vec2(20.0, 32.0));
                ui.put(x_label_rect, egui::Label::new(
                    egui::RichText::new("X").size(TEXTFIELD_FONT_SIZE),
                ));
                let x_drag_rect = egui::Rect::from_min_size(
                    MAPSIZE_X_DRAG_POS, MAPSIZE_FIELD_SIZE);
                ui.put(
                    x_drag_rect,
                    egui::DragValue::new(&mut self.map_x)
                        .range(MAPSIZE_MIN..=MAPSIZE_MAX)
                        .speed(8.0),
                );

                let z_label_rect = egui::Rect::from_min_size(
                    MAPSIZE_Z_LABEL_POS, egui::vec2(20.0, 32.0));
                ui.put(z_label_rect, egui::Label::new(
                    egui::RichText::new("Z").size(TEXTFIELD_FONT_SIZE),
                ));
                let z_drag_rect = egui::Rect::from_min_size(
                    MAPSIZE_Z_DRAG_POS, MAPSIZE_FIELD_SIZE);
                ui.put(
                    z_drag_rect,
                    egui::DragValue::new(&mut self.map_z)
                        .range(MAPSIZE_MIN..=MAPSIZE_MAX)
                        .speed(8.0),
                );

                // ── Adjust-colony-dimensions checkbox ───────────────
                // Only rendered when a colony is selected — it's meaningless
                // otherwise (gates the spawn widgets for the load-a-colony case).
                let colony_loaded = !self.colony_path.trim().is_empty();
                if colony_loaded {
                    let adjustcol_rect = egui::Rect::from_min_size(
                        ADJUSTCOL_POS, ADJUSTCOL_SIZE);
                    ui.put(
                        adjustcol_rect,
                        egui::Checkbox::new(
                            &mut self.adjust_colony_dimensions,
                            "Adjust colony dimensions",
                        ),
                    );
                }

                // Spawn widgets greyed out only when a colony is loaded AND
                // "Adjust colony dimensions" is unticked; editable otherwise.
                let spawn_widgets_enabled =
                    !colony_loaded || self.adjust_colony_dimensions;

                // ── Water-level field (beside the map-size row) ─────
                // Unlike the always-editable map-size fields, this follows the
                // spawn-control gate (a loaded colony restores its saved water
                // level unless "Adjust colony dimensions" is ticked).
                let waterlevel_label_rect = egui::Rect::from_min_size(
                    WATERLEVEL_LABEL_POS, WATERLEVEL_LABEL_SIZE);
                ui.put(waterlevel_label_rect, egui::Label::new(
                    egui::RichText::new("Water level:").size(TEXTFIELD_FONT_SIZE),
                ));
                let waterlevel_drag_rect = egui::Rect::from_min_size(
                    WATERLEVEL_DRAG_POS, WATERLEVEL_FIELD_SIZE);
                ui.add_enabled_ui(spawn_widgets_enabled, |ui| {
                    ui.put(
                        waterlevel_drag_rect,
                        egui::DragValue::new(&mut self.water_level)
                            .range(WATERLEVEL_MIN..=WATERLEVEL_MAX)
                            .speed(1.0),
                    );
                });

                ui.add_enabled_ui(spawn_widgets_enabled, |ui| {
                    // ── Max-organisms row ───────────────────────────
                    // Sizes the GPU brain-pool tensors at startup; the runtime
                    // panel can lower the soft cap but never above this. Editor ignores.
                    let maxorg_label_rect = egui::Rect::from_min_size(
                        MAXPHOTO_LABEL_POS, MAXPHOTO_LABEL_SIZE);
                    ui.put(
                        maxorg_label_rect,
                        egui::Label::new(
                            egui::RichText::new("Max Phototrophic Organisms:").size(TEXTFIELD_FONT_SIZE),
                        ),
                    );
                    let maxorg_drag_rect = egui::Rect::from_min_size(
                        MAXPHOTO_DRAG_POS, MAXPHOTO_FIELD_SIZE);
                    ui.put(
                        maxorg_drag_rect,
                        egui::DragValue::new(&mut self.max_phototrophs)
                            .range(1..=LAUNCHER_POPULATION_CAP)
                            .speed(1.0),
                    );

                    // ── Spawn-phototrophic-organisms row ─────────────
                    // Initial-cohort photo count; below `MaxPhotoautotrophs`
                    // leaves room for reproduction to backfill.
                    let spawnphoto_label_rect = egui::Rect::from_min_size(
                        SPAWNPHOTO_LABEL_POS, SPAWNPHOTO_LABEL_SIZE);
                    ui.put(
                        spawnphoto_label_rect,
                        egui::Label::new(
                            egui::RichText::new("Spawn Phototrophic Organisms:").size(TEXTFIELD_FONT_SIZE),
                        ),
                    );
                    let spawnphoto_drag_rect = egui::Rect::from_min_size(
                        SPAWNPHOTO_DRAG_POS, SPAWNPHOTO_FIELD_SIZE);
                    ui.put(
                        spawnphoto_drag_rect,
                        egui::DragValue::new(&mut self.start_photoautotrophs)
                            .range(0..=LAUNCHER_POPULATION_CAP)
                            .speed(1.0),
                    );

                    // ── Max-herbivores row ──────────────────────────
                    // Seeds the `MaxHerbivores` reproduction cap. Clamped to
                    // `[1, LAUNCHER_POPULATION_CAP]` so it can't exceed pool size.
                    let maxherb_label_rect = egui::Rect::from_min_size(
                        MAXHERB_LABEL_POS, MAXHERB_LABEL_SIZE);
                    ui.put(
                        maxherb_label_rect,
                        egui::Label::new(
                            egui::RichText::new("Max Herbivores:").size(TEXTFIELD_FONT_SIZE),
                        ),
                    );
                    let maxherb_drag_rect = egui::Rect::from_min_size(
                        MAXHERB_DRAG_POS, MAXHERB_FIELD_SIZE);
                    ui.put(
                        maxherb_drag_rect,
                        egui::DragValue::new(&mut self.max_herbivores)
                            .range(1..=LAUNCHER_POPULATION_CAP)
                            .speed(1.0),
                    );

                    // ── Start-heterotrophs row ───────────────────
                    // Initial-cohort herbivore count; below `MaxHerbivores`
                    // leaves headroom for reproduction.
                    let starthet_label_rect = egui::Rect::from_min_size(
                        STARTHET_LABEL_POS, STARTHET_LABEL_SIZE);
                    ui.put(
                        starthet_label_rect,
                        egui::Label::new(
                            egui::RichText::new("Start Heterotroph Number:").size(TEXTFIELD_FONT_SIZE),
                        ),
                    );
                    let starthet_drag_rect = egui::Rect::from_min_size(
                        STARTHET_DRAG_POS, STARTHET_FIELD_SIZE);
                    ui.put(
                        starthet_drag_rect,
                        egui::DragValue::new(&mut self.start_heterotrophs)
                            .range(0..=LAUNCHER_POPULATION_CAP)
                            .speed(1.0),
                    );
                });

                // ── Action button ───────────────────────────────────
                // Single launch button; editor / species-editor modes are
                // reachable from the in-app mode bar after the sim starts.
                let total_w  = ACTION_BTN_SIZE.x;
                let row_left = (LAUNCHER_WINDOW_SIZE.x - total_w) * 0.5;
                let start_pos  = egui::pos2(row_left, ACTION_ROW_TOP);
                let start_rect = egui::Rect::from_min_size(start_pos, ACTION_BTN_SIZE);
                let start_label = egui::RichText::new(START_BTN_LABEL).size(ACTION_BTN_FONT_SIZE).strong();

                if ui.put(start_rect, egui::Button::new(start_label)).clicked() {
                    self.launch_simulation(ctx);
                }

                if !self.status.is_empty() {
                    let status_pos  = egui::pos2(row_left, ACTION_ROW_TOP + ACTION_BTN_SIZE.y + 6.0);
                    let status_rect = egui::Rect::from_min_size(
                        status_pos, egui::vec2(total_w, 24.0),
                    );
                    ui.put(status_rect, egui::Label::new(&self.status));
                }
            });
    }
}
