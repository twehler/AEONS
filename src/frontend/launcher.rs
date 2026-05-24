// Combined launcher (eframe / egui).
//
// Two action buttons side-by-side: START AEONS and COLONY EDITOR.
// Returns a `LaunchMode` that `main.rs` consumes to re-spawn the
// binary with the appropriate argv (subprocess re-spawn is required
// because winit's `EventLoop` is a singleton — we can't show this
// eframe window and then a Bevy window in the same process).

use std::env;
use std::sync::mpsc;

use eframe::egui;

use crate::simulation_settings::{
    DEFAULT_MAX_ORGANISMS, DEFAULT_MAX_HERBIVORES, DEFAULT_START_HETEROTROPHS,
    DEFAULT_MAP_X, DEFAULT_MAP_Z,
};

/// Soft practical upper bound for the launcher's "Max Organisms" field.
/// Higher values still work (the simulation simply allocates a larger
/// GPU brain pool), but the DragValue would feel runaway at very high
/// magnitudes. Picked to comfortably exceed any realistic colony size.
const LAUNCHER_MAX_ORGANISMS_CAP: usize = 100_000;


// ── Layout constants ─────────────────────────────────────────────────────────

const LAUNCHER_WINDOW_SIZE:    egui::Vec2 = egui::vec2(800.0, 680.0);
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

// Map-size row, sits between the colony textfield and the action
// buttons. Two `DragValue` widgets (X / Z) plus a heading label.
const MAPSIZE_ROW_TOP:         f32        = 345.0;
const MAPSIZE_LABEL_POS:       egui::Pos2 = egui::pos2(60.0, MAPSIZE_ROW_TOP);
const MAPSIZE_LABEL_SIZE:      egui::Vec2 = egui::vec2(130.0, 32.0);
const MAPSIZE_X_LABEL_POS:     egui::Pos2 = egui::pos2(200.0, MAPSIZE_ROW_TOP);
const MAPSIZE_X_DRAG_POS:      egui::Pos2 = egui::pos2(225.0, MAPSIZE_ROW_TOP);
const MAPSIZE_Z_LABEL_POS:     egui::Pos2 = egui::pos2(345.0, MAPSIZE_ROW_TOP);
const MAPSIZE_Z_DRAG_POS:      egui::Pos2 = egui::pos2(370.0, MAPSIZE_ROW_TOP);
const MAPSIZE_FIELD_SIZE:      egui::Vec2 = egui::vec2(110.0, 32.0);
const MAPSIZE_MIN:             f32        = 50.0;
const MAPSIZE_MAX:             f32        = 8192.0;

// Max-organisms row. Single `DragValue` capped at
// `LAUNCHER_MAX_ORGANISMS_CAP`. The chosen value is forwarded to the
// child subprocess via `--max-organisms N` argv and sizes both the
// GPU brain-pool tensors (`OrganismPoolSize`) AND the initial
// reproduction soft cap (`MaxOrganisms`) at simulation startup.
// Adjust-colony-dimensions checkbox. Acts as the gate over the
// downstream spawn-control widgets (Max Organisms, AI-training mode,
// Max Herbivores) when a colony file is loaded. The user-facing
// contract is documented on the field in `LauncherApp`.
const ADJUSTCOL_ROW_TOP:       f32        = 380.0;
const ADJUSTCOL_POS:           egui::Pos2 = egui::pos2(60.0, ADJUSTCOL_ROW_TOP);
const ADJUSTCOL_SIZE:          egui::Vec2 = egui::vec2(400.0, 24.0);

const MAXORG_ROW_TOP:          f32        = 415.0;
const MAXORG_LABEL_POS:        egui::Pos2 = egui::pos2(60.0, MAXORG_ROW_TOP);
const MAXORG_LABEL_SIZE:       egui::Vec2 = egui::vec2(180.0, 32.0);
const MAXORG_DRAG_POS:         egui::Pos2 = egui::pos2(245.0, MAXORG_ROW_TOP);
const MAXORG_FIELD_SIZE:       egui::Vec2 = egui::vec2(110.0, 32.0);

// AI-training-mode checkbox — sits between the Max-Organisms row and
// the action button. The checkbox is the final word on whether the
// child process boots with training mode active: it initialises from
// the `--trainingmode` flag (if the launcher itself was invoked with
// it) but the user can override the initial value by toggling, and
// the final committed state is forwarded via
// `LaunchMode::RunSimulation::training_mode`.
const TRAININGMODE_ROW_TOP:    f32        = 450.0;
const TRAININGMODE_POS:        egui::Pos2 = egui::pos2(60.0, TRAININGMODE_ROW_TOP);
const TRAININGMODE_SIZE:       egui::Vec2 = egui::vec2(400.0, 24.0);

// Max-herbivores row. Same layout pattern as the Max-organisms row,
// just one slot below the AI-training-mode checkbox. The value seeds
// the `MaxHerbivores` reproduction cap at simulation startup. Tuned
// from the launcher so AI-training runs can fix the cohort size before
// the brain pool is allocated.
const MAXHERB_ROW_TOP:         f32        = 485.0;
const MAXHERB_LABEL_POS:       egui::Pos2 = egui::pos2(60.0, MAXHERB_ROW_TOP);
const MAXHERB_LABEL_SIZE:      egui::Vec2 = egui::vec2(180.0, 32.0);
const MAXHERB_DRAG_POS:        egui::Pos2 = egui::pos2(245.0, MAXHERB_ROW_TOP);
const MAXHERB_FIELD_SIZE:      egui::Vec2 = egui::vec2(110.0, 32.0);

// Start-heterotrophs row. Sits one slot below "Max Herbivores",
// shares the same label / drag layout. Drives the initial-cohort
// herbivore count at `spawn_colony`. Independent from
// `MaxHerbivores`, which caps the running population.
const STARTHET_ROW_TOP:        f32        = 520.0;
const STARTHET_LABEL_POS:      egui::Pos2 = egui::pos2(60.0, STARTHET_ROW_TOP);
const STARTHET_LABEL_SIZE:     egui::Vec2 = egui::vec2(220.0, 32.0);
const STARTHET_DRAG_POS:       egui::Pos2 = egui::pos2(285.0, STARTHET_ROW_TOP);
const STARTHET_FIELD_SIZE:     egui::Vec2 = egui::vec2(110.0, 32.0);

// Two side-by-side buttons. "START AEONS" on the left, "COLONY EDITOR"
// on the right — both styled the same so neither feels like a footnote.
const ACTION_BTN_SIZE:         egui::Vec2 = egui::vec2(330.0, 90.0);
const ACTION_ROW_TOP:          f32        = 565.0;
const ACTION_BTN_FONT_SIZE:    f32        = 24.0;
const START_BTN_LABEL:         &str       = "START AEONS";
const DEFAULT_MAP_PATH:        &str       = "assets/world_superflat.glb";
// Empty by default so the launcher opens in fresh-spawn mode: the
// Max Organisms / Max Herbivores fields drive the initial cohort
// directly. The user can paste or browse to a `.colony` file
// whenever they explicitly want to load one — that flips the spawn-
// control widgets to greyed-out (gated by "Adjust colony
// dimensions").
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
        max_organisms:  usize,
        /// Reproduction soft cap on herbivores. Forwarded to the
        /// child as `--max-herbivores N`. Seeded from the launcher's
        /// text field — defaults to `DEFAULT_MAX_HERBIVORES`.
        max_herbivores: usize,
        /// Number of heterotrophs spawned at startup. Drives the
        /// initial cohort in `spawn_colony`; the cap (`MaxHerbivores`)
        /// is independent. Forwarded via `--start-heteros N`.
        start_heterotrophs: usize,
        /// Final AI-training-mode state at click-of-START — the
        /// launcher's checkbox is the source of truth. It seeds
        /// from `--trainingmode` if that argv flag was set when the
        /// launcher process started, but the user may toggle it
        /// before clicking and the toggle wins.
        training_mode:  bool,
    },
    /// Boot the colony editor (Bevy + minimal plugin set, no AI).
    RunEditor {
        map_path: String,
        map_x:    f32,
        map_z:    f32,
    },
}

/// Open the launcher window. Returns `Some(mode)` when the user clicked
/// either action button, `None` if they closed the window.
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
    max_organisms:  usize,
    /// Herbivore reproduction soft cap. Editable via the Max
    /// Herbivores text field; forwarded to the child via
    /// `--max-herbivores N`.
    max_herbivores: usize,
    /// Initial-cohort herbivore count. Editable via the
    /// Start Heterotroph Number text field; forwarded via
    /// `--start-heteros N`.
    start_heterotrophs: usize,
    /// AI-training-mode checkbox state. Initialised from
    /// `--trainingmode` in argv; mutable while the launcher is open.
    /// Whatever value is held when the user clicks START AEONS is
    /// what the child process boots with.
    training_mode:  bool,
    /// When a colony save is selected, the spawn-control widgets
    /// (Max Organisms, AI-training mode, Max Herbivores) are
    /// disabled by default so the colony loads exactly as recorded.
    /// Checking this box re-enables those widgets so the user can
    /// override the cap values for the loaded colony. Has no effect
    /// when no colony is selected (the widgets are always editable
    /// in that case).
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
            // Default seed: the global "starter" value. The user can
            // dial it freely up to `LAUNCHER_MAX_ORGANISMS_CAP`; the
            // chosen value flows through to the GPU brain-pool size.
            max_organisms:  DEFAULT_MAX_ORGANISMS,
            // Herbivore cap default — mirrors `MaxHerbivores::default()`.
            // Capped at `LAUNCHER_MAX_ORGANISMS_CAP` from above for
            // simplicity (the herbivore count can never exceed the
            // overall organism cap anyway).
            max_herbivores: DEFAULT_MAX_HERBIVORES,
            // Initial herbivore cohort default. Defaults to the same
            // value as the cap so an out-of-the-box run with default
            // settings still produces a visible starter cohort.
            start_heterotrophs: DEFAULT_START_HETEROTROPHS,
            training_mode,
            // Default OFF — if the user pre-fills a colony path,
            // they'll see the spawn widgets greyed out until they
            // explicitly opt in to overriding the loaded colony's
            // dimensions.
            adjust_colony_dimensions: false,
            banner:         load_banner(&cc.egui_ctx),
            status:         String::new(),
            tx,
        }
    }

    fn launch_simulation(&mut self, ctx: &egui::Context) {
        let map = self.map_path.trim();
        if map.is_empty() {
            self.status = "Please enter a path to a .glb world file.".into();
            return;
        }
        let colony = self.colony_path.trim();
        let mode = LaunchMode::RunSimulation {
            map_path:       map.to_string(),
            colony_path:    if colony.is_empty() { None } else { Some(colony.to_string()) },
            wireframe:      self.wireframe,
            map_x:          self.map_x,
            map_z:          self.map_z,
            max_organisms:  self.max_organisms.max(1),
            max_herbivores: self.max_herbivores,
            start_heterotrophs: self.start_heterotrophs,
            training_mode:  self.training_mode,
        };
        let _ = self.tx.send(mode);
        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
    }

    fn launch_editor(&mut self, ctx: &egui::Context) {
        let map = self.map_path.trim();
        if map.is_empty() {
            self.status = "Please enter a path to a .glb world file.".into();
            return;
        }
        // The colony field is silently ignored in editor mode — it's a
        // simulation-only concept. We don't error if it's set; the user
        // may have typed it and then changed their mind.
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

                // ── Map field + Open dialog ─────────────────────────
                let textfield_rect = egui::Rect::from_min_size(TEXTFIELD_POS, TEXTFIELD_SIZE);
                ui.put(
                    textfield_rect,
                    egui::TextEdit::singleline(&mut self.map_path)
                        .hint_text("Path to .glb world file")
                        .font(egui::FontId::proportional(TEXTFIELD_FONT_SIZE)),
                );
                let open_map_rect = egui::Rect::from_min_size(OPEN_MAP_BTN_POS, OPEN_BTN_SIZE);
                if ui.put(open_map_rect, egui::Button::new("Open…")).clicked() {
                    let start = dialog_start_dir(&self.map_path);
                    if let Some(p) = rfd::FileDialog::new()
                        .add_filter("glTF binary (.glb)", &["glb"])
                        .add_filter("All files", &["*"])
                        .set_directory(&start)
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
                        .pick_file()
                    {
                        self.colony_path = p.to_string_lossy().into_owned();
                    }
                }

                // ── Map-size row ────────────────────────────────────
                // Two DragValue widgets that bind to the X and Z
                // dimensions the loaded world is normalised to. The
                // values are forwarded to the child subprocess via
                // `--map-size X Z` argv and inserted into the `MapSize`
                // resource at simulation / editor startup.
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
                // Always interactive — controls whether the three
                // spawn-control widgets below stay editable when a
                // colony file is loaded. When no colony is loaded,
                // the checkbox is irrelevant (spawn controls are
                // always editable in that case) so it just acts as a
                // user-state memo.
                let adjustcol_rect = egui::Rect::from_min_size(
                    ADJUSTCOL_POS, ADJUSTCOL_SIZE);
                ui.put(
                    adjustcol_rect,
                    egui::Checkbox::new(
                        &mut self.adjust_colony_dimensions,
                        "Adjust colony dimensions",
                    ),
                );

                // Editability gate for Max Organisms / AI-training /
                // Max Herbivores: when a colony is loaded AND the
                // user hasn't ticked "Adjust colony dimensions",
                // these are greyed out so the colony loads as
                // recorded. Otherwise they're editable (no colony,
                // or user has explicitly opted in to override).
                let spawn_widgets_enabled =
                    self.colony_path.trim().is_empty()
                    || self.adjust_colony_dimensions;

                ui.add_enabled_ui(spawn_widgets_enabled, |ui| {
                    // ── Max-organisms row ───────────────────────────
                    // This value sizes the GPU brain-pool tensors at
                    // simulation startup. After launch the
                    // statistics-panel "Max Organisms" field can lower
                    // this soft cap further, but never above what was
                    // allocated here. Ignored by the editor.
                    let maxorg_label_rect = egui::Rect::from_min_size(
                        MAXORG_LABEL_POS, MAXORG_LABEL_SIZE);
                    ui.put(
                        maxorg_label_rect,
                        egui::Label::new(
                            egui::RichText::new("Max Organisms:").size(TEXTFIELD_FONT_SIZE),
                        ),
                    );
                    let maxorg_drag_rect = egui::Rect::from_min_size(
                        MAXORG_DRAG_POS, MAXORG_FIELD_SIZE);
                    ui.put(
                        maxorg_drag_rect,
                        egui::DragValue::new(&mut self.max_organisms)
                            .range(1..=LAUNCHER_MAX_ORGANISMS_CAP)
                            .speed(1.0),
                    );

                    // ── AI-training-mode checkbox ───────────────────
                    // Seeded from `--trainingmode` argv on launcher
                    // start (see `LauncherApp::new`). Whatever the
                    // user leaves it on when clicking START is what's
                    // forwarded — so the launcher checkbox overrides
                    // the argv flag and they "compete" via
                    // last-touched-wins.
                    let training_rect = egui::Rect::from_min_size(
                        TRAININGMODE_POS, TRAININGMODE_SIZE);
                    ui.put(
                        training_rect,
                        egui::Checkbox::new(&mut self.training_mode, "AI-training mode"),
                    );

                    // ── Max-herbivores row ──────────────────────────
                    // Seeds the `MaxHerbivores` reproduction cap. The
                    // statistics panel exposes the same value at
                    // runtime so the user can lower or raise it
                    // later, but the initial-cohort sizing is
                    // anchored here. Clamped to
                    // `[1, LAUNCHER_MAX_ORGANISMS_CAP]` so the
                    // herbivore cap can never exceed the overall
                    // pool size.
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
                            .range(1..=LAUNCHER_MAX_ORGANISMS_CAP)
                            .speed(1.0),
                    );

                    // ── Start-heterotrophs row ───────────────────
                    // Initial-cohort herbivore count, independent
                    // from the running-population cap. Defaults to
                    // `DEFAULT_START_HETEROTROPHS = 5`. Setting it
                    // below `MaxHerbivores` (and `MaxOrganisms`)
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
                            .range(0..=LAUNCHER_MAX_ORGANISMS_CAP)
                            .speed(1.0),
                    );
                });

                // ── Action button ───────────────────────────────────
                // Single launch button now that the standalone Colony
                // Editor entry has been removed. Editor / species-editor
                // modes are reachable from the in-app mode bar after
                // the simulation starts.
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
