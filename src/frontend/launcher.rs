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


// ── Layout constants ─────────────────────────────────────────────────────────

const LAUNCHER_WINDOW_SIZE:    egui::Vec2 = egui::vec2(800.0, 540.0);
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

// Two side-by-side buttons. "START AEONS" on the left, "COLONY EDITOR"
// on the right — both styled the same so neither feels like a footnote.
const ACTION_BTN_SIZE:         egui::Vec2 = egui::vec2(330.0, 90.0);
const ACTION_BTN_GAP:          f32        = 20.0;
const ACTION_ROW_TOP:          f32        = 380.0;
const ACTION_BTN_FONT_SIZE:    f32        = 24.0;
const START_BTN_LABEL:         &str       = "START AEONS";
const EDITOR_BTN_LABEL:        &str       = "COLONY EDITOR";

const DEFAULT_MAP_PATH:        &str       = "assets/world.glb";
const DEFAULT_COLONY_PATH:     &str       = "";


// ── Public API ───────────────────────────────────────────────────────────────

/// Outcome of the launcher: which workflow the user picked.
pub enum LaunchMode {
    /// Boot the full simulation (Bevy + Burn brain pools + every plugin).
    RunSimulation {
        map_path:    String,
        colony_path: Option<String>,
        wireframe:   bool,
    },
    /// Boot the colony editor (Bevy + minimal plugin set, no AI).
    RunEditor {
        map_path: String,
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
    map_path:    String,
    colony_path: String,
    wireframe:   bool,
    banner:      Option<egui::TextureHandle>,
    status:      String,
    tx:          mpsc::Sender<LaunchMode>,
}

impl LauncherApp {
    fn new(cc: &eframe::CreationContext<'_>, tx: mpsc::Sender<LaunchMode>) -> Self {
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

    fn launch_simulation(&mut self, ctx: &egui::Context) {
        let map = self.map_path.trim();
        if map.is_empty() {
            self.status = "Please enter a path to a .glb world file.".into();
            return;
        }
        let colony = self.colony_path.trim();
        let mode = LaunchMode::RunSimulation {
            map_path:    map.to_string(),
            colony_path: if colony.is_empty() { None } else { Some(colony.to_string()) },
            wireframe:   self.wireframe,
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

                // ── Action buttons row ──────────────────────────────
                let total_w  = ACTION_BTN_SIZE.x * 2.0 + ACTION_BTN_GAP;
                let row_left = (LAUNCHER_WINDOW_SIZE.x - total_w) * 0.5;
                let start_pos  = egui::pos2(row_left, ACTION_ROW_TOP);
                let editor_pos = egui::pos2(row_left + ACTION_BTN_SIZE.x + ACTION_BTN_GAP, ACTION_ROW_TOP);

                let start_rect  = egui::Rect::from_min_size(start_pos,  ACTION_BTN_SIZE);
                let editor_rect = egui::Rect::from_min_size(editor_pos, ACTION_BTN_SIZE);

                let start_label  = egui::RichText::new(START_BTN_LABEL).size(ACTION_BTN_FONT_SIZE).strong();
                let editor_label = egui::RichText::new(EDITOR_BTN_LABEL).size(ACTION_BTN_FONT_SIZE).strong();

                if ui.put(start_rect,  egui::Button::new(start_label)).clicked() {
                    self.launch_simulation(ctx);
                }
                if ui.put(editor_rect, egui::Button::new(editor_label)).clicked() {
                    self.launch_editor(ctx);
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
