// Combined launcher (eframe / egui).
//
// Returns a `LaunchMode` that `main.rs` consumes to re-spawn the binary with
// the chosen argv. Subprocess re-spawn is required because winit's `EventLoop`
// is a singleton — eframe and Bevy windows can't coexist in one process.

use std::env;
use std::sync::mpsc;

use eframe::egui;

// ── Layout constants ─────────────────────────────────────────────────────────
//
// The launcher is minimal: a banner, a single `.aeonsw` field + Open button, two
// stacked action buttons (Start / Continue without a file), and a status line.
// (The old multi-field form is gone — everything else is chosen from the loaded
// `.aeonsw` or the in-engine setup dialogue.)

const LAUNCHER_WINDOW_SIZE:    egui::Vec2 = egui::vec2(800.0, 470.0);
const BANNER_PATH:             &str       = "logos/dna.png";
const BANNER_POS:              egui::Pos2 = egui::pos2(0.0, 0.0);
const BANNER_SIZE:             egui::Vec2 = egui::vec2(800.0, 200.0);

const TEXTFIELD_POS:           egui::Pos2 = egui::pos2(60.0, 230.0);
const TEXTFIELD_SIZE:          egui::Vec2 = egui::vec2(580.0, 32.0);
const TEXTFIELD_FONT_SIZE:     f32        = 16.0;
const OPEN_BTN_SIZE:           egui::Vec2 = egui::vec2(90.0, 32.0);
const OPEN_MAP_BTN_POS:        egui::Pos2 = egui::pos2(650.0, 230.0);

// Two stacked action buttons under the `.aeonsw` field: Start, then Continue.
// `update` places Continue at `ACTION_ROW_TOP + height + 8` and the status line
// below that — all sized to fit inside `LAUNCHER_WINDOW_SIZE`.
const ACTION_BTN_SIZE:         egui::Vec2 = egui::vec2(330.0, 64.0);
const ACTION_ROW_TOP:          f32        = 290.0;
const ACTION_BTN_FONT_SIZE:    f32        = 22.0;
const START_BTN_LABEL:         &str       = "Start";
/// Default `.aeonsw` shown in the picker (the bundled sample world).
const DEFAULT_MAP_PATH:        &str       = "assets/waterworld1.aeonsw";


// ── Public API ───────────────────────────────────────────────────────────────

/// Outcome of the launcher: which workflow the user picked.
pub enum LaunchMode {
    /// Load a `.aeonsw` world and boot the full simulation. The `.aeonsw` is
    /// self-contained (terrain + water + optional colony) and authoritative on
    /// load, so the launcher passes only the path + the two render/training
    /// passthroughs; everything else comes from the file or runtime defaults.
    RunSimulation {
        map_path:      String,
        wireframe:     bool,
        training_mode: bool,
    },
    /// No `.aeonsw` chosen: boot AEONS into the in-engine setup dialogue (choose a
    /// `.glb` or a flat world, set water + map dims), then into the Map Editor.
    NewWorld,
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
    /// The chosen `.aeonsw` path (the only user-editable input).
    map_path: String,
    /// Render-wireframe + AI-training passthroughs, seeded from this process's
    /// argv (`--wireframe` / `--trainingmode`) and forwarded to the child.
    wireframe:     bool,
    training_mode: bool,
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
            wireframe,
            training_mode,
            banner:         load_banner(&cc.egui_ctx),
            status:         String::new(),
            tx,
        }
    }

    /// Start with the chosen `.aeonsw` (load terrain + water + colony, boot the sim).
    fn launch_simulation(&mut self, ctx: &egui::Context) {
        let map = self.map_path.trim();
        if map.is_empty() {
            self.status = "Choose a .aeonsw world file, or click \"Continue without a file\".".into();
            return;
        }
        let _ = self.tx.send(LaunchMode::RunSimulation {
            map_path:      map.to_string(),
            wireframe:     self.wireframe,
            training_mode: self.training_mode,
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

                // ── .aeonsw field + Open dialog ─────────────────────
                // The launcher ONLY asks for a `.aeonsw` (terrain + colony + water,
                // self-contained). Everything else is chosen in-engine: a loaded
                // `.aeonsw` boots the sim (or the Colony Editor if it has no colony);
                // "Continue without a file" boots the in-engine setup dialogue.
                let textfield_rect = egui::Rect::from_min_size(TEXTFIELD_POS, TEXTFIELD_SIZE);
                ui.put(
                    textfield_rect,
                    egui::TextEdit::singleline(&mut self.map_path)
                        .hint_text("Path to a .aeonsw world file")
                        .font(egui::FontId::proportional(TEXTFIELD_FONT_SIZE)),
                );
                let open_map_rect = egui::Rect::from_min_size(OPEN_MAP_BTN_POS, OPEN_BTN_SIZE);
                if ui.put(open_map_rect, egui::Button::new("Open…")).clicked() {
                    let start = dialog_start_dir(&self.map_path);
                    // `set_parent(frame)` ties the dialog to the window so the
                    // portal stacks it in front (Wayland places it behind otherwise).
                    if let Some(p) = rfd::FileDialog::new()
                        .add_filter("AEONS world (.aeonsw)", &["aeonsw"])
                        .add_filter("All files", &["*"])
                        .set_directory(&start)
                        .set_parent(frame)
                        .pick_file()
                    {
                        self.map_path = p.to_string_lossy().into_owned();
                    }
                }

                // ── Action buttons: Start (load .aeonsw) + Continue (new world) ──
                let total_w  = ACTION_BTN_SIZE.x;
                let row_left = (LAUNCHER_WINDOW_SIZE.x - total_w) * 0.5;
                let start_rect = egui::Rect::from_min_size(
                    egui::pos2(row_left, ACTION_ROW_TOP), ACTION_BTN_SIZE);
                let start_label = egui::RichText::new(START_BTN_LABEL).size(ACTION_BTN_FONT_SIZE).strong();
                if ui.put(start_rect, egui::Button::new(start_label)).clicked() {
                    self.launch_simulation(ctx);
                }

                let cont_pos  = egui::pos2(row_left, ACTION_ROW_TOP + ACTION_BTN_SIZE.y + 8.0);
                let cont_rect = egui::Rect::from_min_size(cont_pos, ACTION_BTN_SIZE);
                let cont_label = egui::RichText::new("Continue without a file").size(ACTION_BTN_FONT_SIZE);
                if ui.put(cont_rect, egui::Button::new(cont_label)).clicked() {
                    let _ = self.tx.send(LaunchMode::NewWorld);
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }

                if !self.status.is_empty() {
                    let status_pos  = egui::pos2(row_left, cont_pos.y + ACTION_BTN_SIZE.y + 6.0);
                    let status_rect = egui::Rect::from_min_size(status_pos, egui::vec2(total_w, 24.0));
                    ui.put(status_rect, egui::Label::new(&self.status));
                }
            });
    }
}
