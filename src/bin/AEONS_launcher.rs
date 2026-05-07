//! AEONS_launcher — a small GUI front-end that asks the user for the path to
//! a `.glb` world file and then spawns the main AEONS simulation against it.
//!
//! Layout is fully manual: change the constants in the `LAYOUT` block to
//! reposition or resize the banner image, the path text field, and the launch
//! button without touching the rest of the file.

use eframe::egui;
use std::process::Command;

// =============================================================================
// LAYOUT — adjust freely. All values are in logical pixels with the origin at
// the top-left corner of the launcher window.
// =============================================================================

const WINDOW_SIZE: egui::Vec2 = egui::vec2(800.0, 500.0);

// Banner image (drawn straight via the painter so layout never shifts it).
const BANNER_PATH: &str = "logos/dna.png";
const BANNER_POS: egui::Pos2 = egui::pos2(0.0, 0.0);
const BANNER_SIZE: egui::Vec2 = egui::vec2(800.0, 200.0);

// Map-path text field.
const TEXTFIELD_POS: egui::Pos2 = egui::pos2(60.0, 250.0);
const TEXTFIELD_SIZE: egui::Vec2 = egui::vec2(680.0, 32.0);
const TEXTFIELD_FONT_SIZE: f32 = 16.0;

// Launch button.
const BUTTON_POS: egui::Pos2 = egui::pos2(220.0, 360.0);
const BUTTON_SIZE: egui::Vec2 = egui::vec2(360.0, 90.0);
const BUTTON_FONT_SIZE: f32 = 28.0;
const BUTTON_LABEL: &str = "LAUNCH AEONS";

// Default value pre-filled in the text field on startup.
const DEFAULT_MAP_PATH: &str = "assets/world.glb";

// =============================================================================

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("AEONS Launcher")
            .with_inner_size(WINDOW_SIZE)
            .with_resizable(false),
        ..Default::default()
    };

    eframe::run_native(
        "AEONS Launcher",
        options,
        Box::new(|cc| Ok(Box::new(LauncherApp::new(cc)) as Box<dyn eframe::App>)),
    )
}

struct LauncherApp {
    map_path: String,
    banner: Option<egui::TextureHandle>,
    status: String,
}

impl LauncherApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            map_path: DEFAULT_MAP_PATH.to_string(),
            banner: load_banner(&cc.egui_ctx),
            status: String::new(),
        }
    }

    fn launch(&mut self) {
        let map = self.map_path.trim();
        if map.is_empty() {
            self.status = "Please enter a path to a .glb world file.".into();
            return;
        }

        match Command::new("cargo")
            .args(["run", "--release", "--bin", "AEONS", "--", map])
            .spawn()
        {
            Ok(_) => self.status = format!("Launching AEONS with `{map}` …"),
            Err(e) => self.status = format!("Failed to launch AEONS: {e}"),
        }
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

                let button_rect = egui::Rect::from_min_size(BUTTON_POS, BUTTON_SIZE);
                let label = egui::RichText::new(BUTTON_LABEL).size(BUTTON_FONT_SIZE).strong();
                if ui.put(button_rect, egui::Button::new(label)).clicked() {
                    self.launch();
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
