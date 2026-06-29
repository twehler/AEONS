// Map editor — paint materials, brushes, and session state.
//
// `MapMaterial` is the small terrain-paint palette (analogous to `CellType` but
// for the bare map): five flat colours the user paints onto the terrain mesh's
// vertex colours. Painting is a purely VISUAL recolour — it never touches the
// simulation or any saved data.
//
// `MapBrush` is the brush selector's backing enum (one brush for now,
// structured so more can be added later). `MapEditorSession` holds the editor's
// transient UI selection state, mirroring the relevant fields of
// `SpeciesSession`.

use bevy::prelude::*;


/// The five terrain paint colours.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MapMaterial {
    Yellow,
    Brown,
    Grey,
    DarkGreen,
    LightGreen,
}

impl MapMaterial {
    /// Every paint colour, in palette order (drives the bottom-panel swatches).
    pub const ALL: [MapMaterial; 5] = [
        Self::Yellow,
        Self::Brown,
        Self::Grey,
        Self::DarkGreen,
        Self::LightGreen,
    ];

    /// sRGB display triple.
    pub fn srgb(self) -> (f32, f32, f32) {
        match self {
            Self::Yellow     => (0.749, 0.647, 0.278), // #bfa547
            Self::Brown      => (0.310, 0.204, 0.137), // #4f3423
            Self::Grey       => (0.55, 0.55, 0.55),
            Self::DarkGreen  => (0.13, 0.35, 0.13),
            Self::LightGreen => (0.45, 0.72, 0.32),
        }
    }

    /// UI swatch colour (sRGB).
    pub fn ui_color(self) -> Color {
        let (r, g, b) = self.srgb();
        Color::srgb(r, g, b)
    }

    /// sRGB byte quadruple for writing directly into an `Rgba8UnormSrgb` paint
    /// texture. The paint texture is sampled by a `StandardMaterial` with a white
    /// `base_color`, and an Srgb-format texture is hardware-decoded to linear on
    /// sample — so the bytes we store must be the raw sRGB triple (×255), NOT the
    /// linear value (storing linear here would double-decode and darken).
    pub fn srgb_u8(self) -> [u8; 4] {
        let (r, g, b) = self.srgb();
        let to_u8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        [to_u8(r), to_u8(g), to_u8(b), 255]
    }

    /// LINEAR rgba for this colour. (The brush now stamps the sRGB byte triple
    /// directly into the `Rgba8UnormSrgb` paint texture via `srgb_u8`; this linear
    /// accessor is retained as a reference for any future linear-space consumer.)
    #[allow(dead_code)]
    pub fn linear_f32(self) -> [f32; 4] {
        LinearRgba::from(self.ui_color()).to_f32_array()
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Yellow     => "Yellow",
            Self::Brown      => "Brown",
            Self::Grey       => "Grey",
            Self::DarkGreen  => "Dark Green",
            Self::LightGreen => "Light Green",
        }
    }
}


/// The brush selector's backing enum. `Basic` hard-replaces texels under the
/// screen disc; `Soft` feathers the colour with a `softness`-controlled falloff
/// (and single-pass-cap accumulation). Modelled as an enum so further brushes can
/// be added without reworking the dropdown.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum MapBrush {
    #[default]
    Basic,
    Soft,
}

impl MapBrush {
    pub const ALL: [MapBrush; 2] = [Self::Basic, Self::Soft];

    pub fn label(self) -> &'static str {
        match self {
            Self::Basic => "Basic Brush",
            Self::Soft  => "Soft Brush",
        }
    }
}


/// Transient map-editor UI state. Mirrors the relevant fields of `SpeciesSession`.
#[derive(Resource)]
pub struct MapEditorSession {
    /// The currently-selected paint colour (none until a swatch is clicked).
    pub selected_material:   Option<MapMaterial>,
    /// The currently-selected brush.
    pub selected_brush:      MapBrush,
    /// Whether the brush dropdown is expanded.
    pub brush_dropdown_open: bool,
    /// Screen-space (viewport-pixel) brush radius. Stays visually constant on
    /// screen regardless of camera distance/zoom (derived from the live camera
    /// each stroke). Editable via the tool-panel "Brush size (px)" field.
    pub brush_radius_px:     f32,
    /// Soft-brush edge softness in `[0,1]`: 0 = hard edge (identical to `Basic`),
    /// 1 = fully feathered. Only consumed by `MapBrush::Soft`. Editable via the
    /// tool-panel "Softness" field.
    pub softness:            f32,
}

impl Default for MapEditorSession {
    fn default() -> Self {
        Self {
            selected_material:   None,
            selected_brush:      MapBrush::default(),
            brush_dropdown_open: false,
            brush_radius_px:     crate::simulation_settings::DEFAULT_BRUSH_RADIUS_PX,
            softness:            crate::simulation_settings::DEFAULT_BRUSH_SOFTNESS,
        }
    }
}
