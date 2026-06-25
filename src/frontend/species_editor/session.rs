// Species editor — central state. One `SpeciesSession` Resource holds the
// in-progress species: cycler values, the body parts under construction, and
// transient interaction state.

use bevy::prelude::*;
use std::path::PathBuf;

use crate::cell::{BodyPartKind, CellType};
use crate::colony::{IntelligenceLevel, Symmetry};
use crate::organism::MovementMode;


// ── Cycler enums ────────────────────────────────────────────────────────────
//
// Local to this module (not shared with the colony editor) to avoid coupling
// the two editors.

/// Trophic strategy. Determines the starter cell type (Photo /
/// DigestionCell) and the marker component on the final spawned organism.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metabolism { Photoautotroph, Heterotroph }

impl Metabolism {
    pub fn cycle(self) -> Self {
        match self {
            Metabolism::Photoautotroph => Metabolism::Heterotroph,
            Metabolism::Heterotroph    => Metabolism::Photoautotroph,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Metabolism::Photoautotroph => "Photoautotroph",
            Metabolism::Heterotroph    => "Heterotroph",
        }
    }
    /// Cell type for the auto-seeded base cell. Subsequent cells are picked
    /// freely from the bottom panel, so a species can be mixed-type.
    pub fn starter_cell_type(self) -> CellType {
        match self {
            Metabolism::Photoautotroph => CellType::Photo,
            Metabolism::Heterotroph    => CellType::DigestionCell,
        }
    }
}

/// Variable-form vs fixed-form body plan, mirroring `Organism::has_variable_form`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form { Variable, Fixed }

impl Form {
    pub fn cycle(self) -> Self {
        match self {
            Form::Variable => Form::Fixed,
            Form::Fixed    => Form::Variable,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Form::Variable => "Variable",
            Form::Fixed    => "Fixed",
        }
    }
    pub fn as_bool(self) -> bool { matches!(self, Form::Variable) }
}

/// Heterotroph sub-classification (only meaningful when
/// `metabolism == Heterotroph`): Herbivore eats photoautotrophs, Carnivore eats
/// other heterotrophs. Interaction with the intelligence cycler: Herbivore
/// auto-sets `Level1`; Carnivore auto-sets `Level2` (Levels 0/1 invalid for
/// carnivores). User can upgrade afterwards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Classification { Herbivore, Carnivore }

impl Classification {
    pub fn cycle(self) -> Self {
        match self {
            Classification::Herbivore => Classification::Carnivore,
            Classification::Carnivore => Classification::Herbivore,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Classification::Herbivore => "Herbivore",
            Classification::Carnivore => "Carnivore",
        }
    }
}

/// Sessile vs mobile toggle, mirroring `Organism::is_sessile`. Interaction with
/// the intelligence cycler: Sessile → `Level0` (auto-set, locked); Mobile →
/// auto-set `Level1`, freely cycleable afterward.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mobility { Sessile, Mobile }

impl Mobility {
    pub fn cycle(self) -> Self {
        match self {
            Mobility::Sessile => Mobility::Mobile,
            Mobility::Mobile  => Mobility::Sessile,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Mobility::Sessile => "Sessile",
            Mobility::Mobile  => "Mobile",
        }
    }
    pub fn is_sessile(self) -> bool { matches!(self, Mobility::Sessile) }
}

/// Which editor interaction mode is active (chosen from the bottom-right
/// dropdown). `Addition` is the default — left-click places the selected cell.
/// `Deletion` highlights the hovered cell and left-click removes it.
/// `Diagnostics` highlights the hovered cell and reports its `CellType` in a
/// text field; it never mutates the body. Placement is only live in `Addition`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditorMode { Addition, Deletion, Diagnostics }

impl EditorMode {
    /// Every mode, in dropdown display order.
    pub const ALL: [EditorMode; 3] =
        [EditorMode::Addition, EditorMode::Deletion, EditorMode::Diagnostics];

    /// Full label for the dropdown option rows.
    pub fn label(self) -> &'static str {
        match self {
            EditorMode::Addition    => "Addition Mode",
            EditorMode::Deletion    => "Deletion Mode",
            EditorMode::Diagnostics => "Diagnostics Mode",
        }
    }

    /// Short label for the header button (`"Editor Mode: <short>"`).
    pub fn short_label(self) -> &'static str {
        match self {
            EditorMode::Addition    => "Addition",
            EditorMode::Deletion    => "Deletion",
            EditorMode::Diagnostics => "Diagnostics",
        }
    }
}

/// Species-editor camera-movement style, toggled from the tool panel.
/// `Free` flies the shared flycam (WASD + middle-hold look, same as the sim);
/// `Orbit` spins around the editor origin (middle-drag) with scroll-zoom.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CameraMode { Free, Orbit }

impl CameraMode {
    pub fn label(self) -> &'static str {
        match self {
            CameraMode::Free  => "Free",
            CameraMode::Orbit => "Orbit",
        }
    }
    pub fn toggle(self) -> Self {
        match self {
            CameraMode::Free  => CameraMode::Orbit,
            CameraMode::Orbit => CameraMode::Free,
        }
    }
}

// Movement paradigm uses the shared `crate::organism::MovementMode` directly
// (no editor-local enum). The author-able subset is cycled in `top_panel.rs`
// (Sliding → LimbBasedWalking → Swimming); `Flying` is a placeholder, not
// reachable from the cycler.

// Wrappers around the shared enums to keep cycle-label helpers local.
pub fn cycle_intelligence(level: IntelligenceLevel) -> IntelligenceLevel {
    match level {
        IntelligenceLevel::Level0 => IntelligenceLevel::Level1,
        IntelligenceLevel::Level1 => IntelligenceLevel::Level2,
        IntelligenceLevel::Level2 => IntelligenceLevel::Level3,
        IntelligenceLevel::Level3 => IntelligenceLevel::Level0,
    }
}
pub fn intelligence_label(level: IntelligenceLevel) -> &'static str {
    match level {
        IntelligenceLevel::Level0 => "Level 0",
        IntelligenceLevel::Level1 => "Level 1",
        IntelligenceLevel::Level2 => "Level 2",
        IntelligenceLevel::Level3 => "Level 3",
    }
}

pub fn cycle_symmetry(sym: Symmetry) -> Symmetry {
    match sym {
        Symmetry::NoSymmetry => Symmetry::Bilateral,
        Symmetry::Bilateral  => Symmetry::NoSymmetry,
    }
}
pub fn symmetry_label(sym: Symmetry) -> &'static str {
    match sym {
        Symmetry::NoSymmetry => "No Symmetry",
        Symmetry::Bilateral  => "Bilateral",
    }
}


// ── Draft species (cycler values) ───────────────────────────────────────────

/// Phototroph grounding mode — the three authorable states of the top-panel
/// Grounding cycler. Editable for PHOTOTROPHS (all three states) and for
/// ground-moving HETEROTROPHS (terrestrial ↔ ocean-floor — a benthic slider);
/// locked to water-based for swimmers. `OceanFloor` is benthic: it is
/// ground-anchored exactly like `GroundBased` (gravity holds it down, identical
/// runtime behaviour — it slides on / sinks to the terrain) but SPAWNS on
/// submerged terrain — the seafloor — instead
/// of dry land (see `colony::spawn_species_instance`). The runtime `Organism`
/// stores only the `ground_based` bool, so `OceanFloor` differs solely in spawn
/// PLACEMENT; its seafloor-ness is then captured by the spawned position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Grounding {
    GroundBased,
    WaterBased,
    OceanFloor,
}

impl Grounding {
    /// Cycle order for the editor button: Ground → Water → Ocean-Floor → …
    pub fn cycle(self) -> Self {
        match self {
            Grounding::GroundBased => Grounding::WaterBased,
            Grounding::WaterBased  => Grounding::OceanFloor,
            Grounding::OceanFloor  => Grounding::GroundBased,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Grounding::GroundBased => "Ground-Based",
            Grounding::WaterBased  => "Water-Based",
            Grounding::OceanFloor  => "Ocean-Floor",
        }
    }
    /// Ground-anchored? Both terrestrial and benthic (ocean-floor) are; only
    /// floating water-based is not. This is the runtime `Organism::ground_based`.
    pub fn is_ground_based(self) -> bool {
        matches!(self, Grounding::GroundBased | Grounding::OceanFloor)
    }
    /// Wants seafloor (submerged-terrain) spawn placement?
    pub fn is_ocean_floor(self) -> bool {
        matches!(self, Grounding::OceanFloor)
    }
    /// `.species` grounding byte. 0/1 PRESERVE the pre-v13 BOOL meaning
    /// (0 = water, 1 = ground) so older files load unchanged; 2 = ocean-floor (v13+).
    pub fn to_species_tag(self) -> u8 {
        match self {
            Grounding::WaterBased  => 0,
            Grounding::GroundBased => 1,
            Grounding::OceanFloor  => 2,
        }
    }
    /// Inverse of `to_species_tag` (the caller applies the movement-mode clamp).
    pub fn from_species_tag(b: u8) -> Self {
        match b {
            2 => Grounding::OceanFloor,
            1 => Grounding::GroundBased,
            _ => Grounding::WaterBased,
        }
    }
    /// Rebuild from runtime/template flags (no `OceanFloor` is recoverable from a
    /// live organism — it carries only the `ground_based` bool).
    pub fn from_flags(ground_based: bool, ocean_floor: bool) -> Self {
        if ocean_floor { Grounding::OceanFloor }
        else if ground_based { Grounding::GroundBased }
        else { Grounding::WaterBased }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DraftSpecies {
    pub metabolism:     Metabolism,
    pub intelligence:   IntelligenceLevel,
    pub symmetry:       Symmetry,
    pub form:           Form,
    pub mobility:       Mobility,
    pub classification: Classification,
    pub movement:       MovementMode,
    /// Phototroph grounding (the top-panel Grounding cycler): ground / water /
    /// ocean-floor. Only editable for PHOTOTROPHS; heterotrophs always derive it
    /// from `movement` — see `effective_grounding` / `effective_ground_based`.
    pub grounding:      Grounding,
}

impl DraftSpecies {
    /// The tri-state grounding this draft actually produces: phototrophs honour
    /// the Grounding cycler (clamped by the movement mode — a fluid mode can
    /// never be ground-anchored); heterotrophs derive ground-vs-water from the
    /// movement mode but may still choose ocean-floor (benthic) placement while
    /// ground-moving. Mirrors `spawn_organism`'s coercion so the saved
    /// `.species` byte equals the spawned organism's placement + behaviour.
    pub fn effective_grounding(&self) -> Grounding {
        match self.metabolism {
            Metabolism::Photoautotroph => {
                // A fluid movement mode forces water-based regardless of cycler.
                if self.grounding.is_ground_based() && !self.movement.default_ground_based() {
                    Grounding::WaterBased
                } else {
                    self.grounding
                }
            }
            Metabolism::Heterotroph => {
                // Ground-vs-water is movement-derived (a swimmer floats), but a
                // ground-moving heterotroph may still be a BENTHIC slider that
                // spawns on the ocean floor — honour that choice.
                if !self.movement.default_ground_based() {
                    Grounding::WaterBased
                } else if self.grounding.is_ocean_floor() {
                    Grounding::OceanFloor
                } else {
                    Grounding::GroundBased
                }
            }
        }
    }

    /// The `Organism::ground_based` bool this draft produces (ground-anchored?).
    pub fn effective_ground_based(&self) -> bool {
        self.effective_grounding().is_ground_based()
    }
}

impl Default for DraftSpecies {
    fn default() -> Self {
        Self {
            metabolism:     Metabolism::Heterotroph,
            intelligence:   IntelligenceLevel::Level1,
            symmetry:       Symmetry::Bilateral,
            form:           Form::Fixed,
            mobility:       Mobility::Mobile,
            classification: Classification::Herbivore,
            // Sliding is the opt-out default; limb movement is the user's
            // deliberate opt-in to physics + PPO locomotion.
            movement:       MovementMode::Sliding,
            // Ground-based is the default; phototroph drafts may opt into
            // water-based (floating algae) or ocean-floor (benthic) grounding.
            grounding:      Grounding::GroundBased,
        }
    }
}


// ── Editor body part ─────────────────────────────────────────────────────────

/// One body part under construction in the editor.
///
/// For Bilateral species `ocg` holds the RIGHT-half cells only (x ≥ 0); the
/// left half is generated by mirroring at render / save / spawn time. The
/// base body (index 0) becomes one welded runtime part; each appendage
/// (index ≥ 1) becomes a mirrored pair of runtime parts attached to the
/// base. For NoSymmetry `ocg` is the full part.
#[derive(Clone, Debug, PartialEq)]
pub struct EditorBodyPart {
    /// User-facing name shown / editable in the Body-part index panel.
    /// Persisted to the `.species` file.
    pub name: String,
    /// Right-half (Bilateral) or full (NoSymmetry) OCG, editor-local
    /// coords, sequential indices 0..N.
    pub ocg: Vec<(usize, Vec3, CellType)>,
    /// Part kind, set via the Body-part panel's "Kind" dropdown. Appendages are
    /// `Limb` (paired, splits in Bilateral), `Segment` (midline moving, fuses in
    /// Bilateral) or `Static` (midline, rigid fixed joint, no brain movement).
    /// All rotate/attach around their FIRST cell (the attachment seed). The base
    /// body (index 0) is always `Body` and ignores this field.
    pub kind: BodyPartKind,
    /// Index (into `body_parts`) of the part this one attaches to. `0` = main
    /// body. Pointing at another LIMB makes this a sub-limb. Base body has
    /// `parent = 0` (self / ignored).
    pub parent: usize,
}


// ── Session resource ────────────────────────────────────────────────────────

/// Root state for the species editor.
///
/// `body_parts[0]` is the base body; entries beyond it are appendages added
/// via the Body-part index panel. `active_body_part` selects which one new
/// cells are placed on.
#[derive(Resource)]
pub struct SpeciesSession {
    pub draft: DraftSpecies,

    /// All body parts. Index 0 is the base body (auto-seeded with one cell on
    /// creation/reset, so this is never empty); later entries are appendages.
    pub body_parts: Vec<EditorBodyPart>,

    /// Index into `body_parts` of the part new cells are placed on.
    pub active_body_part: usize,

    /// Cell type selected in the bottom panel. `None` means no preview
    /// cell follows the cursor — left-click is ignored.
    pub selected_cell_type: Option<CellType>,

    /// Always `true` now that the base cell is auto-seeded — kept so systems
    /// that gate on "growth mode" keep working. Cycler locking uses
    /// `has_appended_cells()` instead.
    pub first_cell_spawned: bool,

    /// Save dispatch: when `Some(path)`, the next Update tick writes a
    /// `.species` binary file to `path` and clears this field.
    pub save_requested: Option<PathBuf>,

    /// Unsaved-changes flag. Set by every mutating action; cleared on
    /// successful `.species` write and on load. Gates the Clear/New and Load
    /// confirmation modals.
    pub dirty: bool,

    /// Toggled by the Clear/New button when dirty; the modal lifecycle system
    /// spawns/despawns the modal on this.
    pub show_clear_modal: bool,

    /// Toggled by the "Load Species" button when dirty; the modal lifecycle
    /// system spawns/despawns the confirmation, "Yes" proceeds with file dialog
    /// + load.
    pub show_load_modal: bool,

    /// Active editor interaction mode (bottom-right dropdown). `Addition` =
    /// place cells; `Deletion` = click to remove the hovered cell; `Diagnostics`
    /// = inspect the hovered cell's type. Placement is suppressed outside
    /// `Addition`. See [`EditorMode`].
    pub editor_mode: EditorMode,

    /// `true` while the editor-mode dropdown is expanded. Toggled by its header
    /// button; the dropdown lifecycle system shows/hides the option rows.
    pub mode_dropdown_open: bool,

    /// Body part currently being renamed, if any. While `Some`, keystrokes edit
    /// `rename_buffer` instead of anything else.
    pub renaming_body_part: Option<usize>,

    /// In-progress rename text. Committed to the part's `name` on Enter,
    /// discarded on Escape.
    pub rename_buffer: String,

    /// Body-part row whose "Kind" dropdown is currently open, if any. The
    /// dropdown overlay lifecycle (`body_part_panel`) keys off this.
    pub kind_dropdown_open: Option<usize>,

    /// Camera-movement style (tool-panel toggle). See [`CameraMode`].
    pub camera_mode: CameraMode,
}

impl Default for SpeciesSession {
    fn default() -> Self {
        let mut s = Self {
            draft:              DraftSpecies::default(),
            body_parts:         Vec::new(),
            active_body_part:   0,
            selected_cell_type: None,
            first_cell_spawned: true,
            save_requested:     None,
            dirty:              false,
            show_clear_modal:   false,
            show_load_modal:    false,
            editor_mode:        EditorMode::Addition,
            mode_dropdown_open: false,
            renaming_body_part: None,
            rename_buffer:      String::new(),
            kind_dropdown_open: None,
            camera_mode:        CameraMode::Free,
        };
        s.seed_base();
        s
    }
}

impl SpeciesSession {
    /// Reset to a fresh state (a single auto-seeded base cell).
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// (Re)create the base body with a single seed cell, typed by metabolism and
    /// positioned for symmetry (origin for NoSymmetry, `(MIN_X_BILATERAL,0,0)`
    /// for Bilateral). Re-run when metabolism/symmetry flips *before any cell is
    /// appended*, so the base preview tracks those options live.
    pub fn seed_base(&mut self) {
        let starter = self.draft.metabolism.starter_cell_type();
        let pos = match self.draft.symmetry {
            Symmetry::NoSymmetry => Vec3::ZERO,
            Symmetry::Bilateral  => Vec3::new(crate::body_part::MIN_X_BILATERAL, 0.0, 0.0),
        };
        self.body_parts = vec![EditorBodyPart {
            name:   "Base Body".to_string(),
            ocg:    vec![(0usize, pos, starter)],
            kind:   BodyPartKind::Body,
            parent: 0,
        }];
        self.active_body_part   = 0;
        self.first_cell_spawned = true;
    }

    /// `true` once any cell exists beyond the single base seed. Locks the
    /// Metabolism + Symmetry cyclers, which can't be reconciled with an
    /// already-built body.
    pub fn has_appended_cells(&self) -> bool {
        self.body_parts.iter().map(|p| p.ocg.len()).sum::<usize>() > 1
    }

    /// Editor-mode predicates (see [`EditorMode`]).
    pub fn is_addition(&self)    -> bool { self.editor_mode == EditorMode::Addition }
    pub fn is_deletion(&self)    -> bool { self.editor_mode == EditorMode::Deletion }
    pub fn is_diagnostics(&self) -> bool { self.editor_mode == EditorMode::Diagnostics }

    /// The body part new cells are placed on, if any exists.
    pub fn active_part(&self) -> Option<&EditorBodyPart> {
        self.body_parts.get(self.active_body_part)
    }

    /// The base body part (index 0), if it exists.
    pub fn base_part(&self) -> Option<&EditorBodyPart> {
        self.body_parts.first()
    }

    /// `true` if the active part is an appendage (not the base body).
    pub fn active_is_appendage(&self) -> bool {
        self.active_body_part != 0
    }

    /// Combined OCG of one part for mesh rendering: for Bilateral this
    /// is right-half + mirrored-left (re-numbered sequentially), as
    /// `build_mesh_from_ocg` expects. For NoSymmetry it's the part's OCG
    /// unchanged.
    pub fn combined_ocg(&self, ocg: &[(usize, Vec3, CellType)]) -> Vec<(usize, Vec3, CellType)> {
        match self.draft.symmetry {
            Symmetry::NoSymmetry => ocg.to_vec(),
            Symmetry::Bilateral => {
                let left = crate::body_part::mirror_right_to_left(ocg);
                let mut combined: Vec<(usize, Vec3, CellType)> =
                    ocg.iter().chain(left.iter()).copied().collect();
                for (i, entry) in combined.iter_mut().enumerate() {
                    entry.0 = i;
                }
                combined
            }
        }
    }
}
