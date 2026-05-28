// Species editor — central state.
//
// Holds the in-progress species definition: cycler values (metabolism,
// intelligence, symmetry, form), the cell list under construction, and
// transient interaction state (which cell type the user has currently
// selected from the bottom panel, whether the first cell has been
// spawned yet, and the entity handles for the current mesh + preview
// + bilateral-axis visuals).
//
// One ECS Resource — `SpeciesSession` — wraps everything. Its inner
// state machine has three phases:
//   1. Configuration: cyclers active, first cell not yet spawned.
//      `ocg` is empty. The "Spawn first Cell" button gates the
//      transition to phase 2.
//   2. Growth: at least one cell present. The user picks a cell type
//      from the bottom panel; the preview cell follows the cursor and
//      snaps to lattice frontier positions. Left-click commits a cell.
//   3. Save: the user clicks "Create Species" — opens a Save-As
//      dialog, then `save_requested` carries the chosen path to the
//      save system on the next Update tick.

use bevy::prelude::*;
use std::path::PathBuf;

use crate::cell::CellType;
use crate::colony::{IntelligenceLevel, Symmetry};


// ── Cycler enums ────────────────────────────────────────────────────────────
//
// Mirror the colony-editor cycler types but live in this module so the
// species editor isn't coupled to the colony editor (which would make
// removing or refactoring either one a chore). They cycle through the
// fixed enums in the wider codebase (`Symmetry`, `IntelligenceLevel`)
// and the two species-editor-local toggles (`Metabolism`, `Form`).

/// Trophic strategy. Determines the starter cell type (Photo /
/// NonPhoto) and the marker component on the final spawned organism.
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
    /// Cell type used for the very first cell when the user clicks
    /// "Spawn first Cell." Subsequent cells are picked freely from the
    /// bottom panel — the species can be mixed-type (e.g. a hetero
    /// body with Photo "spots").
    pub fn starter_cell_type(self) -> CellType {
        match self {
            Metabolism::Photoautotroph => CellType::Photo,
            Metabolism::Heterotroph    => CellType::NonPhoto,
        }
    }
}

/// Variable-form vs fixed-form body plan, mirroring the
/// `Organism::has_variable_form` boolean. Stored as an enum here so
/// the cycler UI can display a string instead of a checkbox.
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

/// Heterotroph sub-classification: Herbivore (eats photoautotrophs)
/// vs Carnivore (eats other heterotrophs). Photoautotroph species
/// keep this set to `Herbivore` at the type level but it's ignored
/// — the field is only meaningful when `metabolism == Heterotroph`.
///
/// Interactions:
///   * Switching to Herbivore → intelligence auto-sets to `Level1`
///     (the supervised herbivore_1 brain). User can manually
///     upgrade to Level 2 or 3 afterwards.
///   * Switching to Carnivore → intelligence auto-sets to `Level2`
///     (the smaller predator pool). User can upgrade to Level 3.
///     Levels 0 and 1 are NOT valid choices for carnivores.
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

/// Sessile vs mobile toggle, mirroring `Organism::is_sessile`. Exposed
/// as its own cycler because the user wants explicit control AND
/// because the choice has a documented interaction with the
/// intelligence cycler:
///   * Sessile → `IntelligenceLevel::Level0` (auto-set, locked).
///   * Mobile  → auto-set to `Level1` on toggle, but freely
///                cycleable to higher levels afterward.
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

#[derive(Clone, Copy, Debug)]
pub struct DraftSpecies {
    pub metabolism:     Metabolism,
    pub intelligence:   IntelligenceLevel,
    pub symmetry:       Symmetry,
    pub form:           Form,
    pub mobility:       Mobility,
    pub classification: Classification,
}

impl Default for DraftSpecies {
    fn default() -> Self {
        Self {
            metabolism:     Metabolism::Heterotroph,
            intelligence:   IntelligenceLevel::Level1,
            // Heterotrophs default to Bilateral symmetry (see
            // `default_symmetry_for`); photoautotrophs default to
            // NoSymmetry.
            symmetry:       Symmetry::Bilateral,
            form:           Form::Fixed,
            mobility:       Mobility::Mobile,
            classification: Classification::Herbivore,
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
}


// ── Session resource ────────────────────────────────────────────────────────

/// Root state for the species editor.
///
/// `body_parts[0]` is the base body; entries beyond it are appendages added
/// via the Body-part index panel. `active_body_part` selects which one new
/// cells are placed on.
#[derive(Resource, Default)]
pub struct SpeciesSession {
    pub draft: DraftSpecies,

    /// All body parts. Index 0 is the base body; later entries are
    /// appendages. Empty until "Spawn first Cell" seeds the base.
    pub body_parts: Vec<EditorBodyPart>,

    /// Index into `body_parts` of the part new cells are placed on.
    pub active_body_part: usize,

    /// Cell type selected in the bottom panel. `None` means no preview
    /// cell follows the cursor — left-click is ignored.
    pub selected_cell_type: Option<CellType>,

    /// `true` once the user has clicked "Spawn first Cell" (the base body
    /// has its seed). Locks the cyclers and reveals the
    /// bottom-panel + viewport-placement workflow.
    pub first_cell_spawned: bool,

    /// Save dispatch: when `Some(path)`, the next Update tick writes a
    /// `.species` binary file to `path` and clears this field.
    pub save_requested: Option<PathBuf>,

    /// `true` whenever the user has made changes that have not yet been
    /// saved. Set by every mutating action; cleared on successful
    /// `.species` write. Consumed by the Clear/New button.
    pub dirty: bool,

    /// Rising-edge flag toggled by the Clear/New button when the
    /// session is dirty. The clear-modal lifecycle system spawns the
    /// modal on `true` and despawns it on `false`.
    pub show_clear_modal: bool,

    /// Index of the body part currently being renamed in the Body-part
    /// index panel, if any. While `Some`, keystrokes edit `rename_buffer`
    /// instead of doing anything else.
    pub renaming_body_part: Option<usize>,

    /// In-progress rename text. Committed to the part's `name` on Enter,
    /// discarded on Escape.
    pub rename_buffer: String,
}

impl SpeciesSession {
    /// Reset to a fresh state. Called by the Clear/New flow once the
    /// user confirms (or unconditionally, if there were no unsaved
    /// changes); `Default` covers the spawn-time case.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

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
