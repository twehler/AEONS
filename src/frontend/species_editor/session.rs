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

/// Movement paradigm. Maps directly to `Organism::sliding_movement`:
///   * `Sliding`     → brain writes velocity, root translates kinematically.
///   * `LimbMovement` → Avian physics per body part; PPO brain outputs
///                      PD target joint angles. Limbs push the body via
///                      friction at the ground.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpeciesMovement { Sliding, LimbMovement }

impl SpeciesMovement {
    pub fn cycle(self) -> Self {
        match self {
            SpeciesMovement::Sliding      => SpeciesMovement::LimbMovement,
            SpeciesMovement::LimbMovement => SpeciesMovement::Sliding,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            SpeciesMovement::Sliding      => "Sliding",
            SpeciesMovement::LimbMovement => "Limb-Movement",
        }
    }
    /// Inverse of `Organism::sliding_movement`'s naming.
    pub fn is_sliding(self) -> bool { matches!(self, SpeciesMovement::Sliding) }
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
    pub movement:       SpeciesMovement,
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
            // Default to Sliding so new species inherit current
            // behaviour. The toggle is the user's deliberate opt-in
            // to physics + PPO locomotion.
            movement:       SpeciesMovement::Sliding,
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
    /// `true` when the user marked this body part as a "Limb" in the
    /// body-part index panel. Limbs spawn with `BodyPartKind::Limb` and
    /// visually rotate around their FIRST cell (the attachment seed)
    /// in the running simulation. The base body (index 0) is never a
    /// limb. Persisted to the `.species` file (v05+).
    pub is_limb: bool,
    /// Index (into `body_parts`) of the body part this one attaches to.
    /// `0` = the main body (a normal limb). A value pointing at another
    /// LIMB makes this a first-grade **sub-limb** (a 2-DOF leg segment):
    /// it gets a hinge joint to its parent limb exactly like a limb does
    /// to the main body. Set at creation time from the active body part.
    /// The base body (index 0) has `parent = 0` (self / ignored).
    /// Persisted to the `.species` file (v07+).
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

    /// All body parts. Index 0 is the base body; later entries are
    /// appendages. The base is auto-seeded with one cell on session
    /// creation / reset (see `seed_base`), so this is never empty —
    /// there is no longer a "spawn first cell" step.
    pub body_parts: Vec<EditorBodyPart>,

    /// Index into `body_parts` of the part new cells are placed on.
    pub active_body_part: usize,

    /// Cell type selected in the bottom panel. `None` means no preview
    /// cell follows the cursor — left-click is ignored.
    pub selected_cell_type: Option<CellType>,

    /// Always `true` now that the base cell is auto-seeded — kept so the
    /// bottom-panel / viewport-placement / preview systems that gate on
    /// "growth mode" keep working unchanged. Cycler locking is driven by
    /// `has_appended_cells()` instead.
    pub first_cell_spawned: bool,

    /// Save dispatch: when `Some(path)`, the next Update tick writes a
    /// `.species` binary file to `path` and clears this field.
    pub save_requested: Option<PathBuf>,

    /// `true` whenever the user has made changes that have not yet been
    /// saved. Set by every mutating action (cell placement, cycler
    /// changes, renames); cleared on successful `.species` write and on
    /// load. Consumed by the Clear/New button and the Load confirmation.
    pub dirty: bool,

    /// Rising-edge flag toggled by the Clear/New button when the
    /// session is dirty. The clear-modal lifecycle system spawns the
    /// modal on `true` and despawns it on `false`.
    pub show_clear_modal: bool,

    /// Rising-edge flag toggled by the "Load Species" button when the
    /// session is dirty. The load-modal lifecycle system spawns the
    /// confirmation on `true` and despawns it on `false`; "Yes" then
    /// proceeds with the file dialog + load.
    pub show_load_modal: bool,

    /// `true` while "Cell-Deletion Mode" is active: hovered cells are
    /// highlighted deep-red and a left-click deletes the hovered cell.
    /// Placement (preview + click-to-place) is suppressed while active.
    pub deletion_mode: bool,

    /// Index of the body part currently being renamed in the Body-part
    /// index panel, if any. While `Some`, keystrokes edit `rename_buffer`
    /// instead of doing anything else.
    pub renaming_body_part: Option<usize>,

    /// In-progress rename text. Committed to the part's `name` on Enter,
    /// discarded on Escape.
    pub rename_buffer: String,
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
            deletion_mode:      false,
            renaming_body_part: None,
            rename_buffer:      String::new(),
        };
        s.seed_base();
        s
    }
}

impl SpeciesSession {
    /// Reset to a fresh state (a single auto-seeded base cell). Called by
    /// the Clear/New flow once the user confirms (or unconditionally, if
    /// there were no unsaved changes); `Default` covers the spawn-time case.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// (Re)create the base body with a single seed cell, coloured by the
    /// current metabolism (Photo → green, Heterotroph → red) and positioned
    /// for the current symmetry (origin for NoSymmetry, the right-half seed
    /// `(MIN_X_BILATERAL, 0, 0)` for Bilateral). Called on session
    /// creation/reset and whenever the metabolism or symmetry cycler is
    /// flipped *before any cell has been appended* — so the base preview
    /// tracks those two options live, but only while it's still just the base.
    pub fn seed_base(&mut self) {
        let starter = self.draft.metabolism.starter_cell_type();
        let pos = match self.draft.symmetry {
            Symmetry::NoSymmetry => Vec3::ZERO,
            Symmetry::Bilateral  => Vec3::new(crate::body_part::MIN_X_BILATERAL, 0.0, 0.0),
        };
        self.body_parts = vec![EditorBodyPart {
            name:    "Base Body".to_string(),
            ocg:     vec![(0usize, pos, starter)],
            is_limb: false,
            parent:  0,
        }];
        self.active_body_part   = 0;
        self.first_cell_spawned = true;
    }

    /// `true` once the user has appended any cell beyond the single base
    /// seed (extra cells on the base, or any appendage). Locks the
    /// Metabolism + Symmetry cyclers, since both reinterpret / recolour the
    /// base in ways that can't be reconciled with an already-built body.
    pub fn has_appended_cells(&self) -> bool {
        self.body_parts.iter().map(|p| p.ocg.len()).sum::<usize>() > 1
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
