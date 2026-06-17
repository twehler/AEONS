// Per-organism data the editor manipulates.
//
// `OrganismTemplate` is the editor's working representation: trophic
// strategy + intelligence + symmetry + world position. Its mesh entity
// is kept in lock-step via the `entity` field.

use bevy::prelude::*;

use crate::cell::{BodyPartKind, CellType};
use crate::organism::{IntelligenceLevel, MovementMode, Symmetry};
use crate::body_part::MIN_X_BILATERAL;


/// Trophic strategy. Editor-local mirror of `OrganismKind`, convertible
/// back for save-time emission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metabolism {
    Heterotroph,
    Photoautotroph,
}

impl Metabolism {
    pub fn label(self) -> &'static str {
        match self {
            Metabolism::Heterotroph    => "Heterotroph",
            Metabolism::Photoautotroph => "Photoautotroph",
        }
    }

    pub fn cycle(self) -> Self {
        match self {
            Metabolism::Heterotroph    => Metabolism::Photoautotroph,
            Metabolism::Photoautotroph => Metabolism::Heterotroph,
        }
    }

    pub fn cell_type(self) -> CellType {
        match self {
            Metabolism::Heterotroph    => CellType::NonPhoto,
            Metabolism::Photoautotroph => CellType::Photo,
        }
    }

    /// Preview-mesh colour: green photoautotroph, red heterotroph.
    pub fn preview_colour(self) -> Color {
        match self {
            Metabolism::Photoautotroph => Color::srgb(0.20, 0.80, 0.20),
            Metabolism::Heterotroph    => Color::srgb(0.85, 0.20, 0.20),
        }
    }
}


/// Label helper for `IntelligenceLevel`.
pub fn intel_label(level: IntelligenceLevel) -> &'static str {
    match level {
        IntelligenceLevel::Level0 => "Intel 0 (sessile)",
        IntelligenceLevel::Level1 => "Intel 1",
        IntelligenceLevel::Level2 => "Intel 2",
        IntelligenceLevel::Level3 => "Intel 3",
    }
}

pub fn intel_cycle(level: IntelligenceLevel) -> IntelligenceLevel {
    match level {
        IntelligenceLevel::Level0 => IntelligenceLevel::Level1,
        IntelligenceLevel::Level1 => IntelligenceLevel::Level2,
        IntelligenceLevel::Level2 => IntelligenceLevel::Level3,
        IntelligenceLevel::Level3 => IntelligenceLevel::Level0,
    }
}


/// Cycle helper for `Symmetry`.
pub fn sym_label(sym: Symmetry) -> &'static str {
    match sym {
        Symmetry::NoSymmetry => "No Symmetry",
        Symmetry::Bilateral  => "Bilateral",
    }
}

pub fn sym_cycle(sym: Symmetry) -> Symmetry {
    match sym {
        Symmetry::NoSymmetry => Symmetry::Bilateral,
        Symmetry::Bilateral  => Symmetry::NoSymmetry,
    }
}


/// Body-plan form → `Organism::has_variable_form`. `Variable` ⇒ sessile;
/// `Fixed` ⇒ mobile. Invariant `Variable ⇒ NoSymmetry` is enforced by
/// the simulation and mirrored by the creation panel's cyclers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    Variable,
    Fixed,
}

impl Form {
    pub fn is_variable(self) -> bool {
        matches!(self, Form::Variable)
    }
}

pub fn form_label(form: Form) -> &'static str {
    match form {
        Form::Variable => "Sessile / Variable Form",
        Form::Fixed    => "Mobile / Fixed Form",
    }
}

pub fn form_cycle(form: Form) -> Form {
    match form {
        Form::Variable => Form::Fixed,
        Form::Fixed    => Form::Variable,
    }
}


/// One created organism, keyed by a stable monotonic `id` so the
/// inventory UI can reference rows without entity churn.
#[derive(Clone, Debug)]
pub struct OrganismTemplate {
    pub id:           u32,
    pub metabolism:   Metabolism,
    pub intelligence: IntelligenceLevel,
    pub symmetry:     Symmetry,
    pub form:         Form,
    pub position:     Vec3,
    /// Bevy entity holding the visual marker in the 3D world. Kept
    /// in lock-step with `position` by the placement system.
    pub entity:       Entity,
    /// If `Some`, this template is `.species`-sourced and carries the
    /// full OCG (bilateral-expanded if applicable); `build_ocg` returns
    /// it verbatim.
    pub custom_ocg:    Option<Vec<(usize, Vec3, CellType)>>,
    /// Appendage parts as `(OCG, kind, parent)`. OCG is the raw stored shape
    /// (right-half for bilateral); `kind` is the `BodyPartKind` (Limb/Segment/
    /// Static/Organ — all rebase to a first-cell pivot); `parent` is the EDITOR
    /// parent index (0 = main body). A Bilateral `Limb`/`Organ` expands to a
    /// mirrored pair; `Segment`/`Static` fuse to one midline part. Empty for
    /// single-part templates.
    pub custom_appendages: Vec<(Vec<(usize, Vec3, CellType)>, BodyPartKind, usize)>,
    /// Display name (species filename stem); `None` ⇒ "Hetero/Photo #N".
    pub species_name:  Option<String>,
    /// Carnivore flag. When `true`, spawn attaches a `Carnivore` marker
    /// so IL2/IL3 brains hunt heterotrophs instead of photoautotrophs.
    pub is_carnivore:  bool,
    /// Maps to `Organism::movement_mode`. Defaults `MovementMode::Sliding`
    /// for cycler templates; `.species`-sourced templates carry the loaded mode.
    pub movement_mode: MovementMode,
    /// Maps to `Organism::is_sessile` (species-editor Mobility cycler).
    /// `spawn_organism` independently coerces `has_variable_form ⇒ sessile`;
    /// the `Fixed + Sessile` pairing is only carryable through this field.
    pub is_sessile: bool,
    /// Maps to `Organism::ground_based`. Only an override for phototrophs
    /// (floating, water-based algae); `spawn_organism` coerces heterotrophs
    /// to the movement-mode default.
    pub ground_based: bool,
    /// Trained brain carried from a `.species` import (or a loaded `.colony`),
    /// so an editor-saved colony PERSISTS the weights instead of fresh-init.
    /// `Some` ⇒ `save::write_organism` encodes the matching `.colony` brain
    /// block (sliding or limb/swim); `None` ⇒ both blocks written absent.
    pub brain: Option<crate::species_editor::save::LoadedBrain>,
}

impl OrganismTemplate {
    /// Inventory-panel display name: "Hetero #3" / "Photo #5", or
    /// "<species> #N" for species-loaded templates.
    pub fn display_name(&self) -> String {
        if let Some(ref name) = self.species_name {
            return format!("{name} #{}", self.id);
        }
        let prefix = match self.metabolism {
            Metabolism::Heterotroph    => "Hetero",
            Metabolism::Photoautotroph => "Photo",
        };
        format!("{prefix} #{}", self.id)
    }

    /// Maps to `Organism::has_variable_form`. Invariant `Variable ⇒
    /// NoSymmetry` is enforced at the cycler, so it never disagrees with `symmetry`.
    pub fn has_variable_form(&self) -> bool {
        self.form.is_variable()
    }

    /// Sessile flag. `is_sessile` and `has_variable_form` are independent
    /// in the species file (a fixed-form sessile organism is valid); the
    /// OR preserves the invariant `Variable ⇒ Sessile`.
    pub fn is_sessile(&self) -> bool {
        self.is_sessile || self.form.is_variable()
    }

    /// Build the per-body-part OCG. Returns `custom_ocg` verbatim if set,
    /// else the cycler shape: 1 cell (NoSymmetry) or a `±MIN_X_BILATERAL`
    /// pair (Bilateral).
    pub fn build_ocg(&self) -> Vec<(usize, Vec3, CellType)> {
        if let Some(ref ocg) = self.custom_ocg {
            return ocg.clone();
        }
        let ct = self.metabolism.cell_type();
        match self.symmetry {
            Symmetry::NoSymmetry => vec![(0, Vec3::ZERO, ct)],
            Symmetry::Bilateral  => vec![
                (0, Vec3::new( MIN_X_BILATERAL, 0.0, 0.0), ct),
                (1, Vec3::new(-MIN_X_BILATERAL, 0.0, 0.0), ct),
            ],
        }
    }

    /// Bounding-sphere radius for right-click hit-testing. One RD cell
    /// (`EDGE_LEN = 1.0`) has long-radius `MIN_X_BILATERAL ≈ 1.155`; a
    /// bilateral pair reaches `≈ 2·MIN_X_BILATERAL`.
    pub fn pick_radius(&self) -> f32 {
        // Species OCG: bounding sphere plus one cell of padding.
        if let Some(ref ocg) = self.custom_ocg {
            let max_r = ocg.iter()
                .map(|(_, p, _)| p.length())
                .fold(0.0_f32, |a, b| a.max(b));
            return max_r + MIN_X_BILATERAL + 0.25;
        }
        match self.symmetry {
            Symmetry::NoSymmetry => MIN_X_BILATERAL + 0.25,
            Symmetry::Bilateral  => 2.0 * MIN_X_BILATERAL + 0.25,
        }
    }
}
