// Per-organism data the editor manipulates.
//
// `OrganismTemplate` is the editor's working representation of an
// organism: just enough metadata for the user to pick its trophic
// strategy + intelligence + symmetry, plus the world position. The
// actual mesh / Bevy entity is kept in lock-step via a separate
// `template_entity` field that the placement system updates.

use bevy::prelude::*;

use crate::cell::CellType;
use crate::organism::{IntelligenceLevel, Symmetry};
use crate::body_part::MIN_X_BILATERAL;


/// Trophic strategy. Mirrors `OrganismKind` from the simulation but
/// kept as an editor-local enum so the UI can iterate variants
/// without pulling in the simulation's invariants. Convertible
/// back to `OrganismKind` for save-time emission.
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

    /// Trophic colour used for the editor preview mesh — green for
    /// photoautotrophs, red for heterotrophs.
    pub fn preview_colour(self) -> Color {
        match self {
            Metabolism::Photoautotroph => Color::srgb(0.20, 0.80, 0.20),
            Metabolism::Heterotroph    => Color::srgb(0.85, 0.20, 0.20),
        }
    }
}


/// Cycle helper for `IntelligenceLevel` — we re-export here so the
/// UI can treat it like the other editor enums without an extra
/// `match` site.
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


/// Body-plan form. Maps directly to `Organism::has_variable_form`:
///   * `Variable` ⇒ has_variable_form = true ⇒ sessile, plant-like.
///   * `Fixed`    ⇒ has_variable_form = false ⇒ mobile, animal-like.
///
/// The simulation enforces the invariant `Variable ⇒ NoSymmetry`;
/// the editor's creation panel mirrors that automatically (cycling
/// to Variable forces symmetry to NoSymmetry, and vice versa).
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


/// One organism the user has created in the editor.
///
/// Identified by a stable `id` (a monotonic counter) so the
/// inventory-panel UI can reference rows without entity churn.
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
}

impl OrganismTemplate {
    /// Display name used in the inventory panel and as a save-file
    /// fingerprint. Format: "Hetero #3", "Photo #5".
    pub fn display_name(&self) -> String {
        let prefix = match self.metabolism {
            Metabolism::Heterotroph    => "Hetero",
            Metabolism::Photoautotroph => "Photo",
        };
        format!("{prefix} #{}", self.id)
    }

    /// Maps directly to `Organism::has_variable_form` in the save
    /// format. Set by the user via the bottom-panel "Form" cycler;
    /// the simulation invariant `Variable ⇒ NoSymmetry` is enforced
    /// at the cycler level so this never disagrees with `symmetry`.
    pub fn has_variable_form(&self) -> bool {
        self.form.is_variable()
    }

    /// Sessile is conceptually paired with variable-form in this
    /// simulation: all variable-form organisms are sessile, and the
    /// editor doesn't expose sessile as an independent toggle.
    pub fn is_sessile(&self) -> bool {
        self.form.is_variable()
    }

    /// Build the per-body-part OCG for the save format. Single-cell
    /// for `NoSymmetry`; two cells at `(±MIN_X_BILATERAL, 0, 0)` for
    /// `Bilateral` (matching `colony.rs::spawn_colony`'s seed layout).
    pub fn build_ocg(&self) -> Vec<(usize, Vec3, CellType)> {
        let ct = self.metabolism.cell_type();
        match self.symmetry {
            Symmetry::NoSymmetry => vec![(0, Vec3::ZERO, ct)],
            Symmetry::Bilateral  => vec![
                (0, Vec3::new( MIN_X_BILATERAL, 0.0, 0.0), ct),
                (1, Vec3::new(-MIN_X_BILATERAL, 0.0, 0.0), ct),
            ],
        }
    }

    /// Bounding-sphere radius used for right-click hit-testing
    /// against the rendered mesh. Sized just generously enough to
    /// cover the visible rhombic-dodecahedron extent for each
    /// supported symmetry.
    ///
    /// One RD cell with `EDGE_LEN = 1.0` has long-radius
    /// `MIN_X_BILATERAL ≈ 1.155` from the cell centre. A bilateral
    /// pair sits two cell-centres apart along X, so its outermost
    /// point sits at `|center| + cell_radius ≈ 2 · MIN_X_BILATERAL`.
    pub fn pick_radius(&self) -> f32 {
        match self.symmetry {
            Symmetry::NoSymmetry => MIN_X_BILATERAL + 0.25,
            Symmetry::Bilateral  => 2.0 * MIN_X_BILATERAL + 0.25,
        }
    }
}
