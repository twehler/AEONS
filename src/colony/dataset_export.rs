// CSV dataset export.
//
// One-shot worker driven by the `ExportDatasetRequested` resource.
// When the user clicks "Export Simulation Dataset" in the statistics
// panel, that handler pauses the simulation, opens a native save
// dialog, and writes the chosen path into the resource. The system
// below picks the path up on the next Update tick, snapshots every
// living organism, and writes a `;`-separated CSV file.
//
// Row order: Carnivores → Herbivores → Photoautotrophs. Within each
// group, organisms are emitted in whatever order the Bevy query
// yields them (effectively archetype-then-spawn order).
//
// Columns: identity + transform + every scalar/enum field of
// `Organism` (the `body_parts` Vec is summarised by two derived
// counts; the raw nested structure would not fit in a flat CSV
// cleanly), followed by all 19 DNA slots emitted as individual
// `dna_<name>` columns — the DNA vector itself never appears as one
// opaque column.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use bevy::prelude::*;

use crate::colony::{Carnivore, Heterotroph, Organism, OrganismRoot, Photoautotroph};
use crate::lineages::dna::{DNA_DIM, DNA_FIELD_NAMES};
use crate::organism::{IntelligenceLevel, Symmetry};


/// Filled by the statistics-panel handler on a click. Consumed by
/// `export_dataset_system` on the next Update tick. `None` means
/// "nothing to do".
#[derive(Resource, Default)]
pub struct ExportDatasetRequested(pub Option<PathBuf>);


/// Snapshot every live organism into a `;`-separated CSV at the
/// requested path. The path is `.take()`-d so a single click produces
/// a single write — subsequent ticks find the resource empty and
/// skip. Errors are logged at `error!` and silently swallowed
/// otherwise so a file-system failure doesn't disrupt the simulation.
pub fn export_dataset_system(
    mut req: ResMut<ExportDatasetRequested>,
    query:   Query<
        (
            Entity, &Organism, &Transform,
            Has<Photoautotroph>, Has<Heterotroph>, Has<Carnivore>,
        ),
        With<OrganismRoot>,
    >,
) {
    let Some(path) = req.0.take() else { return };

    let file = match File::create(&path) {
        Ok(f) => f,
        Err(e) => {
            error!("export-dataset: failed to create {}: {}", path.display(), e);
            return;
        }
    };
    let mut out = BufWriter::new(file);

    if let Err(e) = write_csv(&mut out, &query) {
        error!("export-dataset: write failure on {}: {}", path.display(), e);
        return;
    }

    if let Err(e) = out.flush() {
        error!("export-dataset: flush failure on {}: {}", path.display(), e);
        return;
    }

    info!("exported simulation dataset to {}", path.display());
}


/// Trophic class used for both row-ordering and the `trophic_class`
/// column. Determined from the entity's marker components, not from
/// `Organism` content (cell counts can drift mid-consumption).
#[derive(Clone, Copy, PartialEq, Eq)]
enum TrophicClass {
    Carnivore,
    Herbivore,
    Photoautotroph,
}

impl TrophicClass {
    fn label(self) -> &'static str {
        match self {
            TrophicClass::Carnivore      => "Carnivore",
            TrophicClass::Herbivore      => "Herbivore",
            TrophicClass::Photoautotroph => "Photoautotroph",
        }
    }

    /// Sort key — lower = earlier row.
    fn order(self) -> u8 {
        match self {
            TrophicClass::Carnivore      => 0,
            TrophicClass::Herbivore      => 1,
            TrophicClass::Photoautotroph => 2,
        }
    }
}


fn classify(is_photo: bool, is_hetero: bool, is_carn: bool) -> Option<TrophicClass> {
    if is_carn  && is_hetero { return Some(TrophicClass::Carnivore); }
    if is_hetero             { return Some(TrophicClass::Herbivore); }
    if is_photo              { return Some(TrophicClass::Photoautotroph); }
    None
}


fn write_csv<W: Write>(
    out:   &mut W,
    query: &Query<
        (
            Entity, &Organism, &Transform,
            Has<Photoautotroph>, Has<Heterotroph>, Has<Carnivore>,
        ),
        With<OrganismRoot>,
    >,
) -> std::io::Result<()> {
    // ── Header ────────────────────────────────────────────────
    let mut header = String::new();
    let cols = [
        // Identity / context
        "entity_id", "trophic_class", "species_id",
        "pos_x", "pos_y", "pos_z",
        // Body plan
        "symmetry", "intelligence_level", "is_sessile",
        "has_variable_form", "adult",
        "photo_cell_count", "non_photo_cell_count",
        "grown_cell_count", "alive_body_part_count",
        // Energetics / RL signals
        "energy", "in_sunlight",
        "reproduced", "reproductions", "predations",
        "hunger", "dopamine", "target_distance",
        // Movement
        "movement_speed",
        "movement_dir_x", "movement_dir_y", "movement_dir_z",
        "velocity_x",     "velocity_y",     "velocity_z",
        "is_climbing", "climb_energy_debt", "cached_bounding_radius",
    ];
    for (i, c) in cols.iter().enumerate() {
        if i > 0 { header.push(';'); }
        header.push_str(c);
    }
    for name in DNA_FIELD_NAMES.iter() {
        header.push(';');
        header.push_str("dna_");
        header.push_str(name);
    }
    header.push('\n');
    out.write_all(header.as_bytes())?;

    // ── Collect + sort rows ───────────────────────────────────
    // Bevy queries iterate archetype-order which mixes the
    // trophic classes; collect with the (TrophicClass, entity)
    // sort key to honour the user-requested ordering
    // (carnivores → herbivores → photoautotrophs).
    let mut rows: Vec<(TrophicClass, Entity, &Organism, &Transform)> = query.iter()
        .filter_map(|(e, org, tf, is_photo, is_hetero, is_carn)| {
            classify(is_photo, is_hetero, is_carn).map(|tc| (tc, e, org, tf))
        })
        .collect();
    rows.sort_by_key(|(tc, e, _, _)| (tc.order(), e.index()));

    // ── Per-row write ─────────────────────────────────────────
    for (tc, entity, organism, transform) in rows {
        write_row(out, tc, entity, organism, transform)?;
    }

    Ok(())
}


fn write_row<W: Write>(
    out:       &mut W,
    tc:        TrophicClass,
    entity:    Entity,
    organism:  &Organism,
    transform: &Transform,
) -> std::io::Result<()> {
    let pos = transform.translation;

    // Build the row as a single owned String so we can write it in
    // one syscall.  Per-cell precision is .6 (matches the .colony
    // save format) so the file round-trips with negligible loss.
    let mut row = String::with_capacity(512);

    macro_rules! push {
        ($($t:tt)*) => {{ use std::fmt::Write as _; let _ = write!(row, $($t)*); }};
    }
    macro_rules! sep { () => { row.push(';'); } }

    // Identity
    push!("{}", entity.index());
    sep!(); push!("{}", tc.label());
    sep!(); match organism.species_id {
        Some(id) => push!("{}", id),
        None     => (),                  // empty cell on None
    };
    // Position
    sep!(); push!("{:.6}", pos.x);
    sep!(); push!("{:.6}", pos.y);
    sep!(); push!("{:.6}", pos.z);

    // Body plan
    sep!(); push!("{}", symmetry_label(organism.symmetry));
    sep!(); push!("{}", intelligence_label(organism.intelligence_level));
    sep!(); push!("{}", bool01(organism.is_sessile));
    sep!(); push!("{}", bool01(organism.has_variable_form));
    sep!(); push!("{}", bool01(organism.adult));
    sep!(); push!("{}", organism.photo_cell_count);
    sep!(); push!("{}", organism.non_photo_cell_count);
    sep!(); push!("{}", organism.grown_cell_count());
    sep!(); push!("{}", organism.alive_body_part_count());

    // Energetics / RL signals
    sep!(); push!("{:.6}", organism.energy);
    sep!(); push!("{}", bool01(organism.in_sunlight));
    sep!(); push!("{}", bool01(organism.reproduced));
    sep!(); push!("{}", organism.reproductions);
    sep!(); push!("{}", organism.predations);
    sep!(); push!("{:.6}", organism.hunger);
    sep!(); push!("{:.6}", organism.dopamine);
    sep!(); push!("{:.6}", organism.target_distance);

    // Movement
    sep!(); push!("{:.6}", organism.movement_speed);
    sep!(); push!("{:.6}", organism.movement_direction.x);
    sep!(); push!("{:.6}", organism.movement_direction.y);
    sep!(); push!("{:.6}", organism.movement_direction.z);
    sep!(); push!("{:.6}", organism.velocity.x);
    sep!(); push!("{:.6}", organism.velocity.y);
    sep!(); push!("{:.6}", organism.velocity.z);
    sep!(); push!("{}", bool01(organism.is_climbing));
    sep!(); push!("{:.6}", organism.climb_energy_debt);
    sep!(); push!("{:.6}", organism.cached_bounding_radius);

    // DNA — exactly DNA_DIM slots, matching DNA_FIELD_NAMES order.
    // Defend against a corrupt / partially-populated dna vector by
    // emitting empty cells past the available length.
    for i in 0..DNA_DIM {
        sep!();
        if let Some(v) = organism.dna.get(i) {
            push!("{:.6}", v);
        }
    }

    row.push('\n');
    out.write_all(row.as_bytes())
}


#[inline]
fn bool01(b: bool) -> u8 { if b { 1 } else { 0 } }

fn symmetry_label(s: Symmetry) -> &'static str {
    match s {
        Symmetry::NoSymmetry => "NoSymmetry",
        Symmetry::Bilateral  => "Bilateral",
    }
}

fn intelligence_label(il: IntelligenceLevel) -> &'static str {
    match il {
        IntelligenceLevel::Level0 => "Level0",
        IntelligenceLevel::Level1 => "Level1",
        IntelligenceLevel::Level2 => "Level2",
        IntelligenceLevel::Level3 => "Level3",
    }
}
