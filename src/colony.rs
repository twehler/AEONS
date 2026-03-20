use crate::cell::*;
use bevy::prelude::*;
use std::collections::HashMap;

pub struct ColonyPlugin;

impl Plugin for ColonyPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_colony);
    }
}


pub struct Colony {
    organisms: Vec<Organism>,
}


// The occg vector (short for "order of cell collection growth") dictates which cell-collections
// grow in which order. Vector of strings representing cell-collections; multiple collections
// can be represented in one string, separated by '|'.

pub struct Organism {
    collections: Vec<CellCollection>,
    pos: Vec3,
    orientation: Vec3,
    ocg: Vec<CellType>, // Order of cell growth — CellType instead of Cell so we can Copy/Clone freely
}


// The ocg vector (short for "order of cell growth") dictates the order in which individual
// cells will be appended to the organism across all cell-collections.
// It also dictates whether certain cell-collections grow independently or sequentially.

pub struct CellCollection {
    // Maps CellType → relative position offset from starter_cell_position.
    // Using CellType as the key (cheap Copy) avoids the double-ownership problem
    // that arises when the same Cell value is also placed in the ocg vec.
    cells: HashMap<CellType, [f32; 3]>,

    // Position of this collection's starter cell relative to the organism's prime cell.
    // Acts as the rotation base for the whole collection.
    starter_cell_position: Vec3,
}

impl CellCollection {
    /// Look up the world-space position of a cell type within this collection,
    /// given the organism's world position.
    fn world_pos(&self, cell_type: &CellType, organism_pos: Vec3) -> Option<Vec3> {
        let rel = self.cells.get(cell_type)?;
        let collection_origin = organism_pos + self.starter_cell_position;
        Some(collection_origin + Vec3::from(*rel))
    }
}


fn spawn_colony(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // ── Define the cell collection ───────────────────────────────────────────
    // Keys are CellType (Copy), values are [x,y,z] offsets relative to the
    // collection's starter cell.
    let o1_coll_1 = CellCollection {
        cells: HashMap::from([
            (CellType::RedCell,       [0.0,  0.0,  1.0]),
            (CellType::YellowCell,    [0.0,  0.0, -1.0]),
            (CellType::GreenCell,     [0.0, -1.0,  0.0]),
            (CellType::OrangeCell,    [0.0,  1.0,  0.0]),
            (CellType::LightBlueCell, [1.0,  0.0,  1.0]),
            (CellType::BlueCell,      [1.0,  1.0,  1.0]),
        ]),
        starter_cell_position: Vec3::ZERO,
    };

    // ── Define the organism ──────────────────────────────────────────────────
    // ocg lists CellType values in the order cells should be spawned.
    // Duplicate types are allowed if the same cell type appears multiple times
    // in different collections (they'll each get looked up independently).
    let o1 = Organism {
        collections: vec![o1_coll_1],
        pos: Vec3::new(100.0, 50.0, 100.0),
        orientation: Vec3::ZERO,
        ocg: vec![
            CellType::RedCell,
            CellType::YellowCell,
            CellType::GreenCell,
            CellType::OrangeCell,
            CellType::LightBlueCell,
            CellType::BlueCell,
        ],
    };

    let mut colony = Colony {
        organisms: vec![o1],
    };

    // ── Spawn loop ───────────────────────────────────────────────────────────
    // Walk the OCG vector in order. For each cell type, search all collections
    // of the organism until we find one that owns that type, then spawn it at
    // the correct world position.
    //
    // Complexity: O(ocg.len × collections.len) — negligible for biological
    // organism sizes; can be pre-indexed if organisms grow very large.
    

    for organism in &colony.organisms {
        for cell_type in &organism.ocg {
            // Find the first collection that contains this cell type and
            // compute its world-space position.
            let spawn_pos = organism
                .collections
                .iter()
                .find_map(|collection| collection.world_pos(cell_type, organism.pos));

            match spawn_pos {
                Some(pos) => {
                    spawn_rhombic_dodecahedron(
                        pos,
                        cell_type,
                        &mut commands,
                        &mut meshes,
                        &mut materials,
                    );
                }
                None => {
                    // The OCG references a cell type that isn't registered in
                    // any collection — log a warning and skip rather than panic.
                    warn!(
                        "OCG contains {:?} but no collection owns it — skipping.",
                        cell_type
                    );
                }
            }
        }
    }
}
