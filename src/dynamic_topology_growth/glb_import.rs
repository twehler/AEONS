//! Import a `.glb` (binary glTF) triangle mesh as a seed `Topology`.
//!
//! The imported file's vertex positions and triangle indices are copied
//! verbatim — no remeshing, no reorientation, no smoothing. Vertex 0 of the
//! imported mesh is used as `Topology::relative_origin`, and an OCG ledger
//! is initialized exactly as for the cube seed (one entry per vertex,
//! position = absolute - origin). Subsequent growth then operates on this
//! seed instead of the cube.

use std::path::Path;

use bevy::prelude::*;

use super::{Face, Topology};

#[derive(Debug)]
pub enum GlbImportError {
    Gltf(gltf::Error),
    NoMesh,
    NoPositions,
    NotTriangulated,
}

impl std::fmt::Display for GlbImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GlbImportError::Gltf(e) => write!(f, "glTF parse error: {}", e),
            GlbImportError::NoMesh => write!(f, "GLB contains no mesh primitives"),
            GlbImportError::NoPositions => write!(f, "primitive has no POSITION attribute"),
            GlbImportError::NotTriangulated => {
                write!(f, "primitive is not a triangle list")
            }
        }
    }
}

impl std::error::Error for GlbImportError {}

/// Load a `.glb` file and produce a `Topology` whose vertices and faces
/// mirror the imported mesh, with OCG initialized in the same pattern as
/// `Topology::initial_cube`.
pub fn topology_from_glb(path: &Path) -> Result<Topology, GlbImportError> {
    let (gltf, buffers, _images) =
        gltf::import(path).map_err(GlbImportError::Gltf)?;

    let mut vertices: Vec<Vec3> = Vec::new();
    let mut triangles: Vec<[u32; 3]> = Vec::new();

    for mesh in gltf.meshes() {
        for prim in mesh.primitives() {
            if prim.mode() != gltf::mesh::Mode::Triangles {
                return Err(GlbImportError::NotTriangulated);
            }
            let reader = prim.reader(|b| Some(&buffers[b.index()]));
            let positions = reader
                .read_positions()
                .ok_or(GlbImportError::NoPositions)?;
            let base = vertices.len() as u32;
            let mut prim_vert_count: u32 = 0;
            for p in positions {
                vertices.push(Vec3::from_array(p));
                prim_vert_count += 1;
            }
            // Indexed primitives have an explicit index buffer; non-indexed
            // ones (common from Blender) imply a sequential triangle list
            // 0,1,2, 3,4,5, … over the position stream.
            let inds: Vec<u32> = match reader.read_indices() {
                Some(idx) => idx.into_u32().collect(),
                None => (0..prim_vert_count).collect(),
            };
            if inds.len() % 3 != 0 {
                return Err(GlbImportError::NotTriangulated);
            }
            for tri in inds.chunks_exact(3) {
                triangles.push([base + tri[0], base + tri[1], base + tri[2]]);
            }
        }
    }

    if vertices.is_empty() || triangles.is_empty() {
        return Err(GlbImportError::NoMesh);
    }

    let origin = vertices[0];
    let faces: Vec<Face> = triangles
        .into_iter()
        .map(|[a, b, c]| {
            let pa = vertices[a as usize];
            let pb = vertices[b as usize];
            let pc = vertices[c as usize];
            let normal = (pb - pa).cross(pc - pa).normalize_or_zero();
            Face {
                vertices: vec![a, b, c],
                normal,
            }
        })
        .collect();

    Ok(Topology::from_seed_mesh(vertices, faces, origin))
}
