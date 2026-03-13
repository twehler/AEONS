use bevy::prelude::*;
use bevy::mesh::Indices;
pub use bevy::render::render_resource::PrimitiveTopology;
use bevy::asset::RenderAssetUsages;

pub fn generate_rhombic_dodecahedron(pos: Vec3, total_width: f32) -> Mesh {
    let s = total_width / 4.0;

    // 14 vertices — same layout as the Python version
    let v = [
        // Cube corners (0-7)
        Vec3::new( s,  s,  s), Vec3::new( s,  s, -s),
        Vec3::new( s, -s,  s), Vec3::new( s, -s, -s),
        Vec3::new(-s,  s,  s), Vec3::new(-s,  s, -s),
        Vec3::new(-s, -s,  s), Vec3::new(-s, -s, -s),
        // Octahedron tips (8-13)
        Vec3::new( 2.0*s,  0.0,    0.0  ), // +X (8)
        Vec3::new(-2.0*s,  0.0,    0.0  ), // -X (9)
        Vec3::new( 0.0,    2.0*s,  0.0  ), // +Y (10)
        Vec3::new( 0.0,   -2.0*s,  0.0  ), // -Y (11)
        Vec3::new( 0.0,    0.0,    2.0*s), // +Z (12)
        Vec3::new( 0.0,    0.0,   -2.0*s), // -Z (13)
    ];

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals:   Vec<[f32; 3]> = Vec::new();
    let mut indices:   Vec<u32>      = Vec::new();

    // Mirrors add_face(p1, p2, p3, p4) from the Python version.
    // p1 = center tip, p2/p4 = side cube corners, p3 = opposite tip.
    let mut add_face = |p1: usize, p2: usize, p3: usize, p4: usize| {
        let pts = [v[p1], v[p2], v[p3], v[p4]];

        let edge1 = pts[1] - pts[0];
        let edge2 = pts[2] - pts[0];
        let norm  = edge1.cross(edge2).normalize();

        let base = positions.len() as u32;
        for p in &pts {
            positions.push((*p + pos).into());
            normals.push(norm.into());
        }

        // Two triangles per diamond face
        indices.extend([base, base+1, base+2, base, base+2, base+3]);
    };

    // Top cap — connected to +Z tip (index 12)
    add_face(12,  0, 10,  4); // Top-Front (+Y)
    add_face(12,  4,  9,  6); // Top-Left  (-X)
    add_face(12,  6, 11,  2); // Top-Back  (-Y)
    add_face(12,  2,  8,  0); // Top-Right (+X)

    // Bottom cap — connected to -Z tip (index 13)
    add_face( 5, 10,  1, 13); // Bottom-Front (+Y)
    add_face( 7,  9,  5, 13); // Bottom-Left  (-X)
    add_face( 3, 11,  7, 13); // Bottom-Back  (-Y)
    add_face( 1,  8,  3, 13); // Bottom-Right (+X)

    // Middle ring
    add_face( 1, 10,  0,  8); // Side +X/+Y
    add_face( 5,  9,  4, 10); // Side +Y/-X
    add_face( 7, 11,  6,  9); // Side -X/-Y
    add_face( 3,  8,  2, 11); // Side -Y/+X

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL,   normals);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}



pub fn spawn_rhombic_dodecahedron(
    mut commands:  Commands,
    mut meshes:    ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mesh = generate_rhombic_dodecahedron(Vec3::new(100.0, 100.0, 80.0), 1.0);

    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.8, 0.7, 0.6),
            ..default()
        })),
        Transform::IDENTITY,
    ));
}


