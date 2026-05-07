use bevy::prelude::Vec3;

#[allow(dead_code)]
pub const VERT_COUNT: usize = 4;

/// Returns [T0, T1, T2, T3]. T0–T2 are the base triangle (CCW viewed from the
/// attachment side, i.e. the side facing the existing mesh). T3 is the apex.
/// `base_center` is the centroid of the base triangle; `outward` points from
/// base toward apex.
pub fn cell_vertices(base_center: Vec3, outward: Vec3, edge: f32) -> [Vec3; 4] {
    let up = if outward.dot(Vec3::Z).abs() < 0.9 { Vec3::Z } else { Vec3::X };
    let u = outward.cross(up).normalize();
    let v = outward.cross(u);

    let r = edge / 3.0_f32.sqrt();
    let sqrt3_half = 3.0_f32.sqrt() / 2.0;

    let t0 = base_center + r * u;
    let t1 = base_center + r * (-0.5 * u + sqrt3_half * v);
    let t2 = base_center + r * (-0.5 * u - sqrt3_half * v);
    let t3 = base_center + edge * (2.0_f32 / 3.0).sqrt() * outward;
    [t0, t1, t2, t3]
}

/// The base triangle (interior face after merging). Relative vertex indices.
pub const BASE_TRI: [u32; 3] = [0, 1, 2];

/// All triangles of the tetrahedron except the base. Relative vertex indices.
/// These become open faces after attachment.
pub fn non_base_tris() -> [[u32; 3]; 3] {
    [[0, 1, 3], [1, 2, 3], [2, 0, 3]]
}
