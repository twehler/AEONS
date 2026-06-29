// Texture-driven terrain properties, averaged per heightmap cell.
//
// INFRASTRUCTURE ONLY (this version): the painted terrain texture is reduced to a
// per-cell **average colour** (over each cell's TOP surface), and that colour is
// run through a single pluggable `props_from_color` classifier to derive per-cell
// `TerrainCellProperties` (first concrete property: nutrients, from the hue +
// saturation of brown). The result is a `TerrainProperties` resource the
// simulation can later read via `properties_at(x, z)`. **Nothing here is wired into
// plant growth / photosynthesis yet** — that is a deliberate follow-up.
//
// The grid mirrors `HeightmapSampler` 1:1 (`width`/`depth`/`min_x`/`min_z`,
// `HEIGHTMAP_CELL_SIZE`), so a cell here is the same 1×1 world-unit XZ square.
// Because the paint texture is atlas-space (xatlas UV unwrap — NOT aligned to the
// XZ grid), the builder rasterises each terrain triangle's atlas-UV footprint,
// maps every painted texel's interpolated WORLD position back to its XZ cell, and
// averages. A texel contributes only if its world Y is within
// `TERRAIN_PROP_TOP_SURFACE_TOL` of the heightmap's stored max-Y for that cell, so
// a cave ceiling / overhang underside never pollutes the floor cell.
//
// (The barycentric texel walk below is a small, self-contained reimplementation of
// the same math the Map Editor brush uses in `gpu_paint::rasterise_triangle`. It is
// duplicated deliberately rather than shared: this builder lives in the WORLD layer
// and runs at load, and reaching into `frontend::map_editor` for it would invert
// the dependency direction.)

use bevy::prelude::*;

use crate::simulation_settings::{
    BROWN_HUE_CENTER, BROWN_HUE_HALF_WIDTH, BROWN_HUE_MAX, BROWN_HUE_MIN, BROWN_SAT_MIN,
    BROWN_VAL_MAX, BROWN_VAL_MIN, NUTRIENT_BASELINE, NUTRIENT_MAX, TERRAIN_PROP_TOP_SURFACE_TOL,
};
use crate::world_geometry::HEIGHTMAP_CELL_SIZE;


// ── Per-cell properties ─────────────────────────────────────────────────────────

/// Derived properties of one terrain cell. Add fields here as new properties are
/// introduced; `Default` is the neutral/baseline value used for unpainted cells.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainCellProperties {
    /// Nutrient richness available to plants growing on this cell (currently
    /// derived from brown hue + saturation; consumed by no system yet).
    pub nutrients: f32,
}

impl Default for TerrainCellProperties {
    fn default() -> Self {
        Self { nutrients: NUTRIENT_BASELINE }
    }
}


// ── The grid resource ───────────────────────────────────────────────────────────

/// Per-cell terrain properties, dimensioned 1:1 with `HeightmapSampler`. `cells`
/// and `avg_color` are row-major `width × depth` (index `z*width + x`); `avg_color`
/// is retained for debugging + re-derivation.
#[derive(Resource, Clone)]
pub struct TerrainProperties {
    pub width:     u32,
    pub depth:     u32,
    pub min_x:     i32, // world-X of column 0 (cell units)
    pub min_z:     i32, // world-Z of row 0 (cell units)
    pub cells:     Vec<TerrainCellProperties>,
    pub avg_color: Vec<[u8; 4]>,
}

// Infrastructure for future texture-driven terrain effects: the lookup API +
// fields are intentionally not consumed by any system YET (see the module header),
// so allow the dead-code until a consumer (plant nutrient uptake) is wired in.
#[allow(dead_code)]
impl TerrainProperties {
    /// A neutral grid (every cell baseline / mid-grey) sized to a heightmap. Used
    /// for `.glb` worlds (no paint) and load fallbacks.
    pub fn neutral(width: u32, depth: u32, min_x: i32, min_z: i32) -> Self {
        let n = (width.max(1) * depth.max(1)) as usize;
        Self {
            width: width.max(1),
            depth: depth.max(1),
            min_x,
            min_z,
            cells:     vec![TerrainCellProperties::default(); n],
            avg_color: vec![[128, 128, 128, 255]; n],
        }
    }

    /// Cell index for world XZ (nearest cell, clamped to the grid), or `None` if the
    /// grid is empty.
    fn cell_index(&self, x: f32, z: f32) -> Option<usize> {
        if self.width == 0 || self.depth == 0 { return None; }
        let cell = HEIGHTMAP_CELL_SIZE;
        let col = ((x / cell).floor() as i32 - self.min_x).clamp(0, self.width as i32 - 1) as u32;
        let row = ((z / cell).floor() as i32 - self.min_z).clamp(0, self.depth as i32 - 1) as u32;
        Some((row * self.width + col) as usize)
    }

    /// Properties at world XZ (Default if out of range / empty grid).
    pub fn properties_at(&self, x: f32, z: f32) -> TerrainCellProperties {
        self.cell_index(x, z)
            .and_then(|i| self.cells.get(i).copied())
            .unwrap_or_default()
    }

    /// Averaged surface colour at world XZ (mid-grey if out of range / empty grid).
    pub fn avg_color_at(&self, x: f32, z: f32) -> [u8; 4] {
        self.cell_index(x, z)
            .and_then(|i| self.avg_color.get(i).copied())
            .unwrap_or([128, 128, 128, 255])
    }
}


// ── Builder input ───────────────────────────────────────────────────────────────

/// One terrain submesh as the property builder needs it: the model matrix (local →
/// world), local vertex positions, atlas UVs, and triangle indices.
pub struct PropMesh<'a> {
    pub model:     Mat4,
    pub positions: &'a [[f32; 3]],
    pub uv0:       &'a [[f32; 2]],
    pub indices:   &'a [u32],
}


// ── Builder ─────────────────────────────────────────────────────────────────────

/// Build the per-cell property grid from the painted atlas texture + terrain
/// geometry. `heights` is the heightmap's per-cell max-Y (same `width`/`depth`/
/// `min_x`/`min_z` grid). `tex_bytes` is row-major RGBA8 sRGB of an
/// `atlas_edge × atlas_edge` texture. O(total terrain paint texels) — runs once at
/// load, never per frame.
#[allow(clippy::too_many_arguments)]
pub fn build_terrain_properties(
    width: u32,
    depth: u32,
    min_x: i32,
    min_z: i32,
    heights: &[f32],
    tex_bytes: &[u8],
    atlas_edge: u32,
    meshes: &[PropMesh],
) -> TerrainProperties {
    let n = (width.max(1) * depth.max(1)) as usize;
    if width == 0 || depth == 0 || atlas_edge == 0
        || tex_bytes.len() < (atlas_edge as usize) * (atlas_edge as usize) * 4
        || heights.len() < n
    {
        return TerrainProperties::neutral(width, depth, min_x, min_z);
    }

    // Per-cell colour accumulators (f64 sums + count).
    let mut sum = vec![[0.0f64; 3]; n];
    let mut count = vec![0u32; n];

    let edge = atlas_edge;
    let edge_f = edge as f32;
    let cell = HEIGHTMAP_CELL_SIZE;
    let tol = TERRAIN_PROP_TOP_SURFACE_TOL;

    for mesh in meshes {
        let vcount = mesh.positions.len().min(mesh.uv0.len());
        for tri in mesh.indices.chunks_exact(3) {
            let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            if i0 >= vcount || i1 >= vcount || i2 >= vcount { continue; }

            let uv0 = Vec2::from_array(mesh.uv0[i0]);
            let uv1 = Vec2::from_array(mesh.uv0[i1]);
            let uv2 = Vec2::from_array(mesh.uv0[i2]);
            let p0 = Vec2::new(uv0.x * edge_f, uv0.y * edge_f);
            let p1 = Vec2::new(uv1.x * edge_f, uv1.y * edge_f);
            let p2 = Vec2::new(uv2.x * edge_f, uv2.y * edge_f);

            // Texel AABB clamped to the atlas.
            let edge_i = edge as i64;
            let min_tx = (p0.x.min(p1.x).min(p2.x).floor() as i64).clamp(0, edge_i - 1);
            let max_tx = (p0.x.max(p1.x).max(p2.x).ceil() as i64).clamp(0, edge_i - 1);
            let min_ty = (p0.y.min(p1.y).min(p2.y).floor() as i64).clamp(0, edge_i - 1);
            let max_ty = (p0.y.max(p1.y).max(p2.y).ceil() as i64).clamp(0, edge_i - 1);
            if min_tx > max_tx || min_ty > max_ty { continue; }

            let denom = edge_fn(p0, p1, p2);
            if denom.abs() < 1e-9 { continue; }
            let inv = 1.0 / denom;

            let w0 = Vec3::from_array(mesh.positions[i0]);
            let w1 = Vec3::from_array(mesh.positions[i1]);
            let w2 = Vec3::from_array(mesh.positions[i2]);

            for ty in min_ty..=max_ty {
                for tx in min_tx..=max_tx {
                    let s = Vec2::new(tx as f32 + 0.5, ty as f32 + 0.5);
                    let b0 = edge_fn(p1, p2, s) * inv;
                    let b1 = edge_fn(p2, p0, s) * inv;
                    let b2 = edge_fn(p0, p1, s) * inv;
                    const E: f32 = -1e-4;
                    if b0 < E || b1 < E || b2 < E { continue; }

                    // World surface point at this texel.
                    let p_local = w0 * b0 + w1 * b1 + w2 * b2;
                    let p_world = mesh.model.transform_point3(p_local);

                    // XZ → cell (nearest, must be in range).
                    let col = (p_world.x / cell).floor() as i32 - min_x;
                    let row = (p_world.z / cell).floor() as i32 - min_z;
                    if col < 0 || row < 0 || col as u32 >= width || row as u32 >= depth { continue; }
                    let ci = (row as u32 * width + col as u32) as usize;

                    // Top-surface filter: only the surface near the cell's max-Y.
                    if (p_world.y - heights[ci]).abs() > tol { continue; }

                    // Sample this texel's sRGB colour and accumulate.
                    let idx = (ty as usize * edge as usize + tx as usize) * 4;
                    sum[ci][0] += tex_bytes[idx] as f64;
                    sum[ci][1] += tex_bytes[idx + 1] as f64;
                    sum[ci][2] += tex_bytes[idx + 2] as f64;
                    count[ci] += 1;
                }
            }
        }
    }

    // Reduce to per-cell average colour + derived properties.
    let mut cells = vec![TerrainCellProperties::default(); n];
    let mut avg_color = vec![[128u8, 128, 128, 255]; n];
    for ci in 0..n {
        if count[ci] > 0 {
            let c = count[ci] as f64;
            let to_u8 = |v: f64| (v / c).round().clamp(0.0, 255.0) as u8;
            let col = [to_u8(sum[ci][0]), to_u8(sum[ci][1]), to_u8(sum[ci][2]), 255];
            avg_color[ci] = col;
            cells[ci] = props_from_color(col);
        }
        // Uncovered cells keep the neutral default (mid-grey + baseline props).
    }

    TerrainProperties { width, depth, min_x, min_z, cells, avg_color }
}

/// 2× signed area of triangle (a, b, c) — the edge function for barycentrics.
#[inline]
fn edge_fn(a: Vec2, b: Vec2, c: Vec2) -> f32 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}


// ── Colour → properties classifier (the single pluggable mapping) ────────────────

/// Map an averaged sRGB cell colour to its derived properties. The first concrete
/// mapping is brown(hue + saturation) → nutrients; extend this one function as more
/// properties / colour families are added.
pub fn props_from_color(rgba: [u8; 4]) -> TerrainCellProperties {
    let (h, s, v) = rgb_to_hsv(rgba);
    let nutrients = if (BROWN_HUE_MIN..=BROWN_HUE_MAX).contains(&h)
        && s >= BROWN_SAT_MIN
        && (BROWN_VAL_MIN..=BROWN_VAL_MAX).contains(&v)
    {
        // Richer (more saturated) and more central-hue brown ⇒ more nutrients.
        // (A `v` term is a natural future extension for "darker = richer".)
        let centrality = (1.0 - (h - BROWN_HUE_CENTER).abs() / BROWN_HUE_HALF_WIDTH).clamp(0.0, 1.0);
        NUTRIENT_BASELINE + NUTRIENT_MAX * s * centrality
    } else {
        NUTRIENT_BASELINE
    };
    TerrainCellProperties { nutrients }
}

/// sRGB bytes → HSV (`h` in degrees `[0,360)`, `s`/`v` in `[0,1]`). Hue is 0 for
/// achromatic colours.
pub fn rgb_to_hsv(rgba: [u8; 4]) -> (f32, f32, f32) {
    let r = rgba[0] as f32 / 255.0;
    let g = rgba[1] as f32 / 255.0;
    let b = rgba[2] as f32 / 255.0;
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    let v = max;
    let s = if max <= 0.0 { 0.0 } else { delta / max };
    let h = if delta < 1e-6 {
        0.0
    } else {
        let h = if max == r {
            60.0 * (((g - b) / delta) % 6.0)
        } else if max == g {
            60.0 * (((b - r) / delta) + 2.0)
        } else {
            60.0 * (((r - g) / delta) + 4.0)
        };
        if h < 0.0 { h + 360.0 } else { h }
    };
    (h, s, v)
}


// ── Tests ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hsv_known_triples() {
        // Palette Brown #4f3423 ≈ H 23°, S 0.56, V 0.31.
        let (h, s, v) = rgb_to_hsv([79, 52, 35, 255]);
        assert!((h - 23.0).abs() < 2.0, "h = {h}");
        assert!((s - 0.56).abs() < 0.03, "s = {s}");
        assert!((v - 0.31).abs() < 0.02, "v = {v}");

        // Pure primaries + grey.
        let (hr, sr, vr) = rgb_to_hsv([255, 0, 0, 255]);
        assert!(hr.abs() < 1e-3 && (sr - 1.0).abs() < 1e-3 && (vr - 1.0).abs() < 1e-3);
        let (hg, _, _) = rgb_to_hsv([0, 255, 0, 255]);
        assert!((hg - 120.0).abs() < 1e-3, "hg = {hg}");
        let (_, sgrey, _) = rgb_to_hsv([128, 128, 128, 255]);
        assert!(sgrey.abs() < 1e-3, "grey saturation = {sgrey}");
    }

    #[test]
    fn brown_yields_nutrients_others_baseline() {
        // Palette Brown → clearly above baseline.
        let brown = props_from_color([79, 52, 35, 255]).nutrients;
        assert!(brown > NUTRIENT_BASELINE + 0.05, "brown nutrients = {brown}");

        // Grey / DarkGreen / Yellow → baseline (out of the brown band).
        assert_eq!(props_from_color([140, 140, 140, 255]).nutrients, NUTRIENT_BASELINE);
        assert_eq!(props_from_color([33, 89, 33, 255]).nutrients, NUTRIENT_BASELINE);
        // Yellow #bfa547 (H≈47°) sits just outside the band → baseline.
        assert_eq!(props_from_color([191, 165, 71, 255]).nutrients, NUTRIENT_BASELINE);
    }

    #[test]
    fn richer_brown_more_nutrients() {
        // A saturated central brown beats a washed-out one of the same hue family.
        let rich = props_from_color([79, 52, 35, 255]).nutrients;   // S≈0.56
        let washed = props_from_color([120, 105, 92, 255]).nutrients; // low S, same-ish hue
        assert!(rich > washed, "rich {rich} should exceed washed {washed}");
    }

    /// A flat 1×2-cell quad (two triangles), half painted Brown / half Grey, with a
    /// second LOWER triangle under the same XZ that must be excluded by the
    /// top-surface filter.
    #[test]
    fn per_cell_average_and_top_surface_filter() {
        // Grid: width 2, depth 1, origin (0,0). Cells cover x∈[0,1) and x∈[1,2).
        let width = 2u32;
        let depth = 1u32;
        let heights = vec![5.0f32, 5.0]; // top surface at Y=5 for both cells

        // 4×4 atlas: left half Brown, right half Grey.
        let edge = 4u32;
        let brown = [79u8, 52, 35, 255];
        let grey = [140u8, 140, 140, 255];
        let mut tex = vec![0u8; (edge * edge * 4) as usize];
        for ty in 0..edge {
            for tx in 0..edge {
                let idx = ((ty * edge + tx) * 4) as usize;
                let c = if tx < edge / 2 { brown } else { grey };
                tex[idx..idx + 4].copy_from_slice(&c);
            }
        }

        // Two top quads at Y=5 spanning x∈[0,2], z∈[0,1], UVs covering the full
        // atlas: left quad → left (brown) half, right quad → right (grey) half.
        // Plus a LOWER quad (Y=0) over the SAME XZ mapped to the brown half — it
        // must be filtered out (|0-5| > tol), so the grey cell stays grey.
        let positions: Vec<[f32; 3]> = vec![
            // left top quad (cell x=0): x 0..1
            [0.0, 5.0, 0.0], [1.0, 5.0, 0.0], [1.0, 5.0, 1.0], [0.0, 5.0, 1.0],
            // right top quad (cell x=1): x 1..2
            [1.0, 5.0, 0.0], [2.0, 5.0, 0.0], [2.0, 5.0, 1.0], [1.0, 5.0, 1.0],
            // lower quad under the right cell (x 1..2) at Y=0
            [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, 1.0], [1.0, 0.0, 1.0],
        ];
        let uv0: Vec<[f32; 2]> = vec![
            // left quad → brown half u∈[0,0.5]
            [0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0],
            // right quad → grey half u∈[0.5,1]
            [0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0],
            // lower quad → brown half (would wrongly turn the grey cell brown if not filtered)
            [0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0],
        ];
        let indices: Vec<u32> = vec![
            0, 1, 2, 0, 2, 3,
            4, 5, 6, 4, 6, 7,
            8, 9, 10, 8, 10, 11,
        ];

        let meshes = [PropMesh { model: Mat4::IDENTITY, positions: &positions, uv0: &uv0, indices: &indices }];
        let props = build_terrain_properties(width, depth, 0, 0, &heights, &tex, edge, &meshes);

        // Left cell ≈ brown → above baseline; right cell ≈ grey → baseline (the
        // lower brown quad was filtered out by the top-surface gate).
        let left = props.properties_at(0.5, 0.5).nutrients;
        let right = props.properties_at(1.5, 0.5).nutrients;
        assert!(left > NUTRIENT_BASELINE + 0.05, "left (brown) nutrients = {left}");
        assert_eq!(right, NUTRIENT_BASELINE, "right cell must stay grey/baseline");

        let lc = props.avg_color_at(0.5, 0.5);
        assert!(lc[0] > lc[2], "left avg colour should read brownish (R>B): {lc:?}");
        assert_eq!(props.avg_color_at(1.5, 0.5), grey, "right avg colour should be grey");
    }
}
