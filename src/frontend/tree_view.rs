// Lineages window mode — pan/zoom-able 2D tree of life.
//
// Lives under `frontend/` (not `lineages/`) because it's purely UI /
// render plumbing: the `SpeciesRegistry` is the domain model, and
// this module just renders it. `FrontendPlugin` registers every
// system below and owns the `apply_mode_transition` Display flip,
// keeping all Window-Mode plumbing in one file.
//
// Architecture (scale-first — designed for tens of thousands of
// species on a cluster):
//
//   * Dedicated `Camera2d` rendering on `RenderLayers::layer(LAYER)`
//     into an off-screen `Image` (the same render-target trick the
//     simulation viewport uses for its 3D camera). The
//     `LineagesPanel` UI entity is an `ImageNode` displaying that
//     texture, so the panel composites cleanly with the rest of the
//     UI tree and the Window-Mode `Display::None` toggle just
//     hides it.
//   * One `Mesh2d` for ALL edges (`LineList` topology), rebuilt
//     only when `SpeciesRegistry` changes (1 Hz speciation tick).
//     The static-mesh approach is what makes the renderer cheap at
//     n species: one upload on change, zero CPU work on frames
//     where nothing has changed.
//   * One `Text2d` per species (label + member count). Bevy
//     frustum-culls them automatically; we additionally LOD-gate
//     visibility behind a zoom threshold so labels disappear when
//     the user is zoomed all the way out over thousands of nodes.
//
// Layout: leaf-counting / centred-parent (a simplified
// Reingold-Tilford). O(n) walk, produces non-overlapping
// coordinates with each subtree owning a contiguous x-range.
// Auto-expands naturally: a subtree with many leaves takes a wider
// slot than a sparse sibling.
//
// Interaction:
//   * `Pointer<Drag>` on the panel image pans the camera
//     (translation delta scaled by `1 / zoom` so screen-pixel drag
//     distance matches visual distance regardless of zoom).
//   * `MouseWheel` over the panel multiplies the orthographic
//     scale (clamped to `[MIN_ZOOM, MAX_ZOOM]`).

use std::collections::HashMap;

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::RenderLayers;
use bevy::camera::{OrthographicProjection, RenderTarget};
use bevy::input::mouse::MouseWheel;
use bevy::mesh::PrimitiveTopology;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureFormat};
use bevy::sprite::Text2d;
use bevy::sprite_render::{ColorMaterial, MeshMaterial2d};
use bevy::window::PrimaryWindow;

use crate::frontend::{PANEL_BG_COLOR, TOP_BAR_HEIGHT_PX};
use crate::lineages::species::SpeciesRegistry;
use crate::simulation_settings::WindowMode;


// ── Tunables ────────────────────────────────────────────────────────────────

/// RenderLayers index for the lineages 2D scene. The lineages
/// camera renders only entities tagged with this layer; the rest
/// of the world (3D simulation, UI overlays, etc.) sit on layer 0.
const LAYER: usize = 1;

/// World-space vertical distance between depth levels in the tree.
const LAYER_HEIGHT_PX: f32 = 100.0;

/// World-space horizontal distance between adjacent leaves. Each
/// subtree's width grows automatically as `leaf_count * NODE_SPACING_PX`,
/// so wide subtrees push their right-siblings further right — the
/// "auto-expanding edges" the user asked for.
const NODE_SPACING_PX: f32 = 160.0;

const MIN_ZOOM: f32 = 0.02;
const MAX_ZOOM: f32 = 8.0;
/// Multiplicative zoom factor per mouse-wheel notch. 1.2 = ~20%
/// per click — fast enough to traverse the full zoom range in a
/// handful of scrolls, smooth enough to land on intuitive scales.
const ZOOM_STEP: f32 = 1.2;
/// Label LOD: below this zoom level (heavily zoomed out) we hide
/// labels so tens-of-thousands of `Text2d` entities don't sit in
/// the visible viewport at once.
const LABEL_VISIBLE_MIN_ZOOM: f32 = 0.5;

/// Background colour painted INSIDE the render-target texture (i.e.
/// behind the tree itself). The panel's own background colour
/// surrounds the image when there's letterboxing. Hex #8ba4d6 — a
/// muted slate-blue.
const SCENE_BG: Color = Color::srgb(0.5451, 0.6431, 0.8392);
const EDGE_COLOR: Color = Color::srgb(0.55, 0.58, 0.65);

const LABEL_FONT_SIZE: f32 = 16.0;
const NAME_COLOR:      Color = Color::srgb(0.95, 0.95, 0.95);
const NAME_COLOR_EXT:  Color = Color::srgb(0.75, 0.35, 0.35);


// ── Resources ───────────────────────────────────────────────────────────────

/// Handle to the off-screen image the lineages camera renders into,
/// plus the last-applied size so we only realloc when the panel
/// pixel dimensions actually change.
#[derive(Resource)]
pub struct LineagesViewportRender {
    pub image:        Handle<Image>,
    pub current_size: UVec2,
}

/// Pan + zoom state, mirrored onto the lineages camera each frame.
/// Persists across mode flips so re-entering the Lineages mode
/// preserves the user's view.
#[derive(Resource)]
pub struct LineagesViewState {
    pub pan:  Vec2,
    pub zoom: f32,
}

impl Default for LineagesViewState {
    fn default() -> Self {
        // Default centred on the origin (where the first root
        // species lands) at 1:1 zoom.
        Self { pan: Vec2::ZERO, zoom: 1.0 }
    }
}

/// Output of the layout pass. World-space `(x, y)` per species id.
/// Recomputed only when `SpeciesRegistry::is_changed()`.
#[derive(Resource, Default)]
pub struct TreeLayout {
    pub positions: HashMap<u32, Vec2>,
    /// Rendered AABB extent — useful for an eventual "fit-to-view"
    /// button. Not currently consumed.
    pub bounds:    Vec2,
}


// ── Marker components ───────────────────────────────────────────────────────

/// Marker on the Lineages mode's outermost panel — toggled
/// `Display::None`/`Display::Flex` by `apply_mode_transition`. The
/// panel itself is an `ImageNode` showing
/// `LineagesViewportRender::image`.
#[derive(Component)]
pub struct LineagesPanel;

/// Marker on the dedicated 2D camera that owns the lineages render
/// target. `apply_lineages_camera_state` writes pan/zoom onto its
/// Transform + OrthographicProjection.
#[derive(Component)]
pub struct LineagesCamera;

/// Marker on the single mesh entity holding every elbow-connector
/// line in the current tree. Rebuilt on layout change.
#[derive(Component)]
pub struct LineagesEdgesMesh;

/// Marker on each species' label entity (Text2d) — the per-species
/// `id` lets us diff existing labels against the species list
/// without rebuilding every frame.
#[derive(Component)]
pub struct LineagesNodeLabel { pub id: u32 }

/// Captured at the start of each LMB drag so subsequent Drag events
/// (cumulative distance from start point) resolve to absolute
/// camera positions.
#[derive(Resource, Default)]
struct LineagesDragState {
    pan_at_drag_start: Vec2,
}


// ── Panel spawn (called from `frontend.rs::setup_panes`) ────────────────────

/// Creates the off-screen render-target image and the panel
/// `ImageNode` that displays it. The image handle is inserted as a
/// `LineagesViewportRender` resource the layout/render systems
/// query. Returns the new panel `Entity` so the caller can wire it
/// into the rest of the layout tree.
pub fn spawn_lineages_panel(
    parent:          &mut ChildSpawnerCommands,
    top_offset_px:   f32,
    images:          &mut Assets<Image>,
) -> Handle<Image> {
    // Initial dimensions are placeholder — `resize_lineages_target`
    // resizes on the first Update that observes a non-zero
    // ComputedNode size for the panel.
    let image = images.add(Image::new_target_texture(
        1, 1,
        TextureFormat::Rgba8Unorm,
        Some(TextureFormat::Rgba8UnormSrgb),
    ));
    parent
        .spawn((
            LineagesPanel,
            // ImageNode shows the render target. We start with a
            // tiny image and resize via `resize_lineages_target`
            // once the panel's ComputedNode reports its actual px
            // dimensions.
            ImageNode::new(image.clone()),
            Node {
                position_type:  PositionType::Absolute,
                top:            Val::Px(top_offset_px),
                left:           Val::Px(0.0),
                right:          Val::Px(0.0),
                bottom:         Val::Px(0.0),
                display:        Display::None,
                ..default()
            },
            // Background painted by the camera, but a fallback colour
            // here ensures the panel isn't see-through on the first
            // frame before the camera has produced its first image.
            BackgroundColor(PANEL_BG_COLOR),
            // `Pickable` so the pan-drag + zoom-wheel observers
            // attached below receive events.
            Pickable {
                should_block_lower: true,
                is_hoverable:       true,
            },
        ))
        .observe(lineages_pan_start)
        .observe(lineages_pan_drag);
    image
}


// ── Startup: spawn the dedicated 2D camera ──────────────────────────────────

pub fn setup_lineages_camera(
    mut commands: Commands,
    viewport:     Res<LineagesViewportRender>,
) {
    commands.spawn((
        LineagesCamera,
        Camera2d,
        Camera {
            // Order doesn't matter relative to the main UI camera
            // because we have a separate render target — pick a
            // distinct value so debugger logs aren't ambiguous.
            order: 2,
            clear_color: bevy::camera::ClearColorConfig::Custom(SCENE_BG),
            ..default()
        },
        // `RenderTarget` is a separate component in Bevy 0.18
        // (split off from `Camera` in 0.17). Matches the simulation
        // viewport's wiring in `bind_main_camera_to_viewport`.
        RenderTarget::Image(viewport.image.clone().into()),
        Projection::Orthographic(OrthographicProjection {
            scale: 1.0,
            ..OrthographicProjection::default_2d()
        }),
        Transform::default(),
        RenderLayers::layer(LAYER),
    ));
}


// ── Render-target resize ────────────────────────────────────────────────────

pub fn resize_lineages_target(
    mut images:    ResMut<Assets<Image>>,
    mut viewport:  ResMut<LineagesViewportRender>,
    panel_q:       Query<&ComputedNode, With<LineagesPanel>>,
) {
    let Ok(node) = panel_q.single() else { return };
    let size = node.size();
    let new_w = size.x.max(1.0).round() as u32;
    let new_h = size.y.max(1.0).round() as u32;
    let new_size = UVec2::new(new_w, new_h);
    if new_size == viewport.current_size { return; }
    if let Some(image) = images.get_mut(&viewport.image) {
        image.resize(Extent3d {
            width:  new_w,
            height: new_h,
            depth_or_array_layers: 1,
        });
        viewport.current_size = new_size;
    }
}


// ── Layout pass ─────────────────────────────────────────────────────────────

pub fn rebuild_tree_layout(
    registry:   Res<SpeciesRegistry>,
    mut layout: ResMut<TreeLayout>,
) {
    if !registry.is_changed() { return; }

    // Adjacency: parent_id → children, sorted by id for stable
    // left-to-right order across runs.
    let mut children_of: HashMap<Option<u32>, Vec<u32>> = HashMap::new();
    for s in &registry.species {
        children_of.entry(s.parent_id).or_default().push(s.id);
    }
    for kids in children_of.values_mut() { kids.sort(); }

    let mut positions: HashMap<u32, Vec2> = HashMap::with_capacity(registry.species.len());
    let mut next_leaf: f32 = 0.0;

    // Iterative DFS — recursion at registry sizes of 10k+ could
    // overflow the stack on degenerate (linear chain) shapes. We
    // emit nodes in post-order so each parent reads its children's
    // x positions when assigning its own.
    enum Frame { Enter(u32), Exit(u32, u32 /* depth */, f32 /* first leaf x */) }
    let roots: Vec<u32> = children_of.get(&None).cloned().unwrap_or_default();
    let mut stack: Vec<Frame> = roots.iter().rev().map(|&r| Frame::Enter(r)).collect();

    // For Enter frames we need the depth too — push it as a side
    // map keyed by id. We DFS each root independently.
    let mut depth_of: HashMap<u32, u32> = HashMap::new();
    for &r in &roots { depth_of.insert(r, 0); }

    while let Some(frame) = stack.pop() {
        match frame {
            Frame::Enter(id) => {
                let depth = *depth_of.get(&id).unwrap_or(&0);
                let kids  = children_of.get(&Some(id)).cloned().unwrap_or_default();
                if kids.is_empty() {
                    // Leaf — claim the next x slot.
                    positions.insert(
                        id,
                        Vec2::new(
                            next_leaf * NODE_SPACING_PX,
                            depth as f32 * LAYER_HEIGHT_PX,
                        ),
                    );
                    next_leaf += 1.0;
                } else {
                    // Internal — record the first-leaf-x at this point
                    // and schedule the post-order Exit frame after
                    // its children. Note: we push Exit FIRST (stack
                    // pops in reverse), so the Exit runs AFTER all
                    // children's Enter+Exit frames complete.
                    let first_leaf_x = next_leaf;
                    stack.push(Frame::Exit(id, depth, first_leaf_x));
                    for &k in kids.iter().rev() {
                        depth_of.insert(k, depth + 1);
                        stack.push(Frame::Enter(k));
                    }
                }
            }
            Frame::Exit(id, depth, first_leaf_x) => {
                // All descendants have been placed. The internal
                // node's x is the centre of the [first_leaf_x,
                // last_leaf_x] range (last is `next_leaf - 1` —
                // we've already advanced past it).
                let last_leaf_x = next_leaf - 1.0;
                let centre_x    = (first_leaf_x + last_leaf_x) * 0.5;
                positions.insert(
                    id,
                    Vec2::new(
                        centre_x * NODE_SPACING_PX,
                        depth as f32 * LAYER_HEIGHT_PX,
                    ),
                );
            }
        }
    }

    // Translate so the tree's bounding box is centred horizontally
    // around x=0 (more pleasant default view than left-aligned).
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = 0.0_f32;
    for v in positions.values() {
        min_x = min_x.min(v.x);
        max_x = max_x.max(v.x);
        max_y = max_y.max(v.y);
    }
    if min_x.is_finite() {
        let centre_x = (min_x + max_x) * 0.5;
        for v in positions.values_mut() { v.x -= centre_x; }
        layout.bounds = Vec2::new(max_x - min_x, max_y);
    } else {
        layout.bounds = Vec2::ZERO;
    }
    layout.positions = positions;
}


// ── Visual rebuild (mesh + labels) ──────────────────────────────────────────

pub fn rebuild_tree_visuals(
    mut commands:   Commands,
    layout:         Res<TreeLayout>,
    registry:       Res<SpeciesRegistry>,
    mut meshes:     ResMut<Assets<Mesh>>,
    mut materials:  ResMut<Assets<ColorMaterial>>,
    edges_q:        Query<Entity, With<LineagesEdgesMesh>>,
    labels_q:       Query<Entity, With<LineagesNodeLabel>>,
) {
    if !layout.is_changed() { return; }

    // ── Wipe the previous frame's edges + labels ──────────────────
    for e in &edges_q  { commands.entity(e).despawn(); }
    for e in &labels_q { commands.entity(e).despawn(); }

    // ── Build the single LineList edges mesh ──────────────────────
    // Bevy 2D's positive Y points UP; our layout uses depth*Y for
    // "lower in the tree", so we negate Y here so deeper species
    // sit DOWN-screen, matching the natural top-down tree
    // metaphor.
    let mut positions: Vec<[f32; 3]> = Vec::new();
    for s in &registry.species {
        let Some(parent_id) = s.parent_id else { continue; };
        let (Some(&p), Some(&c)) = (
            layout.positions.get(&parent_id),
            layout.positions.get(&s.id),
        ) else { continue; };
        let p_world = Vec2::new(p.x, -p.y);
        let c_world = Vec2::new(c.x, -c.y);
        let mid_y   = (p_world.y + c_world.y) * 0.5;

        // 3 segments: down from parent, across, down to child.
        // LineList topology = each pair of vertices is one segment.
        positions.push([p_world.x, p_world.y, 0.0]);
        positions.push([p_world.x, mid_y,    0.0]);
        positions.push([p_world.x, mid_y,    0.0]);
        positions.push([c_world.x, mid_y,    0.0]);
        positions.push([c_world.x, mid_y,    0.0]);
        positions.push([c_world.x, c_world.y, 0.0]);
    }

    if !positions.is_empty() {
        let mut mesh = Mesh::new(
            PrimitiveTopology::LineList,
            RenderAssetUsages::default(),
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
        let mesh_handle = meshes.add(mesh);
        let mat_handle  = materials.add(ColorMaterial::from_color(EDGE_COLOR));
        commands.spawn((
            LineagesEdgesMesh,
            Mesh2d(mesh_handle),
            MeshMaterial2d(mat_handle),
            Transform::default(),
            RenderLayers::layer(LAYER),
        ));
    }

    // ── One Text2d per species ───────────────────────────────────
    for s in &registry.species {
        let Some(&pos) = layout.positions.get(&s.id) else { continue; };
        let world = Vec2::new(pos.x, -pos.y);
        let colour = if s.extinct { NAME_COLOR_EXT } else { NAME_COLOR };
        let count_line = if s.extinct {
            String::from("extinct")
        } else {
            format!("{} member(s)", s.member_count)
        };
        commands.spawn((
            LineagesNodeLabel { id: s.id },
            Text2d::new(format!("{}\n{}", s.name, count_line)),
            TextFont { font_size: LABEL_FONT_SIZE, ..default() },
            TextColor(colour),
            // z=1 so labels sit slightly forward of the edges mesh
            // (which is at z=0). Bevy 2D paints higher-z later.
            Transform::from_translation(Vec3::new(world.x, world.y, 1.0)),
            RenderLayers::layer(LAYER),
        ));
    }
}


// ── Pan: Pointer drag observers ─────────────────────────────────────────────

fn lineages_pan_start(
    ev:           On<Pointer<Press>>,
    mut state:    ResMut<LineagesDragState>,
    view:         Res<LineagesViewState>,
    window_mode:  Res<WindowMode>,
) {
    if !matches!(ev.button, PointerButton::Primary) { return; }
    if *window_mode != WindowMode::Lineages { return; }
    state.pan_at_drag_start = view.pan;
}

fn lineages_pan_drag(
    ev:           On<Pointer<Drag>>,
    state:        Res<LineagesDragState>,
    mut view:     ResMut<LineagesViewState>,
    window_mode:  Res<WindowMode>,
) {
    if !matches!(ev.button, PointerButton::Primary) { return; }
    if *window_mode != WindowMode::Lineages { return; }
    // `ev.distance` is the cumulative drag in logical pixels since
    // press. Convert to world units by dividing by zoom — a 100px
    // drag at 0.5× zoom moves the camera 200 world units, matching
    // visual pixel-to-pixel feel. Negate X so dragging right
    // pulls the world right (camera moves left).
    let dx = -ev.distance.x / view.zoom;
    // Drag Y is "down is positive" in screen space; our world Y is
    // "up is positive". Negate so dragging DOWN pulls the world
    // down (camera moves up).
    let dy =  ev.distance.y / view.zoom;
    view.pan = state.pan_at_drag_start + Vec2::new(dx, dy);
}


// ── Zoom: mouse wheel over the panel ────────────────────────────────────────

pub fn lineages_zoom_wheel(
    mut wheel:    MessageReader<MouseWheel>,
    windows:      Query<&Window, With<PrimaryWindow>>,
    panel_q:      Query<(&ComputedNode, &bevy::ui::UiGlobalTransform), With<LineagesPanel>>,
    mut view:     ResMut<LineagesViewState>,
    window_mode:  Res<WindowMode>,
) {
    if *window_mode != WindowMode::Lineages {
        // Drain so events don't pile up for the next entry.
        for _ in wheel.read() {}
        return;
    }
    let Ok(window) = windows.single() else { return };
    let Some(cursor) = window.cursor_position() else {
        for _ in wheel.read() {}
        return;
    };
    let Ok((node, ui_xf)) = panel_q.single() else {
        for _ in wheel.read() {}
        return;
    };
    let inv_scale = node.inverse_scale_factor;
    let size      = node.size() * inv_scale;
    let centre    = ui_xf.translation * inv_scale;
    let min       = centre - size * 0.5;
    let max       = min + size;
    let over_panel =
        cursor.x >= min.x && cursor.x <= max.x &&
        cursor.y >= min.y && cursor.y <= max.y;
    if !over_panel {
        for _ in wheel.read() {}
        return;
    }

    let mut accum = 0.0_f32;
    for ev in wheel.read() { accum += ev.y; }
    if accum == 0.0 { return; }
    let factor = if accum > 0.0 { ZOOM_STEP } else { 1.0 / ZOOM_STEP };
    view.zoom = (view.zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);
}


// ── Apply state → camera ────────────────────────────────────────────────────

pub fn apply_lineages_camera_state(
    view:    Res<LineagesViewState>,
    mut cam: Query<(&mut Transform, &mut Projection), With<LineagesCamera>>,
) {
    let Ok((mut tf, mut proj)) = cam.single_mut() else { return };
    tf.translation.x = view.pan.x;
    tf.translation.y = view.pan.y;
    if let Projection::Orthographic(ref mut ortho) = *proj {
        // Bevy ortho `scale` is "world units per pixel" — larger
        // value = more world fits on screen (= "zoomed out"). Our
        // `LineagesViewState::zoom` follows the natural-language
        // convention where higher zoom = "more zoomed in", so we
        // invert.
        ortho.scale = 1.0 / view.zoom.max(MIN_ZOOM);
    }
}


// ── Label LOD (hide when zoomed far out) ────────────────────────────────────

pub fn apply_label_lod(
    view:        Res<LineagesViewState>,
    mut labels:  Query<&mut Visibility, With<LineagesNodeLabel>>,
) {
    if !view.is_changed() { return; }
    let want = if view.zoom < LABEL_VISIBLE_MIN_ZOOM {
        Visibility::Hidden
    } else {
        Visibility::Inherited
    };
    for mut v in &mut labels {
        if *v != want { *v = want; }
    }
}


// ── Init resources ──────────────────────────────────────────────────────────

/// Called by `FrontendPlugin::build`. Inserting `LineagesDragState`
/// here keeps it co-located with the other lineages state without
/// forcing a separate Plugin.
pub fn init_resources(app: &mut App) {
    app
        .init_resource::<LineagesViewState>()
        .init_resource::<LineagesDragState>()
        .init_resource::<TreeLayout>();
}
