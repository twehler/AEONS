// Lineages window mode — pan/zoom-able 2D tree of life.
//
// Lives under `frontend/` (not `lineages/`): it's pure UI/render plumbing over
// the `SpeciesRegistry` domain model. `FrontendPlugin` registers the systems
// and owns the `apply_mode_transition` Display flip.
//
// Architecture (scale-first, for tens of thousands of species):
//   * Dedicated `Camera2d` on `RenderLayers::layer(LAYER)` rendering into an
//     off-screen `Image`; the `LineagesPanel` `ImageNode` displays it, so the
//     Window-Mode `Display::None` toggle just hides it.
//   * One `Mesh2d` for ALL edges (`LineList`), rebuilt only when
//     `SpeciesRegistry` changes — one upload on change, zero CPU otherwise.
//   * One `Text2d` per species; frustum-culled, plus LOD-gated by zoom.
//
// Layout: leaf-counting / centred-parent (simplified Reingold-Tilford). O(n),
// non-overlapping, each subtree owning a contiguous x-range; wider subtrees
// take wider slots.
//
// Interaction: `Pointer<Drag>` pans (delta × 1/zoom for pixel-match);
// `MouseWheel` multiplies ortho scale (clamped `[MIN_ZOOM, MAX_ZOOM]`).

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

/// RenderLayers index for the lineages 2D scene (rest of world is layer 0).
const LAYER: usize = 1;

/// World-space vertical distance between tree depth levels.
const LAYER_HEIGHT_PX: f32 = 100.0;

/// World-space horizontal distance between adjacent leaves; subtree width
/// grows as `leaf_count * NODE_SPACING_PX`.
const NODE_SPACING_PX: f32 = 160.0;

const MIN_ZOOM: f32 = 0.02;
const MAX_ZOOM: f32 = 8.0;
/// Multiplicative zoom factor per wheel notch (~20%/click).
const ZOOM_STEP: f32 = 1.2;
/// Below this zoom, labels are hidden (LOD).
const LABEL_VISIBLE_MIN_ZOOM: f32 = 0.5;

/// Background painted inside the render-target (behind the tree). #8ba4d6.
const SCENE_BG: Color = Color::srgb(0.5451, 0.6431, 0.8392);
const EDGE_COLOR: Color = Color::srgb(0.55, 0.58, 0.65);

const LABEL_FONT_SIZE: f32 = 16.0;
const NAME_COLOR:      Color = Color::srgb(0.95, 0.95, 0.95);
const NAME_COLOR_EXT:  Color = Color::srgb(0.75, 0.35, 0.35);


// ── Resources ───────────────────────────────────────────────────────────────

/// Off-screen image the lineages camera renders into, + last-applied size so
/// we only realloc when the panel's pixel dimensions change.
#[derive(Resource)]
pub struct LineagesViewportRender {
    pub image:        Handle<Image>,
    pub current_size: UVec2,
}

/// Pan + zoom state, mirrored onto the camera each frame. Persists across mode
/// flips so re-entering Lineages preserves the view.
#[derive(Resource)]
pub struct LineagesViewState {
    pub pan:  Vec2,
    pub zoom: f32,
}

impl Default for LineagesViewState {
    fn default() -> Self {
        // Centred on origin (first root species) at 1:1 zoom.
        Self { pan: Vec2::ZERO, zoom: 1.0 }
    }
}

/// Output of the layout pass. World-space `(x, y)` per species id.
/// Recomputed only when `SpeciesRegistry::is_changed()`.
#[derive(Resource, Default)]
pub struct TreeLayout {
    pub positions: HashMap<u32, Vec2>,
    /// Rendered AABB extent (for a future fit-to-view). Not currently consumed.
    pub bounds:    Vec2,
}


// ── Marker components ───────────────────────────────────────────────────────

/// Lineages mode's outermost panel (an `ImageNode` of
/// `LineagesViewportRender::image`); toggled by `apply_mode_transition`.
#[derive(Component)]
pub struct LineagesPanel;

/// Dedicated 2D camera owning the lineages render target.
#[derive(Component)]
pub struct LineagesCamera;

/// Single mesh entity holding every elbow-connector line; rebuilt on layout change.
#[derive(Component)]
pub struct LineagesEdgesMesh;

/// Per-species label entity (Text2d); `id` lets us diff against the species list.
#[derive(Component)]
pub struct LineagesNodeLabel { pub id: u32 }

/// Captured at LMB drag start so cumulative Drag distance resolves to absolute
/// camera positions.
#[derive(Resource, Default)]
struct LineagesDragState {
    pan_at_drag_start: Vec2,
}


// ── Panel spawn (called from `frontend.rs::setup_panes`) ────────────────────

/// Create the off-screen render-target image and the panel `ImageNode` showing
/// it; returns the image handle.
pub fn spawn_lineages_panel(
    parent:          &mut ChildSpawnerCommands,
    top_offset_px:   f32,
    images:          &mut Assets<Image>,
) -> Handle<Image> {
    // Placeholder size; `resize_lineages_target` resizes once the panel reports
    // a non-zero ComputedNode size.
    let image = images.add(Image::new_target_texture(
        1, 1,
        TextureFormat::Rgba8Unorm,
        Some(TextureFormat::Rgba8UnormSrgb),
    ));
    parent
        .spawn((
            LineagesPanel,
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
            // Fallback colour so the panel isn't see-through before the camera's
            // first image lands.
            BackgroundColor(PANEL_BG_COLOR),
            // `Pickable` so the pan-drag observers below receive events.
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
            // Separate render target, so order is independent of the main camera;
            // distinct value just for unambiguous debugger logs.
            order: 2,
            clear_color: bevy::camera::ClearColorConfig::Custom(SCENE_BG),
            ..default()
        },
        // `RenderTarget` is a separate component in Bevy 0.18 (split from `Camera`).
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

    // Adjacency: parent_id → children, sorted by id for stable order across runs.
    let mut children_of: HashMap<Option<u32>, Vec<u32>> = HashMap::new();
    for s in &registry.species {
        children_of.entry(s.parent_id).or_default().push(s.id);
    }
    for kids in children_of.values_mut() { kids.sort(); }

    let mut positions: HashMap<u32, Vec2> = HashMap::with_capacity(registry.species.len());
    let mut next_leaf: f32 = 0.0;

    // Iterative DFS (recursion could stack-overflow on degenerate chains at
    // 10k+ species). Post-order so each parent reads its children's x first.
    enum Frame { Enter(u32), Exit(u32, u32 /* depth */, f32 /* first leaf x */) }
    let roots: Vec<u32> = children_of.get(&None).cloned().unwrap_or_default();
    let mut stack: Vec<Frame> = roots.iter().rev().map(|&r| Frame::Enter(r)).collect();

    // Depth tracked in a side map keyed by id; each root DFS'd independently.
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
                    // Internal — record first-leaf-x and schedule the post-order
                    // Exit. Push Exit FIRST so it pops AFTER all children's frames.
                    let first_leaf_x = next_leaf;
                    stack.push(Frame::Exit(id, depth, first_leaf_x));
                    for &k in kids.iter().rev() {
                        depth_of.insert(k, depth + 1);
                        stack.push(Frame::Enter(k));
                    }
                }
            }
            Frame::Exit(id, depth, first_leaf_x) => {
                // Descendants placed; x = centre of [first_leaf_x, last_leaf_x]
                // (last = next_leaf - 1, already advanced past).
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

    // Centre the bounding box horizontally around x=0.
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
    // Bevy 2D's +Y is up; layout uses depth*Y for "lower in tree", so negate Y
    // here so deeper species sit down-screen.
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

        // 3 segments (down, across, down); LineList = each vertex pair is one segment.
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
            // z=1 so labels paint in front of the edges mesh (z=0).
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
    // `ev.distance` is cumulative logical px since press; ÷zoom → world units
    // for pixel-match feel. Negate X so dragging right pulls the world right.
    let dx = -ev.distance.x / view.zoom;
    // Screen Y is down-positive, world Y up-positive; negate so down pulls down.
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
        // Bevy ortho `scale` is world-units-per-pixel (larger = zoomed out);
        // our `zoom` is higher = zoomed in, so invert.
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

/// Called by `FrontendPlugin::build`; co-locates lineages state without a
/// separate Plugin.
pub fn init_resources(app: &mut App) {
    app
        .init_resource::<LineagesViewState>()
        .init_resource::<LineagesDragState>()
        .init_resource::<TreeLayout>();
}
