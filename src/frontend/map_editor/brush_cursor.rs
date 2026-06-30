// Map editor — brush-radius cursor ring.
//
// A thin (~1.5px) screen-space circle outline at the brush radius, centred on the
// cursor, whose line colour is the per-pixel INVERT of whatever is behind it
// (light->black, dark->white). It is a custom `UiMaterial` node parented to the
// viewport image; the material's `specialize` overrides the fragment blend to an
// INVERT blend, so the effect needs NO framebuffer read (the cheapest possible
// approach). See /todo/brush_circle.md.
//
// Purely cosmetic: `Pickable::IGNORE` + no `Interaction`, so it never captures the
// LMB paint / MMB rotate inputs; it never touches the paint texture; and it is
// hidden whenever the cursor is off the paintable viewport, a brush field is
// focused, or we're not in the Map Editor.
//
// Render correctness: the ring node is a CHILD of the `ViewportImage` node, so in
// the UI pass the terrain image is drawn first and the ring draws on top of it —
// i.e. the invert blend's `dst` already contains the composited terrain.

use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, BlendComponent, BlendFactor, BlendOperation, BlendState,
    RenderPipelineDescriptor,
};
use bevy::shader::ShaderRef;
use bevy::ui::{ComputedNode, UiGlobalTransform};
use bevy::window::PrimaryWindow;

use crate::frontend::ViewportImage;
use crate::simulation_settings::WindowMode;

use super::gpu_paint::{BrushSizeEditState, SoftnessEditState};
use super::material::MapEditorSession;

/// Outline thickness in logical screen pixels.
const LINE_WIDTH_PX: f32 = 1.5;
/// Extra room around the ring (for the AA margin) when sizing the node.
const NODE_PAD_PX: f32 = 4.0;

/// Custom UI material: draws an antialiased ring (white + coverage), drawn with an
/// INVERT blend set in `specialize`.
#[derive(AsBindGroup, Asset, TypePath, Clone, Debug)]
pub struct BrushCursorMaterial {
    /// `x` = ring radius (px), `y` = line half-width (px); `z`, `w` unused.
    #[uniform(0)]
    pub params: Vec4,
}

impl UiMaterial for BrushCursorMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/brush_cursor.wgsl".into()
    }

    fn specialize(descriptor: &mut RenderPipelineDescriptor, _key: UiMaterialKey<Self>) {
        // INVERT blend. The fragment outputs PREMULTIPLIED white — coverage `m` in
        // RGB (see brush_cursor.wgsl) — because these COLOR factors key off the RGB
        // output, not alpha:
        //   out_rgb = src_rgb·(1-dst) + dst·(1-src_rgb) = m·(1-dst) + dst·(1-m)
        //     m=1 (on ring)  -> 1 - dst (invert: light->black, dark->white)
        //     m=0 (off ring) -> dst     (unchanged)
        // The alpha component (Zero/One) leaves the framebuffer alpha untouched.
        if let Some(fragment) = descriptor.fragment.as_mut() {
            if let Some(target) = fragment.targets.get_mut(0).and_then(|t| t.as_mut()) {
                target.blend = Some(BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::OneMinusDst,
                        dst_factor: BlendFactor::OneMinusSrc,
                        operation: BlendOperation::Add,
                    },
                    alpha: BlendComponent {
                        src_factor: BlendFactor::Zero,
                        dst_factor: BlendFactor::One,
                        operation: BlendOperation::Add,
                    },
                });
            }
        }
    }
}

/// Marker for the single cursor-ring node.
#[derive(Component)]
pub struct BrushCursorRing;

/// Spawn the ring node ONCE, as a child of the viewport image, hidden. Retries each
/// frame until the viewport exists; no-ops after it's spawned.
pub fn spawn_brush_cursor_ring(
    mut commands: Commands,
    mut materials: ResMut<Assets<BrushCursorMaterial>>,
    viewport: Query<Entity, With<ViewportImage>>,
    existing: Query<(), With<BrushCursorRing>>,
) {
    if !existing.is_empty() {
        return;
    }
    let Ok(vp) = viewport.single() else {
        return; // viewport not instantiated yet
    };

    let handle = materials.add(BrushCursorMaterial { params: Vec4::ZERO });
    let ring = commands
        .spawn((
            BrushCursorRing,
            MaterialNode(handle),
            Node {
                position_type: PositionType::Absolute,
                width: Val::Px(0.0),
                height: Val::Px(0.0),
                ..default()
            },
            // Draw above the viewport image (its parent) and the fly hint.
            ZIndex(1000),
            Visibility::Hidden,
            // Cosmetic only — must never intercept the paint / rotate pointer.
            Pickable::IGNORE,
        ))
        .id();
    commands.entity(vp).add_child(ring);
}

/// Each frame: gate visibility (Map Editor + cursor over the viewport + no brush
/// field focused), and when shown, centre + size the node on the cursor and push
/// the radius / line width into the material uniform.
#[allow(clippy::type_complexity)]
pub fn update_brush_cursor_ring(
    mode: Res<WindowMode>,
    session: Res<MapEditorSession>,
    brush_edit: Res<BrushSizeEditState>,
    softness_edit: Res<SoftnessEditState>,
    windows: Query<&Window, With<PrimaryWindow>>,
    viewport_q: Query<(&ComputedNode, &UiGlobalTransform), With<ViewportImage>>,
    mut ring: Query<(&mut Node, &mut Visibility, &MaterialNode<BrushCursorMaterial>), With<BrushCursorRing>>,
    mut materials: ResMut<Assets<BrushCursorMaterial>>,
) {
    let Ok((mut node, mut vis, mat)) = ring.single_mut() else {
        return; // ring not spawned yet
    };

    // Compute show/position; `break 'show false` for any reason to hide.
    let show = 'show: {
        if *mode != WindowMode::MapEditor {
            break 'show false;
        }
        if brush_edit.focused || softness_edit.focused {
            break 'show false;
        }
        let Ok(window) = windows.single() else { break 'show false };
        let Some(cursor) = window.cursor_position() else { break 'show false };
        let Ok((vp_node, vp_xf)) = viewport_q.single() else { break 'show false };

        // Cursor in viewport-local logical px (mirrors gpu_paint::cursor_to_viewport_px).
        let inv = vp_node.inverse_scale_factor;
        let size = vp_node.size() * inv;
        let top_left = vp_xf.translation * inv - size * 0.5;
        let local = cursor - top_left;

        // Hide unless the cursor is over the paintable viewport rect.
        if local.x < 0.0 || local.y < 0.0 || local.x > size.x || local.y > size.y {
            break 'show false;
        }

        let radius = session.brush_radius_px.max(0.5);
        let half = radius + LINE_WIDTH_PX + NODE_PAD_PX;
        let dim = half * 2.0;
        node.width = Val::Px(dim);
        node.height = Val::Px(dim);
        node.left = Val::Px(local.x - half);
        node.top = Val::Px(local.y - half);

        if let Some(m) = materials.get_mut(&mat.0) {
            let want = Vec4::new(radius, LINE_WIDTH_PX * 0.5, 0.0, 0.0);
            if m.params != want {
                m.params = want;
            }
        }
        true
    };

    let target = if show { Visibility::Visible } else { Visibility::Hidden };
    if *vis != target {
        *vis = target;
    }
}
