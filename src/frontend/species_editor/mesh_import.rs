// Species editor — transient `.glb` mesh import + Blender-style scaling.
//
// PURE, TRANSIENT: the imported mesh never touches sim/persistence data —
// it's not written to a `.species` file nor converted to cells/OCG. It is
// an in-editor visual template for a future species-generation step.
//
// Workflow: import button (warning modal if dirty, since import deletes the
// cell body) → file dialog → the `.glb` is copied into `assets/` (the
// `AssetServer` roots there, so arbitrary on-disk paths can't load directly)
// and spawned at the editor origin on the editor's `RenderLayers`. An orange
// dot marks the mesh origin (Blender-style). `s` begins scaling along the
// dot→cursor radius (uniform/linear vs. the radius at `s`-press; cursor-on-dot
// → scale 0); `Enter` confirms, `Esc` reverts. All state lives in `MeshImport`.

use bevy::camera::visibility::RenderLayers;
use bevy::gltf::GltfAssetLabel;
use bevy::math::Rot2;
use bevy::prelude::*;
use bevy::scene::{Scene, SceneRoot};
use bevy::ui::{ComputedNode, UiGlobalTransform, UiTransform};
use bevy::window::PrimaryWindow;

use std::path::{Path, PathBuf};

use crate::frontend::ViewportImage;
use crate::player_plugin::FlyCam;
use crate::simulation_settings::WindowMode;

use super::session::SpeciesSession;
use super::SPECIES_EDITOR_ORIGIN;
use super::SPECIES_EDITOR_LAYER;
use crate::ui_modal::{
    self, ConfirmModalSpec, NoButtonStyle, YES_BTN_COLOR, YES_BTN_HOVER,
};


// ── Tunables ─────────────────────────────────────────────────────────────────

/// Sub-directory under `assets/` imported meshes are copied into so the
/// `AssetServer` (rooted at `assets/`) can load them. Files accumulate harmlessly.
const IMPORT_DIR: &str = "imported_meshes";

const DOT_DIAMETER_PX:   f32 = 14.0;
const DOT_COLOR:         Color = Color::srgb(1.0, 0.55, 0.0);   // orange (Blender origin)
const LINE_THICKNESS_PX: f32 = 2.0;
const LINE_COLOR:        Color = Color::srgba(1.0, 0.55, 0.0, 0.9);

/// Floor on the initial cursor-radius so a `s`-press with the cursor
/// sitting exactly on the dot can't divide by zero.
const MIN_INITIAL_RADIUS_PX: f32 = 1.0;


// ── Resource ─────────────────────────────────────────────────────────────────

/// All transient state for the mesh-import / scaling feature.
#[derive(Resource, Default)]
pub struct MeshImport {
    /// The spawned `SceneRoot` entity, if a mesh is currently loaded.
    pub root: Option<Entity>,
    /// `true` once the scene's descendants have had the editor
    /// `RenderLayers` applied (the scene instantiates a frame or two later).
    pub layers_applied: bool,
    /// Committed uniform scale (also the reference captured at `s`-press).
    pub base_scale: f32,
    /// Live uniform scale currently applied to the mesh transform.
    pub current_scale: f32,
    /// `true` while a `s`-initiated scaling gesture is in progress.
    pub scaling: bool,
    /// Pixel distance from the centre dot to the cursor at `s`-press.
    pub initial_radius_px: f32,
    /// Rising-edge flag: the import button sets this when the session is
    /// dirty so the warning modal is raised before the file dialog.
    pub show_warning_modal: bool,
}

impl MeshImport {
    pub fn active(&self) -> bool { self.root.is_some() }
}


// ── Markers ──────────────────────────────────────────────────────────────────

#[derive(Component)] pub struct ImportedMeshRoot;
#[derive(Component)] pub struct ImportCenterDot;
#[derive(Component)] pub struct ImportScaleLine;

/// Top-panel button (spawned by `top_panel::spawn_top_panel`).
#[derive(Component, Default)] pub struct ImportMeshButton;

#[derive(Component)] pub struct ImportWarnModalRoot;
#[derive(Component)] pub struct ImportWarnYesButton;
#[derive(Component)] pub struct ImportWarnNoButton;


// ── Import (file dialog + spawn) ──────────────────────────────────────────────

/// File dialog → copy the `.glb` into `assets/` → spawn → delete the cell body.
/// Shared by the direct (non-dirty) path and the warning modal's "Yes".
fn begin_import(
    asset_server: &AssetServer,
    commands:     &mut Commands,
    session:      &mut SpeciesSession,
    mesh:         &mut MeshImport,
) {
    let initial_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let Some(path) = rfd::FileDialog::new()
        .add_filter("glTF binary (.glb)", &["glb", "gltf"])
        .set_directory(initial_dir)
        .pick_file()
    else { return };

    // Copy into assets/imported_meshes/<file> so the AssetServer can
    // resolve it (it roots at assets/; arbitrary absolute paths fail).
    let Some(file_name) = path.file_name() else {
        error!("mesh import: chosen path has no file name: {}", path.display());
        return;
    };
    let rel  = Path::new(IMPORT_DIR).join(file_name);
    let dest = Path::new("assets").join(&rel);
    if let Some(parent) = dest.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            error!("mesh import: failed to create {}: {}", parent.display(), e);
            return;
        }
    }
    if let Err(e) = std::fs::copy(&path, &dest) {
        error!("mesh import: failed to copy {} → {}: {}", path.display(), dest.display(), e);
        return;
    }

    // Forward-slash the relative path for the asset path (cross-platform).
    let rel_str = rel.to_string_lossy().replace('\\', "/");
    let scene: Handle<Scene> = asset_server.load(GltfAssetLabel::Scene(0).from_asset(rel_str));

    // Replace any previously-imported mesh.
    if let Some(old) = mesh.root.take() {
        commands.entity(old).despawn();
    }
    let root = commands
        .spawn((
            ImportedMeshRoot,
            SceneRoot(scene),
            Transform::from_translation(SPECIES_EDITOR_ORIGIN),
            Visibility::Visible,
            RenderLayers::layer(SPECIES_EDITOR_LAYER),
        ))
        .id();

    mesh.root             = Some(root);
    mesh.layers_applied   = false;
    mesh.base_scale       = 1.0;
    mesh.current_scale    = 1.0;
    mesh.scaling          = false;
    mesh.initial_radius_px = MIN_INITIAL_RADIUS_PX;

    // Delete the cell body — the editor now shows only the imported mesh.
    session.body_parts.clear();
    session.active_body_part   = 0;
    session.selected_cell_type = None;
    session.dirty              = false;

    info!("mesh import: loaded {} (copied to {})", path.display(), dest.display());
}

/// "Import Mesh (.glb)" button handler.
pub fn handle_import_button(
    mode:             Res<WindowMode>,
    mut interactions: Query<(&Interaction, &mut BackgroundColor),
                            (Changed<Interaction>, With<ImportMeshButton>)>,
    asset_server:     Res<AssetServer>,
    mut commands:     Commands,
    mut session:      ResMut<SpeciesSession>,
    mut mesh:         ResMut<MeshImport>,
) {
    if *mode != WindowMode::SpeciesEditor { return; }
    for (interaction, mut bg) in &mut interactions {
        match *interaction {
            Interaction::Pressed => {
                if session.dirty {
                    mesh.show_warning_modal = true;
                } else {
                    begin_import(&asset_server, &mut commands, &mut session, &mut mesh);
                }
                *bg = BackgroundColor(IMPORT_BG_HOVER);
            }
            Interaction::Hovered => *bg = BackgroundColor(IMPORT_BG_HOVER),
            Interaction::None    => *bg = BackgroundColor(IMPORT_BG),
        }
    }
}

pub const IMPORT_BG:       Color = Color::srgb(0.18, 0.48, 0.52); // teal
pub const IMPORT_BG_HOVER: Color = Color::srgb(0.24, 0.60, 0.66);


// ── Warning modal (unsaved changes) ───────────────────────────────────────────

pub fn manage_warning_modal_visibility(
    mut commands: Commands,
    mesh:         Res<MeshImport>,
    existing:     Query<Entity, With<ImportWarnModalRoot>>,
) {
    ui_modal::sync_modal_visibility(
        &mut commands, mesh.show_warning_modal, &existing, spawn_warning_modal);
}

fn spawn_warning_modal(commands: &mut Commands) {
    ui_modal::spawn_confirm_modal(
        commands,
        &ConfirmModalSpec {
            title:      None,
            body:       "Importing a mesh will delete the current cells, and there \
                         are unsaved changes. Are you sure?".to_string(),
            body_font_size: ui_modal::BODY_FONT_SIZE_LG,
            body_color:     ui_modal::BODY_COLOR_LG,
            card_width: 580.0,
            z_index:    120,
            no_style:   NoButtonStyle::Safe,
        },
        ImportWarnModalRoot,
        ImportWarnNoButton,
        ImportWarnYesButton,
    );
}

pub fn handle_warning_modal_buttons(
    mut yes_q:    Query<(&Interaction, &mut BackgroundColor),
                        (Changed<Interaction>, With<ImportWarnYesButton>, Without<ImportWarnNoButton>)>,
    mut no_q:     Query<(&Interaction, &mut BackgroundColor),
                        (Changed<Interaction>, With<ImportWarnNoButton>, Without<ImportWarnYesButton>)>,
    asset_server: Res<AssetServer>,
    mut commands: Commands,
    mut session:  ResMut<SpeciesSession>,
    mut mesh:     ResMut<MeshImport>,
) {
    for (interaction, mut bg) in &mut yes_q {
        if ui_modal::modal_button_pressed(interaction, &mut bg, YES_BTN_COLOR, YES_BTN_HOVER) {
            mesh.show_warning_modal = false;
            begin_import(&asset_server, &mut commands, &mut session, &mut mesh);
        }
    }
    let no_style = NoButtonStyle::Safe;
    for (interaction, mut bg) in &mut no_q {
        if ui_modal::modal_button_pressed(interaction, &mut bg, no_style.base(), no_style.hover()) {
            mesh.show_warning_modal = false;
        }
    }
}


// ── Render-layer fixup + scale application ────────────────────────────────────

/// Put the instantiated scene's descendants on the editor `RenderLayers`
/// (not auto-propagated to children) so the editor camera can see them.
pub fn apply_imported_mesh_render_layers(
    mut mesh:    ResMut<MeshImport>,
    mut commands: Commands,
    children_q:  Query<&Children>,
) {
    if mesh.layers_applied { return; }
    let Some(root) = mesh.root else { return };

    let mut stack = vec![root];
    let mut descendants = Vec::new();
    while let Some(e) = stack.pop() {
        if let Ok(children) = children_q.get(e) {
            for &c in children {
                stack.push(c);
                descendants.push(c);
            }
        }
    }
    if descendants.is_empty() { return; } // scene not instantiated yet

    for e in descendants {
        commands.entity(e).try_insert((
            RenderLayers::layer(SPECIES_EDITOR_LAYER),
            bevy::light::NotShadowCaster,
        ));
    }
    mesh.layers_applied = true;
}

/// Keep the imported mesh's transform synced to the current uniform scale.
pub fn apply_mesh_scale(
    mesh:  Res<MeshImport>,
    mut q: Query<&mut Transform, With<ImportedMeshRoot>>,
) {
    if !mesh.active() { return; }
    if let Ok(mut t) = q.single_mut() {
        t.scale       = Vec3::splat(mesh.current_scale.max(0.0));
        t.translation = SPECIES_EDITOR_ORIGIN;
    }
}


// ── Overlay entities (dot + line) ─────────────────────────────────────────────

/// Spawn/despawn the centre dot + scale line with mesh-loaded state.
/// Parented under `ViewportImage` so their `Val::Px` coords share the
/// space `world_to_viewport` returns.
pub fn manage_overlay_entities(
    mode:        Res<WindowMode>,
    mesh:        Res<MeshImport>,
    mut commands: Commands,
    viewport_q:  Query<Entity, With<ViewportImage>>,
    dot_q:       Query<Entity, With<ImportCenterDot>>,
    line_q:      Query<Entity, With<ImportScaleLine>>,
) {
    let want = mesh.active() && *mode == WindowMode::SpeciesEditor;
    if !want {
        for e in &dot_q  { commands.entity(e).despawn(); }
        for e in &line_q { commands.entity(e).despawn(); }
        return;
    }
    let Ok(viewport) = viewport_q.single() else { return };

    if dot_q.is_empty() {
        let dot = commands.spawn((
            ImportCenterDot,
            Node {
                position_type: PositionType::Absolute,
                width:  Val::Px(DOT_DIAMETER_PX),
                height: Val::Px(DOT_DIAMETER_PX),
                border_radius: BorderRadius::MAX,
                ..default()
            },
            BackgroundColor(DOT_COLOR),
            GlobalZIndex(60),
            Visibility::Hidden,
            Pickable::IGNORE,
        )).id();
        commands.entity(viewport).add_child(dot);
    }
    if line_q.is_empty() {
        let line = commands.spawn((
            ImportScaleLine,
            Node {
                position_type: PositionType::Absolute,
                width:  Val::Px(0.0),
                height: Val::Px(LINE_THICKNESS_PX),
                ..default()
            },
            BackgroundColor(LINE_COLOR),
            UiTransform::IDENTITY,
            GlobalZIndex(59),
            Visibility::Hidden,
            Pickable::IGNORE,
        )).id();
        commands.entity(viewport).add_child(line);
    }
}

/// Window-logical cursor → viewport-image-local logical px (same space as
/// `world_to_viewport(..) * inv_scale`). Mirrors `colony_editor::placement`.
fn cursor_to_viewport(cursor_window: Vec2, node: &ComputedNode, ui_xf: &UiGlobalTransform) -> Vec2 {
    let inv_scale = node.inverse_scale_factor;
    let size      = node.size() * inv_scale;
    let centre    = ui_xf.translation * inv_scale;
    let top_left  = centre - size * 0.5;
    cursor_window - top_left
}

/// Interaction core: read `s`/`Enter`/`Esc`, update the uniform scale from
/// the screen-space cursor radius, and drive the dot + line overlays.
#[allow(clippy::too_many_arguments)]
pub fn update_scale_interaction(
    mode:        Res<WindowMode>,
    mut mesh:    ResMut<MeshImport>,
    keys:        Res<ButtonInput<KeyCode>>,
    session:     Res<SpeciesSession>,
    windows:     Query<&Window, With<PrimaryWindow>>,
    cameras:     Query<(&Camera, &GlobalTransform), With<FlyCam>>,
    viewport_q:  Query<(&ComputedNode, &UiGlobalTransform), With<ViewportImage>>,
    mut dot_q:   Query<(&mut Node, &mut Visibility),
                       (With<ImportCenterDot>, Without<ImportScaleLine>)>,
    mut line_q:  Query<(&mut Node, &mut Visibility, &mut UiTransform),
                       (With<ImportScaleLine>, Without<ImportCenterDot>)>,
) {
    if *mode != WindowMode::SpeciesEditor || !mesh.active() { return; }

    let Ok((camera, cam_xf)) = cameras.single() else { return };
    let Ok((vp_node, vp_xf)) = viewport_q.single() else { return };
    let Ok(window)           = windows.single() else { return };
    let inv_scale            = vp_node.inverse_scale_factor;
    let viewport_size        = vp_node.size() * inv_scale;

    // Centre dot in viewport-local logical px (None if behind camera).
    let dot_screen = camera
        .world_to_viewport(cam_xf, SPECIES_EDITOR_ORIGIN)
        .ok()
        .map(|vp| vp * inv_scale);
    let cursor_screen = window
        .cursor_position()
        .map(|c| cursor_to_viewport(c, vp_node, vp_xf));

    // ── Input: start / confirm / cancel ──
    // Ignore keystrokes during a body-part rename (defensive; gesture isolation).
    let typing = session.renaming_body_part.is_some();
    if !typing {
        if keys.just_pressed(KeyCode::KeyS) && !mesh.scaling {
            if let (Some(d), Some(c)) = (dot_screen, cursor_screen) {
                mesh.initial_radius_px = (c - d).length().max(MIN_INITIAL_RADIUS_PX);
                mesh.base_scale        = mesh.current_scale;
                mesh.scaling           = true;
            }
        } else if mesh.scaling {
            if keys.just_pressed(KeyCode::Enter) || keys.just_pressed(KeyCode::NumpadEnter) {
                mesh.base_scale = mesh.current_scale; // commit
                mesh.scaling    = false;
            } else if keys.just_pressed(KeyCode::Escape) {
                mesh.current_scale = mesh.base_scale; // revert
                mesh.scaling       = false;
            }
        }
    }

    // ── Live scale update while gesturing ──
    if mesh.scaling {
        if let (Some(d), Some(c)) = (dot_screen, cursor_screen) {
            let radius = (c - d).length();
            let factor = radius / mesh.initial_radius_px;
            mesh.current_scale = (mesh.base_scale * factor).max(0.0);
        }
    }

    // ── Dot overlay ──
    if let Ok((mut node, mut vis)) = dot_q.single_mut() {
        match dot_screen {
            Some(d) if in_rect(d, viewport_size) => {
                node.left = Val::Px(d.x - DOT_DIAMETER_PX * 0.5);
                node.top  = Val::Px(d.y - DOT_DIAMETER_PX * 0.5);
                *vis = Visibility::Inherited;
            }
            _ => *vis = Visibility::Hidden,
        }
    }

    // ── Line overlay (only while scaling) ──
    if let Ok((mut node, mut vis, mut ui_xf)) = line_q.single_mut() {
        match (mesh.scaling, dot_screen, cursor_screen) {
            (true, Some(d), Some(c)) => {
                let delta  = c - d;
                let length = delta.length().max(0.001);
                let mid    = (d + c) * 0.5;
                node.width  = Val::Px(length);
                node.height = Val::Px(LINE_THICKNESS_PX);
                // Centre the horizontal bar on the midpoint (UI rotation pivots
                // about the node centre).
                node.left = Val::Px(mid.x - length * 0.5);
                node.top  = Val::Px(mid.y - LINE_THICKNESS_PX * 0.5);
                // Screen Y is down and UI rotation is clockwise, so θ =
                // atan2(Δy, Δx) points the +X bar along the dot→cursor direction.
                ui_xf.rotation = Rot2::radians(delta.y.atan2(delta.x));
                *vis = Visibility::Inherited;
            }
            _ => *vis = Visibility::Hidden,
        }
    }
}

fn in_rect(p: Vec2, size: Vec2) -> bool {
    p.x >= 0.0 && p.x <= size.x && p.y >= 0.0 && p.y <= size.y
}


// ── Lifecycle cleanup ──────────────────────────────────────────────────────────

/// When the user leaves SpeciesEditor mode, drop the imported mesh and
/// restore a fresh base cell so the editor isn't empty on return.
pub fn cleanup_on_mode_exit(
    mode:        Res<WindowMode>,
    mut mesh:    ResMut<MeshImport>,
    mut commands: Commands,
    mut session: ResMut<SpeciesSession>,
) {
    if !mode.is_changed() || *mode == WindowMode::SpeciesEditor { return; }
    if let Some(root) = mesh.root.take() {
        commands.entity(root).despawn();
    }
    *mesh = MeshImport::default();
    if session.body_parts.is_empty() {
        session.seed_base();
        session.dirty = false;
    }
}

/// If the cell body reappears while a mesh is loaded (the user clicked
/// Clear/New or loaded a `.species`), drop the imported mesh — the editor
/// is back in cell-authoring mode.
pub fn sync_with_session(
    mode:        Res<WindowMode>,
    mut mesh:    ResMut<MeshImport>,
    mut commands: Commands,
    session:     Res<SpeciesSession>,
) {
    if *mode != WindowMode::SpeciesEditor { return; }
    if mesh.active() && !session.body_parts.is_empty() {
        if let Some(root) = mesh.root.take() {
            commands.entity(root).despawn();
        }
        *mesh = MeshImport::default();
    }
}
