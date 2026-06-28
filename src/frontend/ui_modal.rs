// Shared confirm-modal scaffolding for the colony + species editors.
//
// Both editors raise several near-identical full-screen "Are you sure?"
// confirmation modals (exit / clear / load colony; species clear / load /
// mesh-import warning). Each one is the same shape: a translucent backdrop, a
// centred dark card with an optional title + a body line, and a `No` / `Yes`
// button row. They differ only in card width, z-index, the exact text, the
// `No`-button colour (the colony exit/load modals use a slightly darker green
// than the destructive-prompt modals), and whether `No` carries a highlighted
// border (the "safe default" variant). This module factors that scaffolding
// out so each call site only supplies its config + its own marker components
// (the markers MUST stay per-modal so each lifecycle/handler system queries
// only its own entities).
//
// Behaviour is preserved exactly: the spawned node tree, colours, sizes and
// z-indices match the hand-rolled versions this replaces.

use bevy::prelude::*;


// ── Shared palette ────────────────────────────────────────────────────────────

pub const MODAL_BACKDROP_COLOR: Color = Color::srgba(0.0, 0.0, 0.0, 0.55);
pub const MODAL_CARD_COLOR:     Color = Color::srgb(0.15, 0.15, 0.18);
pub const MODAL_CARD_BORDER:    Color = Color::srgb(0.40, 0.40, 0.45);

pub const MODAL_BTN_WIDTH:      f32   = 110.0;
pub const MODAL_BTN_HEIGHT:     f32   = 36.0;
pub const MODAL_BTN_GAP:        f32   = 16.0;

/// Destructive `Yes` (muted red) used by every confirm modal.
pub const YES_BTN_COLOR: Color = Color::srgb(0.55, 0.18, 0.18);
pub const YES_BTN_HOVER: Color = Color::srgb(0.68, 0.22, 0.22);

/// `No` palette as used by the colony exit/load modals (slightly darker green,
/// no highlight border).
pub const NO_BTN_COLOR_PLAIN: Color = Color::srgb(0.20, 0.45, 0.30);
pub const NO_BTN_HOVER_PLAIN: Color = Color::srgb(0.26, 0.55, 0.36);

/// `No` palette as the highlighted "safe default" (brighter green + white
/// border) used by the clear / load-species / import-warning modals.
pub const NO_BTN_COLOR_SAFE:  Color = Color::srgb(0.24, 0.56, 0.36);
pub const NO_BTN_HOVER_SAFE:  Color = Color::srgb(0.32, 0.66, 0.42);
pub const NO_BTN_BORDER:      Color = Color::srgb(0.95, 0.95, 0.95);
pub const NO_BTN_BORDER_WIDTH: f32  = 2.0;


// ── Modal config ──────────────────────────────────────────────────────────────

/// Look of the `No` button: a plain darker-green button, or the highlighted
/// "safe default" (brighter green + white border).
#[derive(Clone, Copy)]
pub enum NoButtonStyle {
    Plain,
    Safe,
}

impl NoButtonStyle {
    pub fn base(self) -> Color {
        match self {
            NoButtonStyle::Plain => NO_BTN_COLOR_PLAIN,
            NoButtonStyle::Safe  => NO_BTN_COLOR_SAFE,
        }
    }
    pub fn hover(self) -> Color {
        match self {
            NoButtonStyle::Plain => NO_BTN_HOVER_PLAIN,
            NoButtonStyle::Safe  => NO_BTN_HOVER_SAFE,
        }
    }
}

/// Declarative description of one confirm modal. Strings are owned so call
/// sites can build them from `format!`/`&str` alike.
pub struct ConfirmModalSpec {
    /// Optional bold (18px white) title line above the body. `None` ⇒ body only.
    pub title:     Option<String>,
    /// Body / question text.
    pub body:      String,
    /// Body text font size (colony modals use 14, species modals use 15).
    pub body_font_size: f32,
    /// Body text colour (colony modals use 0.85 grey, species modals 0.9).
    pub body_color: Color,
    /// Card width in logical px.
    pub card_width: f32,
    /// `GlobalZIndex` of the backdrop (modals stack by this).
    pub z_index:   i32,
    pub no_style:  NoButtonStyle,
}

/// Colony-editor body style (14px, 0.85 grey) — the most common case.
pub const BODY_FONT_SIZE: f32 = 14.0;
pub const BODY_COLOR:     Color = Color::srgb(0.85, 0.85, 0.85);
/// Species-editor body style (15px, 0.9 grey).
pub const BODY_FONT_SIZE_LG: f32 = 15.0;
pub const BODY_COLOR_LG:     Color = Color::srgb(0.9, 0.9, 0.9);


// ── Spawn ───────────────────────────────────────────────────────────────────

/// Spawn a full-screen confirm modal: backdrop + card + (title?) + body +
/// `No`/`Yes` button row. `root_marker`, `no_marker`, `yes_marker` are the
/// caller's per-modal marker components (so its own lifecycle/handler systems
/// query only its entities). `No` is on the left (the safe default), `Yes` on
/// the right.
pub fn spawn_confirm_modal<R, N, Y>(
    commands:    &mut Commands,
    spec:        &ConfirmModalSpec,
    root_marker: R,
    no_marker:   N,
    yes_marker:  Y,
)
where
    R: Component,
    N: Component,
    Y: Component,
{
    commands
        .spawn((
            root_marker,
            // Full-screen backdrop blocks clicks falling through to the editor.
            Node {
                position_type: PositionType::Absolute,
                top:    Val::Px(0.0),
                left:   Val::Px(0.0),
                width:  Val::Percent(100.0),
                height: Val::Percent(100.0),
                justify_content: JustifyContent::Center,
                align_items:     AlignItems::Center,
                ..default()
            },
            BackgroundColor(MODAL_BACKDROP_COLOR),
            GlobalZIndex(spec.z_index),
        ))
        .with_children(|root| {
            root.spawn((
                Node {
                    width:  Val::Px(spec.card_width),
                    flex_direction: FlexDirection::Column,
                    align_items:    AlignItems::Center,
                    padding: UiRect::all(Val::Px(22.0)),
                    border:  UiRect::all(Val::Px(1.0)),
                    ..default()
                },
                BackgroundColor(MODAL_CARD_COLOR),
                BorderColor::all(MODAL_CARD_BORDER),
            ))
            .with_children(|card| {
                if let Some(title) = &spec.title {
                    card.spawn((
                        Text::new(title.clone()),
                        TextFont { font_size: 18.0, ..default() },
                        TextColor(Color::WHITE),
                        Node { margin: UiRect::bottom(Val::Px(6.0)), ..default() },
                        Pickable::IGNORE,
                    ));
                }
                card.spawn((
                    Text::new(spec.body.clone()),
                    TextFont { font_size: spec.body_font_size, ..default() },
                    TextColor(spec.body_color),
                    Node { margin: UiRect::bottom(Val::Px(20.0)), ..default() },
                    Pickable::IGNORE,
                ));

                // Button row — No (left, safe default) then Yes.
                card.spawn(Node {
                    flex_direction:  FlexDirection::Row,
                    justify_content: JustifyContent::Center,
                    align_items:     AlignItems::Center,
                    ..default()
                })
                .with_children(|row| {
                    // ── No ──
                    let mut no_node = Node {
                        width:  Val::Px(MODAL_BTN_WIDTH),
                        height: Val::Px(MODAL_BTN_HEIGHT),
                        margin: UiRect::right(Val::Px(MODAL_BTN_GAP)),
                        align_items:     AlignItems::Center,
                        justify_content: JustifyContent::Center,
                        ..default()
                    };
                    let mut no_entity = match spec.no_style {
                        NoButtonStyle::Safe => {
                            no_node.border = UiRect::all(Val::Px(NO_BTN_BORDER_WIDTH));
                            row.spawn((
                                no_marker,
                                Button,
                                no_node,
                                BackgroundColor(NoButtonStyle::Safe.base()),
                                BorderColor::all(NO_BTN_BORDER),
                            ))
                        }
                        NoButtonStyle::Plain => row.spawn((
                            no_marker,
                            Button,
                            no_node,
                            BackgroundColor(NoButtonStyle::Plain.base()),
                        )),
                    };
                    no_entity.with_children(|btn| {
                        btn.spawn((
                            Text::new("No"),
                            TextFont { font_size: 16.0, ..default() },
                            TextColor(Color::WHITE),
                            Pickable::IGNORE,
                        ));
                    });

                    // ── Yes (destructive) ──
                    row.spawn((
                        yes_marker,
                        Button,
                        Node {
                            width:  Val::Px(MODAL_BTN_WIDTH),
                            height: Val::Px(MODAL_BTN_HEIGHT),
                            align_items:     AlignItems::Center,
                            justify_content: JustifyContent::Center,
                            ..default()
                        },
                        BackgroundColor(YES_BTN_COLOR),
                    ))
                    .with_children(|btn| {
                        btn.spawn((
                            Text::new("Yes"),
                            TextFont { font_size: 16.0, ..default() },
                            TextColor(Color::WHITE),
                            Pickable::IGNORE,
                        ));
                    });
                });
            });
        });
}


// ── Lifecycle / button helpers ────────────────────────────────────────────────

/// The classic spawn-when-`want`-rises / despawn-when-it-falls toggle every
/// modal lifecycle system performs. `existing` is the root-marker query.
pub fn sync_modal_visibility<R: Component>(
    commands: &mut Commands,
    want:     bool,
    existing: &Query<Entity, With<R>>,
    spawn:    impl FnOnce(&mut Commands),
) {
    let is_visible = !existing.is_empty();
    if want && !is_visible {
        spawn(commands);
    } else if !want && is_visible {
        for e in existing { commands.entity(e).despawn(); }
    }
}

/// Run the standard button colour state-machine for a modal button and return
/// `true` exactly once, on the `Pressed` edge, so the caller can run its
/// side-effect. `base`/`hover` are the button's two colours.
pub fn modal_button_pressed(
    interaction: &Interaction,
    bg:          &mut BackgroundColor,
    base:        Color,
    hover:       Color,
) -> bool {
    match *interaction {
        Interaction::Pressed => {
            *bg = BackgroundColor(hover);
            true
        }
        Interaction::Hovered => {
            *bg = BackgroundColor(hover);
            false
        }
        Interaction::None => {
            *bg = BackgroundColor(base);
            false
        }
    }
}
