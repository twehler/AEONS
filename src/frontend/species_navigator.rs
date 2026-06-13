// Simulation-mode "Species Navigator" — left-side panel.
//
// Lists active organism categories + live counts, derived per `OrganismRoot`:
// Photoautotroph (marker); Heterotroph+!Carnivore+IL ⇒ "Herbivore L<N>";
// Heterotroph+Carnivore+IL ⇒ "Carnivore L<N>". L0 heteros fold into
// "Herbivore L0"; zero-count rows are hidden.
//
// Fixed rows spawn once and never churn — the 2 Hz refresh just rewrites count
// text and toggles `Display::None`, so the `OrganismRoot` walk runs ≤ 2x/sec.

use bevy::prelude::*;
use bevy::time::common_conditions::on_timer;
use std::time::Duration;

use crate::colony::{Carnivore, Heterotroph, IntelligenceLevel, Organism, OrganismRoot, Photoautotroph};
use crate::frontend::PANEL_BG_COLOR;
use crate::simulation_settings::WindowMode;


// ── Tunables ─────────────────────────────────────────────────────────────────

/// Panel width (logical px); mirrors `NAV_INITIAL_WIDTH_PX` for symmetry.
pub const SPECIES_NAV_WIDTH_PX: f32 = 220.0;

const PANEL_PADDING_PX:    f32 = 8.0;
const ROW_HEIGHT_PX:       f32 = 34.0;
const ROW_GAP_PX:          f32 = 4.0;
const ROW_BG:              Color = Color::srgb(0.20, 0.20, 0.22);
/// 2 Hz refresh.
const REFRESH_INTERVAL:    Duration = Duration::from_millis(500);


// ── Category enum ────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SpeciesCategory {
    Photoautotroph,
    HerbivoreL0,
    HerbivoreL1,
    HerbivoreL2,
    HerbivoreL3,
    CarnivoreL2,
    CarnivoreL3,
}

impl SpeciesCategory {
    /// Canonical display name, used as the static row label.
    pub fn label(self) -> &'static str {
        match self {
            SpeciesCategory::Photoautotroph => "Photoautotroph",
            SpeciesCategory::HerbivoreL0    => "Herbivore L0",
            SpeciesCategory::HerbivoreL1    => "Herbivore L1",
            SpeciesCategory::HerbivoreL2    => "Herbivore L2",
            SpeciesCategory::HerbivoreL3    => "Herbivore L3",
            SpeciesCategory::CarnivoreL2    => "Carnivore L2",
            SpeciesCategory::CarnivoreL3    => "Carnivore L3",
        }
    }

    /// All categories, in display order.
    pub const ALL: [SpeciesCategory; 7] = [
        SpeciesCategory::Photoautotroph,
        SpeciesCategory::HerbivoreL0,
        SpeciesCategory::HerbivoreL1,
        SpeciesCategory::HerbivoreL2,
        SpeciesCategory::HerbivoreL3,
        SpeciesCategory::CarnivoreL2,
        SpeciesCategory::CarnivoreL3,
    ];
}


// ── Markers ──────────────────────────────────────────────────────────────────

#[derive(Component)]
pub struct SpeciesNavigatorPanel;

#[derive(Component)]
struct SpeciesNavigatorList;

#[derive(Component, Clone, Copy)]
struct SpeciesRow(SpeciesCategory);

#[derive(Component, Clone, Copy)]
struct SpeciesRowText(SpeciesCategory);


// ── Plugin ───────────────────────────────────────────────────────────────────

pub struct SpeciesNavigatorPlugin;

impl Plugin for SpeciesNavigatorPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, refresh_species_navigator
            .run_if(on_timer(REFRESH_INTERVAL)));
    }
}


// ── Spawn ────────────────────────────────────────────────────────────────────

/// Append the panel as a child of the `ViewportRow` (from
/// `frontend::setup_panes`), making it a flex sibling of the viewport image so
/// it inherits the row's vertical extent. Must be spawned BEFORE the viewport
/// so flex places it on the left.
pub fn spawn_species_navigator(parent: &mut ChildSpawnerCommands) {
    parent
        .spawn((
            SpeciesNavigatorPanel,
            Node {
                width:           Val::Px(SPECIES_NAV_WIDTH_PX),
                height:          Val::Percent(100.0),
                flex_shrink:     0.0,
                flex_direction:  FlexDirection::Column,
                padding:         UiRect::all(Val::Px(PANEL_PADDING_PX)),
                display:         Display::None, // shown only in Simulation mode
                ..default()
            },
            BackgroundColor(PANEL_BG_COLOR),
        ))
        .with_children(|panel| {
            // Title.
            panel.spawn((
                Text::new("Species Navigator"),
                TextFont { font_size: 15.0, ..default() },
                TextColor(Color::srgb(0.92, 0.92, 0.92)),
                Node { margin: UiRect::bottom(Val::Px(8.0)), ..default() },
                Pickable::IGNORE,
            ));

            // Scrollable list of fixed rows; zero-count rows hidden via `Display::None`.
            panel.spawn((
                SpeciesNavigatorList,
                Node {
                    flex_grow:      1.0,
                    flex_basis:     Val::Px(0.0),
                    min_height:     Val::Px(0.0),
                    flex_direction: FlexDirection::Column,
                    overflow:       Overflow::scroll_y(),
                    ..default()
                },
                ScrollPosition::default(),
            ))
            .with_children(|list| {
                for cat in SpeciesCategory::ALL {
                    list.spawn((
                        SpeciesRow(cat),
                        Node {
                            width:       Val::Percent(100.0),
                            height:      Val::Px(ROW_HEIGHT_PX),
                            margin:      UiRect::bottom(Val::Px(ROW_GAP_PX)),
                            padding:     UiRect::axes(Val::Px(8.0), Val::Px(4.0)),
                            align_items: AlignItems::Center,
                            flex_shrink: 0.0,
                            display:     Display::None,
                            ..default()
                        },
                        BackgroundColor(ROW_BG),
                    ))
                    .with_children(|row| {
                        row.spawn((
                            SpeciesRowText(cat),
                            // Placeholder; populated by the refresh system.
                            Text::new(format!("{}: 0", cat.label())),
                            TextFont { font_size: 13.0, ..default() },
                            TextColor(Color::WHITE),
                            Pickable::IGNORE,
                        ));
                    });
                }
            });
        });
}


// ── Counting + refresh system ───────────────────────────────────────────────

/// Walk every `OrganismRoot` once, bucket into categories, rewrite row text and
/// toggle `Display::None` for empty ones. 2 Hz; skipped outside Simulation mode.
#[allow(clippy::type_complexity)]
fn refresh_species_navigator(
    window_mode:   Res<WindowMode>,
    organisms:     Query<
        (
            &Organism,
            Has<Photoautotroph>,
            Has<Heterotroph>,
            Has<Carnivore>,
        ),
        With<OrganismRoot>,
    >,
    mut rows:      Query<(&SpeciesRow, &mut Node)>,
    mut row_text:  Query<(&SpeciesRowText, &mut Text)>,
) {
    if *window_mode != WindowMode::Simulation { return; }

    // 7-bucket tally, keyed by order in `SpeciesCategory::ALL`.
    let mut counts = [0usize; 7];

    for (organism, is_photo, is_hetero, is_carn) in &organisms {
        let category = if is_photo {
            SpeciesCategory::Photoautotroph
        } else if is_hetero {
            match (is_carn, organism.intelligence_level) {
                (true,  IntelligenceLevel::Level2) => SpeciesCategory::CarnivoreL2,
                (true,  IntelligenceLevel::Level3) => SpeciesCategory::CarnivoreL3,
                // Carnivore L0/L1 shouldn't occur (editor enforces ≥ L2);
                // fold stragglers in rather than drop the count.
                (true,  IntelligenceLevel::Level0) => SpeciesCategory::HerbivoreL0,
                (true,  IntelligenceLevel::Level1) => SpeciesCategory::HerbivoreL1,
                (false, IntelligenceLevel::Level0) => SpeciesCategory::HerbivoreL0,
                (false, IntelligenceLevel::Level1) => SpeciesCategory::HerbivoreL1,
                (false, IntelligenceLevel::Level2) => SpeciesCategory::HerbivoreL2,
                (false, IntelligenceLevel::Level3) => SpeciesCategory::HerbivoreL3,
            }
        } else {
            continue; // not a tracked organism
        };
        counts[category_index(category)] += 1;
    }

    // Rewrite row texts on change. Separate text query avoids a borrow
    // conflict with the organism walk (independent archetype sets).
    for (marker, mut text) in &mut row_text {
        let c = counts[category_index(marker.0)];
        let new = format!("{}: {}", marker.0.label(), c);
        if text.0 != new { text.0 = new; }
    }

    // Toggle row visibility — only non-empty categories appear.
    for (row, mut node) in &mut rows {
        let want = if counts[category_index(row.0)] > 0 {
            Display::Flex
        } else {
            Display::None
        };
        if node.display != want { node.display = want; }
    }
}

fn category_index(c: SpeciesCategory) -> usize {
    match c {
        SpeciesCategory::Photoautotroph => 0,
        SpeciesCategory::HerbivoreL0    => 1,
        SpeciesCategory::HerbivoreL1    => 2,
        SpeciesCategory::HerbivoreL2    => 3,
        SpeciesCategory::HerbivoreL3    => 4,
        SpeciesCategory::CarnivoreL2    => 5,
        SpeciesCategory::CarnivoreL3    => 6,
    }
}
