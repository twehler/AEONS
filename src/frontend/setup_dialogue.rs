// In-engine "new world" setup dialogue (the no-`.aeonsw` boot path).
//
// When AEONS is launched with `--setup` (the launcher's "Continue without a file"),
// the world load is deferred (`WorldSource::Pending`) and this bevy_ui modal is
// shown. The user picks a `.glb` to import OR a flat world, sets the water level +
// map dimensions (+ an Advanced block of population/training/time-speed), and
// confirms. On confirm we insert the chosen resources, set `WorldSource` (kicking
// off the deferred load / flat generation), and boot into the Map Editor.
//
// Numeric fields use simple −/+ steppers (no text-entry plumbing); the source is a
// pair of toggle buttons; AI-training is a toggle. Everything stays runtime-editable
// later in the stats panel, so the dialogue only needs sensible starting points.

use bevy::prelude::*;

use crate::simulation_settings::{
    AiTrainingMode, MaxHerbivores, MaxPhotoautotrophs, OrganismPoolSize,
    StartEmptyColony, StartHeterotrophs, StartPhotoautotrophs, TimeSpeed, WindowMode,
    DEFAULT_MAP_X, DEFAULT_MAP_Z, DEFAULT_MAX_HERBIVORES, DEFAULT_MAX_PHOTOAUTOTROPHS,
    DEFAULT_START_HETEROTROPHS, DEFAULT_START_PHOTOAUTOTROPHS,
};
use crate::environment::{WaterLevel, DEFAULT_WATER_LEVEL};
use crate::world_geometry::{LoadedWorldPath, MapSize, WorldSource};


// ── Plugin ──────────────────────────────────────────────────────────────────────

pub struct SetupDialoguePlugin;

impl Plugin for SetupDialoguePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SetupActive>()
            .init_resource::<SetupConfig>()
            .add_systems(Update, (
                spawn_setup_ui,
                handle_source_buttons,
                handle_steppers,
                handle_training_toggle,
                sync_setup_labels,
                handle_confirm,
            ));
    }
}


// ── Resources ─────────────────────────────────────────────────────────────────────

/// Active only on the `--setup` boot path. The modal shows + the dialogue systems
/// run only while this is true; confirming clears it.
#[derive(Resource, Default)]
pub struct SetupActive(pub bool);

/// Whether the new world is an imported `.glb` or a generated flat world.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum WorldChoice {
    #[default]
    Flat,
    Glb,
}

/// The dialogue's working values (seeded from defaults; the stats panel can change
/// them later, so these are only starting points).
#[derive(Resource)]
pub struct SetupConfig {
    pub choice:      WorldChoice,
    pub glb_path:    Option<String>,
    pub water:       f32,
    pub map_x:       f32,
    pub map_z:       f32,
    pub max_photo:   f32,
    pub max_herb:    f32,
    pub start_photo: f32,
    pub start_het:   f32,
    pub time_speed:  f32,
    pub training:    bool,
}

impl Default for SetupConfig {
    fn default() -> Self {
        Self {
            choice:      WorldChoice::Flat,
            glb_path:    None,
            water:       DEFAULT_WATER_LEVEL,
            map_x:       DEFAULT_MAP_X,
            map_z:       DEFAULT_MAP_Z,
            max_photo:   DEFAULT_MAX_PHOTOAUTOTROPHS as f32,
            max_herb:    DEFAULT_MAX_HERBIVORES as f32,
            start_photo: DEFAULT_START_PHOTOAUTOTROPHS as f32,
            start_het:   DEFAULT_START_HETEROTROPHS as f32,
            time_speed:  1.0,
            training:    false,
        }
    }
}


// ── Field model (the steppers) ─────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Field {
    Water, MapX, MapZ, MaxPhoto, MaxHerb, StartPhoto, StartHet, TimeSpeed,
}

impl Field {
    /// `(label, step, min, max, integral)`.
    fn def(self) -> (&'static str, f32, f32, f32, bool) {
        match self {
            Field::Water      => ("Water level",        5.0,  -8192.0, 8192.0, false),
            Field::MapX       => ("Map size X",          10.0,    50.0, 8192.0, false),
            Field::MapZ       => ("Map size Z",          10.0,    50.0, 8192.0, false),
            Field::MaxPhoto   => ("Max phototrophs",    100.0,     1.0, 100000.0, true),
            Field::MaxHerb    => ("Max herbivores",      10.0,     1.0, 100000.0, true),
            Field::StartPhoto => ("Start phototrophs",   50.0,     0.0, 100000.0, true),
            Field::StartHet   => ("Start heterotrophs",  10.0,     0.0, 100000.0, true),
            Field::TimeSpeed  => ("Time speed",           0.5,    0.01,  100.0, false),
        }
    }
    fn get(self, c: &SetupConfig) -> f32 {
        match self {
            Field::Water => c.water, Field::MapX => c.map_x, Field::MapZ => c.map_z,
            Field::MaxPhoto => c.max_photo, Field::MaxHerb => c.max_herb,
            Field::StartPhoto => c.start_photo, Field::StartHet => c.start_het,
            Field::TimeSpeed => c.time_speed,
        }
    }
    fn set(self, c: &mut SetupConfig, v: f32) {
        match self {
            Field::Water => c.water = v, Field::MapX => c.map_x = v, Field::MapZ => c.map_z = v,
            Field::MaxPhoto => c.max_photo = v, Field::MaxHerb => c.max_herb = v,
            Field::StartPhoto => c.start_photo = v, Field::StartHet => c.start_het = v,
            Field::TimeSpeed => c.time_speed = v,
        }
    }
    fn display(self, c: &SetupConfig) -> String {
        let (_, _, _, _, integral) = self.def();
        let v = self.get(c);
        if integral { format!("{}", v.round() as i64) } else { format!("{v:.1}") }
    }
}

const ADVANCED: [Field; 5] = [
    Field::MaxPhoto, Field::MaxHerb, Field::StartPhoto, Field::StartHet, Field::TimeSpeed,
];
const BASIC: [Field; 3] = [Field::Water, Field::MapX, Field::MapZ];


// ── Markers ─────────────────────────────────────────────────────────────────────

#[derive(Component)] struct SetupRoot;
#[derive(Component)] struct SourceButton(WorldChoice);
#[derive(Component)] struct SourceLabel; // shows the chosen source + glb path
#[derive(Component)] struct StepButton { field: Field, dir: f32 }
#[derive(Component)] struct ValueLabel(Field);
#[derive(Component)] struct TrainingButton;
#[derive(Component)] struct ConfirmButton;


// ── Colours ─────────────────────────────────────────────────────────────────────

const PANEL_BG:   Color = Color::srgba(0.10, 0.11, 0.14, 0.98);
const SCRIM:      Color = Color::srgba(0.0, 0.0, 0.0, 0.6);
const BTN:        Color = Color::srgb(0.20, 0.30, 0.40);
const BTN_HOVER:  Color = Color::srgb(0.30, 0.42, 0.55);
const BTN_ACTIVE: Color = Color::srgb(0.20, 0.50, 0.60);
const CONFIRM:    Color = Color::srgb(0.20, 0.50, 0.35);


// ── Spawn the modal (once, while active) ─────────────────────────────────────────

fn spawn_setup_ui(
    active:     Res<SetupActive>,
    config:     Res<SetupConfig>,
    existing:   Query<(), With<SetupRoot>>,
    mut commands: Commands,
) {
    if !active.0 { return; }
    if !existing.is_empty() { return; } // already spawned

    commands
        .spawn((
            SetupRoot,
            Node {
                position_type:   PositionType::Absolute,
                left: Val::Px(0.0), top: Val::Px(0.0),
                width:  Val::Percent(100.0),
                height: Val::Percent(100.0),
                align_items:     AlignItems::Center,
                justify_content: JustifyContent::Center,
                ..default()
            },
            BackgroundColor(SCRIM),
            GlobalZIndex(1000),
        ))
        .with_children(|scrim| {
            scrim
                .spawn((
                    Node {
                        flex_direction: FlexDirection::Column,
                        width:   Val::Px(440.0),
                        padding: UiRect::all(Val::Px(16.0)),
                        row_gap: Val::Px(6.0),
                        ..default()
                    },
                    BackgroundColor(PANEL_BG),
                ))
                .with_children(|p| {
                    title(p, "New World");

                    // Source choice.
                    section(p, "World source");
                    p.spawn(Node { column_gap: Val::Px(8.0), ..default() }).with_children(|row| {
                        source_btn(row, "Flat world", WorldChoice::Flat);
                        source_btn(row, "Import .glb…", WorldChoice::Glb);
                    });
                    p.spawn((
                        SourceLabel,
                        Text::new(source_text(&config)),
                        TextFont { font_size: 12.0, ..default() },
                        TextColor(Color::srgb(0.7, 0.7, 0.7)),
                        Pickable::IGNORE,
                    ));

                    section(p, "World settings");
                    for f in BASIC { stepper_row(p, f, &config); }

                    section(p, "Advanced");
                    for f in ADVANCED { stepper_row(p, f, &config); }
                    // AI-training toggle.
                    p.spawn((
                        TrainingButton, Button,
                        Node {
                            height: Val::Px(26.0), padding: UiRect::all(Val::Px(4.0)),
                            align_items: AlignItems::Center, justify_content: JustifyContent::Center,
                            ..default()
                        },
                        BackgroundColor(BTN),
                    )).with_children(|b| {
                        b.spawn((Text::new(training_text(config.training)),
                                 TextFont { font_size: 13.0, ..default() },
                                 TextColor(Color::WHITE), Pickable::IGNORE));
                    });

                    // Confirm.
                    p.spawn(Node { height: Val::Px(8.0), ..default() });
                    p.spawn((
                        ConfirmButton, Button,
                        Node {
                            height: Val::Px(34.0),
                            align_items: AlignItems::Center, justify_content: JustifyContent::Center,
                            ..default()
                        },
                        BackgroundColor(CONFIRM),
                    )).with_children(|b| {
                        b.spawn((Text::new("Create World"),
                                 TextFont { font_size: 15.0, ..default() },
                                 TextColor(Color::WHITE), Pickable::IGNORE));
                    });
                });
        });
}

fn title(p: &mut ChildSpawnerCommands, t: &str) {
    p.spawn((Text::new(t), TextFont { font_size: 20.0, ..default() },
             TextColor(Color::WHITE), Pickable::IGNORE));
}
fn section(p: &mut ChildSpawnerCommands, t: &str) {
    p.spawn((Text::new(t), TextFont { font_size: 13.0, ..default() },
             TextColor(Color::srgb(0.55, 0.7, 0.9)),
             Node { margin: UiRect::top(Val::Px(6.0)), ..default() }, Pickable::IGNORE));
}
fn source_btn(row: &mut ChildSpawnerCommands, label: &str, choice: WorldChoice) {
    row.spawn((
        SourceButton(choice), Button,
        Node { flex_grow: 1.0, height: Val::Px(28.0),
               align_items: AlignItems::Center, justify_content: JustifyContent::Center, ..default() },
        BackgroundColor(BTN),
    )).with_children(|b| {
        b.spawn((Text::new(label), TextFont { font_size: 13.0, ..default() },
                 TextColor(Color::WHITE), Pickable::IGNORE));
    });
}
fn stepper_row(p: &mut ChildSpawnerCommands, field: Field, config: &SetupConfig) {
    let (label, _, _, _, _) = field.def();
    p.spawn(Node { column_gap: Val::Px(6.0), align_items: AlignItems::Center, ..default() })
        .with_children(|row| {
            row.spawn((Text::new(label), TextFont { font_size: 12.0, ..default() },
                       TextColor(Color::srgb(0.8, 0.8, 0.8)),
                       Node { width: Val::Px(160.0), ..default() }, Pickable::IGNORE));
            step_btn(row, field, -1.0, "−");
            row.spawn((ValueLabel(field),
                       Text::new(field.display(config)),
                       TextFont { font_size: 13.0, ..default() }, TextColor(Color::WHITE),
                       Node { width: Val::Px(70.0), justify_content: JustifyContent::Center, ..default() },
                       Pickable::IGNORE));
            step_btn(row, field, 1.0, "+");
        });
}
fn step_btn(row: &mut ChildSpawnerCommands, field: Field, dir: f32, label: &str) {
    row.spawn((
        StepButton { field, dir }, Button,
        Node { width: Val::Px(28.0), height: Val::Px(24.0),
               align_items: AlignItems::Center, justify_content: JustifyContent::Center, ..default() },
        BackgroundColor(BTN),
    )).with_children(|b| {
        b.spawn((Text::new(label), TextFont { font_size: 16.0, ..default() },
                 TextColor(Color::WHITE), Pickable::IGNORE));
    });
}

fn source_text(c: &SetupConfig) -> String {
    match c.choice {
        WorldChoice::Flat => "Source: flat world (seafloor at Y=0)".to_string(),
        WorldChoice::Glb  => match &c.glb_path {
            Some(p) => format!("Source: {p}"),
            None    => "Source: .glb (none chosen — click Import .glb…)".to_string(),
        },
    }
}
fn training_text(on: bool) -> String {
    format!("AI-training mode: {}", if on { "ON" } else { "OFF" })
}


// ── Interaction handlers ──────────────────────────────────────────────────────────

fn handle_source_buttons(
    active:     Res<SetupActive>,
    mut config: ResMut<SetupConfig>,
    mut q: Query<(&Interaction, &SourceButton, &mut BackgroundColor), Changed<Interaction>>,
) {
    if !active.0 { return; }
    for (interaction, btn, mut bg) in &mut q {
        match *interaction {
            Interaction::Pressed => {
                config.choice = btn.0;
                if btn.0 == WorldChoice::Glb {
                    // Blocking native picker (one-shot, same as the launcher/editors).
                    if let Some(p) = rfd::FileDialog::new()
                        .add_filter("glTF (.glb/.gltf)", &["glb", "gltf"])
                        .add_filter("All files", &["*"])
                        .pick_file()
                    {
                        config.glb_path = Some(p.to_string_lossy().into_owned());
                    }
                }
                *bg = BackgroundColor(BTN_ACTIVE);
            }
            Interaction::Hovered => *bg = BackgroundColor(BTN_HOVER),
            Interaction::None    => *bg = BackgroundColor(BTN),
        }
    }
}

fn handle_steppers(
    active:     Res<SetupActive>,
    mut config: ResMut<SetupConfig>,
    mut q: Query<(&Interaction, &StepButton, &mut BackgroundColor), Changed<Interaction>>,
) {
    if !active.0 { return; }
    for (interaction, step, mut bg) in &mut q {
        match *interaction {
            Interaction::Pressed => {
                let (_, st, min, max, _) = step.field.def();
                let v = (step.field.get(&config) + st * step.dir).clamp(min, max);
                step.field.set(&mut config, v);
                *bg = BackgroundColor(BTN_HOVER);
            }
            Interaction::Hovered => *bg = BackgroundColor(BTN_HOVER),
            Interaction::None    => *bg = BackgroundColor(BTN),
        }
    }
}

fn handle_training_toggle(
    active:     Res<SetupActive>,
    mut config: ResMut<SetupConfig>,
    mut q: Query<(&Interaction, &mut BackgroundColor), (Changed<Interaction>, With<TrainingButton>)>,
) {
    if !active.0 { return; }
    for (interaction, mut bg) in &mut q {
        match *interaction {
            Interaction::Pressed => { config.training = !config.training; *bg = BackgroundColor(BTN_HOVER); }
            Interaction::Hovered => *bg = BackgroundColor(BTN_HOVER),
            Interaction::None    => *bg = BackgroundColor(BTN),
        }
    }
}

/// Keep value labels + source/training text in sync with `SetupConfig`. The three
/// `&mut Text` queries are kept disjoint by filters (ValueLabel / SourceLabel /
/// neither) so Bevy's access checker is satisfied.
#[allow(clippy::type_complexity)]
fn sync_setup_labels(
    active: Res<SetupActive>,
    config: Res<SetupConfig>,
    mut values:   Query<(&ValueLabel, &mut Text), Without<SourceLabel>>,
    mut src:      Query<&mut Text, (With<SourceLabel>, Without<ValueLabel>)>,
    training_q:   Query<&Children, With<TrainingButton>>,
    mut texts:    Query<&mut Text, (Without<ValueLabel>, Without<SourceLabel>)>,
) {
    if !active.0 || !config.is_changed() { return; }
    for (vl, mut text) in &mut values {
        let d = vl.0.display(&config);
        if text.0 != d { text.0 = d; }
    }
    for mut text in &mut src {
        let t = source_text(&config);
        if text.0 != t { text.0 = t; }
    }
    // Training button's child text (a Text with neither ValueLabel nor SourceLabel).
    for children in &training_q {
        for child in children.iter() {
            if let Ok(mut text) = texts.get_mut(child) {
                let t = training_text(config.training);
                if text.0 != t { text.0 = t; }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_confirm(
    mut active:   ResMut<SetupActive>,
    config:       Res<SetupConfig>,
    mut commands: Commands,
    mut world_source: ResMut<WorldSource>,
    mut loaded_path:  ResMut<LoadedWorldPath>,
    roots:        Query<Entity, With<SetupRoot>>,
    q: Query<&Interaction, (Changed<Interaction>, With<ConfirmButton>)>,
) {
    if !active.0 { return; }
    let mut confirmed = false;
    for interaction in &q {
        if matches!(interaction, Interaction::Pressed) { confirmed = true; }
    }
    if !confirmed { return; }

    // Apply the chosen values (the file/flat world is authoritative thereafter).
    commands.insert_resource(MapSize { x: config.map_x.max(1.0), z: config.map_z.max(1.0) });
    commands.insert_resource(WaterLevel(config.water));
    let max_herb = config.max_herb.round().max(1.0) as usize;
    commands.insert_resource(MaxPhotoautotrophs(config.max_photo.round().max(1.0) as usize));
    commands.insert_resource(MaxHerbivores(max_herb));
    commands.insert_resource(OrganismPoolSize((max_herb * 4).max(16)));
    commands.insert_resource(StartPhotoautotrophs(config.start_photo.round().max(0.0) as usize));
    commands.insert_resource(StartHeterotrophs(config.start_het.round().max(0.0) as usize));
    commands.insert_resource(AiTrainingMode(config.training));
    commands.insert_resource(TimeSpeed(config.time_speed.max(0.01)));
    // New world starts EMPTY — the user builds the colony in the editors.
    commands.insert_resource(StartEmptyColony(true));

    // Kick off the deferred world load.
    match config.choice {
        WorldChoice::Glb if config.glb_path.is_some() => {
            let path = config.glb_path.clone().unwrap();
            loaded_path.0 = path.clone();
            *world_source = WorldSource::Glb(path);
        }
        _ => {
            *world_source = WorldSource::Flat; // flat fallback (also if .glb not chosen)
        }
    }

    // Boot into the Map Editor (paused); the user designs the map, then builds a
    // colony, then starts the sim from the mode bar.
    commands.insert_resource(WindowMode::MapEditor);

    // Tear down the modal + deactivate.
    for e in &roots { commands.entity(e).despawn(); }
    active.0 = false;
    info!("setup dialogue confirmed: {} world, map {}x{}, water {}",
          match config.choice { WorldChoice::Glb => "glb", WorldChoice::Flat => "flat" },
          config.map_x, config.map_z, config.water);
}
