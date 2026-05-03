// Predation — heterotrophs eat photoautotrophs one body part at a time.
//
// New architecture: every `OrganismContactEvent` from `organism_collision.rs`
// carries the body-part indices that touched on each side. When a
// heterotroph touches a photoautotroph, the predator consumes the *single*
// body part of the prey that made contact, takes a proportional share of
// the prey's total energy, and the body part is soft-deleted (cells
// cleared, `consumed = true`, child mesh entity despawned). The prey
// organism only despawns once *all* of its body parts have been eaten.
//
// This makes predation gradient: a prey with many body parts can be
// nibbled over multiple encounters, surviving long enough to spawn
// reactions in the world (escape attempts, brain training signals, etc).
// It also keeps the contact-event system compatible with future
// non-fatal interactions (mating, parasitism) without further changes.

use bevy::prelude::*;
use std::collections::HashSet;

use crate::cell::*;
use crate::colony::*;
use crate::organism_collision::OrganismContactEvent;


/// Fraction of the prey body part's energy share that becomes predator
/// energy. The "lost" 20% models metabolic inefficiency in digestion.
const ENERGY_TRANSFER_RATE: f32 = 0.8;


pub struct PredationPlugin;

impl Plugin for PredationPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<OrganismContactEvent>();
        app.add_systems(Update, predation_system);
    }
}


fn predation_system(
    mut commands:        Commands,
    mut contact_events:  MessageReader<OrganismContactEvent>,
    mut heterotrophs:    Query<&mut Organism, (With<Heterotroph>, Without<Photoautotroph>)>,
    mut phototrophs:     Query<&mut Organism, (With<Photoautotroph>, Without<Heterotroph>)>,
    children_query:      Query<&Children>,
    body_part_idx_query: Query<&BodyPartIndex>,
) {
    // Per-frame dedup: a single body part can be the subject of more than
    // one contact event in the same tick (multiple cell-pair contacts feed
    // distinct events). Eating it more than once would double-credit the
    // predator and let multiple predators "share" the same body part.
    let mut already_eaten:    HashSet<(Entity, usize)> = HashSet::new();
    let mut already_despawned: HashSet<Entity>          = HashSet::new();

    for event in contact_events.read() {
        // Identify predator + prey + which body-part index belongs to prey.
        let a_is_pred = heterotrophs.contains(event.a);
        let b_is_pred = heterotrophs.contains(event.b);
        let a_is_prey = phototrophs.contains(event.a);
        let b_is_prey = phototrophs.contains(event.b);

        let (predator, prey, prey_bp_idx) = if a_is_pred && b_is_prey {
            (event.a, event.b, event.body_part_b)
        } else if b_is_pred && a_is_prey {
            (event.b, event.a, event.body_part_a)
        } else {
            // Either both predators, both prey, or one despawned — nothing
            // for predation to do.
            continue;
        };

        if already_despawned.contains(&prey) { continue; }
        if !already_eaten.insert((prey, prey_bp_idx)) { continue; }

        // ── Mutate prey: consume the body part, decrement energy ─────────
        let (energy_share, prey_dead) = {
            let Ok(mut prey_org) = phototrophs.get_mut(prey) else { continue };

            let alive = prey_org.alive_body_part_count();
            if alive == 0 {
                // No alive body parts — should already be despawned. Be
                // defensive and despawn now.
                already_despawned.insert(prey);
                commands.entity(prey).despawn();
                continue;
            }

            // Bounds + already-eaten guards. Necessary because
            // `consumed_body_parts` is a per-frame set; multiple events
            // for the same body part are rare but possible across frames
            // if the contact persists.
            if prey_bp_idx >= prey_org.body_parts.len()
                || !prey_org.body_parts[prey_bp_idx].is_alive()
            {
                continue;
            }

            // Even split of remaining prey energy across alive body parts.
            // The eaten body part's share moves to the predator (minus
            // metabolic loss); remaining parts retain their shares.
            let share = prey_org.energy / alive as f32;
            prey_org.energy = (prey_org.energy - share).max(0.0);

            let bp = &mut prey_org.body_parts[prey_bp_idx];
            bp.consumed = true;
            bp.cells.clear();

            let new_alive = prey_org.alive_body_part_count();
            (share, new_alive == 0)
        };

        // ── Credit predator ──────────────────────────────────────────────
        if let Ok(mut pred_org) = heterotrophs.get_mut(predator) {
            pred_org.energy += energy_share * ENERGY_TRANSFER_RATE;
        }

        // ── Despawn the eaten body part's child mesh entity ──────────────
        // We don't reindex the prey's `body_parts` Vec; the slot stays in
        // place with `consumed = true`. That keeps sibling children's
        // `BodyPartIndex` stable and avoids walking the children list to
        // patch components.
        if let Ok(children) = children_query.get(prey) {
            for child in children.iter() {
                if let Ok(idx) = body_part_idx_query.get(child) {
                    if idx.0 == prey_bp_idx {
                        commands.entity(child).despawn();
                        break;
                    }
                }
            }
        }

        // ── Despawn the prey root once every body part has been eaten ─────
        if prey_dead {
            commands.entity(prey).despawn();
            already_despawned.insert(prey);
        }
    }
}
