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
    mut commands:           Commands,
    mut contact_events:     MessageReader<OrganismContactEvent>,
    mut heterotrophs:       Query<&mut Organism, (With<Heterotroph>, Without<Photoautotroph>)>,
    mut phototrophs:        Query<&mut Organism, (With<Photoautotroph>, Without<Heterotroph>)>,
    carnivores:             Query<(), With<crate::colony::Carnivore>>,
    children_query:         Query<&Children>,
    body_part_idx_query:    Query<&BodyPartIndex>,
    mut already_eaten:      Local<HashSet<(Entity, usize)>>,
    mut already_despawned:  Local<HashSet<Entity>>,
) {
    // Per-frame dedup: a single body part can be the subject of more than
    // one contact event in the same tick (multiple cell-pair contacts feed
    // distinct events). Eating it more than once would double-credit the
    // predator and let multiple predators "share" the same body part.
    // `Local<HashSet>` reuses allocations across ticks instead of
    // freshly allocating them at the top of every Update — they're just
    // cleared.
    already_eaten.clear();
    already_despawned.clear();

    for event in contact_events.read() {
        // Predator-prey routing:
        //   * Heterotroph + Photoautotroph  ⇒  hetero eats photo (legacy).
        //   * Carnivore (a heterotroph subset) + Heterotroph  ⇒
        //     carnivore eats the other hetero, regardless of the
        //     prey's own classification. A carnivore eating another
        //     carnivore is allowed (food-chain dynamics).
        let a_is_hetero = heterotrophs.contains(event.a);
        let b_is_hetero = heterotrophs.contains(event.b);
        let a_is_photo  = phototrophs.contains(event.a);
        let b_is_photo  = phototrophs.contains(event.b);
        let a_is_carn   = carnivores.contains(event.a);
        let b_is_carn   = carnivores.contains(event.b);

        // (predator, prey, prey_bp_idx, prey_is_photo)
        let (predator, prey, prey_bp_idx, prey_is_photo) =
            if a_is_hetero && b_is_photo {
                (event.a, event.b, event.body_part_b, true)
            } else if b_is_hetero && a_is_photo {
                (event.b, event.a, event.body_part_a, true)
            } else if a_is_carn && b_is_hetero && event.a != event.b {
                (event.a, event.b, event.body_part_b, false)
            } else if b_is_carn && a_is_hetero && event.a != event.b {
                (event.b, event.a, event.body_part_a, false)
            } else {
                continue;
            };

        if already_despawned.contains(&prey) { continue; }
        if !already_eaten.insert((prey, prey_bp_idx)) { continue; }

        // ── Mutate prey: consume the body part, decrement energy ─────────
        //
        // Shared mutation logic for either prey type. Returns
        // `(energy_share, prey_dead)` or `None` if the prey reference
        // can't be acquired or the body-part index is stale.
        let mutate_prey = |prey_org: &mut Organism, commands: &mut Commands,
                           already_despawned: &mut HashSet<Entity>| -> Option<(f32, bool)> {
            let alive = prey_org.alive_body_part_count();
            if alive == 0 {
                already_despawned.insert(prey);
                commands.entity(prey).try_despawn();
                return None;
            }
            if prey_bp_idx >= prey_org.body_parts.len()
                || !prey_org.body_parts[prey_bp_idx].is_alive()
            {
                return None;
            }
            let share = prey_org.energy / alive as f32;
            prey_org.energy = (prey_org.energy - share).max(0.0);

            let bp_counts = prey_org.body_parts[prey_bp_idx].cell_counts();
            prey_org.photo_cell_count     -= bp_counts.0 as i32;
            prey_org.non_photo_cell_count -= bp_counts.1 as i32;

            let bp = &mut prey_org.body_parts[prey_bp_idx];
            bp.consumed = true;
            bp.cells.clear();
            bp.ocg.clear();

            prey_org.recompute_bounding_radius();
            let new_alive = prey_org.alive_body_part_count();
            // Bilateral organisms can't survive losing a half.
            let bilateral_collapse = matches!(prey_org.symmetry, Symmetry::Bilateral);
            // Losing body_parts[0] (the structural root / trunk) kills
            // the organism. For variable-form plants this matters
            // visually: after the flat-hierarchy fix branches no
            // longer cascade with their nominal parent body part, so
            // eating the trunk would otherwise leave the surviving
            // branches floating in space (and unable to grow back,
            // since `continuous_growth` seeds from
            // `body_parts[0].ocg`). Treating index-0 consumption as
            // fatal restores the intuitive "plant is dead once the
            // trunk is gone" semantics across every symmetry.
            let trunk_lost = prey_bp_idx == 0;
            Some((share, new_alive == 0 || bilateral_collapse || trunk_lost))
        };

        let mutation_result = if prey_is_photo {
            let Ok(mut prey_org) = phototrophs.get_mut(prey) else { continue };
            mutate_prey(&mut prey_org, &mut commands, &mut already_despawned)
        } else {
            // Carnivore eating a heterotroph. Same logic, different
            // query. Predator credit (below) re-borrows heterotrophs
            // mutably for the predator entity — that's safe because
            // the prey borrow ends with this block, and Bevy's NLL
            // releases the reference before the next get_mut.
            let Ok(mut prey_org) = heterotrophs.get_mut(prey) else { continue };
            mutate_prey(&mut prey_org, &mut commands, &mut already_despawned)
        };
        let Some((energy_share, prey_dead)) = mutation_result else { continue };

        // ── Credit predator ──────────────────────────────────────────────
        if let Ok(mut pred_org) = heterotrophs.get_mut(predator) {
            pred_org.energy += energy_share * ENERGY_TRANSFER_RATE;
            // Bump the predation counter (saturating, since u8 wraps).
            // The active hetero pools read the per-tick delta of this
            // field as their eat-event reward signal.
            pred_org.predations = pred_org.predations.saturating_add(1);
            // RL reward: +0.6 dopamine on every successful consumption,
            // clamped at 1.0. The herbivore brain's REINFORCE update
            // consumes the per-tick `Δdopamine` as its reward signal.
            pred_org.dopamine = (pred_org.dopamine + 0.6).min(1.0);
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
                        commands.entity(child).try_despawn();
                        break;
                    }
                }
            }
        }

        // ── Despawn the prey root once every body part has been eaten ─────
        if prey_dead {
            commands.entity(prey).try_despawn();
            already_despawned.insert(prey);
        }
    }
}
