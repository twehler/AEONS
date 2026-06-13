// Predation — heterotrophs eat photoautotrophs one body part at a time.
//
// Each `OrganismContactEvent` carries the body-part indices that touched.
// The predator consumes the single contacted prey body part, takes a
// proportional share of the prey's energy, and soft-deletes the part
// (cells cleared, `consumed = true`, child mesh despawned). The prey
// organism despawns only once all its body parts are eaten — making
// predation gradient (many-part prey can be nibbled over encounters).

use bevy::prelude::*;
use std::collections::HashSet;

use crate::cell::*;
use crate::colony::*;
use crate::organism_collision::OrganismContactEvent;


use crate::simulation_settings::ENERGY_TRANSFER_RATE;


pub struct PredationPlugin;

impl Plugin for PredationPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<OrganismContactEvent>();
        app.add_systems(Update, emit_proximity_predation.before(predation_system));
        app.add_systems(Update, predation_system);
    }
}


/// Emit a predation contact event when a LIMB herbivore gets within `EAT_RADIUS` of
/// a phototroph — eating by PROXIMITY rather than physical collider contact (the
/// limb↔prey physical collision is filtered in `SelfCollisionFilter`). Reaching prey
/// then eating is preserved, but herbivores pass THROUGH prey so no dense limb↔prey
/// contact pile forms at feeding clusters (that pile craters FPS). Reuses all the
/// existing predation routing/logic via the same event.
pub fn emit_proximity_predation(
    sim_running: Res<crate::simulation_settings::SimulationRunning>,
    grid:        Option<Res<crate::world_model::WorldModelGrid>>,
    bases:       Query<(&bevy::prelude::ChildOf, &crate::cell::BodyPartIndex, &bevy::prelude::GlobalTransform)>,
    orgs:        Query<&Organism>,
    heteros:     Query<(), With<Heterotroph>>,
    mut out:     MessageWriter<OrganismContactEvent>,
) {
    if !sim_running.0 { return; }
    let Some(grid) = grid else { return };
    let eat = crate::simulation_settings::EAT_RADIUS;
    for (co, idx, gt) in &bases {
        if idx.0 != 0 { continue; } // base body part
        let root = co.parent();
        if !heteros.contains(root) { continue; }
        if orgs.get(root).map(|o| o.movement_mode.is_sliding()).unwrap_or(true) { continue; } // limb only
        let pos = gt.translation();
        if let Some((_, dist, prey)) = crate::world_model::nearest_prey(&grid, pos) {
            if dist < eat {
                // Eat the prey's trunk (part 0) → consumes the whole phototroph.
                out.write(OrganismContactEvent { a: root, b: prey, body_part_a: 0, body_part_b: 0 });
            }
        }
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
    // Per-frame dedup: one body part can produce multiple contact events
    // per tick; eating it twice would double-credit the predator (or let
    // two predators share it). `Local<HashSet>` is reused (cleared) across
    // ticks rather than reallocated.
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
            // Losing body_parts[0] (the structural root/trunk) is fatal:
            // surviving branches don't cascade with it (flat hierarchy)
            // and can't regrow (`continuous_growth` seeds from
            // body_parts[0].ocg), so eating the trunk kills the organism.
            let trunk_lost = prey_bp_idx == 0;
            Some((share, new_alive == 0 || bilateral_collapse || trunk_lost))
        };

        let mutation_result = if prey_is_photo {
            let Ok(mut prey_org) = phototrophs.get_mut(prey) else { continue };
            mutate_prey(&mut prey_org, &mut commands, &mut already_despawned)
        } else {
            // Carnivore eating a heterotroph — same logic, different query.
            // Predator credit re-borrows heterotrophs mutably; safe because
            // the prey borrow ends here (NLL releases it before get_mut).
            let Ok(mut prey_org) = heterotrophs.get_mut(prey) else { continue };
            mutate_prey(&mut prey_org, &mut commands, &mut already_despawned)
        };
        let Some((energy_share, prey_dead)) = mutation_result else { continue };

        // ── Credit predator ──────────────────────────────────────────────
        if let Ok(mut pred_org) = heterotrophs.get_mut(predator) {
            pred_org.energy += energy_share * ENERGY_TRANSFER_RATE;
            // Bump predation counter (saturating). Hetero pools read its
            // per-tick delta as their eat-event reward.
            pred_org.predations = pred_org.predations.saturating_add(1);
            // +0.6 dopamine per eat, clamped at 1.0 (consumed as a reward).
            pred_org.dopamine = (pred_org.dopamine + 0.6).min(1.0);
        }

        // ── Despawn the eaten body part's child mesh entity ──────────────
        // The `body_parts` slot stays in place (`consumed = true`), not
        // reindexed, so sibling `BodyPartIndex` values stay stable.
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
