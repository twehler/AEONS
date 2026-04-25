use bevy::prelude::*;
use std::collections::HashSet;
use crate::colony::{Heterotroph, Organism, Photoautotroph};
use crate::organism_collision::OrganismContactEvent; 

// ── Constants ────────────────────────────────────────────────────────────────

const ENERGY_TRANSFER_RATE: f32 = 0.8; // 80% of prey's energy is transferred

// ── Plugin ───────────────────────────────────────────────────────────────────

pub struct PredationPlugin;

impl Plugin for PredationPlugin {
    fn build(&self, app: &mut App) {
        // 1. THE FIX: Initialize the message queue so MessageReader can find it!
        app.add_message::<OrganismContactEvent>();
        
        // 2. Register your system
        app.add_systems(Update, predation_system);
    }
}

// ── Systems ──────────────────────────────────────────────────────────────────

fn predation_system(
    mut commands: Commands,
    mut contact_events: MessageReader<OrganismContactEvent>,
    mut heterotrophs: Query<&mut Organism, With<Heterotroph>>,
    phototrophs: Query<&Organism, (With<Photoautotroph>, Without<Heterotroph>)>,
) {
    // Keep track to prevent double-eating if multiple events fire in one frame
    let mut despawned_phototrophs = HashSet::new();

    for event in contact_events.read() {
        // 1. Identify which entity is the predator and which is the prey
        let (predator_entity, prey_entity) = if heterotrophs.contains(event.a) && phototrophs.contains(event.b) {
            (event.a, event.b)
        } else if heterotrophs.contains(event.b) && phototrophs.contains(event.a) {
            (event.b, event.a)
        } else {
            // Not a heterotroph hitting a phototroph (could be two plants bumping)
            continue; 
        };

        // 2. Skip if the prey was already eaten by someone else this frame
        if despawned_phototrophs.contains(&prey_entity) {
            continue;
        }

        // 3. Get the prey's energy (immutable read)
        let prey_energy = if let Ok(p_org) = phototrophs.get(prey_entity) {
            p_org.energy
        } else {
            continue;
        };

        // 4. Transfer energy and eradicate the prey
        if let Ok(mut h_org) = heterotrophs.get_mut(predator_entity) {
            h_org.energy += prey_energy * ENERGY_TRANSFER_RATE;
            
            commands.entity(prey_entity).despawn();
            despawned_phototrophs.insert(prey_entity);
        }
    }
}
