use bevy::prelude::*;
use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{
        backend::{AutodiffBackend, Backend},
        Tensor, TensorData,
    },
    optim::{AdamConfig, GradientsParams, Optimizer},
};

/* CPU-Logic
use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;

// ── Backend Configuration ───────────────────────────────────────────────────
pub type MyBackend = NdArray;

*/
use burn_cuda::{Cuda, CudaDevice};


// Set the backend to LibTorch (f32 precision)
pub type MyBackend = Cuda;
pub type MyAutodiffBackend = Autodiff<MyBackend>;
use burn::backend::Autodiff;

use crate::colony::{Organism, OrganismRoot, Heterotroph, Photoautotroph};



// ── Bevy Components & Resources ─────────────────────────────────────────────

const SCAN_RADIUS: f32 = 30.0;
const MAX_NEIGHBORS: usize = 16; // Fixed size for fast tensor batching

#[derive(Component, Default)]
pub struct LocalGraph {
    pub target: Option<Entity>,
    pub non_prey: Vec<Entity>,
    pub potential_prey: Vec<Entity>,
}

#[derive(Resource)]
pub struct ScannerTimer {
    pub timer: Timer,
}

impl Default for ScannerTimer {
    fn default() -> Self {
        Self {
            timer: Timer::from_seconds(1.0, TimerMode::Repeating),
        }
    }
}

// ── Neural Network Architecture (GNN) ───────────────────────────────────────

#[derive(Module, Debug)]
pub struct GnnPursuitModel<B: Backend> {
    // Message Layer: Processes the [Rel X, Rel Y, Rel Z, IsTarget, IsNonPrey, IsPotPrey]
    msg_layer: Linear<B>,
    // Update Layers: Processes the aggregated mean of all neighbor messages
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl<B: Backend> GnnPursuitModel<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            // Input: 6 features -> 16 hidden
            msg_layer: LinearConfig::new(6, 16).init(device),
            // Input: 16 aggregated features -> 16 hidden
            fc1: LinearConfig::new(16, 16).init(device),
            // Output: 5 (Dir X, Dir Y, Dir Z, Speed, Rotation Yaw)
            fc2: LinearConfig::new(16, 5).init(device),
        }
    }

    pub fn forward(&self, neighbors: Tensor<B, 3>) -> Tensor<B, 2> {
        // 1. Message Phase: evaluate each neighbor
        let x = self.msg_layer.forward(neighbors);
        let x = burn::tensor::activation::relu(x);
        
        // 2. Aggregate Phase: collapse the neighbor dimension (Dim 1)
        // This is what makes it a true egocentric Graph Neural Network!
        let x = x.mean_dim(1).squeeze_dim(1); 
        
        // 3. Update Phase: make a decision based on the aggregated context
        let x = self.fc1.forward(x);
        let x = burn::tensor::activation::relu(x);
        let x = self.fc2.forward(x);
        burn::tensor::activation::tanh(x) 
    }
}

// ── THE BULLETPROOF OPTIMIZER WRAPPER ───────────────────────────────────────

pub trait ModelOptimizer {
    fn step_model(&mut self, lr: f64, model: GnnPursuitModel<MyAutodiffBackend>, grads: GradientsParams) -> GnnPursuitModel<MyAutodiffBackend>;
}[patch.crates-io]
cudarc = "=0.18.1"

impl<O> ModelOptimizer for O
where
    O: Optimizer<GnnPursuitModel<MyAutodiffBackend>, MyAutodiffBackend>,
{
    fn step_model(&mut self, lr: f64, model: GnnPursuitModel<MyAutodiffBackend>, grads: GradientsParams) -> GnnPursuitModel<MyAutodiffBackend> {
        self.step(lr, model, grads)
    }
}

pub struct BrainResource {
    pub model: GnnPursuitModel<MyAutodiffBackend>,
    pub optimizer: Box<dyn ModelOptimizer>,
    pub device: <MyAutodiffBackend as Backend>::Device,
}

impl FromWorld for BrainResource {
    fn from_world(_world: &mut World) -> Self {
        let device = CudaDevice::default();
        let model = GnnPursuitModel::new(&device);
        let optimizer = Box::new(AdamConfig::new().init()); 
        
        Self {
            model,
            optimizer,
            device,
        }
    }
}

// ── Bevy Plugin ─────────────────────────────────────────────────────────────

pub struct BehaviourPlugin;

impl Plugin for BehaviourPlugin {
    fn build(&self, app: &mut App) {
        app.init_non_send_resource::<BrainResource>();
        app.init_resource::<ScannerTimer>();
        app.add_systems(Update, (
            initialize_local_graphs,
            scan_environment,
            batched_gnn_think_and_learn
        ).chain());
    }
}

// ── Systems ─────────────────────────────────────────────────────────────────

// Ensures newly reproduced heterotrophs get a graph
fn initialize_local_graphs(
    mut commands: Commands,
    query: Query<Entity, (With<Heterotroph>, Without<LocalGraph>)>,
) {
    for entity in query.iter() {
        commands.entity(entity).insert(LocalGraph::default());
    }
}

// The Scanner Function: Rebuilds the graph edges every 1 second
fn scan_environment(
    time: Res<Time>,
    mut timer: ResMut<ScannerTimer>,
    mut heterotrophs: Query<(Entity, &Transform, &mut LocalGraph), With<Heterotroph>>,
    phototrophs: Query<(Entity, &Transform), With<Photoautotroph>>,
    all_heteros: Query<(Entity, &Transform), With<Heterotroph>>,
) {
    timer.timer.tick(time.delta());
    if !timer.timer.just_finished() {
        return;
    }

    for (h_entity, h_trans, mut graph) in heterotrophs.iter_mut() {
        graph.target = None;
        graph.non_prey.clear();
        graph.potential_prey.clear();

        let mut min_dist = f32::MAX;
        let mut closest_photo = None;

        // 1. Scan for Phototrophs (Prey)
        for (p_entity, p_trans) in phototrophs.iter() {
            let dist = h_trans.translation.distance(p_trans.translation);
            if dist <= SCAN_RADIUS {
                if dist < min_dist {
                    // Push the previous closest into potential prey
                    if let Some(prev_closest) = closest_photo {
                        graph.potential_prey.push(prev_closest);
                    }
                    min_dist = dist;
                    closest_photo = Some(p_entity);
                } else {
                    graph.potential_prey.push(p_entity);
                }
            }
        }
        graph.target = closest_photo;

        // 2. Scan for Heterotrophs (Non-Prey / Competitors)
        for (other_h, o_trans) in all_heteros.iter() {
            if other_h != h_entity {
                let dist = h_trans.translation.distance(o_trans.translation);
                if dist <= SCAN_RADIUS {
                    graph.non_prey.push(other_h);
                }
            }
        }
    }
}

// The Core GNN Loop
fn batched_gnn_think_and_learn(
    time: Res<Time<Virtual>>,
    mut brain: NonSendMut<BrainResource>,
    mut organisms: Query<&mut Organism>,
    graphs: Query<(Entity, &Transform, &LocalGraph), With<Heterotroph>>,
    transforms: Query<&Transform>, // Global lookup for any entity's position
) {
    if time.is_paused() {
        return;
    }

    let mut batch_inputs = Vec::new();
    let mut ideal_outputs_raw = Vec::new();
    let mut active_entities = Vec::new();

    for (h_entity, h_trans, graph) in graphs.iter() {
        // If there is no target, we skip training to simulate "wandering/idle"
        let target_entity = match graph.target {
            Some(e) => e,
            None => continue, 
        };

        // Ensure the target still exists in the world
        let target_pos = match transforms.get(target_entity) {
            Ok(t) => t.translation,
            Err(_) => continue,
        };

        let mut neighbors_raw = Vec::new();
        let mut neighbor_count = 0;

        // Helper closure to build edge features [Rel X, Rel Y, Rel Z, Target, NonPrey, PotPrey]
        let mut add_neighbor = |entity: Entity, edge_type: [f32; 3]| {
            if neighbor_count >= MAX_NEIGHBORS { return; }
            if let Ok(t) = transforms.get(entity) {
                let rel = t.translation - h_trans.translation;
                neighbors_raw.extend_from_slice(&[rel.x, rel.y, rel.z, edge_type[0], edge_type[1], edge_type[2]]);
                neighbor_count += 1;
            }
        };

        // Add Target Edge (One-Hot: [1, 0, 0])
        add_neighbor(target_entity, [1.0, 0.0, 0.0]);

        // Add Potential Prey Edges (One-Hot: [0, 0, 1])
        for &prey_entity in graph.potential_prey.iter() {
            add_neighbor(prey_entity, [0.0, 0.0, 1.0]);
        }

        // Add Non-Prey Edges (One-Hot: [0, 1, 0])
        for &competitor_entity in graph.non_prey.iter() {
            add_neighbor(competitor_entity, [0.0, 1.0, 0.0]);
        }

        // Pad the remaining slots with zeros to maintain tensor shape
        for _ in neighbor_count..MAX_NEIGHBORS {
            neighbors_raw.extend_from_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        }

        // Calculate proxy ideal behavior for Reward/Punishment 
        // (MSE loss towards this ideal vector enforces distance minimization)
        let relative_pos = target_pos - h_trans.translation;
        let ideal_dir = relative_pos.normalize_or_zero();
        
        ideal_outputs_raw.push(ideal_dir.x);
        ideal_outputs_raw.push(ideal_dir.y);
        ideal_outputs_raw.push(ideal_dir.z);
        ideal_outputs_raw.push(1.0); // Ideal speed (max)
        
        let forward = Vec3::Z;
        let dot = forward.dot(ideal_dir);
        let cross = forward.cross(ideal_dir);
        let ideal_yaw = f32::atan2(cross.y, dot);
        ideal_outputs_raw.push(ideal_yaw.clamp(-1.0, 1.0));

        batch_inputs.extend(neighbors_raw);
        active_entities.push(h_entity);
    }

    let batch_size = active_entities.len();
    if batch_size == 0 { return; }

    // Convert data to Tensors
    // Input is 3D: [Batch Size, Number of Neighbors, 6 Features per Neighbor]
    let input_tensor = Tensor::<MyAutodiffBackend, 3>::from_data(
        TensorData::new(batch_inputs, [batch_size, MAX_NEIGHBORS, 6]),
        &brain.device,
    );
    
    let target_tensor = Tensor::<MyAutodiffBackend, 2>::from_data(
        TensorData::new(ideal_outputs_raw, [batch_size, 5]),
        &brain.device,
    );

    // Forward, Loss, and Backprop
    let output_tensor = brain.model.forward(input_tensor);
    let loss = burn::nn::loss::MseLoss::new().forward(output_tensor.clone(), target_tensor, burn::nn::loss::Reduction::Mean);
    
    let gradients = loss.backward();
    
    // Decouple borrows safely
    let cloned_model = brain.model.clone();
    let grad_params = GradientsParams::from_grads(gradients, &brain.model);
    
    let new_model = brain.optimizer.step_model(1e-3, cloned_model, grad_params);
    brain.model = new_model;

    // Apply Decisions
    let output_data = output_tensor.into_data().into_vec::<f32>().expect("Failed to convert tensor to Vec<f32>");

    for (i, entity) in active_entities.into_iter().enumerate() {
        if let Ok(mut organism) = organisms.get_mut(entity) {
            let offset = i * 5;
            
            let dir_x = output_data[offset + 0];
            let dir_y = output_data[offset + 1];
            let dir_z = output_data[offset + 2];
            let speed_out = output_data[offset + 3];
            let yaw_out = output_data[offset + 4];

            let new_dir = Vec3::new(dir_x, 0.0, dir_z);
            if new_dir.length_squared() > 0.01 {
                organism.movement_direction = new_dir.normalize();
            }

            organism.movement_speed = ((speed_out + 1.0) / 2.0) * 20.0;

            let current_rotation = organism.target_rotation; // Fallback to current target
            let target_rotation = Quat::from_rotation_y(yaw_out * std::f32::consts::PI);
            
            organism.target_rotation = current_rotation.slerp(target_rotation, 0.1);
        }
    }
}
