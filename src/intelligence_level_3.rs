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

use crate::colony::{Organism, OrganismRoot, Heterotroph, Photoautotroph, MAXIMUM_ORGANISMS};



// ── Bevy Components & Resources ─────────────────────────────────────────────

const SCAN_RADIUS: f32 = 60.0;
const MAX_NEIGHBORS: usize = 16; // Fixed size for fast tensor batching
const INPUTS: usize = 17;

// ── Warmup Logic ────────────────────────────────────────────────────────────

pub fn warmup_level_3_cache(device: &CudaDevice) {
    // Cast the u32 constant to usize, as Burn requires usize for tensor shapes.
    let batch_size = MAXIMUM_ORGANISMS as usize;
        
    println!("#############################################################");
    println!("###### Priming GNN training kernels for batch size: {} ######", batch_size);
    println!("#############################################################");


    // Initialize the dummy model and optimizer
    let model = GnnPursuitModel::<MyAutodiffBackend>::new(device);
    let mut optimizer: Box<dyn ModelOptimizer> = Box::new(AdamConfig::new().init());
    

    let input_tensor = Tensor::<MyAutodiffBackend, 3>::zeros([batch_size, MAX_NEIGHBORS, INPUTS], device);
    let target_tensor = Tensor::<MyAutodiffBackend, 2>::zeros([batch_size, 5], device);
    
    // --- THE FULL TRAINING CYCLE (1 Iteration) ---
    let output_tensor = model.forward(input_tensor);
    
    let loss = burn::nn::loss::MseLoss::new().forward(
        output_tensor, 
        target_tensor, 
        burn::nn::loss::Reduction::Mean
    );
    
    let gradients = loss.backward();
    let grad_params = GradientsParams::from_grads(gradients, &model);
    
    // We bind the result to `_` to suppress Rust compiler warnings about unused variables
    let _ = optimizer.step_model(1e-3, model, grad_params);
    
    println!("################################################################");
    println!("######### Tuning for Intelligence Level 3 complete. ############");
    println!("################################################################");
}





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
// Advanced Model
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

            msg_layer: LinearConfig::new(INPUTS, 32).init(device),

            fc1: LinearConfig::new(32, 32).init(device),
            // Output: 5 (Dir X, Dir Y, Dir Z, Speed, Rotation Yaw)
            fc2: LinearConfig::new(32, 5).init(device),
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



pub trait ModelOptimizer {
    fn step_model(&mut self, lr: f64, model: GnnPursuitModel<MyAutodiffBackend>, grads: GradientsParams) -> GnnPursuitModel<MyAutodiffBackend>;
}

impl<O> ModelOptimizer for O
where
    O: Optimizer<GnnPursuitModel<MyAutodiffBackend>, MyAutodiffBackend>,
{
    fn step_model(&mut self, lr: f64, model: GnnPursuitModel<MyAutodiffBackend>, grads: GradientsParams) -> GnnPursuitModel<MyAutodiffBackend> {
        self.step(lr, model, grads)
    }
}

pub struct BrainResourceLevel3 {
    pub model: GnnPursuitModel<MyAutodiffBackend>,
    pub optimizer: Box<dyn ModelOptimizer>,
    pub device: <MyAutodiffBackend as Backend>::Device,
}

impl FromWorld for BrainResourceLevel3 {
    fn from_world(_world: &mut World) -> Self {
        let device = CudaDevice::default();
        warmup_level_3_cache(&device);

        let model = GnnPursuitModel::new(&device);
        let optimizer = Box::new(AdamConfig::new().init()); 
        
        Self {
            model,
            optimizer,
            device,
        }
    }
}

// ── Systems ─────────────────────────────────────────────────────────────────

// Ensures newly reproduced heterotrophs get a graph
pub fn initialize_local_graphs(
    mut commands: Commands,
    query: Query<Entity, (With<Heterotroph>, Without<LocalGraph>)>,
) {
    for entity in query.iter() {
        commands.entity(entity).insert(LocalGraph::default());
    }
}

// The Scanner Function: Rebuilds the graph edges every 1 second
pub fn scan_environment(
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

pub fn apply_intelligence_level_3(
    time: Res<Time<Virtual>>,
    mut brain: NonSendMut<BrainResourceLevel3>,
    mut organisms: Query<&mut Organism>,
    graphs: Query<(Entity, &Transform, &LocalGraph), With<Heterotroph>>,
    transforms: Query<&Transform>, // Global lookup for any entity's position
) {
    if time.is_paused() {
        return;
    }

    let max_orgs = MAXIMUM_ORGANISMS as usize;

    // Pre-allocate vectors with exact maximum capacity to prevent re-allocations
    let mut batch_inputs = Vec::with_capacity(max_orgs * MAX_NEIGHBORS * INPUTS);
    let mut ideal_outputs = Vec::with_capacity(max_orgs * 5);
    let mut active_entities = Vec::with_capacity(max_orgs);


    for (h_entity, h_trans, graph) in graphs.iter() {
        if active_entities.len() >= max_orgs { break; }

        // 1. Check for target
        let target_entity = match graph.target {
            Some(e) => e,
            None => continue, 
        };

        // 2. Get target position
        let target_pos = match transforms.get(target_entity) {
            Ok(t) => t.translation,
            Err(_) => continue,
        };

        // 3. Fetch the organism to read its internal self-state
        let organism = match organisms.get(h_entity) {
            Ok(o) => o,
            Err(_) => continue,
        };

        // package the 11 new self-state features 

        let self_features = [
            organism.last_movement_speed,
            organism.last_movement_direction.x,
            organism.last_movement_direction.y,
            organism.last_movement_direction.z,
            organism.rotation.x,          // Directly accessing Vec3
            organism.rotation.y,          // Directly accessing Vec3
            organism.rotation.z,          // Directly accessing Vec3
            organism.last_rotation.x,     // Directly accessing Vec3
            organism.last_rotation.y,     // Directly accessing Vec3
            organism.last_rotation.z,     // Directly accessing Vec3
            organism.last_rotation_speed,
        ];

        let mut neighbors_data = [0.0; MAX_NEIGHBORS * INPUTS]; // INPUTS features
        let mut n_count = 0;

        // 5. Build neighbor features
        let mut add_neighbor = |entity: Entity, edge_type: [f32; 3]| {
            if n_count < MAX_NEIGHBORS {
                if let Ok(t) = transforms.get(entity) {
                    let rel = t.translation - h_trans.translation;
                    let offset = n_count * INPUTS;
                    
                    // Write Relative Position (3)
                    neighbors_data[offset..offset + 3].copy_from_slice(&[rel.x, rel.y, rel.z]);
                    // Write Edge Type (3)
                    neighbors_data[offset + 3..offset + 6].copy_from_slice(&edge_type);
                    // Write Self State (11)
                    neighbors_data[offset + 6..offset + INPUTS].copy_from_slice(&self_features);
                    
                    n_count += 1;
                }
            }
        };

        // Add Edges
        add_neighbor(target_entity, [1.0, 0.0, 0.0]);
        for &prey_entity in graph.potential_prey.iter() { add_neighbor(prey_entity, [0.0, 0.0, 1.0]); }
        for &competitor_entity in graph.non_prey.iter() { add_neighbor(competitor_entity, [0.0, 1.0, 0.0]); }

        // Calculate proxy ideal behavior
        let relative_pos = target_pos - h_trans.translation;
        let ideal_dir = relative_pos.normalize_or_zero();
        
        let dot = Vec3::Z.dot(ideal_dir);
        let cross = Vec3::Z.cross(ideal_dir);
        let ideal_yaw = f32::atan2(cross.y, dot).clamp(-1.0, 1.0);

        // Append to batch
        batch_inputs.extend_from_slice(&neighbors_data);
        ideal_outputs.extend_from_slice(&[ideal_dir.x, ideal_dir.y, ideal_dir.z, 1.0, ideal_yaw]);
        active_entities.push(h_entity);
    }


    let active_count = active_entities.len();
    
    // If no one is active, skip GPU work to save power
    if active_count == 0 { return; }

    // 3. INSTANT PADDING: Resize fills the rest of the Vec with 0.0 automatically.
    // This creates "ghost" organisms to perfectly satisfy the MAXIMUM_ORGANISMS batch size.
    batch_inputs.resize(max_orgs * MAX_NEIGHBORS * INPUTS, 0.0);
    ideal_outputs.resize(max_orgs * 5, 0.0);

    // 4. FIXED TENSOR CREATION: The GPU now only ever sees `max_orgs`
    let input_tensor = Tensor::<MyAutodiffBackend, 3>::from_data(
        TensorData::new(batch_inputs, [max_orgs, MAX_NEIGHBORS, INPUTS]),
        &brain.device,
    );
    
    let target_tensor = Tensor::<MyAutodiffBackend, 2>::from_data(
        TensorData::new(ideal_outputs, [max_orgs, 5]),
        &brain.device,
    );

    // Forward, Loss, and Backprop
    let output_tensor = brain.model.forward(input_tensor);
    let loss = burn::nn::loss::MseLoss::new().forward(
        output_tensor.clone(), 
        target_tensor, 
        burn::nn::loss::Reduction::Mean
    );
    
    let gradients = loss.backward();
    
    let cloned_model = brain.model.clone();
    let grad_params = GradientsParams::from_grads(gradients, &brain.model);
    
    let new_model = brain.optimizer.step_model(1e-3, cloned_model, grad_params);
    brain.model = new_model;

    // Apply Decisions
    let output_data = output_tensor.into_data().into_vec::<f32>().expect("Failed to convert tensor");

    // 5. SELECTIVE APPLICATION: We only iterate up to `active_entities`, 
    // completely ignoring the math results of the zero-padded ghosts!
    for (i, entity) in active_entities.into_iter().enumerate() {
        if let Ok(mut organism) = organisms.get_mut(entity) {
            let offset = i * 5;

            let dir_x = output_data[offset + 0];
            // output_data[offset + 1] is Y, ignored for 2D movement
            let dir_z = output_data[offset + 2];
            let speed_out = output_data[offset + 3];
            let yaw_out = output_data[offset + 4];

            let new_dir = Vec3::new(dir_x, 0.0, dir_z);
            if new_dir.length_squared() > 0.01 {
                organism.movement_direction = new_dir.normalize();
            }

            organism.movement_speed = ((speed_out + 1.0) / 2.0) * 20.0;

            let current_rotation = organism.target_rotation; 


            let target_rotation = Vec3::new(0.0, yaw_out * std::f32::consts::PI, 0.0);

            organism.target_rotation = current_rotation.slerp(target_rotation, 0.1);
        }
    }
}
