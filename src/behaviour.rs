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
use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::Autodiff;

use crate::colony::{Organism, OrganismRoot, Heterotroph, Photoautotroph};

// ── Backend Configuration ───────────────────────────────────────────────────
pub type MyBackend = NdArray;
pub type MyAutodiffBackend = Autodiff<MyBackend>;

// ── Neural Network Architecture ─────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct PursuitModel<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
}

impl<B: Backend> PursuitModel<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            fc1: LinearConfig::new(3, 5).init(device),
            fc2: LinearConfig::new(5, 5).init(device),
            fc3: LinearConfig::new(5, 5).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(input);
        let x = burn::tensor::activation::relu(x);
        let x = self.fc2.forward(x);
        let x = burn::tensor::activation::relu(x);
        let x = self.fc3.forward(x);
        burn::tensor::activation::tanh(x) 
    }
}

// ── THE BULLETPROOF OPTIMIZER WRAPPER ───────────────────────────────────────
// This custom trait is 100% "dyn compatible" (object safe). 
// It safely erases the ugly 'OptimizerAdaptor' type.

pub trait ModelOptimizer {
    fn step_model(&mut self, lr: f64, model: PursuitModel<MyAutodiffBackend>, grads: GradientsParams) -> PursuitModel<MyAutodiffBackend>;
}

// Automatically implement our trait for ANY valid Burn optimizer
impl<O> ModelOptimizer for O
where
    O: Optimizer<PursuitModel<MyAutodiffBackend>, MyAutodiffBackend>,
{
    fn step_model(&mut self, lr: f64, model: PursuitModel<MyAutodiffBackend>, grads: GradientsParams) -> PursuitModel<MyAutodiffBackend> {
        // Just proxy to Burn's native step method
        self.step(lr, model, grads)
    }
}

// ── Bevy Resources ──────────────────────────────────────────────────────────

pub struct BrainResource {
    pub model: PursuitModel<MyAutodiffBackend>,
    
    // We store it as a Boxed Trait Object using OUR safe trait!
    // We no longer care what internal struct Burn is using.
    pub optimizer: Box<dyn ModelOptimizer>,
    
    pub device: <MyAutodiffBackend as Backend>::Device,
}

impl FromWorld for BrainResource {
    fn from_world(_world: &mut World) -> Self {
        let device = NdArrayDevice::Cpu; 
        let model = PursuitModel::new(&device);
        
        // Wrap the unnameable optimizer cleanly in a Box
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
        // Initialized as NonSend to accommodate the Autodiff graph
        app.init_non_send_resource::<BrainResource>();
        app.add_systems(Update, batched_pursuit_think_and_learn);
    }
}

// ── Systems ─────────────────────────────────────────────────────────────────

fn batched_pursuit_think_and_learn(
    mut brain: NonSendMut<BrainResource>,
    mut heterotrophs: Query<(Entity, &mut Organism, &Transform), (With<OrganismRoot>, With<Heterotroph>)>,
    phototrophs: Query<&Transform, (With<OrganismRoot>, With<Photoautotroph>)>,
) {
    let mut inputs_raw = Vec::new();
    let mut ideal_outputs_raw = Vec::new();
    let mut active_entities = Vec::new();

    for (entity, _organism, h_transform) in heterotrophs.iter() {
        let h_pos = h_transform.translation;

        let mut closest_target: Option<Vec3> = None;
        let mut min_dist_sq = f32::MAX;

        for p_transform in phototrophs.iter() {
            let dist_sq = h_pos.distance_squared(p_transform.translation);
            if dist_sq < min_dist_sq {
                min_dist_sq = dist_sq;
                closest_target = Some(p_transform.translation);
            }
        }

        if let Some(t_pos) = closest_target {
            let relative_pos = t_pos - h_pos;
            inputs_raw.push(relative_pos.x);
            inputs_raw.push(relative_pos.y);
            inputs_raw.push(relative_pos.z);

            let ideal_dir = relative_pos.normalize_or_zero();
            
            ideal_outputs_raw.push(ideal_dir.x);
            ideal_outputs_raw.push(ideal_dir.y);
            ideal_outputs_raw.push(ideal_dir.z);
            ideal_outputs_raw.push(1.0); 
            
            let forward = Vec3::Z;
            let dot = forward.dot(ideal_dir);
            let cross = forward.cross(ideal_dir);
            let ideal_yaw = f32::atan2(cross.y, dot);
            ideal_outputs_raw.push(ideal_yaw.clamp(-1.0, 1.0));

            active_entities.push(entity);
        }
    }

    let batch_size = active_entities.len();
    if batch_size == 0 { return; }

    let input_tensor = Tensor::<MyAutodiffBackend, 2>::from_data(
        TensorData::new(inputs_raw, [batch_size, 3]),
        &brain.device,
    );
    
    let target_tensor = Tensor::<MyAutodiffBackend, 2>::from_data(
        TensorData::new(ideal_outputs_raw, [batch_size, 5]),
        &brain.device,
    );

    let output_tensor = brain.model.forward(input_tensor);

    // 4. TRAINING (Reward / Punishment)
    let loss = burn::nn::loss::MseLoss::new().forward(output_tensor.clone(), target_tensor, burn::nn::loss::Reduction::Mean);
    
    let gradients = loss.backward();
    
    // -- THE FIX: Decouple the borrows --
    // We explicitly extract the immutable reads so they finish and release 
    // the NonSendMut smart pointer lock.
    let cloned_model = brain.model.clone();
    let grad_params = GradientsParams::from_grads(gradients, &brain.model);
    
    // Now we safely acquire the mutable lock exclusively for the optimizer step
    let new_model = brain.optimizer.step_model(1e-3, cloned_model, grad_params);
    brain.model = new_model;

    // 5. SCATTER PHASE (Applying outputs back to Bevy)
    let output_data = output_tensor.into_data().into_vec::<f32>().expect("Failed to convert tensor to Vec<f32>");

    for (i, entity) in active_entities.into_iter().enumerate() {
        if let Ok((_, mut organism, mut transform)) = heterotrophs.get_mut(entity) {
            let offset = i * 5;
            
            let dir_x = output_data[offset + 0];
            let dir_y = output_data[offset + 1];
            let dir_z = output_data[offset + 2];
            let speed_out = output_data[offset + 3];
            let yaw_out = output_data[offset + 4];

            let new_dir = Vec3::new(dir_x, dir_y, dir_z);
            if new_dir.length_squared() > 0.01 {
                organism.movement_direction = new_dir.normalize();
            }

            organism.movement_speed = ((speed_out + 1.0) / 2.0) * 20.0;

            let current_rotation = transform.rotation;
            let target_rotation = Quat::from_rotation_y(yaw_out * std::f32::consts::PI);
            
            organism.target_rotation = current_rotation.slerp(target_rotation, 0.1);
        }
    }
}
