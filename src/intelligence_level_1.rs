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

use burn_cuda::{Cuda, CudaDevice};
use burn::backend::Autodiff;

// Set the backend to LibTorch (f32 precision)
pub type MyBackend = Cuda;
pub type MyAutodiffBackend = Autodiff<MyBackend>;

use crate::colony::{Organism, Photoautotroph, Heterotroph, MAXIMUM_ORGANISMS};


// ── Custom Optimizer Trait ──────────────────────────────────────────────────
// This trait abstracts Burn's generic optimizer specifically for our MLP, 
// allowing it to be safely stored inside a Bevy Resource as a Boxed trait object.

pub trait ModelOptimizer {
    fn step_model(
        &mut self,
        lr: f64,
        model: BasicMlpModel<MyAutodiffBackend>,
        grads: GradientsParams,
    ) -> BasicMlpModel<MyAutodiffBackend>;
}

// A blanket implementation that automatically adapts any compatible Burn 
// optimizer (like Adam) to our custom trait.
impl<O> ModelOptimizer for O
where
    O: Optimizer<BasicMlpModel<MyAutodiffBackend>, MyAutodiffBackend>,
{
    fn step_model(
        &mut self,
        lr: f64,
        model: BasicMlpModel<MyAutodiffBackend>,
        grads: GradientsParams,
    ) -> BasicMlpModel<MyAutodiffBackend> {
        // Calls the native Burn optimizer step function
        self.step(lr, model, grads)
    }
}


// ── Intelligence Level 1 (Basic MLP for Phototrophs) ────────────────────────
#[derive(Module, Debug)]
pub struct BasicMlpModel<B: Backend> {
    layer: Linear<B>, // Bare minimum: Single layer mapped directly to outputs
}

impl<B: Backend> BasicMlpModel<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            // 8 Inputs -> 7 Outputs
            layer: LinearConfig::new(8, 7).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.layer.forward(input);
        // Tanh bounds directions and speeds safely between [-1.0, 1.0]
        burn::tensor::activation::tanh(x) 
    }
}

// ── Resource Definition ─────────────────────────────────────────────────────
pub struct BrainResourceLevel1 {
    pub model: BasicMlpModel<MyAutodiffBackend>,
    pub optimizer: Box<dyn ModelOptimizer>,
    pub device: CudaDevice,
}

impl FromWorld for BrainResourceLevel1 {
    fn from_world(_world: &mut World) -> Self {
        let device = CudaDevice::default();
        let batch_size = MAXIMUM_ORGANISMS as usize;
        
        println!("🌱 Priming INTELLIGENCE LEVEL 1 MLP kernels for fixed batch size: {}...", batch_size);
        
        let model = BasicMlpModel::<MyAutodiffBackend>::new(&device);
        let mut optimizer: Box<dyn ModelOptimizer> = Box::new(AdamConfig::new().init());
        
        // Shape: [Batch Size, 8 Inputs] & [Batch Size, 7 Outputs]
        let input_tensor = Tensor::<MyAutodiffBackend, 2>::zeros([batch_size, 8], &device);
        let target_tensor = Tensor::<MyAutodiffBackend, 2>::zeros([batch_size, 7], &device);
        
        let output_tensor = model.forward(input_tensor);
        let loss = burn::nn::loss::MseLoss::new().forward(output_tensor, target_tensor, burn::nn::loss::Reduction::Mean);
        let gradients = loss.backward();
        let grad_params = GradientsParams::from_grads(gradients, &model);
        let _ = optimizer.step_model(1e-3, model.clone(), grad_params);
        
        Self { model, optimizer, device }
    }
}

// ── Main System ─────────────────────────────────────────────────────────────
pub fn apply_intelligence_level_1(
    time: Res<Time<Virtual>>,
    mut brain: NonSendMut<BrainResourceLevel1>,
    mut organisms: Query<(Entity, &mut Organism), (With<Photoautotroph>, Without<Heterotroph>)>,
) {
    if time.is_paused() { return; }

    let max_orgs = MAXIMUM_ORGANISMS as usize;
    let mut batch_inputs = Vec::with_capacity(max_orgs * 8);
    let mut ideal_outputs = Vec::with_capacity(max_orgs * 7);
    let mut active_entities = Vec::with_capacity(max_orgs);

    for (entity, organism) in organisms.iter() {
        if active_entities.len() >= max_orgs { break; }

        let max_energy = crate::energy::get_max_energy(&organism).max(1.0);
        let energy_ratio = (organism.energy / max_energy).clamp(0.0, 1.0);

        // 8 explicitly requested Inputs for Intelligence Level 1
        batch_inputs.extend_from_slice(&[
            energy_ratio,
            organism.last_movement_speed,
            organism.last_movement_direction.x,
            organism.last_movement_direction.y,
            organism.last_movement_direction.z,
            organism.last_rotation.x,
            organism.last_rotation.y,
            organism.last_rotation.z,
        ]);

        // Proximal Training Data: Plants scatter if starving to find new grounds
        let ideal_speed = if energy_ratio < 0.3 { 1.0 } else { 0.0 };
        let ideal_dir = if organism.last_movement_direction.length_squared() > 0.01 {
            organism.last_movement_direction.normalize()
        } else {
            Vec3::Z
        };
        let ideal_rot = ideal_dir; 

        // 7 Output dimensions for training targets
        ideal_outputs.extend_from_slice(&[
            ideal_speed,
            ideal_rot.x,
            ideal_rot.y,
            ideal_rot.z,
            ideal_dir.x,
            ideal_dir.y,
            ideal_dir.z,
        ]);

        active_entities.push(entity);
    }

    if active_entities.is_empty() { return; }

    // Instant Zero-Padding 
    batch_inputs.resize(max_orgs * 8, 0.0);
    ideal_outputs.resize(max_orgs * 7, 0.0);

    let input_tensor = Tensor::<MyAutodiffBackend, 2>::from_data(
        TensorData::new(batch_inputs, [max_orgs, 8]),
        &brain.device,
    );
    let target_tensor = Tensor::<MyAutodiffBackend, 2>::from_data(
        TensorData::new(ideal_outputs, [max_orgs, 7]),
        &brain.device,
    );

    let output_tensor = brain.model.forward(input_tensor);
    let loss = burn::nn::loss::MseLoss::new().forward(output_tensor.clone(), target_tensor, burn::nn::loss::Reduction::Mean);
    
    let gradients = loss.backward();
    let cloned_model = brain.model.clone();
    let grad_params = GradientsParams::from_grads(gradients, &brain.model);
    
    brain.model = brain.optimizer.step_model(1e-3, cloned_model, grad_params);

    let output_data = output_tensor.into_data().into_vec::<f32>().expect("Failed to convert tensor");

    // Map outputs directly back to Vec3 directions (treating output as Cartesian vectors)
    for (i, entity) in active_entities.into_iter().enumerate() {
        if let Ok((_, mut organism)) = organisms.get_mut(entity) {
            let offset = i * 7;
            
            // Output bounds handled by Tanh activation [-1.0, 1.0]
            let speed_out = output_data[offset + 0].clamp(0.0, 1.0);
            let rot_x = output_data[offset + 1];
            let rot_y = output_data[offset + 2];
            let rot_z = output_data[offset + 3];
            let dir_x = output_data[offset + 4];
            let dir_y = output_data[offset + 5];
            let dir_z = output_data[offset + 6];

            organism.movement_speed = speed_out * 20.0; // Assuming 20.0 is your base max speed
            
            // Map rotation directly
            let new_rot = Vec3::new(rot_x, rot_y, rot_z);
            if new_rot.length_squared() > 0.01 {
                organism.target_rotation = new_rot.normalize(); 
            }

            // Map direction
            let new_dir = Vec3::new(dir_x, dir_y, dir_z);
            if new_dir.length_squared() > 0.01 {
                organism.movement_direction = new_dir.normalize();
            }
        }
    }
}
