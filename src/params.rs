use crate::activation::ActivationFunction;
use crate::mat::WeightsInitializer;

#[derive(Clone)]
pub struct Params {
    pub layer_size: usize,
    pub input_size: usize,
    pub weights: Vec<u32>,
    pub biases: Vec<u32>,
    pub last_layer: bool,
    pub activation: ActivationFunction,
}

impl Params {
    pub fn init_weights(&mut self, initializer: WeightsInitializer) {
        self.weights = vec![0; self.layer_size * self.input_size];
        todo!()
    }

    pub fn init_bias(&mut self, initializer: WeightsInitializer) {
        self.biases = vec![0; self.layer_size];
        todo!()
    }
}
