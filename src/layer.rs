use crate::activation::ActivationFunction;
use crate::mat::WeightsInitializer;
use crate::params::Params;
use std::io::Error;

pub trait PocketLayer<T> {
    fn new(
        layer_size: usize,
        input_size: usize,
        activation: ActivationFunction,
        params: Params,
    ) -> Self;

    fn initialize(
        &mut self,
        weights_initializer: WeightsInitializer,
        bias_initializer: WeightsInitializer,
    ) -> Result<bool, Error>;

    fn forward(&self, node_input: Box<Vec<u32>>) -> Box<Vec<u32>>;

    fn dfa(&self, dfa_weights: Box<Vec<u32>>, node_input: Box<Vec<u32>>, node_output: Box<Vec<u32>>) -> (Box<Vec<u32>>, u32);

    fn has_weights(&self) -> bool {
        return false;
    }

}
