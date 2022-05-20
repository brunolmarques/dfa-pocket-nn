use crate::activation::ActivationFunction;
use crate::mat::WeightsInitializer;
use crate::params::Params;

use std::collections::HashMap;
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

    fn forward(&self, input: Vec<u32>) -> Vec<u32>;

    fn dfa(&self, dfa_weights: Vec<u32>) -> (&Vec<u32>, &Vec<u32>);

    fn reset_params(&self);

    fn has_weights(&self) -> bool {
        return false;
    }

    fn get_params(&self) -> HashMap<&str, &str>;

    fn get_param(&self, key: &str) -> (&str, &str);

    fn set_param(&self, key: &str, value: T);
}
