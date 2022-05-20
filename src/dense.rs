use crate::activation::ActivationFunction;
use crate::layer::PocketLayer;
use crate::mat::{self, WeightsInitializer};
use crate::params::Params;

use std::collections::HashMap;
use std::io::{Error, ErrorKind};

pub const SHRT_MAX: i32 = 32767;

#[derive(Clone)]
pub struct Dense {
    layer_size: usize,
    input_size: usize, //flattened 2D matrix to 1D vector/array
    activation: ActivationFunction,
    params: Params,
}

impl<T> PocketLayer<T> for Dense {
    fn new(
        layer_size: usize,
        input_size: usize,
        activation: ActivationFunction,
        params: Params,
    ) -> Self {
        Dense {
            layer_size,
            input_size,
            activation,
            params,
        }
    }

    fn initialize(
        &mut self,
        weights_initializer: WeightsInitializer,
        bias_initializer: WeightsInitializer,
    ) -> Result<bool, Error> {
        if self.input_size <= 1 {
            let msg = format!("Invalid input size! Size must be greater than 1.");
            return Err(Error::new(ErrorKind::InvalidData, msg));
        }

        self.params.layer_size = self.layer_size;
        self.params.input_size = self.input_size;
        self.params.init_weights(weights_initializer);
        self.params.init_bias(bias_initializer);
        self.params.last_layer = false;
        self.params.activation = self.activation;

        Ok(true)
    }

    fn forward(&self, input: Vec<u32>) -> Vec<u32> {
        let z = mat::forward_pass(
            &input,
            &self.params.weights,
            &self.params.biases,
            self.layer_size,
            self.input_size,
        );
        self.params.activation.apply_function(&z)
    }

    fn dfa(&self, dfa_weights: Vec<u32>) -> (&Vec<u32>, &Vec<u32>) {
        todo!()
    }

    fn reset_params(&self) {
        todo!()
    }

    fn has_weights(&self) -> bool {
        return false;
    }

    fn get_params(&self) -> HashMap<&str, &str> {
        todo!()
    }

    fn get_param(&self, key: &str) -> (&str, &str) {
        todo!()
    }

    fn set_param(&self, key: &str, value: T) {
        todo!()
    }

    /*
    fn get_output(&self) -> &Box<dyn Array> {
        &self.params.output
    }

    fn get_weights(&self) -> &Box<dyn Array> {
        &self.params.weights
    }

    fn set_random_dfa_weights(&mut self, i: usize, c: usize) {
        // Using He initialization. Maybe other randomization can work better
        let range = mat::floor_sqrt((12 * SHRT_MAX) / (self.input_size + self.output_size) as i32);
        self.params.weights.setRandom(false, -range, range);
    }
    */
}
