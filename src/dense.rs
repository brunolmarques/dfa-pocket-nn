use crate::activation::*;
use crate::layer::PocketLayer;
use crate::mat::{self, WeightsInitializer};
use crate::params::Params;
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

    fn forward(&self, node_input: Box<Vec<u32>>) -> Box<Vec<u32>> {

        let z = mat::forward_pass(
            &node_input,
            &self.params.weights,
            &self.params.biases,
            self.layer_size,
            self.input_size,
        );
        
        self.params.activation.forward(z)
    }

    fn dfa(&self, dfa_weights: Box<Vec<u32>>, node_input: Box<Vec<u32>>, node_output: Box<Vec<u32>>) -> (Box<Vec<u32>>, u32) {
        let mut temp_weights: Box<Vec<u32>> = Box::new(vec![]);

        if !self.params.last_layer {
            temp_weights = mat::dot(
                &dfa_weights,
                &self.params.biases,
                self.layer_size,
                self.input_size,
            );
        } else {
            temp_weights = dfa_weights;
        };

        let grad = self.activation.gradient(node_output);

        let egrad = Box::new(temp_weights.into_iter()
            .zip(*grad)
            .map(|(e, g)| e * g)
            .collect::<Vec<_>>());

        let dw = mat::dot(
            &mat::transpose(&node_input, self.layer_size, self.input_size),
            &egrad,
            self.layer_size,
            self.input_size,
        );
        let db = egrad.iter().sum::<u32>();
        (dw, db)
    }

    fn has_weights(&self) -> bool {
        return false;
    }

    /*
    fn set_random_dfa_weights(&mut self, i: usize, c: usize) {
        // Using He initialization. Maybe other randomization can work better
        let range = mat::floor_sqrt((12 * SHRT_MAX) / (self.input_size + self.output_size) as i32);
        self.params.weights.setRandom(false, -range, range);
    }
    */
}
