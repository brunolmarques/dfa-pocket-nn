#[derive(Clone, Copy)]
pub enum ActivationFunction {
    PocketSigmoid,
    PocketSoftmax,
    PocketTanh,
    PocketRelu8bit,
    PocketLeakyRelu,
}

impl ActivationFunction {
    pub fn apply_function(&self, input: &Vec<u32>) -> Vec<u32> {
        match self {
            ActivationFunction::PocketSigmoid => self.pocket_sigmoid(input),
            ActivationFunction::PocketSoftmax => self.pocket_softmax(input),
            ActivationFunction::PocketTanh => self.pocket_tanh(input),
            ActivationFunction::PocketRelu8bit => self.pocket_relu8bit(input),
            ActivationFunction::PocketLeakyRelu => self.pocket_leaky_relu(input),
        }
    }

    fn pocket_sigmoid(&self, input: &Vec<u32>) -> Vec<u32> {
        todo!()
    }

    fn pocket_softmax(&self, input: &Vec<u32>) -> Vec<u32> {
        todo!()
    }

    fn pocket_tanh(&self, input: &Vec<u32>) -> Vec<u32> {
        todo!()
    }

    fn pocket_relu8bit(&self, input: &Vec<u32>) -> Vec<u32> {
        todo!()
    }

    fn pocket_leaky_relu(&self, input: &Vec<u32>) -> Vec<u32> {
        todo!()
    }
}
