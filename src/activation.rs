
pub trait Activation {
    fn forward(&self, x: Box<Vec<u32>>) -> Box<Vec<u32>>;

    fn gradient(&self, x: Box<Vec<u32>>) -> Box<Vec<u32>>;
}

#[derive(Clone, Copy)]
pub enum ActivationFunction {
    PocketSigmoid,
    PocketSoftmax,
    PocketTanh,
    PocketRelu,
    PocketLeakyRelu,
    LinearDirect,
}

impl Activation for ActivationFunction {
    fn forward(&self, input: Box<Vec<u32>>) -> Box<Vec<u32>> {
        use ActivationFunction::*;

        match self {
            PocketSigmoid => self.pocket_sigmoid_forward(input),
            PocketSoftmax => self.pocket_softmax_forward(input),
            PocketTanh => self.pocket_tanh_forward(input),
            PocketRelu => self.pocket_relu_forward(input),
            PocketLeakyRelu => self.pocket_leaky_relu_forward(input),
            LinearDirect => input,
        }
    }

    fn gradient(&self, input: Box<Vec<u32>>) -> Box<Vec<u32>> {
        use ActivationFunction::*;

        match self {
            PocketSigmoid => self.pocket_sigmoid_gradient(input),
            PocketSoftmax => self.pocket_softmax_gradient(input),
            PocketTanh => self.pocket_tanh_gradient(input),
            PocketRelu => self.pocket_relu_gradient(input),
            PocketLeakyRelu => self.pocket_leaky_relu_gradient(input),
            LinearDirect => input,
        }
    }
}

impl ActivationFunction {
    fn pocket_sigmoid_forward(&self, input: Box<Vec<u32>>) -> Box<Vec<u32>> {
        // return 1.0 / (1.0 + np.exp(-x))
        todo!()
    }

    fn pocket_sigmoid_gradient(&self, input: Box<Vec<u32>>) -> Box<Vec<u32>> {
        todo!()
    }

    fn pocket_softmax_forward(&self, input: Box<Vec<u32>>) -> Box<Vec<u32>> {
        todo!()
    }

    fn pocket_softmax_gradient(&self, input: Box<Vec<u32>>) -> Box<Vec<u32>> {
        todo!()
    }

    fn pocket_tanh_forward(&self, input: Box<Vec<u32>>) -> Box<Vec<u32>> {
        todo!()
    }

    fn pocket_tanh_gradient(&self, input: Box<Vec<u32>>) -> Box<Vec<u32>> {
        todo!()
    }

    fn pocket_relu_forward(&self, input: Box<Vec<u32>>) -> Box<Vec<u32>> {
        todo!()
    }

    fn pocket_relu_gradient(&self, input: Box<Vec<u32>>) -> Box<Vec<u32>> {
        todo!()
    }

    fn pocket_leaky_relu_forward(&self, input: Box<Vec<u32>>) -> Box<Vec<u32>> {
        todo!()
    }

    fn pocket_leaky_relu_gradient(&self, input: Box<Vec<u32>>) -> Box<Vec<u32>> {
        todo!()
    }
}
