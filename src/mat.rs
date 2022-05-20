pub fn forward_pass(
    input: &Vec<u32>,
    weights: &Vec<u32>,
    bias: &Vec<u32>,
    layer_size: usize,
    input_size: usize,
) -> Vec<u32> {
    let mut forward_weights = dot(input, weights, layer_size, input_size);
    forward_weights = add_bias(forward_weights, bias, layer_size);
    todo!();
    
}

pub fn dot(
    a_matrix: &Vec<u32>,
    b_matrix: &Vec<u32>,
    layer_size: usize,
    input_size: usize,
) -> Vec<u32> {
    let mut result_matrix: Vec<u32> = vec![0; layer_size * input_size];

    for (c_i, a_i) in result_matrix
        .chunks_exact_mut(layer_size)
        .zip(a_matrix.chunks_exact(layer_size))
    {
        for (a_ik, b_k) in a_i.iter().zip(b_matrix.chunks_exact(layer_size)) {
            for (c_ij, b_kj) in c_i.iter_mut().zip(b_k.iter()) {
                *c_ij += (*a_ik) * (*b_kj);
            }
        }
    }
    result_matrix
}

pub fn add_bias(mut weights: Vec<u32>, bias: &Vec<u32>, layer_size: usize) -> Vec<u32> {
    weights
        .chunks_exact_mut(layer_size)
        .zip(bias.iter())
        .flat_map(|(row, b)| row.iter_mut().map(move |w_value| *w_value + b))
        .collect()
}

#[derive(Clone)]
pub enum WeightsInitializer {
    ConstantValue,
    RandomHe,
    Zeros,
}
