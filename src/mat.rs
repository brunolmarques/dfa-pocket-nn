
pub fn forward_pass(
    input: &Box<Vec<u32>>,
    weights: &Box<Vec<u32>>,
    bias: &Box<Vec<u32>>,
    layer_size: usize,
    input_size: usize,
) -> Box<Vec<u32>> {
    let forward_weights = dot(input, weights, layer_size, input_size);
    add_bias(forward_weights, bias, layer_size)
}

/*
Linearized matrix dot multiplcation.
layer_size breaks chunks of the vector into the exact matrix row size.
Each chunk represents a matrix row.
*/
pub fn dot(
    a_matrix: &Box<Vec<u32>>,
    b_matrix: &Box<Vec<u32>>,
    layer_size: usize,
    input_size: usize,
) -> Box<Vec<u32>> {
    let mut result_matrix: Vec<u32> = vec![0; layer_size * input_size];

    for (c_i, a_i) in result_matrix
        .chunks_exact_mut(layer_size)
        .zip(a_matrix.chunks_exact(layer_size))
    {
        for (a_ik, b_k) in a_i.iter().zip(b_matrix.chunks_exact(layer_size)) {
            for (c_ij, b_kj) in c_i.iter_mut().zip(b_k.iter()) { //loops c rown until a_ik is depleted
                *c_ij += (*a_ik) * (*b_kj);
            }
        }
    }
    Box::new(result_matrix)
}

pub fn add_bias(
    mut weights: Box<Vec<u32>>,
    bias: &Box<Vec<u32>>,
    layer_size: usize,
) -> Box<Vec<u32>> {
    Box::new(
        (*weights
            .chunks_exact_mut(layer_size)
            .zip(bias.iter())
            .flat_map(|(row, b)| row.iter_mut().map(move |w_value| *w_value + b))
            .collect::<Vec<u32>>())
        .to_vec(),
    )
}

pub fn transpose(vec: &Box<Vec<u32>>, layer_size: usize, input_size: usize) -> Box<Vec<u32>> {
    let mut transposed_vec: Vec<u32> = vec![];
    
    for i in 0..input_size {
        for l in 0..layer_size {
            if l == 0 {
                if let Some(elem) = vec.get(i) {
                    transposed_vec.push(*elem)
                }
            } else {
                let index = i + (l * 3);
                if let Some(elem) = vec.get(index) {
                    transposed_vec.push(*elem)
                }
            }
        }
    } 
    
    Box::new(transposed_vec)
}

#[derive(Clone)]
pub enum WeightsInitializer {
    ConstantValue,
    RandomHe,
    Zeros,
}
