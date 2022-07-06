extern crate dfa_pocket_nn as pcknn;

use pcknn::mat;

#[test]
fn test_dot_fn() {
    let mock_input_1 = Box::new(vec![2, 4, 3, 5]);
    let mock_input_2 = Box::new(vec![4, 2, 1, 1]);
    let expectected = Box::new(vec![12, 8, 17, 11]);

    assert!(mock_input_1.len() == expectected.len(), "input and output lengths must be the same");
    assert!(mock_input_2.len() == expectected.len(), "input and output lengths must be the same");
  
    let result = mat::dot(&mock_input_1, &mock_input_2, 2, 2);

    assert_eq!(result, expectected)
}

#[test]
fn test_add_bias_fn() {
    let weights = Box::new(vec![2, 4, 3, 5]);
    let bias = &Box::new(vec![7, 2]);
    let expectected = Box::new(vec![9, 11, 5, 7]);
  
    let result = mat::add_bias(weights, bias, 2);

    assert_eq!(result, expectected)
}

#[test]
fn test_transpose_fn() {
    let input_matrix = Box::new(vec![2, 4, 3, 5, 3, 9, 1, 2, 1]);
    let expectected = Box::new(vec![2, 5, 1, 4, 3, 2, 3, 9, 1]);
  
    let result = mat::transpose(&input_matrix, 3, 3);

    assert_eq!(result, expectected)
}