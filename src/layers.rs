use crate::linear_algebra::Matrix;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Normal, Distribution};

/// Represents a linear neural network layer
pub struct Linear {
    weights: Matrix,
    biases: Matrix,
}

impl Linear {
    /// Returns a new linear layer with the given number of inputs and units
    pub fn new(inputs: usize, units: usize) -> Linear {
        // initialize weights using Kaiming He initialization scheme
        let weights = Linear::kaiming_he_init(inputs, units);
        // initialize biases to 0
        let biases = Matrix::new(1, units);
        // create and return new linear layer
        Linear {
            weights,
            biases,
        }
    }

    /// Returns a weights matrix of the given size whose elements have 
    /// been randomly initialized based on the Kaiming He method
    fn kaiming_he_init(inputs: usize, units: usize) -> Matrix {
        // set up random number generation
        let mut rng = StdRng::seed_from_u64(43);
        let sd = (2.0 / inputs as f64).sqrt();
        let normal = Normal::new(0., sd).unwrap();
        // set up matrix data
        let mut data: Vec<f64> = vec![0.; inputs * units];
        // set each element to a random number drawn from a normal
        // distrubution with a mean of 0 and a sd of 2 / rows
        for i in 0..data.len() {
            data[i] = normal.sample(&mut rng);
        }
        // create and return matrix
        Matrix::from_vec(inputs, units, data)
    }
}

impl Layer for Linear {
    /// Returns the matrix product of the given inputs and the layer's
    /// weights, plus the layer's biases
    fn forward(&self, inputs: &Matrix) -> Matrix {
        // output = inputs . weights + biases
        inputs.matmul(&self.weights).add(&self.biases)
    }

    /// Caclulates and sets the gradients of the weights and biases
    /// and calculates and returns the gradients of the layer's inputs
    fn backward(&mut self, inputs: &Matrix, outputs: &Matrix) -> Matrix {
        // gradients of weights = transpose of inputs . gradients of outputs
        // i.e. gradient of each weight is input associated with that weight *
        // gradient of output of unit associated with that weight
        self.weights.set_grad(&inputs.t().matmul(outputs.get_grad()));
        // gradients of biases are gradients of outputs
        self.biases.set_grad(outputs.get_grad());
        // gradients of inputs = gradients of outputs . transpose of weights
        // i.e. gradient of each input is sum of products of each weight
        // associated with that input and the gradient of the output of
        // unit associated with that weight
        outputs.get_grad().matmul(&self.weights.t())
    }
}

impl TrainableLayer for Linear {
    /// Updates the weights and biases based on their gradients and
    /// the given learning rate
    fn update(&mut self, lr: f64) {
        self.weights.gd_step(lr);
        self.biases.gd_step(lr);
    }
}

pub struct ReLU();

pub struct Sigmoid();

pub struct MSELoss();

/// Forward pass and backward pass methods associated with
/// a layer of an artificial neural network
pub trait Layer {
    /// Returns the result of passing the input through the layer
    fn forward(&self, inputs: &Matrix) -> Matrix;
    /// Returns the gradients of the inputs as well as
    /// setting any internal gradients
    fn backward(&mut self, inputs: &Matrix, outputs: &Matrix) -> Matrix;
}

/// Update method to update the parameters of a trainable neural network layer
pub trait TrainableLayer {
    /// Updates the layer's parameters
    fn update(&mut self, lr: f64);
}

// Unit tests
#[cfg(test)]
mod tests {
    use super::*;

    // test the forward pass of the linear layer
    #[test]
    fn linear_forward() {
        let lin = Linear::new(4, 5);
        let inputs = Matrix::from_vec(1, 4, vec![1., 2., 3., 4.,]);
        let outputs = lin.forward(&inputs);
        let expected_result = round_vector(&vec![0.97200704663574553,
            0.611102402371291228,
            2.17676979487117434,
            -1.38731209511734939,
            0.09459345315137568]);
        assert_eq!(outputs.get_rows(), 1);
        assert_eq!(outputs.get_cols(), 5);
        assert_eq!(round_vector(outputs.get_data()), expected_result);
    }

    // test the backward pass of the linear layer
    #[test]
    fn linear_backward() {
        let mut lin = Linear::new(4, 5);
        let mut inputs = Matrix::from_vec(1, 4, vec![1., 2., 3., 4.,]);
        inputs.init_grad();
        let mut outputs = lin.forward(&inputs);
        outputs.init_grad();
        outputs.set_grad(&Matrix::from_vec(1, 5, vec![1.; 5]));
        let input_grads = lin.backward(&inputs, &outputs);
        let expected_input_grads = round_vector(&vec![-0.47602755502906417, 
            -4.0489898635333251, 
            0.42156254984864825, 
            2.444120058615501707]);
        assert_eq!(round_vector(input_grads.get_data()), expected_input_grads);
        assert_eq!(input_grads.get_rows(), 1);
        assert_eq!(input_grads.get_cols(), 4);
        let expected_weight_grads = vec![1., 1., 1., 1., 1.,
                                         2., 2., 2., 2., 2.,
                                         3., 3., 3., 3., 3.,
                                         4., 4., 4., 4., 4.];
        assert_eq!(lin.weights.get_grad().get_data(), &expected_weight_grads);
        assert_eq!(lin.weights.get_grad().get_rows(), 4);
        assert_eq!(lin.weights.get_grad().get_cols(), 5);
        assert_eq!(lin.biases.get_grad().get_data(), outputs.get_grad().get_data());
        assert_eq!(lin.biases.get_grad().get_rows(), 1);
        assert_eq!(lin.biases.get_grad().get_cols(), 5);
    }

    // helper function which returns a new vector in which each element
    // is the corresponding element of the given vector rounded to
    // eight decimal places
    fn round_vector(data: &Vec<f64>) -> Vec<f64> {
        data.iter().map(|num| {
            (num * 100_000_000.0).round() / 100_000_000.0
        }).collect()
    }
}