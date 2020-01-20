/// A structure representing a two-dimensional coordinate
pub struct Coord(pub usize, pub usize);

/// A structure representing a matrix
pub struct Matrix {
    /// The number of rows in the matrix
    rows: usize,
    /// The number of columns in the matrix
    cols: usize,
    /// The data contained in the matrix
    data: Vec<f64>,
    /// An optional matrix of gradients
    pub grad: Option<Box<Matrix>>,
}

impl Matrix {
    /// Returns a Matrix with the number of rows and columns given,
    /// with each element initially set to 0.0
    pub fn new(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows,
            cols,
            // set data to zeroes
            data: vec![0.0; rows * cols],
            // set grad to None
            grad: None,
        }
    }

    /// Creates and returns a Matrix using an existing vector
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Matrix {
        if data.len() != rows * cols {
            panic!("Vector does not match the given dimensions");
        }
        Matrix {
            rows,
            cols,
            data,
            grad: None,
        }
    }

    /// Returns a clone of the current matrix, excluding its gradients
    pub fn clone(&self) -> Matrix {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.clone(),
            grad: None,
        }
    }

    /// Returns the number of rows in the matrix
    pub fn get_rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns in the matrix
    pub fn get_cols(&self) -> usize {
        self.cols
    }

    /// Returns the element whose position is specified by the given Coord
    pub fn get_at(&self, loc: &Coord) -> f64 {
        // ensure the coordinates are valid
        self.validate_coord(&loc);
        // return element at given location
        self.data[self.coord_to_index(&loc)]
    }

    /// Sets the element whose position is specified by the given Coord
    /// to the value given
    pub fn set_at(&mut self, loc: &Coord, val: f64) {
        // ensure the coordinates are valid
        self.validate_coord(&loc);
        // calculate the correct index within the data vector
        let index = self.coord_to_index(&loc);
        // set the element to the given value
        self.data[index] = val;
    }

    /// Increments the element whose position is specified by the 
    /// given Coord by the value given
    pub fn increment_at(&mut self, loc: &Coord, val: f64) {
        // ensure the coordinates are valid
        self.validate_coord(&loc);
        // calculate the correct index within the data vector
        let index = self.coord_to_index(&loc);
        // increment the element by the given value
        self.data[index] += val;
    }

    /// Returns a reference to the data vector
    pub fn get_data(&self) -> &Vec<f64> {
        &self.data
    }

    /// Initializes the optional matrix of gradients
    pub fn init_grad(&mut self) {
        self.grad = Some(Box::new(Matrix::new(self.rows, self.cols)));
    }

    /// Returns a reference to the matrix of gradients, if present
    pub fn get_grad(&self) -> &Matrix {
        match &self.grad {
            Some(grad_ptr) => &*grad_ptr,
            None => panic!("No gradients set"),
        }
    }

    /// Sets the matrix of gradients to the given matrix
    pub fn set_grad(&mut self, grads: &Matrix) {
        self.grad = Some(Box::new(grads.clone()));
    }

    /// Displays the matrix for testing and debugging purposes
    pub fn print(&self) {
        for i in self.data.iter() {
            print!("{} ", i);
        }
        print!("\n");
    }

    /// Return the matrix which is the result of adding the given matrix
    /// to the current matrix
    pub fn add(&self, other: &Matrix) -> Matrix {
        // ensure the given matrix has the correct dimensions
        self.validate_equal_dimensions(other);
        // initialize new matrix
        let mut matrix = Matrix::new(self.rows, self.cols);
        // set each elelment of the new matrix by adding each 
        // corresponding pair of elements in the current and given matrices
        for i in 0..self.data.len() {
            matrix.data[i] = self.data[i] + other.data[i];
        }
        // return the new matrix
        matrix
    }

    /// Return the matrix which is the result of multiplying the given
    /// matrix with the current matrix
    pub fn matmul(&self, other: &Matrix) -> Matrix {
        // ensure given matrix is compatible
        self.validate_matmul_compatible(other);
        // create result matrix
        let mut matrix = Matrix::new(self.rows, other.cols);
        // current result of element multiplication
        let mut x: f64;
        // perform matrix multiplication
        for i in 0..self.rows {
            for k in 0..self.cols {
                for j in 0..other.cols {
                    x = self.get_at(&Coord(i, k)) * 
                        other.get_at(&Coord(k, j));
                    matrix.increment_at(&Coord(i, j), x);
                }
            }
        }
        matrix
    }

    /// Returns the transpose of the current matrix
    pub fn t(&self) -> Matrix {
        // create result matrix
        let mut matrix = Matrix::new(self.cols, self.rows);
        // calculate transposition
        for i in 0..self.rows {
            for j in 0..self.cols {
                matrix.set_at(&Coord(j, i), 
                    self.get_at(&Coord(i, j)));
            }
        }
        matrix
    }

    /// Multiplies each gradient by the given learning rate and
    /// adds this value to the corresponding data element
    pub fn gd_step(&mut self, lr: f64) {
        for i in 0..self.data.len() {
            self.data[i] += self.get_grad().get_data()[i] * lr;
        }
    }

    // panic if the given coordinate is invalid for the current matrix
    fn validate_coord(&self, coord: &Coord) {
        if coord.0 >= self.rows || coord.1 >= self.cols {
            panic!("Coord row or col too large for this matrix");
        }
    }

    // panic if the given matrix's dimensions do not match the current matrix
    fn validate_equal_dimensions(&self, other: &Matrix) {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Matrices do not have equal dimensions");
        }
    }

    // panic if the given matrix's dimensions do not match the current matrix
    fn validate_matmul_compatible(&self, other: &Matrix) {
        if self.cols != other.rows {
            panic!("Matrices do not have compatible dimensions");
        }
    }

    // return the index in the data vector corresponding to the given Coord
    fn coord_to_index(&self, coord: &Coord) -> usize {
        coord.0 * self.cols + coord.1
    }
}

// Unit tests
#[cfg(test)]
mod tests {
    use super::*;

    // test whether matrix rows is set correctly
    #[test]
    fn initial_rows() {
        let matrix = Matrix::new(2, 4);
        assert_eq!(2, matrix.rows);
    }

    // test whether matrix cols is set correctly
    #[test]
    fn initial_cols() {
        let matrix = Matrix::new(2, 4);
        assert_eq!(4, matrix.cols);
    }

    // test whether data is initialized to the correct length
    #[test]
    fn initial_data_len() {
        let matrix = Matrix::new(2, 4);
        assert_eq!(matrix.data.len(), 2 * 4);
    }

    // test from_vec
    #[test]
    fn create_from_vector() {
        let vector = vec![0., 1., 2., 3., 4., 5.];
        let matrix = Matrix::from_vec(2, 3, vector.clone());
        assert_eq!(matrix.data, vector);
    }

    // test whether from_vec panics correctly if dimensions do not match
    #[test]
    #[should_panic(expected = "Vector does not match the given dimensions")]
    fn create_from_invalid_vector() {
        let vector = vec![0., 1., 2., 3., 4., 5.];
        let matrix = Matrix::from_vec(4, 3, vector.clone());
        assert_eq!(matrix.data, vector);
    }

    // test the conversion of a Coord into an index in the data vector
    #[test]
    fn coord_index_conversion() {
        let mut matrix = Matrix::new(2, 3);
        matrix.data = vec![1., 2., 3., 4., 5., 6.];
        let coord = Coord(1, 1);
        assert_eq!(matrix.coord_to_index(&coord), 4);
    }

    // test element access using get_at
    #[test]
    fn element_access() {
        let mut matrix = Matrix::new(2, 3);
        matrix.data = vec![1., 2., 3., 4., 5., 6.];
        let coord = Coord(1, 1);
        assert_eq!(matrix.get_at(&coord), 5.);
    }

    // test element modification using set_at
    #[test]
    fn element_modification() {
        let mut matrix = Matrix::new(2, 3);
        let coord = Coord(1, 1);
        matrix.set_at(&coord, 5.);
        assert_eq!(matrix.get_at(&coord), 5.);
    }

    // test whether element access panics correctly if Coord is invalid
    #[test]
    #[should_panic(expected = "Coord row or col too large for this matrix")]
    fn out_of_bounds_access() {
        let matrix = Matrix::new(2, 3);
        matrix.get_at(&Coord(2, 2));
    }

    // test elementwise addition
    #[test]
    fn elementwise_addition() {
        let mut a = Matrix::new(2, 3);
        a.data = vec![1., 2., 3., 4., 5., 6.];
        let mut b = Matrix::new(2, 3);
        b.data = vec![2., 4., 6., 8., 10., 12.];
        let c = a.add(&b);
        assert_eq!(c.data, vec![3., 6., 9., 12., 15., 18.]);
    }

    // test whether elementwise add panics correctly if matrices
    // do not have equal dimensions
    #[test]
    #[should_panic(expected = "Matrices do not have equal dimensions")]
    fn elementwise_add_wrong_size() {
        let a = Matrix::new(2, 3);
        let b = Matrix::new(2, 4);
        a.add(&b);
    }

    // test matrix multiply
    #[test]
    fn matrix_multiply() {
        let mut a = Matrix::new(2, 3);
        a.data = vec![1., 2., 3., 4., 5., 6.];
        let mut b = Matrix::new(3, 1);
        b.data = vec![7., 8., 9.];
        let c = a.matmul(&b);
        assert_eq!(c.data, vec![50., 122.]);
    }

    // test whether matrix multiply panics correctly if matrices
    // do not have compatible dimensions
    #[test]
    #[should_panic(expected = "Matrices do not have compatible dimensions")]
    fn matrix_multiply_wrong_size() {
        let a = Matrix::new(2, 3);
        let b = Matrix::new(2, 3);
        a.matmul(&b);
    }

    // test transpose
    #[test]
    fn matrix_transposition() {
        let mut a = Matrix::new(2, 3);
        a.data = vec![1., 2., 3., 4., 5., 6.];
        let b = a.t();
        assert_eq!(b.data, vec![1., 4., 2., 5., 3., 6.]);
    }

    // test setting of gradients
    #[test]
    fn initialize_gradients() {
        let mut matrix = Matrix::new(2, 3);
        matrix.init_grad();
        match matrix.grad {
            Some(grad_matrix) => assert_eq!(grad_matrix.data, vec![0.; 6]),
            None => panic!("Gradients have not been set"),
        }
    }

    // test updating of elements by gradient descent
    #[test]
    fn gradient_descent() {
        let mut matrix = Matrix::new(2, 3);
        matrix.init_grad();
        matrix.set_grad(&Matrix::from_vec(2, 3, vec![1.; 6]));
        matrix.gd_step(-0.01);
        assert_eq!(matrix.get_data(), &vec![-0.01; 6]);
    }
}