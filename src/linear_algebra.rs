/// A structure representing a two-dimensional coordinate
pub struct Coord {
    /// The row element of the coordinate
    pub row: usize,
    /// The column element of the coordinate
    pub col: usize,
}

impl Coord {
    /// Returns a Coord with the given row and column values
    fn new(row: usize, col: usize) -> Coord {
        Coord {
            row,
            col,
        }
    }
}

/// A structure representing a matrix
pub struct Matrix {
    /// The number of rows in the matrix
    rows: usize,
    /// The number of columns in the matrix
    cols: usize,
    /// The data contained in the Matrix
    data: Vec<f64>,
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
        }
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

    /// Return the matrix which is the result of adding the given matrix
    /// to the current matrix
    pub fn add(&mut self, other: &Matrix) -> Matrix {
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

    // panic if the given coordinate is invalid for the current matrix
    fn validate_coord(&self, coord: &Coord) {
        if coord.row >= self.rows || coord.col >= self.cols {
            panic!("Coord row or col too large for this matrix");
        }
    }

    // panic if the given matrix's dimensions do not match the current matrix
    fn validate_equal_dimensions(&self, other: &Matrix) {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Matrices do not have equal dimensions");
        }
    }

    // return the index in the data vector corresponding to the given Coord
    fn coord_to_index(&self, coord: &Coord) -> usize {
        coord.row * self.cols + coord.col
    }
}

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

    // test the conversion of a Coord into an index in the data vector
    #[test]
    fn coord_index_conversion() {
        let mut matrix = Matrix::new(2, 3);
        matrix.data = vec![1., 2., 3., 4., 5., 6.];
        let coord = Coord::new(1, 1);
        assert_eq!(matrix.coord_to_index(&coord), 4);
    }

    // test element access using get_at
    #[test]
    fn element_access() {
        let mut matrix = Matrix::new(2, 3);
        matrix.data = vec![1., 2., 3., 4., 5., 6.];
        let coord = Coord::new(1, 1);
        assert_eq!(matrix.get_at(&coord), 5.);
    }

    // test element modification using set_at
    #[test]
    fn element_modification() {
        let mut matrix = Matrix::new(2, 3);
        let coord = Coord::new(1, 1);
        matrix.set_at(&coord, 5.);
        assert_eq!(matrix.get_at(&coord), 5.);
    }

    // test whether element access panics correctly if Coord is invalid
    #[test]
    #[should_panic(expected = "Coord row or col too large for this matrix")]
    fn out_of_bounds_access() {
        let matrix = Matrix::new(2, 3);
        matrix.get_at(&Coord::new(2, 2));
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
        let mut a = Matrix::new(2, 3);
        let b = Matrix::new(2, 4);
        a.add(&b);
    }
}