use std::fs;
use crate::linear_algebra::Matrix;

pub struct Row {
    x: Matrix,
    y: f64,
}

/// A structure representing a training or validation dataset
pub struct Dataset {
    /// The number of rows in the dataset
    rows: usize,
    /// The number of independent variables in the dataset
    x_vars: usize,
    /// The rows of the dataset
    data: Vec<Row>,
}

impl Dataset {
    /// Returns an empty dataset
    pub fn new() -> Dataset {
        Dataset {
            rows: 0,
            x_vars: 0,
            data: Vec::new(),
        }
    }

    /// Loads data from a csv file
    pub fn load_data(&mut self, filename: &str) {
        // read data from file
        let contents = fs::read_to_string(filename)
            .expect("Reading file failed");
        // create variables for each row of data
        let mut x: Vec<f64>;
        let mut y: f64;
        // parse and save each row of the dataset
        for line in contents.lines() {
            x = line.split(",").map(|field| {
                field.parse::<f64>().unwrap()
            }).collect();
            y = x.pop().unwrap();
            self.data.push(Row { 
                x: Matrix::from_vec(1, x.len(), x),
                y,
            });
        }
        // set number of rows and x_vars
        self.rows = self.data.len();
        self.x_vars = self.data[0].x.get_cols();
    }

    /// Prints the dataset for testing and debugging purposes
    pub fn print(&self) {
        for m in self.data.iter() {
            m.x.print();
            print!("{}", m.y);
        }
    }

    /// Returns the length of the dataset
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns a reference to the given row
    pub fn get_at(&self, index: usize) -> &Row {
        &self.data[index]
    }

    /// Returns an iterator over the dataset
    pub fn iter(&mut self) -> DatasetIterator {
        DatasetIterator::new(self)
    }
}

pub struct DatasetIterator<'a> {
    dataset: &'a Dataset,
    index: usize,
}

impl<'a> DatasetIterator<'a> {
    fn new(dataset: &mut Dataset) -> DatasetIterator {
        DatasetIterator {
            dataset,
            index: 0,
        }
    }
}

impl<'a> Iterator for DatasetIterator<'a> {
    type Item = &'a Row;

    fn next(&mut self) -> Option<Self::Item> {
        let mut result = None;
        if self.index < self.dataset.len() {
            result = Some(self.dataset.get_at(self.index));
        }
        self.index += 1;
        result
    }
}

// Unit tests
#[cfg(test)]
mod tests {
    use super::*;

    // test loading of data
    #[test]
    fn data_loading() {
        let mut dataset = Dataset::new();
        dataset.load_data("data/banknote_train.csv");
        assert_eq!(dataset.x_vars, 4);
        assert_eq!(dataset.rows, 1097);
        assert_eq!(dataset.data.len(), 1097);
    }


    // test whether iterating over the dataset works
    #[test]
    fn dataset_iteration() {
        let mut dataset = Dataset::new();
        dataset.load_data("data/banknote_train.csv");
        let mut count;
        for i in 0..2 {
            count = 0;
            for m in dataset.iter() {
                //print!("{}: ", i + 1);
                //m.print();
                count += 1;
            }
            assert_eq!(count, 1097);
        }
    }
}