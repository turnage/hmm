use std::ops::{Index, IndexMut};

use base::Matrix;

pub struct Cube<T: Copy> {
    state: Vec<Matrix<T>>,
}

impl<T: Copy> Cube<T> {
    pub fn with_dims(i: usize, j: usize, k: usize, default: T) -> Self {
        let mut state = Vec::with_capacity(k);
        for _ in 0..k {
            state.push(Matrix::with_dims(i, j, default))
        }
        Cube { state: state }
    }
}

impl<T: Copy> Index<usize> for Cube<T> {
    type Output = Matrix<T>;

    fn index(&self, i: usize) -> &Self::Output {
        &self.state[i]
    }
}

impl<T: Copy> IndexMut<usize> for Cube<T> {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        &mut self.state[i]
    }
}
