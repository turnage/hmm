use std::ops::{Index, IndexMut};

use float_cmp::ApproxEqUlps;

const FLOAT_TOLERANCE: i64 = 1;

#[derive(Debug, PartialEq)]
pub struct Matrix {
    state: Vec<Vec<f64>>,
}

impl Matrix {
    pub fn from(state: Vec<Vec<f64>>) -> Self {
        Matrix { state: state }
    }

    pub fn with_dims(n: usize, m: usize) -> Self {
        let mut state: Vec<Vec<f64>> = Vec::with_capacity(n);
        for row in 0..n {
            state[row] = [0.0, 0.0].iter().cloned().cycle().take(m).collect();
        }
        Matrix { state: state }
    }

    pub fn row_stochastic(&self) -> bool {
        self.state.iter().all(|row| row.iter().sum::<f64>().approx_eq_ulps(&1.0, FLOAT_TOLERANCE))
    }

    pub fn dims(&self) -> (usize, usize) {
        let n = self.state.len();
        if n == 0 {
            (0, 0)
        } else {
            (n, self.state[0].len())
        }
    }
}

impl Index<usize> for Matrix {
    type Output = Vec<f64>;

    fn index(&self, i: usize) -> &Self::Output {
        &self.state[i]
    }
}

impl IndexMut<usize> for Matrix {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        &mut self.state[i]
    }
}

#[cfg(test)]
impl Clone for Matrix {
    fn clone(&self) -> Matrix {
        let mut rows = Vec::new();
        for i in 0..self.state.len() {
            rows.push(self.state[i].clone())
        }
        Matrix::from(rows)
    }
}
