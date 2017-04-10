use std::iter::{IntoIterator, FromIterator};
use std::ops::{Index, IndexMut};
use std::slice::Iter;

use float_cmp::ApproxEqUlps;

use base::FLOAT_TOLERANCE;
use base::model::Emitter;

#[derive(Debug, PartialEq)]
pub struct Matrix<T> {
    state: Vec<Vec<T>>,
}

impl<T> Matrix<T> {
    pub fn from(state: Vec<Vec<T>>) -> Self {
        Matrix { state: state }
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

impl<T: Copy> Matrix<T> {
    pub fn with_dims(n: usize, m: usize, default: T) -> Self {
        let mut state: Vec<Vec<T>> = Vec::with_capacity(n);
        for _ in 0..n {
            state.push([default].iter().cloned().cycle().take(m).collect());
        }
        Matrix { state: state }
    }
}

impl Matrix<f64> {
    pub fn row_stochastic(&self) -> bool {
        self.state.iter().all(|row| row.iter().sum::<f64>().approx_eq_ulps(&1.0, FLOAT_TOLERANCE))
    }
}

/// The Emitter implementation for Matrix assumes the matrix's rows are each row stochastic
/// distributions for a hidden state, and the observations are indexes of observation classes, so
/// m[i][o] is the probability of state i emitting observation o.
impl Emitter for Matrix<f64> {
    type Observation = usize;

    fn emitp(&self, state: usize, observation: &Self::Observation) -> Result<f64, String> {
        let (states, emissions) = self.dims();
        if state < states && *observation < emissions {
            Ok(self.state[state][*observation])
        } else {
            Err(format!("no emission entry ay {}x{}; dist table is {}x{}",
                        state,
                        *observation,
                        states,
                        emissions))
        }

    }
}

impl<T> Index<usize> for Matrix<T> {
    type Output = Vec<T>;

    fn index(&self, i: usize) -> &Self::Output {
        &self.state[i]
    }
}

impl<T> IndexMut<usize> for Matrix<T> {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        &mut self.state[i]
    }
}

#[cfg(test)]
impl<T: Clone> Clone for Matrix<T> {
    fn clone(&self) -> Matrix<T> {
        let mut rows = Vec::new();
        for i in 0..self.state.len() {
            rows.push(self.state[i].clone())
        }
        Matrix::from(rows)
    }
}
