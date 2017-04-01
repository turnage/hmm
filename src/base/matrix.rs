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
        self.state.iter().all(|row| row.iter().sum::<f64>() == 1.0)
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
