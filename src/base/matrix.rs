use base::model::{Starter, Emitter, Transor};

pub trait Matrix {
    fn dimensions(&self) -> (usize, usize);
}

impl Matrix for Vec<Vec<f64>> {
    fn dimensions(&self) -> (usize, usize) {
        if self.len() == 0 {
            (0, 0)
        } else {
            (self.len(), self[0].len())
        }
    }
}

/// The Starter implementation for a vector assumes the vector is a stochastic distribution of
/// starting probabilities such that v[i] is the probability that state i begins a sequence.
impl Starter<usize> for Vec<f64> {
    fn startp(&self, s: usize) -> Result<f64, String> {
        if s < self.len() {
            Ok(self[s])
        } else {
            Err(format!("no start entry for {}; have {} entries", s, self.len()))
        }
    }
}

/// The Emitter implementation for a matrix assumes the matrix's rows are each stochastic
/// distributions for a hidden state, and the observations are indexes of observation classes, so
/// m[i][o] is the probability of state i emitting observation o.
impl Emitter<usize, usize> for Vec<Vec<f64>> {
    fn emitp(&self, state: usize, observation: usize) -> Result<f64, String> {
        let (states, emissions) = self.dimensions();
        if state < states && observation < emissions {
            Ok(self[state][observation])
        } else {
            Err(format!("no emission entry at {}x{}; dist table is {}x{}",
                        state,
                        observation,
                        states,
                        emissions))
        }

    }
}

/// The Transor implementation for a matrix assumes each row is a stochastic distribution of
/// transition probabilities to next states; such that m[a][b] is the probability of transitioning
/// from state a to state b.
impl Transor<usize> for Vec<Vec<f64>> {
    fn transp(&self, a: usize, b: usize) -> Result<f64, String> {
        let (states, _) = self.dimensions();
        if a < states && b < states {
            Ok(self[a][b])
        } else {
            Err(format!("no transition entry at {}x{}; dist table is {}x{}",
                        a,
                        b,
                        states,
                        states))
        }
    }

    fn states(&self) -> Vec<usize> {
        (0..self.len()).collect()
    }
}
