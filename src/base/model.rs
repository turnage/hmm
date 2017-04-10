use float_cmp::ApproxEqUlps;

use base::Matrix;
use base::FLOAT_TOLERANCE;

pub trait Emitter {
    type Observation;
    /// emitp returns the probability of a given state emitting the given observation.
    /// If the hidden state or observation is not recognized, the emitting should return an error
    /// explaining why it is incompatible with the emitter.
    fn emitp(&self, state: usize, observation: &Self::Observation) -> Result<f64, String>;
}

#[derive(Debug, PartialEq)]
pub struct Model<E> {
    pub n: usize,
    pub init: Vec<f64>,
    pub trans: Matrix<f64>,
    pub emit: E,
}

impl<E> Model<E> {
    /// from returns a Model hmm from the initial distribution of N hidden states, probability
    /// matrix of hidden state transitions, and probability distributions of M possible emissions
    /// from each hidden state (traditionally denoted pi, a, and b respectively).
    pub fn from(init: Vec<f64>, trans: Matrix<f64>, emit: E) -> Result<Self, String> {
        let check = &|valid, error| if valid { Ok(()) } else { Err(error) };
        let (n, (trans_rows, trans_cols)) = (init.len(), trans.dims());
        check(init.iter().sum::<f64>().approx_eq_ulps(&1.0, FLOAT_TOLERANCE),
              format!("initial dist is not row stochastic"))
            .and(check(trans.row_stochastic(),
                       format!("transform dist is not row stochastic")))
            .and(check(n > 1,
                       format!("got {} hidden states in initial dist; need > 1", n)))
            .and(check(trans_rows == n && trans_cols == n,
                       format!("got {}x{} transform dist; need {}x{}",
                               trans_rows,
                               trans_cols,
                               n,
                               n)))
            .and(Ok(Model {
                n: n,
                init: init,
                trans: trans,
                emit: emit,
            }))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use base::test_model;

    #[test]
    fn from() {
        let model = test_model();
        assert_eq!(Model::from(model.init.clone(), model.trans.clone(), model.emit.clone()),
                   Ok(Model {
                       n: 2,
                       init: model.init,
                       trans: model.trans,
                       emit: model.emit,
                   }));
    }
}
