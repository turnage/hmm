use base::Matrix;

/// Emitters represent emission distributions of hidden states.
pub trait Emitter {
    type Observation;
    /// emitp returns the probability of a given state emitting the given observation.
    /// If the hidden state or observation is not recognized, the emitting should return an error
    /// explaining why it is incompatible with the emitter.
    fn emitp(&self, state: usize, observation: Self::Observation) -> Result<f64, String>;
}

#[derive(Debug, PartialEq)]
pub struct Model<E> {
    n: usize,
    init: Matrix,
    trans: Matrix,
    emit: E,
}

impl<E> Model<E> {
    /// from returns a Model hmm from the initial distribution of N hidden states, probability matrix
    /// of hidden state transitions, and probability distributions of M possible emissions from
    /// each hidden state (traditionally denoted pi, a, and b respectively).
    pub fn from(init: Matrix, trans: Matrix, emit: E) -> Result<Self, String> {
        let check = &|valid, error| if valid { Ok(()) } else { Err(error) };
        let ((_, init_states), (trans_rows, trans_cols)) = (init.dims(), trans.dims());
        check(init.row_stochastic(),
              format!("initial dist is not row stochastic"))
            .and(check(trans.row_stochastic(),
                       format!("transform dist is not row stochastic")))
            .and(check(init_states > 1,
                       format!("got {} hidden states in initial dist; need > 1",
                               init_states)))
            .and(check(trans_rows == init_states && trans_cols == init_states,
                       format!("got {}x{} transform dist; need {}x{}",
                               trans_rows,
                               trans_cols,
                               init_states,
                               init_states)))
            .and(Ok(Model {
                n: init_states,
                init: init,
                trans: trans,
                emit: emit,
            }))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn test_model() -> Model<Matrix> {
        Model {
            n: 2,
            init: Matrix::from(vec![vec![0.4, 0.6]]),
            trans: Matrix::from(vec![vec![0.7, 0.3], vec![0.4, 0.6]]),
            emit: Matrix::from(vec![vec![0.1, 0.4, 0.5], vec![0.7, 0.2, 0.1]]),
        }
    }

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
