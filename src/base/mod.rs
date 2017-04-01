//! Base implements algorithms shared for all or many hmm architectures.

mod matrix;

use self::matrix::Matrix;

pub struct Base {
    init: Matrix,
    trans: Matrix,
    emit: Matrix,
}

impl Base {
    /// from returns a Base hmm from the initial distribution of hidden states, probability matrix
    /// of hidden state transitions, and probability distributions of emissions from each hidden
    /// state (traditionally denoted pi, a, and b respectively).
    ///
    /// The provided matrices must be row-stochastic. Example format:
    ///
    ///     init: [0.6 0.4]
    ///     trans: [0.7 0.3 // trans[i][j] is the probability of transitioning from state i to j.
    ///             0.4 0.6]
    ///     emit: [0.1 0.4 0.5  // each column is an emission; each row is a state's emission dist.
    ///            0.7 0.2 0.1]
    pub fn from(init: Matrix, trans: Matrix, emit: Matrix) -> Result<Self, String> {
        let check = &|valid, error| if valid { Ok(()) } else { Err(error) };
        let ((ir, ic), (tr, tc), (er, ec)) = (init.dims(), trans.dims(), emit.dims());
        check(init.row_stochastic(),
              format!("initial dist is not row stochastic"))
            .and(check(trans.row_stochastic(),
                       format!("transform dist is not row stochastic")))
            .and(check(emit.row_stochastic(),
                       format!("emissions dist is not row stochastic")))
            .and(check(ir > 1,
                       format!("got {} hidden states in initial dist; need > 1", ir)))
            .and(check(tr != ir || tc != ir,
                       format!("got {}x{} transform dist; need {}x{}", tr, tc, ir, ir)))
            .and(check(er != ir,
                       format!("got {} emissions dists; need N={}", er, ir)))
            .and(check(ec > 1, format!("got {} possible emissions; need > 1", ec)))
            .and(Ok(Base {
                init: init,
                trans: trans,
                emit: emit,
            }))
    }
}
