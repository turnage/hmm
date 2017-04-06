use base::Matrix;

#[derive(Debug, PartialEq)]
pub struct Params {
    n: usize,
    m: usize,
    init: Matrix,
    trans: Matrix,
    emit: Matrix,
}

impl Params {
    /// from returns a Params hmm from the initial distribution of N hidden states, probability matrix
    /// of hidden state transitions, and probability distributions of M possible emissions from
    /// each hidden state (traditionally denoted pi, a, and b respectively).
    pub fn from(init: Matrix, trans: Matrix, emit: Matrix) -> Result<Self, String> {
        let (n, m) = emit.dims();
        let check = &|valid, error| if valid { Ok(()) } else { Err(error) };
        let ((_, ic), (tr, tc), (er, ec)) = (init.dims(), trans.dims(), emit.dims());
        check(init.row_stochastic(),
              format!("initial dist is not row stochastic"))
            .and(check(trans.row_stochastic(),
                       format!("transform dist is not row stochastic")))
            .and(check(emit.row_stochastic(),
                       format!("emissions dist is not row stochastic")))
            .and(check(ic > 1,
                       format!("got {} hidden states in initial dist; need > 1", ic)))
            .and(check(tr == ic && tc == ic,
                       format!("got {}x{} transform dist; need {}x{}", tr, tc, ic, ic)))
            .and(check(er == ic,
                       format!("got {} emissions dists; need N={}", er, ic)))
            .and(check(ec > 1, format!("got {} possible emissions; need > 1", ec)))
            .and(Ok(Params {
                n: n,
                m: m,
                init: init,
                trans: trans,
                emit: emit,
            }))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn test_params() -> Params {
        Params {
            n: 2,
            m: 3,
            init: Matrix::from(vec![vec![0.4, 0.6]]),
            trans: Matrix::from(vec![vec![0.7, 0.3], vec![0.4, 0.6]]),
            emit: Matrix::from(vec![vec![0.1, 0.4, 0.5], vec![0.7, 0.2, 0.1]]),
        }
    }

    #[test]
    fn from() {
        let base = test_base();
        assert_eq!(Params::from(base.init.clone(), base.trans.clone(), base.emit.clone()),
                   Ok(Params {
                       n: 2,
                       m: 3,
                       init: base.init,
                       trans: base.trans,
                       emit: base.emit,
                   }));
    }
}
