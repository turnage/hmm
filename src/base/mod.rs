mod matrix;

use self::matrix::Matrix;

pub struct Base {
    init: Matrix,
    trans: Matrix,
    emit: Matrix,
}

impl Base {
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
