//! Base implements algorithms shared for all or many hmm architectures.

mod matrix;
mod model;
mod train;

pub use self::matrix::Matrix;
pub use self::model::{Model, Emitter};
pub use self::train::Train;

const FLOAT_TOLERANCE: i64 = 1;


/// test_model defines the discrete hmm used in Mark Stamp's HMM paper published by SJSU, which
/// includes test results for the model problems which are used throughout this library. Follow
/// this URL for a reference: http://www.cs.sjsu.edu/~stamp/RUA/HMM.pdf
#[cfg(test)]
fn test_model() -> Model<Matrix> {
    Model {
        n: 2,
        init: vec![0.4, 0.6],
        trans: Matrix::from(vec![vec![0.7, 0.3], vec![0.4, 0.6]]),
        emit: Matrix::from(vec![vec![0.1, 0.4, 0.5], vec![0.7, 0.2, 0.1]]),
    }
}
