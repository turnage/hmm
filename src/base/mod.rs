//! Base implements algorithms shared for all or many hmm architectures.

mod matrix;
mod model;
mod solve;

pub use self::matrix::Matrix;
pub use self::model::{Model, Starter, Emitter, Transor};
pub use self::solve::Solve;

const FLOAT_TOLERANCE: i64 = 2;

/// test_model defines the discrete hmm used in Mark Stamp's HMM paper published by SJSU, which
/// includes test results for the model problems which are used throughout this library. Follow
/// this URL for a reference: http://www.cs.sjsu.edu/~stamp/RUA/HMM.pdf
#[cfg(test)]
fn test_model() -> Model<usize, usize, Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>> {
    Model::from(vec![0.6, 0.4],
                vec![vec![0.1, 0.4, 0.5], vec![0.7, 0.2, 0.1]],
                vec![vec![0.7, 0.3], vec![0.4, 0.6]])
}
