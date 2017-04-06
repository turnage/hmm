//! Base implements algorithms shared for all or many hmm architectures.

mod matrix;
mod params;

//use float_cmp::ApproxEqUlps;

pub use self::matrix::Matrix;
pub use self::params::Params;

const FLOAT_TOLERANCE: i64 = 1;

#[derive(Debug, PartialEq)]
pub struct Model {
    params: Params,
    seq: Vec<usize>,
    alpha: Option<Matrix>,
    normal_coef: Option<Vec<f64>>,
}
