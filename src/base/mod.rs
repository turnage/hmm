//! Base implements algorithms shared for all or many hmm architectures.

mod matrix;
mod model;

//use float_cmp::ApproxEqUlps;

pub use self::matrix::Matrix;
pub use self::model::Model;

const FLOAT_TOLERANCE: i64 = 1;
