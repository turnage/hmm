//! Base implements algorithms shared for all or many hmm architectures.

mod matrix;
mod model;
mod train;

pub use self::matrix::Matrix;
pub use self::model::{Model, Emitter};
pub use self::train::Train;

const FLOAT_TOLERANCE: i64 = 1;
