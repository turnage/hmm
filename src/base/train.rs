use base::{Matrix, Model, Emitter};

pub struct Train<E: Emitter> {
    model: Model<E>,
    obs: Vec<E::Observation>,
    alpha: Option<Matrix>,
    normal_coef: Option<Vec<f64>>,
}


impl<E: Emitter> Train<E> {
    pub fn from(model: Model<E>, observations: Vec<E::Observation>) -> Result<Self, String> {
        if observations.len() < 1 {
            Err(format!("no observations provided"))
        } else {
            Ok(Train {
                model: model,
                obs: observations,
                alpha: None,
                normal_coef: None,
            })
        }
    }
}
