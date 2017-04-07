use base::{Matrix, Model, Emitter};

pub struct Train;

impl Train {
    /// p returns the probability of the observation sequence using the given alpha and
    /// normalization coefficient vector.
    pub fn p(alpha: &Matrix, normal_coefs: &Vec<f64>) -> Result<f64, String> {
        let (t, _) = alpha.dims();
        if t == 0 || normal_coefs.is_empty() {
            Err(format!("alpha matrix or normalization coefficients missing"))
        } else if t != normal_coefs.len() {
            Err(format!("need {} normalization coefficients; have {}",
                        t,
                        normal_coefs.len()))
        } else {
            Ok(normal_coefs.iter().map(|v| -1.0 * v.log2()).sum::<f64>().exp2())
        }
    }

    /// alpha_pass computes the alpha function and normalization coefficients for the train's
    /// observations.
    pub fn alpha_pass<E: Emitter>(obs: &Vec<E::Observation>,
                                  model: &Model<E>)
                                  -> Result<(Matrix, Vec<f64>), String> {
        let (tlen, n) = (obs.len(), model.n);
        let mut raw_alpha = Matrix::with_dims(tlen, n);
        let mut normal_alpha = Matrix::with_dims(tlen, n);
        let mut normal_coefs = Vec::with_capacity(obs.len());
        let trans = |i| if i == 0 { 1.0 } else { model.trans[i - 1][i] };
        for (t, o) in obs.iter().enumerate() {
            let mut normalizer = 0.0;
            for (i, pi) in model.init.iter().enumerate() {
                let reducer = if t == 0 { pi * trans(i) } else { trans(i) };
                raw_alpha[t][i] = model.emit.emitp(i, o).map(|v| v * reducer)?;
                normalizer += raw_alpha[t][i];
            }
            normal_coefs.push(normalizer.recip());
            for i in 0..model.n {
                normal_alpha[t][i] = normal_coefs[t] * raw_alpha[t][i];
            }
        }
        Ok((normal_alpha, normal_coefs))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use base::test_model;

    #[test]
    fn p() {
        let (model, obs) = (test_model(), vec![0, 0, 0, 0]);
        match Train::alpha_pass(&obs, &model) {
            Ok((alpha, normal_coefs)) => println!("P: {:?}", Train::p(&alpha, &normal_coefs)),
            Err(e) => panic!(e),
        }
    }
}
