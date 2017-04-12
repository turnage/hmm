use std::f64;

use base::{Cube, Matrix, Model, Emitter};

struct Path {
    states: Vec<usize>,
    p: f64,
}

pub struct Alpha {
    normal: Matrix<f64>,
    coefs: Vec<f64>,
}

pub struct Train;

impl Train {
    pub fn most_probable_sequence<E: Emitter>(obs: &Vec<E::Observation>,
                                              model: &Model<E>)
                                              -> Result<Vec<usize>, String> {
        let mut paths = Vec::new();
        for (t, o) in obs.iter().enumerate() {
            for (i, pi) in model.init.iter().enumerate() {
                if t == 0 {
                    paths.push(Path {
                        states: Vec::new(),
                        p: model.emitter.emitp(i, o).map(|p| p.log2() + pi.log2())?,
                    })
                } else {
                    let emitp = model.emitter.emitp(i, o)?;
                    let connect =
                        |p: f64, prev: usize| p + model.trans[prev][i].log2() + emitp.log2();
                    let (prev, p) = Train::best_path(&paths, connect);
                    if prev != i {
                        paths[i].states = paths[prev].states.to_vec();
                    }
                    paths[i].states.push(prev);
                    paths[i].p = p;
                }
            }
        }

        let (path_end, _) = Train::best_path(&paths, |p: f64, _: usize| p);
        paths[path_end].states.push(path_end);

        Ok(paths.swap_remove(path_end).states)
    }

    pub fn probability_of_sequence(alpha: &Alpha) -> f64 {
        alpha.coefs.iter().fold(0f64, |p: f64, c: &f64| p + c.log2()).exp2().recip()
    }

    pub fn alpha<E: Emitter>(obs: &Vec<E::Observation>, model: &Model<E>) -> Result<Alpha, String> {
        let mut alpha = Alpha {
            normal: Matrix::with_dims(obs.len(), model.n, 0f64),
            coefs: vec![0f64],
        };

        for (i, pi) in model.init.iter().enumerate() {
            alpha.normal[0][i] = pi * model.emitter.emitp(i, obs.first().unwrap())?;
            alpha.coefs[0] += alpha.normal[0][i];
        }
        alpha.coefs[0] = alpha.coefs[0].recip();

        for i in 0..model.n {
            alpha.normal[0][i] *= alpha.coefs[0];
        }

        for (t, o) in obs.iter().enumerate().skip(1) {
            alpha.coefs.push(0f64);
            for i in 0..model.n {
                for j in 0..model.n {
                    alpha.normal[t][i] += alpha.normal[t - 1][j] * model.trans[j][i];
                }
                alpha.normal[t][i] *= model.emitter.emitp(i, o)?;
                alpha.coefs[t] += alpha.normal[t][i];
            }
            alpha.coefs[t] = alpha.coefs[t].recip();
            for i in 0..model.n {
                alpha.normal[t][i] *= alpha.coefs[t];
            }
        }

        Ok(alpha)
    }

    pub fn beta<E: Emitter>(obs: &Vec<E::Observation>,
                            model: &Model<E>,
                            coefs: &Vec<f64>)
                            -> Result<Matrix<f64>, String> {
        let mut beta = Matrix::with_dims(obs.len(), model.n, 0f64);
        for i in 0..model.n {
            beta[obs.len() - 1][i] = coefs[obs.len() - 1];
        }

        for (t, o) in obs.iter().enumerate().skip(1).rev() {
            for i in 0..model.n {
                beta[t][i] = 0f64;
                let emitp = model.emitter.emitp(i, o)?;
                for j in 0..model.n {
                    beta[t][i] += model.trans[i][j] * emitp * beta[t + 1][j];
                }
                beta[t][i] *= coefs[t];
            }
        }

        Ok(beta)
    }

    /// gammas returns the gap gamma g(i,j) and unit gamma g(i).
    pub fn gammas<E: Emitter>(obs: &Vec<E::Observation>,
                              model: &Model<E>,
                              alpha: &Matrix<f64>,
                              beta: &Matrix<f64>)
                              -> Result<(Cube<f64>, Matrix<f64>), String> {
        let mut gamma = Matrix::with_dims(obs.len(), model.n, 0f64);
        let mut gap_gamma = Cube::with_dims(obs.len(), model.n, model.n, 0f64);
        for t in 0..(obs.len() - 2) {
            let mut denom = 0f64;
            for i in 0..model.n {
                for j in 0..model.n {
                    let emitp = model.emitter.emitp(i, &obs[t + 1])?;
                    denom += alpha[t][i] * model.trans[i][j] * emitp * beta[t + 1][j];
                }
            }
            for i in 0..model.n {
                gamma[t][i] = 0f64;
                for j in 0..model.n {
                    let emitp = model.emitter.emitp(i, &obs[t + 1])?;
                    gap_gamma[t][i][j] = alpha[t][i] * model.trans[i][j] * emitp * beta[t + 1][j] /
                                         denom;
                    gamma[t][i] += gap_gamma[t][i][j];
                }
            }
        }

        let denom = alpha[obs.len() - 1].iter().sum::<f64>();
        for i in 0..model.n {
            gamma[obs.len() - 1][i] = alpha[obs.len() - 1][i] / denom;
        }

        Ok((gap_gamma, gamma))
    }

    /// estimates the initial distribution of hidden states using gamma.
    pub fn estimate_pi(gamma: &Matrix<f64>) -> Vec<f64> {
        gamma[0].iter().cloned().collect()
    }

    /// estimates the transition matrix of hidden states using the gap_gamma and gamma.
    pub fn estimate_trans(gap_gamma: &Cube<f64>, gamma: &Matrix<f64>) -> Matrix<f64> {
        let (t, n) = gamma.dims();
        let mut trans = Matrix::with_dims(n, n, 0f64);
        for i in 0..n {
            for j in 0..n {
                let mut numer = 0f64;
                let mut denom = 0f64;
                for t in 0..(t - 1) {
                    numer += gap_gamma[t][i][j];
                    denom += gamma[t][i];
                }
                trans[i][j] = numer / denom;
            }
        }
        trans
    }

    /// estimates a discrete emission matrix using gamma.
    pub fn estimate_discrete_emissions(m: usize,
                                       obs: Vec<usize>,
                                       gamma: &Matrix<f64>)
                                       -> Matrix<f64> {
        let (n, t) = gamma.dims();
        let mut emitter = Matrix::with_dims(n, m, 0f64);
        for i in 0..n {
            for j in 0..m {
                let mut numer = 0f64;
                let mut denom = 0f64;
                for t in 0..t {
                    if obs[t] == j {
                        numer += gamma[t][i];
                    }
                    denom += gamma[t][i];
                }
                emitter[i][j] = numer / denom;
            }
        }
        emitter
    }

    fn best_path<F>(ps: &Vec<Path>, f: F) -> (usize, f64)
        where F: Fn(f64, usize) -> f64
    {
        let mut max_state = 0;
        let mut max = f64::MIN;
        for (i, p) in ps.iter().enumerate() {
            let p = f(p.p, i);
            if p > max {
                max = p;
                max_state = i;
            }
        }
        (max_state, max)
    }
}

#[cfg(test)]
mod test {
    use float_cmp::ApproxEqUlps;

    use super::*;

    use base::{FLOAT_TOLERANCE, test_model};

    #[test]
    fn delta() {
        let (model, obs) = (test_model(), vec![0, 1, 0, 2]);
        match Train::most_probable_sequence(&obs, &model) {
            Ok(seq) => assert_eq!(seq, vec![1, 1, 1, 0]),
            Err(e) => panic!(e),
        }
    }

    struct AlphaTest {
        obs: Vec<usize>,
        p: f64,
    }

    #[test]
    fn alpha() {
        let model = test_model();
        // These test values were created by hand using the test model.
        let tests = vec![AlphaTest {
                             obs: vec![0, 0],
                             p: 0.1456,
                         },
                         AlphaTest {
                             obs: vec![1, 0],
                             p: 0.104,
                         }];
        for t in tests.iter() {
            let alpha = Train::alpha(&t.obs, &model).expect("failed to create alpha");
            let p = Train::probability_of_sequence(&alpha);
            assert!(p.approx_eq_ulps(&t.p, FLOAT_TOLERANCE),
                    "got {}; expected {}",
                    p,
                    t.p);
        }
    }
}
