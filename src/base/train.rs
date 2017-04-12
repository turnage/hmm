use std::f64;

use base::{Matrix, Model, Emitter};

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
        println!("COEFS: {:?}", alpha.coefs);
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
