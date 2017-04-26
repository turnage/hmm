use std::f64;
use std::fmt::Debug;

use float_cmp::ApproxEqUlps;

use base::{FLOAT_TOLERANCE, Model, Starter, Emitter, Transor};

#[derive(Debug)]
struct Path<S> {
    states: Vec<S>,
    p: f64,
}

pub struct Solve;

impl Solve {
    pub fn most_probable_sequence<S, O, St, E, T>(obs: &Vec<O>,
                                                  model: &Model<S, O, St, E, T>)
                                                  -> Result<Vec<S>, String>
        where S: Copy + Debug,
              O: Copy,
              St: Starter<S>,
              E: Emitter<S, O>,
              T: Transor<S>
    {
        let states = model.trans.states();
        let mut paths = Vec::new();
        for (t, &o) in obs.iter().enumerate() {
            let mut path_updates = Vec::new();
            for (i, &s) in states.iter().enumerate() {
                if t == 0 {
                    let pi = model.start.startp(s)?;
                    paths.push(Path {
                        states: Vec::new(),
                        p: model.emitter.emitp(s, o).map(|p| p.log2() + pi.log2())?,
                    })
                } else {
                    let emitp = model.emitter.emitp(s, o)?;
                    let (prev_path, prev_state, p) = if 
                        !emitp.approx_eq_ulps(&0f64, FLOAT_TOLERANCE) {
                        let connect = |p: f64, prev: S| {
                            let tp = model.trans.transp(prev, s)?;
                            if tp.approx_eq_ulps(&0f64, FLOAT_TOLERANCE) {
                                Ok(None)
                            } else {
                                Ok(Some(p + tp.log2() + emitp.log2()))
                            }
                        };
                        Solve::best_path(&paths, &states, connect)?
                    } else {
                        (i, s, 0f64.log2())
                    };
                    let statepath = if prev_path != i {
                        Some(paths[prev_path].states.to_vec())
                    } else {
                        None
                    };
                    path_updates.push((statepath, prev_state, p));
                }
            }
            for (i, (statepath, new_state, p)) in path_updates.drain(0..).enumerate() {
                if let Some(new_path) = statepath {
                    paths[i].states = new_path;
                }
                paths[i].states.push(new_state);
                paths[i].p = p;
            }
        }

        // for (i, path) in paths.iter().enumerate() {
        // println!("{:?} -> {:?}", states[i], path);
        // }

        let (path_end, last_state, _) =
            Solve::best_path(&paths, &states, |p: f64, _: S| Ok(Some(p)))?;
        paths[path_end].states.push(last_state);

        Ok(paths.swap_remove(path_end).states)
    }

    fn best_path<F, S: Copy + Debug>(ps: &Vec<Path<S>>,
                                     states: &Vec<S>,
                                     f: F)
                                     -> Result<(usize, S, f64), String>
        where F: Fn(f64, S) -> Result<Option<f64>, String>
    {
        let mut max_path = 0;
        let mut max_state = states[max_path];
        let mut max = 0f64.log2();
        for (i, p) in ps.iter().enumerate() {
            if let Some(p) = f(p.p, states[i])? {
                if p > max {
                    max = p;
                    max_state = states[i];
                    max_path = i;
                }
            }
        }
        Ok((max_path, max_state, max))
    }

    pub fn probability_of_sequence(coefs: &Vec<f64>) -> f64 {
        coefs.iter().fold(0f64, |p: f64, c: &f64| p + c.log2()).exp2().recip()
    }

    pub fn alpha<S, O, St, E, T>(obs: &Vec<O>,
                                 model: &Model<S, O, St, E, T>)
                                 -> Result<(Vec<Vec<f64>>, Vec<f64>), String>
        where S: Copy,
              O: Copy,
              St: Starter<S>,
              E: Emitter<S, O>,
              T: Transor<S>
    {
        let states = model.trans.states();
        let mut normal = vec![vec![0f64; states.len()]; obs.len()];
        let mut coefs = vec![0f64];

        for (i, &s) in states.iter().enumerate() {
            let pi = model.start.startp(s)?;
            normal[0][i] = pi * model.emitter.emitp(s, obs[0])?;
            coefs[0] += normal[0][i];
        }
        coefs[0] = coefs[0].recip();

        for i in 0..states.len() {
            normal[0][i] *= coefs[0];
        }

        for (t, &o) in obs.iter().enumerate().skip(1) {
            coefs.push(0f64);
            for (i, &a) in states.iter().enumerate() {
                for (j, &b) in states.iter().enumerate() {
                    normal[t][i] += normal[t - 1][j] * model.trans.transp(b, a)?;
                }
                normal[t][i] *= model.emitter.emitp(a, o)?;
                coefs[t] += normal[t][i];
            }
            coefs[t] = coefs[t].recip();
            for i in 0..states.len() {
                normal[t][i] *= coefs[t];
            }
        }

        Ok((normal, coefs))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use base::test_model;

    #[test]
    fn delta() {
        let (model, obs) = (test_model(), vec![0, 1, 0, 2]);
        match Solve::most_probable_sequence(&obs, &model) {
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
            let (_, coefs) = Solve::alpha(&t.obs, &model).expect("failed to create alpha");
            let p = Solve::probability_of_sequence(&coefs);
            assert!(p.approx_eq_ulps(&t.p, FLOAT_TOLERANCE),
                    "got {}; expected {}",
                    p,
                    t.p);
        }
    }
}
