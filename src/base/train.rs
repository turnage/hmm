use std::f64;

use base::{Matrix, Model, Emitter, scale};

pub struct Train;

#[derive(Debug)]
struct Path {
    states: Vec<usize>,
    p: f64,
}

impl Train {
    pub fn delta_pass<E: Emitter>(obs: &Vec<E::Observation>,
                                  model: &Model<E>)
                                  -> Result<Vec<usize>, String> {
        // we track a state path which arrives at each possible state.
        let mut paths = Vec::new();
        for (t, o) in obs.iter().enumerate() {
            for (i, pi) in model.init.iter().enumerate() {
                if t == 0 {
                    paths.push(Path {
                        states: vec![i],
                        p: model.emit.emitp(i, o).map(|p| p * pi)?,
                    })
                } else {
                    let emitp = model.emit.emitp(i, o)?;
                    let connect = |p: f64, prev: usize| p * model.trans[prev][i] * emitp;
                    let (prev, p) = Train::max(&paths, connect);
                    paths[i].states.push(prev);
                    paths[i].p = p;
                }

            }
            println!("PATHS t={}: {:?}", t, paths);
        }

        println!("PATHS: {:?}", paths);

        Ok(Vec::new())
    }

    fn max<F>(ps: &Vec<Path>, f: F) -> (usize, f64)
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
    use super::*;

    use base::test_model;

    #[test]
    fn delta() {
        let (model, obs) = (test_model(), vec![0, 1, 0, 2]);
        match Train::delta_pass(&obs, &model) {
            Ok(seq) => assert_eq!(seq, vec![1, 1, 1, 0]),
            Err(e) => panic!(e),
        }
    }
}
