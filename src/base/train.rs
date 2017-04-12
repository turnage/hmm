use std::f64;

use base::{Model, Emitter};

pub struct Train;

struct Path {
    states: Vec<usize>,
    p: f64,
}

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
                        p: model.emit.emitp(i, o).map(|p| p.log2() + pi.log2())?,
                    })
                } else {
                    let emitp = model.emit.emitp(i, o)?;
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
    use super::*;

    use base::test_model;

    #[test]
    fn delta() {
        let (model, obs) = (test_model(), vec![0, 1, 0, 2]);
        match Train::most_probable_sequence(&obs, &model) {
            Ok(seq) => assert_eq!(seq, vec![1, 1, 1, 0]),
            Err(e) => panic!(e),
        }
    }
}
