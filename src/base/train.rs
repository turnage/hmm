use std::fmt::Debug;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::hash::Hash;

use base::{Model, Starter, Emitter, Transor};

pub fn discrete<S: Debug + Copy + Eq + Hash, O: Debug + Copy + Eq + Hash>
    (paths: &Vec<Vec<(S, O)>>,
     trans_tunings: Option<Vec<((S, S), f64)>>)
     -> Model<S, O, HashMap<S, f64>, HashMap<S, HashMap<O, f64>>, HashMap<S, HashMap<S, f64>>> {
    let mut train = Discrete::new();
    for path in paths.iter() {
        train.observe_state_path(path);
    }
    train.stochast();
    if let Some(tunings) = trans_tunings {
        for ((s1, s2), p) in tunings {
            *train.trans.entry(s1).or_insert(HashMap::new()).entry(s2).or_insert(p) = p
        }
    }
    train.model()
}

struct Discrete<S: Copy, O: Copy> {
    start: HashMap<S, f64>,
    emit: HashMap<S, HashMap<O, f64>>,
    trans: HashMap<S, HashMap<S, f64>>,
}

impl<S: Debug + Copy + Eq + Hash, O: Debug + Copy + Eq + Hash> Discrete<S, O> {
    fn new() -> Self {
        Discrete {
            start: HashMap::new(),
            emit: HashMap::new(),
            trans: HashMap::new(),
        }
    }

    fn model
        (self)
         -> Model<S, O, HashMap<S, f64>, HashMap<S, HashMap<O, f64>>, HashMap<S, HashMap<S, f64>>> {
        Model::from(self.start, self.emit, self.trans)
    }

    fn observe_state_path(&mut self, points: &Vec<(S, O)>) {
        let &(start_state, _) = points.first().unwrap();
        *self.start.entry(start_state).or_insert(0f64) += 1f64;

        let mut prev = None;
        for &(state, obs) in points.iter() {
            *self.emit
                .entry(state)
                .or_insert(HashMap::new())
                .entry(obs)
                .or_insert(0f64) += 1f64;
            if let Some(prev) = prev {
                *self.trans
                    .entry(prev)
                    .or_insert(HashMap::new())
                    .entry(state)
                    .or_insert(0f64) += 1f64;
            }
            prev = Some(state);
        }
    }

    fn stochast(&mut self) {
        let start_denom = self.start.values().sum::<f64>();
        for (_, pi) in self.start.iter_mut() {
            *pi /= start_denom;
        }

        for (_, dist) in self.emit.iter_mut() {
            let dist_denom = dist.values().sum::<f64>();
            for (_, val) in dist.iter_mut() {
                *val /= dist_denom;
            }
        }

        for (_, dist) in self.trans.iter_mut() {
            let dist_denom = dist.values().sum::<f64>();
            for (_, val) in dist.iter_mut() {
                *val /= dist_denom;
            }
        }
    }
}

impl<S: Debug + Copy + Eq + Hash> Starter<S> for HashMap<S, f64> {
    fn startp(&self, s: S) -> Result<f64, String> {
        if let Some(&p) = self.get(&s) {
            Ok(p)
        } else {
            Ok(0f64)
        }
    }
}

impl<S: Debug + Copy + Eq + Hash, O: Debug + Copy + Eq + Hash> Emitter<S, O> for HashMap<S, HashMap<O, f64>> {
    fn emitp(&self, s: S, o: O) -> Result<f64, String> {
        if let Some(dist) = self.get(&s) {
            if let Some(&p) = dist.get(&o) {
                Ok(p)
            } else {
                Ok(0f64)
            }
        } else {
            Err(format!("state not found"))
        }
    }
}

impl<S: Debug + Copy + Eq + Hash> Transor<S> for HashMap<S, HashMap<S, f64>> {
    fn transp(&self, a: S, b: S) -> Result<f64, String> {
        if let Some(dist) = self.get(&a) {
            if let Some(&p) = dist.get(&b) {
                Ok(p)
            } else {
                Ok(0f64)
            }
        } else {
            Err(format!("state not found"))
        }
    }

    fn states(&self) -> Vec<S> {
        self.keys().cloned().collect()
    }
}
