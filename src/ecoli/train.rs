use std::collections::HashMap;

use base::Model;
use base::train;
use ecoli::corpus::TrainingSet;

pub fn model(dna: &String,
             tset: &Vec<(usize, usize)>)
             -> (Model<char,
                       char,
                       HashMap<char, f64>,
                       HashMap<char, HashMap<char, f64>>,
                       HashMap<char, HashMap<char, f64>>>) {
    let mut state_paths = Vec::new();
    for &range in tset.iter() {
        state_paths.push(extract_range(dna, range))
    }
    train::discrete(&state_paths)
}

fn extract_range(dna: &String, (start, end): (usize, usize)) -> Vec<(char, char)> {
    dna.chars().skip(start).take(end - start).map(|c| (c, c)).collect()
}

#[cfg(test)]
mod test {
    use float_cmp::ApproxEqUlps;

    use super::*;

    use base::{FLOAT_TOLERANCE, Solve};

    #[test]
    fn basic_train() {
        let dna = String::from("actgcctggctgtctg");
        let tset = vec![(0, 4), (4, 8), (8, 12), (12, 16)];
        let model = model(&dna, &tset);
        let (_, coefs) = Solve::alpha(&vec!['a'], &model).expect("alpha");
        let p = Solve::probability_of_sequence(&coefs);
        assert!(p.approx_eq_ulps(&0.25, FLOAT_TOLERANCE),
                "got {}; expected 0.25",
                p);
    }
}
