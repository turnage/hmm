use base::Model;

pub struct Multinomial {
    n: usize,
    m: usize,
    model: Model<usize, usize, Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>>,
}

impl Multinomial {
    pub fn new(n: usize, m: usize) -> Self {
        Multinomial {
            n: n,
            m: m,
            model: Model::from(Vec::new(), Vec::new(), Vec::new()),
        }
    }

    /// learn trains the model on sequences, which are pairs of (hidden state, emission)
    /// so that it can predict similar sequences.
    pub fn learn(n: usize,
                 m: usize,
                 seqs: Vec<Vec<(usize, usize)>>)
                 -> Model<usize, usize, Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>> {
        let mut starts = vec![0f64; n];
        let mut emit = vec![vec![0f64; m]; n];
        let mut trans = vec![vec![0f64; n]; n];
        let mut prev: Option<usize> = None;
        for seq in seqs.iter() {
            let (state, _) = seq[0];
            starts[state] += 1f64;
            for (i, &(state, emission)) in seq.iter().enumerate() {
                if let Some(prev) = prev {
                    trans[prev][state] += 1f64;
                }
                emit[state][emission] += 1f64;
                prev = Some(state);
            }
        }

        Model::from(Multinomial::stochast(&starts),
                    emit.iter().map(|dist| Multinomial::stochast(dist)).collect(),
                    trans.iter().map(|dist| Multinomial::stochast(dist)).collect())
    }

    /// stochast makes a vector stochastic by dividing all elements by the vector sum.
    fn stochast(v: &Vec<f64>) -> Vec<f64> {
        let sum = v.iter().sum::<f64>();
        v.iter().map(|v| v / sum).collect()
    }
}
