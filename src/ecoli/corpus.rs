#[derive(Debug, PartialEq)]
pub struct TrainingSet {
    pub noncoding: Vec<(usize, usize)>,
    pub start: Vec<(usize, usize)>,
    pub gene: Vec<(usize, usize)>,
}

#[derive(Debug, PartialEq)]
struct Sequence {
    start: usize,
    end: usize,
    genes: Vec<Gene>,
}

impl Sequence {
    pub fn from(desc: &str) -> Self {
        let terms = desc.split_whitespace()
            .map(|s| s.trim_matches(|c| c == ',' || c == '[' || c == ']'))
            .map(|v| v.parse::<usize>().unwrap())
            .collect::<Vec<usize>>();
        let mut pairs = terms.chunks(2).map(|p| (p[0] - 1, p[1]));

        let (seq_start, seq_end) = pairs.next().unwrap();

        let mut genes = Vec::new();
        while let Some((start, end)) = pairs.next() {
            genes.push(Gene {
                start: start,
                end: end,
            })
        }

        Sequence {
            start: seq_start,
            end: seq_end,
            genes: genes,
        }
    }

    pub fn training_set(&self) -> TrainingSet {
        let mut set = TrainingSet {
            noncoding: Vec::new(),
            start: Vec::new(),
            gene: Vec::new(),
        };
        for (i, gene) in self.genes.iter().enumerate() {
            if i == 0 {
                set.noncoding.push((self.start, gene.start));
            }
            set.start.push((gene.start, gene.start + 3));
            set.gene.push((gene.start + 3, gene.end));
            let noncoding_end = if i == self.genes.len() - 1 {
                self.end
            } else {
                self.genes[i + 1].start
            };
            set.noncoding.push((gene.end, noncoding_end));
        }

        set
    }
}

#[derive(Debug, PartialEq)]
struct Gene {
    start: usize,
    end: usize,
}

pub struct Genome {
    seqs: Vec<Sequence>,
}

impl Genome {
    pub fn from(seqs: String) -> Self {
        Genome { seqs: seqs.lines().map(|li| Sequence::from(li)).collect() }
    }

    pub fn training_set(&self) -> TrainingSet {
        self.seqs.iter().map(|s| s.training_set()).fold(TrainingSet {
                                                            noncoding: Vec::new(),
                                                            start: Vec::new(),
                                                            gene: Vec::new(),
                                                        },
                                                        |mut acc, mut tset| {
            acc.noncoding.append(&mut tset.noncoding);
            acc.start.append(&mut tset.noncoding);
            acc.gene.append(&mut tset.gene);
            acc
        })
    }
}


#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn sequence_from() {
        assert_eq!(Sequence::from("1\t2800\t[190, 255] [337, 2799]"),
                   Sequence {
                       start: 0,
                       end: 2800,
                       genes: vec![Gene {
                                       start: 189,
                                       end: 255,
                                   },
                                   Gene {
                                       start: 336,
                                       end: 2799,
                                   }],
                   })
    }

    #[test]
    fn training_set() {
        let seq = Sequence {
            start: 0,
            end: 2800,
            genes: vec![Gene {
                            start: 189,
                            end: 255,
                        },
                        Gene {
                            start: 336,
                            end: 2799,
                        }],
        };
        assert_eq!(seq.training_set(),
                   TrainingSet {
                       noncoding: vec![(0, 189), (255, 336), (2799, 2800)],
                       start: vec![(189, 192), (336, 339)],
                       gene: vec![(192, 255), (339, 2799)],
                   });
    }
}
