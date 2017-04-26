use std::collections::HashMap;
use std::cmp::Ordering;
use std::fmt;
use std::str::FromStr;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Label {
    NonCoding,
    StartCodon1,
    StartCodon2,
    StartCodon3,
    InternalCodon1(usize),
    InternalCodon2(usize),
    InternalCodon3(usize),
    StopCodon1(usize),
    StopCodon2(usize),
    StopCodon3(usize),
    Invalid,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Base {
    A,
    C,
    T,
    G,
}

impl Base {
    fn from(c: char) -> Result<Base, String> {
        match c {
            'a' => Ok(Base::A),
            'c' => Ok(Base::C),
            't' => Ok(Base::T),
            'g' => Ok(Base::G),
            x => Err(format!("{} is not a valid dna base", x)),
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct Sequence {
    start: usize,
    end: usize,
    genes: Vec<(usize, usize)>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum LabelClass {
    NonCoding,
    StartCodon,
    InternalCodon,
    StopCodon,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct LabeledRange {
    class: LabelClass,
    range: (usize, usize),
}

impl PartialOrd for LabeledRange {
    fn partial_cmp(&self, other: &LabeledRange) -> Option<Ordering> {
        self.range.0.partial_cmp(&other.range.0)
    }

    fn lt(&self, other: &LabeledRange) -> bool {
        self.range.0 < other.range.0
    }

    fn le(&self, other: &LabeledRange) -> bool {
        self.range.0 <= other.range.0
    }

    fn gt(&self, other: &LabeledRange) -> bool {
        self.range.0 > other.range.0
    }

    fn ge(&self, other: &LabeledRange) -> bool {
        self.range.0 >= other.range.0
    }
}

impl Ord for LabeledRange {
    fn cmp(&self, other: &LabeledRange) -> Ordering {
        self.range.0.cmp(&other.range.0)
    }
}

impl Sequence {
    pub fn label_dna(&self, dna: &[Base]) -> Vec<(Label, Base)> {
        let mut emissions = Vec::new();
        let mut internal_codon_table = HashMap::new();
        let mut stop_codon_table = HashMap::new();
        for range in self.labeled_ranges().iter() {
            let (start, end) = range.range;
            let bases = &dna[start..end];
            let mut range_emissions = match range.class {
                LabelClass::NonCoding => Labeler::noncoding(bases),
                LabelClass::StartCodon => Labeler::start_codon(bases),
                LabelClass::StopCodon => Labeler::stop_codon(bases, &mut stop_codon_table),
                LabelClass::InternalCodon => {
                    Labeler::internal_codons(bases, &mut internal_codon_table)
                }
            };
            emissions.append(&mut range_emissions);
        }
        emissions
    }

    fn labeled_ranges(&self) -> Vec<LabeledRange> {
        let labeler = |c| {
            move |&r| {
                LabeledRange {
                    class: c,
                    range: r,
                }
            }
        };
        let mut ranges: Vec<LabeledRange> = self.start_codon_ranges()
            .iter()
            .map(labeler(LabelClass::StartCodon))
            .chain(self.stop_codon_ranges()
                .iter()
                .map(labeler(LabelClass::StopCodon))
                .chain(self.internal_codon_ranges()
                    .iter()
                    .map(labeler(LabelClass::InternalCodon))
                    .chain(self.noncoding_ranges()
                        .iter()
                        .map(labeler(LabelClass::NonCoding)))))
            .collect();
        ranges.sort();
        ranges
    }

    fn start_codon_ranges(&self) -> Vec<(usize, usize)> {
        self.genes.iter().cloned().map(|(start, _)| (start, start + 3)).collect()
    }

    fn stop_codon_ranges(&self) -> Vec<(usize, usize)> {
        self.genes.iter().cloned().map(|(_, end)| (end - 3, end)).collect()
    }

    fn internal_codon_ranges(&self) -> Vec<(usize, usize)> {
        self.genes.iter().cloned().map(|(start, end)| (start + 3, end - 3)).collect()
    }

    fn noncoding_ranges(&self) -> Vec<(usize, usize)> {
        let mut noncoding_ranges = Vec::new();
        for i in 0..(self.genes.len()) {
            let (gstart, gend) = self.genes[i];
            if i == 0 {
                noncoding_ranges.push((self.start, gstart));
            } else {
                let (_, last_gend) = self.genes[i - 1];
                noncoding_ranges.push((last_gend, gstart));
            }

            if i == self.genes.len() - 1 {
                noncoding_ranges.push((gend, self.end));
            }
        }
        noncoding_ranges
    }
}

impl FromStr for Sequence {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let terms = s.split_whitespace()
            .map(|s| s.trim_matches(|c| c == ',' || c == '[' || c == ']'))
            .map(|v| v.parse::<usize>().unwrap())
            .collect::<Vec<usize>>();
        let mut pairs = terms.chunks(2).map(|p| (p[0] - 1, p[1]));

        let (seq_start, seq_end) = pairs.next().unwrap();

        let mut genes = Vec::new();
        while let Some((start, end)) = pairs.next() {
            genes.push((start, end))
        }

        Ok(Sequence {
            start: seq_start,
            end: seq_end,
            genes: genes,
        })
    }
}

impl fmt::Display for Sequence {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let result = write!(f, "{}-{}:", self.start + 1, self.end);
        let mut gene_results = Vec::new();
        for &(gene_start, gene_end) in self.genes.iter() {
            gene_results.push(write!(f, " [{}, {}]", gene_start + 1, gene_end));
        }
        let composite_result = gene_results.drain(0..).fold(result, |acc, r| acc.and(r));
        composite_result
    }
}

struct Labeler;

impl Labeler {
    fn noncoding(bases: &[Base]) -> Vec<(Label, Base)> {
        bases.iter().cloned().map(|b| (Label::NonCoding, b)).collect()
    }

    fn start_codon(bases: &[Base]) -> Vec<(Label, Base)> {
        vec![(Label::StartCodon1, bases[0]),
             (Label::StartCodon2, bases[1]),
             (Label::StartCodon3, bases[2])]
    }

    fn internal_codons(bases: &[Base],
                       instances: &mut HashMap<(Base, Base, Base), usize>)
                       -> Vec<(Label, Base)> {
        Labeler::multi_instance_codons(bases, instances, |i, k| match i {
            0 => Label::InternalCodon1(k),
            1 => Label::InternalCodon2(k),
            2 => Label::InternalCodon3(k),
            _ => Label::Invalid,
        })
    }

    fn stop_codon(bases: &[Base],
                  instances: &mut HashMap<(Base, Base, Base), usize>)
                  -> Vec<(Label, Base)> {
        Labeler::multi_instance_codons(bases, instances, |i, k| match i {
            0 => Label::StopCodon1(k),
            1 => Label::StopCodon2(k),
            2 => Label::StopCodon3(k),
            _ => Label::Invalid,
        })
    }

    fn multi_instance_codons<F>(bases: &[Base],
                                instances: &mut HashMap<(Base, Base, Base), usize>,
                                f: F)
                                -> Vec<(Label, Base)>
        where F: Fn(usize, usize) -> Label
    {
        let mut emissions = Vec::new();
        for codon in bases.chunks(3) {
            let key = (codon[0], codon[1], codon[2]);
            let instance = if let Some(&instance) = instances.get(&key) {
                instance
            } else {
                let instance = instances.len();
                instances.insert(key, instance);
                instance
            };
            for i in 0..3 {
                emissions.push((f(i, instance), codon[i]))
            }
        }
        emissions
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn sequence_from() {
        assert_eq!("1\t2800\t[190, 255] [337, 2799]".parse::<Sequence>(),
                   Ok(Sequence {
                       start: 0,
                       end: 2800,
                       genes: vec![(189, 255), (336, 2799)],
                   }))
    }

    #[test]
    fn sequence_label_dna() {
        assert_eq!(Sequence {
                           start: 0,
                           end: 14,
                           genes: vec![(1, 13)],
                       }
                       .label_dna(&[Base::A, Base::A, Base::T, Base::G, Base::A, Base::A,
                                    Base::A, Base::C, Base::C, Base::C, Base::T, Base::G,
                                    Base::A, Base::T]),
                   vec![(Label::NonCoding, Base::A),
                        (Label::StartCodon1, Base::A),
                        (Label::StartCodon2, Base::T),
                        (Label::StartCodon3, Base::G),
                        (Label::InternalCodon1(0), Base::A),
                        (Label::InternalCodon2(0), Base::A),
                        (Label::InternalCodon3(0), Base::A),
                        (Label::InternalCodon1(1), Base::C),
                        (Label::InternalCodon2(1), Base::C),
                        (Label::InternalCodon3(1), Base::C),
                        (Label::StopCodon1(0), Base::T),
                        (Label::StopCodon2(0), Base::G),
                        (Label::StopCodon3(0), Base::A),
                        (Label::NonCoding, Base::T)])

    }

    #[test]
    fn labeler_internal_codons() {
        let mut unique_checker = HashMap::new();
        unique_checker.insert((Base::A, Base::C, Base::C), 0);
        assert_eq!(Labeler::internal_codons(&[Base::A, Base::C, Base::C, Base::A, Base::A,
                                              Base::A],
                                            &mut unique_checker),
                   vec![(Label::InternalCodon1(0), Base::A),
                        (Label::InternalCodon2(0), Base::C),
                        (Label::InternalCodon3(0), Base::C),
                        (Label::InternalCodon1(1), Base::A),
                        (Label::InternalCodon2(1), Base::A),
                        (Label::InternalCodon3(1), Base::A)]);
    }

    #[test]
    fn sequence_labeled_ranges() {
        assert_eq!(Sequence {
                           start: 0,
                           end: 1000,
                           genes: vec![(100, 200), (600, 700)],
                       }
                       .labeled_ranges(),
                   vec![LabeledRange {
                            class: LabelClass::NonCoding,
                            range: (0, 100),
                        },
                        LabeledRange {
                            class: LabelClass::StartCodon,
                            range: (100, 103),
                        },
                        LabeledRange {
                            class: LabelClass::InternalCodon,
                            range: (103, 197),
                        },
                        LabeledRange {
                            class: LabelClass::StopCodon,
                            range: (197, 200),
                        },
                        LabeledRange {
                            class: LabelClass::NonCoding,
                            range: (200, 600),
                        },
                        LabeledRange {
                            class: LabelClass::StartCodon,
                            range: (600, 603),
                        },
                        LabeledRange {
                            class: LabelClass::InternalCodon,
                            range: (603, 697),
                        },
                        LabeledRange {
                            class: LabelClass::StopCodon,
                            range: (697, 700),
                        },
                        LabeledRange {
                            class: LabelClass::NonCoding,
                            range: (700, 1000),
                        }]);
    }

    #[test]
    fn sequence_noncoding_ranges() {
        assert_eq!(Sequence {
                           start: 0,
                           end: 1000,
                           genes: vec![(100, 200), (600, 999)],
                       }
                       .noncoding_ranges(),
                   vec![(0, 100), (200, 600), (999, 1000)]);
        assert_eq!(Sequence {
                           start: 0,
                           end: 14,
                           genes: vec![(1, 13)],
                       }
                       .noncoding_ranges(),
                   vec![(0, 1), (13, 14)]);
    }

    #[test]
    fn sequence_internal_codon_ranges() {
        assert_eq!(Sequence {
                           start: 0,
                           end: 1000,
                           genes: vec![(100, 200), (600, 700)],
                       }
                       .internal_codon_ranges(),
                   vec![(103, 197), (603, 697)])
    }

    #[test]
    fn sequence_display() {
        assert_eq!(format!("{}",
                           Sequence {
                               start: 0,
                               end: 2800,
                               genes: vec![(189, 255)],
                           }),
                   "1-2800: [190, 255]")
    }
}
