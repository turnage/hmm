use std::fmt;
use std::str::FromStr;

#[derive(Debug, PartialEq)]
struct Sequence {
    start: usize,
    end: usize,
    genes: Vec<(usize, usize)>,
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
