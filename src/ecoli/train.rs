use ecoli::corpus::Genome;

fn translate(c: char) -> usize {
    match c {
        'A' => 0,
        'C' => 1,
        'T' => 2,
        'G' => 3,
        _ => panic!("invalid DNA"),
    }
}
