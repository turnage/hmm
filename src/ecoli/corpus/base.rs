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
