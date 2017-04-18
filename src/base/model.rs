use std::marker;

pub trait Starter<S> {
    /// startp returns the probability of state s beginning a state sequence.
    fn startp(&self, s: S) -> Result<f64, String>;
}

pub trait Emitter<S, O> {
    /// emitp returns the probability of a given state emitting the given observation.
    /// If the hidden state or observation is not recognized, the emitter should return an error
    /// explaining why it is incompatible with the emitter.
    fn emitp(&self, state: S, observation: O) -> Result<f64, String>;
}

pub trait Transor<S> {
    /// transp returns the probability of state a transitioning to state b. If either hidden state
    /// is invalid the transor should return an error explaining why it is invalid.
    fn transp(&self, a: S, b: S) -> Result<f64, String>;
    fn states(&self) -> Vec<S>;
}

pub struct Model<S: Copy, O: Copy, St: Starter<S>, E: Emitter<S, O>, T: Transor<S>> {
    pub start: St,
    pub emitter: E,
    pub trans: T,
    _state_marker: marker::PhantomData<S>,
    _observation_marker: marker::PhantomData<O>,
}

impl<S: Copy, O: Copy, St: Starter<S>, E: Emitter<S, O>, T: Transor<S>> Model<S, O, St, E, T> {
    pub fn from(start: St, emitter: E, trans: T) -> Self {
        Model {
            start: start,
            emitter: emitter,
            trans: trans,
            _state_marker: marker::PhantomData,
            _observation_marker: marker::PhantomData,
        }
    }
}
