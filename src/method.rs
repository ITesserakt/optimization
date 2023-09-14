use crate::approx_model::ApproxModel;
use crate::backward::Backward;
use crate::binary::Binary;
use crate::conjugate_directions::ConjugateDirections;
use crate::enumerate::{Enumerate, MonteCarlo};
use crate::fibonacci::GoldenRatio;
use crate::functions::Point;
use crate::uniform::Uniform;
use crate::zeidel::GaussZeidel;
use derive_more::From;
use std::fmt::{Display, Formatter};

pub trait Optimizer {
    type X = f64;
    type F = f64;
    type Metadata = ();

    fn optimize(&self, f: impl Fn(Self::X) -> Self::F) -> (Self::X, Self::F, Self::Metadata);
}

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub struct Steps(pub usize);

impl Display for Steps {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Clone, From)]
pub enum OneDimensionalMethod {
    Uniform(Uniform),
    Binary(Binary),
    Enumerate(Enumerate),
    MonteCarlo(MonteCarlo<1>),
    ApproxModel(ApproxModel),
    Backward(Backward<1>),
    GoldenRatio(GoldenRatio),
}

#[derive(Clone, From)]
pub enum GlobalOneDimensionalMethod {
    ApproxModel(ApproxModel),
    MonteCarlo(MonteCarlo<1>),
    Enumerate(Enumerate),
}

#[derive(Clone)]
pub enum GlobalMultiMethod<const N: usize> {
    GaussZeidel(GaussZeidel<N>),
    ConjugateDirections(ConjugateDirections<N>),
}

impl Optimizer for GlobalOneDimensionalMethod {
    type X = Point<1>;

    fn optimize(&self, f: impl Fn(Point<1>) -> Self::F) -> (Point<1>, Self::F, Self::Metadata) {
        match self {
            GlobalOneDimensionalMethod::ApproxModel(x) => {
                let (x, f, _) = x.optimize(|x| f([x].into()));
                ([x].into(), f, ())
            }
            GlobalOneDimensionalMethod::MonteCarlo(x) => x.optimize(f),
            GlobalOneDimensionalMethod::Enumerate(x) => {
                let (x, f, _) = x.optimize(|x| f([x].into()));
                ([x].into(), f, ())
            }
        }
    }
}

impl Optimizer for OneDimensionalMethod {
    type Metadata = ();

    fn optimize(&self, f: impl Fn(Self::X) -> Self::F) -> (Self::X, Self::F, Self::Metadata) {
        match self {
            OneDimensionalMethod::Uniform(x) => {
                let (x, f, _) = x.optimize(f);
                (x, f, ())
            }
            OneDimensionalMethod::Binary(x) => {
                let (x, f, _) = x.optimize(f);
                (x, f, ())
            }
            OneDimensionalMethod::Enumerate(x) => {
                let (x, f, _) = x.optimize(f);
                (x, f, ())
            }
            OneDimensionalMethod::MonteCarlo(x) => {
                let (x, f, _) = x.optimize(|x| f(x[0]));
                (x[0], f, ())
            }
            OneDimensionalMethod::ApproxModel(x) => {
                let (x, f, _) = x.optimize(f);
                (x, f, ())
            }
            OneDimensionalMethod::Backward(x) => {
                let (x, f, _) = x.optimize(|x| f(x[0]));
                (x[0], f, ())
            }
            OneDimensionalMethod::GoldenRatio(x) => {
                let (x, f, _) = x.optimize(f);
                (x, f, ())
            }
        }
    }
}

impl<const N: usize> Optimizer for GlobalMultiMethod<N> {
    type X = Point<N>;
    type Metadata = Steps;

    fn optimize(&self, f: impl Fn(Point<N>) -> Self::F) -> (Point<N>, Self::F, Steps) {
        match self {
            GlobalMultiMethod::GaussZeidel(x) => x.optimize(f),
            GlobalMultiMethod::ConjugateDirections(x) => x.optimize(f),
        }
    }
}

impl<const N: usize> From<GaussZeidel<N>> for GlobalMultiMethod<N> {
    fn from(value: GaussZeidel<N>) -> Self {
        GlobalMultiMethod::GaussZeidel(value)
    }
}

impl<const N: usize> From<ConjugateDirections<N>> for GlobalMultiMethod<N> {
    fn from(value: ConjugateDirections<N>) -> Self {
        GlobalMultiMethod::ConjugateDirections(value)
    }
}
