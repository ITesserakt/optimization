use crate::approx_model::ApproxModel;
use crate::binary::Binary;
use crate::enumerate::MonteCarlo;
use crate::fibonacci::GoldenRatio;
use crate::functions::Point;
use crate::zeidel::GaussZeidel;
use derive_more::From;

pub trait Optimizer {
    type X = f64;
    type F = f64;
    type Metadata = ();

    fn optimize(&self, f: impl FnMut(Self::X) -> Self::F) -> (Self::X, Self::F, Self::Metadata);
}

#[derive(Debug)]
pub struct Steps(pub usize);

#[derive(Clone, From)]
pub enum OneDimensionalMethod {
    GoldenRatio(GoldenRatio),
    Binary(Binary),
    ApproxModel(ApproxModel),
}

#[derive(From)]
pub enum GlobalOneDimensionalMethod {
    MonteCarlo(MonteCarlo<1>),
    ApproxModel(ApproxModel),
}

#[derive(From)]
pub enum GlobalMultiMethod<const N: usize> {
    GaussZeidel(GaussZeidel<N>),
}

impl Optimizer for OneDimensionalMethod {
    type Metadata = Steps;

    fn optimize(&self, f: impl FnMut(Self::X) -> Self::F) -> (Self::X, Self::F, Self::Metadata) {
        match self {
            OneDimensionalMethod::GoldenRatio(x) => x.optimize(f),
            OneDimensionalMethod::Binary(x) => x.optimize(f),
            OneDimensionalMethod::ApproxModel(x) => x.optimize(f),
        }
    }
}

impl Optimizer for GlobalOneDimensionalMethod {
    type X = Point<1>;

    fn optimize(
        &self,
        mut f: impl FnMut(Self::X) -> Self::F,
    ) -> (Self::X, Self::F, Self::Metadata) {
        match self {
            GlobalOneDimensionalMethod::MonteCarlo(x) => {
                let (x, f, _) = x.optimize(f);
                (x, f, ())
            }
            GlobalOneDimensionalMethod::ApproxModel(x) => {
                let (x, f, _) = x.optimize(|it| f([it].into()));
                ([x].into(), f, ())
            }
        }
    }
}

impl<const N: usize> Optimizer for GlobalMultiMethod<N> {
    type X = Point<N>;
    type Metadata = Steps;

    fn optimize(&self, f: impl FnMut(Self::X) -> Self::F) -> (Self::X, Self::F, Self::Metadata) {
        match self {
            GlobalMultiMethod::GaussZeidel(x) => x.optimize(f),
        }
    }
}
