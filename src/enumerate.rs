use crate::functions::Point;
use crate::method::Optimizer;
use crate::utils::linspace;
use nalgebra::SVector;
use ordered_float::OrderedFloat;
use std::ops::RangeInclusive;

#[derive(Clone)]
pub struct Enumerate {
    range: RangeInclusive<<Self as Optimizer>::X>,
    n: usize,
}

impl Optimizer for Enumerate {
    type Metadata = ();

    fn optimize(
        &self,
        mut f: impl FnMut(Self::X) -> Self::F,
    ) -> (Self::X, Self::F, Self::Metadata) {
        let a = *self.range.start();
        let b = *self.range.end();
        let x = linspace(a, b, self.n)
            .min_by_key(|&x| OrderedFloat(f(x)))
            .unwrap();

        (x, f(x), ())
    }
}

impl Enumerate {
    pub fn new<T: Into<<Self as Optimizer>::X> + Clone>(
        range: RangeInclusive<T>,
        n: usize,
    ) -> Self {
        Self {
            range: RangeInclusive::new(range.start().clone().into(), range.end().clone().into()),
            n,
        }
    }
}

#[derive(Clone)]
pub struct MonteCarlo<const N: usize> {
    distributions: SVector<RangeInclusive<f64>, N>,
    n: usize,
}

impl<const N: usize> Optimizer for MonteCarlo<N> {
    type X = Point<N>;
    type Metadata = ();

    fn optimize(
        &self,
        mut f: impl FnMut(Self::X) -> Self::F,
    ) -> (Self::X, Self::F, Self::Metadata) {
        let map_to = |x, y: RangeInclusive<f64>| return (y.end() - y.start()) * x + y.start();

        let x = (0..self.n)
            .map(|_| Point::<N>::new_random().zip_map(&self.distributions, map_to))
            .min_by_key(|x| OrderedFloat(f(*x)));

        (x.unwrap(), f(x.unwrap()), ())
    }
}

impl<const N: usize> MonteCarlo<N> {
    pub fn new(distributions: [RangeInclusive<f64>; N], n: usize) -> MonteCarlo<N> {
        Self {
            distributions: distributions.into(),
            n,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::enumerate::{Enumerate, MonteCarlo};
    use crate::functions::{Booth, Tang};
    use crate::task::Task;

    #[test]
    fn test_enumeration() {
        Task::new(Enumerate::new(-5.0..=5.0, 1000000), Tang)
            .solve_check()
            .check();
    }

    #[test]
    fn test_monte_karlo_tang() {
        Task::new(
            MonteCarlo::new([-2.904..=-2.903, -2.904..=-2.903], 100000),
            Tang,
        )
        .solve_space_check()
        .check();
    }

    #[test]
    fn test_monte_karlo_booth() {
        Task::new(MonteCarlo::new([0.0..=3.5, 0.0..=3.5], 100000), Booth)
            .solve_space_check()
            .with_eps_x(1e-2)
            .with_eps_y(1e-3)
            .check();
    }
}
