use crate::functions::{Function, Point};
use crate::method::Optimizer;

use crate::restriction::Restriction;
#[cfg(test)]
use approx::assert_relative_eq;
#[cfg(test)]
use std::fmt::Debug;
#[cfg(test)]
use std::marker::PhantomData;

pub struct Task<O, F> {
    optimizer: O,
    _unused: F,
}

impl<O, F> Task<O, F> {
    pub fn new(optimizer: O, function: F) -> Self {
        Self {
            optimizer,
            _unused: function,
        }
    }
}

#[cfg(test)]
pub struct Check<const N: usize, F: Function<N>> {
    eps_x: f64,
    eps_y: f64,
    x: Point<N>,
    f: f64,
    restrictions: Vec<Restriction<N>>,
    _phantom: PhantomData<F>,
}

#[cfg(test)]
impl<const N: usize, F: Function<N>> Check<N, F> {
    pub fn with_eps_x(self, eps_x: f64) -> Self {
        Self { eps_x, ..self }
    }

    pub fn with_eps_y(self, eps_y: f64) -> Self {
        Self { eps_y, ..self }
    }

    pub fn check(self) {
        let any_x_eq = F::X().into_iter().any(|point| {
            self.x
                .into_iter()
                .zip(point.into_iter())
                .all(|(actual, expected)| {
                    approx::relative_eq!(actual, expected, epsilon = self.eps_x)
                })
        });

        assert!(
            any_x_eq,
            "All points did not converged: actual={}, expected={}",
            self.x,
            F::X()
                .into_iter()
                .map(|x| x.to_string())
                .intersperse(" or ".to_string())
                .collect::<String>()
        );

        assert!(
            self.restrictions
                .into_iter()
                .all(|r| r.satisfies(self.x, self.eps_y)),
            "Point does not satisfy given restrictions"
        );

        assert_relative_eq!(self.f, F::F, epsilon = self.eps_y);
    }

    pub fn satisfy_restrictions(self, restrictions: Vec<Restriction<N>>) -> Self {
        Self {
            restrictions,
            ..self
        }
    }
}

impl<O, F> Task<O, F>
where
    O: Optimizer<X = f64, F = f64>,
    F: Function<1>,
{
    pub fn solve(self) -> (O::X, O::F, O::Metadata) {
        let method = self.optimizer;
        method.optimize(|x| F::f([x].into()))
    }

    #[cfg(test)]
    #[must_use]
    pub fn solve_check(self) -> Check<1, F>
    where
        O::Metadata: Debug,
    {
        let (x, f, m) = self.solve();
        dbg!(m);

        Check {
            eps_x: 1e-5,
            eps_y: 1e-5,
            x: [x].into(),
            f,
            restrictions: vec![],
            _phantom: Default::default(),
        }
    }
}

impl<O, F, const N: usize> Task<O, F>
where
    O: Optimizer<X = Point<N>, F = f64>,
    F: Function<N>,
{
    pub fn solve_space(self) -> (Point<N>, O::F, O::Metadata) {
        self.optimizer.optimize(F::f)
    }

    #[cfg(test)]
    #[must_use]
    pub fn solve_space_check(self) -> Check<N, F>
    where
        O::Metadata: Debug,
    {
        let (xs, f, m) = self.solve_space();
        dbg!(m);

        Check {
            eps_x: 1e-5,
            eps_y: 1e-5,
            x: xs,
            f,
            restrictions: vec![],
            _phantom: Default::default(),
        }
    }
}
