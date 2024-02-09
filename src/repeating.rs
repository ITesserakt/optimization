use crate::functions::Point;
use crate::method::{Optimizer, Steps};
use derive_more::Constructor;

#[derive(Constructor, Clone)]
pub struct RepeatingStochastic<const N: usize> {
    start: [Point<N>; 3],
    lambda: f64,
    smoothing: [f64; 2],
    eps_x: f64,
    eps_y: f64,
    m: usize,
}

impl<const N: usize> Optimizer for RepeatingStochastic<N> {
    type X = Point<N>;
    type Metadata = Steps;

    fn optimize(
        &self,
        mut f: impl FnMut(Self::X) -> Self::F,
    ) -> (Self::X, Self::F, Self::Metadata) {
        let mut r = 2;
        let beta = self.smoothing[0];
        let gamma = self.smoothing[1];
        let mut x = self.start[2];
        let mut x1 = self.start[1];
        let mut x0 = self.start[0];
        let mut s1 = x1 - x0;
        let mut s2 = x - x1;
        let mut lambda = self.lambda;

        'outer: loop {
            let mut next_x = x;
            let mut tries = 0;
            'inner: loop {
                let p: Point<N> = (Point::<N>::new_random() * 2.0) - Point::from([1.0; N]);
                let s = gamma * s1 + (1.0 - gamma) * s2;
                let delta = beta * s.normalize() + (1.0 - beta) * p.normalize();

                let x_ = x;
                next_x = x + lambda * delta;

                if f(next_x) < f(x_) {
                    if (f(next_x) - f(x_)).abs() < self.eps_y || (next_x - x_).norm() < self.eps_x {
                        break 'outer;
                    } else {
                        lambda *= 2.0;
                        break 'inner;
                    }
                } else if tries < self.m {
                    tries += 1;
                    continue;
                } else {
                    tries = 0;
                    lambda *= 0.5;
                }
            }
            x0 = x1;
            x1 = x;
            x = next_x;
            s1 = x1 - x0;
            s2 = x - x1;
            r += 1;
        }

        (x, f(x), Steps(r))
    }
}

#[cfg(test)]
mod tests {
    use crate::functions::{Booth, Function, Himmelblau, Sphere};
    use crate::repeating::RepeatingStochastic;
    use crate::task::Task;
    use lazy_static::lazy_static;
    use test_case::test_case;

    lazy_static! {
        static ref OPTIMIZER: RepeatingStochastic<2> = RepeatingStochastic::new(
            [[10.0, 5.0].into(), [2.0, 3.0].into(), [4.0, 9.0].into()],
            0.5,
            [0.5, 0.5],
            1e-12,
            1e-12,
            10,
        );
    }

    #[test_case(Booth; "booth")]
    #[test_case(Sphere; "sphere")]
    #[test_case(Himmelblau; "himmelblau")]
    fn test<F: Function<2>>(f: F) {
        Task::new(OPTIMIZER.clone(), f)
            .solve_space_check()
            .with_eps_x(1e-4)
            .check();
    }
}
