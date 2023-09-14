use crate::functions::Point;
use crate::method::{Optimizer, Steps};
use derive_more::Constructor;
use ordered_float::OrderedFloat;
use std::mem::swap;

#[derive(Clone)]
pub struct Backward<const N: usize> {
    start: Point<N>,
    step: f64,
    k: usize,
    alpha: f64,
    eps: f64,
}

impl<const N: usize> Backward<N> {
    pub fn new(start: Point<N>, step: f64, alpha: f64, eps: f64) -> Self {
        Self {
            start,
            step,
            k: 3 * N,
            alpha,
            eps,
        }
    }
}

impl<const N: usize> Optimizer for Backward<N> {
    type X = Point<N>;
    type Metadata = Steps;

    fn optimize(&self, f: impl Fn(Point<N>) -> Self::F) -> (Point<N>, Self::F, Steps) {
        let mut r = 0;
        let mut step = self.step;
        let mut x = self.start;
        let mut x_ = x + x;

        while (f(x) - f(x_)).abs() > self.eps {
            let mut k = 1;

            while f(x) < f(x_) && k < self.k {
                let phi: Point<N> = Point::new_random() * 2.0 - Point::from_element(1.0);
                x_ = x + step * phi.normalize();
                k += 1;
            }

            if k == self.k {
                step *= self.alpha;
            }

            swap(&mut x, &mut x_);
            r += 1;
        }

        (x, f(x), Steps(r))
    }
}

#[derive(Constructor)]
pub struct BestChoice<const N: usize> {
    start: Point<N>,
    step: f64,
    k: usize,
    alpha: f64,
    eps: f64,
}

impl<const N: usize> Optimizer for BestChoice<N> {
    type X = Point<N>;
    type Metadata = Steps;

    fn optimize(&self, f: impl Fn(Point<N>) -> Self::F) -> (Point<N>, Self::F, Steps) {
        let mut r = 0;
        let mut step = self.step;
        let mut x = self.start;
        let mut x_ = x + x;

        while (f(x) - f(x_)).abs() > self.eps {
            while f(x_) > f(x) {
                let ps = (0..self.k)
                    .map(|_| Point::new_random() * 2.0 - Point::from_element(1.0))
                    .min_by_key(|p| OrderedFloat(f(*p)));
                x_ = x + step * ps.unwrap();
            }

            step *= self.alpha;
            swap(&mut x, &mut x_);
            r += 1;
        }

        (x, f(x), Steps(r))
    }
}

#[cfg(test)]
mod tests {
    use crate::backward::{Backward, BestChoice};
    use crate::functions::{Booth, Rosenbrok, Sphere};
    use crate::task::Task;

    #[test]
    fn test_backward_sphere() {
        Task::new(Backward::new([10.0, 5.0].into(), 10.0, 0.5, 1e-15), Sphere)
            .solve_space_check()
            .check();
    }

    #[test]
    fn test_backward_booth() {
        Task::new(Backward::new([-10.0, -5.0].into(), 10.0, 0.6, 1e-15), Booth)
            .solve_space_check()
            .check();
    }

    #[test]
    fn test_best_choice_sphere() {
        Task::new(
            BestChoice::new([-1.0, 0.1].into(), 8., 6, 0.7, 1e-9),
            Sphere,
        )
        .solve_space_check()
        .with_eps_x(1e-3)
        .check();
    }

    #[test]
    fn test_best_choice_booth() {
        Task::new(BestChoice::new([1.5, 3.2].into(), 8., 6, 0.7, 1e-9), Booth)
            .solve_space_check()
            .with_eps_x(1e-3)
            .check();
    }

    #[test]
    fn test_best_choice_rosenbrok() {
        Task::new(
            BestChoice::new([1.5, 3.2].into(), 10., 6, 0.8, 1e-9),
            Rosenbrok,
        )
        .solve_space_check()
        .with_eps_x(1e-4)
        .check();
    }
}
