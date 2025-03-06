use crate::functions::Point;
use crate::method::{OneDimensionalMethod, Optimizer, Steps};
use crate::zeidel::GaussZeidel;
use derive_more::Constructor;

#[derive(Constructor, Clone)]
pub struct ConjugateDirections<const N: usize> {
    start: Point<N>,
    optimizer: OneDimensionalMethod,
    eps_x: f64,
    eps_y: f64,
}

impl<const N: usize> Optimizer for ConjugateDirections<N> {
    type X = Point<N>;
    type Metadata = Steps;

    fn optimize(
        &self,
        mut f: impl FnMut(Self::X) -> Self::F,
    ) -> (Self::X, Self::F, Self::Metadata) {
        let mut r = 0;
        let mut x = self.start;

        loop {
            let gauss = GaussZeidel::new(x, self.optimizer.clone(), self.eps_x, self.eps_y);
            let x0 = gauss.step(&mut f, 0, &x);

            let gauss = GaussZeidel::new(x0, self.optimizer.clone(), self.eps_x, self.eps_x);
            let xn = gauss.full_step(&mut f, x0);

            let p = xn - x0;
            let step = |lambda: f64| x0 + lambda * p;
            let (lambda, _, _) = self.optimizer.optimize(|lambda| f(step(lambda)));

            let x_ = x;
            x = step(lambda);
            if (f(x) - f(x_)).abs() < self.eps_y || (x - x_).norm() < self.eps_x {
                break;
            }

            r += 1;
        }

        (x, f(x), Steps(r))
    }
}

#[cfg(test)]
mod tests {
    extern crate test;

    use crate::conjugate_directions::ConjugateDirections;
    use crate::fibonacci::GoldenRatio;
    use crate::functions::{Booth, Himmelblau, Sphere};
    use crate::task::Task;
    use std::sync::LazyLock;
    use test::Bencher;

    static OPTIMIZER: LazyLock<ConjugateDirections<2>> = LazyLock::new(|| {
        ConjugateDirections::new(
            [-2.0, -5.0].into(),
            GoldenRatio::new(-10.0..=10.0, 1e-9).into(),
            1e-9,
            1e-10,
        )
    });

    #[bench]
    fn test_conjugate_dirs_booth_golden_ratio(b: &mut Bencher) {
        b.iter(|| {
            Task::new(OPTIMIZER.clone(), Booth)
                .solve_space_check()
                .check();
        })
    }

    #[test]
    fn test_conjugate_dirs_himmelblau_golden_ratio() {
        Task::new(OPTIMIZER.clone(), Himmelblau)
            .solve_space_check()
            .check();
    }

    #[test]
    fn test_conjugate_dirs_sphere_golden_ratio() {
        Task::new(OPTIMIZER.clone(), Sphere)
            .solve_space_check()
            .check();
    }
}
