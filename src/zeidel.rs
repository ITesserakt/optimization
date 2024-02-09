use crate::functions::Point;
use crate::method::{OneDimensionalMethod, Optimizer, Steps};
use nalgebra::SVector;

#[derive(Clone)]
pub struct GaussZeidel<const N: usize> {
    pub optimizer: OneDimensionalMethod,
    start: Point<N>,
    eps_x: f64,
    eps_y: f64,
}

impl<const N: usize> Optimizer for GaussZeidel<N> {
    type X = Point<N>;
    type Metadata = Steps;

    fn optimize(
        &self,
        mut f: impl FnMut(Self::X) -> Self::F,
    ) -> (Self::X, Self::F, Self::Metadata) {
        let mut x = self.start;
        let mut x_ = self.start.clone().map(|x| x + 2.0 * self.eps_x);
        let mut r = 1;

        while (x - x_).norm() > self.eps_x && (f(x) - f(x_)).abs() > self.eps_y {
            x_ = x;
            x = self.full_step(&mut f, x);
            r += 1;
        }

        (x, f(x), Steps(r))
    }
}

impl<const N: usize> GaussZeidel<N> {
    pub fn new(start: Point<N>, inner: OneDimensionalMethod, eps_x: f64, eps_y: f64) -> Self {
        Self {
            optimizer: inner,
            start,
            eps_x,
            eps_y,
        }
    }

    pub fn step(
        &self,
        f: &mut impl FnMut(Point<N>) -> f64,
        direction: usize,
        x: &Point<N>,
    ) -> Point<N> {
        let mut l: Point<N> = SVector::from_element(0.0);
        l[direction] = 1.0;
        let next_x = |lambda: f64| x + lambda * l;
        let (lambda, _, _) = self.optimizer.optimize(|lambda| f(next_x(lambda)));

        next_x(lambda)
    }

    pub fn full_step(&self, f: &mut impl FnMut(Point<N>) -> f64, mut x: Point<N>) -> Point<N> {
        for i in 0..N {
            x = self.step(f, i, &x);
        }
        x
    }
}

#[cfg(test)]
mod tests {
    extern crate test;

    use crate::binary::Binary;
    use crate::fibonacci::GoldenRatio;
    use crate::functions::{Booth, Function, Himmelblau, Rosenbrok, Sphere};
    use crate::method::OneDimensionalMethod;
    use crate::task::{Check, Task};
    use crate::zeidel::GaussZeidel;
    use lazy_static::lazy_static;
    use test::Bencher;

    #[test]
    fn test_gauss_zeidel_rosenbrok() {
        Task::new(
            GaussZeidel::new(
                [2.0, 1.5].into(),
                GoldenRatio::new(-10.0..=10.0, 1e-12).into(),
                1e-12,
                1e-12,
            ),
            Rosenbrok,
        )
        .solve_space_check()
        .with_eps_x(1e-4)
        .check();
    }

    lazy_static! {
        static ref METHODS: [OneDimensionalMethod; 2] = [
            GoldenRatio::new(-10.0..=10.0, 1e-6).into(),
            Binary::new(-10.0..=10.0, 1e-6, 1e-7).into(),
        ];
    }

    #[must_use]
    fn helper<F: Function<2>>(f: F, method: &OneDimensionalMethod) -> Check<2, F> {
        Task::new(
            GaussZeidel::new([-4.0, -4.0].into(), method.clone(), 1e-12, 1e-12),
            f,
        )
        .solve_space_check()
    }

    #[bench]
    fn test_gauss_zeidel_booth_binary(b: &mut Bencher) {
        b.iter(|| helper(Booth, &METHODS[1]).check())
    }

    #[bench]
    fn test_gauss_zeidel_booth_golden_ratio(b: &mut Bencher) {
        b.iter(|| helper(Booth, &METHODS[0]).check())
    }

    #[bench]
    fn test_gauss_zeidel_himmelblau_golden_ratio(b: &mut Bencher) {
        b.iter(|| helper(Himmelblau, &METHODS[0]).check())
    }

    #[test]
    fn test_gauss_zeidel_sphere_golden_ratio() {
        helper(Sphere, &METHODS[0]).check()
    }
}
