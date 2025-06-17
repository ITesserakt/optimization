use crate::method::{Optimizer, Steps};
use ordered_float::OrderedFloat;
use polynomial::Polynomial;
use rand::{Rng, rng};
use roots::find_root_inverse_quadratic;
use scilib::math::polynomial::Poly;
use std::cell::RefCell;
use std::cmp::min_by_key;
use std::ops::RangeInclusive;

#[derive(Clone)]
pub struct ApproxModel {
    range: RangeInclusive<f64>,
    n: usize,
    m: usize,
    eps: f64,
}

impl ApproxModel {
    pub fn new(range: RangeInclusive<f64>, n: usize, m: usize, eps: f64) -> Self {
        Self { range, n, m, eps }
    }

    fn build_polynomial(&self, f: &impl Fn(f64) -> f64) -> (Polynomial<f64>, Steps) {
        let a = *self.range.start();
        let b = *self.range.end();
        let mut polynomial: Option<Polynomial<f64>> = None;
        let mut random = rng();
        let mut r = 0;

        for _ in 0..self.m {
            let test = random.random_range(self.range.clone());
            let y = f(test);

            if let Some(ref poly) = polynomial
                && (poly.eval(test) - y).abs() < self.eps
            {
                continue;
            }

            polynomial = Polynomial::chebyshev(f, self.n + r, a, b);
            r += 1;
        }

        (polynomial.unwrap(), Steps(r))
    }

    fn find_minimum(&self, poly: Polynomial<f64>, f: &impl Fn(f64) -> f64) -> f64 {
        let a = *self.range.start();
        let b = *self.range.end();
        let mut poly = Poly::from(&poly.data().iter().cloned().enumerate().collect::<Vec<_>>());
        poly.derive(1);
        let poly = |x| poly.compute(x);
        let mut eps = 1e-5;
        let root = find_root_inverse_quadratic(a, b, poly, &mut eps);

        root.unwrap_or_else(|_| min_by_key(f(a), f(b), |x| OrderedFloat(*x)))
    }

    fn fn_mut_to_fn<I, O, F: FnMut(I) -> O>(f: F) -> impl Fn(I) -> O {
        let cell = RefCell::new(f);
        move |x| (cell.borrow_mut())(x)
    }
}

impl Optimizer for ApproxModel {
    type Metadata = Steps;

    fn optimize(&self, f: impl FnMut(Self::X) -> Self::F) -> (Self::X, Self::F, Self::Metadata) {
        let f = ApproxModel::fn_mut_to_fn(f);
        let (polynomial, tries) = self.build_polynomial(&f);
        let minimum = self.find_minimum(polynomial, &f);

        (minimum, f(minimum), tries)
    }
}

#[cfg(test)]
mod tests {
    use crate::approx_model::ApproxModel;
    use crate::functions::{Sphere, Tang};
    use crate::task::Task;

    #[test]
    fn test_approx_model_tang() {
        Task::new(ApproxModel::new(-5.0..=5.0, 5, 10, 1e-7), Tang)
            .solve_check()
            .check();
    }

    #[test]
    fn test_approx_model_sphere() {
        Task::new(ApproxModel::new(-5.0..=5.0, 5, 10, 1e-7), Sphere)
            .solve_check()
            .check();
    }
}
