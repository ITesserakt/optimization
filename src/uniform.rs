use crate::method::{Optimizer, Steps};
use crate::utils::linspace;
use ordered_float::OrderedFloat;
use std::ops::RangeInclusive;

#[derive(Clone)]
pub struct Uniform {
    range: RangeInclusive<<Self as Optimizer>::X>,
    n: usize,
    eps: <Self as Optimizer>::X,
}

impl Optimizer for Uniform {
    type Metadata = Steps;

    fn optimize(&self, f: impl Fn(Self::X) -> Self::F) -> (Self::X, Self::F, Steps) {
        let mut r = 1;
        let mut a = *self.range.start();
        let mut b = *self.range.end();
        let mut x = None;
        let step = 1.0 / self.n as f64;

        while (b - a) > self.eps {
            x = linspace(a, b, self.n).min_by_key(|&x| OrderedFloat(f(x)));

            match x {
                None => unreachable!(),
                Some(x) => {
                    a = x - step * (b - a);
                    b = x + step * (b - a);
                    r += 1;
                }
            }
        }

        (x.unwrap(), f(x.unwrap()), Steps(r))
    }
}

impl Uniform {
    pub fn new<T: Into<<Self as Optimizer>::X> + Clone>(
        range: RangeInclusive<T>,
        n: usize,
        eps: T,
    ) -> Self {
        Self {
            range: RangeInclusive::new(range.start().clone().into(), range.end().clone().into()),
            n,
            eps: eps.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::functions::{Sphere, Tang};
    use crate::task::Task;
    use crate::uniform::Uniform;

    #[test]
    fn test_uniform_tang() {
        Task::new(Uniform::new(-5.0..=0.0, 100, 1e-6), Tang)
            .solve_check()
            .check();
    }

    #[test]
    fn test_uniform_sphere() {
        Task::new(Uniform::new(-5.0..=5.0, 1000, 1e-7), Sphere)
            .solve_check()
            .check();
    }
}
