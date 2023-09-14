use crate::method::{Optimizer, Steps};
use derive_more::Constructor;
use std::ops::RangeInclusive;

#[derive(Constructor, Clone)]
pub struct GoldenRatio {
    range: RangeInclusive<f64>,
    eps: f64,
}

impl Optimizer for GoldenRatio {
    type Metadata = Steps;

    fn optimize(&self, f: impl Fn(Self::X) -> Self::F) -> (Self::X, Self::F, Steps) {
        #[derive(Eq, PartialEq)]
        enum Side {
            Left,
            Right,
        }

        let mut r = 0;
        let mut a = *self.range.start();
        let mut b = *self.range.end();
        let mut fx = f(a);
        let mut side = Side::Left;

        while (b - a) > self.eps {
            let x1 = b - (b - a) * Self::TAU;
            let x2 = a + (b - a) * Self::TAU;

            if (side == Side::Left && fx < f(x2)) || (side == Side::Right && f(x1) < fx) {
                b = x2;
                fx = f(x1);
                side = Side::Right
            } else {
                a = x1;
                fx = f(x2);
                side = Side::Left;
            }
            r += 1;
        }

        let x = (a + b) / 2.0;
        (x, f(x), Steps(r))
    }
}

impl GoldenRatio {
    const TAU: f64 = 0.618033988749894;
}

#[cfg(test)]
mod tests {
    use crate::fibonacci::GoldenRatio;
    use crate::functions::{Sphere, Tang};
    use crate::task::Task;

    #[test]
    fn test_golden_ratio_tang() {
        Task::new(GoldenRatio::new(-5.0..=0.0, 1e-6), Tang)
            .solve_check()
            .check();
    }

    #[test]
    fn test_golden_ratio_sphere() {
        Task::new(GoldenRatio::new(-5.0..=0.0, 1e-6), Sphere)
            .solve_check()
            .check();
    }
}
