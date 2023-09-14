use crate::method::{Optimizer, Steps};
use std::ops::RangeInclusive;

#[derive(Clone)]
pub struct Binary {
    range: RangeInclusive<<Self as Optimizer>::X>,
    eps: <Self as Optimizer>::X,
    delta: <Self as Optimizer>::X,
}

impl Optimizer for Binary {
    type Metadata = Steps;

    fn optimize(&self, f: impl Fn(Self::X) -> Self::F) -> (Self::X, Self::F, Steps) {
        let mut a = *self.range.start();
        let mut b = *self.range.end();
        let mut r = 1;
        let mut x = a;

        while (b - a) > self.eps {
            x = (b + a) / 2.0;
            let x_1 = x - self.delta / 2.0;
            let x_2 = x + self.delta / 2.0;
            let f_1 = f(x_1);
            let f_2 = f(x_2);

            if f_1 < f_2 {
                b = x_2;
            } else {
                a = x_1;
            }
            r += 1;
        }

        (x, f(x), Steps(r))
    }
}

impl Binary {
    pub fn new<T: Into<<Self as Optimizer>::X> + Clone>(
        range: RangeInclusive<T>,
        eps: T,
        delta: T,
    ) -> Self {
        Self {
            range: RangeInclusive::new(range.start().clone().into(), range.end().clone().into()),
            eps: eps.into(),
            delta: delta.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::binary::Binary;
    use crate::functions::{Sphere, Tang};
    use crate::task::Task;

    #[test]
    fn test_binary_tang() {
        Task::new(Binary::new(-5.0..=0.0, 1e-6, 1e-7), Tang)
            .solve_check()
            .check();
    }

    #[test]
    fn test_binary_sphere() {
        Task::new(Binary::new(-5.0..=0.0, 1e-6, 1e-7), Sphere)
            .solve_check()
            .check();
    }
}
