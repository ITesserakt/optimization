use crate::functions::Point;
use crate::method::{GlobalOneDimensionalMethod, Optimizer};
use derive_more::Constructor;
use nalgebra::Const;
use std::cell::Cell;
use std::ops::RangeInclusive;
use std::rc::Rc;

type Restriction = RangeInclusive<f64>;

#[derive(Constructor)]
pub struct NestedTasks<const N: usize> {
    restrictions: [Restriction; N],
    builder: Rc<dyn Fn(Restriction) -> GlobalOneDimensionalMethod>,
}

impl<const N: usize> Optimizer for NestedTasks<N>
where
    Optimize<N>: Helper<N>,
{
    type X = Point<N>;

    fn optimize(&self, f: impl FnMut(Self::X) -> Self::F) -> (Point<N>, Self::F, Self::Metadata) {
        let (x, f) = Optimize {
            builder: self.builder.clone(),
            restrictions: self.restrictions.clone(),
        }
        .run(f);
        (x, f, ())
    }
}

trait Helper<const N: usize> {
    fn run(&self, f: impl FnMut(Point<N>) -> f64) -> (Point<N>, f64);
}

struct Optimize<const N: usize>
where
    Const<{ N }>:,
{
    builder: Rc<dyn Fn(Restriction) -> GlobalOneDimensionalMethod>,
    restrictions: [Restriction; N],
}

impl Helper<1> for Optimize<1> {
    fn run(&self, f: impl FnMut(Point<1>) -> f64) -> (Point<1>, f64) {
        let head = self.restrictions.clone();
        let optimizer = (self.builder)(head[0].clone());
        let (x, f, _) = optimizer.optimize(f);
        (x, f)
    }
}

impl<const N: usize> Optimize<N> {
    fn deconstruct(&self) -> (Optimize<1>, Optimize<{ N - 1 }>)
    where
        [(); N - 1]:,
    {
        let (r, rs) = self.restrictions.split_first().unwrap();
        let lesser = Optimize {
            builder: self.builder.clone(),
            restrictions: [r.clone()],
        };

        let higher = if N > 1 {
            let rs = rs.first_chunk().unwrap().clone();

            Optimize {
                builder: self.builder.clone(),
                restrictions: rs,
            }
        } else {
            panic!("Deconstruct should not be called with N = 1")
        };
        (lesser, higher)
    }
}

macro_rules! impl_helper {
    ($n:literal) => {
        impl Helper<$n> for Optimize<$n>
        {
            fn run(&self, mut f: impl FnMut(Point<$n>) -> f64) -> (Point<$n>, f64) {
                let (head, tail) = self.deconstruct();
                let optimizer = (self.builder)(head.restrictions[0].clone());
                let mut y = Default::default();
                let mut concat_buffer = [0.0; $n];
                let mut concat = move |x: Point<1>, y: Point<{ $n - 1 }>| {
                    concat_buffer[..1].copy_from_slice(x.as_slice());
                    concat_buffer[1..].copy_from_slice(y.as_slice());
                    concat_buffer
                };

                let (x, f, _) = optimizer.optimize(|x| {
                    let (x, f) = tail.run(|y| f(concat(x, y).into()));
                    y = x;
                    f
                });
                (concat(x, y).into(), f)
            }
        }

    };
    ($n:literal, $($ns:literal),+) => {
        impl_helper!($n);
        impl_helper!($($ns),+);
    };
}

impl_helper!(2, 3, 4);

#[cfg(test)]
mod tests {
    use crate::approx_model::ApproxModel;
    use crate::compound::NestedTasks;
    use crate::enumerate::MonteCarlo;
    use crate::functions::{Booth, Function, Himmelblau, Rastrigin, Rosenbrok, Sphere, Tang};
    use crate::task::Task;
    use std::rc::Rc;
    use test_case::test_case;

    #[test_case(Booth)]
    #[test_case(Tang)]
    #[test_case(Rastrigin)]
    #[test_case(Rosenbrok)]
    #[test_case(Sphere)]
    #[test_case(Himmelblau)]
    fn test_second_dimension<F: Function<2>>(f: F) {
        let k = 1000;
        let range = -10.0..=10.0;
        let eps = (range.end() - range.start()) / k as f64 * 2.0;

        Task::new(
            NestedTasks::new(
                [range.clone(), range.clone()],
                Rc::new(move |r| MonteCarlo::new([r], k).into()),
            ),
            f,
        )
        .solve_space_check()
        .with_eps_x(1e-1)
        .with_eps_y(1e-1)
        .check();
    }

    #[test_case(Tang)]
    #[test_case(Rastrigin)]
    #[test_case(Sphere)]
    fn test_third_dimension<F: Function<3>>(f: F) {
        let optimizer = NestedTasks::new(
            [-5.0..=5.0, -5.0..=5.0, -5.0..=5.0],
            Rc::new(|r| ApproxModel::new(r, 10, 2, f64::EPSILON).into()),
        );

        Task::new(optimizer, f)
            .solve_space_check()
            .with_eps_x(1e-2)
            .with_eps_y(1e-2)
            .check();
    }
}
