use crate::method::{GlobalOneDimensionalMethod, Optimizer};
use derive_more::Constructor;
use nalgebra::{DVector, SVector};
use std::ops::RangeInclusive;
use std::rc::Rc;

type Restriction = RangeInclusive<f64>;
type Point = DVector<f64>;

#[derive(Constructor)]
pub struct NestedTasks {
    restrictions: Vec<Restriction>,
    builder: Rc<dyn Fn(Restriction) -> GlobalOneDimensionalMethod>,
}

impl Optimizer for NestedTasks {
    type X = Point;
    type F = f64;
    type Metadata = ();

    fn optimize(
        &self,
        mut f: impl FnMut(Self::X) -> Self::F,
    ) -> (Self::X, Self::F, Self::Metadata) {
        let (x, f) = Optimize {
            builder: self.builder.clone(),
            restrictions: &self.restrictions,
        }
        .run(Box::new(|p| f(p)));
        (x, f, ())
    }
}

struct Optimize<'a> {
    builder: Rc<dyn Fn(Restriction) -> GlobalOneDimensionalMethod>,
    restrictions: &'a [Restriction],
}

impl Optimize<'_> {
    #[inline]
    fn deconstruct(&self) -> (Optimize<'_>, Optimize<'_>) {
        let (r, rs) = self.restrictions.split_first().unwrap();
        let lesser = Optimize {
            builder: self.builder.clone(),
            restrictions: std::slice::from_ref(r),
        };

        let higher = if rs.len() > 0 {
            Optimize {
                builder: self.builder.clone(),
                restrictions: rs,
            }
        } else {
            unreachable!("Deconstruct should not be called with N = 1")
        };
        (lesser, higher)
    }

    fn run<'a>(&self, mut f: Box<dyn FnMut(Point) -> f64 + 'a>) -> (Point, f64) {
        let n = self.restrictions.len();
        if n == 1 {
            let [head] = self.restrictions else {
                unreachable!()
            };
            let optimizer = (self.builder)(head.clone());
            let (x, f, _) = optimizer.optimize(|p| f(Point::from_vec(vec![p.into_scalar()])));
            return (Point::from_element(1, x.into_scalar()), f);
        }

        let concat = |a: SVector<f64, 1>, b: Point| {
            let x = a.into_scalar();
            b.insert_row(0, x)
        };

        let (head, tail) = self.deconstruct();
        let optimizer = (self.builder)(head.restrictions[0].clone());
        let mut ys = Point::default();

        let (x, f, _) = optimizer.optimize(|x| {
            let (x, f) = tail.run(Box::new(|y| f(concat(x, y))));
            ys = x;
            f
        });
        (concat(x, ys), f)
    }
}

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

        Task::new(
            NestedTasks::new(
                vec![range.clone(), range.clone()],
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
    fn test_sixth_dimension<F: Function<5>>(f: F) {
        let optimizer = NestedTasks::new(
            [0; 5].map(|_| -5.0..=5.0).to_vec(),
            Rc::new(|r| ApproxModel::new(r, 4, 2, f64::EPSILON).into()),
        );

        Task::new(optimizer, f)
            .solve_space_check()
            .with_eps_x(1e-2)
            .with_eps_y(1e-2)
            .check();
    }
}
