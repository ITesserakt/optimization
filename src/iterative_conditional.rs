use crate::functions::Point;
use crate::method::{GlobalMultiMethod, Optimizer, Steps};
use crate::restriction::Restriction;

pub struct IterativeConditional<const N: usize> {
    inequalities: Vec<Restriction<N>>,
    equalities: Vec<Restriction<N>>,
    start: Point<N>,
    optimizer: Box<dyn Fn(Point<N>) -> GlobalMultiMethod<N>>,
    eps: f64,
    parameters: Parameters,
    control_hook: Box<dyn Fn(&mut Parameters)>,
}

impl<const N: usize> IterativeConditional<N> {
    pub fn new<
        F: Fn(&mut Parameters) + 'static,
        G: Fn(Point<N>) -> GlobalMultiMethod<N> + 'static,
    >(
        restrictions: Vec<Restriction<N>>,
        start: Point<N>,
        optimizer: G,
        eps: f64,
        parameters: Parameters,
        control_hook: F,
    ) -> Option<Self> {
        let (ineq, eq): (Vec<_>, _) = restrictions.into_iter().partition(|r| r.is_inequality());

        if ineq.len() != parameters.mu.len() || eq.len() != parameters.lambda.len() {
            return None;
        }

        Some(Self {
            inequalities: ineq,
            equalities: eq,
            start,
            optimizer: Box::new(optimizer),
            eps,
            parameters,
            control_hook: Box::new(control_hook),
        })
    }
}

#[derive(Clone)]
pub struct Parameters {
    pub lambda: Vec<f64>,
    pub mu: Vec<f64>,
    pub alpha_h: f64,
    pub alpha_g: f64,
}

impl<const N: usize> IterativeConditional<N> {
    fn tax(&self, x: Point<N>, params: &Parameters) -> f64 {
        let tax_func = |r: &Restriction<N>| match r {
            Restriction::Inequality(f) => f(x).min(0.0).powi(2),
            Restriction::Equality(f) => f(x).powi(2),
        };

        params.alpha_h
            * (params
                .lambda
                .iter()
                .zip(self.equalities.iter().map(tax_func))
                .map(|(x, y)| x * y)
                .sum::<f64>())
            + params.alpha_g
                * (params
                    .mu
                    .iter()
                    .zip(self.inequalities.iter().map(tax_func))
                    .map(|(x, y)| x * y)
                    .sum::<f64>())
    }
}

impl<const N: usize> Optimizer for IterativeConditional<N> {
    type X = Point<N>;
    type Metadata = Steps;

    fn optimize(
        &self,
        mut f: impl FnMut(Self::X) -> Self::F,
    ) -> (Self::X, Self::F, Self::Metadata) {
        let mut r = 0;
        let mut x = self.start;
        let mut x_ = self.start;

        let mut params = self.parameters.clone();

        loop {
            let q = |x: Point<N>| f(x) + self.tax(x, &params);
            let (new_x, _, _) = (self.optimizer)(x).optimize(q);
            x = new_x;
            (self.control_hook)(&mut params);

            if r % 2 == 0 {
                if (x - x_).norm() < self.eps {
                    break;
                }
                x_ = x;
            }

            r += 1;
        }

        (x, f(x), Steps(r))
    }
}

#[cfg(test)]
mod tests {
    use crate::approx_model::ApproxModel;
    use crate::fibonacci::GoldenRatio;
    use crate::functions::{Function, Point};
    use crate::iterative_conditional::{IterativeConditional, Parameters};
    use crate::method::OneDimensionalMethod;
    use crate::restriction::Restriction;
    use crate::task::Task;
    use crate::zeidel::GaussZeidel;
    use std::sync::LazyLock;

    struct Func;

    impl Function<2> for Func {
        const F: f64 = 0.5;

        fn X() -> Vec<Point<2>> {
            vec![[2.5, 1.5].into()]
        }

        fn f(xs: Point<2>) -> f64 {
            let x = xs[0];
            let y = xs[1];
            (x - 3.0).powi(2) + (y - 2.0).powi(2)
        }
    }

    struct OtherFunc;

    impl Function<4> for OtherFunc {
        const F: f64 = -47.0;

        fn X() -> Vec<Point<4>> {
            vec![[0.2, 0.9, 2.1, -0.1].into()]
        }

        fn f(x: Point<4>) -> f64 {
            x[0].powi(2) + x[1].powi(2) + x[2].powi(2) + x[3].powi(2)
                - 5.0 * x[0]
                - 5.0 * x[1]
                - 21.0 * x[2]
                + 7.0 * x[3]
        }
    }

    struct AnotherFunc;

    impl Function<2> for AnotherFunc {
        const F: f64 = 0.0;

        fn X() -> Vec<Point<2>> {
            vec![[1.0, 1.0].into()]
        }

        fn f(x: Point<2>) -> f64 {
            (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
        }
    }

    static LOCAL_OPTIMIZER: LazyLock<OneDimensionalMethod> =
        LazyLock::new(|| ApproxModel::new(-10.0..=10.0, 3, 1, 1e-15).into());

    #[test]
    fn test_iterative_conditional() {
        let restriction = Restriction::equality(|xs| xs[0] + xs[1] - 4.0);
        let method = IterativeConditional::new(
            vec![restriction.clone()],
            [-4.0, 2.3].into(),
            |start| GaussZeidel::new(start, LOCAL_OPTIMIZER.clone(), 1e-15, 1e-15).into(),
            1e-5,
            Parameters {
                lambda: vec![1.0],
                mu: vec![],
                alpha_h: 1.0,
                alpha_g: 1.0,
            },
            |p| p.alpha_h *= 4.0,
        );

        Task::new(method.unwrap(), Func)
            .solve_space_check()
            .satisfy_restrictions(vec![restriction])
            .check();
    }

    #[test]
    fn test_iterative_conditional_slow() {
        let restrictions = vec![
            Restriction::inequality(|x| {
                8.0 - x[0].powi(2) - x[1].powi(2) - x[2].powi(2) - x[3].powi(2) - x[0] + x[1] - x[2]
                    + x[3]
            }),
            Restriction::inequality(|x| {
                10.0 - x[0].powi(2) - 2.0 * x[1].powi(2) - x[2].powi(2) - 2.0 * x[3].powi(2) + x[0]
                    - x[3]
            }),
            Restriction::inequality(|x| {
                5.0 - 2.0 * x[0].powi(2) - x[1].powi(2) - x[2].powi(2) - 2.0 * x[3].powi(2)
                    + x[1]
                    + x[3]
            }),
        ];
        let method = IterativeConditional::new(
            restrictions.clone(),
            [-0.01, 0.99, 1.99, -0.99].into(),
            |start| {
                GaussZeidel::new(
                    start,
                    GoldenRatio::new(-10.0..=10.0, 1e-12).into(),
                    1e-4,
                    1e-4,
                )
                .into()
            },
            1e-5,
            Parameters {
                lambda: vec![],
                mu: vec![1.0, 1.0, 1.0],
                alpha_h: 1.0,
                alpha_g: 1.0,
            },
            |p| {
                p.alpha_h *= 4.0;
                p.alpha_g *= 4.0;
            },
        )
        .unwrap();

        Task::new(method, OtherFunc)
            .solve_space_check()
            .satisfy_restrictions(restrictions)
            .with_eps_x(1e-1)
            .with_eps_y(1e-1)
            .check();
    }

    #[test]
    fn test_iterative_conditional_rosenbrok() {
        let restrictions = vec![
            Restriction::inequality(|x| x[1] - 1.0 - (x[0] - 1.0).powi(3)),
            Restriction::inequality(|x| 2.0 - x[0] - x[1]),
        ];

        let method = IterativeConditional::new(
            restrictions.clone(),
            [-0.5, 12.0].into(),
            |start| {
                GaussZeidel::new(
                    start,
                    GoldenRatio::new(-10.0..=10.0, 1e-12).into(),
                    1e-2,
                    1e-2,
                )
                .into()
            },
            1e-5,
            Parameters {
                lambda: vec![],
                mu: vec![1.0, 1.0],
                alpha_h: 1.0,
                alpha_g: 1.0,
            },
            |p| p.alpha_g *= 4.0,
        )
        .unwrap();

        Task::new(method, AnotherFunc)
            .solve_space_check()
            .satisfy_restrictions(restrictions)
            .check();
    }
}
