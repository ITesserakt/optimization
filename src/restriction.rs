use crate::functions::Point;
use std::rc::Rc;

pub type Func<const N: usize> = Rc<dyn Fn(Point<N>) -> f64>;

#[derive(Clone)]
pub enum Restriction<const N: usize> {
    /// g(X) >= 0
    Inequality(Func<N>),
    /// h(X) = 0
    Equality(Func<N>),
}

impl Restriction<1> {
    pub fn interval(from: f64, to: f64) -> [Restriction<1>; 2] {
        [
            Restriction::Inequality(Rc::new(move |x| x[0] - from)),
            Restriction::Inequality(Rc::new(move |x| to - x[0])),
        ]
    }
}

impl<const N: usize> Restriction<N> {
    pub fn satisfies(&self, x: Point<N>, eps: f64) -> bool {
        match self {
            Restriction::Inequality(f) => dbg!(f(x) + eps) >= 0.0,
            Restriction::Equality(f) => dbg!(f(x).abs()) < eps,
        }
    }

    pub fn apply(&self, x: Point<N>) -> f64 {
        match self {
            Restriction::Inequality(f) => f(x),
            Restriction::Equality(f) => f(x),
        }
    }

    pub fn is_inequality(&self) -> bool {
        match self {
            Restriction::Inequality(_) => true,
            Restriction::Equality(_) => false,
        }
    }

    pub fn equality<F: Fn(Point<N>) -> f64 + 'static>(f: F) -> Self {
        Restriction::Equality(Rc::new(f))
    }

    pub fn inequality<F: Fn(Point<N>) -> f64 + 'static>(f: F) -> Self {
        Restriction::Inequality(Rc::new(f))
    }
}

#[cfg(test)]
mod tests {
    use crate::restriction::Restriction;

    #[test]
    fn test_intervals() {
        let restrictions = Restriction::interval(0.0, 10.0);

        assert!(restrictions
            .into_iter()
            .all(|r| r.satisfies([4.0].into(), f64::EPSILON)))
    }
}
