use nalgebra::SVector;
use num::traits::FloatConst;
use std::convert::Into;

pub trait Function<const N: usize> {
    const F: f64;

    #[allow(non_snake_case)]
    fn X() -> Vec<Point<N>>;
    fn f(x: Point<N>) -> f64;
}

pub type Point<const N: usize> = SVector<f64, N>;

pub struct Tang<const N: usize>;

impl<const N: usize> Function<N> for Tang<N> {
    const F: f64 = -39.16617 * (N as f64);

    #[allow(non_snake_case)]
    fn X() -> Vec<Point<N>> {
        vec![SVector::from_element(-2.903534)]
    }

    #[inline]
    fn f(x: Point<N>) -> f64 {
        x.map(|x| x.powi(4) - 16.0 * x.powi(2) + 5.0 * x).sum() / 2.0
    }
}

pub struct Rastrigin<const N: usize>;

impl<const N: usize> Function<N> for Rastrigin<N> {
    const F: f64 = 0.0;

    fn X() -> Vec<Point<N>> {
        vec![SVector::from_element(0.0)]
    }

    #[inline]
    fn f(x: Point<N>) -> f64 {
        10.0 * N as f64 + x.map(|x| x * x - 10.0 * (f64::TAU() * x).cos()).sum()
    }
}

pub struct Sphere<const N: usize>;

impl<const N: usize> Function<N> for Sphere<N> {
    const F: f64 = 0.0;

    fn X() -> Vec<Point<N>> {
        vec![SVector::from_element(0.0)]
    }

    #[inline]
    fn f(x: Point<N>) -> f64 {
        x.map(|x| x * x).sum()
    }
}

pub struct Rosenbrok;

impl Function<2> for Rosenbrok {
    const F: f64 = 0.0;

    fn X() -> Vec<Point<2>> {
        vec![[1.0, 1.0].into()]
    }

    #[inline]
    fn f(xs: Point<2>) -> f64 {
        let x = xs[0];
        let y = xs[1];

        100.0 * (y - x.powi(2)).powi(2) + (1.0 - x).powi(2)
    }
}

pub struct Bukin6;

impl Function<2> for Bukin6 {
    const F: f64 = 0.0;

    fn X() -> Vec<Point<2>> {
        vec![[-10.0, 1.0].into()]
    }

    #[inline]
    fn f(xs: Point<2>) -> f64 {
        let x = xs[0];
        let y = xs[1];

        100.0 * (y - 0.01 * x.powi(2)).abs().sqrt() + 0.01 * (x + 10.0).abs()
    }
}

pub struct Himmelblau;

impl Function<2> for Himmelblau {
    const F: f64 = 0.0;

    fn X() -> Vec<Point<2>> {
        vec![
            [-3.779310, -3.283186].into(),
            [3.0, 2.0].into(),
            [3.584428, -1.848126].into(),
            [-2.805118, 3.131312].into(),
        ]
    }

    #[inline]
    fn f(xs: Point<2>) -> f64 {
        let x = xs[0];
        let y = xs[1];

        (x * x + y - 11.0).powi(2) + (x + y * y - 7.0).powi(2)
    }
}

pub struct Booth;

impl Function<2> for Booth {
    const F: f64 = 0.0;

    fn X() -> Vec<Point<2>> {
        vec![[1.0, 3.0].into()]
    }

    #[inline]
    fn f(xs: Point<2>) -> f64 {
        let x = xs[0];
        let y = xs[1];

        (x + 2.0 * y - 7.0).powi(2) + (2.0 * x + y - 5.0).powi(2)
    }
}
