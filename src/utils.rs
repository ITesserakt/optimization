use num::{Float, FromPrimitive};

pub fn linspace<T: Float + FromPrimitive + 'static>(
    start: T,
    end: T,
    n: usize,
) -> impl Iterator<Item = T> {
    let to_float = |i: usize| T::from_usize(i).unwrap();
    let dx = (end - start) / to_float(n - 1);
    (0..n).map(move |i| start + to_float(i) * dx)
}

pub struct Bool<const B: bool>;
pub trait True {}
pub trait False {}

impl False for Bool<false> {}
impl True for Bool<true> {}
