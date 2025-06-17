use num::{FromPrimitive, Num};

pub fn linspace<T: Num + PartialOrd + Copy + FromPrimitive>(
    start: T,
    end: T,
    n: usize,
) -> impl Iterator<Item = T> {
    let dx = (end - start) / T::from_usize(n - 1).unwrap();
    let mut current = start;
    std::iter::from_fn(move || {
        if current > dx {
            None
        } else {
            current = current + dx;
            Some(current)
        }
    })
}

pub struct Bool<const B: bool>;
pub trait True {}
pub trait False {}

impl False for Bool<false> {}
impl True for Bool<true> {}
