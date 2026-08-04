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

#[derive(Debug, Clone, Copy)]
pub struct Intersperse<I: Iterator> {
    started: bool,
    separator: I::Item,
    next_item: Option<I::Item>,
    iter: I,
}

impl<I> Iterator for Intersperse<I>
where
    I::Item: Clone,
    I: Iterator,
{
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.started {
            if let Some(v) = self.next_item.take() {
                Some(v)
            } else {
                let next_item = self.iter.next();
                if next_item.is_some() {
                    self.next_item = next_item;
                    Some(self.separator.clone())
                } else {
                    None
                }
            }
        } else {
            let item = self.iter.next();
            self.started = item.is_some();
            item
        }
    }
}

pub trait IntersperseIteratorExt: Iterator + Sized {
    #[inline]
    fn intersperse_ext(self, separator: Self::Item) -> Intersperse<Self> {
        Intersperse {
            started: false,
            separator,
            next_item: None,
            iter: self,
        }
    }
}

impl<I: Iterator> IntersperseIteratorExt for I {}

pub struct Bool<const B: bool>;
pub trait True {}
pub trait False {}

impl False for Bool<false> {}
impl True for Bool<true> {}
