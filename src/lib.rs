#![allow(dead_code)]
#![cfg_attr(feature = "nightly", test)]

mod approx_model;
mod backward;
mod binary;
// mod compound;
mod conjugate_directions;
mod enumerate;
mod fibonacci;
mod functions;
mod iterative_conditional;
mod method;
mod repeating;
mod restriction;
mod task;
// mod uniform;
mod utils;
mod zeidel;

#[cfg(test)]
mod test {
    #[cfg(feature = "nightly")]
    extern crate test;
    
    #[cfg(feature = "nightly")]
    pub type Bencher = test::Bencher;

    #[macro_export]
    macro_rules! def_test {
        ($name:ident $body:block) => {
            #[cfg(feature = "nightly")]
            fn $name(b: &mut crate::test::Bencher) {
                b.iter(|| $body);
            }

            #[cfg(not(feature = "nightly"))]
            fn $name() {
                $body
            }
        };
    }
}
