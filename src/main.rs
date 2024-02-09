#![feature(associated_type_defaults)]
#![feature(let_chains)]
#![feature(test)]
#![feature(iter_intersperse)]
#![feature(generic_const_exprs)]
#![feature(slice_as_chunks)]
#![feature(slice_first_last_chunk)]
#![feature(cell_update)]
#![deny(clippy::perf)]

extern crate core;

mod approx_model;
mod backward;
mod binary;
mod compound;
mod conjugate_directions;
mod enumerate;
mod fibonacci;
mod functions;
mod iterative_conditional;
mod method;
mod repeating;
mod restriction;
mod task;
mod uniform;
mod utils;
mod zeidel;

fn main() {}
