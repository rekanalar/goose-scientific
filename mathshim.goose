// The scalar functions of the standard library, under names that do not
// collide with the array operations. The package holds array functions
// named sqrt, exp and log. This module holds the only calls of the
// stdlib versions, so no other file needs the import.
namespace mathshim;

import math;

fn root(x: f64) -> f64 { return sqrt(x); }
fn power_e(x: f64) -> f64 { return exp(x); }
fn ln(x: f64) -> f64 { return log(x); }
