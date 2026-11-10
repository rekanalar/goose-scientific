// The scalar functions of the standard library, under names that do not
// collide with the array operations. The package holds array functions
// named sqrt, exp and log. This module holds the only calls of the
// stdlib versions, so no other file needs the import.
namespace mathshim;

import math;

fn root(x: f64) -> f64 { return sqrt(x); }
fn power_e(x: f64) -> f64 { return exp(x); }
fn ln(x: f64) -> f64 { return log(x); }
fn cosine(x: f64) -> f64 { return cos(x); }
fn sine(x: f64) -> f64 { return sin(x); }
fn log2_of(x: f64) -> f64 { return log2(x); }
fn log10_of(x: f64) -> f64 { return log10(x); }
fn power(x: f64, y: f64) -> f64 { return pow(x, y); }

// trunc drops the fraction. A number of 2^62 or more, an infinity and
// NaN hold no fraction to drop, so they pass through.
fn trunc_of(x: f64) -> f64 {
    if x != x || x >= 4611686018427387904.0 || x <= -4611686018427387904.0 { return x; }
    let t = x as i64;
    return t;
}

fn floor_of(x: f64) -> f64 {
    let t = trunc_of(x);
    if t > x { return t - 1.0; }
    return t;
}

fn ceil_of(x: f64) -> f64 {
    let t = trunc_of(x);
    if t < x { return t + 1.0; }
    return t;
}

// round sends a half to the even neighbour, like NumPy.
fn round_of(x: f64) -> f64 {
    let f = floor_of(x);
    let d = x - f;
    if d < 0.5 { return f; }
    if d > 0.5 { return f + 1.0; }
    let half = f / 2.0;
    if trunc_of(half) == half { return f; }
    return f + 1.0;
}
