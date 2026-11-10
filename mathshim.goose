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

// abs_of gives the size of a number away from zero.
fn abs_of(x: f64) -> f64 {
    if x < 0.0 { return -x; }
    return x;
}

// min_of gives the smaller of two numbers.
fn min_of(a: f64, b: f64) -> f64 {
    if a < b { return a; }
    return b;
}

// tan_of gives the tangent of an angle. The standard library holds
// it on both supported versions.
fn tan_of(x: f64) -> f64 {
    return tan(x);
}

// arcsine_of gives the angle of a sine.
fn arcsine_of(x: f64) -> f64 {
    return asin(x);
}

// arccosine_of gives the angle of a cosine.
fn arccosine_of(x: f64) -> f64 {
    return acos(x);
}

// arctangent_of gives the angle of a tangent.
fn arctangent_of(x: f64) -> f64 {
    return atan(x);
}

// arctangent2_of gives the angle of a point of the plane. The first
// number stands above the second one.
fn arctangent2_of(y: f64, x: f64) -> f64 {
    return atan2(y, x);
}

// deg2rad_of turns degrees into radians.
fn deg2rad_of(x: f64) -> f64 {
    return radians(x);
}

// rad2deg_of turns radians into degrees.
fn rad2deg_of(x: f64) -> f64 {
    return degrees(x);
}

// sinh_of gives the hyperbolic sine. The power of the number gives
// the value for a large number. The difference of the two powers of
// the number gives it for a small one, where the power loses its
// digits.
fn sinh_of(x: f64) -> f64 {
    if x != x { return x - x; }
    if abs_of(x) < 0.5 {
        return (expm1_of(x) - expm1_of(-x)) / 2.0;
    }
    let e = power_e(x);
    return (e - 1.0 / e) / 2.0;
}

// cosh_of gives the hyperbolic cosine. The power of the number gives
// the value for a large number. The sum of the two powers of the
// number gives it for a small one.
fn cosh_of(x: f64) -> f64 {
    if x != x { return x - x; }
    if abs_of(x) < 0.5 {
        return 1.0 + (expm1_of(x) + expm1_of(-x)) / 2.0;
    }
    let e = power_e(x);
    return (e + 1.0 / e) / 2.0;
}

// tanh_of gives the hyperbolic tangent. The power of the number gives
// it for a large one, where the power would pass its size. The power
// of the double of the number gives it for a small one.
fn tanh_of(x: f64) -> f64 {
    if x != x { return x - x; }
    if abs_of(x) < 0.5 {
        let u = expm1_of(2.0 * x);
        return u / (u + 2.0);
    }
    let e = power_e(-2.0 * abs_of(x));
    let t = (1.0 - e) / (1.0 + e);
    if x < 0.0 { return -t; }
    return t;
}

// arcsinh_of gives the arc of the hyperbolic sine. The sum of the
// number and the root of its square holds the power to take the log.
fn arcsinh_of(x: f64) -> f64 {
    if x != x { return x - x; }
    if x >= 0.0 { return ln(x + root(x * x + 1.0)); }
    return -ln(-x + root(x * x + 1.0));
}

// arccosh_of gives the arc of the hyperbolic cosine. A number below
// one holds no arc, so the root of its square gives NaN.
fn arccosh_of(x: f64) -> f64 {
    if x != x { return x - x; }
    return ln(x + root((x - 1.0) * (x + 1.0)));
}

// arctanh_of gives the arc of the hyperbolic tangent. The ratio of
// the sum and the difference holds the power to take the log.
fn arctanh_of(x: f64) -> f64 {
    if x != x { return x - x; }
    return 0.5 * ln((1.0 + x) / (1.0 - x));
}

// expm1_of gives the power of a number less one. A series gives it
// for a small number, where the power loses its digits.
fn expm1_of(x: f64) -> f64 {
    if x != x { return x - x; }
    if abs_of(x) < 0.5 {
        var term: f64 = x;
        var s: f64 = x;
        var k: f64 = 1.0;
        while k < 30.0 {
            term = term * x / (k + 1.0);
            s += term;
            k += 1.0;
        }
        return s;
    }
    return power_e(x) - 1.0;
}

// log1p_of gives the log of a number plus one. A series gives it for
// a small number, where the sum loses its digits.
fn log1p_of(x: f64) -> f64 {
    if x != x { return x - x; }
    if abs_of(x) < 0.25 {
        var term: f64 = x;
        var s: f64 = x;
        var k: f64 = 1.0;
        while k < 40.0 {
            term = -term * x * k / (k + 1.0);
            s += term;
            k += 1.0;
        }
        return s;
    }
    return ln(1.0 + x);
}

// signbit_of tells whether a number holds the sign of minus. A zero of
// the negative side holds it too.
fn signbit_of(x: f64) -> f64 {
    if x != x { return 0.0; }
    if x < 0.0 { return 1.0; }
    if x == 0.0 && 1.0 / x < 0.0 { return 1.0; }
    return 0.0;
}

// isneginf_of tells whether a number holds the infinity below zero.
fn isneginf_of(x: f64) -> f64 {
    if x < 0.0 && is_inf(x) { return 1.0; }
    return 0.0;
}

// isposinf_of tells whether a number holds the infinity above zero.
fn isposinf_of(x: f64) -> f64 {
    if x > 0.0 && is_inf(x) { return 1.0; }
    return 0.0;
}

// reciprocal_of gives the number that multiplies to one.
fn reciprocal_of(x: f64) -> f64 {
    return 1.0 / x;
}

// is_infinite_of tells whether a number holds an infinity. The rule of
// the standard library holds. The result holds 1.0 or 0.0.
fn is_infinite_of(x: f64) -> f64 {
    if is_inf(x) { return 1.0; }
    return 0.0;
}

// is_finite_of tells whether a number holds a finite value. The result
// holds 1.0 or 0.0.
fn is_finite_of(x: f64) -> f64 {
    if x - x == 0.0 { return 1.0; }
    return 0.0;
}

// floor_divide_of gives the quotient that rounds down. The remainder
// comes first, and the quotient follows from it, as NumPy does.
fn floor_divide_of(a: f64, b: f64) -> f64 {
    if b == 0.0 { return a / b; }
    let m = a % b;
    // The remainder of an infinite number holds no value, so the
    // quotient comes from the division alone.
    if m != m { return a / b; }
    var div = (a - m) / b;
    if m != 0.0 && (b < 0.0) != (m < 0.0) { div = div - 1.0; }
    let fl = floor_of(div);
    if div - fl > 0.5 { return fl + 1.0; }
    return fl;
}
