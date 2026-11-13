// The scalar functions of the standard library, under names that do not collide with the array operations.
//
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
// NaN hold no fraction to drop, so they pass through. A zero keeps its
// sign, as NumPy keeps it.
fn trunc_of(x: f64) -> f64 {
    if x != x || x >= 4611686018427387904.0 || x <= -4611686018427387904.0 { return x; }
    let t = x as i64;
    if t == 0 { return signed_zero(x); }
    return t;
}

// signed_zero gives a zero with the sign of the number.
fn signed_zero(x: f64) -> f64 {
    if x < 0.0 { return -0.0; }
    if x == 0.0 && 1.0 / x < 0.0 { return -0.0; }
    return 0.0;
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

// round sends a half to the even neighbour, like NumPy. A result of
// zero keeps the sign of the number, as NumPy keeps it.
fn round_of(x: f64) -> f64 {
    if x != x { return x - x; }
    let f = floor_of(x);
    let d = x - f;
    var r = f;
    if d > 0.5 { r = f + 1.0; }
    if d == 0.5 {
        let half = f / 2.0;
        if trunc_of(half) != half { r = f + 1.0; }
    }
    if r == 0.0 { return signed_zero(x); }
    return r;
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

// expm1_of gives the power of a number less one. The method is the
// classic libc reduction: a shift by whole steps of the natural log of
// two, and a rational form for the short range that stays near the
// result. The error stays under one unit in the last place.
fn expm1_of(x: f64) -> f64 {
    if x != x { return x; }
    let o_threshold: f64 = 7.09782712893383973096e+02;
    let ln2_hi: f64 = 6.93147180369123816490e-01;
    let ln2_lo: f64 = 1.90821492927058770002e-10;
    let invln2: f64 = 1.44269504088896338700e+00;
    let q1: f64 = -3.33333333333331316428e-02;
    let q2: f64 = 1.58730158725481460165e-03;
    let q3: f64 = -7.93650757867487942473e-05;
    let q4: f64 = 4.00821782732936239552e-06;
    let q5: f64 = -2.01099218183624371326e-07;
    let t_big: f64 = 38.81622314453125;
    let t_half: f64 = 0.3465733528137207;
    let t_one: f64 = 1.0397205352783203;
    let t_tiny: f64 = 5.551115123125783e-17;
    let sign: bool = x < 0.0;
    let ax = abs_of(x);
    if ax >= t_big {
        if sign { return -1.0; }
        if x > o_threshold { return x * power(2.0, 1023.0); }
    }
    var k: f64 = 0.0;
    var hi: f64 = x;
    var lo: f64 = 0.0;
    var cc: f64 = 0.0;
    var xr: f64 = x;
    if ax > t_half {
        if ax < t_one {
            if !sign { hi = x - ln2_hi; lo = ln2_lo; k = 1.0; }
            else { hi = x + ln2_hi; lo = -ln2_lo; k = -1.0; }
        } else {
            if sign { k = trunc_of(invln2 * x - 0.5); }
            else { k = trunc_of(invln2 * x + 0.5); }
            hi = x - k * ln2_hi;
            lo = k * ln2_lo;
        }
        xr = hi - lo;
        cc = (hi - xr) - lo;
    } else if ax < t_tiny { return x; }
    let hfx = 0.5 * xr;
    let hxs = xr * hfx;
    let r1 = 1.0 + hxs * (q1 + hxs * (q2 + hxs * (q3 + hxs * (q4 + hxs * q5))));
    let t = 3.0 - r1 * hfx;
    var ev = hxs * ((r1 - t) / (6.0 - xr * t));
    if k == 0.0 { return xr - (xr * ev - hxs); }
    ev = xr * (ev - cc) - cc;
    ev -= hxs;
    if k == -1.0 { return 0.5 * (xr - ev) - 0.5; }
    if k == 1.0 {
        if xr < -0.25 { return -2.0 * (ev - (xr + 0.5)); }
        return 1.0 + 2.0 * (xr - ev);
    }
    let pk = power(2.0, k);
    if k < 0.0 || k > 56.0 {
        return (xr - ev + 1.0) * pk - 1.0;
    }
    let nm = power(2.0, -k);
    if k < 20.0 { return (xr - ev + (1.0 - nm)) * pk; }
    return (xr - (ev + nm) + 1.0) * pk;
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

// cbrt_of gives the real root of the cube. The power of the log of the
// number over three gives the first value, and one step of Newton
// takes the digits back. The step holds the cube of the value as two
// numbers, so the correction keeps every digit. A number near the
// ends of the range scales by a power of eight, so no cube passes
// over the size of a number.
fn cbrt_of(x: f64) -> f64 {
    if x != x { return x - x; }
    if x == 0.0 { return x; }
    if is_inf(x) { return x; }
    var sign: f64 = 1.0;
    if x < 0.0 { sign = -1.0; }
    let a = abs_of(x);
    var scaled = a;
    var back: f64 = 1.0;
    if a > 1.0e200 { scaled = a / 4.149515568880993e180; back = 1.6069380442589903e60; }
    if a < 1.0e-200 { scaled = a * 4.149515568880993e180; back = 6.223015277861142e-61; }
    let y = power_e(ln(scaled) / 3.0);
    // The cube of y as two numbers: y*y and then the cube, each as a
    // value and the part that the value loses.
    let p, pe = two_product(y, y);
    let c, ce = two_product(p, y);
    let lo = pe * y + ce;
    // The cube less the number holds the error of the first value.
    let r0 = c - scaled;
    let rh, rl = two_sum(r0, lo);
    let s = 3.0 * p;
    let q = rh / s;
    let qq, qe = two_product(s, q);
    let rem_h, rem_l = two_sum(rh - qq, rl - qe);
    let corr = q + (rem_h + rem_l) / s;
    return sign * (y - corr) * back;
}

