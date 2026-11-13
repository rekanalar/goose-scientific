// The functions of the floating point format and of whole numbers.
//
namespace mathshim;

import math;

// two_product gives a product as two numbers. The first holds the
// value and the second holds the part that the value loses. The split
// of Dekker takes each number apart at 27 bits, so the parts hold no
// overlap.
fn two_product(x: f64, y: f64) -> f64, f64 {
    let split = 134217729.0;
    let cx = x * split;
    let hi_x = cx - (cx - x);
    let lo_x = x - hi_x;
    let cy = y * split;
    let hi_y = cy - (cy - y);
    let lo_y = y - hi_y;
    let p = x * y;
    let err = ((hi_x * hi_y - p) + hi_x * lo_y + lo_x * hi_y) + lo_x * lo_y;
    return p, err;
}

// two_sum gives a sum as two numbers. The first holds the value and
// the second holds the part that the value loses.
fn two_sum(a: f64, b: f64) -> f64, f64 {
    let s = a + b;
    let bb = s - a;
    return s, (a - (s - bb)) + (b - bb);
}

// exp2_of gives two to the power of the number. The power of the
// standard library takes the whole work, so the result holds the same
// digits as the power of NumPy holds.
fn exp2_of(x: f64) -> f64 {
    if x != x { return x - x; }
    return power(2.0, x);
}

// copysign_of gives the size of the first number with the sign of the
// second. A zero of the negative side holds the sign of minus.
fn copysign_of(x: f64, y: f64) -> f64 {
    if y < 0.0 { return -abs_of(x); }
    if y == 0.0 && 1.0 / y < 0.0 { return -abs_of(x); }
    return abs_of(x);
}

// mod_of gives the remainder with the sign of the divisor, as Python
// holds it. The remainder of the language holds the sign of the
// dividend, so the divisor comes on when the signs differ.
fn mod_of(x: f64, y: f64) -> f64 {
    let m = x % y;
    if m != m { return m; }
    if m == 0.0 { return signed_zero(y); }
    if (m < 0.0) != (y < 0.0) { return m + y; }
    return m;
}

// fmod_of gives the remainder with the sign of the dividend, as the C
// library holds it. The remainder of the language holds it already.
fn fmod_of(x: f64, y: f64) -> f64 {
    return x % y;
}

// hypot_of gives the length of two numbers. The squares hold their
// lost parts as two numbers, so the root sees the exact sum and the
// result holds every digit, as the C library holds it. A size close
// to the ends of the range scales by a power of two, and the power
// of two comes back without a loss.
fn hypot_of(x: f64, y: f64) -> f64 {
    let a = abs_of(x);
    let b = abs_of(y);
    var m = a;
    if b > m { m = b; }
    if m == 0.0 { return 0.0; }
    if is_inf(m) { return m; }
    var s: f64 = 1.0;
    if m >= 1e150 { s = 2.409919865102884e-181; }
    if m <= 1e-150 { s = 4.149515568880993e180; }
    let u = a * s;
    let v = b * s;
    let s1, e1 = two_product(u, u);
    let s2, e2 = two_product(v, v);
    let hi, err = two_sum(s1, s2);
    let lo = e1 + e2 + err;
    var r = root(hi);
    // One step of Newton on the exact sum takes the lost digits back.
    let ph, pe = two_product(r, r);
    let d = (ph - hi) + (pe - lo);
    r = r - d / (2.0 * r);
    return r / s;
}

// exponent_of gives the exponent of the power of two that holds a
// number. The log of two gives it, and the powers of two around it
// correct it.
fn exponent_of(x: f64) -> f64 {
    let a = abs_of(x);
    let e = floor_of(log2_of(a));
    if power(2.0, e) > a { return e - 1.0; }
    if power(2.0, e + 1.0) <= a { return e + 1.0; }
    return e;
}

// spacing_of gives the distance to the next number of the machine above
// the number. The distance holds the sign of the number.
fn spacing_of(x: f64) -> f64 {
    if x != x { return x - x; }
    if is_inf(x) { return x - x; }
    let a = abs_of(x);
    if a == 0.0 { return 4.9406564584124654e-324; }
    // The largest number has no neighbour above it but the infinity.
    if a == 1.7976931348623157e308 {
        if x < 0.0 { return -1.0 / 0.0; }
        return 1.0 / 0.0;
    }
    // The numbers below the smallest normal come one step apart.
    if a < 2.2250738585072014e-308 {
        if x < 0.0 { return -4.9406564584124654e-324; }
        return 4.9406564584124654e-324;
    }
    let gap = power(2.0, exponent_of(x) - 52.0);
    if x < 0.0 { return -gap; }
    return gap;
}

// nextafter_of gives the next number of the machine toward the second
// number. The distance below a power of two is half the distance above
// it, so the step follows the direction of the second number.
fn nextafter_of(x: f64, y: f64) -> f64 {
    if x != x { return x - x; }
    if y != y { return y - y; }
    if x == y { return y; }
    // The largest number of the machine stands next to the infinity.
    if is_inf(x) {
        if x > 0.0 { return 1.7976931348623157e308; }
        return -1.7976931348623157e308;
    }
    // The smallest number of the machine stands next to the zero.
    if x == 0.0 {
        if y > 0.0 { return 4.9406564584124654e-324; }
        return -4.9406564584124654e-324;
    }
    let a = abs_of(x);
    // The numbers below the smallest normal come one step apart.
    if a < 2.2250738585072014e-308 {
        var r = x - 4.9406564584124654e-324;
        if y > x { r = x + 4.9406564584124654e-324; }
        if r == 0.0 { return signed_zero(x); }
        return r;
    }
    let up = power(2.0, exponent_of(x) - 52.0);
    var down = up;
    // Below the smallest normal the step stays one subnormal step.
    if a == 2.2250738585072014e-308 { down = 4.9406564584124654e-324; }
    else if power(2.0, exponent_of(x)) == a { down = up / 2.0; }
    if y > x {
        if x > 0.0 { return x + up; }
        return x + down;
    }
    if x > 0.0 { return x - down; }
    return x - up;
}

// fmax_of gives the larger number and skips NaN, as the C library
// does: a NaN gives way to the other number.
fn fmax_of(a: f64, b: f64) -> f64 {
    if a >= b || b != b { return a; }
    return b;
}

// fmin_of gives the smaller number and skips NaN.
fn fmin_of(a: f64, b: f64) -> f64 {
    if a <= b || b != b { return a; }
    return b;
}

// heaviside_of gives the step function. A zero takes the value h.
fn heaviside_of(x: f64, h: f64) -> f64 {
    if x != x { return x; }
    if x == 0.0 { return h; }
    if x < 0.0 { return 0.0; }
    return 1.0;
}

// logaddexp_of gives the log of the sum of the powers of e, without
// the overflow of the powers.
fn logaddexp_of(x: f64, y: f64) -> f64 {
    if x == y { return x + 0.6931471805599453; }
    let t = x - y;
    if t > 0.0 { return x + log1p_of(power_e(-t)); }
    if t <= 0.0 { return y + log1p_of(power_e(t)); }
    return t;
}

// logaddexp2_of gives the log base two of the sum of the powers of two.
fn logaddexp2_of(x: f64, y: f64) -> f64 {
    if x == y { return x + 1.0; }
    let t = x - y;
    if t > 0.0 { return x + 1.4426950408889634 * log1p_of(exp2_of(-t)); }
    if t <= 0.0 { return y + 1.4426950408889634 * log1p_of(exp2_of(t)); }
    return t;
}

// ldexp_of gives x times two to the power e. The count e cuts toward
// zero. A result below the smallest normal number may round twice.
fn ldexp_of(x: f64, e: f64) -> f64 {
    if x != x || x == 0.0 || is_inf(x) { return x; }
    if e != e { return e; }
    var n = e;
    if n > 4000.0 { n = 4000.0; }
    if n < -4000.0 { n = -4000.0; }
    var k = n as i64;
    var r = x;
    while k > 1000 { r = r * power(2.0, 1000.0); k -= 1000; }
    while k < -1000 { r = r * power(2.0, -1000.0); k += 1000; }
    return r * power(2.0, k);
}

// frexp_man_of gives the mantissa of a number, in the range of one
// half to one, with the sign of the number.
fn frexp_man_of(x: f64) -> f64 {
    if x != x || x == 0.0 || is_inf(x) { return x; }
    return x / power(2.0, exponent_of(x)) * 0.5;
}

// frexp_exp_of gives the power of two that goes with the mantissa.
fn frexp_exp_of(x: f64) -> f64 {
    if x != x || x == 0.0 || is_inf(x) { return 0.0; }
    return exponent_of(x) + 1.0;
}

// modf_int_of gives the whole part of a number.
fn modf_int_of(x: f64) -> f64 {
    if x != x || is_inf(x) { return x; }
    return trunc_of(x);
}

// modf_frac_of gives the fraction of a number with its sign.
fn modf_frac_of(x: f64) -> f64 {
    if x != x { return x; }
    if is_inf(x) { return signed_zero(x); }
    let f = x - trunc_of(x);
    if f == 0.0 { return signed_zero(x); }
    return f;
}

// sinc_of gives sin(pi x) over pi x. A zero gives one, as NumPy gives
// it.
fn sinc_of(x: f64) -> f64 {
    var y = x;
    if x == 0.0 { y = 1e-20; }
    y = 3.141592653589793 * y;
    return sin(y) / y;
}

// i0_of gives the modified Bessel function of the first kind and of
// order zero. The series holds only positive terms, so it loses no
// digit to a difference. A number above 713.9 gives the infinity.
fn i0_of(x: f64) -> f64 {
    if x != x { return x; }
    let a = abs_of(x);
    if a > 713.98 { return 1.0 / 0.0; }
    let q = a * a / 4.0;
    var term: f64 = 1.0;
    var sum: f64 = 1.0;
    var k: f64 = 1.0;
    while k < 2000.0 {
        term = term * q / (k * k);
        sum = sum + term;
        if term < sum * 1e-17 { break; }
        k = k + 1.0;
    }
    return sum;
}

// gcd_of gives the greatest common divisor of two whole numbers that
// the numbers of 64 bits hold.
fn gcd_of(a: f64, b: f64) -> f64 {
    var x = abs_of(a) as i64;
    var y = abs_of(b) as i64;
    while y != 0 {
        let t = x % y;
        x = y;
        y = t;
    }
    return x;
}

// lcm_of gives the least common multiple of two whole numbers.
fn lcm_of(a: f64, b: f64) -> f64 {
    let g = gcd_of(a, b);
    if g == 0.0 { return 0.0; }
    return abs_of(a) / g * abs_of(b);
}

fn bit_and_of(a: f64, b: f64) -> f64 { return (a as i64) & (b as i64); }
fn bit_or_of(a: f64, b: f64) -> f64 { return (a as i64) | (b as i64); }
fn bit_xor_of(a: f64, b: f64) -> f64 { return (a as i64) ^ (b as i64); }
fn bit_not_of(a: f64) -> f64 { return -1.0 - a; }

// bit_left_of shifts to the left. A count outside zero to 63 gives
// zero, as NumPy gives it.
fn bit_left_of(a: f64, b: f64) -> f64 {
    if b < 0.0 || b >= 64.0 { return 0.0; }
    return (((a as i64) as u64) << (b as u64)) as i64;
}

// bit_right_of shifts to the right and keeps the sign. A count outside
// zero to 63 gives zero or minus one.
fn bit_right_of(a: f64, b: f64) -> f64 {
    if b < 0.0 || b >= 64.0 {
        if a < 0.0 { return -1.0; }
        return 0.0;
    }
    return (a as i64) >> (b as i64);
}

// bit_count_of counts the bits of the size of a whole number.
fn bit_count_of(a: f64) -> f64 {
    var v = abs_of(a) as u64;
    var n: i64 = 0;
    while v != 0 {
        n += (v & 1) as i64;
        v = v >> 1;
    }
    return n;
}
