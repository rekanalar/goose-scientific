// The rest of the elementwise math of NumPy: the maximum that skips
// NaN, the step, the sums of logs, the split of a number, the sinc, the
// Bessel function, the functions of whole numbers and the unwrap of an
// angle. A name of NumPy that is an alias of a function of the package
// calls that function.
namespace scientific;

import mathshim;

fn fmax(a, b) -> Arr, Fault { return bapply(a, b, mathshim::fmax_of); }
fn fmin(a, b) -> Arr, Fault { return bapply(a, b, mathshim::fmin_of); }

// heaviside gives the step function. The second array gives the value
// at zero.
fn heaviside(a, b) -> Arr, Fault { return bapply(a, b, mathshim::heaviside_of); }

fn logaddexp(a, b) -> Arr, Fault { return bapply(a, b, mathshim::logaddexp_of); }
fn logaddexp2(a, b) -> Arr, Fault { return bapply(a, b, mathshim::logaddexp2_of); }
fn sinc(a) -> Arr, Fault { return uapply(a, mathshim::sinc_of); }

// i0 gives the modified Bessel function of order zero.
fn i0(a) -> Arr, Fault { return uapply(a, mathshim::i0_of); }

// frexp splits every element into a mantissa of one half to one and a
// power of two. The result holds the mantissas and the powers.
fn frexp(a) -> Arr, Arr, Fault {
    let m, f1 = uapply(a, mathshim::frexp_man_of);
    if f1.kind != F_NONE { return empty_arr(), empty_arr(), f1; }
    let e, f2 = uapply(a, mathshim::frexp_exp_of);
    if f2.kind != F_NONE { return empty_arr(), empty_arr(), f2; }
    return m, e, no_fault();
}

// modf splits every element into a fraction and a whole part, both
// with the sign of the element. The result holds the fractions and the
// whole parts.
fn modf(a) -> Arr, Arr, Fault {
    let fr, f1 = uapply(a, mathshim::modf_frac_of);
    if f1.kind != F_NONE { return empty_arr(), empty_arr(), f1; }
    let ip, f2 = uapply(a, mathshim::modf_int_of);
    if f2.kind != F_NONE { return empty_arr(), empty_arr(), f2; }
    return fr, ip, no_fault();
}

// whole_check checks that every element holds a whole number that a
// double holds exactly: at most 2^53 in size.
fn whole_check(a, tag: const u8[:]) -> Fault {
    var i: i64 = 0;
    while i < a.data.len {
        let x = a.data[i];
        if !(x == x) || x != mathshim::trunc_of(x) || x > 9007199254740992.0 || x < -9007199254740992.0 {
            return fault(F_VALUE, tag, "an element holds no whole number of 53 bits");
        }
        i += 1;
    }
    return no_fault();
}

// ldexp gives every element times two to the power of the whole number
// in the second array.
fn ldexp(a, b) -> Arr, Fault {
    let f = whole_check(b, "ldexp");
    if f.kind != F_NONE { return empty_arr(), f; }
    return bapply(a, b, mathshim::ldexp_of);
}

// The functions of whole numbers hold the numbers in doubles. They
// refuse an element that is no whole number of 53 bits.
fn gcd(a, b) -> Arr, Fault {
    let f1 = whole_check(a, "gcd");
    if f1.kind != F_NONE { return empty_arr(), f1; }
    let f2 = whole_check(b, "gcd");
    if f2.kind != F_NONE { return empty_arr(), f2; }
    return bapply(a, b, mathshim::gcd_of);
}

fn lcm(a, b) -> Arr, Fault {
    let f1 = whole_check(a, "lcm");
    if f1.kind != F_NONE { return empty_arr(), f1; }
    let f2 = whole_check(b, "lcm");
    if f2.kind != F_NONE { return empty_arr(), f2; }
    return bapply(a, b, mathshim::lcm_of);
}

fn bitwise_and(a, b) -> Arr, Fault {
    let f1 = whole_check(a, "bitwise_and");
    if f1.kind != F_NONE { return empty_arr(), f1; }
    let f2 = whole_check(b, "bitwise_and");
    if f2.kind != F_NONE { return empty_arr(), f2; }
    return bapply(a, b, mathshim::bit_and_of);
}

fn bitwise_or(a, b) -> Arr, Fault {
    let f1 = whole_check(a, "bitwise_or");
    if f1.kind != F_NONE { return empty_arr(), f1; }
    let f2 = whole_check(b, "bitwise_or");
    if f2.kind != F_NONE { return empty_arr(), f2; }
    return bapply(a, b, mathshim::bit_or_of);
}

fn bitwise_xor(a, b) -> Arr, Fault {
    let f1 = whole_check(a, "bitwise_xor");
    if f1.kind != F_NONE { return empty_arr(), f1; }
    let f2 = whole_check(b, "bitwise_xor");
    if f2.kind != F_NONE { return empty_arr(), f2; }
    return bapply(a, b, mathshim::bit_xor_of);
}

// left_shift shifts the bits of the whole numbers to the left. A result
// above 2^53 in size loses its low digits, because the array holds
// doubles.
fn left_shift(a, b) -> Arr, Fault {
    let f1 = whole_check(a, "left_shift");
    if f1.kind != F_NONE { return empty_arr(), f1; }
    let f2 = whole_check(b, "left_shift");
    if f2.kind != F_NONE { return empty_arr(), f2; }
    return bapply(a, b, mathshim::bit_left_of);
}

fn right_shift(a, b) -> Arr, Fault {
    let f1 = whole_check(a, "right_shift");
    if f1.kind != F_NONE { return empty_arr(), f1; }
    let f2 = whole_check(b, "right_shift");
    if f2.kind != F_NONE { return empty_arr(), f2; }
    return bapply(a, b, mathshim::bit_right_of);
}

fn invert(a) -> Arr, Fault {
    let f = whole_check(a, "invert");
    if f.kind != F_NONE { return empty_arr(), f; }
    return uapply(a, mathshim::bit_not_of);
}

// bitwise_count counts the bits of the size of every whole number.
fn bitwise_count(a) -> Arr, Fault {
    let f = whole_check(a, "bitwise_count");
    if f.kind != F_NONE { return empty_arr(), f; }
    return uapply(a, mathshim::bit_count_of);
}

// unwrap turns the jumps of a phase into the smooth phase. A jump
// larger than discont, which is half the period by default, takes the
// multiple of the period that brings it back. A negative axis counts
// from the end.
fn unwrap(p, discont: f64 = 0.0 / 0.0, axis: i64 = -1, period: f64 = 6.283185307179586) -> Arr, Fault {
    var ax = axis;
    if ax < 0 { ax += p.ndim; }
    if ax < 0 || ax >= p.ndim {
        return empty_arr(), fault(F_RANGE, "axis", "the axis stands outside the shape");
    }
    if !(period > 0.0) {
        return empty_arr(), fault(F_VALUE, "unwrap", "the period holds a value that is not above zero");
    }
    var disc = discont;
    if disc != disc { disc = period / 2.0; }
    let width = p.shape[ax];
    let inner = product_after_of(p.shape, ax);
    let outer = product_before_of(p.shape, ax);
    let high = period / 2.0;
    let low = -high;
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < p.data.len { d.push(p.data[i]); i += 1; }
    var cell: i64 = 0;
    while cell < outer * inner {
        let base = (cell / inner) * (width * inner) + cell % inner;
        var run: f64 = 0.0;
        var k: i64 = 1;
        while k < width {
            let at = base + k * inner;
            let dd = p.data[at] - p.data[at - inner];
            var dm = mathshim::mod_of(dd - low, period) + low;
            if dm == low && dd > 0.0 { dm = high; }
            var corr = dm - dd;
            if mathshim::abs_of(dd) < disc { corr = 0.0; }
            run += corr;
            d[at] = p.data[at] + run;
            k += 1;
        }
        cell += 1;
    }
    return Arr { ndim: p.ndim, shape: same_shape(p), data: d }, no_fault();
}

// The names of NumPy that stand for a function of the package.
fn float_power(a, b) -> Arr, Fault { return power(a, b); }
fn pow(a, b) -> Arr, Fault { return power(a, b); }
fn remainder(a, b) -> Arr, Fault { return mod(a, b); }
fn radians(a) -> Arr, Fault { return deg2rad(a); }
fn degrees(a) -> Arr, Fault { return rad2deg(a); }
fn acos(a) -> Arr, Fault { return arccos(a); }
fn asin(a) -> Arr, Fault { return arcsin(a); }
fn atan(a) -> Arr, Fault { return arctan(a); }
fn atan2(a, b) -> Arr, Fault { return arctan2(a, b); }
fn acosh(a) -> Arr, Fault { return arccosh(a); }
fn asinh(a) -> Arr, Fault { return arcsinh(a); }
fn atanh(a) -> Arr, Fault { return arctanh(a); }
fn bitwise_not(a) -> Arr, Fault { return invert(a); }
fn bitwise_invert(a) -> Arr, Fault { return invert(a); }
fn bitwise_left_shift(a, b) -> Arr, Fault { return left_shift(a, b); }
fn bitwise_right_shift(a, b) -> Arr, Fault { return right_shift(a, b); }
fn concat(axis: i64, a, b) -> Arr, Fault { return concatenate(axis, a, b); }
fn copy_of(a) -> Arr, Fault { return clone(a); }
fn amax(a) -> f64, Fault { return max(a); }
fn amin(a) -> f64, Fault { return min(a); }

// ndim gives the count of the dimensions of an array.
fn ndim(a) -> i64 { return a.ndim; }
