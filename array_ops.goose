// The elementwise operations and the whole-array reductions.
//
// A binary operation broadcasts its two shapes, walks the result in
// row-major order and reads each source by its strides. The IEEE
// rules hold: a division by zero gives an infinity, a log of a number
// below zero gives NaN.
namespace scientific;

import mathshim;

// bwalk walks a broadcast shape and gives the flat position of each
// source. The caller gets one element pair per call through a
// callback-style walk: this version returns nothing and fills out_a
// and out_b with flat indices.
fn bcast_indices(shape: i64[:], sa: i64[:], sb: i64[:],
                 out_a: i64[>..]&, out_b: i64[>..]&) {
    var strides: i64[>..] = [];
    strides_of(shape, strides);
    var sa_strides: i64[>..] = [];
    var sb_strides: i64[>..] = [];
    var i: i64 = 0;
    while i < shape.len {
        sa_strides.push(0);
        sb_strides.push(0);
        i += 1;
    }
    // A source dimension of length 1 strides by 0: it repeats.
    i = 0;
    while i < shape.len {
        let off = sa.len - shape.len + i;
        if off >= 0 && sa[off] != 1 { sa_strides[i] = strides_of_at(sa, off); }
        let off2 = sb.len - shape.len + i;
        if off2 >= 0 && sb[off2] != 1 { sb_strides[i] = strides_of_at(sb, off2); }
        i += 1;
    }
    let total = numel(shape);
    var pos: i64[>..] = [];
    i = 0;
    while i < shape.len { pos.push(0); i += 1; }
    var n: i64 = 0;
    while n < total {
        var fa: i64 = 0;
        var fb: i64 = 0;
        i = 0;
        while i < shape.len { fa += pos[i] * sa_strides[i]; fb += pos[i] * sb_strides[i]; i += 1; }
        out_a.push(fa);
        out_b.push(fb);
        // Count one up in row-major order.
        i = shape.len - 1;
        while i >= 0 {
            pos[i] += 1;
            if pos[i] < shape[i] { break; }
            pos[i] = 0;
            i -= 1;
        }
        n += 1;
    }
    return;
}

fn strides_of_at(s: i64[:], at: i64) -> i64 {
    var acc: i64 = 1;
    var i: i64 = s.len - 1;
    while i > at {
        acc *= s[i];
        i -= 1;
    }
    return acc;
}

// bshape gives the shape that two arrays share under the broadcast
// rule. The rule of NumPy holds: the lengths are equal, or one side
// holds one, or one side has no position left.
fn bshape(a, b) -> i64[..8], Fault {
    let shape, f = broadcast_shapes(a.shape, b.shape);
    if f.kind != F_NONE { return [], f; }
    let total = numel(shape);
    if total > MAX_ELEMENTS {
        return [], fault(F_LIMIT, "broadcast", "the result holds more elements than the limit");
    }
    var s: i64[..8] = [];
    var i: i64 = 0;
    while i < shape.len { s.push(shape[i]); i += 1; }
    return s, no_fault();
}

// bop applies one arithmetic rule to two arrays under the broadcast
// rule. The rule picks the operation: 0 add, 1 sub, 2 mul, 3 div,
// 4 maximum, 5 minimum.
fn bop(a, b, rule: i32) -> Arr, Fault {
    let s, fs = bshape(a, b);
    if fs.kind != F_NONE { return empty_arr(), fs; }
    let total = numel(s);
    var ia: i64[>..] = [];
    var ib: i64[>..] = [];
    bcast_indices(s, a.shape, b.shape, ia, ib);
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < total {
        let x = a.data[ia[i]];
        let y = b.data[ib[i]];
        if rule == 0 { d.push(x + y); }
        else if rule == 1 { d.push(x - y); }
        else if rule == 2 { d.push(x * y); }
        else if rule == 3 { d.push(x / y); }
        else if rule == 4 { if x > y { d.push(x); } else { d.push(y); } }
        else if rule == 5 { if x < y { d.push(x); } else { d.push(y); } }
        else if rule == 6 { d.push(mathshim::power(x, y)); }
        else {
            var r: f64 = 0.0;
            if rule == 7 { if x == y { r = 1.0; } }
            else if rule == 8 { if x != y { r = 1.0; } }
            else if rule == 9 { if x < y { r = 1.0; } }
            else if rule == 10 { if x <= y { r = 1.0; } }
            else if rule == 11 { if x > y { r = 1.0; } }
            else if rule == 12 { if x >= y { r = 1.0; } }
            else if rule == 13 { if x != 0.0 && y != 0.0 { r = 1.0; } }
            else if rule == 14 { if x != 0.0 || y != 0.0 { r = 1.0; } }
            else { if (x != 0.0) != (y != 0.0) { r = 1.0; } }
            d.push(r);
        }
        i += 1;
    }
    return Arr { ndim: s.len, shape: s, data: d }, no_fault();
}

fn add(a, b) -> Arr, Fault { return bop(a, b, 0); }
fn sub(a, b) -> Arr, Fault { return bop(a, b, 1); }
fn mul(a, b) -> Arr, Fault { return bop(a, b, 2); }
fn div(a, b) -> Arr, Fault { return bop(a, b, 3); }
fn maximum(a, b) -> Arr, Fault { return bop(a, b, 4); }
fn minimum(a, b) -> Arr, Fault { return bop(a, b, 5); }
fn power(a, b) -> Arr, Fault { return bop(a, b, 6); }
fn equal(a, b) -> Arr, Fault { return bop(a, b, 7); }
fn not_equal(a, b) -> Arr, Fault { return bop(a, b, 8); }
fn less(a, b) -> Arr, Fault { return bop(a, b, 9); }
fn less_equal(a, b) -> Arr, Fault { return bop(a, b, 10); }
fn greater(a, b) -> Arr, Fault { return bop(a, b, 11); }
fn greater_equal(a, b) -> Arr, Fault { return bop(a, b, 12); }
fn logical_and(a, b) -> Arr, Fault { return bop(a, b, 13); }
fn logical_or(a, b) -> Arr, Fault { return bop(a, b, 14); }
fn logical_xor(a, b) -> Arr, Fault { return bop(a, b, 15); }

// uop applies one math rule to every element. The rule picks the
// operation: 0 neg, 1 abs, 2 sqrt, 3 exp, 4 log, 5 sign, 6 floor,
// 7 ceil, 8 round, 9 trunc, 10 square, 11 log2, 12 log10, 13 isnan,
// 14 isinf, 15 isfinite, 16 logical_not. The tests give 1.0 or 0.0.
fn uop(a, rule: i32) -> Arr, Fault {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < a.data.len {
        let x = a.data[i];
        var r: f64 = 0.0;
        if rule == 0 { r = -x; }
        else if rule == 1 { r = x; if x < 0.0 { r = -x; } }
        else if rule == 2 { r = mathshim::root(x); }
        else if rule == 3 { r = mathshim::power_e(x); }
        else if rule == 4 { r = mathshim::ln(x); }
        else if rule == 5 {
            r = x;
            if x > 0.0 { r = 1.0; } else if x < 0.0 { r = -1.0; } else if x == 0.0 { r = 0.0; }
        }
        else if rule == 6 { r = mathshim::floor_of(x); }
        else if rule == 7 { r = mathshim::ceil_of(x); }
        else if rule == 8 { r = mathshim::round_of(x); }
        else if rule == 9 { r = mathshim::trunc_of(x); }
        else if rule == 10 { r = x * x; }
        else if rule == 11 { r = mathshim::log2_of(x); }
        else if rule == 12 { r = mathshim::log10_of(x); }
        else if rule == 13 { if x != x { r = 1.0; } }
        else if rule == 14 { r = mathshim::is_infinite_of(x); }
        else if rule == 15 { r = mathshim::is_finite_of(x); }
        else { if x == 0.0 { r = 1.0; } }
        d.push(r);
        i += 1;
    }
    var s: i64[..8] = [];
    i = 0;
    while i < a.ndim { s.push(a.shape[i]); i += 1; }
    return Arr { ndim: a.ndim, shape: s, data: d }, no_fault();
}

fn neg(a) -> Arr, Fault { return uop(a, 0); }
fn abs(a) -> Arr, Fault { return uop(a, 1); }
fn sqrt(a) -> Arr, Fault { return uop(a, 2); }
fn exp(a) -> Arr, Fault { return uop(a, 3); }
fn log(a) -> Arr, Fault { return uop(a, 4); }
fn sign(a) -> Arr, Fault { return uop(a, 5); }
fn floor(a) -> Arr, Fault { return uop(a, 6); }
fn ceil(a) -> Arr, Fault { return uop(a, 7); }
fn trunc(a) -> Arr, Fault { return uop(a, 9); }
fn square(a) -> Arr, Fault { return uop(a, 10); }
fn log2(a) -> Arr, Fault { return uop(a, 11); }
fn log10(a) -> Arr, Fault { return uop(a, 12); }
fn isnan(a) -> Arr, Fault { return uop(a, 13); }
fn isinf(a) -> Arr, Fault { return uop(a, 14); }
fn isfinite(a) -> Arr, Fault { return uop(a, 15); }
fn logical_not(a) -> Arr, Fault { return uop(a, 16); }

// round sends every element to a whole number. The count of the
// decimals gives the place of the last digit. A half goes to the even
// neighbour, as NumPy does.
fn round(a, decimals: i64 = 0) -> Arr, Fault {
    if decimals == 0 { return uop(a, 8); }
    var d: f64[>..] = [];
    var i: i64 = 0;
    if decimals > 0 {
        let p = mathshim::power(10.0, decimals as f64);
        while i < a.data.len { d.push(mathshim::round_of(a.data[i] * p) / p); i += 1; }
    } else {
        let p = mathshim::power(10.0, -(decimals as f64));
        while i < a.data.len { d.push(mathshim::round_of(a.data[i] / p) * p); i += 1; }
    }
    return Arr { ndim: a.ndim, shape: same_shape(a), data: d }, no_fault();
}

// uapply applies one function to every element of an array. The
// caller passes the function at the end of the call.
fn uapply<F: fn(f64) -> f64>(a) -> Arr, Fault {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < a.data.len { d.push(F(a.data[i])); i += 1; }
    return Arr { ndim: a.ndim, shape: same_shape(a), data: d }, no_fault();
}

// bapply applies one function to two arrays under the broadcast rule.
// The caller passes the function at the end of the call.
fn bapply<F: fn(f64, f64) -> f64>(a, b) -> Arr, Fault {
    let s, fs = bshape(a, b);
    if fs.kind != F_NONE { return empty_arr(), fs; }
    let total = numel(s);
    var ia: i64[>..] = [];
    var ib: i64[>..] = [];
    bcast_indices(s, a.shape, b.shape, ia, ib);
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < total { d.push(F(a.data[ia[i]], b.data[ib[i]])); i += 1; }
    return Arr { ndim: s.len, shape: s, data: d }, no_fault();
}

// sin gives the sine of every element.
fn sin(a) -> Arr, Fault { return uapply(a, mathshim::sine); }

// cos gives the cosine of every element.
fn cos(a) -> Arr, Fault { return uapply(a, mathshim::cosine); }

// tan gives the tangent of every element.
fn tan(a) -> Arr, Fault { return uapply(a, mathshim::tan_of); }

// arcsin gives the angle of the sine of every element.
fn arcsin(a) -> Arr, Fault { return uapply(a, mathshim::arcsine_of); }

// arccos gives the angle of the cosine of every element.
fn arccos(a) -> Arr, Fault { return uapply(a, mathshim::arccosine_of); }

// arctan gives the angle of the tangent of every element.
fn arctan(a) -> Arr, Fault { return uapply(a, mathshim::arctangent_of); }

// arctan2 gives the angle of every point of two arrays.
fn arctan2(a, b) -> Arr, Fault { return bapply(a, b, mathshim::arctangent2_of); }

// sinh gives the hyperbolic sine of every element.
fn sinh(a) -> Arr, Fault { return uapply(a, mathshim::sinh_of); }

// cosh gives the hyperbolic cosine of every element.
fn cosh(a) -> Arr, Fault { return uapply(a, mathshim::cosh_of); }

// tanh gives the hyperbolic tangent of every element.
fn tanh(a) -> Arr, Fault { return uapply(a, mathshim::tanh_of); }

// arcsinh gives the arc of the hyperbolic sine of every element.
fn arcsinh(a) -> Arr, Fault { return uapply(a, mathshim::arcsinh_of); }

// arccosh gives the arc of the hyperbolic cosine of every element.
fn arccosh(a) -> Arr, Fault { return uapply(a, mathshim::arccosh_of); }

// arctanh gives the arc of the hyperbolic tangent of every element.
fn arctanh(a) -> Arr, Fault { return uapply(a, mathshim::arctanh_of); }

// expm1 gives the power of every element less one.
fn expm1(a) -> Arr, Fault { return uapply(a, mathshim::expm1_of); }

// log1p gives the log of every element plus one.
fn log1p(a) -> Arr, Fault { return uapply(a, mathshim::log1p_of); }

// reciprocal gives the number that multiplies every element to one.
fn reciprocal(a) -> Arr, Fault { return uapply(a, mathshim::reciprocal_of); }

// deg2rad turns the degrees of every element into radians.
fn deg2rad(a) -> Arr, Fault { return uapply(a, mathshim::deg2rad_of); }

// rad2deg turns the radians of every element into degrees.
fn rad2deg(a) -> Arr, Fault { return uapply(a, mathshim::rad2deg_of); }

// signbit tells whether every element holds the sign of minus.
fn signbit(a) -> Arr, Fault { return uapply(a, mathshim::signbit_of); }

// isneginf tells whether every element holds the infinity below zero.
fn isneginf(a) -> Arr, Fault { return uapply(a, mathshim::isneginf_of); }

// isposinf tells whether every element holds the infinity above zero.
fn isposinf(a) -> Arr, Fault { return uapply(a, mathshim::isposinf_of); }

// floor_divide gives the quotient of two arrays that rounds down.
fn floor_divide(a, b) -> Arr, Fault { return bapply(a, b, mathshim::floor_divide_of); }

// subtract takes the difference of two arrays. It is the name of
// NumPy for sub.
fn subtract(a, b) -> Arr, Fault { return sub(a, b); }

// multiply multiplies two arrays. It is the name of NumPy for mul.
fn multiply(a, b) -> Arr, Fault { return mul(a, b); }

// divide divides two arrays. It is the name of NumPy for div.
fn divide(a, b) -> Arr, Fault { return div(a, b); }

// true_divide divides two arrays. It is the name of NumPy for the
// division of numbers that hold a fraction.
fn true_divide(a, b) -> Arr, Fault { return div(a, b); }

// negative turns the sign of every element around. It is the name of
// NumPy for neg.
fn negative(a) -> Arr, Fault { return neg(a); }

// positive gives every element as it stands. It is the name of NumPy
// for the sign that a number keeps.
fn positive(a) -> Arr, Fault { return clone(a); }

// fabs gives the size of every element away from zero. It is the name
// of NumPy for abs.
fn fabs(a) -> Arr, Fault { return abs(a); }

// absolute gives the size of every element away from zero. It is the
// name of NumPy for abs.
fn absolute(a) -> Arr, Fault { return abs(a); }

// around sends every element to a whole number. It is the name of
// NumPy for round.
fn around(a, decimals: i64 = 0) -> Arr, Fault { return round(a, decimals); }

// scale multiplies every element by a number.
fn scale(a, s: f64) -> Arr, Fault {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < a.data.len { d.push(a.data[i] * s); i += 1; }
    var sh: i64[..8] = [];
    i = 0;
    while i < a.ndim { sh.push(a.shape[i]); i += 1; }
    return Arr { ndim: a.ndim, shape: sh, data: d }, no_fault();
}

// clip keeps every element inside a range.
fn clip(a, lo: f64, hi: f64) -> Arr, Fault {
    if lo > hi {
        return empty_arr(), fault(F_VALUE, "clip", "the low bound stands above the high bound");
    }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < a.data.len {
        let x = a.data[i];
        if x < lo { d.push(lo); } else if x > hi { d.push(hi); } else { d.push(x); }
        i += 1;
    }
    var sh: i64[..8] = [];
    i = 0;
    while i < a.ndim { sh.push(a.shape[i]); i += 1; }
    return Arr { ndim: a.ndim, shape: sh, data: d }, no_fault();
}

// The reductions over the whole array. A reduction of an empty array
// is a fault, except sum and prod, which give 0 and 1.

fn sum(a) -> f64, Fault {
    var t: f64 = 0.0;
    var i: i64 = 0;
    while i < a.data.len { t += a.data[i]; i += 1; }
    return t, no_fault();
}

fn prod(a) -> f64, Fault {
    var t: f64 = 1.0;
    var i: i64 = 0;
    while i < a.data.len { t *= a.data[i]; i += 1; }
    return t, no_fault();
}

fn min(a) -> f64, Fault {
    if a.data.len == 0 {
        return 0.0, fault(F_VALUE, "min", "the array holds no element");
    }
    var t: f64 = a.data[0];
    var i: i64 = 1;
    while i < a.data.len {
        if a.data[i] < t { t = a.data[i]; }
        i += 1;
    }
    return t, no_fault();
}

fn max(a) -> f64, Fault {
    if a.data.len == 0 {
        return 0.0, fault(F_VALUE, "max", "the array holds no element");
    }
    var t: f64 = a.data[0];
    var i: i64 = 1;
    while i < a.data.len {
        if a.data[i] > t { t = a.data[i]; }
        i += 1;
    }
    return t, no_fault();
}

fn mean(a) -> f64, Fault {
    if a.data.len == 0 {
        return 0.0, fault(F_VALUE, "mean", "the array holds no element");
    }
    var t: f64 = 0.0;
    var i: i64 = 0;
    while i < a.data.len { t += a.data[i]; i += 1; }
    return t / (a.data.len as f64), no_fault();
}

// shape_without gives the shape of a reduction along one axis. The
// axis drops out of the shape.
fn shape_without(a_shape: i64[:], axis: i64, out: i64[>..]&) -> Fault {
    if axis < 0 || axis >= a_shape.len {
        return fault(F_RANGE, "axis", "the axis stands outside the shape");
    }
    var i: i64 = 0;
    while i < a_shape.len {
        if i != axis { out.push(a_shape[i]); }
        i += 1;
    }
    if out.len == 0 { out.push(1); }
    return no_fault();
}

// reduce_axis runs one rule along one axis. The rule picks the
// operation: 0 sum, 1 prod, 2 min, 3 max, 4 mean.
fn reduce_axis(a, axis: i64, rule: i32) -> Arr, Fault {
    var s: i64[>..] = [];
    let sf = shape_without(a.shape, axis, s);
    if sf.kind != F_NONE { return empty_arr(), sf; }
    let inner = product_after_of(a.shape, axis);
    let width = a.shape[axis];
    let outer = product_before_of(a.shape, axis);
    var d: f64[>..] = [];
    var cell: i64 = 0;
    while cell < outer * inner {
        let base = (cell / inner) * (width * inner) + cell % inner;
        var t: f64 = 0.0;
        if rule == 1 { t = 1.0; }
        else if rule == 2 || rule == 3 {
            t = a.data[base];
        }
        var k: i64 = 0;
        while k < width {
            let x = a.data[base + k * inner];
            if rule == 0 { t += x; }
            else if rule == 1 { t *= x; }
            else if rule == 2 { if x < t { t = x; } }
            else if rule == 3 { if x > t { t = x; } }
            else { t += x; }
            k += 1;
        }
        if rule == 4 { t /= width as f64; }
        d.push(t);
        cell += 1;
    }
    var sc: i64[..8] = [];
    var i: i64 = 0;
    while i < s.len { sc.push(s[i]); i += 1; }
    return Arr { ndim: sc.len, shape: sc, data: d }, no_fault();
}

fn product_before_of(s: i64[:], axis: i64) -> i64 {
    var n: i64 = 1;
    var i: i64 = 0;
    while i < axis { n *= s[i]; i += 1; }
    return n;
}

fn product_after_of(s: i64[:], axis: i64) -> i64 {
    var n: i64 = 1;
    var i: i64 = axis + 1;
    while i < s.len { n *= s[i]; i += 1; }
    return n;
}

// sum_axis adds the elements along one axis. The axis drops from the
// shape, as in NumPy.
fn sum_axis(a, axis: i64) -> Arr, Fault { return reduce_axis(a, axis, 0); }

// prod_axis multiplies the elements along one axis.
fn prod_axis(a, axis: i64) -> Arr, Fault { return reduce_axis(a, axis, 1); }

// min_axis takes the smallest element along one axis.
fn min_axis(a, axis: i64) -> Arr, Fault { return reduce_axis(a, axis, 2); }

// max_axis takes the largest element along one axis.
fn max_axis(a, axis: i64) -> Arr, Fault { return reduce_axis(a, axis, 3); }

// mean_axis averages the elements along one axis.
fn mean_axis(a, axis: i64) -> Arr, Fault { return reduce_axis(a, axis, 4); }

// argmin gives the flat position of the smallest element. The first
// of equal elements wins. An empty array is a fault.
fn argmin(a) -> i64, Fault {
    if a.data.len == 0 {
        return -1, fault(F_VALUE, "argmin", "the array holds no element");
    }
    var best: i64 = 0;
    var i: i64 = 1;
    while i < a.data.len {
        if a.data[i] < a.data[best] { best = i; }
        i += 1;
    }
    return best, no_fault();
}

// argmax gives the flat position of the largest element. The first of
// equal elements wins. An empty array is a fault.
fn argmax(a) -> i64, Fault {
    if a.data.len == 0 {
        return -1, fault(F_VALUE, "argmax", "the array holds no element");
    }
    var best: i64 = 0;
    var i: i64 = 1;
    while i < a.data.len {
        if a.data[i] > a.data[best] { best = i; }
        i += 1;
    }
    return best, no_fault();
}

// rint gives every element rounded to a whole number. It is the name
// of NumPy for round.
fn rint(a) -> Arr, Fault { return round(a); }

// fix gives every element cut toward zero. It is the name of NumPy for
// trunc.
fn fix(a) -> Arr, Fault { return trunc(a); }

// mod gives the remainder of two arrays with the sign of the divisor.
fn mod(a, b) -> Arr, Fault { return bapply(a, b, mathshim::mod_of); }

// fmod gives the remainder of two arrays with the sign of the dividend.
fn fmod(a, b) -> Arr, Fault { return bapply(a, b, mathshim::fmod_of); }

// hypot gives the length of two numbers.
fn hypot(a, b) -> Arr, Fault { return bapply(a, b, mathshim::hypot_of); }

// nextafter gives the next number of the machine toward the second
// number.
fn nextafter(a, b) -> Arr, Fault { return bapply(a, b, mathshim::nextafter_of); }

// copysign gives the size of every element with the sign of the second.
fn copysign(a, b) -> Arr, Fault { return bapply(a, b, mathshim::copysign_of); }

// cbrt gives the real root of the cube of every element.
fn cbrt(a) -> Arr, Fault { return uapply(a, mathshim::cbrt_of); }

// exp2 gives two to the power of every element.
fn exp2(a) -> Arr, Fault { return uapply(a, mathshim::exp2_of); }

// spacing gives the distance to the next number of the machine above
// every element.
fn spacing(a) -> Arr, Fault { return uapply(a, mathshim::spacing_of); }
