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

// bop applies one arithmetic rule to two arrays under the broadcast
// rule. The rule picks the operation: 0 add, 1 sub, 2 mul, 3 div,
// 4 maximum, 5 minimum.
fn bop(a, b, rule: i32) -> Arr, Fault {
    var s: i64[..8] = [];
    let n = max_i(a.ndim, b.ndim);
    var i: i64 = 0;
    while i < n {
        let x = at_or_one(a.shape, a.ndim - n + i);
        let y = at_or_one(b.shape, b.ndim - n + i);
        if x == y { s.push(x); }
        else if x == 1 { s.push(y); }
        else if y == 1 { s.push(x); }
        else {
            return empty_arr(), fault(F_SHAPE, "broadcast", "the shapes hold two lengths that are not equal and not one");
        }
        i += 1;
    }
    if s.len == 0 { s.push(1); }
    let total = numel(s);
    if total > MAX_ELEMENTS {
        return empty_arr(), fault(F_LIMIT, "broadcast", "the result holds more elements than the limit");
    }
    var ia: i64[>..] = [];
    var ib: i64[>..] = [];
    bcast_indices(s, a.shape, b.shape, ia, ib);
    var d: f64[>..] = [];
    i = 0;
    while i < total {
        let x = a.data[ia[i]];
        let y = b.data[ib[i]];
        if rule == 0 { d.push(x + y); }
        else if rule == 1 { d.push(x - y); }
        else if rule == 2 { d.push(x * y); }
        else if rule == 3 { d.push(x / y); }
        else if rule == 4 { if x > y { d.push(x); } else { d.push(y); } }
        else { if x < y { d.push(x); } else { d.push(y); } }
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

// uop applies one math rule to every element. The rule picks the
// operation: 0 neg, 1 abs, 2 sqrt, 3 exp, 4 log.
fn uop(a, rule: i32) -> Arr, Fault {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < a.data.len {
        let x = a.data[i];
        if rule == 0 { d.push(-x); }
        else if rule == 1 { if x < 0.0 { d.push(-x); } else { d.push(x); } }
        else if rule == 2 { d.push(mathshim::root(x)); }
        else if rule == 3 { d.push(mathshim::power_e(x)); }
        else { d.push(mathshim::ln(x)); }
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
