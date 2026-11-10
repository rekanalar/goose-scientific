// Selection, truth tests, running sums, differences, flips and the
// spread of the values.
//
// A comparison gives an array of 1.0 and 0.0, because the array holds
// one element type. A zero is false and any other value is true.
namespace scientific;

import mathshim;

// where picks from a where the condition holds and from b where it does
// not. The three shapes broadcast together.
fn where(cond, a, b) -> Arr, Fault {
    let sh, f = broadcast_shapes(cond.shape, a.shape);
    if f.kind != F_NONE { return empty_arr(), f; }
    let sh2, f2 = broadcast_shapes(sh, b.shape);
    if f2.kind != F_NONE { return empty_arr(), f2; }
    let ones_arr, f3 = ones(sh2);
    if f3.kind != F_NONE { return empty_arr(), f3; }
    let c, f4 = mul(cond, ones_arr);
    let x, f5 = mul(a, ones_arr);
    let y, f6 = mul(b, ones_arr);
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < c.data.len {
        if c.data[i] != 0.0 { d.push(x.data[i]); } else { d.push(y.data[i]); }
        i += 1;
    }
    var s: i64[..8] = [];
    i = 0;
    while i < c.ndim { s.push(c.shape[i]); i += 1; }
    return Arr { ndim: c.ndim, shape: s, data: d }, no_fault();
}

// all tells whether every element is true. any tells whether one is.
// near holds the rule of NumPy for one pair of numbers. Two equal
// numbers are close. Two numbers of the missing value are close only
// when the flag says so. The rest follows the size of the difference
// and the size of the second number.
fn near(x: f64, y: f64, rtol: f64, atol: f64, equal_nan: bool) -> bool {
    if x == y { return true; }
    if x != x || y != y {
        return equal_nan && x != x && y != y;
    }
    let d = x - y;
    var size = y;
    if y < 0.0 { size = -y; }
    if d < 0.0 { return -d <= atol + rtol * size; }
    return d <= atol + rtol * size;
}

// isclose tells for every pair of elements whether they sit close to
// each other. The sizes come from NumPy: a relative size of 1e-05 and
// an absolute size of 1e-08.
fn isclose(a, b, rtol: f64 = 1.0e-05, atol: f64 = 1.0e-08, equal_nan: bool = false) -> Arr, Fault {
    let s, fs = bshape(a, b);
    if fs.kind != F_NONE { return empty_arr(), fs; }
    let total = numel(s);
    var ia: i64[>..] = [];
    var ib: i64[>..] = [];
    bcast_indices(s, a.shape, b.shape, ia, ib);
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < total {
        if near(a.data[ia[i]], b.data[ib[i]], rtol, atol, equal_nan) { d.push(1.0); }
        else { d.push(0.0); }
        i += 1;
    }
    return Arr { ndim: s.len, shape: s, data: d }, no_fault();
}

// allclose tells whether every pair of elements sits close to each
// other, under the rule of isclose.
fn allclose(a, b, rtol: f64 = 1.0e-05, atol: f64 = 1.0e-08, equal_nan: bool = false) -> bool, Fault {
    let r, f = isclose(a, b, rtol, atol, equal_nan);
    if f.kind != F_NONE { return false, f; }
    var i: i64 = 0;
    while i < r.data.len {
        if r.data[i] == 0.0 { return false, no_fault(); }
        i += 1;
    }
    return true, no_fault();
}

// array_equal tells whether two arrays hold one shape and equal
// elements. The missing value equals itself only when the flag says
// so, as in NumPy.
fn array_equal(a, b, equal_nan: bool = false) -> bool, Fault {
    if a.ndim != b.ndim { return false, no_fault(); }
    var i: i64 = 0;
    while i < a.ndim {
        if a.shape[i] != b.shape[i] { return false, no_fault(); }
        i += 1;
    }
    if a.data.len != b.data.len { return false, no_fault(); }
    i = 0;
    while i < a.data.len {
        let x = a.data[i];
        let y = b.data[i];
        if x != x || y != y {
            if equal_nan && x != x && y != y { i += 1; continue; }
            return false, no_fault();
        }
        if x != y { return false, no_fault(); }
        i += 1;
    }
    return true, no_fault();
}

fn all(a) -> bool, Fault {
    var i: i64 = 0;
    while i < a.data.len {
        if a.data[i] == 0.0 { return false, no_fault(); }
        i += 1;
    }
    return true, no_fault();
}

fn any(a) -> bool, Fault {
    var i: i64 = 0;
    while i < a.data.len {
        if a.data[i] != 0.0 { return true, no_fault(); }
        i += 1;
    }
    return false, no_fault();
}

fn count_nonzero(a) -> i64, Fault {
    var n: i64 = 0;
    var i: i64 = 0;
    while i < a.data.len {
        if a.data[i] != 0.0 { n += 1; }
        i += 1;
    }
    return n, no_fault();
}

fn same_shape(a) -> i64[..8] {
    var s: i64[..8] = [];
    var i: i64 = 0;
    while i < a.ndim { s.push(a.shape[i]); i += 1; }
    return s;
}

// cum_axis runs a running total along one axis. The rule picks the
// operation: 0 sum, 1 product.
fn cum_axis(a, axis: i64, rule: i32) -> Arr, Fault {
    if axis < 0 || axis >= a.ndim {
        return empty_arr(), fault(F_RANGE, "axis", "the axis stands outside the shape");
    }
    let inner = product_after_of(a.shape, axis);
    let width = a.shape[axis];
    let outer = product_before_of(a.shape, axis);
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < a.data.len { d.push(a.data[i]); i += 1; }
    var cell: i64 = 0;
    while cell < outer * inner {
        let base = (cell / inner) * (width * inner) + cell % inner;
        var k: i64 = 1;
        while k < width {
            let at = base + k * inner;
            if rule == 0 { d[at] = d[at - inner] + d[at]; } else { d[at] = d[at - inner] * d[at]; }
            k += 1;
        }
        cell += 1;
    }
    return Arr { ndim: a.ndim, shape: same_shape(a), data: d }, no_fault();
}

fn cumsum(a, axis: i64) -> Arr, Fault { return cum_axis(a, axis, 0); }
fn cumprod(a, axis: i64) -> Arr, Fault { return cum_axis(a, axis, 1); }

// diff gives the difference of neighbours along an axis. The axis
// loses one element, so its length must be two or more.
fn diff(a, axis: i64) -> Arr, Fault {
    if axis < 0 || axis >= a.ndim {
        return empty_arr(), fault(F_RANGE, "axis", "the axis stands outside the shape");
    }
    let width = a.shape[axis];
    if width < 2 {
        return empty_arr(), fault(F_SHAPE, "diff", "the axis holds fewer than two elements");
    }
    let inner = product_after_of(a.shape, axis);
    let outer = product_before_of(a.shape, axis);
    var d: f64[>..] = [];
    var o: i64 = 0;
    while o < outer {
        var k: i64 = 1;
        while k < width {
            var j: i64 = 0;
            while j < inner {
                let at = o * width * inner + k * inner + j;
                d.push(a.data[at] - a.data[at - inner]);
                j += 1;
            }
            k += 1;
        }
        o += 1;
    }
    var s = same_shape(a);
    s[axis] = width - 1;
    return Arr { ndim: a.ndim, shape: s, data: d }, no_fault();
}

// flip reverses the order of the elements along an axis.
fn flip(a, axis: i64) -> Arr, Fault {
    if axis < 0 || axis >= a.ndim {
        return empty_arr(), fault(F_RANGE, "axis", "the axis stands outside the shape");
    }
    let width = a.shape[axis];
    let inner = product_after_of(a.shape, axis);
    let outer = product_before_of(a.shape, axis);
    var d: f64[>..] = [];
    var o: i64 = 0;
    while o < outer {
        var k: i64 = 0;
        while k < width {
            var j: i64 = 0;
            while j < inner {
                d.push(a.data[o * width * inner + (width - 1 - k) * inner + j]);
                j += 1;
            }
            k += 1;
        }
        o += 1;
    }
    return Arr { ndim: a.ndim, shape: same_shape(a), data: d }, no_fault();
}

// var_axis gives the variance along an axis. The divisor is the length
// minus ddof. A ddof that leaves no positive divisor is a fault.
fn var_axis(a, axis: i64, ddof: i64) -> Arr, Fault {
    var s: i64[>..] = [];
    let sf = shape_without(a.shape, axis, s);
    if sf.kind != F_NONE { return empty_arr(), sf; }
    let width = a.shape[axis];
    if ddof < 0 || width - ddof < 1 {
        return empty_arr(), fault(F_VALUE, "var", "the ddof leaves no positive divisor");
    }
    let inner = product_after_of(a.shape, axis);
    let outer = product_before_of(a.shape, axis);
    var d: f64[>..] = [];
    var cell: i64 = 0;
    while cell < outer * inner {
        let base = (cell / inner) * (width * inner) + cell % inner;
        var m: f64 = 0.0;
        var k: i64 = 0;
        while k < width { m += a.data[base + k * inner]; k += 1; }
        m /= width as f64;
        var t: f64 = 0.0;
        k = 0;
        while k < width {
            let e = a.data[base + k * inner] - m;
            t += e * e;
            k += 1;
        }
        d.push(t / ((width - ddof) as f64));
        cell += 1;
    }
    var sc: i64[..8] = [];
    var i: i64 = 0;
    while i < s.len { sc.push(s[i]); i += 1; }
    return Arr { ndim: sc.len, shape: sc, data: d }, no_fault();
}

fn std_axis(a, axis: i64, ddof: i64) -> Arr, Fault {
    let v, f = var_axis(a, axis, ddof);
    if f.kind != F_NONE { return empty_arr(), f; }
    return sqrt(v);
}

// variance and std run over the whole array.
fn variance(a, ddof: i64) -> f64, Fault {
    let n = a.data.len as i64;
    if ddof < 0 || n - ddof < 1 {
        return 0.0, fault(F_VALUE, "var", "the ddof leaves no positive divisor");
    }
    var m: f64 = 0.0;
    var i: i64 = 0;
    while i < a.data.len { m += a.data[i]; i += 1; }
    m /= n as f64;
    var t: f64 = 0.0;
    i = 0;
    while i < a.data.len { let e = a.data[i] - m; t += e * e; i += 1; }
    return t / ((n - ddof) as f64), no_fault();
}

fn std(a, ddof: i64) -> f64, Fault {
    let v, f = variance(a, ddof);
    if f.kind != F_NONE { return 0.0, f; }
    return mathshim::root(v), no_fault();
}

// ptp gives the span from the smallest to the largest element.
fn ptp(a) -> f64, Fault {
    let lo, f = min(a);
    let hi, f2 = max(a);
    return hi - lo, no_fault();
}
