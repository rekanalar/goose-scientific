// The reductions that skip NaN, and the function that replaces it.
//
// A NaN counts as a gap. Each reduction works on the values that are
// not NaN. A fault comes from an axis or a ddof, never from a value.
namespace scientific;

import mathshim;

// nan_cell reduces one line of the buffer. The line starts at base and
// holds width values that stand step apart. The rule picks the
// operation: 0 sum, 1 prod, 2 min, 3 max, 4 mean, 5 var, 6 std. A
// line with no value gives NaN for the rules that need one.
fn nan_cell(a, base: i64, width: i64, step: i64, rule: i32, ddof: i64) -> f64 {
    var n: i64 = 0;
    var s: f64 = 0.0;
    var p: f64 = 1.0;
    var lo: f64 = 0.0;
    var hi: f64 = 0.0;
    var k: i64 = 0;
    while k < width {
        let x = a.data[base + k * step];
        if x == x {
            if n == 0 || x < lo { lo = x; }
            if n == 0 || x > hi { hi = x; }
            s += x;
            p *= x;
            n += 1;
        }
        k += 1;
    }
    if rule == 0 { return s; }
    if rule == 1 { return p; }
    if n == 0 { return 0.0 / 0.0; }
    if rule == 2 { return lo; }
    if rule == 3 { return hi; }
    let m = s / (n as f64);
    if rule == 4 { return m; }
    if n - ddof < 1 { return 0.0 / 0.0; }
    var t: f64 = 0.0;
    k = 0;
    while k < width {
        let x = a.data[base + k * step];
        if x == x { let e = x - m; t += e * e; }
        k += 1;
    }
    let v = t / ((n - ddof) as f64);
    if rule == 5 { return v; }
    return mathshim::root(v);
}

// nan_whole reduces the whole array to one number.
fn nan_whole(a, rule: i32, ddof: i64, name: const u8[:]) -> f64, Fault {
    if a.data.len == 0 {
        return 0.0, fault(F_VALUE, name, "the array holds no element");
    }
    if ddof < 0 {
        return 0.0, fault(F_VALUE, name, "the ddof holds a value below zero");
    }
    return nan_cell(a, 0, a.data.len, 1, rule, ddof), no_fault();
}

// nan_axis reduces the array along one axis. The axis drops from the
// shape, as in NumPy.
fn nan_axis(a, axis: i64, rule: i32, ddof: i64, name: const u8[:]) -> Arr, Fault {
    var s: i64[>..] = [];
    let sf = shape_without(a.shape, axis, s);
    if sf.kind != F_NONE { return empty_arr(), sf; }
    if ddof < 0 {
        return empty_arr(), fault(F_VALUE, name, "the ddof holds a value below zero");
    }
    let inner = product_after_of(a.shape, axis);
    let width = a.shape[axis];
    let outer = product_before_of(a.shape, axis);
    var d: f64[>..] = [];
    var cell: i64 = 0;
    while cell < outer * inner {
        let base = (cell / inner) * (width * inner) + cell % inner;
        d.push(nan_cell(a, base, width, inner, rule, ddof));
        cell += 1;
    }
    var sc: i64[..8] = [];
    var i: i64 = 0;
    while i < s.len { sc.push(s[i]); i += 1; }
    return Arr { ndim: sc.len, shape: sc, data: d }, no_fault();
}

// nansum adds the values that are not NaN.
fn nansum(a) -> f64, Fault { return nan_whole(a, 0, 0, "nansum"); }

// nanprod multiplies the values that are not NaN.
fn nanprod(a) -> f64, Fault { return nan_whole(a, 1, 0, "nanprod"); }

// nanmin gives the smallest value that is not NaN.
fn nanmin(a) -> f64, Fault { return nan_whole(a, 2, 0, "nanmin"); }

// nanmax gives the largest value that is not NaN.
fn nanmax(a) -> f64, Fault { return nan_whole(a, 3, 0, "nanmax"); }

// nanmean averages the values that are not NaN.
fn nanmean(a) -> f64, Fault { return nan_whole(a, 4, 0, "nanmean"); }

// nanvar gives the variance of the values that are not NaN. The
// divisor is their count less ddof.
fn nanvar(a, ddof: i64 = 0) -> f64, Fault { return nan_whole(a, 5, ddof, "nanvar"); }

// nanstd gives the root of nanvar.
fn nanstd(a, ddof: i64 = 0) -> f64, Fault { return nan_whole(a, 6, ddof, "nanstd"); }

fn nansum_axis(a, axis: i64) -> Arr, Fault { return nan_axis(a, axis, 0, 0, "nansum"); }
fn nanprod_axis(a, axis: i64) -> Arr, Fault { return nan_axis(a, axis, 1, 0, "nanprod"); }
fn nanmin_axis(a, axis: i64) -> Arr, Fault { return nan_axis(a, axis, 2, 0, "nanmin"); }
fn nanmax_axis(a, axis: i64) -> Arr, Fault { return nan_axis(a, axis, 3, 0, "nanmax"); }
fn nanmean_axis(a, axis: i64) -> Arr, Fault { return nan_axis(a, axis, 4, 0, "nanmean"); }
fn nanvar_axis(a, axis: i64, ddof: i64 = 0) -> Arr, Fault { return nan_axis(a, axis, 5, ddof, "nanvar"); }
fn nanstd_axis(a, axis: i64, ddof: i64 = 0) -> Arr, Fault { return nan_axis(a, axis, 6, ddof, "nanstd"); }

// nan_to_num replaces NaN and the two infinities. The defaults are
// zero, the largest double and the least double, as in NumPy.
fn nan_to_num(a, nan: f64 = 0.0, posinf: f64 = 1.7976931348623157e308, neginf: f64 = -1.7976931348623157e308) -> Arr, Fault {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < a.data.len {
        let x = a.data[i];
        if x != x { d.push(nan); }
        else if mathshim::is_infinite_of(x) == 1.0 {
            if x > 0.0 { d.push(posinf); } else { d.push(neginf); }
        }
        else { d.push(x); }
        i += 1;
    }
    return Arr { ndim: a.ndim, shape: same_shape(a), data: d }, no_fault();
}
