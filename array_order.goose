// The order statistics: the median, the quantiles, the mode and the
// weighted mean.
//
// A quantile sorts a copy of the values. A NaN among the values gives
// NaN, as NumPy gives it.
namespace scientific;

import mathshim;

// method_code gives the number of a quantile method, or -1 for a name
// that no method holds. The names are the names of NumPy.
fn method_code(name: const u8[:]) -> i64 {
    if name == "linear" { return 0; }
    if name == "lower" { return 1; }
    if name == "higher" { return 2; }
    if name == "nearest" { return 3; }
    if name == "midpoint" { return 4; }
    if name == "inverted_cdf" { return 5; }
    if name == "averaged_inverted_cdf" { return 6; }
    if name == "closest_observation" { return 7; }
    if name == "interpolated_inverted_cdf" { return 8; }
    if name == "hazen" { return 9; }
    if name == "weibull" { return 10; }
    if name == "median_unbiased" { return 11; }
    if name == "normal_unbiased" { return 12; }
    return -1;
}

// lerp gives the value between a and b at the fraction t. The two
// forms keep the result exact at both ends, as NumPy does.
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    if t >= 0.5 { return b - (b - a) * (1.0 - t); }
    return a + (b - a) * t;
}

// quantile_sorted gives the quantile q of sorted values. The method
// code picks the rule for the virtual index.
fn quantile_sorted(v: f64[:], q: f64, m: i64) -> f64 {
    let n = v.len as f64;
    var vi: f64 = 0.0;
    var prev: f64 = 0.0;
    var gamma: f64 = 0.0;
    if m == 0 || m == 1 || m == 2 || m == 3 || m == 4 { vi = (n - 1.0) * q; }
    else if m == 5 || m == 6 { vi = n * q - 1.0; }
    else if m == 7 { vi = n * q - 1.5; }
    else {
        var alpha: f64 = 0.0;
        var beta: f64 = 0.0;
        if m == 8 { beta = 1.0; }
        else if m == 9 { alpha = 0.5; beta = 0.5; }
        else if m == 11 { alpha = 1.0 / 3.0; beta = 1.0 / 3.0; }
        else if m == 12 { alpha = 0.375; beta = 0.375; }
        vi = n * q + (alpha + q * (1.0 - alpha - beta)) - 1.0;
    }
    prev = mathshim::floor_of(vi);
    gamma = vi - prev;
    if m == 1 { gamma = 0.0; }
    else if m == 2 { if gamma > 0.0 { prev += 1.0; } gamma = 0.0; }
    else if m == 3 { prev = mathshim::round_of(vi); gamma = 0.0; }
    else if m == 4 { if gamma != 0.0 { gamma = 0.5; } }
    else if m == 5 { if gamma > 0.0 { prev += 1.0; } gamma = 0.0; }
    else if m == 6 { if gamma == 0.0 { gamma = 0.5; } else { gamma = 1.0; } }
    else if m == 7 {
        // A tie between two values goes to the one of the odd place.
        var keep = false;
        if gamma == 0.0 { keep = mathshim::mod_of(prev, 2.0) != 0.0; }
        if !keep { prev += 1.0; }
        gamma = 0.0;
       
    }
    var next = prev + 1.0;
    if prev < 0.0 { prev = 0.0; next = 0.0; }
    if prev >= n - 1.0 { prev = n - 1.0; next = n - 1.0; }
    if next > n - 1.0 { next = n - 1.0; }
    let lo = v[prev as i64];
    let hi = v[next as i64];
    if lo == hi { return lo; }
    return lerp(lo, hi, gamma);
}

// quantile_line sorts one line of values and takes its quantile. It
// checks the quantile, the method and the values first.
fn quantile_line(values: f64[:], q: f64, m: i64, name: const u8[:]) -> f64, Fault {
    if values.len == 0 {
        return 0.0, fault(F_VALUE, name, "the array holds no element");
    }
    var i: i64 = 0;
    while i < values.len {
        if values[i] != values[i] { return values[i], no_fault(); }
        i += 1;
    }
    let sv, p, f = sort_values(values);
    if f.kind != F_NONE { return 0.0, f; }
    return quantile_sorted(sv.data, q, m), no_fault();
}

// check_quantile refuses a quantile outside zero to one and a name of
// a method that does not exist.
fn check_quantile(q: f64, method: const u8[:], name: const u8[:]) -> i64, Fault {
    if !(q >= 0.0 && q <= 1.0) {
        return -1, fault(F_VALUE, name, "the quantile lies outside the range");
    }
    let m = method_code(method);
    if m < 0 { return -1, fault(F_VALUE, name, "the method holds an unknown name"); }
    return m, no_fault();
}

// quantile gives the value at a fraction of the sorted values. The
// fraction lies between zero and one.
fn quantile(a, q: f64, method: const u8[:] = "linear") -> f64, Fault {
    let m, f = check_quantile(q, method, "quantile");
    if f.kind != F_NONE { return 0.0, f; }
    return quantile_line(a.data, q, m, "quantile");
}

// percentile gives the value at a percentage. The percentage lies
// between zero and one hundred.
fn percentile(a, q: f64, method: const u8[:] = "linear") -> f64, Fault {
    let m, f = check_quantile(q / 100.0, method, "percentile");
    if f.kind != F_NONE { return 0.0, f; }
    return quantile_line(a.data, q / 100.0, m, "percentile");
}

// median gives the middle value. An even count gives the mean of the
// two middle values, as NumPy does.
fn median(a) -> f64, Fault {
    if a.data.len == 0 { return 0.0, fault(F_VALUE, "median", "the array holds no element"); }
    var i: i64 = 0;
    while i < a.data.len {
        if a.data[i] != a.data[i] { return a.data[i], no_fault(); }
        i += 1;
    }
    let v, p, f = sort_values(a.data);
    if f.kind != F_NONE { return 0.0, f; }
    let n = v.data.len;
    if n % 2 == 1 { return v.data[n / 2], no_fault(); }
    return (v.data[n / 2 - 1] + v.data[n / 2]) / 2.0, no_fault();
}

// order_axis runs one order statistic along one axis. The rule picks
// it: 0 quantile, 1 median. The axis drops from the shape.
fn order_axis(a, axis: i64, q: f64, m: i64, rule: i32, name: const u8[:]) -> Arr, Fault {
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
        var line: f64[>..] = [];
        var k: i64 = 0;
        while k < width { line.push(a.data[base + k * inner]); k += 1; }
        var r: f64 = 0.0;
        var lf = no_fault();
        if rule == 0 { r, lf = quantile_line(line, q, m, name); }
        else {
            var one: i64[..8] = [];
            one.push(width);
            let arr = Arr { ndim: 1, shape: one, data: copy(line) };
            r, lf = median(arr);
        }
        if lf.kind != F_NONE { return empty_arr(), lf; }
        d.push(r);
        cell += 1;
    }
    var sc: i64[..8] = [];
    var i: i64 = 0;
    while i < s.len { sc.push(s[i]); i += 1; }
    return Arr { ndim: sc.len, shape: sc, data: d }, no_fault();
}

// quantile_axis gives the quantile along one axis.
fn quantile_axis(a, q: f64, axis: i64, method: const u8[:] = "linear") -> Arr, Fault {
    let m, f = check_quantile(q, method, "quantile");
    if f.kind != F_NONE { return empty_arr(), f; }
    return order_axis(a, axis, q, m, 0, "quantile");
}

// percentile_axis gives the percentile along one axis.
fn percentile_axis(a, q: f64, axis: i64, method: const u8[:] = "linear") -> Arr, Fault {
    let m, f = check_quantile(q / 100.0, method, "percentile");
    if f.kind != F_NONE { return empty_arr(), f; }
    return order_axis(a, axis, q / 100.0, m, 0, "percentile");
}

// median_axis gives the median along one axis.
fn median_axis(a, axis: i64) -> Arr, Fault {
    return order_axis(a, axis, 0.0, 0, 1, "median");
}

// mode gives the value that comes most often, and its count. Of equal
// counts the smallest value wins, as SciPy gives it. A NaN is not a
// value, so it stays out of the count.
fn mode(a) -> f64, i64, Fault {
    var clean: f64[>..] = [];
    var i: i64 = 0;
    while i < a.data.len {
        if a.data[i] == a.data[i] { clean.push(a.data[i]); }
        i += 1;
    }
    if clean.len == 0 {
        return 0.0, 0, fault(F_VALUE, "mode", "the array holds no value that is not NaN");
    }
    let sv, p, f = sort_values(clean);
    if f.kind != F_NONE { return 0.0, 0, f; }
    var best: f64 = sv.data[0];
    var bestn: i64 = 0;
    var run: i64 = 0;
    i = 0;
    while i < sv.data.len {
        if i > 0 && sv.data[i] == sv.data[i - 1] { run += 1; } else { run = 1; }
        if run > bestn { bestn = run; best = sv.data[i]; }
        i += 1;
    }
    return best, bestn, no_fault();
}

// average gives the mean of the values with a weight for each value.
// The weights have the shape of the array. A sum of weights of zero
// is a fault.
fn average(a, weights) -> f64, Fault {
    if a.data.len == 0 { return 0.0, fault(F_VALUE, "average", "the array holds no element"); }
    if weights.data.len != a.data.len || weights.ndim != a.ndim {
        return 0.0, fault(F_SHAPE, "average", "the weights differ from the array in shape");
    }
    var k: i64 = 0;
    while k < a.ndim {
        if weights.shape[k] != a.shape[k] {
            return 0.0, fault(F_SHAPE, "average", "the weights differ from the array in shape");
        }
        k += 1;
    }
    var sw: f64 = 0.0;
    var t: f64 = 0.0;
    var i: i64 = 0;
    while i < a.data.len {
        sw += weights.data[i];
        t += a.data[i] * weights.data[i];
        i += 1;
    }
    if sw == 0.0 { return 0.0, fault(F_VALUE, "average", "the weights add up to zero"); }
    return t / sw, no_fault();
}

// average_axis gives the weighted mean along one axis. The weights
// hold one value for each position of that axis.
fn average_axis(a, weights, axis: i64) -> Arr, Fault {
    var s: i64[>..] = [];
    let sf = shape_without(a.shape, axis, s);
    if sf.kind != F_NONE { return empty_arr(), sf; }
    let width = a.shape[axis];
    if weights.data.len != width {
        return empty_arr(), fault(F_SHAPE, "average", "the weights differ from the axis in length");
    }
    var sw: f64 = 0.0;
    var k: i64 = 0;
    while k < width { sw += weights.data[k]; k += 1; }
    if sw == 0.0 { return empty_arr(), fault(F_VALUE, "average", "the weights add up to zero"); }
    let inner = product_after_of(a.shape, axis);
    let outer = product_before_of(a.shape, axis);
    var d: f64[>..] = [];
    var cell: i64 = 0;
    while cell < outer * inner {
        let base = (cell / inner) * (width * inner) + cell % inner;
        var t: f64 = 0.0;
        k = 0;
        while k < width { t += a.data[base + k * inner] * weights.data[k]; k += 1; }
        d.push(t / sw);
        cell += 1;
    }
    var sc: i64[..8] = [];
    var i: i64 = 0;
    while i < s.len { sc.push(s[i]); i += 1; }
    return Arr { ndim: sc.len, shape: sc, data: d }, no_fault();
}
