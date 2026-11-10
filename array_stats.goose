// The statistics of NumPy that skip NaN, the running forms of the
// array API, the covariance and the correlation.
namespace scientific;

import mathshim;

// nan_arg gives the flat position of the largest or the smallest value
// that is not NaN. The first of equal values wins. An array of only NaN
// is a fault, as NumPy refuses it.
fn nan_arg(a, want_max: bool, tag: const u8[:]) -> i64, Fault {
    var best: i64 = -1;
    var i: i64 = 0;
    while i < a.data.len {
        let x = a.data[i];
        if x == x {
            if best < 0 { best = i; }
            else if want_max && x > a.data[best] { best = i; }
            else if !want_max && x < a.data[best] { best = i; }
        }
        i += 1;
    }
    if best < 0 { return -1, fault(F_VALUE, tag, "the array holds only NaN"); }
    return best, no_fault();
}

// nanargmax gives the flat position of the largest value that is not
// NaN.
fn nanargmax(a) -> i64, Fault { return nan_arg(a, true, "nanargmax"); }

// nanargmin gives the flat position of the smallest value that is not
// NaN.
fn nanargmin(a) -> i64, Fault { return nan_arg(a, false, "nanargmin"); }

// nan_run runs a running total along an axis. A NaN counts as the
// neutral value: zero for the sum and one for the product.
fn nan_run(a, axis: i64, rule: i32) -> Arr, Fault {
    var d: f64[>..] = [];
    var neutral: f64 = 0.0;
    if rule == 1 { neutral = 1.0; }
    var i: i64 = 0;
    while i < a.data.len {
        let x = a.data[i];
        if x == x { d.push(x); } else { d.push(neutral); }
        i += 1;
    }
    let b = Arr { ndim: a.ndim, shape: same_shape(a), data: copy(d) };
    return cum_axis(b, axis, rule);
}

// nancumsum gives the running sum along an axis. A NaN counts as zero.
fn nancumsum(a, axis: i64) -> Arr, Fault { return nan_run(a, axis, 0); }

// nancumprod gives the running product along an axis. A NaN counts as
// one.
fn nancumprod(a, axis: i64) -> Arr, Fault { return nan_run(a, axis, 1); }

// cumulative runs a running total of the array API. With
// include_initial the result starts with the neutral value, so the axis
// grows by one.
fn cumulative(a, axis: i64, rule: i32, include_initial: bool) -> Arr, Fault {
    let c, f = cum_axis(a, axis, rule);
    if f.kind != F_NONE { return empty_arr(), f; }
    if !include_initial { return c, no_fault(); }
    let width = a.shape[axis];
    let inner = product_after_of(a.shape, axis);
    let outer = product_before_of(a.shape, axis);
    var init: f64 = 0.0;
    if rule == 1 { init = 1.0; }
    var d: f64[>..] = [];
    var o: i64 = 0;
    while o < outer {
        var k: i64 = 0;
        while k <= width {
            var j: i64 = 0;
            while j < inner {
                if k == 0 { d.push(init); }
                else { d.push(c.data[o * width * inner + (k - 1) * inner + j]); }
                j += 1;
            }
            k += 1;
        }
        o += 1;
    }
    var s = same_shape(a);
    s[axis] = width + 1;
    if numel(s) > MAX_ELEMENTS {
        return empty_arr(), fault(F_LIMIT, "cumulative", "the result holds more elements than the limit");
    }
    return Arr { ndim: a.ndim, shape: s, data: d }, no_fault();
}

// cumulative_sum gives the running sum along an axis. It is the name of
// the array API, and it can start with a zero.
fn cumulative_sum(a, axis: i64, include_initial: bool = false) -> Arr, Fault {
    return cumulative(a, axis, 0, include_initial);
}

// cumulative_prod gives the running product along an axis. It can start
// with a one.
fn cumulative_prod(a, axis: i64, include_initial: bool = false) -> Arr, Fault {
    return cumulative(a, axis, 1, include_initial);
}

// ediff1d gives the differences of the neighbours of the flat array.
// The values of to_begin come first and the values of to_end come last.
// A result with no element is a fault.
fn ediff1d(a, to_begin: f64[:] = [], to_end: f64[:] = []) -> Arr, Fault {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < to_begin.len { d.push(to_begin[i]); i += 1; }
    i = 1;
    while i < a.data.len { d.push(a.data[i] - a.data[i - 1]); i += 1; }
    i = 0;
    while i < to_end.len { d.push(to_end[i]); i += 1; }
    if d.len == 0 {
        return empty_arr(), fault(F_SHAPE, "ediff1d", "the result holds no element");
    }
    var s: i64[..8] = [];
    s.push(d.len);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}

// nan_values copies the values that are not NaN into a flat array.
fn nan_values(a, out: f64[>..]&) {
    var i: i64 = 0;
    while i < a.data.len {
        if a.data[i] == a.data[i] { out.push(a.data[i]); }
        i += 1;
    }
}

// nanquantile gives the quantile of the values that are not NaN. An
// array of only NaN gives NaN, as NumPy gives it.
fn nanquantile(a, q: f64, method: const u8[:] = "linear") -> f64, Fault {
    var v: f64[>..] = [];
    nan_values(a, v);
    if v.len == 0 {
        let m, f = check_quantile(q, method, "nanquantile");
        if f.kind != F_NONE { return 0.0, f; }
        return 0.0 / 0.0, no_fault();
    }
    var s: i64[..8] = [];
    s.push(v.len);
    let b = Arr { ndim: 1, shape: s, data: copy(v) };
    return quantile(b, q, method);
}

// nanpercentile gives the percentile of the values that are not NaN.
fn nanpercentile(a, q: f64, method: const u8[:] = "linear") -> f64, Fault {
    if !(q >= 0.0 && q <= 100.0) {
        return 0.0, fault(F_VALUE, "nanpercentile", "the percentage lies outside the range");
    }
    return nanquantile(a, q / 100.0, method);
}

// nanmedian gives the median of the values that are not NaN. An array
// of only NaN gives NaN.
fn nanmedian(a) -> f64, Fault { return nanquantile(a, 0.5); }

// histogram_bin_edges gives the edges of the bins of a histogram with
// equal width. It is the name of NumPy for histogram_edges.
fn histogram_bin_edges(a, bins: i64) -> Arr, Fault { return histogram_edges(a, bins); }

// vars_of reads a matrix as variables and observations. With rowvar
// each row holds one variable. Otherwise each column holds one. A
// vector holds one variable.
fn vars_of(m, rowvar: bool, tag: const u8[:], x: f64[>..]&, nvar: i64&, nobs: i64&) -> Fault {
    if m.ndim > 2 {
        return fault(F_SHAPE, tag, "the array holds more than two dimensions");
    }
    if m.ndim == 1 {
        nvar = 1;
        nobs = m.data.len;
        var i: i64 = 0;
        while i < m.data.len { x.push(m.data[i]); i += 1; }
        return no_fault();
    }
    var rows = m.shape[0];
    var cols = m.shape[1];
    if rowvar {
        nvar = rows;
        nobs = cols;
        var i: i64 = 0;
        while i < m.data.len { x.push(m.data[i]); i += 1; }
    } else {
        nvar = cols;
        nobs = rows;
        var v: i64 = 0;
        while v < cols {
            var k: i64 = 0;
            while k < rows { x.push(m.data[k * cols + v]); k += 1; }
            v += 1;
        }
    }
    return no_fault();
}

// cov_of gives the covariance matrix of variables in rows. The divisor
// is the count of observations less ddof. A divisor below zero counts
// as zero, so the result holds the infinity or NaN, as NumPy gives it.
fn cov_of(x: f64[:], nvar: i64, nobs: i64, ddof: i64) -> Arr {
    var mean: f64[>..] = [];
    var v: i64 = 0;
    while v < nvar {
        var s: f64 = 0.0;
        var k: i64 = 0;
        while k < nobs { s += x[v * nobs + k]; k += 1; }
        mean.push(s / nobs);
        v += 1;
    }
    var fact = (nobs - ddof) as f64;
    if fact < 0.0 { fact = 0.0; }
    var d: f64[>..] = [];
    v = 0;
    while v < nvar {
        var w: i64 = 0;
        while w < nvar {
            var s: f64 = 0.0;
            var k: i64 = 0;
            while k < nobs {
                s += (x[v * nobs + k] - mean[v]) * (x[w * nobs + k] - mean[w]);
                k += 1;
            }
            d.push(s / fact);
            w += 1;
        }
        v += 1;
    }
    var sh: i64[..8] = [];
    sh.push(nvar);
    sh.push(nvar);
    return Arr { ndim: 2, shape: sh, data: d };
}

// cov gives the covariance matrix of the variables. With rowvar each
// row holds one variable. The divisor is the count of observations
// less ddof. A vector gives a matrix of one row and one column.
fn cov(m, rowvar: bool = true, ddof: i64 = 1) -> Arr, Fault {
    var x: f64[>..] = [];
    var nvar: i64 = 0;
    var nobs: i64 = 0;
    let f = vars_of(m, rowvar, "cov", x, nvar, nobs);
    if f.kind != F_NONE { return empty_arr(), f; }
    return cov_of(x, nvar, nobs, ddof), no_fault();
}

// stack_vars joins the variables of two arrays, as the call of NumPy
// with a second array does. Both must hold the same count of
// observations.
fn stack_vars(m, y, rowvar: bool, tag: const u8[:], x: f64[>..]&, nvar: i64&, nobs: i64&) -> Fault {
    var ny: i64 = 0;
    var oy: i64 = 0;
    var yd: f64[>..] = [];
    let f1 = vars_of(m, rowvar, tag, x, nvar, nobs);
    if f1.kind != F_NONE { return f1; }
    let f2 = vars_of(y, rowvar, tag, yd, ny, oy);
    if f2.kind != F_NONE { return f2; }
    if oy != nobs {
        return fault(F_SHAPE, tag, "the two arrays hold a different count of observations");
    }
    var i: i64 = 0;
    while i < yd.len { x.push(yd[i]); i += 1; }
    nvar = nvar + ny;
    return no_fault();
}

// cov_with gives the covariance matrix of the variables of m followed
// by the variables of y.
fn cov_with(m, y, rowvar: bool = true, ddof: i64 = 1) -> Arr, Fault {
    var x: f64[>..] = [];
    var nvar: i64 = 0;
    var nobs: i64 = 0;
    let f = stack_vars(m, y, rowvar, "cov_with", x, nvar, nobs);
    if f.kind != F_NONE { return empty_arr(), f; }
    return cov_of(x, nvar, nobs, ddof), no_fault();
}

// corr_of turns a covariance matrix into correlations. Each value
// clips to the range of minus one to one, as NumPy clips it.
fn corr_of(c: Arr&) {
    let n = c.shape[0];
    var sd: f64[>..] = [];
    var i: i64 = 0;
    while i < n { sd.push(mathshim::root(c.data[i * n + i])); i += 1; }
    i = 0;
    while i < n {
        var j: i64 = 0;
        while j < n {
            var v = c.data[i * n + j] / sd[i] / sd[j];
            if v > 1.0 { v = 1.0; }
            if v < -1.0 { v = -1.0; }
            c.data[i * n + j] = v;
            j += 1;
        }
        i += 1;
    }
}

// corrcoef gives the matrix of the correlation coefficients of the
// variables. A variable that holds no spread gives NaN.
fn corrcoef(m, rowvar: bool = true) -> Arr, Fault {
    var x: f64[>..] = [];
    var nvar: i64 = 0;
    var nobs: i64 = 0;
    let f = vars_of(m, rowvar, "corrcoef", x, nvar, nobs);
    if f.kind != F_NONE { return empty_arr(), f; }
    var c = cov_of(x, nvar, nobs, 1);
    corr_of(c);
    return c, no_fault();
}

// corrcoef_with gives the correlation coefficients of the variables of
// m followed by the variables of y.
fn corrcoef_with(m, y, rowvar: bool = true) -> Arr, Fault {
    var x: f64[>..] = [];
    var nvar: i64 = 0;
    var nobs: i64 = 0;
    let f = stack_vars(m, y, rowvar, "corrcoef_with", x, nvar, nobs);
    if f.kind != F_NONE { return empty_arr(), f; }
    var c = cov_of(x, nvar, nobs, 1);
    corr_of(c);
    return c, no_fault();
}
