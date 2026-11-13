// The assembly of shapes: the swap of the axes, the permutation, the growth, the cut and the padding.
//
// The assembly of shapes: the swap and the permutation of axes, the
// arrays of a least dimension, the growth and the cut of an array, the
// padding, the diagonals, the grids of indices and the tables of
// positions. Each function follows the rule of NumPy for the shape of
// its result.
namespace scientific;

// swapaxes swaps two axes. A negative axis counts from the end.
fn swapaxes(a, axis1: i64, axis2: i64) -> Arr, Fault {
    let x = fix_axis(axis1, a.ndim);
    let y = fix_axis(axis2, a.ndim);
    if x < 0 || y < 0 {
        return empty_arr(), fault(F_RANGE, "swapaxes", "an axis stands outside the shape");
    }
    var perm: i64[..8] = [];
    var i: i64 = 0;
    while i < a.ndim { perm.push(i); i += 1; }
    perm[x] = y;
    perm[y] = x;
    return transpose_perm(a, perm);
}

// permute_dims reorders the axes. A negative axis counts from the end.
fn permute_dims(a, axes: i64[:]) -> Arr, Fault {
    if axes.len != a.ndim {
        return empty_arr(), fault(F_SHAPE, "permute_dims", "the permutation holds a different count than the shape");
    }
    var perm: i64[..8] = [];
    var i: i64 = 0;
    while i < axes.len {
        let ax = fix_axis(axes[i], a.ndim);
        if ax < 0 {
            return empty_arr(), fault(F_RANGE, "permute_dims", "an axis stands outside the shape");
        }
        perm.push(ax);
        i += 1;
    }
    return transpose_perm(a, perm);
}

// atleast_1d gives an array of one dimension or more. Every array of
// the package holds one dimension at least, so the result is a copy.
fn atleast_1d(a) -> Arr, Fault { return clone(a); }

// atleast_2d turns an array of one dimension into one row.
fn atleast_2d(a) -> Arr, Fault {
    if a.ndim >= 2 { return clone(a); }
    return reshape(a, [1, a.shape[0]]);
}

// atleast_3d turns an array of one dimension into the shape 1, N, 1
// and an array of two dimensions into the shape M, N, 1.
fn atleast_3d(a) -> Arr, Fault {
    if a.ndim >= 3 { return clone(a); }
    if a.ndim == 1 { return reshape(a, [1, a.shape[0], 1]); }
    return reshape(a, [a.shape[0], a.shape[1], 1]);
}

// append joins the elements of the second array after the elements of
// the first, both flattened. It is append of NumPy with no axis.
fn append(a, b) -> Arr, Fault {
    let x, fx = ravel(a);
    if fx.kind != F_NONE { return empty_arr(), fx; }
    let y, fy = ravel(b);
    if fy.kind != F_NONE { return empty_arr(), fy; }
    return concatenate(0, x, y);
}

// append_axis joins two arrays along an axis. It is append of NumPy
// with an axis.
fn append_axis(a, b, axis: i64) -> Arr, Fault {
    let ax = fix_axis(axis, a.ndim);
    if ax < 0 {
        return empty_arr(), fault(F_RANGE, "append_axis", "the axis stands outside the shape");
    }
    return concatenate(ax, a, b);
}

// delete_axis drops the entries of an axis at the places of the list.
// A negative place counts from the end. A repeated place counts once.
// A result with no element is a fault.
fn delete_axis(a, places: i64[:], axis: i64) -> Arr, Fault {
    let ax = fix_axis(axis, a.ndim);
    if ax < 0 {
        return empty_arr(), fault(F_RANGE, "delete", "the axis stands outside the shape");
    }
    let width = a.shape[ax];
    var drop: f64[>..] = [];
    var i: i64 = 0;
    while i < width { drop.push(0.0); i += 1; }
    i = 0;
    while i < places.len {
        var p = places[i];
        if p < 0 { p += width; }
        if p < 0 || p >= width {
            return empty_arr(), fault(F_RANGE, "delete", "a place stands outside the axis");
        }
        drop[p] = 1.0;
        i += 1;
    }
    var kept: i64 = 0;
    i = 0;
    while i < width { if drop[i] == 0.0 { kept += 1; } i += 1; }
    if kept == 0 {
        return empty_arr(), fault(F_VALUE, "delete", "the result holds no element");
    }
    let inner = product_after_of(a.shape, ax);
    let outer = product_before_of(a.shape, ax);
    var d: f64[>..] = [];
    var o: i64 = 0;
    while o < outer {
        var k: i64 = 0;
        while k < width {
            if drop[k] == 0.0 {
                var j: i64 = 0;
                while j < inner { d.push(a.data[o * width * inner + k * inner + j]); j += 1; }
            }
            k += 1;
        }
        o += 1;
    }
    var s: i64[..8] = [];
    i = 0;
    while i < a.ndim { s.push(a.shape[i]); i += 1; }
    s[ax] = kept;
    return Arr { ndim: a.ndim, shape: s, data: d }, no_fault();
}

// delete drops elements of the flattened array. It is delete of NumPy
// with no axis.
fn delete(a, places: i64[:]) -> Arr, Fault {
    let x, f = ravel(a);
    if f.kind != F_NONE { return empty_arr(), f; }
    return delete_axis(x, places, 0);
}

// resize gives an array of a new shape. The elements repeat in their
// order until the shape is full, as NumPy repeats them.
fn resize(a, shape: i64[:]) -> Arr, Fault {
    let f = check_shape(shape);
    if f.kind != F_NONE { return empty_arr(), f; }
    let n = numel(shape);
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < n { d.push(a.data[i % a.data.len]); i += 1; }
    var s: i64[..8] = [];
    i = 0;
    while i < shape.len { s.push(shape[i]); i += 1; }
    return Arr { ndim: shape.len, shape: s, data: d }, no_fault();
}

// pad_source gives the place of the source for a place of the result
// along one axis. The value -1 means the constant value.
fn pad_source(k: i64, n: i64, mode: const u8[:]) -> i64 {
    if k >= 0 && k < n { return k; }
    if mode == "constant" || mode == "empty" || pad_is_stat(mode) { return -1; }
    if mode == "edge" {
        if k < 0 { return 0; }
        return n - 1;
    }
    if n == 1 { return 0; }
    var period = n;
    if mode == "reflect" { period = 2 * (n - 1); }
    if mode == "symmetric" { period = 2 * n; }
    var r = k % period;
    if r < 0 { r += period; }
    if mode == "wrap" { return r; }
    if r >= n {
        if mode == "reflect" { return period - r; }
        return period - 1 - r;
    }
    return r;
}

// pad_is_stat tells whether a mode fills the pad from the line itself.
fn pad_is_stat(mode: const u8[:]) -> bool {
    return mode == "linear_ramp" || mode == "maximum" || mode == "minimum" || mode == "mean" || mode == "median";
}

// pad_stat gives the value of one pad place of the statistic modes. The
// position p lies outside the line: below zero before the line, and at
// the width or above after it. The statistic takes the len values
// nearest to that end of the line. A len above the width clips to the
// width, and a len of zero takes the whole line. The value is the end
// value of the ramp for linear_ramp.
fn pad_stat(a, base: i64, width: i64, inner: i64, p: i64, before: i64, after: i64, mode: const u8[:], end: f64, len: i64) -> f64 {
    let low = p < 0;
    var edge: f64 = 0.0;
    if low { edge = a.data[base]; } else { edge = a.data[base + (width - 1) * inner]; }
    if mode == "linear_ramp" {
        if low { return end + (edge - end) * (before + p) / before; }
        let q = p - width;
        return end + (edge - end) * (after - 1 - q) / after;
    }
    var cnt = width;
    if len > 0 && len < width { cnt = len; }
    var vals: f64[>..] = [];
    var i: i64 = 0;
    while i < cnt {
        if low { vals.push(a.data[base + i * inner]); }
        else { vals.push(a.data[base + (width - cnt + i) * inner]); }
        i += 1;
    }
    if mode == "maximum" || mode == "minimum" {
        var best = vals[0];
        i = 1;
        while i < cnt {
            if mode == "maximum" && vals[i] > best { best = vals[i]; }
            if mode == "minimum" && vals[i] < best { best = vals[i]; }
            i += 1;
        }
        return best;
    }
    if mode == "mean" {
        var t: f64 = 0.0;
        i = 0;
        while i < cnt { t += vals[i]; i += 1; }
        return t / cnt;
    }
    // The median sorts a copy by insertion.
    i = 1;
    while i < cnt {
        let v = vals[i];
        var j = i - 1;
        while j >= 0 && vals[j] > v { vals[j + 1] = vals[j]; j -= 1; }
        vals[j + 1] = v;
        i += 1;
    }
    if cnt % 2 == 1 { return vals[cnt / 2]; }
    return (vals[cnt / 2 - 1] + vals[cnt / 2]) / 2.0;
}

// pad_one grows one axis by the counts before and after. Each side holds
// its own statistic length, end value and constant. The mode empty
// fills the pad with zero, so no value of the result is undefined.
fn pad_one(a, axis: i64, before: i64, after: i64, mode: const u8[:], len_b: i64, len_a: i64, end_b: f64, end_a: f64, const_b: f64, const_a: f64) -> Arr, Fault {
    let width = a.shape[axis];
    let inner = product_after_of(a.shape, axis);
    let outer = product_before_of(a.shape, axis);
    // The tests below never form a product that can overflow. Each
    // count meets the limit first, and the product is a quotient.
    if before > MAX_ELEMENTS || after > MAX_ELEMENTS {
        return empty_arr(), fault(F_LIMIT, "pad", "the result holds more elements than the limit");
    }
    let new_width = width + before + after;
    if new_width > MAX_ELEMENTS / (inner * outer) {
        return empty_arr(), fault(F_LIMIT, "pad", "the result holds more elements than the limit");
    }
    var d: f64[>..] = [];
    var o: i64 = 0;
    while o < outer {
        var k: i64 = 0;
        while k < new_width {
            let p = k - before;
            let src = pad_source(p, width, mode);
            let low = p < 0;
            var j: i64 = 0;
            while j < inner {
                if src < 0 && pad_is_stat(mode) {
                    var e = end_a;
                    var n = len_a;
                    if low { e = end_b; n = len_b; }
                    d.push(pad_stat(a, o * width * inner + j, width, inner, p, before, after, mode, e, n));
                }
                else if src < 0 && mode == "empty" { d.push(0.0); }
                else if src < 0 {
                    if low { d.push(const_b); } else { d.push(const_a); }
                }
                else { d.push(a.data[o * width * inner + src * inner + j]); }
                j += 1;
            }
            k += 1;
        }
        o += 1;
    }
    var s: i64[..8] = [];
    var i: i64 = 0;
    while i < a.ndim { s.push(a.shape[i]); i += 1; }
    s[axis] = new_width;
    return Arr { ndim: a.ndim, shape: s, data: d }, no_fault();
}

// pad_expand_i turns a list of whole numbers into one pair for each
// axis. The list holds no value (the pair is fill), one value for every
// side, one pair for every axis, or one pair for each axis. Any other
// count gives false.
fn pad_expand_i(v: i64[:], ndim: i64, fill: i64, out: i64[>..]&) -> bool {
    out.clear();
    var i: i64 = 0;
    while i < 2 * ndim {
        if v.len == 0 { out.push(fill); }
        else if v.len == 1 { out.push(v[0]); }
        else if v.len == 2 { out.push(v[i % 2]); }
        else if v.len == 2 * ndim { out.push(v[i]); }
        else { return false; }
        i += 1;
    }
    return true;
}

// pad_expand_f does the same for a list of real numbers.
fn pad_expand_f(v: f64[:], ndim: i64, out: f64[>..]&) -> bool {
    out.clear();
    var i: i64 = 0;
    while i < 2 * ndim {
        if v.len == 0 { out.push(0.0); }
        else if v.len == 1 { out.push(v[0]); }
        else if v.len == 2 { out.push(v[i % 2]); }
        else if v.len == 2 * ndim { out.push(v[i]); }
        else { return false; }
        i += 1;
    }
    return true;
}

// pad_axes grows every axis (the name pad is a keyword of Goose). The
// lists before and after hold one count per axis. The mode is
// "constant", "edge", "reflect", "symmetric", "wrap", "linear_ramp",
// "maximum", "minimum", "mean", "median" or "empty". The lists
// stat_length, end_values and constant_values hold no value, one
// value, one pair (before, after) or one pair for each axis. The pair
// of each axis comes in turn: (before0, after0, before1, after1, ...).
// The statistic modes take the stat_length values nearest the edge. An
// empty stat_length takes the whole line, and a length above the width
// clips to it. The end values end the ramps of linear_ramp. The
// constants fill the pad of constant. The mode empty fills the pad
// with zero, where NumPy leaves it undefined. A value that is not
// finite passes into the pad. A count below zero or a statistic length
// below one is a fault.
fn pad_axes(a, before: i64[:], after: i64[:], mode: const u8[:] = "constant", stat_length: i64[:] = [], end_values: f64[:] = [], constant_values: f64[:] = []) -> Arr, Fault {
    let fs = check_stack(a.ndim, a.shape, a.data.len, 1, "pad");
    if fs.kind != F_NONE { return empty_arr(), fs; }
    if before.len != a.ndim || after.len != a.ndim {
        return empty_arr(), fault(F_SHAPE, "pad", "the counts do not hold one value per axis");
    }
    var sl: i64[>..] = [];
    var ev: f64[>..] = [];
    var cv: f64[>..] = [];
    if !pad_expand_i(stat_length, a.ndim, 0, sl) || !pad_expand_f(end_values, a.ndim, ev) || !pad_expand_f(constant_values, a.ndim, cv) {
        return empty_arr(), fault(F_SHAPE, "pad", "a list holds a count of values that is not zero, one, two or two for each axis");
    }
    if mode != "constant" && mode != "edge" && mode != "reflect" && mode != "symmetric" && mode != "wrap" && mode != "empty" && !pad_is_stat(mode) {
        return empty_arr(), fault(F_VALUE, "pad", "the mode holds an unknown name");
    }
    var ax: i64 = 0;
    while ax < a.ndim {
        if before[ax] < 0 || after[ax] < 0 {
            return empty_arr(), fault(F_VALUE, "pad", "a count of growth holds a value below zero");
        }
        ax += 1;
    }
    if stat_length.len > 0 {
        var i: i64 = 0;
        while i < sl.len {
            if sl[i] < 1 {
                return empty_arr(), fault(F_VALUE, "pad", "the statistic length holds a value below one");
            }
            i += 1;
        }
    }
    var cur = clone_of(a);
    ax = 0;
    while ax < a.ndim {
        let r, f = pad_one(cur, ax, before[ax], after[ax], mode, sl[2 * ax], sl[2 * ax + 1], ev[2 * ax], ev[2 * ax + 1], cv[2 * ax], cv[2 * ax + 1]);
        if f.kind != F_NONE { return empty_arr(), f; }
        cur = copy(r);
        ax += 1;
    }
    return cur, no_fault();
}

// pad_modes is pad_axes with one value for every axis and side. The
// value fills a constant pad and ends a ramp. The stat_length of zero
// takes the whole line, and a length above the width clips to it. A
// count below zero, or a stat_length below zero, is a fault.
fn pad_modes(a, before: i64[:], after: i64[:], mode: const u8[:] = "constant", value: f64 = 0.0, stat_length: i64 = 0) -> Arr, Fault {
    if stat_length == 0 { return pad_axes(a, before, after, mode, [], [value], [value]); }
    return pad_axes(a, before, after, mode, [stat_length], [value], [value]);
}

fn clone_of(a) -> Arr {
    let c, f = clone(a);
    return c;
}

