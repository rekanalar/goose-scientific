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

// diagonal gives the diagonal of two axes. The diagonal becomes the
// last axis of the result. The offset moves it above (positive) or
// below (negative) the main one. A diagonal with no element is a fault.
fn diagonal(a, offset: i64 = 0, axis1: i64 = 0, axis2: i64 = 1) -> Arr, Fault {
    if a.ndim < 2 {
        return empty_arr(), fault(F_SHAPE, "diagonal", "the array holds fewer than two dimensions");
    }
    let x = fix_axis(axis1, a.ndim);
    let y = fix_axis(axis2, a.ndim);
    if x < 0 || y < 0 {
        return empty_arr(), fault(F_RANGE, "diagonal", "an axis stands outside the shape");
    }
    if x == y {
        return empty_arr(), fault(F_VALUE, "diagonal", "the two axes are the same");
    }
    let n1 = a.shape[x];
    let n2 = a.shape[y];
    var len = n1;
    if offset >= 0 { len = n2 - offset; if n1 < len { len = n1; } }
    else { len = n1 + offset; if n2 < len { len = n2; } }
    if len <= 0 {
        return empty_arr(), fault(F_VALUE, "diagonal", "the diagonal holds no element");
    }
    var strides: i64[>..] = [];
    strides_of(a.shape, strides);
    var rest: i64[..8] = [];
    var s: i64[..8] = [];
    var i: i64 = 0;
    while i < a.ndim {
        if i != x && i != y { rest.push(i); s.push(a.shape[i]); }
        i += 1;
    }
    s.push(len);
    var rs: i64 = 1;
    i = 0;
    while i < rest.len { rs *= a.shape[rest[i]]; i += 1; }
    var i0: i64 = 0;
    var j0: i64 = 0;
    if offset >= 0 { j0 = offset; } else { i0 = -offset; }
    var d: f64[>..] = [];
    var r: i64 = 0;
    while r < rs {
        // The place of the rest axes from the number r.
        var base: i64 = 0;
        var q = r;
        i = rest.len - 1;
        while i >= 0 {
            base += (q % a.shape[rest[i]]) * strides[rest[i]];
            q = q / a.shape[rest[i]];
            i -= 1;
        }
        var t: i64 = 0;
        while t < len {
            d.push(a.data[base + (i0 + t) * strides[x] + (j0 + t) * strides[y]]);
            t += 1;
        }
        r += 1;
    }
    return Arr { ndim: s.len, shape: s, data: d }, no_fault();
}

// fill_diagonal writes a value on the main diagonal, in place. A matrix
// that is taller than wide stops at the square unless wrap is true. An
// array of more than two dimensions must hold equal lengths.
fn fill_diagonal(a, value: f64, wrap: bool = false) -> Fault {
    if a.ndim < 2 {
        return fault(F_SHAPE, "fill_diagonal", "the array holds fewer than two dimensions");
    }
    var step: i64 = 1;
    var end = a.data.len;
    if a.ndim == 2 {
        step = a.shape[1] + 1;
        if !wrap {
            end = a.shape[1] * a.shape[1];
            if end > a.data.len { end = a.data.len; }
        }
    } else {
        var i: i64 = 1;
        var p: i64 = 1;
        while i < a.ndim {
            if a.shape[i] != a.shape[0] {
                return fault(F_SHAPE, "fill_diagonal", "an array of more than two dimensions needs equal lengths");
            }
            p *= a.shape[0];
            step += p;
            i += 1;
        }
    }
    var pos: i64 = 0;
    while pos < end {
        a.data[pos] = value;
        pos += step;
    }
    return no_fault();
}

fn two_rows(first: f64[:], second: f64[:]) -> Arr {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < first.len { d.push(first[i]); i += 1; }
    i = 0;
    while i < second.len { d.push(second[i]); i += 1; }
    var s: i64[..8] = [];
    s.push(2);
    s.push(first.len);
    return Arr { ndim: 2, shape: s, data: d };
}

// tri gives a matrix of n rows and m columns. The entries on and
// below the diagonal k hold one and the others hold zero.
fn tri(n: i64, m: i64, k: i64 = 0) -> Arr, Fault {
    var shape: i64[..8] = [];
    shape.push(n);
    shape.push(m);
    let f = check_shape(shape);
    if f.kind != F_NONE { return empty_arr(), f; }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        var j: i64 = 0;
        while j < m {
            if j <= i + k { d.push(1.0); } else { d.push(0.0); }
            j += 1;
        }
        i += 1;
    }
    return Arr { ndim: 2, shape: shape, data: d }, no_fault();
}

// tri_places gives the places of the lower (lower is true) or the upper
// triangle of a matrix of n rows and m columns, as two rows: the row
// numbers and the column numbers.
fn tri_places(n: i64, k: i64, m: i64, lower: bool, tag: const u8[:]) -> Arr, Fault {
    var cols = m;
    if cols == 0 { cols = n; }
    var shape: i64[..8] = [];
    shape.push(n);
    shape.push(cols);
    let f = check_shape(shape);
    if f.kind != F_NONE { return empty_arr(), fault(f.kind, tag, f.detail); }
    var rows: f64[>..] = [];
    var cs: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        var j: i64 = 0;
        while j < cols {
            var hit = j >= i + k;
            if lower { hit = j <= i + k; }
            if hit { rows.push(i); cs.push(j); }
            j += 1;
        }
        i += 1;
    }
    if rows.len == 0 {
        return empty_arr(), fault(F_VALUE, tag, "the triangle holds no place");
    }
    return two_rows(rows, cs), no_fault();
}

// tril_indices gives the places of the lower triangle as two rows. The
// count m of columns of zero means a square matrix.
fn tril_indices(n: i64, k: i64 = 0, m: i64 = 0) -> Arr, Fault {
    return tri_places(n, k, m, true, "tril_indices");
}

// triu_indices gives the places of the upper triangle as two rows.
fn triu_indices(n: i64, k: i64 = 0, m: i64 = 0) -> Arr, Fault {
    return tri_places(n, k, m, false, "triu_indices");
}

// diag_indices gives the places of the main diagonal of an array of
// ndim axes of length n. The result holds one row per axis.
fn diag_indices(n: i64, ndim: i64 = 2) -> Arr, Fault {
    if n < 1 || ndim < 1 || ndim > MAX_DIMS {
        return empty_arr(), fault(F_RANGE, "diag_indices", "the length or the count of axes stands outside its range");
    }
    var d: f64[>..] = [];
    var a: i64 = 0;
    while a < ndim {
        var i: i64 = 0;
        while i < n { d.push(i); i += 1; }
        a += 1;
    }
    var s: i64[..8] = [];
    s.push(ndim);
    s.push(n);
    return Arr { ndim: 2, shape: s, data: d }, no_fault();
}

// indices gives the grid of the places of an array of a shape. The
// result holds one more axis in front, of the length of the shape.
fn indices(shape: i64[:]) -> Arr, Fault {
    let f = check_shape(shape);
    if f.kind != F_NONE { return empty_arr(), f; }
    if shape.len >= MAX_DIMS {
        return empty_arr(), fault(F_LIMIT, "indices", "the shape holds too many dimensions");
    }
    let n = numel(shape);
    if n * shape.len > MAX_ELEMENTS {
        return empty_arr(), fault(F_LIMIT, "indices", "the result holds more elements than the limit");
    }
    var strides: i64[>..] = [];
    strides_of(shape, strides);
    var d: f64[>..] = [];
    var a: i64 = 0;
    while a < shape.len {
        var i: i64 = 0;
        while i < n { d.push((i / strides[a]) % shape[a]); i += 1; }
        a += 1;
    }
    var s: i64[..8] = [];
    s.push(shape.len);
    var i: i64 = 0;
    while i < shape.len { s.push(shape[i]); i += 1; }
    return Arr { ndim: s.len, shape: s, data: d }, no_fault();
}

// ix_ gives the open grid of two lists of places: a column of the first
// list and a row of the second. The package holds the pair of lists
// only.
fn ix_(first: i64[:], second: i64[:]) -> Arr, Arr, Fault {
    if first.len < 1 || second.len < 1 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "ix_", "a list holds no place");
    }
    var c: f64[>..] = [];
    var r: f64[>..] = [];
    var i: i64 = 0;
    while i < first.len { c.push(first[i]); i += 1; }
    i = 0;
    while i < second.len { r.push(second[i]); i += 1; }
    var sc: i64[..8] = [];
    sc.push(first.len);
    sc.push(1);
    var sr: i64[..8] = [];
    sr.push(1);
    sr.push(second.len);
    return Arr { ndim: 2, shape: sc, data: c }, Arr { ndim: 2, shape: sr, data: r }, no_fault();
}

// unravel_index turns flat places into one row of places per axis of
// the shape. A place outside the array is a fault.
fn unravel_index(flat: i64[:], shape: i64[:]) -> Arr, Fault {
    let f = check_shape(shape);
    if f.kind != F_NONE { return empty_arr(), f; }
    if flat.len < 1 {
        return empty_arr(), fault(F_VALUE, "unravel_index", "the list holds no place");
    }
    let n = numel(shape);
    var strides: i64[>..] = [];
    strides_of(shape, strides);
    var d: f64[>..] = [];
    var a: i64 = 0;
    while a < shape.len {
        var i: i64 = 0;
        while i < flat.len {
            if flat[i] < 0 || flat[i] >= n {
                return empty_arr(), fault(F_RANGE, "unravel_index", "a place stands outside the array");
            }
            d.push((flat[i] / strides[a]) % shape[a]);
            i += 1;
        }
        a += 1;
    }
    var s: i64[..8] = [];
    s.push(shape.len);
    s.push(flat.len);
    return Arr { ndim: 2, shape: s, data: d }, no_fault();
}

// ravel_multi_index turns one row of places per axis into flat places.
// The mode is "raise", "wrap" or "clip" for a place outside its axis.
fn ravel_multi_index(multi, shape: i64[:], mode: const u8[:] = "raise") -> Arr, Fault {
    let f = check_shape(shape);
    if f.kind != F_NONE { return empty_arr(), f; }
    if mode != "raise" && mode != "wrap" && mode != "clip" {
        return empty_arr(), fault(F_VALUE, "ravel_multi_index", "the mode holds an unknown name");
    }
    if multi.ndim != 2 || multi.shape[0] != shape.len {
        return empty_arr(), fault(F_SHAPE, "ravel_multi_index", "the table does not hold one row per axis");
    }
    var strides: i64[>..] = [];
    strides_of(shape, strides);
    let count = multi.shape[1];
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < count {
        var flat: i64 = 0;
        var a: i64 = 0;
        while a < shape.len {
            let v = multi.data[a * count + i];
            if v != v || v != mathshim::trunc_of(v) || v > 4503599627370496.0 || v < -4503599627370496.0 {
                return empty_arr(), fault(F_VALUE, "ravel_multi_index", "a place holds no whole number");
            }
            var p = v as i64;
            if mode == "wrap" {
                p = p % shape[a];
                if p < 0 { p += shape[a]; }
            } else if mode == "clip" {
                if p < 0 { p = 0; }
                if p >= shape[a] { p = shape[a] - 1; }
            } else if p < 0 || p >= shape[a] {
                return empty_arr(), fault(F_RANGE, "ravel_multi_index", "a place stands outside its axis");
            }
            flat += p * strides[a];
            a += 1;
        }
        d.push(flat);
        i += 1;
    }
    var s: i64[..8] = [];
    s.push(count);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}

// unstack cuts an array into pieces along an axis, one per entry of
// the axis. Read a piece with unstack_part.
fn unstack(a, axis: i64 = 0) -> Parts, Fault {
    let ax = fix_axis(axis, a.ndim);
    if ax < 0 {
        return empty_parts(), fault(F_RANGE, "unstack", "the axis stands outside the shape");
    }
    return split(a, a.shape[ax], ax);
}

// unstack_part gives one piece of unstack with the axis of the cut
// dropped. A piece of an array of one dimension keeps its axis of
// length one, because the package holds no scalar array.
fn unstack_part(p, index: i64) -> Arr, Fault {
    let x, f = part(p, index);
    if f.kind != F_NONE { return empty_arr(), f; }
    if x.ndim == 1 { return x, no_fault(); }
    var s: i64[..8] = [];
    var i: i64 = 0;
    while i < x.ndim { if i != p.axis { s.push(x.shape[i]); } i += 1; }
    return Arr { ndim: s.len, shape: s, data: copy(x.data) }, no_fault();
}

// block_grid assembles four arrays into a grid of two rows and two columns.
// Use hstack and vstack for other grids.
fn block_grid(top_left, top_right, bottom_left, bottom_right) -> Arr, Fault {
    let t, ft = hstack(top_left, top_right);
    if ft.kind != F_NONE { return empty_arr(), ft; }
    let b, fb = hstack(bottom_left, bottom_right);
    if fb.kind != F_NONE { return empty_arr(), fb; }
    return vstack(t, b);
}
