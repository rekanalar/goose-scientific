// The rank filters and the derivative filters of arrays of N dimensions,
// as in scipy.ndimage: rank_filter, median_filter, percentile_filter,
// minimum_filter, maximum_filter, minimum_filter1d, maximum_filter1d,
// sobel, prewitt, laplace, gaussian_laplace and gaussian_gradient_magnitude.
//
// A window is a footprint of N dimensions. The value at x takes the
// places x + o - (f / 2) - origin, where o runs over the nonzero places of
// the footprint and f is its shape. The border follows the mode, as in the
// linear filters. A NaN counts as the largest value.
//
// A call with the suffix _fp takes the footprint as an array. The other
// calls take the size of a box.
namespace scientific;

import mathshim;

// The cost of a rank filter is the count of values times the count of
// places of the window. The sort of a window makes it heavier than a sum.
const ND_RANK_COST: i64 = 200000000;

// nd_sift moves the root of a heap of end values down, in a max heap.
fn nd_sift(buf: f64[>..]&, start: i64, end: i64) {
    var root = start;
    var go = true;
    while go {
        var child = 2 * root + 1;
        if child >= end {
            go = false;
        } else {
            if child + 1 < end && nd_less(buf[child], buf[child + 1]) { child += 1; }
            if nd_less(buf[root], buf[child]) {
                let t = buf[root];
                buf[root] = buf[child];
                buf[child] = t;
                root = child;
            } else {
                go = false;
            }
        }
    }
}

// nd_pick gives the value of a rank in the first t values of buf. A rank
// of 0 or of t - 1 scans. Else a heap gives the answer in t log t steps,
// whatever the data. The call changes the order of the values.
fn nd_pick(buf: f64[>..]&, t: i64, rank: i64) -> f64 {
    var m = buf[0];
    var i: i64 = 1;
    if rank == 0 {
        while i < t { if nd_less(buf[i], m) { m = buf[i]; } i += 1; }
        return m;
    }
    if rank == t - 1 {
        while i < t { if nd_less(m, buf[i]) { m = buf[i]; } i += 1; }
        return m;
    }
    var s = t / 2 - 1;
    while s >= 0 { nd_sift(buf, s, t); s -= 1; }
    var end = t - 1;
    let last = rank;
    while end > last {
        let x = buf[0];
        buf[0] = buf[end];
        buf[end] = x;
        nd_sift(buf, 0, end);
        end -= 1;
    }
    return buf[0];
}

// nd_rank gives the value of one rank in the window of every value. The
// footprint fp marks the places with a nonzero value. The buffer out holds
// as many zeros as src holds values.
fn nd_rank(src: f64[:], shape: i64[:], fp: f64[:], fshape: i64[:], shift: i64[:], rank: i64, mode: i64, cval: f64, out: f64[>..]&) {
    let nd = shape.len;
    var offs: i64[>..] = [];
    var ox: i64[..8] = [];
    var d: i64 = 0;
    while d < nd { ox.push(0); d += 1; }
    var t: i64 = 0;
    var q: i64 = 0;
    while q < fp.len {
        if fp[q] != 0.0 {
            d = 0;
            while d < nd { offs.push(ox[d]); d += 1; }
            t += 1;
        }
        d = nd - 1;
        while d >= 0 {
            ox[d] += 1;
            if ox[d] < fshape[d] { d = -1; } else { ox[d] = 0; d -= 1; }
        }
        q += 1;
    }
    var buf: f64[>..] = nd_zeros(t);
    var ix: i64[..8] = [];
    d = 0;
    while d < nd { ix.push(0); d += 1; }
    var p: i64 = 0;
    while p < src.len {
        var k: i64 = 0;
        while k < t {
            var flat: i64 = 0;
            var outside = false;
            d = 0;
            while d < nd && !outside {
                let m = nd_src(ix[d] + offs[k * nd + d] - shift[d], shape[d], mode);
                if m < 0 { outside = true; } else { flat = flat * shape[d] + m; }
                d += 1;
            }
            if outside { buf[k] = cval; } else { buf[k] = src[flat]; }
            k += 1;
        }
        out[p] = nd_pick(buf, t, rank);
        d = nd - 1;
        while d >= 0 {
            ix[d] += 1;
            if ix[d] < shape[d] { d = -1; } else { ix[d] = 0; d -= 1; }
        }
        p += 1;
    }
}

// nd_rank_of gives the rank for a count of places. The kinds are 0 for the
// rank itself, 1 for the median, 2 for a percentile, 3 for the minimum and
// 4 for the maximum. A rank below zero counts from the end. It gives -1
// for a rank outside the window.
fn nd_rank_of(kind: i64, rank: i64, percentile: f64, t: i64) -> i64 {
    var r = rank;
    if kind == 1 { r = t / 2; }
    if kind == 3 { r = 0; }
    if kind == 4 { r = t - 1; }
    if kind == 2 {
        var p = percentile;
        if p < 0.0 { p += 100.0; }
        if !(p >= 0.0 && p <= 100.0) { return -1; }
        if p == 100.0 { r = t - 1; } else { r = (((t as f64) * p / 100.0) as i64); }
    }
    if r < 0 { r += t; }
    if r < 0 || r >= t { return -1; }
    return r;
}

// nd_rank_run checks the operands, resolves the rank and runs the filter
// for a footprint. The footprint is an array of the count of dimensions of
// the data.
fn nd_rank_run(a, fp, kind: i64, rank: i64, percentile: f64, mode: const u8[:], cval: f64, origin: i64[:], tag: const u8[:]) -> Arr, Fault {
    let m, f = nd_check(a, mode, tag);
    if f.kind != F_NONE { return empty_arr(), f; }
    let ff = check_stack(fp.ndim, fp.shape, fp.data.len, 1, tag);
    if ff.kind != F_NONE { return empty_arr(), ff; }
    if fp.ndim != a.ndim {
        return empty_arr(), fault(F_SHAPE, tag, "the footprint holds another count of dimensions than the array");
    }
    var og: i64[..8] = [];
    if !nd_list(origin, a.ndim, 0, og) {
        return empty_arr(), fault(F_SHAPE, tag, "the origin holds a count that is not one for each axis");
    }
    var shift: i64[..8] = [];
    var d: i64 = 0;
    while d < a.ndim {
        let fs = fp.shape[d];
        if og[d] < -(fs / 2) || og[d] > fs - 1 - fs / 2 {
            return empty_arr(), fault(F_RANGE, tag, "the origin stands outside the footprint");
        }
        shift.push(fs / 2 + og[d]);
        d += 1;
    }
    var t: i64 = 0;
    var i: i64 = 0;
    while i < fp.data.len { if fp.data[i] != 0.0 { t += 1; } i += 1; }
    if t < 1 { return empty_arr(), fault(F_VALUE, tag, "the footprint holds no nonzero place"); }
    let r = nd_rank_of(kind, rank, percentile, t);
    if r < 0 { return empty_arr(), fault(F_RANGE, tag, "the rank stands outside the window"); }
    if a.data.len > ND_RANK_COST / t {
        return empty_arr(), fault(F_LIMIT, tag, "the cost of the filter is above the limit");
    }
    var out: f64[>..] = nd_zeros(a.data.len);
    nd_rank(a.data, a.shape, fp.data, fp.shape, shift, r, m, cval, out);
    return arr_of(copy(out), a.shape), no_fault();
}

// nd_box_fp builds the footprint of a box of ones from a list of sizes. The
// cost check runs before the allocation.
fn nd_box_fp(a, size: i64[:], tag: const u8[:]) -> Arr, Fault {
    var sz: i64[..8] = [];
    if size.len == 0 || !nd_list(size, a.ndim, 1, sz) {
        return empty_arr(), fault(F_SHAPE, tag, "the size holds a count that is not one for each axis");
    }
    var total: i64 = 1;
    var d: i64 = 0;
    while d < a.ndim {
        if sz[d] < 1 { return empty_arr(), fault(F_VALUE, tag, "the size holds a value below one"); }
        if total > ND_RANK_COST / sz[d] {
            return empty_arr(), fault(F_LIMIT, tag, "the cost of the filter is above the limit");
        }
        total *= sz[d];
        d += 1;
    }
    var ones: f64[>..] = [];
    var i: i64 = 0;
    while i < total { ones.push(1.0); i += 1; }
    return arr_of(copy(ones), sz), no_fault();
}

// nd_rank_size runs a rank filter with a box.
fn nd_rank_size(a, size: i64[:], kind: i64, rank: i64, percentile: f64, mode: const u8[:], cval: f64, origin: i64[:], tag: const u8[:]) -> Arr, Fault {
    let m, f = nd_check(a, mode, tag);
    if f.kind != F_NONE { return empty_arr(), f; }
    let fp, g = nd_box_fp(a, size, tag);
    if g.kind != F_NONE { return empty_arr(), g; }
    return nd_rank_run(a, fp, kind, rank, percentile, mode, cval, origin, tag);
}

// rank_filter gives the value of a rank in a box around each value.
fn rank_filter(a, rank: i64, size: i64[:], mode: const u8[:] = "reflect", cval: f64 = 0.0, origin: i64[:] = []) -> Arr, Fault {
    return nd_rank_size(a, size, 0, rank, 0.0, mode, cval, origin, "rank_filter");
}

// rank_filter_fp gives the value of a rank in the footprint around each value.
fn rank_filter_fp(a, rank: i64, footprint, mode: const u8[:] = "reflect", cval: f64 = 0.0, origin: i64[:] = []) -> Arr, Fault {
    return nd_rank_run(a, footprint, 0, rank, 0.0, mode, cval, origin, "rank_filter_fp");
}

// median_filter gives the median of a box around each value.
fn median_filter(a, size: i64[:], mode: const u8[:] = "reflect", cval: f64 = 0.0, origin: i64[:] = []) -> Arr, Fault {
    return nd_rank_size(a, size, 1, 0, 0.0, mode, cval, origin, "median_filter");
}

fn median_filter_fp(a, footprint, mode: const u8[:] = "reflect", cval: f64 = 0.0, origin: i64[:] = []) -> Arr, Fault {
    return nd_rank_run(a, footprint, 1, 0, 0.0, mode, cval, origin, "median_filter_fp");
}

// percentile_filter gives a percentile of a box around each value. A
// negative percentile counts from 100, as in SciPy.
fn percentile_filter(a, percentile: f64, size: i64[:], mode: const u8[:] = "reflect", cval: f64 = 0.0, origin: i64[:] = []) -> Arr, Fault {
    return nd_rank_size(a, size, 2, 0, percentile, mode, cval, origin, "percentile_filter");
}

fn percentile_filter_fp(a, percentile: f64, footprint, mode: const u8[:] = "reflect", cval: f64 = 0.0, origin: i64[:] = []) -> Arr, Fault {
    return nd_rank_run(a, footprint, 2, 0, percentile, mode, cval, origin, "percentile_filter_fp");
}

// nd_extreme_all gives the minimum or the maximum of a box. The box is
// separable, so each axis takes one pass.
fn nd_extreme_all(a, size: i64[:], op: i64, mode: const u8[:], cval: f64, origin: i64[:], tag: const u8[:]) -> Arr, Fault {
    let m, f = nd_check(a, mode, tag);
    if f.kind != F_NONE { return empty_arr(), f; }
    var sz: i64[..8] = [];
    if size.len == 0 || !nd_list(size, a.ndim, 1, sz) {
        return empty_arr(), fault(F_SHAPE, tag, "the size holds a count that is not one for each axis");
    }
    var og: i64[..8] = [];
    if !nd_list(origin, a.ndim, 0, og) {
        return empty_arr(), fault(F_SHAPE, tag, "the origin holds a count that is not one for each axis");
    }
    var cur: f64[>..] = [];
    var i: i64 = 0;
    while i < a.data.len { cur.push(a.data[i]); i += 1; }
    var d: i64 = 0;
    while d < a.ndim {
        var nxt: f64[>..] = nd_zeros(a.data.len);
        let g = nd_box_axis(cur, a.shape, d, op, sz[d], og[d], m, cval, tag, nxt);
        if g.kind != F_NONE { return empty_arr(), g; }
        cur = copy(nxt);
        d += 1;
    }
    return arr_of(copy(cur), a.shape), no_fault();
}

// minimum_filter gives the minimum of a box around each value.
fn minimum_filter(a, size: i64[:], mode: const u8[:] = "reflect", cval: f64 = 0.0, origin: i64[:] = []) -> Arr, Fault {
    return nd_extreme_all(a, size, 2, mode, cval, origin, "minimum_filter");
}

// maximum_filter gives the maximum of a box around each value.
fn maximum_filter(a, size: i64[:], mode: const u8[:] = "reflect", cval: f64 = 0.0, origin: i64[:] = []) -> Arr, Fault {
    return nd_extreme_all(a, size, 3, mode, cval, origin, "maximum_filter");
}

fn minimum_filter_fp(a, footprint, mode: const u8[:] = "reflect", cval: f64 = 0.0, origin: i64[:] = []) -> Arr, Fault {
    return nd_rank_run(a, footprint, 3, 0, 0.0, mode, cval, origin, "minimum_filter_fp");
}

fn maximum_filter_fp(a, footprint, mode: const u8[:] = "reflect", cval: f64 = 0.0, origin: i64[:] = []) -> Arr, Fault {
    return nd_rank_run(a, footprint, 4, 0, 0.0, mode, cval, origin, "maximum_filter_fp");
}

// nd_extreme1d runs the minimum or the maximum along one axis.
fn nd_extreme1d(a, size: i64, axis: i64, op: i64, mode: const u8[:], cval: f64, origin: i64, tag: const u8[:]) -> Arr, Fault {
    let m, f = nd_check(a, mode, tag);
    if f.kind != F_NONE { return empty_arr(), f; }
    let ax = nd_axis(axis, a.ndim);
    if ax < 0 { return empty_arr(), fault(F_RANGE, tag, "the axis stands outside the array"); }
    var out: f64[>..] = nd_zeros(a.data.len);
    let g = nd_box_axis(a.data, a.shape, ax, op, size, origin, m, cval, tag, out);
    if g.kind != F_NONE { return empty_arr(), g; }
    return arr_of(copy(out), a.shape), no_fault();
}

// minimum_filter1d gives the minimum of a window along one axis.
fn minimum_filter1d(a, size: i64, axis: i64 = -1, mode: const u8[:] = "reflect", cval: f64 = 0.0, origin: i64 = 0) -> Arr, Fault {
    return nd_extreme1d(a, size, axis, 2, mode, cval, origin, "minimum_filter1d");
}

// maximum_filter1d gives the maximum of a window along one axis.
fn maximum_filter1d(a, size: i64, axis: i64 = -1, mode: const u8[:] = "reflect", cval: f64 = 0.0, origin: i64 = 0) -> Arr, Fault {
    return nd_extreme1d(a, size, axis, 3, mode, cval, origin, "maximum_filter1d");
}

// nd_three correlates along one axis with three weights.
fn nd_three(src: f64[:], shape: i64[:], axis: i64, w0: f64, w1: f64, w2: f64, mode: i64, cval: f64) -> f64[>..] {
    var out: f64[>..] = nd_zeros(src.len);
    nd_along(src, shape, axis, [w0, w1, w2], 0, 0, 1, mode, cval, out);
    return out;
}

// nd_edge gives the derivative along one axis and the smoothing along the
// others. The smoothing weights are s0, s1 and s0.
fn nd_edge(a, axis: i64, s0: f64, s1: f64, mode: const u8[:], cval: f64, tag: const u8[:]) -> Arr, Fault {
    let m, f = nd_check(a, mode, tag);
    if f.kind != F_NONE { return empty_arr(), f; }
    let ax = nd_axis(axis, a.ndim);
    if ax < 0 { return empty_arr(), fault(F_RANGE, tag, "the axis stands outside the array"); }
    var cur: f64[>..] = nd_three(a.data, a.shape, ax, -1.0, 0.0, 1.0, m, cval);
    var d: i64 = 0;
    while d < a.ndim {
        if d != ax {
            let nxt = nd_three(cur, a.shape, d, s0, s1, s0, m, cval);
            cur = copy(nxt);
        }
        d += 1;
    }
    return arr_of(copy(cur), a.shape), no_fault();
}

// sobel gives the Sobel derivative along an axis.
fn sobel(a, axis: i64 = -1, mode: const u8[:] = "reflect", cval: f64 = 0.0) -> Arr, Fault {
    return nd_edge(a, axis, 1.0, 2.0, mode, cval, "sobel");
}

// prewitt gives the Prewitt derivative along an axis.
fn prewitt(a, axis: i64 = -1, mode: const u8[:] = "reflect", cval: f64 = 0.0) -> Arr, Fault {
    return nd_edge(a, axis, 1.0, 1.0, mode, cval, "prewitt");
}

// laplace gives the sum of the second differences of all axes.
fn laplace(a, mode: const u8[:] = "reflect", cval: f64 = 0.0) -> Arr, Fault {
    let m, f = nd_check(a, mode, "laplace");
    if f.kind != F_NONE { return empty_arr(), f; }
    var acc: f64[>..] = nd_zeros(a.data.len);
    var d: i64 = 0;
    while d < a.ndim {
        let t = nd_three(a.data, a.shape, d, 1.0, -2.0, 1.0, m, cval);
        var i: i64 = 0;
        while i < acc.len { acc[i] += t[i]; i += 1; }
        d += 1;
    }
    return arr_of(copy(acc), a.shape), no_fault();
}

// gaussian_laplace gives the sum of the second Gaussian derivatives of all axes.
fn gaussian_laplace(a, sigma: f64[:], mode: const u8[:] = "reflect", cval: f64 = 0.0, truncate: f64 = 4.0) -> Arr, Fault {
    let m, f = nd_check(a, mode, "gaussian_laplace");
    if f.kind != F_NONE { return empty_arr(), f; }
    var acc: f64[>..] = nd_zeros(a.data.len);
    var d: i64 = 0;
    while d < a.ndim {
        var od: i64[..8] = [];
        var k: i64 = 0;
        while k < a.ndim { if k == d { od.push(2); } else { od.push(0); } k += 1; }
        let t, g = gaussian_filter(a, sigma, od, mode, cval, truncate);
        if g.kind != F_NONE { return empty_arr(), g; }
        var i: i64 = 0;
        while i < acc.len { acc[i] += t.data[i]; i += 1; }
        d += 1;
    }
    return arr_of(copy(acc), a.shape), no_fault();
}

// gaussian_gradient_magnitude gives the length of the Gaussian gradient.
fn gaussian_gradient_magnitude(a, sigma: f64[:], mode: const u8[:] = "reflect", cval: f64 = 0.0, truncate: f64 = 4.0) -> Arr, Fault {
    let m, f = nd_check(a, mode, "gaussian_gradient_magnitude");
    if f.kind != F_NONE { return empty_arr(), f; }
    var acc: f64[>..] = nd_zeros(a.data.len);
    var d: i64 = 0;
    while d < a.ndim {
        var od: i64[..8] = [];
        var k: i64 = 0;
        while k < a.ndim { if k == d { od.push(1); } else { od.push(0); } k += 1; }
        let t, g = gaussian_filter(a, sigma, od, mode, cval, truncate);
        if g.kind != F_NONE { return empty_arr(), g; }
        var i: i64 = 0;
        while i < acc.len { acc[i] += t.data[i] * t.data[i]; i += 1; }
        d += 1;
    }
    var i: i64 = 0;
    while i < acc.len { acc[i] = mathshim::root(acc[i]); i += 1; }
    return arr_of(copy(acc), a.shape), no_fault();
}
