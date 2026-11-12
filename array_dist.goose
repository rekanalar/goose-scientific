// The chamfer and brute force distance transforms and the value indices of
// N dimensions, as in scipy.ndimage: distance_transform_cdt,
// distance_transform_bf and value_indices.
//
// Both transforms give each foreground value (nonzero) its distance to the
// nearest background value (zero). The calls with the suffix _indices also
// give the position of that background value. The array of positions holds one
// row for each axis, so its shape is the count of axes followed by the shape.
//
// A chamfer transform makes two passes over the array with the half of a
// structure of side 3 on every axis. A structure holds true where its value is
// not zero. With the metric "taxicab" the structure is the cross. With
// "chessboard" it is the full box. A value that the structure never reaches
// keeps the distance -1, as in SciPy.
//
// The brute force transform tests every foreground value against every
// background value that touches the foreground. A tie goes to the later
// position in row-major order, as in SciPy. The sampling counts for the
// metric "euclidean" only, as in SciPy.
namespace scientific;

import mathshim;

// The brute force transform tests each foreground value against each edge
// value. The count of pairs has this limit.
const DC_BF_COST: i64 = 200000000;

// dc_pass runs one chamfer pass. It reads the offsets of the forward half of
// the structure. The buffers hold flat arrays in the order of the pass.
fn dc_pass(shape: i64[:], offs: i64[:], t: i64, dt: i64[>..]&, ft: i64[>..]&, want: bool) {
    let nd = shape.len;
    var ix: i64[..8] = [];
    var d: i64 = 0;
    while d < nd { ix.push(0); d += 1; }
    var strides: i64[..8] = [];
    d = 0;
    while d < nd { strides.push(0); d += 1; }
    var run: i64 = 1;
    d = nd - 1;
    while d >= 0 { strides[d] = run; run *= shape[d]; d -= 1; }
    var p: i64 = 0;
    while p < dt.len {
        if dt[p] != 0 {
            var mn = dt[p];
            var mi: i64 = -1;
            var k: i64 = 0;
            while k < t {
                var q: i64 = 0;
                var outside = false;
                d = 0;
                while d < nd && !outside {
                    let c = ix[d] + offs[k * nd + d];
                    if c < 0 || c >= shape[d] { outside = true; } else { q += c * strides[d]; }
                    d += 1;
                }
                if !outside {
                    var v = dt[q];
                    if v >= 0 {
                        v += 1;
                        if mn < 0 || v < mn { mn = v; mi = q; }
                    }
                }
                k += 1;
            }
            if mi >= 0 {
                dt[p] = mn;
                if want { ft[p] = ft[mi]; }
            }
        }
        gf_next(ix, shape);
        p += 1;
    }
}

// dc_reverse reverses a flat buffer. It reverses every axis of the array.
fn dc_reverse(b: i64[>..]&) {
    var lo: i64 = 0;
    var hi: i64 = b.len - 1;
    while lo < hi {
        let t = b[lo];
        b[lo] = b[hi];
        b[hi] = t;
        lo += 1;
        hi -= 1;
    }
}

// dc_positions turns flat positions into one row of coordinates for each axis.
fn dc_positions(shape: i64[:], flat: i64[:]) -> Arr {
    let nd = shape.len;
    let n = flat.len;
    var out: f64[>..] = nd_zeros(nd * n);
    var d: i64 = nd - 1;
    var div: i64 = 1;
    while d >= 0 {
        var p: i64 = 0;
        while p < n {
            out[d * n + p] = ((flat[p] / div) % shape[d]) as f64;
            p += 1;
        }
        div *= shape[d];
        d -= 1;
    }
    var sh: i64[..8] = [];
    sh.push(nd);
    d = 0;
    while d < nd { sh.push(shape[d]); d += 1; }
    return arr_of(copy(out), sh);
}

// dc_has_background checks that the array holds a background value.
fn dc_has_background(a) -> bool {
    var p: i64 = 0;
    while p < a.data.len { if a.data[p] == 0.0 { return true; } p += 1; }
    return false;
}

// dc_chamfer runs the two passes with a structure of side 3.
fn dc_chamfer(a, st, want: bool, tag: const u8[:]) -> Arr, Arr, Fault {
    let fa = check_stack(a.ndim, a.shape, a.data.len, 1, tag);
    if fa.kind != F_NONE { return empty_arr(), empty_arr(), fa; }
    let fs = check_stack(st.ndim, st.shape, st.data.len, 1, tag);
    if fs.kind != F_NONE { return empty_arr(), empty_arr(), fs; }
    if st.ndim != a.ndim {
        return empty_arr(), empty_arr(), fault(F_SHAPE, tag, "the metric holds another count of dimensions than the array");
    }
    var d: i64 = 0;
    while d < st.ndim {
        if st.shape[d] != 3 { return empty_arr(), empty_arr(), fault(F_SHAPE, tag, "the metric must hold the size 3 on every axis"); }
        d += 1;
    }
    let nd = a.ndim;
    let n = a.data.len;
    if want && n > MAX_ELEMENTS / nd {
        return empty_arr(), empty_arr(), fault(F_LIMIT, tag, "the result holds more elements than the limit");
    }
    if !dc_has_background(a) {
        return empty_arr(), empty_arr(), fault(F_VALUE, tag, "the array holds no background value");
    }
    var offs: i64[>..] = [];
    var t: i64 = 0;
    let half = st.data.len / 2;
    var j: i64 = 0;
    while j < half {
        if st.data[j] != 0.0 {
            var rem = j;
            var c: i64[..8] = [];
            d = 0;
            while d < nd { c.push(0); d += 1; }
            d = nd - 1;
            while d >= 0 { c[d] = rem % 3 - 1; rem = rem / 3; d -= 1; }
            d = 0;
            while d < nd { offs.push(c[d]); d += 1; }
            t += 1;
        }
        j += 1;
    }
    if t > 0 && n > ND_COST / t {
        return empty_arr(), empty_arr(), fault(F_LIMIT, tag, "the cost of the transform is above the limit");
    }
    var dt: i64[>..] = [];
    var ft: i64[>..] = [];
    var p: i64 = 0;
    while p < n {
        if a.data[p] != 0.0 { dt.push(-1); } else { dt.push(0); }
        if want { ft.push(p); }
        p += 1;
    }
    dc_pass(a.shape, offs, t, dt, ft, want);
    dc_reverse(dt);
    if want { dc_reverse(ft); }
    dc_pass(a.shape, offs, t, dt, ft, want);
    dc_reverse(dt);
    if want { dc_reverse(ft); }
    var out: f64[>..] = [];
    p = 0;
    while p < n { out.push(dt[p] as f64); p += 1; }
    var idx = empty_arr();
    if want { idx = dc_positions(a.shape, ft); }
    return arr_of(copy(out), a.shape), idx, no_fault();
}

// dc_metric gives the structure of a metric name.
fn dc_metric(a, metric: const u8[:], tag: const u8[:]) -> Arr, Fault {
    if metric == "taxicab" || metric == "cityblock" || metric == "manhattan" {
        return generate_binary_structure(a.ndim, 1);
    }
    if metric == "chessboard" { return generate_binary_structure(a.ndim, a.ndim); }
    return empty_arr(), fault(F_VALUE, tag, "the metric is not taxicab, cityblock, manhattan or chessboard");
}

// distance_transform_cdt gives the chamfer distance of each foreground value
// to the nearest background value. The metric is "chessboard", "taxicab",
// "cityblock" or "manhattan".
fn distance_transform_cdt(a, metric: const u8[:] = "chessboard") -> Arr, Fault {
    let st, f = dc_metric(a, metric, "distance_transform_cdt");
    if f.kind != F_NONE { return empty_arr(), f; }
    let d, i, g = dc_chamfer(a, st, false, "distance_transform_cdt");
    return d, g;
}

// distance_transform_cdt_indices gives the distances and the positions of the
// nearest background values.
fn distance_transform_cdt_indices(a, metric: const u8[:] = "chessboard") -> Arr, Arr, Fault {
    let st, f = dc_metric(a, metric, "distance_transform_cdt_indices");
    if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
    return dc_chamfer(a, st, true, "distance_transform_cdt_indices");
}

// distance_transform_cdt_st takes a structure of side 3 on every axis for the
// metric. A value that is not zero counts as true. SciPy reads the buffer of an
// integer or float structure as bytes and gives wrong results for it. This
// package reads the values.
fn distance_transform_cdt_st(a, structure) -> Arr, Fault {
    let d, i, g = dc_chamfer(a, structure, false, "distance_transform_cdt_st");
    return d, g;
}

fn distance_transform_cdt_st_indices(a, structure) -> Arr, Arr, Fault {
    return dc_chamfer(a, structure, true, "distance_transform_cdt_st_indices");
}

// dc_brute runs the brute force transform. The metric is 0 for euclidean, 1
// for taxicab and 2 for chessboard.
fn dc_brute(a, metric: i64, sampling: f64[:], want: bool, tag: const u8[:]) -> Arr, Arr, Fault {
    let fa = check_stack(a.ndim, a.shape, a.data.len, 1, tag);
    if fa.kind != F_NONE { return empty_arr(), empty_arr(), fa; }
    let nd = a.ndim;
    let n = a.data.len;
    if sampling.len != 0 && sampling.len != 1 && sampling.len != nd {
        return empty_arr(), empty_arr(), fault(F_SHAPE, tag, "the sampling holds a count that is not one for each axis");
    }
    var sp: f64[>..] = [];
    var d: i64 = 0;
    while d < nd {
        var s: f64 = 1.0;
        if metric == 0 {
            if sampling.len == 1 { s = sampling[0]; }
            if sampling.len > 1 { s = sampling[d]; }
            if !(s - s == 0.0) || !(s > 0.0) || s * (a.shape[d] as f64) > 1.0e100 {
                return empty_arr(), empty_arr(), fault(F_VALUE, tag, "the sampling holds a value that is not above zero or is too large");
            }
        }
        sp.push(s);
        d += 1;
    }
    if want && n > MAX_ELEMENTS / nd {
        return empty_arr(), empty_arr(), fault(F_LIMIT, tag, "the result holds more elements than the limit");
    }
    var fg: i64 = 0;
    var p: i64 = 0;
    while p < n { if a.data[p] != 0.0 { fg += 1; } p += 1; }
    if fg == n {
        return empty_arr(), empty_arr(), fault(F_VALUE, tag, "the array holds no background value");
    }
    // The background values that touch a foreground value, in row-major order.
    var edge: i64[>..] = [];
    var ix: i64[..8] = [];
    d = 0;
    while d < nd { ix.push(0); d += 1; }
    var boxes: i64 = 1;
    d = 0;
    while d < nd { boxes *= 3; d += 1; }
    var strides: i64[..8] = [];
    d = 0;
    while d < nd { strides.push(0); d += 1; }
    var run: i64 = 1;
    d = nd - 1;
    while d >= 0 { strides[d] = run; run *= a.shape[d]; d -= 1; }
    p = 0;
    while p < n {
        if a.data[p] == 0.0 {
            var touch = false;
            var k: i64 = 0;
            while k < boxes && !touch {
                var rem = k;
                var q: i64 = 0;
                var outside = false;
                d = nd - 1;
                while d >= 0 {
                    let c = ix[d] + rem % 3 - 1;
                    rem = rem / 3;
                    if c < 0 || c >= a.shape[d] { outside = true; } else { q += c * strides[d]; }
                    d -= 1;
                }
                if !outside && a.data[q] != 0.0 { touch = true; }
                k += 1;
            }
            if touch { edge.push(p); }
        }
        gf_next(ix, a.shape);
        p += 1;
    }
    if fg > 0 && edge.len > 0 && fg > DC_BF_COST / edge.len {
        return empty_arr(), empty_arr(), fault(F_LIMIT, tag, "the cost of the transform is above the limit");
    }
    var out: f64[>..] = nd_zeros(n);
    var ft: i64[>..] = [];
    p = 0;
    while p < n { ft.push(p); p += 1; }
    var px: i64[..8] = [];
    var qx: i64[..8] = [];
    d = 0;
    while d < nd { px.push(0); qx.push(0); d += 1; }
    p = 0;
    while p < n {
        if a.data[p] != 0.0 {
            var best: f64 = 0.0;
            var bi: i64 = -1;
            var e: i64 = 0;
            while e < edge.len {
                var rem = edge[e];
                d = nd - 1;
                while d >= 0 { qx[d] = rem % a.shape[d]; rem = rem / a.shape[d]; d -= 1; }
                var v: f64 = 0.0;
                d = 0;
                while d < nd {
                    var df = (px[d] - qx[d]) as f64;
                    if df < 0.0 { df = -df; }
                    if metric == 0 { df = df * sp[d]; v += df * df; }
                    else if metric == 1 { v += df; }
                    else { if df > v { v = df; } }
                    d += 1;
                }
                if bi < 0 || v <= best { best = v; bi = edge[e]; }
                e += 1;
            }
            if metric == 0 { out[p] = mathshim::root(best); } else { out[p] = best; }
            ft[p] = bi;
        }
        gf_next(px, a.shape);
        p += 1;
    }
    var idx = empty_arr();
    if want { idx = dc_positions(a.shape, ft); }
    return arr_of(copy(out), a.shape), idx, no_fault();
}

// dc_name gives the number of a brute force metric, or -1.
fn dc_name(metric: const u8[:]) -> i64 {
    if metric == "euclidean" { return 0; }
    if metric == "taxicab" || metric == "cityblock" || metric == "manhattan" { return 1; }
    if metric == "chessboard" { return 2; }
    return -1;
}

// distance_transform_bf gives the distance of each foreground value to the
// nearest background value by a search of the values that touch the
// foreground. The metric is "euclidean", "taxicab", "cityblock", "manhattan"
// or "chessboard".
fn distance_transform_bf(a, metric: const u8[:] = "euclidean", sampling: f64[:] = []) -> Arr, Fault {
    let m = dc_name(metric);
    if m < 0 { return empty_arr(), fault(F_VALUE, "distance_transform_bf", "the metric is not supported"); }
    let d, i, f = dc_brute(a, m, sampling, false, "distance_transform_bf");
    return d, f;
}

// distance_transform_bf_indices gives the distances and the positions of the
// nearest background values.
fn distance_transform_bf_indices(a, metric: const u8[:] = "euclidean", sampling: f64[:] = []) -> Arr, Arr, Fault {
    let m = dc_name(metric);
    if m < 0 { return empty_arr(), empty_arr(), fault(F_VALUE, "distance_transform_bf_indices", "the metric is not supported"); }
    return dc_brute(a, m, sampling, true, "distance_transform_bf_indices");
}

// dv_less orders two positions of the array by value and then by position.
fn dv_less(v: f64[:], x: i64, y: i64) -> bool {
    return v[x] < v[y] || (v[x] == v[y] && x < y);
}

fn dv_sift(ord: i64[>..]&, v: f64[:], start: i64, end: i64) {
    var root = start;
    var go = true;
    while go {
        var child = 2 * root + 1;
        if child >= end { go = false; }
        else {
            if child + 1 < end && dv_less(v, ord[child], ord[child + 1]) { child += 1; }
            if dv_less(v, ord[root], ord[child]) {
                let t = ord[root];
                ord[root] = ord[child];
                ord[child] = t;
                root = child;
            } else { go = false; }
        }
    }
}

// value_indices gives the positions of each value of an array of whole
// numbers. It gives the distinct values in ascending order, the count of
// positions of each value, and one row of coordinates for each position. The
// rows of one value are consecutive and in row-major order. With has_ignore
// true the value ignore_value gets no entry.
fn value_indices(a, ignore_value: f64 = 0.0, has_ignore: bool = false) -> Arr, Arr, Arr, Fault {
    let tag = "value_indices";
    let fa = check_stack(a.ndim, a.shape, a.data.len, 1, tag);
    if fa.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), fa; }
    let nd = a.ndim;
    let n = a.data.len;
    if n > MAX_ELEMENTS / nd {
        return empty_arr(), empty_arr(), empty_arr(), fault(F_LIMIT, tag, "the result holds more elements than the limit");
    }
    var keep: i64[>..] = [];
    var vals: f64[>..] = [];
    var p: i64 = 0;
    while p < n {
        let x = a.data[p];
        if !(x >= -9.0e18 && x <= 9.0e18) || (x as i64) as f64 != x {
            return empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, tag, "the array holds a value that is not a whole number");
        }
        vals.push(x);
        if !(has_ignore && x == ignore_value) { keep.push(p); }
        p += 1;
    }
    let m = keep.len;
    var s = m / 2 - 1;
    while s >= 0 { dv_sift(keep, vals, s, m); s -= 1; }
    var end = m - 1;
    while end > 0 {
        let t = keep[0];
        keep[0] = keep[end];
        keep[end] = t;
        dv_sift(keep, vals, 0, end);
        end -= 1;
    }
    var uv: f64[>..] = [];
    var uc: f64[>..] = [];
    var i: i64 = 0;
    while i < m {
        let x = vals[keep[i]];
        if uv.len == 0 || uv[uv.len - 1] != x { uv.push(x); uc.push(0.0); }
        uc[uc.len - 1] += 1.0;
        i += 1;
    }
    let pos = dc_positions(a.shape, keep);
    // dc_positions gives one row for each axis. Turn it into one row for each position.
    var rows: f64[>..] = nd_zeros(m * nd);
    var d: i64 = 0;
    while d < nd {
        i = 0;
        while i < m { rows[i * nd + d] = pos.data[d * m + i]; i += 1; }
        d += 1;
    }
    let k = uv.len;
    return arr_of(copy(uv), [k]), arr_of(copy(uc), [k]), arr_of(copy(rows), [m, nd]), no_fault();
}
